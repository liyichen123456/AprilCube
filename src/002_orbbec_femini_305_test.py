#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TARGET_DEVICE_HINT = "Gemini 305"
DEFAULT_FPS = 25

SENSOR_ENUM_CANDIDATES: dict[str, tuple[str, ...]] = {
    "color": ("COLOR_SENSOR",),
    "ir": ("IR_SENSOR",),
    "left_ir": ("LEFT_IR_SENSOR", "IR_LEFT_SENSOR"),
    "right_ir": ("RIGHT_IR_SENSOR", "IR_RIGHT_SENSOR"),
    "depth": ("DEPTH_SENSOR",),
}

FRAME_GETTER_CANDIDATES: dict[str, tuple[str, ...]] = {
    "color": ("get_color_frame",),
    "ir": ("get_ir_frame",),
    "left_ir": ("get_left_ir_frame", "get_ir_frame"),
    "right_ir": ("get_right_ir_frame", "get_ir_frame"),
    "depth": ("get_depth_frame",),
}


@dataclass(frozen=True)
class ProfileInfo:
    index: int
    width: int
    height: int
    fps: int
    format_name: str
    profile: Any

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def label(self) -> str:
        return f"{self.width}x{self.height}@{self.fps} {self.format_name}"


@dataclass(frozen=True)
class TestResult:
    ok: bool
    profile: ProfileInfo
    frames: int = 0
    measured_fps: float = 0.0
    actual_width: int | None = None
    actual_height: int | None = None
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Open an Orbbec {TARGET_DEVICE_HINT} and find the largest single-stream "
            "resolution that can actually start at the requested FPS."
        )
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="FPS to test.")
    parser.add_argument(
        "--sensor",
        choices=tuple(SENSOR_ENUM_CANDIDATES.keys()),
        default="color",
        help="Single stream to test. Default is color, i.e. monocular RGB/visible stream.",
    )
    parser.add_argument("--serial", type=str, default=None, help="Optional Orbbec serial number.")
    parser.add_argument(
        "--seconds",
        type=float,
        default=2.0,
        help="Seconds to capture for each candidate profile.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=1000,
        help="wait_for_frames timeout in milliseconds.",
    )
    parser.add_argument(
        "--try-all",
        action="store_true",
        help="Do not stop at the first successful max-resolution profile.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only enumerate profiles; do not start the camera stream.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a small OpenCV preview while testing each profile.",
    )
    return parser.parse_args()


def import_orbbec_sdk() -> Any:
    try:
        import pyorbbecsdk as ob  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "pyorbbecsdk is not installed in this Python environment.\n"
            "Install the official Python SDK first, for example:\n"
            "  python -m pip install --upgrade pyorbbecsdk2\n"
            "Then run this script again."
        ) from exc
    return ob


def orbbec_udev_install_hint(ob: Any) -> str:
    module_file = getattr(ob, "__file__", None)
    if not module_file:
        return (
            "Install Orbbec udev rules, reload udev, then replug the camera."
        )
    script = Path(module_file).resolve().parent / "shared" / "install_udev_rules.sh"
    if script.is_file():
        return (
            "Install Orbbec udev rules, then replug the camera:\n"
            f"  sudo bash {script}\n"
            "  # unplug/replug Gemini 305 after the command finishes"
        )
    return "Install Orbbec udev rules, reload udev, then replug the camera."


def enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    text = str(value)
    return text.split(".")[-1].split(":")[0].strip("<> ")


def resolve_sensor_type(ob: Any, sensor_key: str) -> Any:
    sensor_enum = getattr(ob, "OBSensorType")
    for name in SENSOR_ENUM_CANDIDATES[sensor_key]:
        if hasattr(sensor_enum, name):
            return getattr(sensor_enum, name)
    available = ", ".join(name for name in dir(sensor_enum) if name.isupper())
    raise RuntimeError(f"SDK does not expose sensor '{sensor_key}'. Available sensors: {available}")


def make_pipeline(ob: Any, device: Any | None) -> Any:
    if device is None:
        return ob.Pipeline()
    try:
        return ob.Pipeline(device)
    except TypeError:
        return ob.Pipeline()


def safe_call(obj: Any, method: str, *args: Any) -> Any | None:
    func = getattr(obj, method, None)
    if func is None:
        return None
    try:
        return func(*args)
    except Exception:
        return None


def select_device(ob: Any, serial: str | None) -> Any | None:
    if not hasattr(ob, "Context"):
        print("[device] SDK has no Context class; using Pipeline() default device.")
        return None

    context = ob.Context()
    devices = context.query_devices()
    count = int(devices.get_count())
    if count <= 0:
        raise RuntimeError("No Orbbec device found.")

    print(f"[device] found {count} Orbbec device(s)")
    for idx in range(count):
        name = safe_call(devices, "get_device_name_by_index", idx)
        sn = safe_call(devices, "get_device_serial_number_by_index", idx)
        uid = safe_call(devices, "get_device_uid_by_index", idx)
        conn = safe_call(devices, "get_device_connection_type_by_index", idx)
        print(f"  [{idx}] name={name} serial={sn} uid={uid} connection={conn}")

    if serial:
        return devices.get_device_by_serial_number(serial)
    return devices.get_device_by_index(0)


def print_device_info(device: Any | None) -> None:
    if device is None:
        return
    info = safe_call(device, "get_device_info")
    if info is None:
        return
    fields = (
        ("name", "get_name"),
        ("serial", "get_serial_number"),
        ("firmware", "get_firmware_version"),
        ("hardware", "get_hardware_version"),
        ("connection", "get_connection_type"),
        ("vid", "get_vid"),
        ("pid", "get_pid"),
    )
    values = []
    for label, method in fields:
        value = safe_call(info, method)
        if value is not None:
            values.append(f"{label}={value}")
    if values:
        print("[device] selected " + " ".join(values))


def enumerate_video_profiles(pipeline: Any, sensor_type: Any) -> list[ProfileInfo]:
    profile_list = pipeline.get_stream_profile_list(sensor_type)
    count = int(profile_list.get_count())
    profiles: list[ProfileInfo] = []
    for idx in range(count):
        profile = profile_list.get_stream_profile_by_index(idx)
        is_video = safe_call(profile, "is_video_stream_profile")
        if is_video is False:
            continue
        video_profile = safe_call(profile, "as_video_stream_profile") or profile
        profiles.append(
            ProfileInfo(
                index=idx,
                width=int(video_profile.get_width()),
                height=int(video_profile.get_height()),
                fps=int(video_profile.get_fps()),
                format_name=enum_name(video_profile.get_format()),
                profile=video_profile,
            )
        )
    return sorted(
        profiles,
        key=lambda p: (p.pixels, p.width, p.height, p.fps, p.format_name),
        reverse=True,
    )


def get_expected_frame(frames: Any, sensor_key: str) -> Any | None:
    for method in FRAME_GETTER_CANDIDATES[sensor_key]:
        frame = safe_call(frames, method)
        if frame is not None:
            return frame

    frame_count = safe_call(frames, "get_frame_count")
    if frame_count is None:
        frame_count = safe_call(frames, "get_count")
    if frame_count:
        return safe_call(frames, "get_frame_by_index", 0)
    return None


def frame_shape(frame: Any) -> tuple[int | None, int | None]:
    width = safe_call(frame, "get_width")
    height = safe_call(frame, "get_height")
    return (
        int(width) if width is not None else None,
        int(height) if height is not None else None,
    )


def frame_to_preview_image(frame: Any) -> Any | None:
    import cv2
    import numpy as np

    width, height = frame_shape(frame)
    if width is None or height is None:
        return None

    fmt = enum_name(frame.get_format())
    data = np.asanyarray(frame.get_data())

    if fmt == "MJPG":
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == "RGB":
        rgb = np.resize(data, (height, width, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if fmt == "BGR":
        return np.resize(data, (height, width, 3))
    if fmt == "YUYV":
        yuyv = np.resize(data, (height, width, 2))
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    if fmt == "UYVY":
        uyvy = np.resize(data, (height, width, 2))
        return cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    if fmt in {"Y16", "YUYV_PACKED", "Y8"}:
        gray = np.resize(data, (height, width))
        return cv2.cvtColor(cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype("uint8"), cv2.COLOR_GRAY2BGR)
    return None


def test_profile(
    ob: Any,
    device: Any | None,
    sensor_key: str,
    profile: ProfileInfo,
    seconds: float,
    timeout_ms: int,
    preview: bool,
) -> TestResult:
    pipeline = make_pipeline(ob, device)
    config = ob.Config()
    started = False
    timestamps: list[float] = []
    actual_width: int | None = None
    actual_height: int | None = None

    try:
        config.enable_stream(profile.profile)
        pipeline.start(config)
        started = True

        deadline = time.monotonic() + max(seconds, 0.1)
        while time.monotonic() < deadline:
            frames = pipeline.wait_for_frames(timeout_ms)
            if frames is None:
                continue
            frame = get_expected_frame(frames, sensor_key)
            if frame is None:
                continue
            timestamps.append(time.monotonic())
            actual_width, actual_height = frame_shape(frame)

            if preview:
                import cv2

                image = frame_to_preview_image(frame)
                if image is not None:
                    max_width = 1280
                    if image.shape[1] > max_width:
                        scale = max_width / image.shape[1]
                        image = cv2.resize(image, (max_width, int(image.shape[0] * scale)))
                    cv2.imshow(f"Orbbec {sensor_key} {profile.label()}", image)
                    if cv2.waitKey(1) in (27, ord("q")):
                        break

        if preview:
            import cv2

            cv2.destroyAllWindows()

        if not timestamps:
            return TestResult(False, profile, error="stream started but no frames arrived")

        elapsed = max(timestamps[-1] - timestamps[0], 1e-9)
        measured_fps = (len(timestamps) - 1) / elapsed if len(timestamps) > 1 else 0.0
        return TestResult(
            True,
            profile,
            frames=len(timestamps),
            measured_fps=measured_fps,
            actual_width=actual_width,
            actual_height=actual_height,
        )
    except Exception as exc:
        return TestResult(False, profile, error=f"{type(exc).__name__}: {exc}")
    finally:
        if started:
            try:
                pipeline.stop()
            except Exception:
                pass


def print_profiles(title: str, profiles: list[ProfileInfo]) -> None:
    print(title)
    if not profiles:
        print("  none")
        return
    for profile in profiles:
        print(f"  [{profile.index:02d}] {profile.label()} pixels={profile.pixels}")


def main() -> int:
    args = parse_args()
    ob = import_orbbec_sdk()
    print(f"[sdk] pyorbbecsdk version={safe_call(ob, 'get_version') or 'unknown'}")

    try:
        device = select_device(ob, args.serial)
        print_device_info(device)
        sensor_type = resolve_sensor_type(ob, args.sensor)

        enum_pipeline = make_pipeline(ob, device)
        profiles = enumerate_video_profiles(enum_pipeline, sensor_type)
        print_profiles(f"[profiles] all {args.sensor} video profiles:", profiles)

        candidates = [profile for profile in profiles if profile.fps == args.fps]
        candidates.sort(key=lambda p: (p.pixels, p.width, p.height, p.format_name), reverse=True)
        print_profiles(f"[profiles] {args.sensor} profiles at exactly {args.fps} fps:", candidates)

        if args.list_only:
            return 0 if candidates else 2
        if not candidates:
            print(f"[result] no {args.sensor} profile advertises exactly {args.fps} fps")
            return 2

        print(f"[test] trying largest {args.sensor} profiles at {args.fps} fps")
        successes: list[TestResult] = []
        failures: list[TestResult] = []
        for candidate in candidates:
            print(f"  try {candidate.label()} ...", flush=True)
            result = test_profile(
                ob=ob,
                device=device,
                sensor_key=args.sensor,
                profile=candidate,
                seconds=args.seconds,
                timeout_ms=args.timeout_ms,
                preview=args.preview,
            )
            if result.ok:
                successes.append(result)
                actual = (
                    f"{result.actual_width}x{result.actual_height}"
                    if result.actual_width and result.actual_height
                    else "unknown"
                )
                print(
                    f"    OK frames={result.frames} measured_fps={result.measured_fps:.2f} "
                    f"actual_frame={actual}"
                )
                if not args.try_all:
                    break
            else:
                failures.append(result)
                print(f"    FAIL {result.error}")

        if successes:
            best = max(
                successes,
                key=lambda r: (r.profile.pixels, r.profile.width, r.profile.height, r.profile.format_name),
            )
            print(
                "[result] max verified "
                f"{args.sensor} resolution at {args.fps} fps: {best.profile.width}x{best.profile.height} "
                f"format={best.profile.format_name} measured_fps={best.measured_fps:.2f} "
                f"frames={best.frames}"
            )
            return 0

        print(f"[result] no advertised {args.fps} fps profile could be opened")
        for failure in failures:
            print(f"  {failure.profile.label()}: {failure.error}")
        return 3
    except KeyboardInterrupt:
        print("\n[stop] interrupted")
        return 130
    except Exception as exc:
        if "openUsbDevice failed" in str(exc):
            print("[hint] " + orbbec_udev_install_hint(ob).replace("\n", "\n[hint] "))
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
