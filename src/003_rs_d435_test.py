#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import sys
import time
from dataclasses import dataclass
from typing import Any


TARGET_DEVICE_HINT = "RealSense D435"
DEFAULT_FPS = 30

SENSOR_SPECS: dict[str, tuple[str, int]] = {
    "color": ("color", 0),
    "depth": ("depth", 0),
    "left_ir": ("infrared", 1),
    "right_ir": ("infrared", 2),
}


@dataclass(frozen=True)
class ProfileInfo:
    index: int
    sensor_name: str
    stream_type: Any
    stream_index: int
    width: int
    height: int
    fps: int
    format_value: Any
    format_name: str

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def label(self) -> str:
        stream_suffix = f"[{self.stream_index}]" if self.stream_index else ""
        return (
            f"{self.width}x{self.height}@{self.fps} {self.format_name} "
            f"stream={enum_name(self.stream_type)}{stream_suffix}"
        )


@dataclass(frozen=True)
class TestResult:
    ok: bool
    profile: ProfileInfo
    frames: int = 0
    measured_fps: float = 0.0
    source_fps: float = 0.0
    dropped_frames: int = 0
    actual_width: int | None = None
    actual_height: int | None = None
    actual_fps: int | None = None
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Open an Intel {TARGET_DEVICE_HINT}, enumerate its video modes, and find "
            "the largest single-stream resolution that can actually start at the "
            "requested FPS."
        )
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Exact FPS to test. Default: {DEFAULT_FPS}.",
    )
    parser.add_argument(
        "--sensor",
        choices=tuple(SENSOR_SPECS.keys()),
        default="color",
        help="Single stream to test. Default: color (RGB/visible camera).",
    )
    parser.add_argument(
        "--serial",
        type=str,
        default=None,
        help="Optional RealSense serial number. By default, select the first D435.",
    )
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
        help="Test every profile at the requested FPS instead of stopping after the first success.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only enumerate profiles and capability summaries; do not start a stream.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a small OpenCV preview while testing each profile.",
    )
    return parser.parse_args()


def import_realsense_sdk() -> Any:
    try:
        import pyrealsense2 as rs  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "pyrealsense2 is not installed in this Python environment.\n"
            "Install the Intel RealSense Python SDK first, for example:\n"
            "  python -m pip install --upgrade pyrealsense2\n"
            "Then run this script again."
        ) from exc
    return rs


def enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    text = str(value)
    return text.split(".")[-1].split(":")[0].strip("<> ")


def safe_get_info(obj: Any, info_key: Any) -> str | None:
    try:
        if hasattr(obj, "supports") and not obj.supports(info_key):
            return None
        return str(obj.get_info(info_key))
    except Exception:
        return None


def device_summary(rs: Any, device: Any) -> str:
    fields = (
        ("name", rs.camera_info.name),
        ("serial", rs.camera_info.serial_number),
        ("firmware", rs.camera_info.firmware_version),
        ("product_id", rs.camera_info.product_id),
        ("usb", rs.camera_info.usb_type_descriptor),
    )
    values = []
    for label, key in fields:
        value = safe_get_info(device, key)
        if value:
            values.append(f"{label}={value}")
    return " ".join(values)


def select_device(rs: Any, serial: str | None) -> tuple[Any, str]:
    devices = list(rs.context().query_devices())
    if not devices:
        raise RuntimeError("No Intel RealSense device found.")

    print(f"[device] found {len(devices)} RealSense device(s)")
    for idx, device in enumerate(devices):
        print(f"  [{idx}] {device_summary(rs, device)}")

    selected = None
    if serial:
        for device in devices:
            if safe_get_info(device, rs.camera_info.serial_number) == serial:
                selected = device
                break
        if selected is None:
            raise RuntimeError(f"RealSense serial {serial!r} was not found.")
    else:
        for device in devices:
            name = safe_get_info(device, rs.camera_info.name) or ""
            if "D435" in name.upper():
                selected = device
                break
        if selected is None:
            selected = devices[0]

    selected_serial = safe_get_info(selected, rs.camera_info.serial_number)
    if not selected_serial:
        raise RuntimeError("Selected RealSense device does not report a serial number.")
    print(f"[device] selected {device_summary(rs, selected)}")
    return selected, selected_serial


def resolve_stream(rs: Any, sensor_key: str) -> tuple[Any, int]:
    stream_name, stream_index = SENSOR_SPECS[sensor_key]
    return getattr(rs.stream, stream_name), stream_index


def enumerate_video_profiles(
    rs: Any,
    device: Any,
    sensor_key: str,
) -> list[ProfileInfo]:
    wanted_stream, wanted_index = resolve_stream(rs, sensor_key)
    profiles: list[ProfileInfo] = []
    seen: set[tuple[int, int, int, str, int]] = set()

    for sensor in device.query_sensors():
        sensor_name = safe_get_info(sensor, rs.camera_info.name) or "unknown sensor"
        for native_profile in sensor.get_stream_profiles():
            try:
                if native_profile.stream_type() != wanted_stream:
                    continue
                stream_index = int(native_profile.stream_index())
                if stream_index != wanted_index:
                    continue
                video = native_profile.as_video_stream_profile()
                width = int(video.width())
                height = int(video.height())
                fps = int(native_profile.fps())
                format_value = native_profile.format()
                format_name = enum_name(format_value)
            except Exception:
                continue

            key = (width, height, fps, format_name, stream_index)
            if key in seen:
                continue
            seen.add(key)
            profiles.append(
                ProfileInfo(
                    index=len(profiles),
                    sensor_name=sensor_name,
                    stream_type=wanted_stream,
                    stream_index=stream_index,
                    width=width,
                    height=height,
                    fps=fps,
                    format_value=format_value,
                    format_name=format_name,
                )
            )

    return sorted(
        profiles,
        key=lambda p: (p.pixels, p.width, p.height, p.fps, p.format_name),
        reverse=True,
    )


def capability_modes(profiles: list[ProfileInfo]) -> list[ProfileInfo]:
    """Collapse format duplicates while keeping one row per resolution/FPS mode."""
    modes: dict[tuple[int, int, int], ProfileInfo] = {}
    for profile in profiles:
        modes.setdefault((profile.width, profile.height, profile.fps), profile)
    return list(modes.values())


def pareto_modes(profiles: list[ProfileInfo]) -> list[ProfileInfo]:
    modes = capability_modes(profiles)
    frontier = []
    for profile in modes:
        dominated = any(
            other.pixels >= profile.pixels
            and other.fps >= profile.fps
            and (other.pixels > profile.pixels or other.fps > profile.fps)
            for other in modes
        )
        if not dominated:
            frontier.append(profile)
    return sorted(frontier, key=lambda p: (p.pixels, p.fps), reverse=True)


def print_capability_summary(sensor_key: str, profiles: list[ProfileInfo]) -> None:
    if not profiles:
        return

    modes = capability_modes(profiles)
    max_resolution = max(modes, key=lambda p: (p.pixels, p.fps, p.width, p.height))
    max_fps = max(modes, key=lambda p: (p.fps, p.pixels, p.width, p.height))
    print(f"[capability] {sensor_key} advertised limits (not yet stream-verified):")
    print(
        "  max-resolution mode: "
        f"{max_resolution.width}x{max_resolution.height}@{max_resolution.fps}"
    )
    print(
        "  max-FPS mode:        "
        f"{max_fps.width}x{max_fps.height}@{max_fps.fps}"
    )
    print("  resolution/FPS Pareto frontier:")
    for profile in pareto_modes(profiles):
        formats = sorted(
            {
                p.format_name
                for p in profiles
                if p.width == profile.width
                and p.height == profile.height
                and p.fps == profile.fps
            }
        )
        print(
            f"    {profile.width}x{profile.height}@{profile.fps} "
            f"formats={','.join(formats)}"
        )


def get_expected_frame(frames: Any, sensor_key: str) -> Any | None:
    if sensor_key == "color":
        return frames.get_color_frame()
    if sensor_key == "depth":
        return frames.get_depth_frame()
    if sensor_key == "left_ir":
        return frames.get_infrared_frame(1)
    if sensor_key == "right_ir":
        return frames.get_infrared_frame(2)
    return None


def frame_to_preview_image(frame: Any) -> Any | None:
    import cv2
    import numpy as np

    width = int(frame.get_width())
    height = int(frame.get_height())
    fmt = enum_name(frame.get_profile().format()).lower()
    data = np.asanyarray(frame.get_data())

    if fmt == "bgr8":
        return data.reshape(height, width, 3)
    if fmt == "rgb8":
        return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
    if fmt == "rgba8":
        return cv2.cvtColor(data.reshape(height, width, 4), cv2.COLOR_RGBA2BGR)
    if fmt == "bgra8":
        return cv2.cvtColor(data.reshape(height, width, 4), cv2.COLOR_BGRA2BGR)
    if fmt in {"yuyv", "yuy2"}:
        return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUY2)
    if fmt == "uyvy":
        return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_UYVY)
    if fmt in {"mjpeg", "mjpg"}:
        return cv2.imdecode(data.reshape(-1), cv2.IMREAD_COLOR)
    if data.size == width * height:
        gray = data.reshape(height, width)
        gray8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    return None


def test_profile(
    rs: Any,
    serial: str,
    sensor_key: str,
    profile: ProfileInfo,
    seconds: float,
    timeout_ms: int,
    preview: bool,
) -> TestResult:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(
        profile.stream_type,
        profile.stream_index,
        profile.width,
        profile.height,
        profile.format_value,
        profile.fps,
    )

    started = False
    timestamps: list[float] = []
    device_timestamps_ms: list[float] = []
    frame_numbers: list[int] = []
    actual_width: int | None = None
    actual_height: int | None = None
    actual_fps: int | None = None

    try:
        pipeline.start(config)
        started = True
        deadline = time.monotonic() + max(seconds, 0.1)
        while time.monotonic() < deadline:
            try:
                frames = pipeline.wait_for_frames(timeout_ms)
            except RuntimeError as exc:
                if "Frame didn't arrive" in str(exc):
                    continue
                raise

            frame = get_expected_frame(frames, sensor_key)
            if frame is None:
                continue
            timestamps.append(time.monotonic())
            device_timestamps_ms.append(float(frame.get_timestamp()))
            frame_numbers.append(int(frame.get_frame_number()))
            actual_width = int(frame.get_width())
            actual_height = int(frame.get_height())
            actual_fps = int(frame.get_profile().fps())

            if preview:
                import cv2

                image = frame_to_preview_image(frame)
                if image is not None:
                    max_width = 1280
                    if image.shape[1] > max_width:
                        scale = max_width / image.shape[1]
                        image = cv2.resize(image, (max_width, int(image.shape[0] * scale)))
                    cv2.imshow(f"RealSense {sensor_key} {profile.label()}", image)
                    if cv2.waitKey(1) in (27, ord("q")):
                        break

        if preview:
            import cv2

            cv2.destroyAllWindows()

        if not timestamps:
            return TestResult(False, profile, error="stream started but no frames arrived")

        elapsed = max(timestamps[-1] - timestamps[0], 1e-9)
        measured_fps = (len(timestamps) - 1) / elapsed if len(timestamps) > 1 else 0.0
        frame_span = (
            frame_numbers[-1] - frame_numbers[0]
            if len(frame_numbers) > 1
            else 0
        )
        device_elapsed = (
            (device_timestamps_ms[-1] - device_timestamps_ms[0]) / 1000.0
            if len(device_timestamps_ms) > 1
            else 0.0
        )
        source_fps = frame_span / device_elapsed if frame_span > 0 and device_elapsed > 0 else 0.0
        dropped_frames = max(frame_span + 1 - len(frame_numbers), 0)
        return TestResult(
            True,
            profile,
            frames=len(timestamps),
            measured_fps=measured_fps,
            source_fps=source_fps,
            dropped_frames=dropped_frames,
            actual_width=actual_width,
            actual_height=actual_height,
            actual_fps=actual_fps,
        )
    except Exception as exc:
        return TestResult(False, profile, error=f"{type(exc).__name__}: {exc}")
    finally:
        if started:
            try:
                pipeline.stop()
            except Exception:
                pass
        if preview:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass


def print_profiles(title: str, profiles: list[ProfileInfo]) -> None:
    print(title)
    if not profiles:
        print("  none")
        return
    for profile in profiles:
        print(
            f"  [{profile.index:02d}] {profile.label()} pixels={profile.pixels} "
            f"sensor={profile.sensor_name}"
        )


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        print("[error] --fps must be greater than zero", file=sys.stderr)
        return 2
    if args.seconds <= 0:
        print("[error] --seconds must be greater than zero", file=sys.stderr)
        return 2
    if args.timeout_ms <= 0:
        print("[error] --timeout-ms must be greater than zero", file=sys.stderr)
        return 2

    rs = import_realsense_sdk()
    try:
        sdk_version = importlib.metadata.version("pyrealsense2")
    except importlib.metadata.PackageNotFoundError:
        sdk_version = getattr(rs, "__version__", "unknown")
    print(f"[sdk] pyrealsense2 version={sdk_version}")

    try:
        device, serial = select_device(rs, args.serial)
        profiles = enumerate_video_profiles(rs, device, args.sensor)
        print_profiles(f"[profiles] all {args.sensor} video profiles:", profiles)
        print_capability_summary(args.sensor, profiles)

        candidates = [profile for profile in profiles if profile.fps == args.fps]
        candidates.sort(
            key=lambda p: (p.pixels, p.width, p.height, p.format_name),
            reverse=True,
        )
        print_profiles(
            f"[profiles] {args.sensor} profiles at exactly {args.fps} fps:",
            candidates,
        )

        if args.list_only:
            return 0 if candidates else 2
        if not candidates:
            available_fps = sorted({profile.fps for profile in profiles})
            print(
                f"[result] no {args.sensor} profile advertises exactly {args.fps} fps; "
                f"available FPS values: {available_fps}"
            )
            return 2

        print(f"[test] trying largest {args.sensor} profiles at {args.fps} fps")
        successes: list[TestResult] = []
        failures: list[TestResult] = []
        for candidate in candidates:
            print(f"  try {candidate.label()} ...", flush=True)
            result = test_profile(
                rs=rs,
                serial=serial,
                sensor_key=args.sensor,
                profile=candidate,
                seconds=args.seconds,
                timeout_ms=args.timeout_ms,
                preview=args.preview,
            )
            if result.ok:
                successes.append(result)
                actual = (
                    f"{result.actual_width}x{result.actual_height}@{result.actual_fps}"
                    if result.actual_width and result.actual_height and result.actual_fps
                    else "unknown"
                )
                print(
                    f"    OK frames={result.frames} delivered_fps={result.measured_fps:.2f} "
                    f"source_fps={result.source_fps:.2f} dropped={result.dropped_frames} "
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
                key=lambda r: (
                    r.profile.pixels,
                    r.profile.width,
                    r.profile.height,
                    r.profile.format_name,
                ),
            )
            print(
                "[result] max verified "
                f"{args.sensor} resolution at {args.fps} fps: "
                f"{best.profile.width}x{best.profile.height} "
                f"format={best.profile.format_name} "
                f"delivered_fps={best.measured_fps:.2f} source_fps={best.source_fps:.2f} "
                f"dropped={best.dropped_frames} frames={best.frames}"
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
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
