#!/usr/bin/env python3
# OpenCV camera frame:
# +x: image right
# +y: image down
# +z: camera forward
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

THIS_FILE = Path(__file__).resolve()
APRILCUBE_ROOT = THIS_FILE.parent.parent
THIRDPARTY_DIR = APRILCUBE_ROOT.parent
PROJECT_ROOT = THIRDPARTY_DIR.parent
RECORDER_UTILS_DIR = PROJECT_ROOT / "scripts" / "utils"

if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))
if str(RECORDER_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(RECORDER_UTILS_DIR))

import aprilcube  # noqa: E402
from aprilcube.detect import _preprocess as preprocess_tag_image  # noqa: E402
from recorder_rs import RealSenseManager  # noqa: E402


DEFAULT_INTRINSICS_YAML = Path("/home/ps/RobotCamCalib1/outputs/intrinsics_realsense_1280x720_0707_171032.yaml")
DEFAULT_CUBE_CFG = Path(
    "/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/cubes/"
    "cube_april_36h11_100_123_2x2x2_outer62p5mm"
)
WINDOW_NAME = "RealSense D435 AprilCube"
PINHOLE_UNDISTORT_ALPHA = 0.0
RECORD_OUTPUT_DIR = APRILCUBE_ROOT / "recordings"


def load_intrinsics_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path).expanduser().resolve()
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    dist = data.get("dist", data.get("D", None))
    if dist is None:
        dist = np.zeros(5, dtype=np.float64)

    return {
        "path": str(yaml_path),
        "camera_model": str(data.get("camera_model", "pinhole")),
        "distortion_model": str(data.get("distortion_model", "")),
        "image_size": tuple(int(v) for v in data["image_size"]),
        "K": np.asarray(data["K"], dtype=np.float64).reshape(3, 3),
        "dist": np.asarray(dist, dtype=np.float64).reshape(-1),
    }


def camera_matrix_to_intrinsic_dict(k: np.ndarray) -> dict[str, float]:
    return {
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
    }


def create_undistort_maps(
    calib: dict[str, Any],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    camera_matrix = np.asarray(calib["K"], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(calib.get("dist", np.zeros(5)), dtype=np.float64).reshape(-1)
    if dist_coeffs.size == 0 or np.allclose(dist_coeffs, 0.0):
        return None

    detection_camera_matrix, _roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        PINHOLE_UNDISTORT_ALPHA,
        image_size,
    )
    detection_camera_matrix = np.asarray(detection_camera_matrix, dtype=np.float64).reshape(3, 3)
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        np.eye(3, dtype=np.float64),
        detection_camera_matrix,
        image_size,
        cv2.CV_16SC2,
    )
    return map1, map2, detection_camera_matrix


def undistort_frame(
    frame: np.ndarray,
    undistort_pack: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    if undistort_pack is None:
        return frame
    map1, map2, _new_camera_matrix = undistort_pack
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)


def make_detector_input_vis(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    enhanced = preprocess_tag_image(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def result_to_text(device_name: str, cube_name: str, result: dict[str, Any]) -> str:
    if not result.get("success", False):
        return (
            f"[{device_name}][{cube_name}] cube not detected "
            f"tags={int(result.get('n_tags', 0))}"
        )

    tvec = np.asarray(result["tvec"], dtype=np.float64).reshape(-1)
    rot_mat, _ = cv2.Rodrigues(np.asarray(result["rvec"], dtype=np.float64).reshape(3, 1))
    sy = float(np.sqrt(rot_mat[0, 0] * rot_mat[0, 0] + rot_mat[1, 0] * rot_mat[1, 0]))
    if sy < 1e-6:
        euler = np.array([
            np.arctan2(-rot_mat[1, 2], rot_mat[1, 1]),
            np.arctan2(-rot_mat[2, 0], sy),
            0.0,
        ])
    else:
        euler = np.array([
            np.arctan2(rot_mat[2, 1], rot_mat[2, 2]),
            np.arctan2(-rot_mat[2, 0], sy),
            np.arctan2(rot_mat[1, 0], rot_mat[0, 0]),
        ])
    euler_deg = np.degrees(euler)
    faces = sorted(list(result.get("visible_faces", set())))
    text = (
        f"[{device_name}][{cube_name}] "
        f"t=({tvec[0]:.1f},{tvec[1]:.1f},{tvec[2]:.1f})mm "
        f"rot=({euler_deg[0]:.1f},{euler_deg[1]:.1f},{euler_deg[2]:.1f}) "
        f"reproj={float(result.get('reproj_error', float('inf'))):.2f}px "
        f"tags={int(result.get('n_tags', 0))} faces={faces}"
    )
    if result.get("single_tag_cfg_pose", False):
        text += (
            " single_tag_cfg_pose"
            f"(id={result.get('single_tag_id', '?')},face={result.get('single_tag_face', '?')})"
        )
    if result.get("predicted", False):
        text += " predicted"
    return text


def draw_text_panel(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    vis = frame.copy()
    y = 24
    for line in lines:
        cv2.putText(
            vis,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24
    cv2.putText(
        vis,
        "press s start rec, p stop/save rec, q or ESC quit",
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


def resize_if_needed(frame: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = image_size
    h, w = frame.shape[:2]
    if (w, h) == (target_w, target_h):
        return frame
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)


class RawFramePklRecorder:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.path: Path | None = None
        self._metadata: dict[str, Any] | None = None
        self._frames: list[dict[str, Any]] = []
        self.started_wall_time: str | None = None
        self.started_monotonic: float | None = None

    @property
    def is_recording(self) -> bool:
        return self._metadata is not None

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def buffered_bytes(self) -> int:
        return int(sum(frame["image_bgr"].nbytes for frame in self._frames))

    def start(self, metadata: dict[str, Any]) -> None:
        if self.is_recording:
            print(f"[INFO] Recording already active: {self.path}")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = self.output_dir / f"012_rs_raw_frames_{stamp}.pkl"
        self.started_wall_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.started_monotonic = time.perf_counter()
        self._frames = []
        self._metadata = dict(metadata)
        print(f"[INFO] Started raw-frame memory buffering: {self.path}")

    def write(
        self,
        *,
        device_name: str,
        loop_frame_idx: int,
        image_bgr: np.ndarray | None,
        capture_timestamp: float | None,
    ) -> None:
        if not self.is_recording or image_bgr is None:
            return

        image_copy = np.array(image_bgr, copy=True)
        self._frames.append(
            {
                "type": "frame",
                "device_name": device_name,
                "camera_name": device_name,
                "loop_frame_idx": int(loop_frame_idx),
                "capture_timestamp": None
                if capture_timestamp is None
                else float(capture_timestamp),
                "write_monotonic": float(time.perf_counter()),
                "shape": tuple(int(v) for v in image_copy.shape),
                "dtype": str(image_copy.dtype),
                "image_bgr": image_copy,
            }
        )

    def _print_save_progress(self, done: int, total: int) -> None:
        width = 36
        ratio = 1.0 if total <= 0 else done / total
        filled = int(round(width * ratio))
        bar = "#" * filled + "-" * (width - filled)
        sys.stdout.write(f"\r[INFO] Saving PKL [{bar}] {done}/{total} frames")
        sys.stdout.flush()
        if done >= total:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def stop(self, reason: str = "user_stop") -> None:
        if not self.is_recording:
            print("[INFO] Recording is not active.")
            return

        path = self.path
        assert path is not None
        assert self._metadata is not None

        total_frames = self.frame_count
        buffered_gb = self.buffered_bytes / (1024**3)
        elapsed = (
            time.perf_counter() - self.started_monotonic
            if self.started_monotonic is not None
            else 0.0
        )
        print(
            f"[INFO] Stopped raw-frame buffering: frames={total_frames} "
            f"buffered={buffered_gb:.2f} GiB duration={elapsed:.2f}s"
        )
        print(f"[INFO] Writing PKL: {path}")

        with path.open("wb") as f:
            pickle.dump(
                {
                    "type": "header",
                    "format": "aprilcube_rs_raw_frame_stream_v1",
                    "created_wall_time": self.started_wall_time,
                    "metadata": self._metadata,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            self._print_save_progress(0, total_frames)
            for idx, frame_record in enumerate(self._frames, start=1):
                pickle.dump(frame_record, f, protocol=pickle.HIGHEST_PROTOCOL)
                if idx == total_frames or idx % 10 == 0:
                    self._print_save_progress(idx, total_frames)
            pickle.dump(
                {
                    "type": "footer",
                    "reason": reason,
                    "frame_count": int(total_frames),
                    "stopped_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        self._frames = []
        self._metadata = None
        self.started_monotonic = None
        print(f"[INFO] Saved raw-frame PKL recording: {path} frames={total_frames}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect one AprilCube from Intel RealSense D435 color stream."
    )
    parser.add_argument("--intrinsics-yaml", type=Path, default=DEFAULT_INTRINSICS_YAML)
    parser.add_argument("--cube-cfg", type=Path, default=DEFAULT_CUBE_CFG)
    parser.add_argument("--fps", type=int, default=15, help="Requested RealSense stream FPS.")
    parser.add_argument("--serial", type=str, default=None, help="Optional RealSense serial number.")
    parser.add_argument("--no-undistort", action="store_true", help="Use raw color image and YAML dist coeffs.")
    parser.add_argument("--slow", action="store_true", help="Use slower high-accuracy AprilTag detector settings.")
    parser.add_argument("--no-filter", action="store_true", help="Disable AprilCube temporal pose filter.")
    parser.add_argument("--prefer-mjpeg", action="store_true", help="Ask recorder_rs to prefer MJPEG color stream.")
    parser.add_argument("--show-depth", action="store_true", help="Append depth colormap next to color visualization.")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=RECORD_OUTPUT_DIR,
        help="Directory for raw color-frame PKL recordings triggered by s/p.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calib = load_intrinsics_yaml(args.intrinsics_yaml)
    image_size = tuple(int(v) for v in calib["image_size"])
    raw_camera_matrix = np.asarray(calib["K"], dtype=np.float64).reshape(3, 3)
    raw_dist_coeffs = np.asarray(calib["dist"], dtype=np.float64).reshape(-1)

    undistort_pack = None
    detection_camera_matrix = raw_camera_matrix.copy()
    detector_dist_coeffs = raw_dist_coeffs
    if not args.no_undistort:
        undistort_pack = create_undistort_maps(calib, image_size)
        if undistort_pack is not None:
            detection_camera_matrix = undistort_pack[2]
            detector_dist_coeffs = np.zeros(5, dtype=np.float64)

    cube_cfg = args.cube_cfg.expanduser().resolve()
    if cube_cfg.is_dir():
        cube_name = cube_cfg.name
    elif cube_cfg.name == "config.json":
        cube_name = cube_cfg.parent.name
    else:
        raise FileNotFoundError(f"Invalid cube cfg path: {cube_cfg}")

    detector = aprilcube.detector(
        cube_cfg,
        intrinsic_cfg=camera_matrix_to_intrinsic_dict(detection_camera_matrix),
        dist_coeffs=detector_dist_coeffs,
        enable_filter=not args.no_filter,
        fast=not args.slow,
    )

    device_serials = [args.serial] if args.serial else None
    manager = RealSenseManager(
        desired_width=image_size[0],
        desired_height=image_size[1],
        desired_fps=args.fps,
        device_serials=device_serials,
        align_to_color=True,
        prefer_mjpeg=bool(args.prefer_mjpeg),
        verbose=True,
    )
    if not manager.is_enabled():
        raise RuntimeError("RealSenseManager is not enabled. Check pyrealsense2 and D435 connection.")

    print(
        f"[INFO] intrinsics_yaml={Path(args.intrinsics_yaml).expanduser().resolve()} "
        f"image_size={image_size} camera_model={calib['camera_model']} "
        f"undistort={not args.no_undistort}"
    )
    print(
        f"[INFO] raw_K fx={raw_camera_matrix[0,0]:.3f} fy={raw_camera_matrix[1,1]:.3f} "
        f"cx={raw_camera_matrix[0,2]:.3f} cy={raw_camera_matrix[1,2]:.3f}"
    )
    print(
        f"[INFO] detection_K fx={detection_camera_matrix[0,0]:.3f} "
        f"fy={detection_camera_matrix[1,1]:.3f} "
        f"cx={detection_camera_matrix[0,2]:.3f} cy={detection_camera_matrix[1,2]:.3f}"
    )
    print(f"[INFO] cube_cfg={cube_cfg}")
    print("[INFO] Press 's' to start raw-frame PKL recording, 'p' to stop/save, 'q' or ESC to quit.")

    recorder = RawFramePklRecorder(Path(args.record_dir))
    recording_metadata = {
        "script": str(THIS_FILE),
        "recorded_image": "raw_realsense_color_bgr",
        "intrinsics_yaml": str(Path(args.intrinsics_yaml).expanduser().resolve()),
        "cube_cfg": str(cube_cfg),
        "image_size": tuple(int(v) for v in image_size),
        "fps": int(args.fps),
        "serial": None if args.serial is None else str(args.serial),
        "prefer_mjpeg": bool(args.prefer_mjpeg),
        "show_depth": bool(args.show_depth),
        "undistort_for_detection": bool(not args.no_undistort),
        "raw_camera_matrix": raw_camera_matrix.tolist(),
        "raw_dist_coeffs": raw_dist_coeffs.tolist(),
        "detection_camera_matrix": detection_camera_matrix.tolist(),
        "detector_dist_coeffs": detector_dist_coeffs.tolist(),
    }

    def handle_key(key: int) -> bool:
        if key in (ord("q"), 27):
            return False
        if key == ord("s"):
            recorder.start(recording_metadata)
        elif key == ord("p"):
            recorder.stop("user_stop")
        return True

    frame_idx = 0
    last_print = time.monotonic()
    fps_count = 0
    fps_value = 0.0
    try:
        while True:
            packs = manager.capture_frames()
            if not packs:
                key = cv2.waitKey(1) & 0xFF
                if not handle_key(key):
                    break
                continue

            for device_name, pack in packs.items():
                raw_color = pack["color"]
                capture_timestamp = pack.get("timestamp", None)
                color = resize_if_needed(raw_color, image_size)
                recorder.write(
                    device_name=device_name,
                    loop_frame_idx=frame_idx,
                    image_bgr=raw_color,
                    capture_timestamp=None
                    if capture_timestamp is None
                    else float(capture_timestamp),
                )
                detect_frame = undistort_frame(color, undistort_pack)
                result = detector.process_frame(
                    detect_frame,
                    timestamp=time.monotonic()
                    if capture_timestamp is None
                    else float(capture_timestamp),
                )

                vis = make_detector_input_vis(detect_frame)
                vis = detector.draw_result(vis, result)
                fps_count += 1
                now = time.monotonic()
                if now - last_print >= 1.0:
                    fps_value = fps_count / max(now - last_print, 1e-9)
                    fps_count = 0
                    last_print = now

                lines = [
                    f"[{device_name}] RealSense AprilCube fps={fps_value:.1f} "
                    f"detect_size={image_size}",
                    result_to_text(device_name, cube_name, result),
                ]
                if recorder.is_recording:
                    buffered_gb = recorder.buffered_bytes / (1024**3)
                    lines.append(
                        f"REC buffering frames={recorder.frame_count} mem={buffered_gb:.2f}GiB"
                    )
                else:
                    lines.append("REC off: press s to start, p to stop/save")
                vis = draw_text_panel(vis, lines)

                if args.show_depth:
                    depth_vis = resize_if_needed(pack["depth_colormap"], image_size)
                    vis = np.hstack([vis, depth_vis])

                cv2.imshow(f"{WINDOW_NAME}: {device_name}", vis)

                if frame_idx % max(int(args.print_every), 1) == 0:
                    print(result_to_text(device_name, cube_name, result))

            frame_idx += 1
            key = cv2.waitKey(1) & 0xFF
            if not handle_key(key):
                break
    finally:
        if recorder.is_recording:
            recorder.stop("shutdown")
        manager.stop()
        cv2.destroyAllWindows()
        print("[INFO] Released RealSense cameras.")


if __name__ == "__main__":
    main()
