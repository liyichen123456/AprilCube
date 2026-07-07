#!/usr/bin/env python3
# OpenCV camera frame:
# +x: image right
# +y: image down
# +z: camera forward
from __future__ import annotations

import argparse
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
    "cube_april_36h11_100_105_1x1x1_50mm"
)
WINDOW_NAME = "RealSense D435 AprilCube"
PINHOLE_UNDISTORT_ALPHA = 0.0


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
        "press q or ESC to quit",
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

    frame_idx = 0
    last_print = time.monotonic()
    fps_count = 0
    fps_value = 0.0
    try:
        while True:
            packs = manager.capture_frames()
            if not packs:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            for device_name, pack in packs.items():
                color = resize_if_needed(pack["color"], image_size)
                detect_frame = undistort_frame(color, undistort_pack)
                result = detector.process_frame(detect_frame, timestamp=float(pack["timestamp"]))

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
                vis = draw_text_panel(vis, lines)

                if args.show_depth:
                    depth_vis = resize_if_needed(pack["depth_colormap"], image_size)
                    vis = np.hstack([vis, depth_vis])

                cv2.imshow(f"{WINDOW_NAME}: {device_name}", vis)

                if frame_idx % max(int(args.print_every), 1) == 0:
                    print(result_to_text(device_name, cube_name, result))

            frame_idx += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        manager.stop()
        cv2.destroyAllWindows()
        print("[INFO] Released RealSense cameras.")


if __name__ == "__main__":
    main()
