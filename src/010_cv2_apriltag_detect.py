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
THIRDPARTY_DIR = THIS_FILE.parent.parent.parent
PROJECT_ROOT = THIRDPARTY_DIR.parent
RECORDER_UTILS_DIR = PROJECT_ROOT / "scripts" / "utils"

if str(RECORDER_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(RECORDER_UTILS_DIR))

from recorder_cv2_cam import CV2CameraManager  # noqa: E402


def load_intrinsics_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path).expanduser().resolve()
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    dist = data.get("dist", data.get("D", None))
    if dist is None:
        dist = np.zeros(5, dtype=np.float64)

    return {
        "path": str(yaml_path),
        "camera_model": str(data.get("camera_model", "")),
        "distortion_model": str(data.get("distortion_model", "")),
        "image_size": tuple(int(v) for v in data["image_size"]),
        "K": np.asarray(data["K"], dtype=np.float64).reshape(3, 3),
        "dist": np.asarray(dist, dtype=np.float64).reshape(-1),
    }


# ============================================================
# User macros
# ============================================================

CAMERA_TO_PORT: dict[str, str] = {
    "cam0": "3-10.1:1.0",
}

CAMERA_TO_INTRINSICS_YAML: dict[str, str] = {
    "cam0": "/home/ps/RobotCamCalib1/outputs/intrinsics_cam0_fisheye_2592x1944_0703_230535.yaml",
}

ACTIVE_CAMERA_NAMES: list[str] = ["cam0"]

FPS = 120
FOURCC = "MJPG"
WINDOW_PREFIX = "CV2 AprilTag 36h11"
PRINT_EVERY_N_FRAMES = 10
UNDISTORT_BEFORE_DETECTION = True
FISHEYE_RECTIFIED_HORIZONTAL_FOV_DEG = 120.0
PINHOLE_UNDISTORT_ALPHA = 0.0


def is_fisheye_calib(calib: dict[str, Any]) -> bool:
    camera_model = str(calib.get("camera_model", "")).lower()
    distortion_model = str(calib.get("distortion_model", "")).lower()
    return camera_model == "fisheye" or distortion_model == "opencv_fisheye"


def make_centered_pinhole_camera_matrix(
    image_size: tuple[int, int],
    horizontal_fov_deg: float,
) -> np.ndarray:
    width, height = image_size
    half_fov_rad = np.radians(horizontal_fov_deg) / 2.0
    if not 0.0 < half_fov_rad < (np.pi / 2.0):
        raise ValueError(f"horizontal_fov_deg must be in (0, 180), got {horizontal_fov_deg}.")

    focal = width / (2.0 * np.tan(half_fov_rad))
    return np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def compute_detection_camera_matrix(
    calib: dict[str, Any],
    image_size: tuple[int, int],
    *,
    undistort_before_detection: bool,
) -> np.ndarray:
    camera_matrix = np.asarray(calib["K"], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(calib.get("dist", np.zeros(5)), dtype=np.float64).reshape(-1)
    if (
        not undistort_before_detection
        or dist_coeffs.size == 0
        or np.allclose(dist_coeffs, 0.0)
    ):
        return camera_matrix.copy()

    if is_fisheye_calib(calib):
        if dist_coeffs.size != 4:
            raise ValueError(f"OpenCV fisheye calibration expects 4 coeffs, got {dist_coeffs.size}.")
        return make_centered_pinhole_camera_matrix(
            image_size,
            FISHEYE_RECTIFIED_HORIZONTAL_FOV_DEG,
        )

    new_camera_matrix, _roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        PINHOLE_UNDISTORT_ALPHA,
        image_size,
    )
    return np.asarray(new_camera_matrix, dtype=np.float64).reshape(3, 3)


def create_undistort_maps(
    calib: dict[str, Any],
    image_size: tuple[int, int],
    detection_camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    camera_matrix = np.asarray(calib["K"], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(calib.get("dist", np.zeros(5)), dtype=np.float64).reshape(-1)
    if dist_coeffs.size == 0 or np.allclose(dist_coeffs, 0.0):
        return None

    detection_camera_matrix = np.asarray(detection_camera_matrix, dtype=np.float64).reshape(3, 3)
    if is_fisheye_calib(calib):
        if dist_coeffs.size != 4:
            raise ValueError(f"OpenCV fisheye calibration expects 4 coeffs, got {dist_coeffs.size}.")
        return cv2.fisheye.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs.reshape(4, 1),
            np.eye(3, dtype=np.float64),
            detection_camera_matrix,
            image_size,
            cv2.CV_16SC2,
        )

    return cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        np.eye(3, dtype=np.float64),
        detection_camera_matrix,
        image_size,
        cv2.CV_16SC2,
    )


def undistort_frame(
    frame: np.ndarray,
    undistort_maps: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    if undistort_maps is None:
        return frame
    map1, map2 = undistort_maps
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)


def get_apriltag_36h11_dictionary() -> Any:
    dict_id = getattr(cv2.aruco, "DICT_APRILTAG_36h11", None)
    if dict_id is None:
        dict_id = getattr(cv2.aruco, "DICT_APRILTAG_36H11", None)
    if dict_id is None:
        raise RuntimeError("This OpenCV build does not expose DICT_APRILTAG_36h11.")
    return cv2.aruco.getPredefinedDictionary(dict_id)


def create_apriltag_detector(*, fast: bool) -> Any:
    dictionary = get_apriltag_36h11_dictionary()
    params = cv2.aruco.DetectorParameters()

    if fast:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        params.adaptiveThreshWinSizeMin = 5
        params.adaptiveThreshWinSizeMax = 41
        params.adaptiveThreshWinSizeStep = 12
        params.minMarkerPerimeterRate = 0.03
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.05
        params.minCornerDistanceRate = 0.05
        params.minDistanceToBorder = 2
        params.perspectiveRemovePixelPerCell = 6
        params.perspectiveRemoveIgnoredMarginPerCell = 0.13
        params.errorCorrectionRate = 0.6
    else:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 50
        params.cornerRefinementMinAccuracy = 0.01
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 53
        params.adaptiveThreshWinSizeStep = 4
        params.minMarkerPerimeterRate = 0.01
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.05
        params.minCornerDistanceRate = 0.02
        params.minDistanceToBorder = 1
        params.perspectiveRemovePixelPerCell = 8
        params.perspectiveRemoveIgnoredMarginPerCell = 0.13
        params.errorCorrectionRate = 0.6

    return cv2.aruco.ArucoDetector(dictionary, params)


def preprocess_tag_image(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def detect_markers(detector: Any, gray: np.ndarray) -> tuple[list[np.ndarray], list[int], int]:
    enhanced = preprocess_tag_image(gray)
    try:
        corners_list, ids, rejected = detector.detectMarkers(enhanced)
    except cv2.error:
        return [], [], 0

    corners = [corner.reshape(4, 2) for corner in corners_list]
    tag_ids = [] if ids is None else [int(x[0]) for x in ids]
    return corners, tag_ids, len(rejected or [])


def draw_detections(
    image: np.ndarray,
    corners_list: list[np.ndarray],
    tag_ids: list[int],
) -> np.ndarray:
    vis = image.copy()
    for corners, tag_id in zip(corners_list, tag_ids, strict=False):
        pts = corners.astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 255), 3)
        for idx, point in enumerate(pts):
            color = (0, 255, 0) if idx == 0 else (255, 255, 0)
            cv2.circle(vis, tuple(point), 4, color, -1)

        center = np.mean(corners, axis=0).astype(int)
        cv2.circle(vis, tuple(center), 4, (0, 0, 255), -1)
        cv2.putText(
            vis,
            f"id={tag_id}",
            tuple(center + np.array([8, -8])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return vis


def draw_text_panel(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    y = 28
    for line in lines:
        cv2.putText(
            out,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 26
    return out


def resolve_common_image_size(calib_by_camera: dict[str, dict[str, Any]]) -> tuple[int, int]:
    image_sizes = {
        camera_name: tuple(int(v) for v in calib["image_size"])
        for camera_name, calib in calib_by_camera.items()
    }
    unique_sizes = set(image_sizes.values())
    if len(unique_sizes) != 1:
        raise ValueError(f"Active cameras use different image sizes: {image_sizes}")
    return next(iter(unique_sizes))


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect all AprilTag 36h11 markers with OpenCV.")
    parser.add_argument(
        "--cameras",
        type=str,
        default=",".join(ACTIVE_CAMERA_NAMES),
        help="Comma-separated logical camera names.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use faster detector parameters without sub-pixel corner refinement.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Detect on the raw camera frame instead of the rectified frame.",
    )
    args = parser.parse_args()

    active_camera_names = [x.strip() for x in args.cameras.split(",") if x.strip()]
    if not active_camera_names:
        print("[ERROR] No active camera names specified.")
        sys.exit(1)

    missing_camera_cfg = [name for name in active_camera_names if name not in CAMERA_TO_PORT]
    if missing_camera_cfg:
        print(f"[ERROR] Missing CAMERA_TO_PORT entries for: {missing_camera_cfg}")
        sys.exit(1)
    missing_intrinsics_cfg = [
        name for name in active_camera_names if name not in CAMERA_TO_INTRINSICS_YAML
    ]
    if missing_intrinsics_cfg:
        print(f"[ERROR] Missing CAMERA_TO_INTRINSICS_YAML entries for: {missing_intrinsics_cfg}")
        sys.exit(1)

    calib_by_camera = {
        name: load_intrinsics_yaml(CAMERA_TO_INTRINSICS_YAML[name])
        for name in active_camera_names
    }
    image_size = resolve_common_image_size(calib_by_camera)
    capture_size = image_size
    detect_img_size = image_size
    vis_img_size = (max(1, detect_img_size[0] // 2), max(1, detect_img_size[1] // 2))
    use_undistort = UNDISTORT_BEFORE_DETECTION and not args.raw

    detection_camera_matrix_by_camera = {
        camera_name: compute_detection_camera_matrix(
            calib,
            detect_img_size,
            undistort_before_detection=use_undistort,
        )
        for camera_name, calib in calib_by_camera.items()
    }
    undistort_maps_by_camera = {
        camera_name: create_undistort_maps(
            calib,
            detect_img_size,
            detection_camera_matrix_by_camera[camera_name],
        )
        if use_undistort
        else None
        for camera_name, calib in calib_by_camera.items()
    }
    detector = create_apriltag_detector(fast=args.fast)

    for camera_name in active_camera_names:
        calib = calib_by_camera[camera_name]
        print(
            f"[INFO] [{camera_name}] intrinsics_yaml={calib['path']} "
            f"image_size={calib['image_size']} "
            f"camera_model={calib['camera_model'] or 'unknown'} "
            f"distortion_model={calib['distortion_model'] or 'unknown'} "
            f"detect_mode={'raw' if args.raw else 'rectified'} "
            f"detector={'fast' if args.fast else 'precise'}"
        )
    print(
        f"[INFO] capture_size={capture_size} "
        f"detect_img_size={detect_img_size} vis_img_size={vis_img_size}"
    )

    camera_manager = CV2CameraManager(
        camera_to_port={name: CAMERA_TO_PORT[name] for name in active_camera_names},
        capture_size=capture_size,
        fps=FPS,
        fourcc=FOURCC,
    )

    try:
        opened = camera_manager.open_all_cameras()
        if opened == 0:
            print("[ERROR] No CV2 camera opened.")
            sys.exit(1)

        opened_names = camera_manager.get_active_camera_names()
        print(f"[INFO] Opened CV2 cameras: {opened_names}")
        print("[INFO] Detecting OpenCV ArUco AprilTag dictionary: DICT_APRILTAG_36h11")
        print("[INFO] Press 'q' or ESC to quit.")

        frame_idx = 0
        last_no_frame_print_time = 0.0
        while True:
            frame_idx += 1
            frames, origin_frames, _timestamps = camera_manager.get_frames(
                camera_names=opened_names,
                img_size=detect_img_size,
            )
            if not frames:
                now = time.time()
                if now - last_no_frame_print_time > 1.0:
                    print("[INFO] No frames received yet.")
                    last_no_frame_print_time = now
                key = cv2.waitKey(1)
                if key == 27 or key == ord("q"):
                    break
                continue

            for camera_name, frame in frames.items():
                origin_frame = origin_frames.get(camera_name)
                if origin_frame is not None and frame_idx % PRINT_EVERY_N_FRAMES == 0:
                    origin_h, origin_w = origin_frame.shape[:2]
                    detect_h, detect_w = frame.shape[:2]
                    print(
                        f"[{camera_name}] origin_size=({origin_w}, {origin_h}) "
                        f"detect_frame_size=({detect_w}, {detect_h})"
                    )

                detect_frame = frame
                if use_undistort:
                    detect_frame = undistort_frame(
                        frame,
                        undistort_maps_by_camera[camera_name],
                    )

                gray = (
                    cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
                    if len(detect_frame.shape) == 3
                    else detect_frame
                )
                corners_list, tag_ids, n_rejected = detect_markers(detector, gray)
                vis = cv2.cvtColor(preprocess_tag_image(gray), cv2.COLOR_GRAY2BGR)
                vis = draw_detections(vis, corners_list, tag_ids)

                fps_text = camera_manager.get_latest_fps(camera_name)
                ids_text = ",".join(str(x) for x in tag_ids) if tag_ids else "none"
                status_lines = [
                    f"[{camera_name}] AprilTag 36h11 mode={'raw' if args.raw else 'rectified'} "
                    f"detector={'fast' if args.fast else 'precise'} fps={fps_text:.1f}"
                    if fps_text is not None
                    else f"[{camera_name}] AprilTag 36h11 mode={'raw' if args.raw else 'rectified'} "
                    f"detector={'fast' if args.fast else 'precise'}",
                    f"detected={len(tag_ids)} rejected={n_rejected} ids=[{ids_text}]",
                    "green dot = corner 0, red dot = center",
                    "press q or ESC to quit",
                ]
                vis = draw_text_panel(vis, status_lines)

                if frame_idx % PRINT_EVERY_N_FRAMES == 0:
                    print(
                        f"[{camera_name}] detected={len(tag_ids)} "
                        f"rejected={n_rejected} ids=[{ids_text}]"
                    )

                vis = cv2.resize(vis, vis_img_size, interpolation=cv2.INTER_AREA)
                cv2.imshow(f"{WINDOW_PREFIX}: {camera_name}", vis)

            key = cv2.waitKey(1)
            if key == 27 or key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        camera_manager.release_all()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
