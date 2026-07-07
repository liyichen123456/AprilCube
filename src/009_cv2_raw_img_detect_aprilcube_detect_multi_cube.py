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

import aprilcube  # noqa: E402
from aprilcube.detect import (  # noqa: E402
    FACE_DEFS,
    _preprocess as preprocess_tag_image,
    _quad_quality,
    estimate_pose,
)
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
    # "cam0": "/home/ps/RobotCamCalib1/outputs/intrinsics_cam0_fisheye_2592x1944_0618_181728.yaml",  1.1鱼眼
    # "cam0": "/home/ps/RobotCamCalib1/outputs/intrinsics_cam0_fisheye_2592x1944_0703_210450.yaml",
    "cam0": "/home/ps/RobotCamCalib1/outputs/intrinsics_cam0_fisheye_2592x1944_0703_230535.yaml",  # 180 degree

}

ACTIVE_CAMERA_NAMES: list[str] = ["cam0"]

FPS = 120
FOURCC = "MJPG"
WINDOW_PREFIX = "CV2 Raw Fisheye AprilCube"
PRINT_EVERY_N_FRAMES = 5

CUBE_CFG_DIRS: list[Path] = [
    THIRDPARTY_DIR / "aprilcube" / "cubes" / "cube_april_36h11_0_5_1x1x1_10mm",
    THIRDPARTY_DIR / "aprilcube" / "cubes" / "cube_april_36h11_6_11_1x1x1_10mm",
    # THIRDPARTY_DIR / "aprilcube" / "cubes" / "cube_april_36h11_12_17_1x1x1_10mm",
    # THIRDPARTY_DIR / "aprilcube" / "cubes" / "cube_april_36h11_18_23_1x1x1_10mm",
    # THIRDPARTY_DIR / "aprilcube" / "cubes" / "cube_april_36h11_24_29_1x1x1_10mm",
    # THIRDPARTY_DIR / "aprilcube" / "cubes" / "cube_april_36h11_30_35_1x1x1_10mm",
]

FAST_DETECTOR = True
MAX_RAW_REPROJ_ERROR_PX = 8.0


def camera_matrix_to_intrinsic_dict(k: np.ndarray) -> dict[str, float]:
    return {
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
    }


def validate_cube_path(cube_path: Path) -> Path:
    cube_path = cube_path.expanduser().resolve()
    if cube_path.is_dir() and (cube_path / "config.json").is_file():
        return cube_path
    if cube_path.is_file() and cube_path.name == "config.json":
        return cube_path
    raise FileNotFoundError(f"Invalid AprilCube cfg path: {cube_path}")


def resolve_common_image_size(calib_by_camera: dict[str, dict[str, Any]]) -> tuple[int, int]:
    image_sizes = {
        camera_name: tuple(int(v) for v in calib["image_size"])
        for camera_name, calib in calib_by_camera.items()
    }
    unique_sizes = set(image_sizes.values())
    if len(unique_sizes) != 1:
        raise ValueError(
            "CV2CameraManager accepts one capture size for this script, "
            f"but active cameras use different YAML image_size values: {image_sizes}"
        )
    return next(iter(unique_sizes))


def create_detector_for_camera(
    cube_path: Path,
    camera_name: str,
    calib_by_camera: dict[str, dict[str, Any]],
    *,
    fast: bool,
) -> Any:
    if camera_name not in calib_by_camera:
        raise KeyError(f"Missing intrinsics for camera '{camera_name}'.")

    calib = calib_by_camera[camera_name]
    intrinsic_cfg = camera_matrix_to_intrinsic_dict(np.asarray(calib["K"], dtype=np.float64))

    return aprilcube.detector(
        cube_path,
        intrinsic_cfg=intrinsic_cfg,
        dist_coeffs=np.zeros(5, dtype=np.float64),
        enable_filter=False,
        fast=fast,
    )


def is_fisheye_calib(calib: dict[str, Any]) -> bool:
    camera_model = str(calib.get("camera_model", "")).lower()
    distortion_model = str(calib.get("distortion_model", "")).lower()
    return camera_model == "fisheye" or distortion_model == "opencv_fisheye"


def undistort_image_points_for_pnp(
    raw_points: np.ndarray,
    calib: dict[str, Any],
    pnp_camera_matrix: np.ndarray,
) -> np.ndarray:
    raw_points = np.asarray(raw_points, dtype=np.float64).reshape(-1, 1, 2)
    camera_matrix = np.asarray(calib["K"], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(calib["dist"], dtype=np.float64).reshape(-1)
    pnp_camera_matrix = np.asarray(pnp_camera_matrix, dtype=np.float64).reshape(3, 3)

    if is_fisheye_calib(calib):
        if dist_coeffs.size != 4:
            raise ValueError(f"OpenCV fisheye calibration expects 4 coeffs, got {dist_coeffs.size}.")
        undistorted = cv2.fisheye.undistortPoints(
            raw_points,
            camera_matrix,
            dist_coeffs.reshape(4, 1),
            R=np.eye(3, dtype=np.float64),
            P=pnp_camera_matrix,
        )
    else:
        undistorted = cv2.undistortPoints(
            raw_points,
            camera_matrix,
            dist_coeffs,
            R=np.eye(3, dtype=np.float64),
            P=pnp_camera_matrix,
        )

    return undistorted.reshape(-1, 2)


def project_object_points_to_raw(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    calib: dict[str, Any],
    pnp_camera_matrix: np.ndarray,
) -> np.ndarray:
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    camera_matrix = np.asarray(calib["K"], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(calib["dist"], dtype=np.float64).reshape(-1)

    if is_fisheye_calib(calib):
        projected, _ = cv2.fisheye.projectPoints(
            object_points,
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            camera_matrix,
            dist_coeffs.reshape(4, 1),
        )
    else:
        projected, _ = cv2.projectPoints(
            object_points,
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            np.asarray(pnp_camera_matrix, dtype=np.float64).reshape(3, 3),
            dist_coeffs,
        )

    return projected.reshape(-1, 2)


def detect_cube_tags_on_raw(estimator: Any, enhanced_gray: np.ndarray) -> list[tuple[int, np.ndarray]]:
    try:
        corners_list, ids, _rejected = estimator.detector.detectMarkers(enhanced_gray)
    except cv2.error:
        corners_list, ids = (), None

    detections: list[tuple[int, np.ndarray]] = []
    seen_ids: set[int] = set()
    if ids is None:
        return detections

    for i in range(len(ids)):
        tag_id = int(ids[i][0])
        if tag_id in estimator.valid_ids and tag_id not in seen_ids:
            corners_2d = corners_list[i].reshape(4, 2)
            if _quad_quality(corners_2d) > 0.15:
                detections.append((tag_id, corners_2d))
                seen_ids.add(tag_id)

    return detections


def result_visible_faces(estimator: Any, detections: list[tuple[int, np.ndarray]]) -> set[str]:
    visible_faces: set[str] = set()
    for tag_id, _corners in detections:
        for face_name, id_set in estimator.face_id_sets.items():
            if tag_id in id_set:
                visible_faces.add(face_name)
    return visible_faces


def face_normals_are_camera_facing(estimator: Any, visible_faces: set[str], rvec: np.ndarray) -> bool:
    rot_mat, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    for face_name in visible_faces:
        for face_def in FACE_DEFS:
            if face_def[0] != face_name:
                continue
            normal_obj = np.zeros(3, dtype=np.float64)
            normal_obj[face_def[1]] = face_def[2]
            normal_cam = rot_mat @ normal_obj
            if normal_cam[2] > 0:
                return False
            break
    return True


def estimate_cube_pose_raw_fisheye(
    estimator: Any,
    detections: list[tuple[int, np.ndarray]],
    calib: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "success": False,
        "rvec": None,
        "tvec": None,
        "T": None,
        "reproj_error": float("inf"),
        "pnp_reproj_error": float("inf"),
        "n_tags": len(detections),
        "n_inliers": 0,
        "detections": detections,
        "tag_ids": [tag_id for tag_id, _ in detections],
        "visible_faces": result_visible_faces(estimator, detections),
        "predicted": False,
    }
    if not detections:
        return result

    object_points = np.vstack([estimator.tag_corner_map[tag_id] for tag_id, _ in detections])
    raw_image_points = np.vstack([corners for _tag_id, corners in detections])
    pnp_camera_matrix = np.asarray(estimator.camera_matrix, dtype=np.float64).reshape(3, 3)
    pnp_image_points = undistort_image_points_for_pnp(raw_image_points, calib, pnp_camera_matrix)

    success, rvec, tvec, pnp_reproj_err, inliers = estimate_pose(
        object_points.astype(np.float64),
        pnp_image_points.astype(np.float64),
        pnp_camera_matrix,
        np.zeros(5, dtype=np.float64),
    )
    if not success or rvec is None or tvec is None:
        return result

    raw_projected = project_object_points_to_raw(
        object_points,
        rvec,
        tvec,
        calib,
        pnp_camera_matrix,
    )
    raw_reproj_err = float(np.mean(np.linalg.norm(raw_image_points - raw_projected, axis=1)))
    if raw_reproj_err > MAX_RAW_REPROJ_ERROR_PX:
        result["pnp_reproj_error"] = pnp_reproj_err
        result["reproj_error"] = raw_reproj_err
        return result

    if not face_normals_are_camera_facing(estimator, result["visible_faces"], rvec):
        result["pnp_reproj_error"] = pnp_reproj_err
        result["reproj_error"] = raw_reproj_err
        return result

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3], _ = cv2.Rodrigues(rvec)
    transform[:3, 3] = tvec.flatten()

    result["success"] = True
    result["rvec"] = rvec
    result["tvec"] = tvec
    result["T"] = transform
    result["reproj_error"] = raw_reproj_err
    result["pnp_reproj_error"] = pnp_reproj_err
    result["n_inliers"] = len(inliers) if inliers is not None else 0
    return result


def make_tag_detection_vis_image(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    enhanced = preprocess_tag_image(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def draw_raw_fisheye_result(
    image: np.ndarray,
    estimator: Any,
    result: dict[str, Any],
    calib: dict[str, Any],
) -> np.ndarray:
    vis = image.copy()

    for _tag_id, corners_2d in result["detections"]:
        pts = corners_2d.astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 255), 4)

    if result["success"]:
        rvec = np.asarray(result["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(result["tvec"], dtype=np.float64).reshape(3, 1)
        pnp_camera_matrix = np.asarray(estimator.camera_matrix, dtype=np.float64).reshape(3, 3)

        axis_len = float(max(estimator.config.box_dims) / 2)
        axes_3d = np.float64([
            [0, 0, 0],
            [axis_len, 0, 0],
            [0, axis_len, 0],
            [0, 0, axis_len],
        ])
        axes_2d = project_object_points_to_raw(axes_3d, rvec, tvec, calib, pnp_camera_matrix)
        axes_2d = axes_2d.astype(int)
        origin = tuple(axes_2d[0])
        cv2.arrowedLine(vis, origin, tuple(axes_2d[1]), (0, 0, 255), 2, tipLength=0.15)
        cv2.arrowedLine(vis, origin, tuple(axes_2d[2]), (0, 255, 0), 2, tipLength=0.15)
        cv2.arrowedLine(vis, origin, tuple(axes_2d[3]), (255, 0, 0), 2, tipLength=0.15)

        box_2d = project_object_points_to_raw(
            estimator.box_corners_3d,
            rvec,
            tvec,
            calib,
            pnp_camera_matrix,
        ).astype(int)
        for i, j in estimator.box_edges:
            cv2.line(vis, tuple(box_2d[i]), tuple(box_2d[j]), (0, 165, 255), 2)

    return vis


def rotation_matrix_to_euler_xyz_deg(rot_mat: np.ndarray) -> np.ndarray:
    r = np.asarray(rot_mat, dtype=np.float64)
    sy = np.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(r[2, 1], r[2, 2])
        y = np.arctan2(-r[2, 0], sy)
        z = np.arctan2(r[1, 0], r[0, 0])
    else:
        x = np.arctan2(-r[1, 2], r[1, 1])
        y = np.arctan2(-r[2, 0], sy)
        z = 0.0
    return np.degrees(np.array([x, y, z], dtype=np.float64))


def result_to_text(camera_name: str, cube_name: str, result: dict[str, Any] | None) -> str:
    prefix = f"[{camera_name}][{cube_name}]"
    if not result:
        return f"{prefix} no result"
    if not result.get("success", False):
        n_tags = int(result.get("n_tags", 0))
        error = result.get("reproj_error", None)
        suffix = "" if error is None or not np.isfinite(error) else f" raw_reproj={float(error):.2f}px"
        return f"{prefix} cube not detected tags={n_tags}{suffix}"

    tvec = np.asarray(result["tvec"], dtype=np.float64).reshape(-1)
    text = f"{prefix} t=({tvec[0]:.1f},{tvec[1]:.1f},{tvec[2]:.1f})"

    if result.get("rvec", None) is not None:
        rot_mat, _ = cv2.Rodrigues(np.asarray(result["rvec"], dtype=np.float64).reshape(3, 1))
        euler = rotation_matrix_to_euler_xyz_deg(rot_mat)
        text += f" rot=({euler[0]:.1f},{euler[1]:.1f},{euler[2]:.1f})"

    text += f" raw_reproj={float(result['reproj_error']):.2f}px"
    text += f" pnp_reproj={float(result['pnp_reproj_error']):.2f}px"
    text += f" tags={int(result.get('n_tags', 0))}"

    faces = result.get("visible_faces", None)
    if faces is not None:
        text += f" faces={sorted(list(faces))}"

    return text


def draw_text_panel(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    y = 28
    for line in lines:
        cv2.putText(
            out,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Detect AprilCube on the raw fisheye image: raw tag detection, "
            "cv2.fisheye.undistortPoints() for PnP, fisheye projection for drawing."
        )
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=",".join(ACTIVE_CAMERA_NAMES),
        help="Comma-separated logical camera names.",
    )
    parser.add_argument(
        "--cube-dirs",
        type=str,
        default=",".join(str(path) for path in CUBE_CFG_DIRS),
        help="Comma-separated AprilCube cfg directories or config.json files.",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Use native AprilCube slow/high-accuracy detector parameters.",
    )
    args = parser.parse_args()

    active_camera_names = [x.strip() for x in args.cameras.split(",") if x.strip()]
    cube_paths = [validate_cube_path(Path(x.strip())) for x in args.cube_dirs.split(",") if x.strip()]
    if not active_camera_names:
        print("[ERROR] No active camera names specified.")
        sys.exit(1)
    if not cube_paths:
        print("[ERROR] No cube cfg paths specified.")
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

    for camera_name in active_camera_names:
        calib = calib_by_camera[camera_name]
        print(
            f"[INFO] [{camera_name}] intrinsics_yaml={calib['path']} "
            f"image_size={calib['image_size']} "
            f"camera_model={calib['camera_model'] or 'unknown'} "
            f"distortion_model={calib['distortion_model'] or 'unknown'} "
            f"raw_fisheye={is_fisheye_calib(calib)}"
        )
    print(
        f"[INFO] capture_size={capture_size} "
        f"detect_img_size={detect_img_size} vis_img_size={vis_img_size}"
    )

    detector_entries_by_camera: dict[str, list[dict[str, Any]]] = {
        name: [] for name in active_camera_names
    }
    for cube_path in cube_paths:
        cube_name = cube_path.name if cube_path.is_dir() else cube_path.parent.name
        for camera_name in active_camera_names:
            estimator = create_detector_for_camera(
                cube_path,
                camera_name,
                calib_by_camera,
                fast=not args.slow,
            )
            detector_entries_by_camera[camera_name].append(
                {
                    "cube_name": cube_name,
                    "estimator": estimator,
                }
            )
            print(f"[INFO] Loaded raw-fisheye AprilCube estimator for {camera_name}: {cube_name}")

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
        print("[INFO] Raw fisheye path: detect raw corners -> undistortPoints -> solvePnP.")
        print("[INFO] This script does not draw predicted-only cube poses.")
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

                calib = calib_by_camera[camera_name]
                detector_entries = detector_entries_by_camera[camera_name]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                enhanced = preprocess_tag_image(gray)
                vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

                fps_text = camera_manager.get_latest_fps(camera_name)
                status_lines = [
                    f"[{camera_name}] raw_fisheye_aprilcube cubes={len(detector_entries)} "
                    f"detect_size={detect_img_size} vis_size={vis_img_size} capture_size={capture_size} "
                    f"fps={fps_text:.1f}" if fps_text is not None else
                    f"[{camera_name}] raw_fisheye_aprilcube cubes={len(detector_entries)} "
                    f"detect_size={detect_img_size} vis_size={vis_img_size} capture_size={capture_size}"
                ]

                for entry in detector_entries:
                    cube_name = entry["cube_name"]
                    estimator = entry["estimator"]
                    detections = detect_cube_tags_on_raw(estimator, enhanced)
                    result = estimate_cube_pose_raw_fisheye(estimator, detections, calib)
                    vis = draw_raw_fisheye_result(vis, estimator, result, calib)

                    line = result_to_text(camera_name, cube_name, result)
                    status_lines.append(line)
                    if frame_idx % PRINT_EVERY_N_FRAMES == 0:
                        print(line)

                status_lines.append("press q or ESC to quit")
                vis = draw_text_panel(vis, status_lines)
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
