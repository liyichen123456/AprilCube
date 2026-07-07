# OpenCV 相机系
# +x: image right
# +y: image down
# +z: camera forward
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pupil_apriltags import Detector

THIS_FILE = Path(__file__).resolve()
THIRDPARTY_DIR = THIS_FILE.parent.parent.parent
APRILCUBE_SRC_DIR = THIRDPARTY_DIR / "aprilcube" / "cubes" / "cube_april_36h11_0_5_1x1x1_10mm"
PROJECT_ROOT = THIRDPARTY_DIR.parent

RECORDER_UTILS_DIR = PROJECT_ROOT / "scripts" / "utils"

if str(RECORDER_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(RECORDER_UTILS_DIR))

import aprilcube  # noqa: E402
from april_tag_detector import TemporalTagPoseEstimator  # noqa: E402
from aprilcube.detect import estimate_pose  # noqa: E402
from aprilcube_runtime import AprilCubeTemporalPoseRuntime  # noqa: E402
from recorder_oak_cam import OAK1WCameraManager, list_oak_devices  # noqa: E402

# ============================================================
# User macros
# ============================================================

PRINT_AVAILABLE_DEVICES = True

# Use the device name you used before.
# If you want l_wrist, add it here and add intrinsics below.
CAMERA_TO_DEVICE: dict[str, str] = {
    "r_wrist": "3.10.4.4.2",
    # "r_wrist": "3.10.4.4.3",
    # "l_wrist": "3.10.4.4.4",
}

ACTIVE_CAMERA_NAMES: list[str] = ["r_wrist"]

# OAK-side output resolution.
# THE_12_MP is about 4056 x 3040.
# ISP_SCALE=(1, 3) gives about 1352 x 1013.
# ISP_SCALE=(1, 4) gives about 1014 x 760.
ISP_SCALE: tuple[int, int] = (1, 3)

# Lower FPS reduces USB bandwidth and camera-board EMI.
FPS = 25

QUEUE_SIZE = 4
QUEUE_BLOCKING = False

ROTATE_180_NAMES: set[str] = set()

# Host-side image size used by AprilCube detector.
# Must match the intrinsics used by the detector.
# DETECT_IMG_SIZE: tuple[int, int] = (4056, 3040)  # width, height
DETECT_IMG_SIZE: tuple[int, int] = (1280, 960)  # width, height

WINDOW_PREFIX = "OAK AprilCube"
ENABLE_VISER = True
VISER_BASE_PORT = 8080

PRINT_EVERY_N_FRAMES = 5
DRAW_TAG_FRAME_2D = True
TAG_AXIS_LENGTH_SCALE = 0.8
UNDISTORT_BEFORE_DETECTION = True

K_ORIGINAL_SIZE: tuple[int, int] = (1280, 960)

# Intrinsics calibrated at K_ORIGINAL_SIZE.
K_BY_CAMERA: dict[str, np.ndarray] = {
    "r_wrist": np.array(
        [
            [734.4013634654287, 0.0, 646.5993494171163],
            [0.0, 736.4711869706985, 474.326864306542],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    "l_wrist": np.array(
        [
            [734.4013634654287, 0.0, 646.5993494171163],
            [0.0, 736.4711869706985, 474.326864306542],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
}

# Optional distortion coefficients.
# For OAK-1-W wide-angle camera, real distortion coefficients are strongly recommended.
# If unknown, zeros are used. Detection may work, but pose accuracy is worse near image edges.
DIST_COEFFS_BY_CAMERA: dict[str, np.ndarray | None] = {
    "r_wrist": np.array(
        [
            -0.2515969903932863,
            0.10485302574350926,
            0.0002484901366271755,
            0.0004693307634594187,
            -0.026532680288426286,
        ],
        dtype=np.float64,
    ),
    "l_wrist": np.array(
        [
            -0.2515969903932863,
            0.10485302574350926,
            0.0002484901366271755,
            0.0004693307634594187,
            -0.026532680288426286,
        ],
        dtype=np.float64,
    ),
}

# Detector settings.
ENABLE_FILTER = True
FAST_DETECTOR = True

# Temporal AprilTag pose estimation from 2D corners.
USE_TEMPORAL_TAG_POSE_ESTIMATOR = True
PUPIL_TO_OBJECT_CORNER_INDEX = [2, 1, 0, 3]
USE_SOLVEPNP_REFINE_LM = True
USE_TEMPORAL_CANDIDATE_SELECTION = True
SOLVEPNP_GENERIC_FLAG = cv2.SOLVEPNP_IPPE
SOLVEPNP_FLAG = cv2.SOLVEPNP_ITERATIVE
TRANSLATION_SCORE_WEIGHT_DEG_PER_MM = 1.0
REJECT_NEGATIVE_CAMERA_Z = True
USE_CLAHE_FOR_TAG_DETECTION = True
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID_SIZE: tuple[int, int] = (8, 8)
TAG_CONSISTENCY_SELECTION_PASSES = 2
TAG_RELATIVE_ROTATION_ERROR_WEIGHT = 1.0
TAG_ADJACENT_NORMAL_ERROR_WEIGHT = 1.5
TAG_MAX_RELATIVE_ROTATION_ERROR_DEG = 75.0
TAG_MAX_ADJACENT_NORMAL_ERROR_DEG = 35.0
CUBE_TEMPORAL_TRANSLATION_GATE_MM = 10.0
CUBE_TEMPORAL_ROTATION_GATE_DEG = 90.0
CUBE_CANDIDATE_TRANSLATION_SCORE_WEIGHT_DEG_PER_MM = 1.0
CUBE_CLUSTER_ROTATION_THRESH_DEG = 35.0
SAVE_PKL_ON_KEY = ord("s")
SNAPSHOT_SAVE_DIR = THIRDPARTY_DIR / "aprilcube" / "logs_002"


# ============================================================
# Utilities
# ============================================================

def scale_intrinsics(
    k: np.ndarray,
    old_size: tuple[int, int],
    new_size: tuple[int, int],
) -> np.ndarray:
    """Scale camera matrix K when resizing images.

    Args:
        k: 3x3 camera matrix for old_size.
        old_size: (width, height) of the image used during calibration.
        new_size: (width, height) of the image used for detection.
    """
    old_w, old_h = old_size
    new_w, new_h = new_size

    sx = new_w / old_w
    sy = new_h / old_h

    k_new = k.astype(np.float64).copy()
    k_new[0, 0] *= sx
    k_new[1, 1] *= sy
    k_new[0, 2] *= sx
    k_new[1, 2] *= sy

    return k_new


def camera_matrix_to_intrinsic_dict(k: np.ndarray) -> dict[str, float]:
    return {
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
    }


def apriltag_family_from_dict_name(dict_name: str) -> str:
    family_map = {
        "apriltag_16h5": "tag16h5",
        "apriltag_25h9": "tag25h9",
        "apriltag_36h10": "tag36h10",
        "apriltag_36h11": "tag36h11",
    }
    if dict_name not in family_map:
        raise ValueError(
            f"Cube config dict '{dict_name}' is not a native AprilTag family supported by pupil_apriltags."
        )
    return family_map[dict_name]


def k_to_camera_params(k: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(k[0, 0]),
        float(k[1, 1]),
        float(k[0, 2]),
        float(k[1, 2]),
    )


def create_detectors(
    cube_path: Path,
    camera_names: list[str],
) -> dict[str, Any]:
    """Create one AprilCube detector per camera."""
    detectors: dict[str, Any] = {}

    for camera_name in camera_names:
        if camera_name not in K_BY_CAMERA:
            raise KeyError(
                f"Missing intrinsics for camera '{camera_name}'. "
                f"Please add K_BY_CAMERA['{camera_name}']."
            )

        k_scaled = scale_intrinsics(
            K_BY_CAMERA[camera_name],
            old_size=K_ORIGINAL_SIZE,
            new_size=DETECT_IMG_SIZE,
        )

        intrinsic_cfg = camera_matrix_to_intrinsic_dict(k_scaled)

        dist_coeffs = DIST_COEFFS_BY_CAMERA.get(camera_name)
        if dist_coeffs is not None:
            dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)

        detector_dist_coeffs = dist_coeffs
        if UNDISTORT_BEFORE_DETECTION:
            detector_dist_coeffs = np.zeros(5, dtype=np.float64)

        detector = aprilcube.detector(
            cube_path,
            intrinsic_cfg=intrinsic_cfg,
            dist_coeffs=detector_dist_coeffs,
            enable_filter=ENABLE_FILTER,
            fast=FAST_DETECTOR,
        )

        detectors[camera_name] = detector

        print(f"[INFO] Created AprilCube detector for {camera_name}")
        print(f"[INFO]   cube_path = {cube_path}")
        print(f"[INFO]   detect_size = {DETECT_IMG_SIZE}")
        print(f"[INFO]   K_scaled =\n{k_scaled}")
        print(f"[INFO]   undistort_before_detection = {UNDISTORT_BEFORE_DETECTION}")
        print(f"[INFO]   input_dist_coeffs = {dist_coeffs}")
        print(f"[INFO]   detector_dist_coeffs = {detector_dist_coeffs}")

    return detectors


def create_pose_estimators(
    detectors: dict[str, Any],
    camera_names: list[str],
) -> dict[str, TemporalTagPoseEstimator]:
    """Create one reusable temporal pose estimator per camera."""
    estimators: dict[str, TemporalTagPoseEstimator] = {}
    for camera_name in camera_names:
        detector = detectors[camera_name]
        tag_size_m = float(detector.config.tag_size_mm) / 1000.0
        estimators[camera_name] = TemporalTagPoseEstimator(
            tag_size_m=tag_size_m,
            pupil_to_object_corner_index=PUPIL_TO_OBJECT_CORNER_INDEX,
            solvepnp_generic_flag=SOLVEPNP_GENERIC_FLAG,
            solvepnp_flag=SOLVEPNP_FLAG,
            use_temporal_candidate_selection=USE_TEMPORAL_TAG_POSE_ESTIMATOR and USE_TEMPORAL_CANDIDATE_SELECTION,
            use_solvepnp_refine_lm=USE_SOLVEPNP_REFINE_LM,
            translation_score_weight_deg_per_mm=TRANSLATION_SCORE_WEIGHT_DEG_PER_MM,
            reject_negative_camera_z=REJECT_NEGATIVE_CAMERA_Z,
        )
        print(f"[INFO] Created TemporalTagPoseEstimator for {camera_name}")
    return estimators


def build_viser_servers(
    detectors: dict[str, Any],
    camera_names: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Start one aprilcube viser server per camera/detector."""
    viser_servers: dict[str, Any] = {}
    gray_image_handles: dict[str, Any] = {}
    record_checkbox_handles: dict[str, Any] = {}

    if not ENABLE_VISER:
        return viser_servers, gray_image_handles, record_checkbox_handles

    for idx, camera_name in enumerate(camera_names):
        detector = detectors[camera_name]
        port = int(VISER_BASE_PORT) + idx
        try:
            detector._viser_show_mesh = False
            detector._viser_show_object_axes = False
            detector._viser_object_frame_axes_length = float(max(detector.config.box_dims)) / 1000.0 * 1.1
            detector._viser_object_frame_axes_radius = 0.001
            detector._viser_object_frame_origin_radius = 0.0
            server = detector.build_viser(port=port)
            viser_servers[camera_name] = server
            gray_folder = server.gui.add_folder("AprilTag Detector Input")
            with gray_folder:
                gray_image_handles[camera_name] = server.gui.add_image(
                    np.zeros((240, 320, 3), dtype=np.uint8),
                    label="Gray / CLAHE",
                    format="jpeg",
                    jpeg_quality=70,
                )
            record_folder = server.gui.add_folder("Recording")
            with record_folder:
                record_checkbox_handles[camera_name] = server.gui.add_checkbox(
                    "Save PKL",
                    initial_value=False,
                )
            print(f"[INFO] AprilCube viser for {camera_name}: http://0.0.0.0:{port}")
        except Exception as exc:
            print(
                f"[WARNING] Failed to start AprilCube viser for {camera_name}: "
                f"{type(exc).__name__}: {exc}"
            )

    return viser_servers, gray_image_handles, record_checkbox_handles


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.uint8)
    if gray.ndim == 2:
        return np.repeat(gray[:, :, None], 3, axis=2)
    return gray


def clone_result_for_pickle(result: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            cloned[key] = np.array(value, copy=True)
        elif isinstance(value, dict):
            sub_dict: dict[str, Any] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, np.ndarray):
                    sub_dict[sub_key] = np.array(sub_value, copy=True)
                elif isinstance(sub_value, dict):
                    sub_sub: dict[str, Any] = {}
                    for k3, v3 in sub_value.items():
                        sub_sub[k3] = np.array(v3, copy=True) if isinstance(v3, np.ndarray) else v3
                    sub_dict[sub_key] = sub_sub
                else:
                    sub_dict[sub_key] = sub_value
            cloned[key] = sub_dict
        elif isinstance(value, list):
            new_list: list[Any] = []
            for item in value:
                if isinstance(item, np.ndarray):
                    new_list.append(np.array(item, copy=True))
                elif isinstance(item, tuple):
                    new_list.append(tuple(np.array(v, copy=True) if isinstance(v, np.ndarray) else v for v in item))
                else:
                    new_list.append(item)
            cloned[key] = new_list
        elif isinstance(value, set):
            cloned[key] = set(value)
        else:
            cloned[key] = value
    return cloned


def save_snapshot_pkl(
    *,
    frame_idx: int,
    cube_path: Path,
    snapshot_by_camera: dict[str, dict[str, Any]],
) -> Path:
    SNAPSHOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    save_path = SNAPSHOT_SAVE_DIR / f"snapshot_{ts}_frame_{frame_idx:06d}.pkl"
    payload = {
        "meta": {
            "source_script": str(THIS_FILE),
            "cube_path": str(cube_path.resolve()),
            "frame_idx": int(frame_idx),
            "detect_img_size": tuple(int(v) for v in DETECT_IMG_SIZE),
            "isp_scale": tuple(int(v) for v in ISP_SCALE),
            "fps": int(FPS),
            "undistort_before_detection": bool(UNDISTORT_BEFORE_DETECTION),
            "use_clahe_for_tag_detection": bool(USE_CLAHE_FOR_TAG_DETECTION),
            "clahe_clip_limit": float(CLAHE_CLIP_LIMIT),
            "clahe_tile_grid_size": tuple(int(v) for v in CLAHE_TILE_GRID_SIZE),
        },
        "cameras": snapshot_by_camera,
    }
    with save_path.open("wb") as f:
        pickle.dump(payload, f)
    return save_path


def set_recording_checkbox_values(
    checkbox_handles: dict[str, Any],
    value: bool,
) -> None:
    for handle in checkbox_handles.values():
        handle.value = bool(value)


def start_recording_state() -> dict[str, Any]:
    return {
        "active": True,
        "start_time_epoch_s": time.time(),
        "frames": [],
    }


def save_recording_pkl(
    *,
    cube_path: Path,
    recording_state: dict[str, Any],
) -> Path:
    SNAPSHOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    start_ts = float(recording_state.get("start_time_epoch_s", time.time()))
    save_name = time.strftime("recording_%Y%m%d_%H%M%S", time.localtime(start_ts))
    save_path = SNAPSHOT_SAVE_DIR / f"{save_name}.pkl"
    payload = {
        "meta": {
            "source_script": str(THIS_FILE),
            "cube_path": str(cube_path.resolve()),
            "detect_img_size": tuple(int(v) for v in DETECT_IMG_SIZE),
            "isp_scale": tuple(int(v) for v in ISP_SCALE),
            "fps": int(FPS),
            "undistort_before_detection": bool(UNDISTORT_BEFORE_DETECTION),
            "use_clahe_for_tag_detection": bool(USE_CLAHE_FOR_TAG_DETECTION),
            "clahe_clip_limit": float(CLAHE_CLIP_LIMIT),
            "clahe_tile_grid_size": tuple(int(v) for v in CLAHE_TILE_GRID_SIZE),
            "recording_start_epoch_s": start_ts,
            "recording_end_epoch_s": time.time(),
            "num_frames": len(recording_state.get("frames", [])),
        },
        "frames": recording_state.get("frames", []),
    }
    with save_path.open("wb") as f:
        pickle.dump(payload, f)
    return save_path


def result_to_text(camera_name: str, result: dict[str, Any] | None) -> str:
    if not result:
        return f"[{camera_name}] no result"

    if not result.get("success", False):
        return f"[{camera_name}] cube not detected"

    tvec = result.get("tvec", None)
    rvec = result.get("rvec", None)
    error = result.get("reproj_error", None)
    visible_faces = result.get("visible_faces", None)
    inward_count = result.get("tag_z_inward_count", None)
    invalid_count = result.get("tag_z_invalid_count", None)

    if tvec is None:
        return f"[{camera_name}] success, but no tvec"

    t = np.asarray(tvec, dtype=np.float64).reshape(-1)

    text = (
        f"[{camera_name}] success "
        f"t_mm=({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f})"
    )

    if rvec is not None:
        r = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        rot_mat, _ = cv2.Rodrigues(r)
        euler_xyz_deg = rotation_matrix_to_euler_xyz_deg(rot_mat)
        text += (
            f" rot_xyz_deg=({euler_xyz_deg[0]:.1f}, "
            f"{euler_xyz_deg[1]:.1f}, {euler_xyz_deg[2]:.1f})"
        )

    if error is not None:
        text += f" reproj={float(error):.2f}px"

    if visible_faces is not None:
        text += f" faces={sorted(list(visible_faces))}"

    if inward_count is not None and invalid_count is not None:
        text += f" tag_z_inward={int(inward_count)} invalid={int(invalid_count)}"

    return text


def tag_edge_length_text(camera_name: str, result: dict[str, Any] | None) -> str:
    if not result:
        return f"[{camera_name}] tag_edge_px: none"

    detections = result.get("detections", [])
    if not detections:
        return f"[{camera_name}] tag_edge_px: none"

    parts: list[str] = []
    for tag_id, corners_2d in detections:
        corners = np.asarray(corners_2d, dtype=np.float64).reshape(4, 2)
        side_lengths = [
            float(np.linalg.norm(corners[(idx + 1) % 4] - corners[idx]))
            for idx in range(4)
        ]
        mean_edge_px = float(np.mean(side_lengths))
        min_edge_px = float(np.min(side_lengths))
        max_edge_px = float(np.max(side_lengths))
        parts.append(
            f"id={int(tag_id)} mean={mean_edge_px:.1f}px min={min_edge_px:.1f}px max={max_edge_px:.1f}px"
        )

    return f"[{camera_name}] tag_edge_px: " + "; ".join(parts)


def is_valid_rotation_matrix(rot: np.ndarray, det_tol: float = 0.2) -> bool:
    """Check whether rot is a valid right-handed rotation matrix."""
    if rot is None:
        return False

    rot = np.asarray(rot, dtype=np.float64)

    if rot.shape != (3, 3):
        return False

    if not np.all(np.isfinite(rot)):
        return False

    det = np.linalg.det(rot)
    if det <= 0.0 or abs(det - 1.0) > det_tol:
        return False

    ortho_err = np.linalg.norm(rot.T @ rot - np.eye(3))
    return ortho_err <= 0.2


def rotation_handedness_text(rot: np.ndarray | None) -> str:
    """Classify a rotation matrix as right-handed, left-handed, or invalid."""
    if rot is None:
        return "missing"

    rot = np.asarray(rot, dtype=np.float64)
    if rot.shape != (3, 3) or not np.all(np.isfinite(rot)):
        return "invalid"

    det = float(np.linalg.det(rot))
    ortho_err = float(np.linalg.norm(rot.T @ rot - np.eye(3)))

    if ortho_err > 0.2:
        return f"invalid(det={det:.4f}, ortho={ortho_err:.4f})"
    if det > 0.0:
        return f"right-handed(det={det:.4f})"
    if det < 0.0:
        return f"left-handed(det={det:.4f})"
    return f"degenerate(det={det:.4f})"


def log_frame_handedness(
    camera_name: str,
    frame_idx: int,
    result: dict[str, Any],
) -> None:
    """Print handedness diagnostics for the cube pose and each detected tag."""
    cube_text = "cube=missing"
    if result.get("success", False) and result.get("rvec", None) is not None:
        cube_rot, _ = cv2.Rodrigues(np.asarray(result["rvec"], dtype=np.float64).reshape(3, 1))
        cube_text = f"cube={rotation_handedness_text(cube_rot)}"

    tag_pose_by_id = result.get("tag_pose_by_id", {})
    if not tag_pose_by_id:
        print(f"[HAND] frame={frame_idx} camera={camera_name} {cube_text} tags=none")
        return

    tag_parts = []
    for tag_id in sorted(tag_pose_by_id):
        pose = tag_pose_by_id[tag_id]
        rot = pose.get("rot_mat", None)
        inward = pose.get("z_inward", None)
        inward_text = ""
        if inward is True:
            inward_text = ",z->in"
        elif inward is False:
            inward_text = ",z->out"
        tag_parts.append(f"id={tag_id}:{rotation_handedness_text(rot)}{inward_text}")

    print(f"[HAND] frame={frame_idx} camera={camera_name} {cube_text} " + " ".join(tag_parts))


def make_handedness_overlay_text(result: dict[str, Any] | None) -> str:
    """Build a short handedness summary for the cv2 overlay."""
    if not result:
        return "handedness: no result"

    tag_pose_by_id = result.get("tag_pose_by_id", {})
    if not tag_pose_by_id:
        return "handedness: no tags"

    parts = []
    for tag_id in sorted(tag_pose_by_id):
        pose = tag_pose_by_id[tag_id]
        rot = pose.get("rot_mat", None)
        handed = rotation_handedness_text(rot)
        if handed.startswith("right-handed"):
            short = "R"
        elif handed.startswith("left-handed"):
            short = "L"
        elif handed.startswith("invalid"):
            short = "I"
        elif handed.startswith("degenerate"):
            short = "D"
        else:
            short = "?"
        parts.append(f"id={tag_id}:{short}")

    return "handedness " + " ".join(parts)


def draw_tag_frame_projection(
    img: np.ndarray,
    pose_R: np.ndarray,
    pose_t: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None,
    axis_length_mm: float,
) -> None:
    """Draw one tag-local coordinate frame projected into the image."""
    pose_R = np.asarray(pose_R, dtype=np.float64)
    pose_t = np.asarray(pose_t, dtype=np.float64).reshape(3, 1)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)

    if not is_valid_rotation_matrix(pose_R):
        return

    obj_pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length_mm, 0.0, 0.0],
            [0.0, axis_length_mm, 0.0],
            [0.0, 0.0, axis_length_mm],
        ],
        dtype=np.float64,
    )

    rvec, _ = cv2.Rodrigues(pose_R)
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float64)

    img_pts, _ = cv2.projectPoints(
        objectPoints=obj_pts,
        rvec=rvec,
        tvec=pose_t,
        cameraMatrix=camera_matrix,
        distCoeffs=np.asarray(dist_coeffs, dtype=np.float64),
    )
    img_pts = np.round(img_pts.reshape(-1, 2)).astype(np.int32)

    origin = tuple(img_pts[0])
    pt_x = tuple(img_pts[1])
    pt_y = tuple(img_pts[2])
    pt_z = tuple(img_pts[3])

    cv2.arrowedLine(img, origin, pt_x, (0, 0, 255), 4, tipLength=0.25)
    cv2.arrowedLine(img, origin, pt_y, (0, 255, 0), 4, tipLength=0.25)
    cv2.arrowedLine(img, origin, pt_z, (255, 0, 0), 4, tipLength=0.25)

    cv2.putText(img, "x", pt_x, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.putText(img, "y", pt_y, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(img, "z", pt_z, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)


def get_tag_object_corners(tag_size_m: float) -> np.ndarray:
    half = float(tag_size_m) / 2.0
    return np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=np.float64,
    )


def reorder_pupil_corners_to_object_order(
    corners_xy: np.ndarray,
    reorder_index: list[int] | tuple[int, int, int, int],
) -> np.ndarray:
    corners = np.asarray(corners_xy, dtype=np.float64).reshape(4, 2)
    return corners[np.asarray(reorder_index, dtype=np.int64)]


def tag_pose_from_native_detection(
    tag: Any,
) -> tuple[np.ndarray, np.ndarray, float | None] | None:
    """Convert pupil_apriltags pose output to the script's mm-based convention."""
    pose_R = getattr(tag, "pose_R", None)
    pose_t = getattr(tag, "pose_t", None)
    if pose_R is None or pose_t is None:
        return None

    pose_R = np.asarray(pose_R, dtype=np.float64).reshape(3, 3)
    if not is_valid_rotation_matrix(pose_R):
        return None

    pose_t_mm = np.asarray(pose_t, dtype=np.float64).reshape(3, 1) * 1000.0
    pose_err = getattr(tag, "pose_err", None)
    reproj_error = float(pose_err) if pose_err is not None else None
    return pose_R, pose_t_mm, reproj_error


def estimate_tag_pose_candidates_solvepnp_ippe(
    detector: Any,
    tag: Any,
) -> list[dict[str, Any]]:
    obj_pts = get_tag_object_corners(float(detector.config.tag_size_mm) / 1000.0)
    img_pts = reorder_pupil_corners_to_object_order(
        np.asarray(tag.corners, dtype=np.float64).reshape(4, 2),
        reorder_index=PUPIL_TO_OBJECT_CORNER_INDEX,
    )
    k = np.asarray(detector.camera_matrix, dtype=np.float64)
    if detector.dist_coeffs is None:
        dist = np.zeros((5, 1), dtype=np.float64)
    else:
        dist = np.asarray(detector.dist_coeffs, dtype=np.float64).reshape(-1, 1)

    candidates: list[dict[str, Any]] = []
    try:
        retval, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
            objectPoints=obj_pts,
            imagePoints=img_pts,
            cameraMatrix=k,
            distCoeffs=dist,
            flags=SOLVEPNP_GENERIC_FLAG,
        )
    except cv2.error:
        retval, rvecs, tvecs, reproj_errs = 0, [], [], None

    if int(retval) > 0:
        if reproj_errs is None:
            reproj_err_list = [None] * len(rvecs)
        else:
            reproj_err_list = np.asarray(reproj_errs, dtype=np.float64).reshape(-1).tolist()

        for cand_idx, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            pose_R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
            pose_t_m = np.asarray(tvec, dtype=np.float64).reshape(3)
            if not is_valid_rotation_matrix(pose_R):
                continue
            if not np.all(np.isfinite(pose_t_m)):
                continue
            if REJECT_NEGATIVE_CAMERA_Z and float(pose_t_m[2]) <= 0.0:
                continue

            reproj, _ = cv2.projectPoints(
                objectPoints=obj_pts,
                rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                tvec=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
                cameraMatrix=k,
                distCoeffs=dist,
            )
            reproj = reproj.reshape(-1, 2)
            reproj_error_px = float(np.mean(np.linalg.norm(reproj - img_pts, axis=1)))
            opencv_reproj = (
                float(reproj_err_list[cand_idx])
                if cand_idx < len(reproj_err_list) and reproj_err_list[cand_idx] is not None
                else None
            )
            candidates.append(
                {
                    "candidate_index": int(cand_idx),
                    "pose_R": pose_R,
                    "pose_t": pose_t_m,
                    "rvec": np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                    "tvec": np.asarray(tvec, dtype=np.float64).reshape(3, 1),
                    "reproj_error_px": reproj_error_px,
                    "opencv_reproj_error_px": opencv_reproj,
                }
            )

    if candidates:
        return candidates

    success, rvec, tvec = cv2.solvePnP(
        objectPoints=obj_pts,
        imagePoints=img_pts,
        cameraMatrix=k,
        distCoeffs=dist,
        flags=SOLVEPNP_FLAG,
    )
    if not success:
        return []

    pose_R, _ = cv2.Rodrigues(rvec)
    pose_t_m = np.asarray(tvec, dtype=np.float64).reshape(3)
    if not is_valid_rotation_matrix(pose_R):
        return []
    if REJECT_NEGATIVE_CAMERA_Z and float(pose_t_m[2]) <= 0.0:
        return []
    reproj, _ = cv2.projectPoints(obj_pts, rvec, tvec, k, dist)
    reproj = reproj.reshape(-1, 2)
    reproj_error_px = float(np.mean(np.linalg.norm(reproj - img_pts, axis=1)))
    return [
        {
            "candidate_index": 0,
            "pose_R": pose_R,
            "pose_t": pose_t_m,
            "rvec": np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            "tvec": np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            "reproj_error_px": reproj_error_px,
            "opencv_reproj_error_px": None,
            "fallback_iterative": True,
        }
    ]


def convert_selected_candidate_to_mm(
    selected: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float | None]:
    pose_R = np.asarray(selected["pose_R"], dtype=np.float64).reshape(3, 3)
    pose_t_mm = np.asarray(selected["pose_t"], dtype=np.float64).reshape(3, 1) * 1000.0
    reproj_error = selected.get("reproj_error_px", None)
    return pose_R, pose_t_mm, (float(reproj_error) if reproj_error is not None else None)


def build_tag_to_cube_transform(
    detector: Any,
    tag_id: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the rigid transform X_cube = R * X_tag + t for one tag."""
    cache = getattr(detector, "_tag_to_cube_transform_cache", None)
    if cache is None:
        cache = {}
        detector._tag_to_cube_transform_cache = cache
    if int(tag_id) in cache:
        return cache[int(tag_id)]

    corners_3d = detector.tag_corner_map.get(int(tag_id))
    if corners_3d is None:
        return None

    tl, tr, _br, bl = np.asarray(corners_3d, dtype=np.float64).reshape(4, 3)
    x_axis = tr - tl
    y_axis = bl - tl

    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    rot_cube_tag = np.column_stack((x_axis, y_axis, z_axis))
    if not is_valid_rotation_matrix(rot_cube_tag):
        return None

    center_cube = np.mean(np.asarray(corners_3d, dtype=np.float64), axis=0).reshape(3, 1)
    cache[int(tag_id)] = (rot_cube_tag, center_cube)
    return cache[int(tag_id)]


def rotation_angle_deg_between(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rot_delta = np.asarray(rot_a, dtype=np.float64).reshape(3, 3).T @ np.asarray(
        rot_b, dtype=np.float64
    ).reshape(3, 3)
    cos_angle = np.clip((np.trace(rot_delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def vector_angle_deg(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = np.asarray(vec_a, dtype=np.float64).reshape(3)
    b = np.asarray(vec_b, dtype=np.float64).reshape(3)
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm <= 1e-12 or b_norm <= 1e-12:
        return 0.0
    cos_angle = np.clip(float(np.dot(a, b) / (a_norm * b_norm)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def get_expected_tag_pair_geometry(
    detector: Any,
    tag_id_a: int,
    tag_id_b: int,
) -> dict[str, Any] | None:
    cache = getattr(detector, "_tag_pair_geometry_cache", None)
    if cache is None:
        cache = {}
        detector._tag_pair_geometry_cache = cache

    key = (int(tag_id_a), int(tag_id_b))
    if key in cache:
        return cache[key]

    transform_a = build_tag_to_cube_transform(detector, int(tag_id_a))
    transform_b = build_tag_to_cube_transform(detector, int(tag_id_b))
    if transform_a is None or transform_b is None:
        return None

    rot_cube_tag_a, _center_a = transform_a
    rot_cube_tag_b, _center_b = transform_b
    expected_rel_rot = rot_cube_tag_a.T @ rot_cube_tag_b
    expected_z_angle_deg = vector_angle_deg(rot_cube_tag_a[:, 2], rot_cube_tag_b[:, 2])
    geom = {
        "expected_rel_rot": expected_rel_rot,
        "expected_z_angle_deg": expected_z_angle_deg,
        "is_adjacent_face": abs(expected_z_angle_deg - 90.0) <= 1e-3,
    }
    cache[key] = geom
    return geom


def score_tag_candidate_against_visible_tags(
    detector: Any,
    tag_id: int,
    candidate_pose_R: np.ndarray,
    other_pose_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    rel_rot_errors_deg: list[float] = []
    adjacent_normal_errors_deg: list[float] = []
    pair_debug: list[dict[str, Any]] = []

    candidate_pose_R = np.asarray(candidate_pose_R, dtype=np.float64).reshape(3, 3)
    candidate_z = candidate_pose_R[:, 2]

    for other_tag_id, other_pose in sorted(other_pose_by_id.items()):
        if int(other_tag_id) == int(tag_id):
            continue

        geom = get_expected_tag_pair_geometry(detector, int(tag_id), int(other_tag_id))
        if geom is None:
            continue

        other_rot = np.asarray(other_pose["rot_mat"], dtype=np.float64).reshape(3, 3)
        observed_rel_rot = candidate_pose_R.T @ other_rot
        rel_rot_err_deg = rotation_angle_deg_between(geom["expected_rel_rot"], observed_rel_rot)
        rel_rot_errors_deg.append(rel_rot_err_deg)

        other_z = other_rot[:, 2]
        observed_z_angle_deg = vector_angle_deg(candidate_z, other_z)
        z_angle_err_deg = abs(observed_z_angle_deg - float(geom["expected_z_angle_deg"]))
        if bool(geom["is_adjacent_face"]):
            adjacent_normal_errors_deg.append(z_angle_err_deg)

        pair_debug.append(
            {
                "other_tag_id": int(other_tag_id),
                "expected_z_angle_deg": float(geom["expected_z_angle_deg"]),
                "observed_z_angle_deg": observed_z_angle_deg,
                "z_angle_err_deg": z_angle_err_deg,
                "rel_rot_err_deg": rel_rot_err_deg,
                "is_adjacent_face": bool(geom["is_adjacent_face"]),
            }
        )

    mean_rel_rot_err_deg = (
        float(np.mean(rel_rot_errors_deg)) if rel_rot_errors_deg else 0.0
    )
    mean_adjacent_normal_err_deg = (
        float(np.mean(adjacent_normal_errors_deg)) if adjacent_normal_errors_deg else 0.0
    )
    hard_reject = (
        (bool(rel_rot_errors_deg) and mean_rel_rot_err_deg > float(TAG_MAX_RELATIVE_ROTATION_ERROR_DEG))
        or (
            bool(adjacent_normal_errors_deg)
            and mean_adjacent_normal_err_deg > float(TAG_MAX_ADJACENT_NORMAL_ERROR_DEG)
        )
    )

    return {
        "mean_rel_rot_err_deg": mean_rel_rot_err_deg,
        "mean_adjacent_normal_err_deg": mean_adjacent_normal_err_deg,
        "num_rel_pairs": int(len(rel_rot_errors_deg)),
        "num_adjacent_pairs": int(len(adjacent_normal_errors_deg)),
        "hard_reject": bool(hard_reject),
        "pair_debug": pair_debug,
    }


def cube_pose_from_tag_pose(
    detector: Any,
    tag_id: int,
    tag_rot_mat: np.ndarray,
    tag_tvec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Convert one independently estimated native AprilTag pose into a cube pose."""
    tag_to_cube = build_tag_to_cube_transform(detector, tag_id)
    if tag_to_cube is None:
        return None

    rot_cube_tag, center_cube = tag_to_cube
    rot_tag_cube = rot_cube_tag.T
    center_tag = -rot_tag_cube @ center_cube

    rot_cam_tag = np.asarray(tag_rot_mat, dtype=np.float64).reshape(3, 3)
    tag_tvec = np.asarray(tag_tvec, dtype=np.float64).reshape(3, 1)

    rot_cam_cube = rot_cam_tag @ rot_tag_cube
    center_cam = rot_cam_tag @ center_tag + tag_tvec

    if not is_valid_rotation_matrix(rot_cam_cube):
        return None

    return rot_cam_cube, center_cam


def tag_pose_from_cube_pose(
    detector: Any,
    tag_id: int,
    cube_rot_mat: np.ndarray,
    cube_tvec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    tag_to_cube = build_tag_to_cube_transform(detector, tag_id)
    if tag_to_cube is None:
        return None

    rot_cube_tag, center_cube = tag_to_cube
    cube_rot_mat = np.asarray(cube_rot_mat, dtype=np.float64).reshape(3, 3)
    cube_tvec = np.asarray(cube_tvec, dtype=np.float64).reshape(3, 1)

    rot_cam_tag = cube_rot_mat @ rot_cube_tag
    tag_tvec = cube_rot_mat @ center_cube + cube_tvec
    if not is_valid_rotation_matrix(rot_cam_tag):
        return None
    return rot_cam_tag, tag_tvec


def tag_z_points_to_cube_interior(
    tag_rot_mat: np.ndarray,
    tag_tvec: np.ndarray,
    cube_center_cam: np.ndarray,
) -> bool:
    """Check whether +z of the independently estimated tag pose points inward."""
    tag_rot_mat = np.asarray(tag_rot_mat, dtype=np.float64).reshape(3, 3)
    tag_tvec = np.asarray(tag_tvec, dtype=np.float64).reshape(3)
    cube_center_cam = np.asarray(cube_center_cam, dtype=np.float64).reshape(3)

    z_axis_cam = tag_rot_mat[:, 2]
    to_cube_center = cube_center_cam - tag_tvec
    return float(np.dot(z_axis_cam, to_cube_center)) > 0.0


def get_previous_cube_pose(detector: Any) -> dict[str, np.ndarray] | None:
    prev_rvec = getattr(detector, "prev_rvec", None)
    prev_tvec = getattr(detector, "prev_tvec", None)
    if prev_rvec is None or prev_tvec is None:
        return None
    prev_rot_mat, _ = cv2.Rodrigues(np.asarray(prev_rvec, dtype=np.float64).reshape(3, 1))
    prev_tvec = np.asarray(prev_tvec, dtype=np.float64).reshape(3, 1)
    if not is_valid_rotation_matrix(prev_rot_mat):
        return None
    return {
        "rot_mat": prev_rot_mat,
        "tvec": prev_tvec,
    }


def relative_cube_pose_delta(
    prev_rot_mat: np.ndarray,
    prev_tvec: np.ndarray,
    curr_rot_mat: np.ndarray,
    curr_tvec: np.ndarray,
) -> tuple[float, float]:
    rot_delta_deg = rotation_angle_deg_between(prev_rot_mat, curr_rot_mat)
    trans_delta_mm = float(
        np.linalg.norm(
            np.asarray(curr_tvec, dtype=np.float64).reshape(3)
            - np.asarray(prev_tvec, dtype=np.float64).reshape(3)
        )
    )
    return rot_delta_deg, trans_delta_mm


def score_cube_candidate_against_tag_measurements(
    detector: Any,
    cube_rot_mat: np.ndarray,
    cube_tvec: np.ndarray,
    tag_meas: list[dict[str, Any]],
) -> dict[str, Any]:
    per_tag_matches: dict[int, dict[str, Any]] = {}
    per_tag_match_scores: list[float] = []

    cube_rot_mat = np.asarray(cube_rot_mat, dtype=np.float64).reshape(3, 3)
    cube_tvec = np.asarray(cube_tvec, dtype=np.float64).reshape(3, 1)

    for meas in tag_meas:
        tag_id = int(meas["tag_id"])
        expected_tag_pose = tag_pose_from_cube_pose(detector, tag_id, cube_rot_mat, cube_tvec)
        if expected_tag_pose is None:
            continue
        expected_rot_mat, expected_tvec = expected_tag_pose

        best_match = None
        best_match_score = float("inf")
        for cand in meas.get("candidate_mm_list", []):
            cand_rot_mat = np.asarray(cand["rot_mat"], dtype=np.float64).reshape(3, 3)
            cand_tvec = np.asarray(cand["tvec"], dtype=np.float64).reshape(3, 1)
            rot_err_deg = rotation_angle_deg_between(expected_rot_mat, cand_rot_mat)
            trans_err_mm = float(np.linalg.norm(cand_tvec.reshape(3) - expected_tvec.reshape(3)))
            reproj_error = cand.get("reproj_error", None)
            reproj_term = 0.0 if reproj_error is None else float(reproj_error)
            match_score = (
                rot_err_deg
                + float(CUBE_CANDIDATE_TRANSLATION_SCORE_WEIGHT_DEG_PER_MM) * trans_err_mm
                + reproj_term
            )
            if match_score < best_match_score:
                best_match_score = match_score
                best_match = {
                    "candidate_index": int(cand.get("candidate_index", -1)),
                    "rot_mat": cand_rot_mat,
                    "tvec": cand_tvec,
                    "reproj_error": reproj_error,
                    "rot_err_deg": rot_err_deg,
                    "trans_err_mm": trans_err_mm,
                    "match_score": match_score,
                }

        if best_match is not None:
            per_tag_matches[tag_id] = best_match
            per_tag_match_scores.append(float(best_match["match_score"]))

    adjacent_normal_errors_deg: list[float] = []
    match_tag_ids = sorted(per_tag_matches)
    for idx_a, tag_id_a in enumerate(match_tag_ids):
        for tag_id_b in match_tag_ids[idx_a + 1:]:
            geom = get_expected_tag_pair_geometry(detector, tag_id_a, tag_id_b)
            if geom is None or not bool(geom["is_adjacent_face"]):
                continue
            rot_a = np.asarray(per_tag_matches[tag_id_a]["rot_mat"], dtype=np.float64).reshape(3, 3)
            rot_b = np.asarray(per_tag_matches[tag_id_b]["rot_mat"], dtype=np.float64).reshape(3, 3)
            observed_z_angle_deg = vector_angle_deg(rot_a[:, 2], rot_b[:, 2])
            adjacent_normal_errors_deg.append(
                abs(observed_z_angle_deg - float(geom["expected_z_angle_deg"]))
            )

    mean_match_score = float(np.mean(per_tag_match_scores)) if per_tag_match_scores else float("inf")
    mean_adjacent_normal_err_deg = (
        float(np.mean(adjacent_normal_errors_deg)) if adjacent_normal_errors_deg else 0.0
    )
    hard_reject = (
        not per_tag_matches
        or (
            bool(adjacent_normal_errors_deg)
            and mean_adjacent_normal_err_deg > float(TAG_MAX_ADJACENT_NORMAL_ERROR_DEG)
        )
    )
    total_score = (
        mean_match_score
        + float(TAG_ADJACENT_NORMAL_ERROR_WEIGHT) * mean_adjacent_normal_err_deg
    )

    return {
        "total_score": total_score,
        "mean_match_score": mean_match_score,
        "mean_adjacent_normal_err_deg": mean_adjacent_normal_err_deg,
        "num_matched_tags": int(len(per_tag_matches)),
        "hard_reject": bool(hard_reject),
        "per_tag_matches": per_tag_matches,
    }


def filter_cube_candidates_with_previous_pose(
    cube_candidates: list[dict[str, Any]],
    previous_pose: dict[str, np.ndarray] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not cube_candidates:
        return [], {
            "had_previous_pose": previous_pose is not None,
            "raw_count": 0,
            "passed_count": 0,
            "fallback_used": False,
            "fallback_reason": "no raw candidates",
        }

    if previous_pose is None:
        return cube_candidates, {
            "had_previous_pose": False,
            "raw_count": len(cube_candidates),
            "passed_count": len(cube_candidates),
            "fallback_used": False,
            "fallback_reason": "",
        }

    prev_rot_mat = np.asarray(previous_pose["rot_mat"], dtype=np.float64).reshape(3, 3)
    prev_tvec = np.asarray(previous_pose["tvec"], dtype=np.float64).reshape(3, 1)
    passed: list[dict[str, Any]] = []

    for cand in cube_candidates:
        rot_delta_deg, trans_delta_mm = relative_cube_pose_delta(
            prev_rot_mat=prev_rot_mat,
            prev_tvec=prev_tvec,
            curr_rot_mat=np.asarray(cand["rot_mat"], dtype=np.float64).reshape(3, 3),
            curr_tvec=np.asarray(cand["tvec"], dtype=np.float64).reshape(3, 1),
        )
        cand["temporal_rotation_delta_deg"] = rot_delta_deg
        cand["temporal_translation_delta_mm"] = trans_delta_mm
        if (
            rot_delta_deg <= float(CUBE_TEMPORAL_ROTATION_GATE_DEG)
            and trans_delta_mm <= float(CUBE_TEMPORAL_TRANSLATION_GATE_MM)
        ):
            passed.append(cand)

    if passed:
        return passed, {
            "had_previous_pose": True,
            "raw_count": len(cube_candidates),
            "passed_count": len(passed),
            "fallback_used": False,
            "fallback_reason": "",
        }

    fallback = sorted(
        cube_candidates,
        key=lambda cand: (
            float(cand.get("cube_score", float("inf"))),
            -int(cand.get("num_support_tags", 0)),
            float(cand.get("reproj_error", float("inf")) if cand.get("reproj_error", None) is not None else float("inf")),
        ),
    )[:1]
    return fallback, {
        "had_previous_pose": True,
        "raw_count": len(cube_candidates),
        "passed_count": 0,
        "fallback_used": True,
        "fallback_reason": "all cube candidates rejected by temporal gate; fallback to best current candidate",
    }


def cluster_cube_candidates(
    detector: Any,
    cube_candidates: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    if not cube_candidates:
        return []

    translation_thresh_mm = float(max(detector.config.box_dims))
    rotation_thresh_deg = float(CUBE_CLUSTER_ROTATION_THRESH_DEG)
    sorted_candidates = sorted(
        cube_candidates,
        key=lambda cand: (
            float(cand.get("cube_score", float("inf"))),
            -int(cand.get("num_support_tags", 0)),
            float(cand.get("reproj_error", float("inf")) if cand.get("reproj_error", None) is not None else float("inf")),
        ),
    )
    clusters: list[list[dict[str, Any]]] = []

    for cand in sorted_candidates:
        assigned = False
        cand_rot_mat = np.asarray(cand["rot_mat"], dtype=np.float64).reshape(3, 3)
        cand_tvec = np.asarray(cand["tvec"], dtype=np.float64).reshape(3, 1)
        for cluster in clusters:
            anchor = cluster[0]
            anchor_rot_mat = np.asarray(anchor["rot_mat"], dtype=np.float64).reshape(3, 3)
            anchor_tvec = np.asarray(anchor["tvec"], dtype=np.float64).reshape(3, 1)
            rot_delta_deg, trans_delta_mm = relative_cube_pose_delta(
                prev_rot_mat=anchor_rot_mat,
                prev_tvec=anchor_tvec,
                curr_rot_mat=cand_rot_mat,
                curr_tvec=cand_tvec,
            )
            if rot_delta_deg <= rotation_thresh_deg and trans_delta_mm <= translation_thresh_mm:
                cluster.append(cand)
                assigned = True
                break
        if not assigned:
            clusters.append([cand])

    return clusters


def select_best_cube_cluster(
    cube_clusters: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if not cube_clusters:
        return []

    def cluster_key(cluster: list[dict[str, Any]]) -> tuple[float, float, float]:
        cluster_scores = [float(cand.get("cube_score", float("inf"))) for cand in cluster]
        support_counts = [int(cand.get("num_support_tags", 0)) for cand in cluster]
        return (
            -float(len(cluster)),
            float(np.mean(cluster_scores)) if cluster_scores else float("inf"),
            -float(np.mean(support_counts)) if support_counts else 0.0,
        )

    return min(cube_clusters, key=cluster_key)

def average_rotations(rot_mats: list[np.ndarray], weights: np.ndarray) -> np.ndarray | None:
    if not rot_mats:
        return None

    accum = np.zeros((3, 3), dtype=np.float64)
    for rot, weight in zip(rot_mats, weights):
        accum += float(weight) * np.asarray(rot, dtype=np.float64)

    u, _s, vt = np.linalg.svd(accum)
    rot_avg = u @ vt
    if np.linalg.det(rot_avg) < 0.0:
        u[:, -1] *= -1.0
        rot_avg = u @ vt

    if not is_valid_rotation_matrix(rot_avg):
        return None
    return rot_avg


def fuse_cube_pose_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray] | None:
    if not candidates:
        return None

    weights = []
    rot_mats = []
    tvecs = []
    for cand in candidates:
        err = cand.get("reproj_error", None)
        weight = 1.0 if err is None else 1.0 / max(float(err), 1e-3)
        weights.append(weight)
        rot_mats.append(np.asarray(cand["rot_mat"], dtype=np.float64).reshape(3, 3))
        tvecs.append(np.asarray(cand["tvec"], dtype=np.float64).reshape(3, 1))

    weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr /= np.sum(weights_arr)

    rot_avg = average_rotations(rot_mats, weights_arr)
    if rot_avg is None:
        return None

    t_avg = np.zeros((3, 1), dtype=np.float64)
    for weight, tvec in zip(weights_arr, tvecs):
        t_avg += float(weight) * tvec

    return rot_avg, t_avg


def detect_native_apriltags(
    detector: Any,
    native_detector: Detector,
    image: np.ndarray,
) -> list[Any]:
    """Detect AprilTags with the native detector and request native pose output."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if USE_CLAHE_FOR_TAG_DETECTION:
        clahe = cv2.createCLAHE(
            clipLimit=float(CLAHE_CLIP_LIMIT),
            tileGridSize=tuple(int(v) for v in CLAHE_TILE_GRID_SIZE),
        )
        gray = clahe.apply(np.asarray(gray, dtype=np.uint8))
    camera_params = k_to_camera_params(np.asarray(detector.camera_matrix, dtype=np.float64))
    tag_size_m = float(detector.config.tag_size_mm) / 1000.0
    tags = native_detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=tag_size_m,
    )
    return [tag for tag in tags if int(tag.tag_id) in detector.valid_ids]


def tag_pose_from_temporal_estimator(
    detector: Any,
    pose_estimator: TemporalTagPoseEstimator,
    camera_name: str,
    tag: Any,
) -> tuple[np.ndarray, np.ndarray, float | None] | None:
    """Estimate a tag pose from cached corners using the shared temporal estimator."""
    solved = pose_estimator.estimate_pose(
        camera_name=camera_name,
        tag_id=int(tag.tag_id),
        corners_xy=np.asarray(tag.corners, dtype=np.float64).reshape(4, 2),
        k=np.asarray(detector.camera_matrix, dtype=np.float64),
        dist_coeffs=np.asarray(detector.dist_coeffs, dtype=np.float64) if detector.dist_coeffs is not None else None,
    )
    if solved is None:
        return None
    pose_R, pose_t, reproj_error_px, _debug_info = solved
    pose_t_mm = np.asarray(pose_t, dtype=np.float64).reshape(3, 1) * 1000.0
    return pose_R, pose_t_mm, float(reproj_error_px)


def compute_reprojection_error(
    detector: Any,
    detections: list[tuple[int, np.ndarray]],
    cube_rvec: np.ndarray,
    cube_tvec: np.ndarray,
) -> float:
    if not detections:
        return float("inf")

    object_points = np.vstack([
        np.asarray(detector.tag_corner_map[int(tag_id)], dtype=np.float64)
        for tag_id, _corners in detections
    ])
    image_points = np.vstack([
        np.asarray(corners, dtype=np.float64)
        for _tag_id, corners in detections
    ])

    projected, _ = cv2.projectPoints(
        object_points,
        cube_rvec,
        cube_tvec,
        detector.camera_matrix,
        detector.dist_coeffs,
    )
    projected = projected.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(image_points - projected, axis=1)))


def process_frame_from_tag_poses(
    camera_name: str,
    detector: Any,
    native_detector: Detector,
    pose_estimator: TemporalTagPoseEstimator,
    image: np.ndarray,
) -> dict[str, Any]:
    """Estimate per-tag poses from corners, then infer cube pose from them."""
    result = {
        "success": False,
        "rvec": None,
        "tvec": None,
        "T": None,
        "reproj_error": float("inf"),
        "n_tags": 0,
        "n_inliers": 0,
        "detections": [],
        "tag_ids": [],
        "visible_faces": set(),
        "predicted": False,
        "tag_z_inward_count": 0,
        "tag_z_invalid_count": 0,
        "tag_pose_by_id": {},
    }

    native_tags = detect_native_apriltags(detector, native_detector, image)
    detections = [
        (int(tag.tag_id), np.asarray(tag.corners, dtype=np.float64).reshape(4, 2))
        for tag in native_tags
    ]
    result["detections"] = detections
    result["n_tags"] = len(detections)
    result["tag_ids"] = [tag_id for tag_id, _ in detections]

    for tag_id, _corners in detections:
        for face_name, id_set in detector.face_id_sets.items():
            if tag_id in id_set:
                result["visible_faces"].add(face_name)

    if not detections:
        return detector._store_latest(result, image)

    object_points = np.vstack(
        [
            np.asarray(detector.tag_corner_map[int(tag_id)], dtype=np.float64).reshape(4, 3)
            for tag_id, _corners in detections
        ]
    ).astype(np.float64)
    image_points = np.vstack(
        [
            np.asarray(corners_2d, dtype=np.float64).reshape(4, 2)
            for _tag_id, corners_2d in detections
        ]
    ).astype(np.float64)

    prev_rvec = getattr(detector, "prev_rvec", None)
    prev_tvec = getattr(detector, "prev_tvec", None)
    success, cube_rvec, cube_tvec, reproj_err, inliers = estimate_pose(
        object_points=object_points,
        image_points=image_points,
        camera_matrix=np.asarray(detector.camera_matrix, dtype=np.float64),
        dist_coeffs=np.asarray(detector.dist_coeffs, dtype=np.float64),
        prev_rvec=prev_rvec,
        prev_tvec=prev_tvec,
    )
    if not success or cube_rvec is None or cube_tvec is None:
        return detector._store_latest(result, image)

    cube_rot_mat, _ = cv2.Rodrigues(np.asarray(cube_rvec, dtype=np.float64).reshape(3, 1))
    cube_tvec = np.asarray(cube_tvec, dtype=np.float64).reshape(3, 1)
    cube_center = cube_tvec.reshape(3)

    inward_count = 0
    invalid_count = 0
    for tag_id, _corners in detections:
        tag_pose = tag_pose_from_cube_pose(detector, int(tag_id), cube_rot_mat, cube_tvec)
        if tag_pose is None:
            continue
        selected_rot_mat, selected_tvec = tag_pose
        inward_ok = tag_z_points_to_cube_interior(
            selected_rot_mat,
            selected_tvec,
            cube_center,
        )
        if inward_ok:
            inward_count += 1
        else:
            invalid_count += 1

        result["tag_pose_by_id"][int(tag_id)] = {
            "rot_mat": selected_rot_mat,
            "tvec": selected_tvec,
            "reproj_error": float(reproj_err),
            "z_inward": inward_ok,
            "consistency_score": None,
            "consistency_debug": None,
        }

        pose_estimator.update_previous_pose(
            camera_name=camera_name,
            tag_id=int(tag_id),
            pose_R=selected_rot_mat,
            pose_t=np.asarray(selected_tvec, dtype=np.float64).reshape(3) / 1000.0,
        )

    result["tag_z_inward_count"] = inward_count
    result["tag_z_invalid_count"] = invalid_count
    single_face_visible = len(result["visible_faces"]) == 1
    if single_face_visible and invalid_count > 0:
        result["cube_candidate_debug"] = {
            "mode": "direct_multitag_pnp",
            "num_tags": int(len(detections)),
            "num_points": int(object_points.shape[0]),
            "used_prev_guess": bool(prev_rvec is not None and prev_tvec is not None),
            "num_inliers": int(len(inliers)) if inliers is not None else None,
            "single_face_inward_constraint_failed": True,
            "visible_faces": sorted(list(result["visible_faces"])),
            "tag_z_inward_count": int(inward_count),
            "tag_z_invalid_count": int(invalid_count),
        }
        return detector._store_latest(result, image)

    result["success"] = True
    result["rvec"] = cube_rvec
    result["tvec"] = cube_tvec
    result["n_inliers"] = int(len(inliers)) if inliers is not None else int(len(detections) * 4)
    result["cube_candidate_debug"] = {
        "mode": "direct_multitag_pnp",
        "num_tags": int(len(detections)),
        "num_points": int(object_points.shape[0]),
        "used_prev_guess": bool(prev_rvec is not None and prev_tvec is not None),
        "num_inliers": int(len(inliers)) if inliers is not None else None,
        "single_face_inward_constraint_failed": False,
    }
    result["reproj_error"] = float(reproj_err)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = cube_rot_mat
    T[:3, 3] = cube_tvec.reshape(3)
    result["T"] = T

    detector.prev_rvec = cube_rvec.copy()
    detector.prev_tvec = cube_tvec.copy()

    return detector._store_latest(result, image)


def draw_detected_tag_visuals(
    img: np.ndarray,
    detector: Any,
    result: dict[str, Any] | None,
) -> np.ndarray:
    """Overlay per-tag AprilTag-style visualization on top of cube results."""
    if not result:
        return img

    detections = result.get("detections", [])
    if not detections:
        return img

    out = img.copy()
    tag_pose_by_id = result.get("tag_pose_by_id", {})

    tag_axis_length_mm = float(detector.config.tag_size_mm) * TAG_AXIS_LENGTH_SCALE

    for tag_id, corners_2d in detections:
        corners = np.round(np.asarray(corners_2d, dtype=np.float64)).astype(np.int32)
        if corners.shape != (4, 2):
            continue

        center_xy = np.round(np.mean(corners, axis=0)).astype(np.int32)
        c_x, c_y = int(center_xy[0]), int(center_xy[1])

        tag_is_inward = None

        cv2.putText(
            out,
            f"ID:{int(tag_id)}",
            (c_x - 18, c_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        cv2.circle(out, (c_x, c_y), 4, (0, 0, 255), -1)

        if not DRAW_TAG_FRAME_2D:
            cv2.polylines(out, [corners], True, (0, 255, 0), 4)
            continue

        tag_pose = tag_pose_by_id.get(int(tag_id))
        if tag_pose is None:
            cv2.polylines(out, [corners], True, (0, 0, 255), 4)
            continue

        tag_rot_mat = np.asarray(tag_pose["rot_mat"], dtype=np.float64)
        tag_tvec = np.asarray(tag_pose["tvec"], dtype=np.float64)
        tag_is_inward = tag_pose.get("z_inward", None)

        border_color = (0, 255, 0) if tag_is_inward is not False else (0, 0, 255)
        cv2.polylines(out, [corners], True, border_color, 4)

        draw_tag_frame_projection(
            img=out,
            pose_R=tag_rot_mat,
            pose_t=tag_tvec,
            camera_matrix=detector.camera_matrix,
            dist_coeffs=detector.dist_coeffs,
            axis_length_mm=tag_axis_length_mm,
        )

        if tag_is_inward is False:
            cv2.putText(
                out,
                "z-out",
                (c_x - 20, c_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
            )

    return out


def rotation_matrix_to_euler_xyz_deg(rot_mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to xyz Euler angles in degrees.

    Avoids requiring scipy.
    """
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
        y += 26

    return out


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Detect AprilCube using OAK camera manager.")
    parser.add_argument(
        "--cameras",
        type=str,
        default=",".join(ACTIVE_CAMERA_NAMES),
        help="Comma-separated logical camera names, e.g. r_wrist or r_wrist,l_wrist.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable AprilCube temporal filter.",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Use slower but more accurate AprilCube detector.",
    )
    args = parser.parse_args()

    global ENABLE_FILTER
    global FAST_DETECTOR

    if args.no_filter:
        ENABLE_FILTER = False

    if args.slow:
        FAST_DETECTOR = False

    active_camera_names = [x.strip() for x in args.cameras.split(",") if x.strip()]

    if not active_camera_names:
        print("[ERROR] No active camera names specified.")
        sys.exit(1)

    missing_camera_cfg = [name for name in active_camera_names if name not in CAMERA_TO_DEVICE]
    if missing_camera_cfg:
        print(f"[ERROR] Missing CAMERA_TO_DEVICE entries for: {missing_camera_cfg}")
        print("[ERROR] Please edit CAMERA_TO_DEVICE at the top of this script.")
        sys.exit(1)

    if PRINT_AVAILABLE_DEVICES:
        list_oak_devices()

    cube_path = APRILCUBE_SRC_DIR.resolve()
    if cube_path.is_dir():
        cube_config_path = cube_path / "config.json"
        if not cube_config_path.is_file():
            print(f"[ERROR] APRILCUBE_SRC_DIR has no config.json: {cube_path}")
            sys.exit(1)
    elif cube_path.is_file() and cube_path.name == "config.json":
        cube_config_path = cube_path
    else:
        print(f"[ERROR] APRILCUBE_SRC_DIR does not exist or is invalid: {cube_path}")
        sys.exit(1)

    print(f"[INFO] Using AprilCube model/config: {cube_path}")

    detectors = create_detectors(
        cube_path=cube_path,
        camera_names=active_camera_names,
    )
    pose_estimators = create_pose_estimators(
        detectors=detectors,
        camera_names=active_camera_names,
    )
    native_family = apriltag_family_from_dict_name(detectors[active_camera_names[0]].config.dict_name)
    native_tag_detector = Detector(
        families=native_family,
        quad_decimate=1.0,
    )
    print(f"[INFO] Native AprilTag detector family: {native_family}")
    print(f"[INFO] Use temporal tag pose estimator: {USE_TEMPORAL_TAG_POSE_ESTIMATOR}")
    print(f"[INFO] Pupil-to-object corner reorder index: {PUPIL_TO_OBJECT_CORNER_INDEX}")
    print(f"[INFO] solvePnPGeneric flag: {SOLVEPNP_GENERIC_FLAG}")
    cube_runtimes = {
        camera_name: AprilCubeTemporalPoseRuntime(
            detector=detectors[camera_name],
            native_detector=native_tag_detector,
            pose_estimator=pose_estimators[camera_name],
            use_clahe_for_native_detection=USE_CLAHE_FOR_TAG_DETECTION,
            clahe_clip_limit=CLAHE_CLIP_LIMIT,
            clahe_tile_grid_size=CLAHE_TILE_GRID_SIZE,
        )
        for camera_name in active_camera_names
    }
    _viser_servers, viser_gray_image_handles, viser_record_checkbox_handles = build_viser_servers(
        detectors=detectors,
        camera_names=active_camera_names,
    )

    camera_manager = OAK1WCameraManager(
        camera_to_device={
            name: CAMERA_TO_DEVICE[name]
            for name in active_camera_names
        },
        isp_scale=ISP_SCALE,
        fps=FPS,
        rotate_180_names=ROTATE_180_NAMES,
        queue_size=QUEUE_SIZE,
        queue_blocking=QUEUE_BLOCKING,
    )

    try:
        opened = camera_manager.open_all_cameras()

        if opened == 0:
            print("[ERROR] No OAK camera opened.")
            sys.exit(1)

        opened_names = camera_manager.get_active_camera_names()
        print(f"[INFO] Opened OAK cameras: {opened_names}")
        print("[INFO] Waiting for first frames...")

        camera_manager.wait_for_first_frames(
            camera_names=opened_names,
            timeout_s=5.0,
        )

        print("[INFO] Press 'q' or ESC to quit.")

        frame_idx = 0
        last_no_frame_print_time = 0.0
        latest_snapshot_by_camera: dict[str, dict[str, Any]] = {}
        recording_state: dict[str, Any] | None = None
        recording_active = False

        while True:
            frame_idx += 1
            latest_snapshot_by_camera = {}

            checkbox_requests = {
                camera_name: bool(handle.value)
                for camera_name, handle in viser_record_checkbox_handles.items()
            }
            if any(request != recording_active for request in checkbox_requests.values()):
                requested_active = any(checkbox_requests.values())
                if requested_active and not recording_active:
                    recording_state = start_recording_state()
                    recording_active = True
                    set_recording_checkbox_values(viser_record_checkbox_handles, True)
                    print("[INFO] Recording started from viser checkbox.")
                elif (not requested_active) and recording_active:
                    set_recording_checkbox_values(viser_record_checkbox_handles, False)
                    recording_active = False
                    if recording_state is not None and recording_state.get("frames"):
                        save_path = save_recording_pkl(
                            cube_path=cube_path,
                            recording_state=recording_state,
                        )
                        print(f"[INFO] Saved recording pkl: {save_path}")
                    else:
                        print("[INFO] Recording stopped with no captured frames.")
                    recording_state = None

            frames, _origin_frames = camera_manager.get_frames(
                camera_names=opened_names,
                img_size=DETECT_IMG_SIZE,
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
                detector = detectors[camera_name]
                cube_runtime = cube_runtimes[camera_name]

                detect_frame = frame
                if UNDISTORT_BEFORE_DETECTION:
                    raw_dist_coeffs = DIST_COEFFS_BY_CAMERA.get(camera_name)
                    if raw_dist_coeffs is not None:
                        raw_dist_coeffs = np.asarray(raw_dist_coeffs, dtype=np.float64)
                        detect_frame = cv2.undistort(
                            frame,
                            detector.camera_matrix,
                            raw_dist_coeffs,
                        )

                detector_gray = cube_runtime.prepare_native_detection_gray(detect_frame)

                result = cube_runtime.process_frame(
                    camera_name=camera_name,
                    image=detect_frame,
                )

                try:
                    vis = detector.draw_result(detect_frame.copy(), result)
                except Exception as exc:
                    print(f"[WARNING] draw_result failed for {camera_name}: {type(exc).__name__}: {exc}")
                    vis = detect_frame.copy()

                vis = cube_runtime.draw_detected_tag_visuals(
                    img=vis,
                    result=result,
                    draw_tag_frame_2d=DRAW_TAG_FRAME_2D,
                    tag_axis_length_scale=TAG_AXIS_LENGTH_SCALE,
                )

                status = result_to_text(camera_name, result)
                tag_edge_status = tag_edge_length_text(camera_name, result)
                log_frame_handedness(
                    camera_name=camera_name,
                    frame_idx=frame_idx,
                    result=result,
                )

                print(status)
                print(tag_edge_status)

                panel_lines = [
                    status,
                    tag_edge_status,
                    make_handedness_overlay_text(result),
                    f"detect_size={DETECT_IMG_SIZE}, isp_scale={ISP_SCALE}, fps={FPS}",
                    f"recording={'ON' if recording_active else 'OFF'} (press s to toggle)",
                    "press q or ESC to quit",
                ]
                vis = draw_text_panel(vis, panel_lines)

                if camera_name in viser_gray_image_handles:
                    viser_gray_image_handles[camera_name].image = gray_to_rgb(detector_gray)

                latest_snapshot_by_camera[camera_name] = {
                    "frame_bgr": np.array(frame, copy=True),
                    "detect_frame_bgr": np.array(detect_frame, copy=True),
                    "detector_gray": np.array(detector_gray, copy=True),
                    "vis_bgr": np.array(vis, copy=True),
                    "camera_matrix": np.array(detector.camera_matrix, copy=True),
                    "dist_coeffs": np.zeros(5, dtype=np.float64)
                    if UNDISTORT_BEFORE_DETECTION
                    else np.array(detector.dist_coeffs, copy=True),
                    "tag_size_mm": float(detector.config.tag_size_mm),
                    "box_dims_mm": np.array(detector.config.box_dims, copy=True),
                    "result": clone_result_for_pickle(result),
                }

                cv2.imshow(f"{WINDOW_PREFIX}: {camera_name}", vis)

            if recording_active and latest_snapshot_by_camera:
                recording_state["frames"].append(
                    {
                        "frame_idx": int(frame_idx),
                        "timestamp_epoch_s": time.time(),
                        "cameras": latest_snapshot_by_camera,
                    }
                )

            key = cv2.waitKey(1)
            if key == SAVE_PKL_ON_KEY:
                if not recording_active:
                    recording_state = start_recording_state()
                    recording_active = True
                    set_recording_checkbox_values(viser_record_checkbox_handles, True)
                    print("[INFO] Recording started from keyboard.")
                else:
                    set_recording_checkbox_values(viser_record_checkbox_handles, False)
                    recording_active = False
                    if recording_state is not None and recording_state.get("frames"):
                        save_path = save_recording_pkl(
                            cube_path=cube_path,
                            recording_state=recording_state,
                        )
                        print(f"[INFO] Saved recording pkl: {save_path}")
                    else:
                        print("[INFO] Recording stopped with no captured frames.")
                    recording_state = None
            elif key == 27 or key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        if 'recording_active' in locals() and recording_active and recording_state is not None:
            if recording_state.get("frames"):
                save_path = save_recording_pkl(
                    cube_path=cube_path,
                    recording_state=recording_state,
                )
                print(f"[INFO] Saved recording pkl on exit: {save_path}")
        camera_manager.release_all()
        cv2.destroyAllWindows()
        print("[INFO] Finished.")


if __name__ == "__main__":
    main()
