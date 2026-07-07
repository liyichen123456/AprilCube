# OpenCV / AprilTag camera convention
# +x: image right
# +y: image down
# +z: camera forward
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import viser
import yaml
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

THIS_FILE = Path(__file__).resolve()
THIRDPARTY_DIR = THIS_FILE.parent.parent.parent
PROJECT_ROOT = THIRDPARTY_DIR.parent
UTILS_DIR = PROJECT_ROOT / "scripts" / "utils"

if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

import aprilcube  # noqa: E402
from april_tag_detector import TemporalTagPoseEstimator  # noqa: E402
from aprilcube_runtime import AprilCubeTemporalPoseRuntime, is_valid_rotation_matrix  # noqa: E402
from recorder_oak_cam import OAK1WCameraManager, list_oak_devices  # noqa: E402

# ============================================================
# User macros
# ============================================================

PRINT_AVAILABLE_DEVICES = True

CAMERA_TO_DEVICE: dict[str, str] = {
    "cam0": "3.10.4.4.2",
    "cam1": "3.10.4.4.3",
    "cam2": "3.10.4.4.4",
}

CAMERA_TO_INTRINSICS_YAML: dict[str, str] = {
    "cam0": "/home/ps/RobotCamCalib1/outputs/intrinsics_cam0.yaml",
    "cam1": "/home/ps/RobotCamCalib1/outputs/intrinsics_cam1.yaml",
    "cam2": "/home/ps/RobotCamCalib1/outputs/intrinsics_cam2.yaml",
}

ACTIVE_CAMERA_NAMES: list[str] = ["cam0", "cam1", "cam2"]

DETECT_IMG_SIZE: tuple[int, int] = (1280, 960)
ISP_SCALE: tuple[int, int] = (1, 3)
FPS = 25
QUEUE_SIZE = 4
QUEUE_BLOCKING = False
ROTATE_180_NAMES: set[str] = set()

INIT_TAG_FAMILY = "tagCustom48h12"
INIT_TAG_ID = 0
INIT_TAG_SIZE_M = 0.1
INIT_REQUIRED_SAMPLES_PER_CAMERA = 10

CUBE_CFG_DIRS: list[Path] = [
    # THIRDPARTY_DIR / "aprilcube" / "cubes" / "cube_april_36h11_0_5_1x1x1_10mm",
    THIRDPARTY_DIR / "aprilcube" / "cubes" / "cube_april_36h11_6_11_1x1x1_10mm",
    THIRDPARTY_DIR / "aprilcube" / "cubes" / "cube_april_36h11_12_17_1x1x1_10mm",
    THIRDPARTY_DIR / "aprilcube" / "cubes" / "cube_april_36h11_18_23_1x1x1_10mm",
    # THIRDPARTY_DIR / "aprilcube" / "cubes" / "cube_april_36h11_24_29_1x1x1_10mm",
    # THIRDPARTY_DIR / "aprilcube" / "cubes" / "cube_april_36h11_30_35_1x1x1_10mm",
]

ENABLE_FILTER = True
FAST_DETECTOR = True
USE_TEMPORAL_TAG_POSE_ESTIMATOR = True
PUPIL_TO_OBJECT_CORNER_INDEX = [2, 1, 0, 3]
USE_SOLVEPNP_REFINE_LM = True
USE_TEMPORAL_CANDIDATE_SELECTION = True
SOLVEPNP_GENERIC_FLAG = cv2.SOLVEPNP_IPPE
SOLVEPNP_FLAG = cv2.SOLVEPNP_ITERATIVE
TRANSLATION_SCORE_WEIGHT_DEG_PER_MM = 1.0
REJECT_NEGATIVE_CAMERA_Z = True

PRINT_EVERY_N_FRAMES = 5
UNDISTORT_BEFORE_DETECTION = True
SHOW_CV2_WINDOWS = False
WINDOW_PREFIX = "OAK Multi-AprilCube World"
PROCESSING_FPS_EMA_ALPHA = 0.2

VISER_HOST = "0.0.0.0"
VISER_PORT = 8080
WORLD_AXES_LENGTH_M = 0.15
WORLD_AXES_RADIUS_M = 0.003
WORLD_ORIGIN_RADIUS_M = 0.004
CAMERA_AXES_LENGTH_M = 0.08
CAMERA_AXES_RADIUS_M = 0.002
CAMERA_ORIGIN_RADIUS_M = 0.003
CUBE_AXES_LENGTH_M = 0.04
CUBE_AXES_RADIUS_M = 0.0015
CUBE_ORIGIN_RADIUS_M = 0.0025
TEMPORAL_TRANSLATION_GATE_M = 0.01
TEMPORAL_ROTATION_GATE_DEG = 90.0
PRINT_TIMING = True


# ============================================================
# Utilities
# ============================================================

def load_intrinsics_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path).expanduser().resolve()
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    image_size = tuple(int(v) for v in data["image_size"])
    k = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(data["dist"], dtype=np.float64).reshape(-1)
    return {
        "path": str(yaml_path),
        "image_size": image_size,
        "K": k,
        "dist": dist,
    }


def scale_intrinsics(
    k: np.ndarray,
    old_size: tuple[int, int],
    new_size: tuple[int, int],
) -> np.ndarray:
    old_w, old_h = old_size
    new_w, new_h = new_size
    sx = new_w / old_w
    sy = new_h / old_h

    k_new = np.asarray(k, dtype=np.float64).copy()
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


def k_to_camera_params(k: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(k[0, 0]),
        float(k[1, 1]),
        float(k[0, 2]),
        float(k[1, 2]),
    )


def apriltag_family_from_dict_name(dict_name: str) -> str:
    family_map = {
        "apriltag_16h5": "tag16h5",
        "apriltag_25h9": "tag25h9",
        "apriltag_36h10": "tag36h10",
        "apriltag_36h11": "tag36h11",
    }
    if dict_name not in family_map:
        raise ValueError(f"Unsupported native AprilTag family: {dict_name}")
    return family_map[dict_name]


def rotation_matrix_to_wxyz(rot: np.ndarray) -> tuple[float, float, float, float]:
    quat_xyzw = R.from_matrix(np.asarray(rot, dtype=np.float64)).as_quat()
    x, y, z, w = quat_xyzw
    return (float(w), float(x), float(y), float(z))


def invert_pose(pose_R: np.ndarray, pose_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose_R = np.asarray(pose_R, dtype=np.float64).reshape(3, 3)
    pose_t = np.asarray(pose_t, dtype=np.float64).reshape(3)
    rot_inv = pose_R.T
    t_inv = -rot_inv @ pose_t
    return rot_inv, t_inv


def compose_pose(
    a_R_b: np.ndarray,
    a_t_b: np.ndarray,
    b_R_c: np.ndarray,
    b_t_c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    a_R_b = np.asarray(a_R_b, dtype=np.float64).reshape(3, 3)
    a_t_b = np.asarray(a_t_b, dtype=np.float64).reshape(3)
    b_R_c = np.asarray(b_R_c, dtype=np.float64).reshape(3, 3)
    b_t_c = np.asarray(b_t_c, dtype=np.float64).reshape(3)
    a_R_c = a_R_b @ b_R_c
    a_t_c = a_R_b @ b_t_c + a_t_b
    return a_R_c, a_t_c


def average_pose_candidates(
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
        tvecs.append(np.asarray(cand["tvec"], dtype=np.float64).reshape(3))

    weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr /= np.sum(weights_arr)

    rot_avg = AprilCubeTemporalPoseRuntime.average_rotations(rot_mats, weights_arr)
    if rot_avg is None:
        return None

    t_avg = np.zeros(3, dtype=np.float64)
    for weight, tvec in zip(weights_arr, tvecs):
        t_avg += float(weight) * tvec
    return rot_avg, t_avg


def rotation_angle_deg_between(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rot_a = np.asarray(rot_a, dtype=np.float64).reshape(3, 3)
    rot_b = np.asarray(rot_b, dtype=np.float64).reshape(3, 3)
    rel = rot_a.T @ rot_b
    trace = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def candidate_sort_key(cand: dict[str, Any]) -> tuple[float, int, str, int]:
    err = cand.get("reproj_error", None)
    reproj = float("inf") if err is None else float(err)
    pnp_idx = int(cand.get("pnp_candidate_index", 999))
    cam = str(cand.get("camera_name", ""))
    tag_id = int(cand.get("tag_id", -1))
    return (reproj, pnp_idx, cam, tag_id)


def add_timing_ms(timing: dict[str, float], key: str, dt_s: float) -> None:
    timing[key] = timing.get(key, 0.0) + float(dt_s) * 1000.0


def format_timing_ms(timing: dict[str, float], keys: list[str]) -> str:
    parts: list[str] = []
    for key in keys:
        if key in timing:
            parts.append(f"{key}={timing[key]:.1f}ms")
    return " ".join(parts)


def filter_cube_candidates_with_previous_pose(
    cube_candidates: list[dict[str, Any]],
    previous_pose: dict[str, np.ndarray] | None,
    translation_gate_m: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not cube_candidates:
        return [], [], {
            "had_previous_pose": previous_pose is not None,
            "raw_count": 0,
            "passed_count": 0,
            "fallback_used": False,
            "fallback_reason": "no raw candidates",
        }

    if previous_pose is None:
        return cube_candidates, [], {
            "had_previous_pose": False,
            "raw_count": len(cube_candidates),
            "passed_count": len(cube_candidates),
            "fallback_used": False,
            "fallback_reason": "",
        }

    prev_rot = np.asarray(previous_pose["rot_mat"], dtype=np.float64).reshape(3, 3)
    prev_t = np.asarray(previous_pose["tvec"], dtype=np.float64).reshape(3)

    gated: list[dict[str, Any]] = []
    rejected_debug: list[dict[str, Any]] = []
    for cand in cube_candidates:
        rot = np.asarray(cand["rot_mat"], dtype=np.float64).reshape(3, 3)
        tvec = np.asarray(cand["tvec"], dtype=np.float64).reshape(3)
        trans_delta_m = float(np.linalg.norm(tvec - prev_t))
        rot_delta_deg = rotation_angle_deg_between(prev_rot, rot)
        cand["temporal_translation_delta_m"] = trans_delta_m
        cand["temporal_rotation_delta_deg"] = rot_delta_deg
        if (
            trans_delta_m <= float(translation_gate_m)
            and rot_delta_deg <= TEMPORAL_ROTATION_GATE_DEG
        ):
            gated.append(cand)
        else:
            rejected_debug.append(
                {
                    "camera_name": cand.get("camera_name", "?"),
                    "tag_id": cand.get("tag_id", -1),
                    "pnp_candidate_index": cand.get("pnp_candidate_index", -1),
                    "trans_delta_m": trans_delta_m,
                    "rot_delta_deg": rot_delta_deg,
                }
            )

    if gated:
        return gated, rejected_debug, {
            "had_previous_pose": True,
            "raw_count": len(cube_candidates),
            "passed_count": len(gated),
            "fallback_used": False,
            "fallback_reason": "",
        }

    fallback = sorted(cube_candidates, key=candidate_sort_key)[:1]
    return fallback, rejected_debug, {
        "had_previous_pose": True,
        "raw_count": len(cube_candidates),
        "passed_count": 0,
            "fallback_used": True,
            "fallback_reason": "all candidates rejected by temporal gate; fallback to best current candidate",
        }


def cube_pose_from_world_tag_pose(
    runtime: AprilCubeTemporalPoseRuntime,
    tag_id: int,
    world_R_tag: np.ndarray,
    world_t_tag_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    tag_to_cube = runtime.build_tag_to_cube_transform(tag_id)
    if tag_to_cube is None:
        return None

    rot_cube_tag, center_cube_mm = tag_to_cube
    rot_tag_cube = rot_cube_tag.T
    center_tag_m = (-rot_tag_cube @ center_cube_mm).reshape(3) / 1000.0

    world_R_tag = np.asarray(world_R_tag, dtype=np.float64).reshape(3, 3)
    world_t_tag_m = np.asarray(world_t_tag_m, dtype=np.float64).reshape(3)

    world_R_cube = world_R_tag @ rot_tag_cube
    world_t_cube = world_R_tag @ center_tag_m + world_t_tag_m
    if not is_valid_rotation_matrix(world_R_cube):
        return None
    return world_R_cube, world_t_cube


def get_tag_object_corners(tag_size_m: float) -> np.ndarray:
    """Object corners in tag frame: [TL, TR, BR, BL]."""
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


def reorder_pupil_corners_to_object_order(corners_xy: np.ndarray) -> np.ndarray:
    """Reorder pupil_apriltags corners to match [TL, TR, BR, BL]."""
    corners = np.asarray(corners_xy, dtype=np.float64).reshape(4, 2)
    return corners[np.asarray(PUPIL_TO_OBJECT_CORNER_INDEX, dtype=np.int64)]


def reprojection_error_px(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    k: np.ndarray,
    dist: np.ndarray,
) -> float:
    proj, _ = cv2.projectPoints(
        objectPoints=obj_pts,
        rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        tvec=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        cameraMatrix=np.asarray(k, dtype=np.float64),
        distCoeffs=np.asarray(dist, dtype=np.float64).reshape(-1, 1),
    )
    proj = proj.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(proj - img_pts, axis=1)))


def estimate_tag_pose_candidates_solvepnp_ippe(
    corners_xy: np.ndarray,
    k: np.ndarray,
    dist_coeffs: np.ndarray | None,
    tag_size_m: float,
) -> list[dict[str, Any]]:
    """
    Return all planar PnP candidates from one detected AprilTag.

    Output convention:
        X_cam = R_cam_tag @ X_tag + t_cam_tag
    """
    obj_pts = get_tag_object_corners(tag_size_m)
    img_pts = reorder_pupil_corners_to_object_order(corners_xy)

    k = np.asarray(k, dtype=np.float64)
    if dist_coeffs is None:
        dist = np.zeros((5, 1), dtype=np.float64)
    else:
        dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)

    try:
        retval, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
            objectPoints=obj_pts,
            imagePoints=img_pts,
            cameraMatrix=k,
            distCoeffs=dist,
            flags=SOLVEPNP_GENERIC_FLAG,
        )
    except cv2.error as exc:
        print(f"[WARNING] solvePnPGeneric failed: {exc}")
        return []

    if int(retval) <= 0:
        return []

    if reproj_errs is None:
        opencv_reproj_errs = [None] * len(rvecs)
    else:
        opencv_reproj_errs = np.asarray(reproj_errs, dtype=np.float64).reshape(-1).tolist()

    candidates: list[dict[str, Any]] = []
    for cand_idx, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        pose_t = np.asarray(tvec, dtype=np.float64).reshape(3)
        pose_R, _ = cv2.Rodrigues(rvec)

        if not is_valid_rotation_matrix(pose_R):
            continue
        if not np.all(np.isfinite(pose_t)):
            continue
        if REJECT_NEGATIVE_CAMERA_Z and float(pose_t[2]) <= 0.0:
            continue

        err_px = reprojection_error_px(
            obj_pts=obj_pts,
            img_pts=img_pts,
            rvec=rvec,
            tvec=pose_t,
            k=k,
            dist=dist,
        )
        candidates.append(
            {
                "candidate_index": int(cand_idx),
                "rot_mat": pose_R,
                "tvec": pose_t,
                "reproj_error": err_px,
                "opencv_reproj_error": opencv_reproj_errs[cand_idx],
            }
        )

    return candidates


def build_cube_candidates_from_tag_candidates(
    runtime: AprilCubeTemporalPoseRuntime,
    camera_name: str,
    tag_id: int,
    tag_pose_candidates_cam: list[dict[str, Any]],
    world_R_cam: np.ndarray,
    world_t_cam: np.ndarray,
) -> list[dict[str, Any]]:
    """Convert all T_cam_tag candidates into T_world_cube candidates."""
    cube_candidates: list[dict[str, Any]] = []

    for cand in tag_pose_candidates_cam:
        cam_R_tag = np.asarray(cand["rot_mat"], dtype=np.float64).reshape(3, 3)
        cam_t_tag = np.asarray(cand["tvec"], dtype=np.float64).reshape(3)

        world_R_tag, world_t_tag = compose_pose(
            world_R_cam,
            world_t_cam,
            cam_R_tag,
            cam_t_tag,
        )

        cube_pose = cube_pose_from_world_tag_pose(
            runtime=runtime,
            tag_id=int(tag_id),
            world_R_tag=world_R_tag,
            world_t_tag_m=world_t_tag,
        )
        if cube_pose is None:
            continue

        world_R_cube, world_t_cube = cube_pose
        cube_candidates.append(
            {
                "camera_name": camera_name,
                "tag_id": int(tag_id),
                "pnp_candidate_index": int(cand.get("candidate_index", -1)),
                "rot_mat": world_R_cube,
                "tvec": world_t_cube,
                "reproj_error": cand.get("reproj_error", None),
                "tag_world_rot_mat": world_R_tag,
                "tag_world_tvec": world_t_tag,
            }
        )

    return cube_candidates


def explain_fused_none_reason(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "no world candidates"

    invalid_rot_indices: list[int] = []
    invalid_t_indices: list[int] = []
    valid_rot_mats: list[np.ndarray] = []
    for idx, cand in enumerate(candidates):
        rot = np.asarray(cand.get("rot_mat", None), dtype=np.float64).reshape(3, 3)
        tvec = np.asarray(cand.get("tvec", None), dtype=np.float64).reshape(3)
        if not is_valid_rotation_matrix(rot):
            invalid_rot_indices.append(idx)
            continue
        if not np.all(np.isfinite(tvec)):
            invalid_t_indices.append(idx)
            continue
        valid_rot_mats.append(rot)

    if invalid_rot_indices:
        return f"invalid rotation candidates at indices={invalid_rot_indices}"
    if invalid_t_indices:
        return f"invalid translation candidates at indices={invalid_t_indices}"
    if not valid_rot_mats:
        return "no valid rotation candidates after filtering"

    weights = []
    for cand in candidates:
        err = cand.get("reproj_error", None)
        weight = 1.0 if err is None else 1.0 / max(float(err), 1e-3)
        weights.append(weight)
    weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr /= np.sum(weights_arr)

    rot_avg = AprilCubeTemporalPoseRuntime.average_rotations(valid_rot_mats, weights_arr[: len(valid_rot_mats)])
    if rot_avg is None:
        dets = [float(np.linalg.det(rot)) for rot in valid_rot_mats]
        return f"rotation average invalid; candidate dets={dets}"
    return "unknown fusion failure"


def validate_cube_path(cube_path: Path) -> Path:
    cube_path = cube_path.resolve()
    if cube_path.is_dir() and (cube_path / "config.json").is_file():
        return cube_path
    if cube_path.is_file() and cube_path.name == "config.json":
        return cube_path
    raise FileNotFoundError(f"Invalid AprilCube cfg path: {cube_path}")


def create_detector_for_camera(cube_path: Path, camera_name: str, calib_by_camera: dict[str, dict[str, Any]]) -> Any:
    calib = calib_by_camera[camera_name]
    k_scaled = scale_intrinsics(
        calib["K"],
        old_size=tuple(calib["image_size"]),
        new_size=DETECT_IMG_SIZE,
    )
    intrinsic_cfg = camera_matrix_to_intrinsic_dict(k_scaled)
    dist_coeffs = np.asarray(calib["dist"], dtype=np.float64)

    detector_dist_coeffs = dist_coeffs
    if UNDISTORT_BEFORE_DETECTION:
        detector_dist_coeffs = np.zeros(5, dtype=np.float64)

    return aprilcube.detector(
        cube_path,
        intrinsic_cfg=intrinsic_cfg,
        dist_coeffs=detector_dist_coeffs,
        enable_filter=ENABLE_FILTER,
        fast=FAST_DETECTOR,
    )


def create_pose_estimator(detector: Any) -> TemporalTagPoseEstimator:
    return TemporalTagPoseEstimator(
        tag_size_m=float(detector.config.tag_size_mm) / 1000.0,
        pupil_to_object_corner_index=PUPIL_TO_OBJECT_CORNER_INDEX,
        solvepnp_generic_flag=SOLVEPNP_GENERIC_FLAG,
        solvepnp_flag=SOLVEPNP_FLAG,
        use_temporal_candidate_selection=USE_TEMPORAL_TAG_POSE_ESTIMATOR and USE_TEMPORAL_CANDIDATE_SELECTION,
        use_solvepnp_refine_lm=USE_SOLVEPNP_REFINE_LM,
        translation_score_weight_deg_per_mm=TRANSLATION_SCORE_WEIGHT_DEG_PER_MM,
        reject_negative_camera_z=REJECT_NEGATIVE_CAMERA_Z,
    )


def create_viser_server() -> tuple[viser.ViserServer, dict[str, Any], dict[str, Any], dict[str, Any]]:
    server = viser.ViserServer(host=VISER_HOST, port=VISER_PORT)
    server.scene.set_up_direction("-y")
    server.scene.world_axes.visible = False

    camera_frame_handles: dict[str, Any] = {}
    cube_frame_handles: dict[str, Any] = {}
    image_handles: dict[str, Any] = {}

    server.scene.add_frame(
        "/world",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        axes_length=WORLD_AXES_LENGTH_M,
        axes_radius=WORLD_AXES_RADIUS_M,
        origin_radius=WORLD_ORIGIN_RADIUS_M,
    )
    server.scene.add_frame(
        "/tag_world",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        axes_length=WORLD_AXES_LENGTH_M,
        axes_radius=WORLD_AXES_RADIUS_M,
        origin_radius=WORLD_ORIGIN_RADIUS_M,
    )

    for camera_name in ACTIVE_CAMERA_NAMES:
        camera_frame_handles[camera_name] = server.scene.add_frame(
            f"/camera/{camera_name}",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            axes_length=CAMERA_AXES_LENGTH_M,
            axes_radius=CAMERA_AXES_RADIUS_M,
            origin_radius=CAMERA_ORIGIN_RADIUS_M,
            visible=False,
        )
        image_handles[camera_name] = server.gui.add_image(
            np.zeros((120, 160, 3), dtype=np.uint8),
            label=camera_name,
        )

    for cube_path in CUBE_CFG_DIRS:
        cube_name = cube_path.name if cube_path.is_dir() else cube_path.parent.name
        cube_frame_handles[cube_name] = server.scene.add_frame(
            f"/cube/{cube_name}",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            axes_length=CUBE_AXES_LENGTH_M,
            axes_radius=CUBE_AXES_RADIUS_M,
            origin_radius=CUBE_ORIGIN_RADIUS_M,
            visible=False,
        )

    print(f"[INFO] Viser server started on http://{VISER_HOST}:{VISER_PORT}")
    print("[INFO] World frame is fixed from the initialization AprilTag.")
    return server, camera_frame_handles, cube_frame_handles, image_handles


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    img_bgr = np.asarray(img_bgr, dtype=np.uint8)
    if img_bgr.ndim == 2:
        return np.repeat(img_bgr[:, :, None], 3, axis=2)
    return img_bgr[:, :, ::-1]


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


def draw_tag_pose_overlay(
    img: np.ndarray,
    tag: Any,
    k: np.ndarray,
    axis_length_m: float = 0.05,
) -> np.ndarray:
    out = img.copy()
    corners = np.round(np.asarray(tag.corners, dtype=np.float64)).astype(np.int32)
    cv2.polylines(out, [corners], True, (0, 255, 0), 2)
    cv2.putText(
        out,
        f"ID:{int(tag.tag_id)}",
        (int(tag.center[0]) - 12, int(tag.center[1]) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )

    pose_R = np.asarray(tag.pose_R, dtype=np.float64).reshape(3, 3)
    pose_t = np.asarray(tag.pose_t, dtype=np.float64).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(pose_R)
    obj_pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length_m, 0.0, 0.0],
            [0.0, axis_length_m, 0.0],
            [0.0, 0.0, axis_length_m],
        ],
        dtype=np.float64,
    )
    img_pts, _ = cv2.projectPoints(obj_pts, rvec, pose_t, np.asarray(k, dtype=np.float64), np.zeros(5))
    img_pts = np.round(img_pts.reshape(-1, 2)).astype(np.int32)
    origin = tuple(img_pts[0])
    pt_x = tuple(img_pts[1])
    pt_y = tuple(img_pts[2])
    pt_z = tuple(img_pts[3])
    cv2.arrowedLine(out, origin, pt_x, (0, 0, 255), 3, tipLength=0.2)
    cv2.arrowedLine(out, origin, pt_y, (0, 255, 0), 3, tipLength=0.2)
    cv2.arrowedLine(out, origin, pt_z, (255, 0, 0), 3, tipLength=0.2)
    return out


def detect_target_tag(
    detector: Detector,
    image_bgr: np.ndarray,
    camera_params: tuple[float, float, float, float],
) -> Any | None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=INIT_TAG_SIZE_M,
    )
    matches = [tag for tag in tags if int(tag.tag_id) == INIT_TAG_ID]
    if len(matches) != 1:
        return None
    return matches[0]


def initialize_world_from_tag(
    camera_manager: OAK1WCameraManager,
    opened_names: list[str],
    calib_by_camera: dict[str, dict[str, Any]],
    tag_detector: Detector,
    camera_frame_handles: dict[str, Any],
    image_handles: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    world_from_camera_samples: dict[str, list[dict[str, np.ndarray]]] = {
        name: [] for name in opened_names
    }
    last_print_time = 0.0

    print("[INFO] Initialization phase started: waiting for the fixed world AprilTag.")

    while True:
        frames, _origin_frames = camera_manager.get_frames(
            camera_names=opened_names,
            img_size=DETECT_IMG_SIZE,
        )

        if not frames:
            time.sleep(0.02)
            continue

        completed = True
        for camera_name in opened_names:
            frame = frames.get(camera_name)
            if frame is None:
                completed = False
                continue

            calib = calib_by_camera[camera_name]
            k_scaled = scale_intrinsics(
                calib["K"],
                old_size=tuple(calib["image_size"]),
                new_size=DETECT_IMG_SIZE,
            )
            dist = np.asarray(calib["dist"], dtype=np.float64).reshape(-1)

            detect_frame = frame
            if UNDISTORT_BEFORE_DETECTION:
                detect_frame = cv2.undistort(frame, k_scaled, dist)

            tag = detect_target_tag(
                detector=tag_detector,
                image_bgr=detect_frame,
                camera_params=k_to_camera_params(k_scaled),
            )

            if tag is not None:
                pose_R = np.asarray(tag.pose_R, dtype=np.float64).reshape(3, 3)
                pose_t = np.asarray(tag.pose_t, dtype=np.float64).reshape(3)
                if is_valid_rotation_matrix(pose_R):
                    world_R_cam, world_t_cam = invert_pose(pose_R, pose_t)
                    world_from_camera_samples[camera_name].append(
                        {
                            "rot_mat": world_R_cam,
                            "tvec": world_t_cam,
                        }
                    )
                    camera_frame_handles[camera_name].wxyz = rotation_matrix_to_wxyz(world_R_cam)
                    camera_frame_handles[camera_name].position = (
                        float(world_t_cam[0]),
                        float(world_t_cam[1]),
                        float(world_t_cam[2]),
                    )
                    camera_frame_handles[camera_name].visible = True
                    detect_frame = draw_tag_pose_overlay(detect_frame, tag, k_scaled)
            else:
                completed = False

            samples = len(world_from_camera_samples[camera_name])
            if samples < INIT_REQUIRED_SAMPLES_PER_CAMERA:
                completed = False

            overlay = draw_text_panel(
                detect_frame,
                [
                    f"[init][{camera_name}] samples={samples}/{INIT_REQUIRED_SAMPLES_PER_CAMERA}",
                    f"family={INIT_TAG_FAMILY} id={INIT_TAG_ID} size_m={INIT_TAG_SIZE_M}",
                ],
            )
            image_handles[camera_name].image = bgr_to_rgb(overlay)
            if SHOW_CV2_WINDOWS:
                cv2.imshow(f"{WINDOW_PREFIX} init: {camera_name}", overlay)

        now = time.time()
        if now - last_print_time > 1.0:
            progress = ", ".join(
                f"{name}:{len(world_from_camera_samples[name])}/{INIT_REQUIRED_SAMPLES_PER_CAMERA}"
                for name in opened_names
            )
            print(f"[INFO] init progress: {progress}")
            last_print_time = now

        if completed:
            break

        key = cv2.waitKey(1)
        if key == 27 or key == ord("q"):
            raise KeyboardInterrupt

    world_from_camera: dict[str, dict[str, np.ndarray]] = {}
    for camera_name in opened_names:
        fused = average_pose_candidates(world_from_camera_samples[camera_name])
        if fused is None:
            raise RuntimeError(f"Failed to fuse initialization poses for {camera_name}")
        rot_mat, tvec = fused
        world_from_camera[camera_name] = {
            "rot_mat": rot_mat,
            "tvec": tvec,
        }
        camera_frame_handles[camera_name].wxyz = rotation_matrix_to_wxyz(rot_mat)
        camera_frame_handles[camera_name].position = (
            float(tvec[0]),
            float(tvec[1]),
            float(tvec[2]),
        )
        camera_frame_handles[camera_name].visible = True
        print(
            f"[INFO] [{camera_name}] fixed world pose from init: "
            f"t_m=({tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f})"
        )

    print("[INFO] Initialization finished. World/camera extrinsics are now treated as time-invariant.")
    return world_from_camera


def build_cube_runtimes(
    opened_names: list[str],
    calib_by_camera: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    runtimes_by_camera: dict[str, list[dict[str, Any]]] = {name: [] for name in opened_names}
    shared_native_detectors: dict[str, Detector] = {}

    for cube_path_raw in CUBE_CFG_DIRS:
        cube_path = validate_cube_path(cube_path_raw)
        cube_name = cube_path.name if cube_path.is_dir() else cube_path.parent.name
        for camera_name in opened_names:
            detector = create_detector_for_camera(cube_path, camera_name, calib_by_camera)
            native_family = apriltag_family_from_dict_name(detector.config.dict_name)
            if native_family not in shared_native_detectors:
                shared_native_detectors[native_family] = Detector(
                    families=native_family,
                    quad_decimate=1.0,
                )

            runtime = AprilCubeTemporalPoseRuntime(
                detector=detector,
                native_detector=shared_native_detectors[native_family],
                pose_estimator=create_pose_estimator(detector),
            )
            runtimes_by_camera[camera_name].append(
                {
                    "cube_name": cube_name,
                    "runtime": runtime,
                    "detector": detector,
                }
            )
            print(f"[INFO] Loaded cube cfg for {camera_name}: {cube_name}")

    return runtimes_by_camera


def main() -> None:
    if PRINT_AVAILABLE_DEVICES:
        list_oak_devices()

    missing_devices = [name for name in ACTIVE_CAMERA_NAMES if name not in CAMERA_TO_DEVICE]
    if missing_devices:
        print(f"[ERROR] Missing CAMERA_TO_DEVICE entries for: {missing_devices}")
        sys.exit(1)

    missing_intrinsics = [name for name in ACTIVE_CAMERA_NAMES if name not in CAMERA_TO_INTRINSICS_YAML]
    if missing_intrinsics:
        print(f"[ERROR] Missing CAMERA_TO_INTRINSICS_YAML entries for: {missing_intrinsics}")
        sys.exit(1)

    calib_by_camera = {
        name: load_intrinsics_yaml(CAMERA_TO_INTRINSICS_YAML[name])
        for name in ACTIVE_CAMERA_NAMES
    }
    for name, calib in calib_by_camera.items():
        print(f"[INFO] [{name}] intrinsics_yaml={calib['path']} image_size={calib['image_size']}")

    init_tag_detector = Detector(
        families=INIT_TAG_FAMILY,
        quad_decimate=1.0,
    )
    print(
        f"[INFO] Init AprilTag detector: family={INIT_TAG_FAMILY}, "
        f"id={INIT_TAG_ID}, size_m={INIT_TAG_SIZE_M}"
    )

    server, camera_frame_handles, cube_frame_handles, image_handles = create_viser_server()
    camera_manager = OAK1WCameraManager(
        camera_to_device={name: CAMERA_TO_DEVICE[name] for name in ACTIVE_CAMERA_NAMES},
        isp_scale=ISP_SCALE,
        fps=FPS,
        rotate_180_names=ROTATE_180_NAMES,
        queue_size=QUEUE_SIZE,
        queue_blocking=QUEUE_BLOCKING,
    )

    try:
        opened = camera_manager.open_all_cameras()
        if opened != len(ACTIVE_CAMERA_NAMES):
            print(f"[ERROR] Need all cameras opened for initialization, got {opened}/{len(ACTIVE_CAMERA_NAMES)}.")
            sys.exit(1)

        opened_names = camera_manager.get_active_camera_names()
        print(f"[INFO] Opened OAK cameras: {opened_names}")
        camera_manager.wait_for_first_frames(camera_names=opened_names, timeout_s=5.0)

        world_from_camera = initialize_world_from_tag(
            camera_manager=camera_manager,
            opened_names=opened_names,
            calib_by_camera=calib_by_camera,
            tag_detector=init_tag_detector,
            camera_frame_handles=camera_frame_handles,
            image_handles=image_handles,
        )

        runtimes_by_camera = build_cube_runtimes(opened_names, calib_by_camera)
        cube_runtime_by_name: dict[str, AprilCubeTemporalPoseRuntime] = {}
        for entry in runtimes_by_camera[opened_names[0]]:
            cube_runtime_by_name[entry["cube_name"]] = entry["runtime"]

        scaled_k_by_camera: dict[str, np.ndarray] = {}
        undistort_dist_by_camera: dict[str, np.ndarray] = {}
        pnp_dist_by_camera: dict[str, np.ndarray] = {}
        for camera_name in opened_names:
            calib = calib_by_camera[camera_name]
            scaled_k_by_camera[camera_name] = scale_intrinsics(
                calib["K"],
                old_size=tuple(calib["image_size"]),
                new_size=DETECT_IMG_SIZE,
            )
            undistort_dist_by_camera[camera_name] = np.asarray(calib["dist"], dtype=np.float64).reshape(-1)
            if UNDISTORT_BEFORE_DETECTION:
                pnp_dist_by_camera[camera_name] = np.zeros(5, dtype=np.float64)
            else:
                pnp_dist_by_camera[camera_name] = np.asarray(calib["dist"], dtype=np.float64).reshape(-1)

        print("[INFO] Switched to multi-cube detection.")
        print("[INFO] Press q or ESC to quit.")
        last_world_cube_pose_by_name: dict[str, dict[str, np.ndarray]] = {}
        processing_fps_ema: float | None = None

        frame_idx = 0
        last_no_frame_print_time = 0.0
        while True:
            frame_start_time = time.perf_counter()
            timing_ms: dict[str, float] = {}
            frame_idx += 1
            t0 = time.perf_counter()
            frames, _origin_frames = camera_manager.get_frames(
                camera_names=opened_names,
                img_size=DETECT_IMG_SIZE,
            )
            add_timing_ms(timing_ms, "get_frames", time.perf_counter() - t0)

            if not frames:
                now = time.time()
                if now - last_no_frame_print_time > 1.0:
                    print("[INFO] No frames received yet.")
                    last_no_frame_print_time = now
                if cv2.waitKey(1) in (27, ord("q")):
                    break
                continue

            cube_candidates_by_cube: dict[str, list[dict[str, Any]]] = {
                (cube_path.name if cube_path.is_dir() else cube_path.parent.name): []
                for cube_path in CUBE_CFG_DIRS
            }

            for camera_name in opened_names:
                camera_loop_start = time.perf_counter()
                runtime_entries = runtimes_by_camera[camera_name]
                frame = frames.get(camera_name)
                if frame is None:
                    image_handles[camera_name].image = np.zeros((120, 160, 3), dtype=np.uint8)
                    continue

                detect_frame = frame
                if UNDISTORT_BEFORE_DETECTION:
                    t0 = time.perf_counter()
                    detect_frame = cv2.undistort(
                        frame,
                        scaled_k_by_camera[camera_name],
                        undistort_dist_by_camera[camera_name],
                    )
                    add_timing_ms(timing_ms, f"{camera_name}.undistort", time.perf_counter() - t0)

                vis = detect_frame.copy()
                status_lines = [
                    f"[{camera_name}] cubes={len(runtime_entries)} detect_size={DETECT_IMG_SIZE} fps={FPS}"
                ]

                grouped_entries: dict[tuple[str, float], list[dict[str, Any]]] = {}
                for entry in runtime_entries:
                    runtime = entry["runtime"]
                    key = (runtime.native_family, round(runtime.tag_size_m, 6))
                    grouped_entries.setdefault(key, []).append(entry)

                for _group_key, group_entries in grouped_entries.items():
                    t0 = time.perf_counter()
                    shared_tags = group_entries[0]["runtime"].detect_native_apriltags_all(detect_frame)
                    add_timing_ms(timing_ms, f"{camera_name}.shared_detect", time.perf_counter() - t0)
                    for entry in group_entries:
                        cube_name = entry["cube_name"]
                        detector = entry["detector"]
                        runtime = entry["runtime"]
                        t0 = time.perf_counter()
                        result = runtime.process_frame(
                            camera_name=camera_name,
                            image=detect_frame,
                            native_tags=shared_tags,
                        )
                        add_timing_ms(timing_ms, f"{camera_name}.process_frame", time.perf_counter() - t0)

                        try:
                            t0 = time.perf_counter()
                            vis = detector.draw_result(vis, result)
                            add_timing_ms(timing_ms, f"{camera_name}.draw_cube", time.perf_counter() - t0)
                        except Exception as exc:
                            print(f"[WARNING] draw_result failed for {camera_name}/{cube_name}: {type(exc).__name__}: {exc}")

                        t0 = time.perf_counter()
                        vis = runtime.draw_detected_tag_visuals(
                            img=vis,
                            result=result,
                            draw_tag_frame_2d=True,
                            tag_axis_length_scale=0.8,
                        )
                        add_timing_ms(timing_ms, f"{camera_name}.draw_tags", time.perf_counter() - t0)

                        line = f"[{camera_name}][{cube_name}] "
                        if result.get("success", False):
                            cam_R_cube, _ = cv2.Rodrigues(np.asarray(result["rvec"], dtype=np.float64).reshape(3, 1))
                            cam_t_cube = np.asarray(result["tvec"], dtype=np.float64).reshape(3) / 1000.0
                            world_R_cam = world_from_camera[camera_name]["rot_mat"]
                            world_t_cam = world_from_camera[camera_name]["tvec"]
                            world_R_cube, world_t_cube = compose_pose(
                                world_R_cam,
                                world_t_cam,
                                cam_R_cube,
                                cam_t_cube,
                            )
                            line += (
                                f"success direct_t_world_m=({world_t_cube[0]:.3f}, {world_t_cube[1]:.3f}, {world_t_cube[2]:.3f}) "
                                f"reproj={float(result.get('reproj_error', float('nan'))):.2f}px"
                            )
                        else:
                            line += "cube not detected"

                        world_R_cam = world_from_camera[camera_name]["rot_mat"]
                        world_t_cam = world_from_camera[camera_name]["tvec"]

                        k_scaled = scaled_k_by_camera[camera_name]
                        pnp_dist = pnp_dist_by_camera[camera_name]

                        for tag in shared_tags:
                            tag_id = int(tag.tag_id)
                            t0 = time.perf_counter()
                            tag_candidates_cam = estimate_tag_pose_candidates_solvepnp_ippe(
                                corners_xy=np.asarray(tag.corners, dtype=np.float64),
                                k=k_scaled,
                                dist_coeffs=pnp_dist,
                                tag_size_m=float(runtime.tag_size_m),
                            )
                            add_timing_ms(timing_ms, f"{camera_name}.ippe", time.perf_counter() - t0)
                            if not tag_candidates_cam:
                                continue

                            t0 = time.perf_counter()
                            cube_candidates_from_this_tag = build_cube_candidates_from_tag_candidates(
                                runtime=runtime,
                                camera_name=camera_name,
                                tag_id=tag_id,
                                tag_pose_candidates_cam=tag_candidates_cam,
                                world_R_cam=world_R_cam,
                                world_t_cam=world_t_cam,
                            )
                            add_timing_ms(timing_ms, f"{camera_name}.tag_to_cube", time.perf_counter() - t0)
                            cube_candidates_by_cube[cube_name].extend(cube_candidates_from_this_tag)

                        status_lines.append(line)
                        if frame_idx % PRINT_EVERY_N_FRAMES == 0:
                            print(line)

                status_lines.append("press q or ESC to quit")
                t0 = time.perf_counter()
                vis = draw_text_panel(vis, status_lines)
                image_handles[camera_name].image = bgr_to_rgb(vis)
                add_timing_ms(timing_ms, f"{camera_name}.overlay", time.perf_counter() - t0)

                if SHOW_CV2_WINDOWS:
                    cv2.imshow(f"{WINDOW_PREFIX}: {camera_name}", vis)
                add_timing_ms(timing_ms, f"{camera_name}.total", time.perf_counter() - camera_loop_start)

            t0 = time.perf_counter()
            for cube_name, cube_candidates in cube_candidates_by_cube.items():
                cube_handle = cube_frame_handles[cube_name]
                cube_runtime = cube_runtime_by_name.get(cube_name)
                if cube_runtime is None:
                    cube_handle.visible = False
                    continue

                translation_gate_m = TEMPORAL_TRANSLATION_GATE_M
                if frame_idx % PRINT_EVERY_N_FRAMES == 0:
                    debug_parts = [
                        f"{cand['camera_name']}/tag{cand['tag_id']}/pnp{cand['pnp_candidate_index']}"
                        for cand in cube_candidates
                    ]
                    print(
                        f"[world][{cube_name}] raw_cube_candidates={len(cube_candidates)} {debug_parts}"
                    )

                gated_candidates, rejected_debug, gate_info = filter_cube_candidates_with_previous_pose(
                    cube_candidates,
                    last_world_cube_pose_by_name.get(cube_name),
                    translation_gate_m,
                )
                if frame_idx % PRINT_EVERY_N_FRAMES == 0:
                    print(
                        f"[world][{cube_name}] temporal_gate "
                        f"had_prev={gate_info['had_previous_pose']} "
                        f"raw={gate_info['raw_count']} passed={gate_info['passed_count']} "
                        f"fallback={gate_info['fallback_used']} "
                        f"trans_gate_m={translation_gate_m:.4f} rot_gate_deg={TEMPORAL_ROTATION_GATE_DEG:.1f}"
                    )
                    if gate_info["fallback_reason"]:
                        print(f"[world][{cube_name}] temporal_gate_reason={gate_info['fallback_reason']}")
                    if rejected_debug:
                        rejected_parts = [
                            (
                                f"{item['camera_name']}/tag{item['tag_id']}/pnp{item['pnp_candidate_index']}"
                                f":dt={item['trans_delta_m']:.3f}m,dr={item['rot_delta_deg']:.1f}deg"
                            )
                            for item in rejected_debug[:12]
                        ]
                        print(f"[world][{cube_name}] temporal_rejected={rejected_parts}")

                fused = average_pose_candidates(gated_candidates)
                if fused is None:
                    if frame_idx % PRINT_EVERY_N_FRAMES == 0:
                        print(
                            f"[world][{cube_name}] fused is None: "
                            f"{explain_fused_none_reason(gated_candidates)}"
                        )
                    cube_handle.visible = False
                    continue

                world_R_cube, world_t_cube = fused
                if frame_idx % PRINT_EVERY_N_FRAMES == 0:
                    print(
                        f"[world][{cube_name}] fused_t_m="
                        f"({world_t_cube[0]:.3f}, {world_t_cube[1]:.3f}, {world_t_cube[2]:.3f})"
                    )
                last_world_cube_pose_by_name[cube_name] = {
                    "rot_mat": world_R_cube,
                    "tvec": world_t_cube,
                }
                cube_handle.wxyz = rotation_matrix_to_wxyz(world_R_cube)
                cube_handle.position = (
                    float(world_t_cube[0]),
                    float(world_t_cube[1]),
                    float(world_t_cube[2]),
                )
                cube_handle.visible = True
            add_timing_ms(timing_ms, "world_fusion", time.perf_counter() - t0)

            t0 = time.perf_counter()
            key = cv2.waitKey(1)
            add_timing_ms(timing_ms, "wait_key", time.perf_counter() - t0)
            add_timing_ms(timing_ms, "frame_total", time.perf_counter() - frame_start_time)
            frame_total_ms = timing_ms.get("frame_total", 0.0)
            if frame_total_ms > 1e-6:
                proc_fps = 1000.0 / frame_total_ms
                if processing_fps_ema is None:
                    processing_fps_ema = proc_fps
                else:
                    alpha = float(PROCESSING_FPS_EMA_ALPHA)
                    processing_fps_ema = (1.0 - alpha) * processing_fps_ema + alpha * proc_fps

            if PRINT_TIMING and frame_idx % PRINT_EVERY_N_FRAMES == 0:
                summary_keys = ["get_frames", "world_fusion", "wait_key", "frame_total"]
                proc_fps_str = ""
                if frame_total_ms > 1e-6:
                    proc_fps_str = f" proc_fps={1000.0 / frame_total_ms:.2f} ema_fps={processing_fps_ema:.2f}"
                print(f"[timing][frame={frame_idx}] {format_timing_ms(timing_ms, summary_keys)}{proc_fps_str}")
                for camera_name in opened_names:
                    camera_keys = [
                        f"{camera_name}.undistort",
                        f"{camera_name}.shared_detect",
                        f"{camera_name}.process_frame",
                        f"{camera_name}.draw_cube",
                        f"{camera_name}.draw_tags",
                        f"{camera_name}.ippe",
                        f"{camera_name}.tag_to_cube",
                        f"{camera_name}.overlay",
                        f"{camera_name}.total",
                    ]
                    camera_line = format_timing_ms(timing_ms, camera_keys)
                    if camera_line:
                        print(f"[timing][frame={frame_idx}][{camera_name}] {camera_line}")

            if key == 27 or key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        camera_manager.release_all()
        cv2.destroyAllWindows()
        print("[INFO] Finished.")


if __name__ == "__main__":
    main()
