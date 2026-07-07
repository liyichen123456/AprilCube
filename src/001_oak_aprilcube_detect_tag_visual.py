# OpenCV 相机系
# +x: image right
# +y: image down
# +z: camera forward
from __future__ import annotations

import argparse
import os
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
from recorder_oak_cam import OAK1WCameraManager, list_oak_devices  # noqa: E402

# ============================================================
# User macros
# ============================================================

PRINT_AVAILABLE_DEVICES = True

# Use the device name you used before.
# If you want l_wrist, add it here and add intrinsics below.
CAMERA_TO_DEVICE: dict[str, str] = {
    "r_wrist": "3.10.4.3",
    # "l_wrist": "3.10.4.4.1",
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


def build_viser_servers(
    detectors: dict[str, Any],
    camera_names: list[str],
) -> dict[str, Any]:
    """Start one aprilcube viser server per camera/detector."""
    viser_servers: dict[str, Any] = {}

    if not ENABLE_VISER:
        return viser_servers

    for idx, camera_name in enumerate(camera_names):
        detector = detectors[camera_name]
        port = int(VISER_BASE_PORT) + idx
        try:
            viser_servers[camera_name] = detector.build_viser(port=port)
            print(f"[INFO] AprilCube viser for {camera_name}: http://0.0.0.0:{port}")
        except Exception as exc:
            print(
                f"[WARNING] Failed to start AprilCube viser for {camera_name}: "
                f"{type(exc).__name__}: {exc}"
            )

    return viser_servers


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


def build_tag_to_cube_transform(
    detector: Any,
    tag_id: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the rigid transform X_cube = R * X_tag + t for one tag."""
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
    return rot_cube_tag, center_cube


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
    """Detect AprilTags with the native detector and estimate per-tag pose."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    camera_params = k_to_camera_params(detector.camera_matrix)
    tag_size_m = float(detector.config.tag_size_mm) / 1000.0
    tags = native_detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=tag_size_m,
    )
    return [tag for tag in tags if int(tag.tag_id) in detector.valid_ids]


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
    detector: Any,
    native_detector: Detector,
    image: np.ndarray,
) -> dict[str, Any]:
    """Estimate native AprilTag poses first, then infer cube pose from them."""
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

    first_pass_candidates: list[dict[str, Any]] = []
    tag_meas: list[dict[str, Any]] = []
    for tag in native_tags:
        tag_id = int(tag.tag_id)
        pose = tag_pose_from_native_detection(tag)
        if pose is None:
            continue
        rot_mat, tvec, reproj = pose
        corners_2d = np.asarray(tag.corners, dtype=np.float64).reshape(4, 2)
        tag_meas.append(
            {
                "tag_id": tag_id,
                "corners_2d": corners_2d,
                "rot_mat": rot_mat,
                "tvec": tvec,
                "reproj_error": reproj,
            }
        )
        cube_pose = cube_pose_from_tag_pose(detector, tag_id, rot_mat, tvec)
        if cube_pose is None:
            continue
        cube_rot_mat, cube_tvec = cube_pose
        first_pass_candidates.append(
            {
                "tag_id": tag_id,
                "rot_mat": cube_rot_mat,
                "tvec": cube_tvec,
                "reproj_error": reproj,
            }
        )

    preliminary_cube = fuse_cube_pose_candidates(first_pass_candidates)
    preliminary_center = None
    if preliminary_cube is not None:
        preliminary_center = np.asarray(preliminary_cube[1], dtype=np.float64).reshape(3)

    chosen_candidates: list[dict[str, Any]] = []
    inward_count = 0
    invalid_count = 0
    for meas in tag_meas:
        tag_id = meas["tag_id"]
        inward_ok = None
        if preliminary_center is not None:
            inward_ok = tag_z_points_to_cube_interior(
                meas["rot_mat"],
                meas["tvec"],
                preliminary_center,
            )
            if inward_ok:
                inward_count += 1
            else:
                invalid_count += 1

        result["tag_pose_by_id"][tag_id] = {
            "rot_mat": meas["rot_mat"],
            "tvec": meas["tvec"],
            "reproj_error": meas["reproj_error"],
            "z_inward": inward_ok,
        }

        if inward_ok is False:
            continue

        cube_pose = cube_pose_from_tag_pose(detector, tag_id, meas["rot_mat"], meas["tvec"])
        if cube_pose is None:
            continue
        cube_rot_mat, cube_tvec = cube_pose
        chosen_candidates.append(
            {
                "tag_id": tag_id,
                "rot_mat": cube_rot_mat,
                "tvec": cube_tvec,
                "reproj_error": meas["reproj_error"],
            }
        )

    result["tag_z_inward_count"] = inward_count
    result["tag_z_invalid_count"] = invalid_count

    if not chosen_candidates:
        chosen_candidates = first_pass_candidates
    final_cube = fuse_cube_pose_candidates(chosen_candidates)
    if final_cube is None:
        return detector._store_latest(result, image)

    cube_rot_mat, cube_tvec = final_cube
    cube_rvec, _ = cv2.Rodrigues(cube_rot_mat)
    result["success"] = True
    result["rvec"] = cube_rvec
    result["tvec"] = cube_tvec
    result["n_inliers"] = len(chosen_candidates) * 4
    result["reproj_error"] = compute_reprojection_error(
        detector=detector,
        detections=detections,
        cube_rvec=cube_rvec,
        cube_tvec=cube_tvec,
    )

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
    native_family = apriltag_family_from_dict_name(detectors[active_camera_names[0]].config.dict_name)
    native_tag_detector = Detector(
        families=native_family,
        quad_decimate=1.0,
    )
    print(f"[INFO] Native AprilTag detector family: {native_family}")
    _viser_servers = build_viser_servers(
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

        while True:
            frame_idx += 1

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

                result = process_frame_from_tag_poses(
                    detector=detector,
                    native_detector=native_tag_detector,
                    image=detect_frame,
                )

                try:
                    vis = detector.draw_result(detect_frame.copy(), result)
                except Exception as exc:
                    print(f"[WARNING] draw_result failed for {camera_name}: {type(exc).__name__}: {exc}")
                    vis = detect_frame.copy()

                vis = draw_detected_tag_visuals(
                    img=vis,
                    detector=detector,
                    result=result,
                )

                status = result_to_text(camera_name, result)
                log_frame_handedness(
                    camera_name=camera_name,
                    frame_idx=frame_idx,
                    result=result,
                )

                if frame_idx % PRINT_EVERY_N_FRAMES == 0:
                    print(status)

                panel_lines = [
                    status,
                    make_handedness_overlay_text(result),
                    f"detect_size={DETECT_IMG_SIZE}, isp_scale={ISP_SCALE}, fps={FPS}",
                    "press q or ESC to quit",
                ]
                vis = draw_text_panel(vis, panel_lines)

                cv2.imshow(f"{WINDOW_PREFIX}: {camera_name}", vis)

            key = cv2.waitKey(1)
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
