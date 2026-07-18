from __future__ import annotations

import copy
import importlib
import importlib.util
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import viser
from PIL import Image

THIS_FILE = Path(__file__).resolve()
DEMO_008_PATH = THIS_FILE.parent / "008_cv2_naive_aprilcube_detect.py"
ASSETS_DIR = THIS_FILE.parent.parent / "assets"
OBJ_MESH_SCALE = 0.001

# ============================================================
# User macros
# ============================================================

PKL_PATH = Path(
    "/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/recordings/"
    "008_raw_frames_20260715_000555.pkl"
)
# Camera names, intrinsics, cube cfgs, and undistortion settings come from the PKL header.
PINHOLE_UNDISTORT_ALPHA = 0.0
ADAPTIVE_CLAHE_DETECTION = True
FAST_DETECTOR = True
ENABLE_RUNTIME_POSE_FILTER = False
SHARED_TAG_DETECTION = False
ENABLE_TEMPORAL_POSTPROCESS = True
RECOMPUTE_POSE = True
PRECOMPUTE_ONLY = False

VISER_HOST = "0.0.0.0"
VISER_PORT = 8092
VISER_MAX_IMAGE_WIDTH = 960  # 0 keeps the original image size.
VISER_JPEG_QUALITY = 85

POSE_CACHE_FORMAT = "aprilcube_008_pose_cache_v1"
POSE_CACHE_FORMAT_020_MULTISTAGE = "aprilcube_020_multistage_008_pose_v1"
POSE_CACHE_FORMAT_023_DEEPTAG_008 = "aprilcube_023_deeptag_008_pose_v1"
OFFLINE_POS_FRAME_FIELD = "offline_pos"
OFFLINE_POS_CACHE_KEY_FIELD = "offline_pos_cache_key"
LEGACY_INLINE_POSE_FRAME_FIELD = "offline_pose_frame"
LEGACY_INLINE_POSE_CACHE_KEY_FIELD = "offline_pose_cache_key"
IMAGE_RECOVERY_VERSION = 10
SINGLE_TAG_FACE_FRAME_SOLVER_VERSION = 1
SINGLE_TAG_FACE_FRAME_MAX_REPROJ_PX = 5.0
SINGLE_TAG_FACE_FRAME_REPROJ_TIE_PX = 1.0
SINGLE_TAG_FACE_FRAME_LM_REFINE = True
SINGLE_TAG_CONTINUITY_GATE_ENABLED = True
SINGLE_TAG_CONTINUITY_MAX_ROTATION_DEG = 45.0
SINGLE_TAG_CONTINUITY_MIN_FACE_OBSERVATIONS = 2
SINGLE_TAG_CONTINUITY_MAX_OBSERVATION_GAP = 8
SINGLE_TAG_CONTINUITY_VERSION = 2
TEMPORAL_OUTLIER_GATE_ENABLED = True
TEMPORAL_OUTLIER_MAX_NEIGHBOR_GAP_FRAMES = 6
TEMPORAL_OUTLIER_NEIGHBOR_MAX_ROTATION_DEG = 35.0
TEMPORAL_OUTLIER_NEIGHBOR_MAX_TRANSLATION_MM = 35.0
TEMPORAL_OUTLIER_MIN_ROTATION_JUMP_DEG = 90.0
TEMPORAL_OUTLIER_MIN_TRANSLATION_JUMP_MM = 70.0
TEMPORAL_OUTLIER_VERSION = 1
TEMPORAL_FILL_MAX_GAP_FRAMES = 30
TEMPORAL_FILL_MAX_ROTATION_DEG = 45.0
TEMPORAL_FILL_VERSION = 5
TEMPORAL_SMOOTHING_ENABLED = True
TEMPORAL_SMOOTHING_WINDOW_RADIUS = 2
TEMPORAL_SMOOTHING_SIGMA_FRAMES = 1.2
TEMPORAL_SMOOTHING_MAX_ROTATION_DEG = 15.0
TEMPORAL_SMOOTHING_MAX_DISPLAY_REPROJ_PX = 12.0
TEMPORAL_SMOOTHING_MAX_REPROJ_RATIO = 2.5
TEMPORAL_SMOOTHING_VERSION = 5
TEMPORAL_ROTATION_JUMP_LIMIT_ENABLED = True
TEMPORAL_ROTATION_JUMP_MAX_DEG = 20.0
TEMPORAL_ROTATION_JUMP_HOLD_DEG = 60.0
TEMPORAL_ROTATION_JUMP_LIMIT_VERSION = 2


def install_numpy_pickle_compat() -> None:
    """Allow NumPy 2.x pickles to load in NumPy 1.x environments."""
    try:
        numpy_core = importlib.import_module("numpy.core")
    except Exception:
        return

    sys.modules.setdefault("numpy._core", numpy_core)
    for module_name in (
        "multiarray",
        "numeric",
        "numerictypes",
        "overrides",
        "fromnumeric",
        "shape_base",
        "umath",
        "_multiarray_umath",
    ):
        try:
            module = importlib.import_module(f"numpy.core.{module_name}")
        except Exception:
            continue
        sys.modules.setdefault(f"numpy._core.{module_name}", module)


install_numpy_pickle_compat()


def load_demo008_module() -> Any:
    spec = importlib.util.spec_from_file_location("aprilcube_demo008", DEMO_008_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load 008 module from {DEMO_008_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_pkl_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser().resolve()
    if path.is_dir():
        candidates = sorted(path.glob("008_raw_frames_*.pkl"))
        if not candidates:
            raise FileNotFoundError(f"No 008_raw_frames_*.pkl found in {path}")
        return candidates[-1].resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PKL file does not exist: {path}")
    return path


def print_index_progress(done_bytes: int, total_bytes: int, *, force_newline: bool = False) -> None:
    width = 36
    ratio = 1.0 if total_bytes <= 0 else min(max(done_bytes / total_bytes, 0.0), 1.0)
    filled = int(round(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(
        f"\r[INFO] Indexing PKL [{bar}] {done_bytes / (1024**2):.1f}/"
        f"{total_bytes / (1024**2):.1f} MiB"
    )
    sys.stdout.flush()
    if force_newline:
        sys.stdout.write("\n")
        sys.stdout.flush()


def build_frame_index(
    path: Path,
) -> tuple[
    dict[str, Any] | None,
    list[int],
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    header: dict[str, Any] | None = None
    footer: dict[str, Any] | None = None
    pose_cache_record: dict[str, Any] | None = None
    offline_pos_cache_key: dict[str, Any] | None = None
    offline_pos_cache: list[dict[str, Any] | None] = []
    offline_pos_cache_complete = True
    offline_pos_cache_keys_match = True
    legacy_inline_pose_cache_key: dict[str, Any] | None = None
    legacy_inline_pose_cache: list[dict[str, Any] | None] = []
    legacy_inline_pose_cache_complete = True
    legacy_inline_pose_cache_keys_match = True
    frame_offsets: list[int] = []
    file_size = path.stat().st_size
    last_print = time.monotonic()

    with path.open("rb") as f:
        while True:
            offset = f.tell()
            try:
                record = pickle.load(f)
            except EOFError:
                break

            if not isinstance(record, dict):
                continue
            record_type = record.get("type", None)
            if record_type == "header":
                header = record
            elif record_type == "frame":
                frame_offsets.append(offset)
                offline_pos = record.get(OFFLINE_POS_FRAME_FIELD, None)
                offline_pos_key = record.get(OFFLINE_POS_CACHE_KEY_FIELD, None)
                if isinstance(offline_pos, dict) and isinstance(offline_pos_key, dict):
                    offline_pos_cache.append(offline_pos)
                    if offline_pos_cache_key is None:
                        offline_pos_cache_key = offline_pos_key
                    elif offline_pos_cache_key != offline_pos_key:
                        offline_pos_cache_keys_match = False
                else:
                    offline_pos_cache.append(None)
                    offline_pos_cache_complete = False

                legacy_inline_pose = record.get(LEGACY_INLINE_POSE_FRAME_FIELD, None)
                legacy_inline_key = record.get(
                    LEGACY_INLINE_POSE_CACHE_KEY_FIELD,
                    None,
                )
                if isinstance(legacy_inline_pose, dict) and isinstance(legacy_inline_key, dict):
                    legacy_inline_pose_cache.append(legacy_inline_pose)
                    if legacy_inline_pose_cache_key is None:
                        legacy_inline_pose_cache_key = legacy_inline_key
                    elif legacy_inline_pose_cache_key != legacy_inline_key:
                        legacy_inline_pose_cache_keys_match = False
                else:
                    legacy_inline_pose_cache.append(None)
                    legacy_inline_pose_cache_complete = False
            elif record_type == "footer":
                footer = record
            elif record_type == "pose_cache":
                pose_cache_record = record

            now = time.monotonic()
            if now - last_print > 0.5:
                print_index_progress(f.tell(), file_size)
                last_print = now

    print_index_progress(file_size, file_size, force_newline=True)
    offline_pos_cache_record = None
    if (
        offline_pos_cache_complete
        and offline_pos_cache_keys_match
        and offline_pos_cache_key is not None
        and len(offline_pos_cache) == len(frame_offsets)
    ):
        offline_pos_cache_record = {
            "type": "pose_cache",
            "format": POSE_CACHE_FORMAT,
            "key": offline_pos_cache_key,
            "pose_cache": offline_pos_cache,
        }
    legacy_inline_pose_cache_record = None
    if (
        legacy_inline_pose_cache_complete
        and legacy_inline_pose_cache_keys_match
        and legacy_inline_pose_cache_key is not None
        and len(legacy_inline_pose_cache) == len(frame_offsets)
    ):
        legacy_inline_pose_cache_record = {
            "type": "pose_cache",
            "format": POSE_CACHE_FORMAT,
            "key": legacy_inline_pose_cache_key,
            "pose_cache": legacy_inline_pose_cache,
        }
    return (
        header,
        frame_offsets,
        footer,
        pose_cache_record,
        offline_pos_cache_record,
        legacy_inline_pose_cache_record,
    )


def load_frame_at_offset(path: Path, offset: int) -> dict[str, Any]:
    with path.open("rb") as f:
        f.seek(offset)
        record = pickle.load(f)
    if not isinstance(record, dict) or record.get("type") != "frame":
        raise ValueError(f"Offset {offset} does not point to a frame record.")
    image = record.get("image_bgr", None)
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Frame at offset {offset} has no ndarray image_bgr.")
    return record


def resize_for_display(image: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return image
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / max(w, 1)
    target_size = (
        max(1, int(round(w * scale))),
        max(1, int(round(h * scale))),
    )
    pil_image = Image.fromarray(image)
    return np.asarray(pil_image.resize(target_size, Image.Resampling.BILINEAR))


def bgr_to_rgb_for_viser(image_bgr: np.ndarray, max_width: int) -> np.ndarray:
    image = resize_for_display(image_bgr, max_width)
    return image[..., ::-1]


def record_summary(record: dict[str, Any], frame_idx: int, total_frames: int) -> str:
    camera_name = record.get("camera_name", "unknown")
    loop_idx = record.get("loop_frame_idx", "unknown")
    capture_ts = record.get("capture_timestamp", None)
    shape = record.get("shape", None)
    dtype = record.get("dtype", None)
    return (
        f"frame {frame_idx + 1}/{total_frames} | camera={camera_name} | "
        f"loop_idx={loop_idx} | shape={shape} | dtype={dtype} | "
        f"capture_ts={capture_ts}"
    )


def print_pose_progress(done: int, total: int, *, force_newline: bool = False) -> None:
    width = 36
    ratio = 1.0 if total <= 0 else min(max(done / total, 0.0), 1.0)
    filled = int(round(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r[INFO] Estimating poses [{bar}] {done}/{total} frames")
    sys.stdout.flush()
    if force_newline:
        sys.stdout.write("\n")
        sys.stdout.flush()


def face_name_for_tag(face_id_sets: dict[str, set[int]], tag_id: int) -> str | None:
    for face_name, tag_ids in face_id_sets.items():
        if int(tag_id) in tag_ids:
            return str(face_name)
    return None


def face_normal_for_name(face_name: str | None) -> np.ndarray | None:
    if face_name is None:
        return None
    from aprilcube.generate import FACE_DEFS

    for definition in FACE_DEFS:
        if definition[0] != face_name:
            continue
        normal = np.zeros(3, dtype=np.float64)
        normal[int(definition[1])] = float(definition[2])
        return normal
    return None


def build_tag_face_frame(
    cube_corners: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    corners = np.asarray(cube_corners, dtype=np.float64).reshape(4, 3)
    center_cube = np.mean(corners, axis=0)
    x_axis = corners[1] - corners[0]
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1e-9:
        return None
    x_axis /= x_norm

    y_axis = corners[3] - corners[0]
    y_axis -= x_axis * float(np.dot(y_axis, x_axis))
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm < 1e-9:
        return None
    y_axis /= y_norm

    z_axis = np.cross(x_axis, y_axis)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm < 1e-9:
        return None
    z_axis /= z_norm

    rotation_cube_from_face = np.column_stack((x_axis, y_axis, z_axis))
    corners_face = (corners - center_cube) @ rotation_cube_from_face
    corners_face[:, 2] = 0.0
    return center_cube, rotation_cube_from_face, corners_face


def pose_continuity_cost(
    rvec: np.ndarray,
    tvec: np.ndarray,
    prev_rvec: np.ndarray | None,
    prev_tvec: np.ndarray | None,
) -> float:
    if prev_rvec is None or prev_tvec is None:
        return 0.0
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    prev_rotation, _ = cv2.Rodrigues(
        np.asarray(prev_rvec, dtype=np.float64).reshape(3, 1)
    )
    angle = np.arccos(
        np.clip((np.trace(prev_rotation.T @ rotation) - 1.0) / 2.0, -1.0, 1.0)
    )
    translation_delta = float(
        np.linalg.norm(
            np.asarray(tvec, dtype=np.float64).reshape(3)
            - np.asarray(prev_tvec, dtype=np.float64).reshape(3)
        )
    )
    return min(translation_delta / 20.0, 20.0) + min(
        float(np.degrees(angle)) / 10.0,
        20.0,
    )


def estimate_single_tag_cube_pose_via_face_frame(
    detections: list[tuple[int, np.ndarray]],
    tag_corner_map: dict[int, np.ndarray],
    face_id_sets: dict[str, set[int]],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    prev_rvec: np.ndarray | None = None,
    prev_tvec: np.ndarray | None = None,
    allow_corner_rotations: bool = False,
) -> tuple[
    bool,
    np.ndarray | None,
    np.ndarray | None,
    float,
    np.ndarray | None,
    dict[str, Any],
]:
    """Solve a tag-local planar pose, then compose it back to the cube frame."""
    candidates: list[dict[str, Any]] = []
    raw_candidate_count = 0
    for tag_id, corners_2d_value in detections:
        base_cube_corners = tag_corner_map.get(int(tag_id))
        if base_cube_corners is None:
            continue
        corners_2d = np.asarray(corners_2d_value, dtype=np.float64).reshape(4, 2)
        face_name = face_name_for_tag(face_id_sets, int(tag_id))
        outward_normal_cube = face_normal_for_name(face_name)
        rotations = range(4) if allow_corner_rotations else range(1)

        for corner_rotation in rotations:
            cube_corners = np.roll(
                np.asarray(base_cube_corners, dtype=np.float64).reshape(4, 3),
                -int(corner_rotation),
                axis=0,
            )
            face_frame = build_tag_face_frame(cube_corners)
            if face_frame is None:
                continue
            center_cube, rotation_cube_from_face, corners_face = face_frame
            try:
                retval, face_rvecs, face_tvecs, _errors = cv2.solvePnPGeneric(
                    corners_face,
                    corners_2d,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE,
                )
            except cv2.error:
                retval, face_rvecs, face_tvecs = 0, (), ()
            if not retval:
                continue

            for face_rvec_value, face_tvec_value in zip(face_rvecs, face_tvecs):
                raw_candidate_count += 1
                face_rvec = np.asarray(face_rvec_value, dtype=np.float64).reshape(3, 1)
                face_tvec = np.asarray(face_tvec_value, dtype=np.float64).reshape(3, 1)
                lm_refined = False
                if SINGLE_TAG_FACE_FRAME_LM_REFINE:
                    try:
                        face_rvec, face_tvec = cv2.solvePnPRefineLM(
                            corners_face,
                            corners_2d,
                            camera_matrix,
                            dist_coeffs,
                            face_rvec,
                            face_tvec,
                        )
                        lm_refined = True
                    except cv2.error:
                        pass

                rotation_camera_from_face, _ = cv2.Rodrigues(face_rvec)
                rotation_camera_from_cube = (
                    rotation_camera_from_face @ rotation_cube_from_face.T
                )
                translation_camera_from_cube = (
                    face_tvec
                    - rotation_camera_from_cube @ center_cube.reshape(3, 1)
                )
                if float(translation_camera_from_cube[2, 0]) <= 0.0:
                    continue
                if (
                    outward_normal_cube is not None
                    and float((rotation_camera_from_cube @ outward_normal_cube)[2]) > 0.0
                ):
                    continue

                cube_rvec, _ = cv2.Rodrigues(rotation_camera_from_cube)
                projected, _ = cv2.projectPoints(
                    cube_corners,
                    cube_rvec,
                    translation_camera_from_cube,
                    camera_matrix,
                    dist_coeffs,
                )
                reproj_error = float(
                    np.mean(
                        np.linalg.norm(
                            corners_2d - projected.reshape(-1, 2),
                            axis=1,
                        )
                    )
                )
                if (
                    not np.isfinite(reproj_error)
                    or reproj_error > SINGLE_TAG_FACE_FRAME_MAX_REPROJ_PX
                ):
                    continue
                candidates.append(
                    {
                        "rvec": cube_rvec,
                        "tvec": translation_camera_from_cube,
                        "reproj_error": reproj_error,
                        "continuity_cost": pose_continuity_cost(
                            cube_rvec,
                            translation_camera_from_cube,
                            prev_rvec,
                            prev_tvec,
                        ),
                        "tag_id": int(tag_id),
                        "face_name": face_name,
                        "corner_rotation": int(corner_rotation),
                        "lm_refined": lm_refined,
                    }
                )

    if not candidates:
        return False, None, None, float("inf"), None, {
            "single_tag_face_frame_pose": True,
            "single_tag_face_frame_raw_candidate_count": raw_candidate_count,
        }

    best_reproj_error = min(float(candidate["reproj_error"]) for candidate in candidates)
    near_best = [
        candidate
        for candidate in candidates
        if float(candidate["reproj_error"])
        <= best_reproj_error + SINGLE_TAG_FACE_FRAME_REPROJ_TIE_PX
    ]
    if prev_rvec is not None and prev_tvec is not None:
        selected = min(
            near_best,
            key=lambda candidate: (
                float(candidate["continuity_cost"]),
                float(candidate["reproj_error"]),
            ),
        )
    else:
        selected = min(near_best, key=lambda candidate: float(candidate["reproj_error"]))

    inliers = np.arange(4, dtype=np.int32).reshape(-1, 1)
    return (
        True,
        selected["rvec"],
        selected["tvec"],
        float(selected["reproj_error"]),
        inliers,
        {
            "single_tag_id": int(selected["tag_id"]),
            "single_tag_face": selected["face_name"],
            "single_tag_candidate_count": len(candidates),
            "single_tag_corner_rotation_deg": int(selected["corner_rotation"]) * 90,
            "single_tag_face_frame_pose": True,
            "single_tag_face_frame_lm_refined": bool(selected["lm_refined"]),
            "single_tag_face_frame_raw_candidate_count": raw_candidate_count,
            "single_tag_face_frame_near_best_count": len(near_best),
            "single_tag_face_frame_best_reproj_error": best_reproj_error,
        },
    )


def process_detections_with_face_frame_solver(
    detector: Any,
    image: np.ndarray,
    tag_detections: list[tuple[int, np.ndarray]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Use the 009 replay face-frame solver without changing other entry points."""
    from aprilcube import detect as detect_mod

    original_solver = detect_mod.estimate_single_tag_cube_pose
    detect_mod.estimate_single_tag_cube_pose = estimate_single_tag_cube_pose_via_face_frame
    try:
        return detector.process_detections(image, tag_detections, **kwargs)
    finally:
        detect_mod.estimate_single_tag_cube_pose = original_solver


def result_copy_for_replay(result: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key in (
        "success",
        "rvec",
        "tvec",
        "T",
        "reproj_error",
        "n_tags",
        "n_inliers",
        "detections",
        "tag_ids",
        "visible_faces",
        "predicted",
        "direct_all_point_pnp",
        "single_tag_cfg_pose",
        "single_tag_id",
        "single_tag_face",
        "single_tag_candidate_count",
        "single_tag_corner_rotation_deg",
        "single_tag_face_frame_pose",
        "single_tag_face_frame_lm_refined",
        "single_tag_face_frame_raw_candidate_count",
        "single_tag_face_frame_near_best_count",
        "single_tag_face_frame_best_reproj_error",
        "failure_reason",
        "temporal_filled",
        "temporal_fill_source",
        "temporal_fill_alpha",
        "temporal_smoothed",
        "temporal_smoothing_source_count",
    ):
        value = result.get(key, None)
        if key == "detections":
            copied[key] = [
                (int(tag_id), np.asarray(corners, dtype=np.float64).copy())
                for tag_id, corners in (value or [])
            ]
        elif key == "visible_faces":
            copied[key] = set(value or [])
        elif isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = value
    return copied


def clone_optional_array(value: np.ndarray | None) -> np.ndarray | None:
    return None if value is None else value.copy()


def snapshot_detector_tracking_state(detector: Any) -> dict[str, Any]:
    return {
        "prev_rvec": clone_optional_array(detector.prev_rvec),
        "prev_tvec": clone_optional_array(detector.prev_tvec),
        "pose_filter": copy.deepcopy(detector.pose_filter),
        "_prev_gray": clone_optional_array(detector._prev_gray),
        "_prev_corners_2d": clone_optional_array(detector._prev_corners_2d),
        "_prev_corners_3d": clone_optional_array(detector._prev_corners_3d),
    }


def restore_detector_tracking_state(detector: Any, state: dict[str, Any]) -> None:
    detector.prev_rvec = clone_optional_array(state["prev_rvec"])
    detector.prev_tvec = clone_optional_array(state["prev_tvec"])
    detector.pose_filter = copy.deepcopy(state["pose_filter"])
    detector._prev_gray = clone_optional_array(state["_prev_gray"])
    detector._prev_corners_2d = clone_optional_array(state["_prev_corners_2d"])
    detector._prev_corners_3d = clone_optional_array(state["_prev_corners_3d"])


def is_measured_pose(result: dict[str, Any]) -> bool:
    return bool(result.get("success", False)) and not bool(result.get("predicted", False))


def rvec_to_quat(rvec: np.ndarray) -> np.ndarray:
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(r))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = r / angle
    half = angle * 0.5
    return np.array(
        [np.cos(half), *(np.sin(half) * axis)],
        dtype=np.float64,
    )


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    return q / max(float(np.linalg.norm(q)), 1e-12)


def align_quat_to_reference(quat: np.ndarray, reference: np.ndarray) -> np.ndarray:
    q = normalize_quat(quat)
    ref = normalize_quat(reference)
    if float(np.dot(ref, q)) < 0.0:
        return -q
    return q


def quat_short_arc_angle_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    q0n = normalize_quat(q0)
    q1n = align_quat_to_reference(q1, q0n)
    dot = abs(float(np.dot(q0n, q1n)))
    return float(np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))


def quat_to_rvec(quat: np.ndarray) -> np.ndarray:
    q = normalize_quat(quat)
    if q[0] < 0:
        q = -q
    sin_half = float(np.linalg.norm(q[1:]))
    if sin_half < 1e-12:
        return np.zeros((3, 1), dtype=np.float64)
    angle = 2.0 * np.arctan2(sin_half, q[0])
    axis = q[1:] / sin_half
    return (angle * axis).reshape(3, 1)


def slerp_quat(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return q / max(float(np.linalg.norm(q)), 1e-12)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


def limit_quat_rotation(
    source: np.ndarray,
    target: np.ndarray,
    max_rotation_deg: float,
) -> tuple[np.ndarray, float, bool]:
    source_q = normalize_quat(source)
    target_q = align_quat_to_reference(target, source_q)
    angle_deg = quat_short_arc_angle_deg(source_q, target_q)
    if angle_deg <= max_rotation_deg:
        return target_q, angle_deg, False
    alpha = max(float(max_rotation_deg), 0.0) / max(angle_deg, 1e-12)
    return normalize_quat(slerp_quat(source_q, target_q, alpha)), angle_deg, True


def pose_transform_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3], _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


class ReplayPoseEstimator:
    def __init__(
        self,
        demo008: Any,
        *,
        active_camera_names: list[str],
        cube_paths: list[Path],
        use_undistort: bool,
        adaptive_clahe: bool,
        shared_tag_detection: bool,
        enable_filter: bool,
        fast: bool,
    ) -> None:
        self.demo008 = demo008
        self.active_camera_names = active_camera_names
        self.cube_paths = cube_paths
        self.use_undistort = use_undistort
        self.adaptive_clahe = adaptive_clahe
        self.shared_tag_detection = shared_tag_detection

        self.calib_by_camera = {
            name: demo008.load_intrinsics_yaml(demo008.CAMERA_TO_INTRINSICS_YAML[name])
            for name in active_camera_names
        }
        self.image_size = demo008.resolve_common_image_size(self.calib_by_camera)
        self.detect_img_size = self.image_size
        self.detection_camera_matrix_by_camera = {
            camera_name: demo008.compute_detection_camera_matrix(
                calib,
                self.detect_img_size,
                undistort_before_detection=use_undistort,
            )
            for camera_name, calib in self.calib_by_camera.items()
        }
        self.undistort_maps_by_camera = {
            camera_name: demo008.create_undistort_maps(
                calib,
                self.detect_img_size,
                self.detection_camera_matrix_by_camera[camera_name],
            )
            if use_undistort
            else None
            for camera_name, calib in self.calib_by_camera.items()
        }

        self.detector_entries_by_camera: dict[str, list[dict[str, Any]]] = {
            name: [] for name in active_camera_names
        }
        self.detector_by_camera_cube: dict[tuple[str, str], Any] = {}
        for cube_path in cube_paths:
            cube_name = cube_path.name if cube_path.is_dir() else cube_path.parent.name
            for camera_name in active_camera_names:
                detector = demo008.create_detector_for_camera(
                    cube_path,
                    camera_name,
                    self.calib_by_camera,
                    self.detection_camera_matrix_by_camera,
                    enable_filter=enable_filter,
                    fast=fast,
                    undistort_before_detection=use_undistort,
                )
                self.detector_entries_by_camera[camera_name].append(
                    {"cube_name": cube_name, "detector": detector}
                )
                self.detector_by_camera_cube[(camera_name, cube_name)] = detector

    def prepare_detect_frame(self, image_bgr: np.ndarray, camera_name: str) -> np.ndarray:
        frame = image_bgr
        h, w = frame.shape[:2]
        if (w, h) != self.detect_img_size:
            frame = cv2.resize(frame, self.detect_img_size, interpolation=cv2.INTER_AREA)
        if self.use_undistort:
            frame = self.demo008.undistort_frame(
                frame,
                self.undistort_maps_by_camera[camera_name],
            )
        return frame

    @staticmethod
    def timestamp_for_record(
        record: dict[str, Any],
        frame_idx: int,
        metadata: dict[str, Any],
    ) -> float:
        capture_ts = record.get("capture_timestamp", None)
        if isinstance(capture_ts, (int, float)):
            return float(capture_ts)
        fps = metadata.get("fps", 30) if isinstance(metadata, dict) else 30
        try:
            fps_f = float(fps)
        except (TypeError, ValueError):
            fps_f = 30.0
        return frame_idx / max(fps_f, 1.0)

    def estimate_record(
        self,
        record: dict[str, Any],
        frame_idx: int,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        camera_name = str(record.get("camera_name", self.active_camera_names[0]))
        image_bgr = record["image_bgr"]
        if camera_name not in self.detector_entries_by_camera:
            return {
                "camera_name": camera_name,
                "status_lines": [f"[{camera_name}] skipped: no detector config"],
                "cube_results": [],
                "decoded_tag_count": 0,
            }

        detector_entries = self.detector_entries_by_camera[camera_name]
        detect_frame = self.prepare_detect_frame(image_bgr, camera_name)
        timestamp = self.timestamp_for_record(record, frame_idx, metadata)

        shared_tags = None
        decoded_tag_ids: set[int] = set()
        adaptive_new_tag_ids: set[int] = set()
        if self.shared_tag_detection:
            shared_tags = detector_entries[0]["detector"].detect_tags(
                detect_frame,
                adaptive_clahe=self.adaptive_clahe,
            )
            decoded_tag_ids.update(int(tag_id) for tag_id, _ in shared_tags["detections"])

        status_lines = [
            f"[{camera_name}] 008 replay cubes={len(detector_entries)} "
            f"detect_size={self.detect_img_size} "
            f"tag_detect_mode={'shared' if self.shared_tag_detection else 'per_cube'} "
            f"adaptive_clahe={self.adaptive_clahe}"
        ]
        cube_results: list[dict[str, Any]] = []
        for entry in detector_entries:
            cube_name = entry["cube_name"]
            detector = entry["detector"]
            if self.shared_tag_detection:
                cube_tags = shared_tags
                assert cube_tags is not None
                result = process_detections_with_face_frame_solver(
                    detector,
                    detect_frame,
                    cube_tags["detections"],
                    rejected_quads=cube_tags["rejected"],
                    gray=cube_tags["gray"],
                    enhanced=cube_tags["enhanced"],
                    timestamp=timestamp,
                )
                recovery_mode = "shared_adaptive" if self.adaptive_clahe else "shared_base"
            else:
                result, cube_tags, recovery_mode = self.estimate_cube_with_clahe_recovery(
                    detector,
                    detect_frame,
                    timestamp,
                )

            decoded_tag_ids.update(int(tag_id) for tag_id, _ in cube_tags["detections"])
            for attempt in cube_tags.get("adaptive_attempts", []):
                if attempt.get("base", False):
                    continue
                adaptive_new_tag_ids.update(int(tag_id) for tag_id in attempt.get("new_ids", []))

            result = result_copy_for_replay(result)
            result["decoded_tags_this_cube_pass"] = len(cube_tags["detections"])
            result["clahe_recovery_mode"] = recovery_mode
            result_text = self.demo008.result_to_text(camera_name, cube_name, result)
            if result.get("single_tag_face_frame_pose", False):
                result_text += " face_frame_ippe"
            status_lines.append(result_text)
            cube_results.append(
                {
                    "cube_name": cube_name,
                    "result": result,
                }
            )

        status_lines[0] += (
            f" decoded_tags={len(decoded_tag_ids)} "
            f"clahe_extra_tags={len(adaptive_new_tag_ids)}"
        )

        return {
            "camera_name": camera_name,
            "status_lines": status_lines,
            "cube_results": cube_results,
            "decoded_tag_count": len(decoded_tag_ids),
            "adaptive_clahe": self.adaptive_clahe,
            "adaptive_new_tags": len(adaptive_new_tag_ids),
            "tag_detect_mode": "shared" if self.shared_tag_detection else "per_cube",
        }

    def estimate_cube_with_clahe_recovery(
        self,
        detector: Any,
        detect_frame: np.ndarray,
        timestamp: float,
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        state_before = snapshot_detector_tracking_state(detector)
        base_tags = detector.detect_tags(detect_frame, adaptive_clahe=False)
        base_result = process_detections_with_face_frame_solver(
            detector,
            detect_frame,
            base_tags["detections"],
            rejected_quads=base_tags["rejected"],
            gray=base_tags["gray"],
            enhanced=base_tags["enhanced"],
            timestamp=timestamp,
        )
        base_state_after = snapshot_detector_tracking_state(detector)
        if is_measured_pose(base_result) or not self.adaptive_clahe:
            return base_result, base_tags, "base"

        from aprilcube import detect as detect_mod

        variants = getattr(
            detect_mod,
            "_adaptive_image_enhancement_variants",
            (),
        )
        if not variants:
            variants = tuple(
                {
                    "name": f"adaptive clip={float(clip_limit):.1f} tile={tuple(tile_grid_size)}",
                    "clahe": (float(clip_limit), tuple(tile_grid_size)),
                }
                for clip_limit, tile_grid_size in getattr(
                    detect_mod,
                    "_adaptive_clahe_variants",
                    (),
                )
            )

        for variant in variants:
            restore_detector_tracking_state(detector, state_before)
            candidate_tags = detector.detect_tags(
                detect_frame,
                adaptive_clahe=True,
                enhancement_variants=(dict(variant),),
            )
            candidate_result = process_detections_with_face_frame_solver(
                detector,
                detect_frame,
                candidate_tags["detections"],
                rejected_quads=candidate_tags["rejected"],
                gray=candidate_tags["gray"],
                enhanced=candidate_tags["enhanced"],
                timestamp=timestamp,
            )
            if is_measured_pose(candidate_result):
                return (
                    candidate_result,
                    candidate_tags,
                    str(variant.get("name", "adaptive enhancement")),
                )

        restore_detector_tracking_state(detector, base_state_after)
        return base_result, base_tags, "base_failed_enhancement_rejected"

    @staticmethod
    def detector_input_mode_for_pose_frame(pose_frame: dict[str, Any]) -> str:
        for cube in pose_frame.get("cube_results", []):
            result = cube.get("result", {})
            mode = str(result.get("clahe_recovery_mode", "base"))
            if result.get("success", False) and mode != "temporal_fill":
                return mode
        for cube in pose_frame.get("cube_results", []):
            result = cube.get("result", {})
            mode = str(result.get("clahe_recovery_mode", "base"))
            if mode != "temporal_fill":
                return mode
        return "base"

    @staticmethod
    def detector_input_gray_for_mode(gray: np.ndarray, mode: str) -> np.ndarray:
        from aprilcube import detect as detect_mod

        if mode in ("base", "shared_base", "base_failed_enhancement_rejected", "temporal_fill"):
            return detect_mod._preprocess(gray)

        variants = getattr(detect_mod, "_adaptive_image_enhancement_variants", ())
        for variant in variants:
            if str(variant.get("name", "")) == mode:
                return detect_mod._preprocess_enhancement_variant(gray, dict(variant))

        return detect_mod._preprocess(gray)

    def draw_detector_input_frame(self, record: dict[str, Any], pose_frame: dict[str, Any]) -> np.ndarray:
        camera_name = pose_frame["camera_name"]
        detect_frame = self.prepare_detect_frame(record["image_bgr"], camera_name)
        gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY) if len(detect_frame.shape) == 3 else detect_frame
        mode = self.detector_input_mode_for_pose_frame(pose_frame)
        enhanced = self.detector_input_gray_for_mode(gray, mode)
        vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        mode_text = f"Detector input: {mode}"
        if mode == "temporal_fill":
            mode_text += " (pose came from temporal fill; showing base detector input)"
        cv2.putText(
            vis,
            mode_text,
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return vis

    def draw_pose_frame(self, record: dict[str, Any], pose_frame: dict[str, Any]) -> np.ndarray:
        camera_name = pose_frame["camera_name"]
        detect_frame = self.prepare_detect_frame(record["image_bgr"], camera_name)
        vis = self.demo008.make_tag_detection_vis_image(detect_frame)
        return self.draw_pose_over_base_frame(vis, camera_name, pose_frame)

    def draw_detector_input_pose_frame(
        self,
        record: dict[str, Any],
        pose_frame: dict[str, Any],
    ) -> np.ndarray:
        camera_name = pose_frame["camera_name"]
        vis = self.draw_detector_input_frame(record, pose_frame)
        return self.draw_pose_over_base_frame(vis, camera_name, pose_frame)

    def draw_pose_over_base_frame(
        self,
        base_frame: np.ndarray,
        camera_name: str,
        pose_frame: dict[str, Any],
    ) -> np.ndarray:
        vis = base_frame.copy()
        for cube in pose_frame["cube_results"]:
            detector = self.detector_by_camera_cube[(camera_name, cube["cube_name"])]
            result = self.normalize_result_for_draw(cube.get("result", {}))
            vis = detector.draw_result(vis, result)
        vis = self.demo008.draw_text_panel(vis, pose_frame.get("status_lines", []))
        temporal_cubes = self.temporal_filled_cube_names(pose_frame)
        if temporal_cubes:
            return self.draw_red_alert_box(
                vis,
                "TEMPORAL FILLED CUBE POSE",
                ", ".join(temporal_cubes[:3]) + (f", +{len(temporal_cubes) - 3}" if len(temporal_cubes) > 3 else ""),
            )
        if not self.pose_frame_has_all_cube_pose(camera_name, pose_frame):
            return self.draw_red_alert_box(vis, "INCOMPLETE CUBE POSE")
        return vis

    @staticmethod
    def normalize_result_for_draw(result: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(result or {})
        normalized.setdefault("success", False)
        normalized.setdefault("detections", [])
        normalized.setdefault("visible_faces", set())
        normalized.setdefault("n_tags", 0)
        normalized.setdefault("reproj_error", float("inf"))
        for key in ("rvec", "tvec"):
            if normalized.get(key) is not None and not isinstance(normalized[key], np.ndarray):
                normalized[key] = np.asarray(normalized[key], dtype=np.float64).reshape(3, 1)
        if normalized.get("T") is not None and not isinstance(normalized["T"], np.ndarray):
            normalized["T"] = np.asarray(normalized["T"], dtype=np.float64).reshape(4, 4)
        return normalized

    @staticmethod
    def draw_red_alert_box(
        vis: np.ndarray,
        label: str,
        detail: str | None = None,
    ) -> np.ndarray:
        h, w = vis.shape[:2]
        border = max(6, min(w, h) // 120)
        cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 0, 255), border)
        cv2.putText(
            vis,
            label,
            (20, max(42, border + 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        if detail:
            cv2.putText(
                vis,
                detail,
                (20, max(84, border + 68)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return vis

    @staticmethod
    def temporal_filled_cube_names(pose_frame: dict[str, Any]) -> list[str]:
        return [
            str(cube.get("cube_name", "unknown"))
            for cube in pose_frame.get("cube_results", [])
            if bool(cube.get("result", {}).get("temporal_filled", False))
        ]

    def pose_frame_has_all_cube_pose(self, camera_name: str, pose_frame: dict[str, Any]) -> bool:
        expected_cubes = {
            entry["cube_name"] for entry in self.detector_entries_by_camera.get(camera_name, [])
        }
        result_cubes = {cube["cube_name"] for cube in pose_frame.get("cube_results", [])}
        if result_cubes != expected_cubes:
            return False
        return all(
            bool(cube.get("result", {}).get("success", False))
            for cube in pose_frame.get("cube_results", [])
        )

    def missing_cube_names_for_pose_frame(
        self,
        camera_name: str,
        pose_frame: dict[str, Any],
    ) -> list[str]:
        expected_cubes = {
            entry["cube_name"] for entry in self.detector_entries_by_camera.get(camera_name, [])
        }
        result_by_cube = {
            cube["cube_name"]: cube.get("result", {})
            for cube in pose_frame.get("cube_results", [])
        }
        return [
            cube_name
            for cube_name in sorted(expected_cubes)
            if not bool(result_by_cube.get(cube_name, {}).get("success", False))
        ]

    def draw_undistorted_debug_frame(
        self,
        record: dict[str, Any],
        pose_frame: dict[str, Any],
    ) -> np.ndarray:
        camera_name = pose_frame["camera_name"]
        vis = self.prepare_detect_frame(record["image_bgr"], camera_name).copy()
        missing = self.missing_cube_names_for_pose_frame(camera_name, pose_frame)
        temporal_cubes = self.temporal_filled_cube_names(pose_frame)
        if temporal_cubes:
            return self.draw_red_alert_box(
                vis,
                "TEMPORAL FILLED CUBE POSE",
                ", ".join(temporal_cubes[:3]) + (f", +{len(temporal_cubes) - 3}" if len(temporal_cubes) > 3 else ""),
            )
        if self.pose_frame_has_all_cube_pose(camera_name, pose_frame):
            return vis

        missing_text = ", ".join(missing[:3])
        if len(missing) > 3:
            missing_text += f", +{len(missing) - 3}"
        return self.draw_red_alert_box(
            vis,
            f"MISSING CUBE POSE: {len(missing)}/{len(pose_frame.get('cube_results', []))}",
            missing_text,
        )


def pose_markdown(pose_frame: dict[str, Any]) -> str:
    lines = [
        f"**camera**: `{pose_frame.get('camera_name', 'unknown')}`",
        f"**tag detect mode**: `{pose_frame.get('tag_detect_mode', 'unknown')}`",
        f"**decoded tags**: `{pose_frame.get('decoded_tag_count', 0)}`",
        f"**adaptive CLAHE**: `{pose_frame.get('adaptive_clahe', False)}`",
        f"**CLAHE extra tags**: `{pose_frame.get('adaptive_new_tags', 0)}`",
        "",
    ]
    for cube in pose_frame.get("cube_results", []):
        result = cube["result"]
        cube_name = cube["cube_name"]
        if not result.get("success", False):
            lines.append(
                f"- `{cube_name}`: no pose, tags={int(result.get('n_tags', 0))}, "
                f"mode={result.get('clahe_recovery_mode', 'unknown')}"
            )
            continue
        tvec = np.asarray(result["tvec"], dtype=np.float64).reshape(-1)
        faces = sorted(list(result.get("visible_faces", set())))
        predicted = " predicted" if result.get("predicted", False) else ""
        temporal_fill = ""
        if result.get("temporal_filled", False):
            source = result.get("temporal_fill_source", {})
            temporal_fill = (
                f", temporal_fill={source.get('before_frame', '?')}"
                f"->{source.get('after_frame', '?')}"
            )
        temporal_smooth = ""
        if result.get("temporal_smoothed", False):
            temporal_smooth = (
                f", smooth_n={int(result.get('temporal_smoothing_source_count', 0))}"
            )
        single_tag_cfg = ""
        if result.get("single_tag_cfg_pose", False):
            single_tag_cfg = (
                f", single_tag_cfg_pose=id{result.get('single_tag_id', '?')}"
                f"/{result.get('single_tag_face', '?')}"
            )
        lines.append(
            f"- `{cube_name}`: t=({tvec[0]:.1f}, {tvec[1]:.1f}, {tvec[2]:.1f}) mm, "
            f"reproj={float(result.get('reproj_error', float('inf'))):.2f}px, "
            f"tags={int(result.get('n_tags', 0))}, faces={faces}{predicted}, "
            f"mode={result.get('clahe_recovery_mode', 'unknown')}"
            f"{single_tag_cfg}{temporal_fill}{temporal_smooth}"
        )
    return "\n".join(lines)


def cube_scene_node_name(cube_name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in cube_name)
    return f"/world_thumb_web_camera/{safe}"


def load_obj_mesh_for_viser(
    obj_name: str,
    color: tuple[int, int, int],
) -> tuple[Any, Path]:
    import trimesh

    obj_path = ASSETS_DIR / f"{obj_name}.obj"
    if not obj_path.is_file():
        raise FileNotFoundError(f"OBJ mesh not found: {obj_path}")

    loaded = trimesh.load(obj_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded

    rgba = np.asarray([color[0], color[1], color[2], 210], dtype=np.uint8)
    mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))
    return mesh, obj_path


def cube_pose_tracks(pose_cache: list[dict[str, Any]]) -> dict[str, list[tuple[int, np.ndarray]]]:
    tracks: dict[str, list[tuple[int, np.ndarray]]] = {}
    for frame_idx, pose_frame in enumerate(pose_cache):
        for cube in pose_frame.get("cube_results", []):
            cube_name = str(cube.get("cube_name", ""))
            result = cube.get("result", {})
            if not cube_name or not bool(result.get("success", False)):
                continue
            tvec = result.get("tvec", None)
            if tvec is None:
                continue
            tracks.setdefault(cube_name, []).append(
                (frame_idx, np.asarray(tvec, dtype=np.float64).reshape(3) / 1000.0)
            )
    return tracks


def make_track_segments(track: list[tuple[int, np.ndarray]]) -> np.ndarray:
    if len(track) < 2:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.asarray(
        [[track[i][1], track[i + 1][1]] for i in range(len(track) - 1)],
        dtype=np.float32,
    )


def create_3d_scene_handles(
    server: viser.ViserServer,
    estimator: ReplayPoseEstimator,
    pose_cache: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    server.scene.set_up_direction("-y")
    server.scene.world_axes.visible = False
    server.scene.add_frame(
        "/world_thumb_web_camera",
        axes_length=0.06,
        axes_radius=0.002,
        origin_radius=0.004,
    )

    grid_lines = []
    grid_half = 0.20
    grid_step = 0.05
    n = int(round(grid_half / grid_step))
    for i in range(-n, n + 1):
        x = i * grid_step
        z = i * grid_step
        grid_lines.append([[x, 0.0, -grid_half], [x, 0.0, grid_half]])
        grid_lines.append([[-grid_half, 0.0, z], [grid_half, 0.0, z]])
    grid_handle = server.scene.add_line_segments(
        "/world_thumb_web_camera/xz_grid_y0",
        points=np.asarray(grid_lines, dtype=np.float32),
        colors=(80, 80, 80),
        line_width=1.0,
        visible=False,
    )
    aspect = estimator.detect_img_size[0] / max(estimator.detect_img_size[1], 1)
    first_camera = estimator.active_camera_names[0]
    camera_matrix = estimator.detection_camera_matrix_by_camera[first_camera]
    fy = float(camera_matrix[1, 1])
    fov_y = float(2.0 * np.arctan(estimator.detect_img_size[1] / max(2.0 * fy, 1e-12)))
    camera_frustum = server.scene.add_camera_frustum(
        "/world_thumb_web_camera/frustum",
        fov=fov_y,
        aspect=aspect,
        scale=0.08,
        line_width=1.5,
        color=(180, 180, 180),
        visible=True,
    )

    palette = [
        (255, 150, 40),
        (80, 180, 255),
        (120, 220, 120),
        (220, 120, 255),
        (255, 220, 80),
        (180, 180, 180),
    ]
    handles: dict[str, dict[str, Any]] = {
        "__scene__": {
            "grid": grid_handle,
            "camera_frustum": camera_frustum,
        }
    }
    tracks = cube_pose_tracks(pose_cache)
    obj_mesh_cache: dict[str, tuple[Any, Path]] = {}
    cfg_to_obj = getattr(estimator.demo008, "CUBE_CFG_NAME_TO_OBJ_NAME", {})
    color_idx = 0
    for camera_name in estimator.active_camera_names:
        for entry in estimator.detector_entries_by_camera.get(camera_name, []):
            cube_name = entry["cube_name"]
            detector = entry["detector"]
            node = cube_scene_node_name(cube_name)
            safe = node.rsplit("/", 1)[-1]
            track_node = f"/world_thumb_web_camera/pose_tracks/{safe}"
            dims_m = tuple(float(v) / 1000.0 for v in detector.config.box_dims)
            color = palette[color_idx % len(palette)]
            color_idx += 1
            frame_handle = server.scene.add_frame(
                node,
                axes_length=max(dims_m) * 0.8,
                axes_radius=max(dims_m) * 0.035,
                origin_radius=0.0,
                visible=False,
            )
            box_handle = server.scene.add_box(
                f"{node}/box",
                dimensions=dims_m,
                color=color,
                opacity=0.35,
                side="double",
                visible=False,
            )
            obj_mesh_handle = None
            obj_name = str(cfg_to_obj.get(cube_name, ""))
            if obj_name:
                try:
                    if obj_name not in obj_mesh_cache:
                        obj_mesh_cache[obj_name] = load_obj_mesh_for_viser(obj_name, color)
                    mesh, obj_path = obj_mesh_cache[obj_name]
                    obj_mesh_handle = server.scene.add_mesh_trimesh(
                        f"{node}/finger_obj",
                        mesh.copy(),
                        scale=OBJ_MESH_SCALE,
                        visible=False,
                        cast_shadow=False,
                        receive_shadow=False,
                    )
                    print(f"[INFO] 3D OBJ mesh: {cube_name} -> {obj_name} path={obj_path}")
                except Exception as exc:
                    print(
                        f"[WARNING] Failed to add 3D OBJ mesh for {cube_name} -> {obj_name}: "
                        f"{type(exc).__name__}: {exc}"
                    )
            track = tracks.get(cube_name, [])
            track_segments = make_track_segments(track)
            trajectory_handle = server.scene.add_line_segments(
                f"{track_node}/trajectory",
                points=track_segments,
                colors=np.asarray(color, dtype=np.uint8),
                line_width=2.0,
                visible=track_segments.shape[0] > 0,
            )
            if track:
                sample_points = np.asarray([pos for _idx, pos in track], dtype=np.float32)
                sample_colors = np.tile(np.asarray(color, dtype=np.uint8), (len(track), 1))
            else:
                sample_points = np.zeros((0, 3), dtype=np.float32)
                sample_colors = np.zeros((0, 3), dtype=np.uint8)
            samples_handle = server.scene.add_point_cloud(
                f"{track_node}/trajectory_samples",
                points=sample_points,
                colors=sample_colors,
                point_size=0.004,
                point_shape="circle",
                visible=sample_points.shape[0] > 0,
            )
            marker_radius = max(max(dims_m) * 0.08, 0.0015)
            current_handle = server.scene.add_icosphere(
                f"{track_node}/current_position",
                radius=marker_radius,
                color=(255, 255, 255),
                subdivisions=2,
                visible=False,
            )
            start_handle = None
            end_handle = None
            if track:
                _start_idx, start_pos = track[0]
                _end_idx, end_pos = track[-1]
                start_handle = server.scene.add_icosphere(
                    f"{track_node}/track_start",
                    radius=marker_radius,
                    color=(40, 220, 80),
                    subdivisions=2,
                    position=start_pos,
                    visible=True,
                )
                end_handle = server.scene.add_icosphere(
                    f"{track_node}/track_end",
                    radius=marker_radius,
                    color=(240, 80, 80),
                    subdivisions=2,
                    position=end_pos,
                    visible=True,
                )
            handles[cube_name] = {
                "frame": frame_handle,
                "box": box_handle,
                "obj_mesh": obj_mesh_handle,
                "base_color": color,
                "trajectory": trajectory_handle,
                "samples": samples_handle,
                "current": current_handle,
                "start": start_handle,
                "end": end_handle,
            }
    return handles


def update_3d_scene(
    scene_handles: dict[str, dict[str, Any]],
    pose_frame: dict[str, Any],
) -> None:
    seen: set[str] = set()
    for cube in pose_frame.get("cube_results", []):
        cube_name = str(cube.get("cube_name", ""))
        if cube_name.startswith("__"):
            continue
        result = cube.get("result", {})
        handles = scene_handles.get(cube_name)
        if handles is None:
            continue
        seen.add(cube_name)
        success = bool(result.get("success", False))
        handles["pose_visible"] = success
        for key in ("frame", "box", "obj_mesh", "current"):
            handle = handles.get(key)
            if handle is not None:
                handle.visible = success
        if not success:
            continue

        rvec = np.asarray(result["rvec"], dtype=np.float64).reshape(3, 1)
        tvec_m = np.asarray(result["tvec"], dtype=np.float64).reshape(3) / 1000.0
        wxyz = rvec_to_quat(rvec)
        handles["frame"].position = tvec_m
        handles["frame"].wxyz = wxyz
        handles["current"].position = tvec_m
        handles["box"].color = (
            (255, 0, 0)
            if bool(result.get("temporal_filled", False))
            else handles["base_color"]
        )

    for cube_name, handles in scene_handles.items():
        if cube_name.startswith("__"):
            continue
        if cube_name in seen:
            continue
        handles["pose_visible"] = False
        for key in ("frame", "box", "obj_mesh", "current"):
            handle = handles.get(key)
            if handle is not None:
                handle.visible = False


def set_optional_visible(handle: Any, visible: bool) -> None:
    if handle is not None:
        handle.visible = bool(visible)


def apply_3d_visibility(
    scene_handles: dict[str, dict[str, Any]],
    *,
    show_box: bool,
    show_obj: bool,
    show_axes: bool,
    show_trajectory: bool,
    show_samples: bool,
    show_endpoints: bool,
    show_grid: bool,
    show_camera: bool,
) -> None:
    scene = scene_handles.get("__scene__", {})
    set_optional_visible(scene.get("grid"), show_grid)
    set_optional_visible(scene.get("camera_frustum"), show_camera)
    for cube_name, handles in scene_handles.items():
        if cube_name.startswith("__"):
            continue
        pose_visible = bool(handles.get("pose_visible", False))
        if "box" in handles:
            handles["box"].visible = bool(show_box) and pose_visible
        if "obj_mesh" in handles and handles["obj_mesh"] is not None:
            handles["obj_mesh"].visible = bool(show_obj) and pose_visible
        if "frame" in handles:
            handles["frame"].visible = bool(show_axes) and pose_visible
        set_optional_visible(handles.get("current"), show_trajectory and pose_visible)
        set_optional_visible(handles.get("trajectory"), show_trajectory)
        set_optional_visible(handles.get("samples"), show_samples)
        set_optional_visible(handles.get("start"), show_endpoints)
        set_optional_visible(handles.get("end"), show_endpoints)


def precompute_pose_cache(
    pkl_path: Path,
    frame_offsets: list[int],
    metadata: dict[str, Any],
    estimator: ReplayPoseEstimator,
) -> list[dict[str, Any]]:
    pose_cache: list[dict[str, Any]] = []
    total = len(frame_offsets)
    last_print = time.monotonic()
    for idx, offset in enumerate(frame_offsets):
        record = load_frame_at_offset(pkl_path, offset)
        pose_cache.append(estimator.estimate_record(record, idx, metadata))
        now = time.monotonic()
        if now - last_print > 0.5:
            print_pose_progress(idx + 1, total)
            last_print = now
    print_pose_progress(total, total, force_newline=True)
    return pose_cache


def cube_result_by_name(pose_frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        cube["cube_name"]: cube
        for cube in pose_frame.get("cube_results", [])
        if isinstance(cube, dict) and "cube_name" in cube
    }


def is_temporal_anchor(result: dict[str, Any]) -> bool:
    return (
        bool(result.get("success", False))
        and not bool(result.get("predicted", False))
        and not bool(result.get("temporal_filled", False))
    )


def interpolate_pose_result(
    before_idx: int,
    before_result: dict[str, Any],
    after_idx: int,
    after_result: dict[str, Any],
    target_idx: int,
) -> dict[str, Any]:
    alpha = (target_idx - before_idx) / max(after_idx - before_idx, 1)
    before_t = np.asarray(before_result["tvec"], dtype=np.float64).reshape(3, 1)
    after_t = np.asarray(after_result["tvec"], dtype=np.float64).reshape(3, 1)
    tvec = (1.0 - alpha) * before_t + alpha * after_t

    q0 = rvec_to_quat(before_result["rvec"])
    q1 = rvec_to_quat(after_result["rvec"])
    anchor_rotation_deg = quat_short_arc_angle_deg(q0, q1)
    q_interp = slerp_quat(q0, q1, alpha)
    rotation_mode = (
        "slerp_large_anchor_rotation"
        if anchor_rotation_deg > TEMPORAL_FILL_MAX_ROTATION_DEG
        else "slerp_short_arc"
    )
    rvec = quat_to_rvec(q_interp)

    before_faces = set(before_result.get("visible_faces", set()) or [])
    after_faces = set(after_result.get("visible_faces", set()) or [])
    before_reproj = float(before_result.get("reproj_error", 0.0))
    after_reproj = float(after_result.get("reproj_error", 0.0))

    return {
        "success": True,
        "rvec": rvec,
        "tvec": tvec,
        "T": pose_transform_from_rvec_tvec(rvec, tvec),
        "reproj_error": (1.0 - alpha) * before_reproj + alpha * after_reproj,
        "n_tags": 0,
        "n_inliers": 0,
        "detections": [],
        "tag_ids": [],
        "visible_faces": before_faces | after_faces,
        "predicted": False,
        "temporal_filled": True,
        "temporal_fill_source": {
            "before_frame": int(before_idx),
            "after_frame": int(after_idx),
        },
        "temporal_fill_alpha": float(alpha),
        "temporal_fill_rotation_deg": float(anchor_rotation_deg),
        "temporal_fill_rotation_mode": rotation_mode,
        "decoded_tags_this_cube_pass": 0,
        "clahe_recovery_mode": "temporal_fill",
    }


def rebuild_pose_frame_status_lines(
    estimator: ReplayPoseEstimator,
    pose_frame: dict[str, Any],
) -> None:
    camera_name = pose_frame.get("camera_name", estimator.active_camera_names[0])
    cube_results = pose_frame.get("cube_results", [])
    header = (
        f"[{camera_name}] 008 replay cubes={len(cube_results)} "
        f"detect_size={estimator.detect_img_size} "
        f"tag_detect_mode={pose_frame.get('tag_detect_mode', 'unknown')} "
        f"adaptive_clahe={pose_frame.get('adaptive_clahe', False)} "
        f"decoded_tags={pose_frame.get('decoded_tag_count', 0)} "
        f"clahe_extra_tags={pose_frame.get('adaptive_new_tags', 0)} "
        f"continuity_rejected={pose_frame.get('continuity_rejected_count', 0)} "
        f"temporal_outlier_rejected={pose_frame.get('temporal_outlier_rejected_count', 0)} "
        f"temporal_filled={pose_frame.get('temporal_filled_count', 0)} "
        f"rotation_limited={pose_frame.get('temporal_rotation_jump_limited_count', 0)} "
        f"smoothing={pose_frame.get('temporal_smoothing_enabled', False)}"
    )
    lines = [header]
    for cube in cube_results:
        lines.append(
            estimator.demo008.result_to_text(
                str(camera_name),
                str(cube["cube_name"]),
                cube.get("result", {}),
            )
        )
    pose_frame["status_lines"] = lines


def is_postprocess_temporal_result(result: dict[str, Any]) -> bool:
    return (
        bool(result.get("temporal_filled", False))
        or result.get("clahe_recovery_mode") == "temporal_fill"
    )


def reject_pose_result_for_temporal_fill(
    result: dict[str, Any],
    reason: str,
    *,
    previous_face: str | None = None,
    rotation_jump_deg: float | None = None,
    previous_frame: int | None = None,
    next_frame: int | None = None,
    next_rotation_jump_deg: float | None = None,
    previous_translation_jump_mm: float | None = None,
    next_translation_jump_mm: float | None = None,
) -> dict[str, Any]:
    rejected = copy.deepcopy(result)
    rejected["success"] = False
    rejected["rvec"] = None
    rejected["tvec"] = None
    rejected["T"] = None
    rejected["reproj_error"] = float("inf")
    rejected["continuity_rejected"] = True
    rejected["continuity_reject_reason"] = reason
    if previous_face is not None:
        rejected["continuity_previous_face"] = previous_face
    if rotation_jump_deg is not None:
        rejected["continuity_rotation_jump_deg"] = float(rotation_jump_deg)
    if previous_frame is not None:
        rejected["continuity_previous_frame"] = int(previous_frame)
    if next_frame is not None:
        rejected["continuity_next_frame"] = int(next_frame)
    if next_rotation_jump_deg is not None:
        rejected["continuity_next_rotation_jump_deg"] = float(next_rotation_jump_deg)
    if previous_translation_jump_mm is not None:
        rejected["continuity_previous_translation_jump_mm"] = float(previous_translation_jump_mm)
    if next_translation_jump_mm is not None:
        rejected["continuity_next_translation_jump_mm"] = float(next_translation_jump_mm)
    return rejected


def single_face_name(result: dict[str, Any]) -> str | None:
    faces = sorted(list(result.get("visible_faces", set()) or []))
    if len(faces) != 1:
        return None
    return str(faces[0])


def reset_temporal_postprocess_outputs(
    pose_cache: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    reset = copy.deepcopy(pose_cache)
    reset_count = 0
    for pose_frame in reset:
        pose_frame["temporal_filled_count"] = 0
        pose_frame["continuity_rejected_count"] = 0
        pose_frame["temporal_outlier_rejected_count"] = 0
        pose_frame["temporal_rotation_jump_limited_count"] = 0
        for cube in pose_frame.get("cube_results", []):
            result = cube.get("result", {})
            for key in (
                "temporal_smoothed",
                "temporal_smoothing_source_count",
                "temporal_smoothing_window_radius",
                "temporal_smoothing_rotation_delta_deg",
                "temporal_smoothing_rotation_limited",
                "temporal_rotation_jump_limited",
                "temporal_rotation_jump_held",
                "temporal_rotation_jump_original_delta_deg",
                "temporal_rotation_jump_max_deg",
                "temporal_rotation_jump_hold_deg",
            ):
                result.pop(key, None)
            if is_postprocess_temporal_result(result):
                cube["result"] = reject_pose_result_for_temporal_fill(
                    result,
                    "reset_previous_temporal_fill",
                )
                reset_count += 1
    return reset, reset_count


def gate_single_tag_pose_cache(
    pose_cache: list[dict[str, Any]],
    estimator: ReplayPoseEstimator,
    *,
    max_rotation_deg: float = SINGLE_TAG_CONTINUITY_MAX_ROTATION_DEG,
) -> tuple[list[dict[str, Any]], int]:
    if not SINGLE_TAG_CONTINUITY_GATE_ENABLED:
        return pose_cache, 0

    gated = copy.deepcopy(pose_cache)
    rejected_count = 0

    for camera_name in estimator.active_camera_names:
        cube_names = [
            entry["cube_name"]
            for entry in estimator.detector_entries_by_camera.get(camera_name, [])
        ]
        frame_indices = [
            idx
            for idx, pose_frame in enumerate(gated)
            if pose_frame.get("camera_name") == camera_name
        ]
        for cube_name in cube_names:
            single_face_observations: list[tuple[int, str, dict[str, Any]]] = []
            for idx in frame_indices:
                pose_frame = gated[idx]
                cube = cube_result_by_name(pose_frame).get(cube_name)
                if cube is None:
                    continue
                result = cube.get("result", {})
                n_tags = int(result.get("n_tags", 0) or 0)
                face = single_face_name(result)
                if (
                    bool(result.get("success", False))
                    and not bool(result.get("predicted", False))
                    and not is_postprocess_temporal_result(result)
                    and n_tags == 1
                    and face is not None
                ):
                    single_face_observations.append((idx, face, result))

            trusted_single_tag_indices: set[int] = set()
            current_run: list[tuple[int, str, dict[str, Any]]] = []

            def commit_run(run: list[tuple[int, str, dict[str, Any]]]) -> None:
                if len(run) < int(SINGLE_TAG_CONTINUITY_MIN_FACE_OBSERVATIONS):
                    return
                trusted_single_tag_indices.update(idx for idx, _face, _result in run)

            for observation in single_face_observations:
                idx, face, result = observation
                if not current_run:
                    current_run = [observation]
                    continue
                prev_idx, prev_face, _prev_result = current_run[-1]
                if (
                    face == prev_face
                    and idx - prev_idx <= int(SINGLE_TAG_CONTINUITY_MAX_OBSERVATION_GAP)
                ):
                    current_run.append(observation)
                    continue
                commit_run(current_run)
                current_run = [observation]
            commit_run(current_run)

            last_trusted_by_face: dict[str, dict[str, Any]] = {}
            for idx in frame_indices:
                pose_frame = gated[idx]
                cube = cube_result_by_name(pose_frame).get(cube_name)
                if cube is None:
                    continue
                result = cube.get("result", {})
                if not bool(result.get("success", False)):
                    continue
                if bool(result.get("predicted", False)):
                    continue
                if is_postprocess_temporal_result(result):
                    continue

                n_tags = int(result.get("n_tags", 0) or 0)
                face = single_face_name(result)
                reject_reason: str | None = None
                rotation_jump_deg: float | None = None
                previous_face: str | None = None

                if n_tags <= 0:
                    reject_reason = "no_decoded_tag_success_pose"
                elif n_tags == 1:
                    if idx not in trusted_single_tag_indices:
                        reject_reason = "single_tag_isolated_face_observation"
                    elif face is not None and face in last_trusted_by_face:
                        previous_face = face
                        rotation_jump_deg = quat_short_arc_angle_deg(
                            rvec_to_quat(last_trusted_by_face[face]["rvec"]),
                            rvec_to_quat(result["rvec"]),
                        )
                        if rotation_jump_deg > max_rotation_deg:
                            reject_reason = "single_tag_same_face_rotation_jump"

                if reject_reason is not None:
                    cube["result"] = reject_pose_result_for_temporal_fill(
                        result,
                        reject_reason,
                        previous_face=previous_face,
                        rotation_jump_deg=rotation_jump_deg,
                    )
                    pose_frame["continuity_rejected_count"] = int(
                        pose_frame.get("continuity_rejected_count", 0)
                    ) + 1
                    rejected_count += 1
                    continue

                if n_tags > 0 and face is not None:
                    last_trusted_by_face[face] = result

    for pose_frame in gated:
        pose_frame["single_tag_continuity_gate_enabled"] = bool(
            SINGLE_TAG_CONTINUITY_GATE_ENABLED
        )
        rebuild_pose_frame_status_lines(estimator, pose_frame)

    return gated, rejected_count


def pose_translation_jump_mm(a: dict[str, Any], b: dict[str, Any]) -> float:
    at = np.asarray(a["tvec"], dtype=np.float64).reshape(3)
    bt = np.asarray(b["tvec"], dtype=np.float64).reshape(3)
    return float(np.linalg.norm(at - bt))


def pose_rotation_jump_deg(a: dict[str, Any], b: dict[str, Any]) -> float:
    return quat_short_arc_angle_deg(rvec_to_quat(a["rvec"]), rvec_to_quat(b["rvec"]))


def gate_temporal_outlier_pose_cache(
    pose_cache: list[dict[str, Any]],
    estimator: ReplayPoseEstimator,
) -> tuple[list[dict[str, Any]], int]:
    if not TEMPORAL_OUTLIER_GATE_ENABLED:
        return pose_cache, 0

    gated = copy.deepcopy(pose_cache)
    rejected_count = 0

    for camera_name in estimator.active_camera_names:
        cube_names = [
            entry["cube_name"]
            for entry in estimator.detector_entries_by_camera.get(camera_name, [])
        ]
        frame_indices = [
            idx
            for idx, pose_frame in enumerate(gated)
            if pose_frame.get("camera_name") == camera_name
        ]
        for cube_name in cube_names:
            anchors: list[tuple[int, dict[str, Any]]] = []
            for idx in frame_indices:
                cube = cube_result_by_name(gated[idx]).get(cube_name)
                if cube is None:
                    continue
                result = cube.get("result", {})
                if is_temporal_anchor(result):
                    anchors.append((idx, result))

            if len(anchors) < 3:
                continue

            for anchor_pos in range(1, len(anchors) - 1):
                prev_idx, prev_result = anchors[anchor_pos - 1]
                idx, result = anchors[anchor_pos]
                next_idx, next_result = anchors[anchor_pos + 1]
                if idx - prev_idx > TEMPORAL_OUTLIER_MAX_NEIGHBOR_GAP_FRAMES:
                    continue
                if next_idx - idx > TEMPORAL_OUTLIER_MAX_NEIGHBOR_GAP_FRAMES:
                    continue

                neighbor_rotation_deg = pose_rotation_jump_deg(prev_result, next_result)
                neighbor_translation_mm = pose_translation_jump_mm(prev_result, next_result)
                if neighbor_rotation_deg > TEMPORAL_OUTLIER_NEIGHBOR_MAX_ROTATION_DEG:
                    continue
                if neighbor_translation_mm > TEMPORAL_OUTLIER_NEIGHBOR_MAX_TRANSLATION_MM:
                    continue

                prev_rotation_deg = pose_rotation_jump_deg(prev_result, result)
                next_rotation_deg = pose_rotation_jump_deg(result, next_result)
                prev_translation_mm = pose_translation_jump_mm(prev_result, result)
                next_translation_mm = pose_translation_jump_mm(result, next_result)
                rotation_flip = (
                    prev_rotation_deg >= TEMPORAL_OUTLIER_MIN_ROTATION_JUMP_DEG
                    and next_rotation_deg >= TEMPORAL_OUTLIER_MIN_ROTATION_JUMP_DEG
                )
                translation_spike = (
                    prev_translation_mm >= TEMPORAL_OUTLIER_MIN_TRANSLATION_JUMP_MM
                    and next_translation_mm >= TEMPORAL_OUTLIER_MIN_TRANSLATION_JUMP_MM
                )
                if not (rotation_flip or translation_spike):
                    continue

                pose_frame = gated[idx]
                cube = cube_result_by_name(pose_frame).get(cube_name)
                if cube is None:
                    continue
                cube["result"] = reject_pose_result_for_temporal_fill(
                    result,
                    "temporal_pose_outlier_between_consistent_neighbors",
                    previous_frame=prev_idx,
                    next_frame=next_idx,
                    rotation_jump_deg=prev_rotation_deg,
                    next_rotation_jump_deg=next_rotation_deg,
                    previous_translation_jump_mm=prev_translation_mm,
                    next_translation_jump_mm=next_translation_mm,
                )
                cube["result"]["temporal_outlier_rejected"] = True
                cube["result"]["temporal_outlier_neighbor_rotation_deg"] = float(neighbor_rotation_deg)
                cube["result"]["temporal_outlier_neighbor_translation_mm"] = float(neighbor_translation_mm)
                pose_frame["continuity_rejected_count"] = int(
                    pose_frame.get("continuity_rejected_count", 0)
                ) + 1
                pose_frame["temporal_outlier_rejected_count"] = int(
                    pose_frame.get("temporal_outlier_rejected_count", 0)
                ) + 1
                rejected_count += 1

    for pose_frame in gated:
        pose_frame["temporal_outlier_gate_enabled"] = bool(TEMPORAL_OUTLIER_GATE_ENABLED)
        rebuild_pose_frame_status_lines(estimator, pose_frame)

    return gated, rejected_count


def complete_pose_cache_temporally(
    pose_cache: list[dict[str, Any]],
    estimator: ReplayPoseEstimator,
    *,
    max_gap_frames: int = TEMPORAL_FILL_MAX_GAP_FRAMES,
) -> tuple[list[dict[str, Any]], int]:
    completed = copy.deepcopy(pose_cache)
    filled_count = 0

    for camera_name in estimator.active_camera_names:
        cube_names = [
            entry["cube_name"]
            for entry in estimator.detector_entries_by_camera.get(camera_name, [])
        ]
        frame_indices = [
            idx
            for idx, pose_frame in enumerate(completed)
            if pose_frame.get("camera_name") == camera_name
        ]
        for cube_name in cube_names:
            anchors: list[tuple[int, dict[str, Any]]] = []
            for idx in frame_indices:
                cube = cube_result_by_name(completed[idx]).get(cube_name)
                if cube is None:
                    continue
                result = cube.get("result", {})
                if is_temporal_anchor(result):
                    anchors.append((idx, result))

            for (before_idx, before_result), (after_idx, after_result) in zip(
                anchors,
                anchors[1:],
            ):
                if after_idx - before_idx - 1 <= 0:
                    continue
                if after_idx - before_idx - 1 > max_gap_frames:
                    continue
                for target_idx in range(before_idx + 1, after_idx):
                    pose_frame = completed[target_idx]
                    cube_map = cube_result_by_name(pose_frame)
                    cube = cube_map.get(cube_name)
                    if cube is not None and bool(cube.get("result", {}).get("success", False)):
                        continue
                    filled_result = interpolate_pose_result(
                        before_idx,
                        before_result,
                        after_idx,
                        after_result,
                        target_idx,
                    )
                    old_result = {} if cube is None else cube.get("result", {})
                    if bool(old_result.get("continuity_rejected", False)):
                        filled_result["temporal_fill_replaced_rejection"] = old_result.get(
                            "continuity_reject_reason",
                            "continuity_rejected",
                        )
                    if cube is None:
                        pose_frame.setdefault("cube_results", []).append(
                            {"cube_name": cube_name, "result": filled_result}
                        )
                    else:
                        cube["result"] = filled_result
                    pose_frame["temporal_filled_count"] = int(
                        pose_frame.get("temporal_filled_count", 0)
                    ) + 1
                    filled_count += 1

    for pose_frame in completed:
        pose_frame["temporal_fill_enabled"] = True
        pose_frame["temporal_fill_max_gap_frames"] = int(max_gap_frames)
        rebuild_pose_frame_status_lines(estimator, pose_frame)

    return completed, filled_count


def pose_result_smoothing_weight(result: dict[str, Any], frame_distance: int) -> float:
    sigma = max(float(TEMPORAL_SMOOTHING_SIGMA_FRAMES), 1e-6)
    time_weight = float(np.exp(-0.5 * (float(frame_distance) / sigma) ** 2))
    if bool(result.get("predicted", False)):
        quality_weight = 0.35
    elif bool(result.get("temporal_filled", False)):
        quality_weight = 0.65
    else:
        quality_weight = 1.0

    reproj = result.get("reproj_error", None)
    if reproj is not None and np.isfinite(float(reproj)):
        quality_weight *= 1.0 / (1.0 + max(float(reproj), 0.0) / 5.0)
    return time_weight * quality_weight


def pose_reprojection_errors_for_result(
    result: dict[str, Any],
    detector: Any,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> tuple[float, dict[int, float]] | None:
    detections = result.get("detections", [])
    if not detections:
        return None

    object_chunks = []
    image_chunks = []
    tag_ids = []
    for tag_id, corners_2d in detections:
        corners_3d = detector.tag_corner_map.get(int(tag_id))
        if corners_3d is None:
            continue
        object_chunks.append(np.asarray(corners_3d, dtype=np.float64).reshape(4, 3))
        image_chunks.append(np.asarray(corners_2d, dtype=np.float64).reshape(4, 2))
        tag_ids.append(int(tag_id))
    if not object_chunks:
        return None

    object_points = np.vstack(object_chunks).astype(np.float64)
    image_points = np.vstack(image_chunks).astype(np.float64)
    projected, _ = cv2.projectPoints(
        object_points,
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        detector.camera_matrix,
        detector.dist_coeffs,
    )
    projected = projected.reshape(-1, 2)
    per_tag: dict[int, float] = {}
    for k, tag_id in enumerate(tag_ids):
        start = k * 4
        end = start + 4
        per_tag[tag_id] = float(np.mean(np.linalg.norm(
            image_points[start:end] - projected[start:end],
            axis=1,
        )))
    return float(np.mean(list(per_tag.values()))), per_tag


def weighted_average_quats(
    quats: list[np.ndarray],
    weights: list[float],
    reference: np.ndarray | None = None,
) -> np.ndarray:
    if not quats:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ref = (
        normalize_quat(reference)
        if reference is not None
        else normalize_quat(quats[len(quats) // 2])
    )
    accum = np.zeros(4, dtype=np.float64)
    for quat, weight in zip(quats, weights):
        q = align_quat_to_reference(quat, ref)
        accum += float(weight) * q
    return accum / max(float(np.linalg.norm(accum)), 1e-12)


def smooth_pose_cache_temporally(
    pose_cache: list[dict[str, Any]],
    estimator: ReplayPoseEstimator,
    *,
    window_radius: int = TEMPORAL_SMOOTHING_WINDOW_RADIUS,
) -> tuple[list[dict[str, Any]], int]:
    if window_radius <= 0:
        return pose_cache, 0

    source = pose_cache
    smoothed = copy.deepcopy(pose_cache)
    smoothed_count = 0

    for camera_name in estimator.active_camera_names:
        cube_names = [
            entry["cube_name"]
            for entry in estimator.detector_entries_by_camera.get(camera_name, [])
        ]
        frame_indices = [
            idx
            for idx, pose_frame in enumerate(source)
            if pose_frame.get("camera_name") == camera_name
        ]
        for cube_name in cube_names:
            for target_idx in frame_indices:
                cube = cube_result_by_name(smoothed[target_idx]).get(cube_name)
                if cube is None:
                    continue
                source_cube = cube_result_by_name(source[target_idx]).get(cube_name)
                source_result = {} if source_cube is None else source_cube.get("result", {})
                if not bool(source_result.get("success", False)):
                    continue

                samples: list[tuple[int, dict[str, Any], float]] = []
                for neighbor_idx in frame_indices:
                    distance = abs(neighbor_idx - target_idx)
                    if distance > window_radius:
                        continue
                    neighbor_cube = cube_result_by_name(source[neighbor_idx]).get(cube_name)
                    if neighbor_cube is None:
                        continue
                    neighbor_result = neighbor_cube.get("result", {})
                    if not bool(neighbor_result.get("success", False)):
                        continue
                    weight = pose_result_smoothing_weight(neighbor_result, distance)
                    if weight <= 0.0:
                        continue
                    samples.append((neighbor_idx, neighbor_result, weight))

                if len(samples) <= 1:
                    continue

                weights = np.asarray([sample[2] for sample in samples], dtype=np.float64)
                weights = weights / max(float(np.sum(weights)), 1e-12)
                t_stack = np.stack(
                    [
                        np.asarray(sample[1]["tvec"], dtype=np.float64).reshape(3)
                        for sample in samples
                    ],
                    axis=0,
                )
                tvec = np.sum(t_stack * weights[:, None], axis=0).reshape(3, 1)
                q_target = rvec_to_quat(source_result["rvec"])
                q_avg = weighted_average_quats(
                    [rvec_to_quat(sample[1]["rvec"]) for sample in samples],
                    [float(w) for w in weights],
                    reference=q_target,
                )
                q_limited, rotation_delta_deg, rotation_limited = limit_quat_rotation(
                    q_target,
                    q_avg,
                    TEMPORAL_SMOOTHING_MAX_ROTATION_DEG,
                )
                rvec = quat_to_rvec(q_limited)

                target_result = cube.get("result", {})
                detector = estimator.detector_by_camera_cube.get((camera_name, cube_name))
                reproj_eval = (
                    None
                    if detector is None
                    else pose_reprojection_errors_for_result(source_result, detector, rvec, tvec)
                )
                if reproj_eval is not None:
                    smoothed_reproj, _smoothed_per_tag = reproj_eval
                    source_reproj = float(source_result.get("reproj_error", smoothed_reproj))
                    max_allowed_reproj = max(
                        TEMPORAL_SMOOTHING_MAX_DISPLAY_REPROJ_PX,
                        source_reproj * TEMPORAL_SMOOTHING_MAX_REPROJ_RATIO,
                    )
                    if smoothed_reproj > max_allowed_reproj:
                        target_result["temporal_smoothing_rejected"] = True
                        target_result["temporal_smoothing_reject_reason"] = (
                            "display_reprojection_too_high"
                        )
                        target_result["temporal_smoothing_candidate_reproj_error"] = float(
                            smoothed_reproj
                        )
                        target_result["temporal_smoothing_max_allowed_reproj_error"] = float(
                            max_allowed_reproj
                        )
                        continue

                target_result["tvec"] = tvec
                target_result["rvec"] = rvec
                target_result["T"] = pose_transform_from_rvec_tvec(rvec, tvec)
                if reproj_eval is not None:
                    smoothed_reproj, smoothed_per_tag = reproj_eval
                    if "reproj_error_before_smoothing" not in target_result:
                        target_result["reproj_error_before_smoothing"] = target_result.get(
                            "reproj_error",
                            None,
                        )
                    if "per_tag_reproj_error_before_smoothing" not in target_result:
                        target_result["per_tag_reproj_error_before_smoothing"] = target_result.get(
                            "per_tag_reproj_error",
                            None,
                        )
                    target_result["reproj_error"] = float(smoothed_reproj)
                    target_result["per_tag_reproj_error"] = smoothed_per_tag
                target_result["temporal_smoothed"] = True
                target_result["temporal_smoothing_source_count"] = int(len(samples))
                target_result["temporal_smoothing_window_radius"] = int(window_radius)
                target_result["temporal_smoothing_rotation_delta_deg"] = float(rotation_delta_deg)
                target_result["temporal_smoothing_rotation_limited"] = bool(rotation_limited)
                smoothed_count += 1

    for pose_frame in smoothed:
        pose_frame["temporal_smoothing_enabled"] = bool(TEMPORAL_SMOOTHING_ENABLED)
        pose_frame["temporal_smoothing_window_radius"] = int(window_radius)
        rebuild_pose_frame_status_lines(estimator, pose_frame)

    return smoothed, smoothed_count


def limit_pose_cache_rotation_jumps(
    pose_cache: list[dict[str, Any]],
    estimator: ReplayPoseEstimator,
    *,
    max_rotation_deg: float = TEMPORAL_ROTATION_JUMP_MAX_DEG,
    hold_rotation_deg: float = TEMPORAL_ROTATION_JUMP_HOLD_DEG,
) -> tuple[list[dict[str, Any]], int]:
    if not TEMPORAL_ROTATION_JUMP_LIMIT_ENABLED:
        return pose_cache, 0

    limited = copy.deepcopy(pose_cache)
    limited_count = 0

    for camera_name in estimator.active_camera_names:
        cube_names = [
            entry["cube_name"]
            for entry in estimator.detector_entries_by_camera.get(camera_name, [])
        ]
        frame_indices = [
            idx
            for idx, pose_frame in enumerate(limited)
            if pose_frame.get("camera_name") == camera_name
        ]
        for cube_name in cube_names:
            previous_quat: np.ndarray | None = None
            for idx in frame_indices:
                pose_frame = limited[idx]
                cube = cube_result_by_name(pose_frame).get(cube_name)
                if cube is None:
                    continue
                result = cube.get("result", {})
                if not bool(result.get("success", False)):
                    previous_quat = None
                    continue
                current_quat = rvec_to_quat(result["rvec"])
                if previous_quat is None:
                    previous_quat = current_quat
                    continue
                limited_quat, rotation_delta_deg, was_limited = limit_quat_rotation(
                    previous_quat,
                    current_quat,
                    max_rotation_deg,
                )
                if was_limited:
                    if rotation_delta_deg > hold_rotation_deg:
                        output_quat = previous_quat
                        result["temporal_rotation_jump_held"] = True
                    else:
                        output_quat = limited_quat
                    rvec = quat_to_rvec(output_quat)
                    tvec = np.asarray(result["tvec"], dtype=np.float64).reshape(3, 1)
                    result["rvec"] = rvec
                    result["T"] = pose_transform_from_rvec_tvec(rvec, tvec)
                    result["temporal_rotation_jump_limited"] = True
                    result["temporal_rotation_jump_original_delta_deg"] = float(rotation_delta_deg)
                    result["temporal_rotation_jump_max_deg"] = float(max_rotation_deg)
                    result["temporal_rotation_jump_hold_deg"] = float(hold_rotation_deg)
                    pose_frame["temporal_rotation_jump_limited_count"] = int(
                        pose_frame.get("temporal_rotation_jump_limited_count", 0)
                    ) + 1
                    limited_count += 1
                    previous_quat = output_quat
                else:
                    previous_quat = current_quat

    for pose_frame in limited:
        pose_frame["temporal_rotation_jump_limit_enabled"] = bool(
            TEMPORAL_ROTATION_JUMP_LIMIT_ENABLED
        )
        pose_frame["temporal_rotation_jump_max_deg"] = float(max_rotation_deg)
        pose_frame["temporal_rotation_jump_hold_deg"] = float(hold_rotation_deg)
        rebuild_pose_frame_status_lines(estimator, pose_frame)

    return limited, limited_count


def complete_and_smooth_pose_cache(
    pose_cache: list[dict[str, Any]],
    estimator: ReplayPoseEstimator,
) -> tuple[list[dict[str, Any]], int, int, int, int]:
    reset_pose_cache, reset_count = reset_temporal_postprocess_outputs(pose_cache)
    gated_pose_cache, rejected_count = gate_single_tag_pose_cache(
        reset_pose_cache,
        estimator,
    )
    outlier_gated_pose_cache, outlier_rejected_count = gate_temporal_outlier_pose_cache(
        gated_pose_cache,
        estimator,
    )
    rejected_count += outlier_rejected_count
    completed, filled_count = complete_pose_cache_temporally(outlier_gated_pose_cache, estimator)
    if not TEMPORAL_SMOOTHING_ENABLED:
        return completed, filled_count, 0, rejected_count, reset_count
    smoothed, smoothed_count = smooth_pose_cache_temporally(completed, estimator)
    limited, limited_count = limit_pose_cache_rotation_jumps(smoothed, estimator)
    return limited, filled_count, smoothed_count + limited_count, rejected_count, reset_count


def make_pose_cache_key(
    *,
    frame_offsets: list[int],
    active_camera_names: list[str],
    cube_paths: list[Path],
    use_undistort: bool,
    adaptive_clahe: bool,
    shared_tag_detection: bool,
    enable_filter: bool,
    fast: bool,
    temporal_postprocess_enabled: bool,
    demo008: Any,
) -> dict[str, Any]:
    return {
        "format": POSE_CACHE_FORMAT,
        "frame_count": len(frame_offsets),
        "active_camera_names": list(active_camera_names),
        "cube_paths": [str(path) for path in cube_paths],
        "intrinsics_yaml": {
            name: demo008.CAMERA_TO_INTRINSICS_YAML[name] for name in active_camera_names
        },
        "use_undistort": bool(use_undistort),
        "adaptive_clahe": bool(adaptive_clahe),
        "image_recovery_version": int(IMAGE_RECOVERY_VERSION),
        "single_tag_face_frame_solver_version": int(
            SINGLE_TAG_FACE_FRAME_SOLVER_VERSION
        ),
        "single_tag_face_frame_max_reproj_px": float(
            SINGLE_TAG_FACE_FRAME_MAX_REPROJ_PX
        ),
        "single_tag_face_frame_reproj_tie_px": float(
            SINGLE_TAG_FACE_FRAME_REPROJ_TIE_PX
        ),
        "single_tag_face_frame_lm_refine": bool(SINGLE_TAG_FACE_FRAME_LM_REFINE),
        "shared_tag_detection": bool(shared_tag_detection),
        "enable_filter": bool(enable_filter),
        "fast": bool(fast),
        "temporal_postprocess_enabled": bool(temporal_postprocess_enabled),
        "single_tag_continuity_gate_enabled": bool(SINGLE_TAG_CONTINUITY_GATE_ENABLED),
        "single_tag_continuity_max_rotation_deg": float(
            SINGLE_TAG_CONTINUITY_MAX_ROTATION_DEG
        ),
        "single_tag_continuity_version": int(SINGLE_TAG_CONTINUITY_VERSION),
        "temporal_outlier_gate_enabled": bool(TEMPORAL_OUTLIER_GATE_ENABLED),
        "temporal_outlier_max_neighbor_gap_frames": int(
            TEMPORAL_OUTLIER_MAX_NEIGHBOR_GAP_FRAMES
        ),
        "temporal_outlier_neighbor_max_rotation_deg": float(
            TEMPORAL_OUTLIER_NEIGHBOR_MAX_ROTATION_DEG
        ),
        "temporal_outlier_neighbor_max_translation_mm": float(
            TEMPORAL_OUTLIER_NEIGHBOR_MAX_TRANSLATION_MM
        ),
        "temporal_outlier_min_rotation_jump_deg": float(
            TEMPORAL_OUTLIER_MIN_ROTATION_JUMP_DEG
        ),
        "temporal_outlier_min_translation_jump_mm": float(
            TEMPORAL_OUTLIER_MIN_TRANSLATION_JUMP_MM
        ),
        "temporal_outlier_version": int(TEMPORAL_OUTLIER_VERSION),
        "temporal_fill_enabled": True,
        "temporal_fill_max_gap_frames": int(TEMPORAL_FILL_MAX_GAP_FRAMES),
        "temporal_fill_max_rotation_deg": float(TEMPORAL_FILL_MAX_ROTATION_DEG),
        "temporal_fill_version": int(TEMPORAL_FILL_VERSION),
        "temporal_smoothing_enabled": bool(TEMPORAL_SMOOTHING_ENABLED),
        "temporal_smoothing_window_radius": int(TEMPORAL_SMOOTHING_WINDOW_RADIUS),
        "temporal_smoothing_sigma_frames": float(TEMPORAL_SMOOTHING_SIGMA_FRAMES),
        "temporal_smoothing_max_rotation_deg": float(TEMPORAL_SMOOTHING_MAX_ROTATION_DEG),
        "temporal_smoothing_version": int(TEMPORAL_SMOOTHING_VERSION),
        "temporal_rotation_jump_limit_enabled": bool(TEMPORAL_ROTATION_JUMP_LIMIT_ENABLED),
        "temporal_rotation_jump_max_deg": float(TEMPORAL_ROTATION_JUMP_MAX_DEG),
        "temporal_rotation_jump_hold_deg": float(TEMPORAL_ROTATION_JUMP_HOLD_DEG),
        "temporal_rotation_jump_limit_version": int(TEMPORAL_ROTATION_JUMP_LIMIT_VERSION),
        "fisheye_rectified_horizontal_fov_deg": (
            None
            if getattr(demo008, "FISHEYE_RECTIFIED_HORIZONTAL_FOV_DEG", None) is None
            else float(getattr(demo008, "FISHEYE_RECTIFIED_HORIZONTAL_FOV_DEG"))
        ),
    }


def load_cached_pose_cache(
    pose_cache_record: dict[str, Any] | None,
    expected_key: dict[str, Any],
    *,
    allow_temporal_postprocess_mismatch: bool = False,
) -> tuple[list[dict[str, Any]], bool] | None:
    if not isinstance(pose_cache_record, dict):
        return None
    if pose_cache_record.get("format") != POSE_CACHE_FORMAT:
        return None
    record_key = pose_cache_record.get("key")
    if isinstance(record_key, dict) and record_key.get("format") == POSE_CACHE_FORMAT_023_DEEPTAG_008:
        pose_cache = pose_cache_record.get("pose_cache", None)
        if isinstance(pose_cache, list) and len(pose_cache) == int(expected_key["frame_count"]):
            return pose_cache, True
        return None
    if isinstance(record_key, dict) and record_key.get("format") == POSE_CACHE_FORMAT_020_MULTISTAGE:
        if not bool(expected_key.get("temporal_postprocess_enabled", True)):
            return None
        pose_cache = pose_cache_record.get("pose_cache", None)
        if isinstance(pose_cache, list) and len(pose_cache) == int(expected_key["frame_count"]):
            return pose_cache, True
        return None
    exact_match = record_key == expected_key
    if not exact_match and isinstance(record_key, dict):
        stable_record_key = {
            key: value for key, value in record_key.items() if key != "frame_offsets"
        }
        stable_expected_key = {
            key: value for key, value in expected_key.items() if key != "frame_offsets"
        }
        exact_match = stable_record_key == stable_expected_key
    compatible_without_temporal = False
    if not exact_match and isinstance(record_key, dict):
        temporal_keys = {
            "frame_offsets",
            "single_tag_continuity_gate_enabled",
            "single_tag_continuity_max_rotation_deg",
            "single_tag_continuity_version",
            "temporal_outlier_gate_enabled",
            "temporal_outlier_max_neighbor_gap_frames",
            "temporal_outlier_neighbor_max_rotation_deg",
            "temporal_outlier_neighbor_max_translation_mm",
            "temporal_outlier_min_rotation_jump_deg",
            "temporal_outlier_min_translation_jump_mm",
            "temporal_outlier_version",
            "temporal_fill_enabled",
            "temporal_fill_max_gap_frames",
            "temporal_fill_max_rotation_deg",
            "temporal_fill_version",
            "temporal_smoothing_enabled",
            "temporal_smoothing_window_radius",
            "temporal_smoothing_sigma_frames",
            "temporal_smoothing_max_rotation_deg",
            "temporal_smoothing_version",
            "temporal_rotation_jump_limit_enabled",
            "temporal_rotation_jump_max_deg",
            "temporal_rotation_jump_hold_deg",
            "temporal_rotation_jump_limit_version",
        }
        if allow_temporal_postprocess_mismatch:
            temporal_keys.add("temporal_postprocess_enabled")
        stripped_record_key = {
            key: value for key, value in record_key.items() if key not in temporal_keys
        }
        stripped_expected_key = {
            key: value for key, value in expected_key.items() if key not in temporal_keys
        }
        compatible_without_temporal = stripped_record_key == stripped_expected_key
    if not exact_match and not compatible_without_temporal:
        return None
    pose_cache = pose_cache_record.get("pose_cache", None)
    if not isinstance(pose_cache, list):
        return None
    if len(pose_cache) != int(expected_key["frame_count"]):
        return None
    return pose_cache, exact_match


def write_pose_cache_into_pkl_frames(
    pkl_path: Path,
    cache_key: dict[str, Any],
    pose_cache: list[dict[str, Any]],
) -> None:
    tmp_path = pkl_path.with_name(f".{pkl_path.name}.rewrite-{time.time_ns()}.tmp")
    frame_idx = 0
    try:
        with pkl_path.open("rb") as src, tmp_path.open("wb") as dst:
            while True:
                try:
                    record = pickle.load(src)
                except EOFError:
                    break

                if isinstance(record, dict) and record.get("type") == "pose_cache":
                    continue

                if isinstance(record, dict) and record.get("type") == "frame":
                    if frame_idx >= len(pose_cache):
                        raise ValueError(
                            f"PKL has more frame records than pose cache entries: >{len(pose_cache)}"
                        )
                    record[OFFLINE_POS_FRAME_FIELD] = pose_cache[frame_idx]
                    record[OFFLINE_POS_CACHE_KEY_FIELD] = cache_key
                    frame_idx += 1

                pickle.dump(record, dst, protocol=pickle.HIGHEST_PROTOCOL)

        if frame_idx != len(pose_cache):
            raise ValueError(
                f"PKL frame count {frame_idx} does not match pose cache count {len(pose_cache)}"
            )
        tmp_path.replace(pkl_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def main() -> None:
    demo008 = load_demo008_module()
    pkl_path = resolve_pkl_path(PKL_PATH)
    print(f"[INFO] PKL: {pkl_path}")
    print("[INFO] Building lightweight frame index. This scans the file once without retaining images.")
    (
        header,
        frame_offsets,
        footer,
        pose_cache_record,
        offline_pos_cache_record,
        legacy_inline_pose_cache_record,
    ) = build_frame_index(pkl_path)
    if not frame_offsets:
        raise ValueError(f"No frame records found in {pkl_path}")

    total_frames = len(frame_offsets)
    metadata = header.get("metadata", {}) if isinstance(header, dict) else {}
    first_record = load_frame_at_offset(pkl_path, frame_offsets[0])
    first_record_camera_name = str(first_record.get("camera_name", ""))
    print(f"[INFO] Indexed frames: {total_frames}")
    if footer is not None:
        print(f"[INFO] Footer frame_count={footer.get('frame_count')} reason={footer.get('reason')}")

    if not isinstance(metadata, dict):
        raise ValueError("PKL header metadata must be a dictionary.")

    recorded_camera_names = metadata.get("opened_cameras", None)
    if not isinstance(recorded_camera_names, (list, tuple)) or not recorded_camera_names:
        raise ValueError("PKL header has no non-empty metadata['opened_cameras'] list.")
    active_camera_names = [str(name) for name in recorded_camera_names]
    if first_record_camera_name and first_record_camera_name not in active_camera_names:
        raise ValueError(
            f"First PKL frame uses camera '{first_record_camera_name}', but "
            f"metadata['opened_cameras']={active_camera_names}."
        )

    recorded_intrinsics = metadata.get("intrinsics_yaml", None)
    if not isinstance(recorded_intrinsics, dict):
        raise ValueError("PKL header has no metadata['intrinsics_yaml'] mapping.")
    camera_to_intrinsics_yaml: dict[str, str] = {}
    for camera_name in active_camera_names:
        intrinsics_path = recorded_intrinsics.get(camera_name, None)
        if not isinstance(intrinsics_path, (str, Path)):
            raise ValueError(
                f"PKL header has no intrinsics YAML for camera '{camera_name}'."
            )
        resolved_intrinsics_path = Path(intrinsics_path).expanduser().resolve()
        if not resolved_intrinsics_path.is_file():
            raise FileNotFoundError(
                f"PKL-recorded intrinsics YAML does not exist: {resolved_intrinsics_path}"
            )
        camera_to_intrinsics_yaml[camera_name] = str(resolved_intrinsics_path)
        print(
            f"[INFO] [{camera_name}] Using PKL-recorded intrinsics YAML: "
            f"{resolved_intrinsics_path}"
        )

    recorded_cube_paths = metadata.get("cube_paths", None)
    if not isinstance(recorded_cube_paths, (list, tuple)) or not recorded_cube_paths:
        raise ValueError("PKL header has no non-empty metadata['cube_paths'] list.")
    indexed_offline_pos_key = (
        offline_pos_cache_record.get("key", {})
        if isinstance(offline_pos_cache_record, dict)
        else {}
    )
    if (
        isinstance(indexed_offline_pos_key, dict)
        and indexed_offline_pos_key.get("format") == POSE_CACHE_FORMAT_023_DEEPTAG_008
    ):
        cached_cube_paths = indexed_offline_pos_key.get("cube_paths", None)
        if isinstance(cached_cube_paths, (list, tuple)) and cached_cube_paths:
            recorded_cube_paths = cached_cube_paths
            print("[INFO] Restricting cube cfgs to the 023 DeepTag cache selection.")
    cube_paths = [
        demo008.validate_cube_path(Path(path).expanduser().resolve())
        for path in recorded_cube_paths
    ]
    print(f"[INFO] Using PKL-recorded cube cfgs: {[str(path) for path in cube_paths]}")

    if "undistort_before_detection" not in metadata:
        raise ValueError(
            "PKL header has no metadata['undistort_before_detection'] setting."
        )
    use_undistort = bool(metadata["undistort_before_detection"])
    fisheye_fov_setting = metadata.get(
        "fisheye_rectified_horizontal_fov_deg_setting",
        None,
    )
    if fisheye_fov_setting is not None:
        fisheye_fov_setting = float(fisheye_fov_setting)

    demo008.ACTIVE_CAMERA_NAMES = list(active_camera_names)
    demo008.CAMERA_TO_INTRINSICS_YAML = camera_to_intrinsics_yaml
    demo008.CUBE_CFG_DIRS = list(cube_paths)
    demo008.UNDISTORT_BEFORE_DETECTION = use_undistort
    demo008.FISHEYE_RECTIFIED_HORIZONTAL_FOV_DEG = fisheye_fov_setting
    demo008.PINHOLE_UNDISTORT_ALPHA = float(PINHOLE_UNDISTORT_ALPHA)
    demo008.ADAPTIVE_CLAHE_DETECTION = bool(ADAPTIVE_CLAHE_DETECTION)
    demo008.ENABLE_FILTER = bool(ENABLE_RUNTIME_POSE_FILTER)
    demo008.FAST_DETECTOR = bool(FAST_DETECTOR)

    adaptive_clahe = bool(ADAPTIVE_CLAHE_DETECTION)
    enable_filter = bool(ENABLE_RUNTIME_POSE_FILTER)
    fast = bool(FAST_DETECTOR)
    temporal_postprocess_enabled = bool(ENABLE_TEMPORAL_POSTPROCESS)
    estimator = ReplayPoseEstimator(
        demo008,
        active_camera_names=active_camera_names,
        cube_paths=cube_paths,
        use_undistort=use_undistort,
        adaptive_clahe=adaptive_clahe,
        shared_tag_detection=bool(SHARED_TAG_DETECTION),
        enable_filter=enable_filter,
        fast=fast,
    )
    pose_cache_key = make_pose_cache_key(
        frame_offsets=frame_offsets,
        active_camera_names=active_camera_names,
        cube_paths=cube_paths,
        use_undistort=use_undistort,
        adaptive_clahe=adaptive_clahe,
        shared_tag_detection=bool(SHARED_TAG_DETECTION),
        enable_filter=enable_filter,
        fast=fast,
        temporal_postprocess_enabled=temporal_postprocess_enabled,
        demo008=demo008,
    )
    print(
        "[INFO] 008 replay detection path: "
        f"{'shared' if SHARED_TAG_DETECTION else 'per-cube'} detect_tags(frame) "
        "+ per-cube process_detections(), sequential over PKL frames."
    )
    print(
        "[INFO] 009 single-tag face-frame solver: "
        f"max_reproj={SINGLE_TAG_FACE_FRAME_MAX_REPROJ_PX:.1f}px "
        f"tie={SINGLE_TAG_FACE_FRAME_REPROJ_TIE_PX:.1f}px "
        f"lm_refine={SINGLE_TAG_FACE_FRAME_LM_REFINE}."
    )
    print(
        "[INFO] 009 offline temporal postprocess: "
        f"enabled={temporal_postprocess_enabled}."
    )
    if RECOMPUTE_POSE:
        offline_pos_cached_pose = None
        legacy_inline_cached_pose = None
        appended_cached_pose = None
        print("[INFO] Ignoring existing pose caches because RECOMPUTE_POSE is enabled.")
    else:
        offline_pos_cached_pose = load_cached_pose_cache(
            offline_pos_cache_record,
            pose_cache_key,
            allow_temporal_postprocess_mismatch=temporal_postprocess_enabled,
        )
        legacy_inline_cached_pose = load_cached_pose_cache(
            legacy_inline_pose_cache_record,
            pose_cache_key,
        )
        appended_cached_pose = load_cached_pose_cache(pose_cache_record, pose_cache_key)
    offline_pos_key = (
        offline_pos_cache_record.get("key", {})
        if isinstance(offline_pos_cache_record, dict)
        else {}
    )
    offline_pos_is_023 = (
        isinstance(offline_pos_key, dict)
        and offline_pos_key.get("format") == POSE_CACHE_FORMAT_023_DEEPTAG_008
    )
    if offline_pos_is_023:
        cache_candidates = (
            ("023 DeepTag offline_pos frame records", offline_pos_cached_pose),
            ("legacy offline_pose_frame records", legacy_inline_cached_pose),
            ("appended PKL cache", appended_cached_pose),
        )
    else:
        cache_candidates = (
            ("legacy offline_pose_frame records", legacy_inline_cached_pose),
            ("offline_pos frame records", offline_pos_cached_pose),
            ("appended PKL cache", appended_cached_pose),
        )
    cache_source, cached_pose = next(
        ((source, cached) for source, cached in cache_candidates if cached is not None),
        ("", None),
    )
    pose_cache_needs_write = bool(RECOMPUTE_POSE)
    if cached_pose is not None:
        pose_cache, cache_exact_match = cached_pose
        using_offline_pos = cache_source in {
            "offline_pos frame records",
            "023 DeepTag offline_pos frame records",
        }
        if not using_offline_pos:
            print(
                f"[INFO] No compatible offline_pos cache; using {cache_source} "
                "as a read-only visualization fallback."
            )
        if cache_exact_match:
            if cache_source == "023 DeepTag offline_pos frame records":
                cache_description = "023 DeepTag-grid measured"
            else:
                cache_description = (
                    "temporal-completed smoothed"
                    if temporal_postprocess_enabled
                    else "raw measured"
                )
            print(
                f"[INFO] Loaded cached {cache_description} pose estimation "
                f"from {cache_source}: frames={len(pose_cache)}"
            )
        elif temporal_postprocess_enabled:
            (
                pose_cache,
                filled_count,
                smoothed_count,
                rejected_count,
                reset_count,
            ) = complete_and_smooth_pose_cache(
                pose_cache,
                estimator,
            )
            pose_cache_needs_write = False
            print(
                f"[INFO] Loaded cached pose estimation from {cache_source} and applied "
                "single-tag gate + temporal completion+smoothing: "
                f"frames={len(pose_cache)} reset={reset_count} "
                f"rejected={rejected_count} filled={filled_count} smoothed={smoothed_count}"
            )
        else:
            pose_cache_needs_write = using_offline_pos
            print(
                f"[INFO] Loaded compatible raw pose estimation from {cache_source}: "
                f"frames={len(pose_cache)}"
            )
    else:
        pose_cache = precompute_pose_cache(pkl_path, frame_offsets, metadata, estimator)
        pose_cache_needs_write = True
        if temporal_postprocess_enabled:
            (
                pose_cache,
                filled_count,
                smoothed_count,
                rejected_count,
                reset_count,
            ) = complete_and_smooth_pose_cache(
                pose_cache,
                estimator,
            )
            print(
                "[INFO] Applied single-tag gate + temporal completion+smoothing: "
                f"reset={reset_count} rejected={rejected_count} "
                f"filled={filled_count} smoothed={smoothed_count}"
            )
        else:
            print("[INFO] Kept raw measured poses; skipped offline temporal postprocess.")

    for pose_frame in pose_cache:
        if not isinstance(pose_frame.get("status_lines"), list):
            rebuild_pose_frame_status_lines(estimator, pose_frame)

    if pose_cache_needs_write:
        write_pose_cache_into_pkl_frames(pkl_path, pose_cache_key, pose_cache)
        cache_description = (
            "temporal-completed smoothed"
            if temporal_postprocess_enabled
            else "raw measured"
        )
        print(
            f"[INFO] Wrote {cache_description} pose estimation into ordered "
            f"PKL frame records: frames={len(pose_cache)}"
        )
    if PRECOMPUTE_ONLY:
        print("[INFO] Precompute-only mode finished; exiting before starting Viser.")
        return

    first_raw_rgb = bgr_to_rgb_for_viser(
        first_record["image_bgr"],
        int(VISER_MAX_IMAGE_WIDTH),
    )
    first_detector_tagpose_bgr = estimator.draw_detector_input_pose_frame(first_record, pose_cache[0])
    first_detector_tagpose_rgb = bgr_to_rgb_for_viser(
        first_detector_tagpose_bgr,
        int(VISER_MAX_IMAGE_WIDTH),
    )
    first_undistorted_debug_bgr = estimator.draw_undistorted_debug_frame(first_record, pose_cache[0])
    first_undistorted_debug_rgb = bgr_to_rgb_for_viser(
        first_undistorted_debug_bgr,
        int(VISER_MAX_IMAGE_WIDTH),
    )

    server = viser.ViserServer(host=VISER_HOST, port=int(VISER_PORT))
    scene_handles = create_3d_scene_handles(server, estimator, pose_cache)
    update_3d_scene(scene_handles, pose_cache[0])

    with server.gui.add_folder("Detector Input TagPose"):
        detector_tagpose_handle = server.gui.add_image(
            first_detector_tagpose_rgb,
            label="",
            format="jpeg",
            jpeg_quality=int(VISER_JPEG_QUALITY),
        )
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=total_frames - 1,
            step=1,
            initial_value=0,
        )
        auto_play_checkbox = server.gui.add_checkbox("Auto play", initial_value=False)
        status_text = server.gui.add_text(
            "Status",
            initial_value=record_summary(first_record, 0, total_frames),
            disabled=True,
        )
        pose_text = server.gui.add_markdown(pose_markdown(pose_cache[0]))

    with server.gui.add_folder("Undistorted Debug Image"):
        undistorted_debug_handle = server.gui.add_image(
            first_undistorted_debug_rgb,
            label="undistorted frame red-box on missing pose",
            format="jpeg",
            jpeg_quality=int(VISER_JPEG_QUALITY),
        )

    with server.gui.add_folder("Raw Image"):
        raw_image_handle = server.gui.add_image(
            first_raw_rgb,
            label="raw origin_frame_bgr",
            format="jpeg",
            jpeg_quality=int(VISER_JPEG_QUALITY),
        )

    with server.gui.add_folder("3D View"):
        show_box_checkbox = server.gui.add_checkbox("Cube box", initial_value=True)
        show_obj_checkbox = server.gui.add_checkbox("Finger OBJ", initial_value=True)
        show_axes_checkbox = server.gui.add_checkbox("Cube axes", initial_value=True)
        show_trajectory_checkbox = server.gui.add_checkbox("Trajectory", initial_value=True)
        show_samples_checkbox = server.gui.add_checkbox("Pose samples", initial_value=True)
        show_endpoints_checkbox = server.gui.add_checkbox("Start/end points", initial_value=True)
        show_camera_checkbox = server.gui.add_checkbox("Camera frustum", initial_value=True)

    with server.gui.add_folder("Replay Metadata"):
        server.gui.add_text("PKL", initial_value=str(pkl_path), disabled=True)
        if isinstance(metadata, dict):
            server.gui.add_markdown(
                "\n".join(
                    [
                        f"`recorded_image`: `{metadata.get('recorded_image', 'unknown')}`",
                        f"`capture_size`: `{metadata.get('capture_size', 'unknown')}`",
                        f"`fps`: `{metadata.get('fps', 'unknown')}`",
                        f"`fourcc`: `{metadata.get('fourcc', 'unknown')}`",
                    ]
                )
            )

    print(f"[INFO] Viser: http://{VISER_HOST}:{int(VISER_PORT)}")
    print(
        "[INFO] Use the sidebar folders: Detector Input TagPose, "
        "Undistorted Debug Image, Raw Image, Replay Metadata."
    )

    current_idx = -1
    last_auto_play_step = time.monotonic()
    while True:
        apply_3d_visibility(
            scene_handles,
            show_box=bool(show_box_checkbox.value),
            show_obj=bool(show_obj_checkbox.value),
            show_axes=bool(show_axes_checkbox.value),
            show_trajectory=bool(show_trajectory_checkbox.value),
            show_samples=bool(show_samples_checkbox.value),
            show_endpoints=bool(show_endpoints_checkbox.value),
            show_grid=False,
            show_camera=bool(show_camera_checkbox.value),
        )
        if bool(auto_play_checkbox.value):
            now = time.monotonic()
            if now - last_auto_play_step >= 0.1:
                frame_slider.value = (int(frame_slider.value) + 1) % total_frames
                last_auto_play_step = now
        else:
            last_auto_play_step = time.monotonic()

        slider_idx = int(frame_slider.value)
        if slider_idx != current_idx:
            try:
                record = load_frame_at_offset(pkl_path, frame_offsets[slider_idx])
                detector_tagpose_bgr = estimator.draw_detector_input_pose_frame(
                    record,
                    pose_cache[slider_idx],
                )
                detector_tagpose_handle.image = bgr_to_rgb_for_viser(
                    detector_tagpose_bgr,
                    int(VISER_MAX_IMAGE_WIDTH),
                )
                undistorted_debug_bgr = estimator.draw_undistorted_debug_frame(
                    record,
                    pose_cache[slider_idx],
                )
                undistorted_debug_handle.image = bgr_to_rgb_for_viser(
                    undistorted_debug_bgr,
                    int(VISER_MAX_IMAGE_WIDTH),
                )
                raw_image_handle.image = bgr_to_rgb_for_viser(
                    record["image_bgr"],
                    int(VISER_MAX_IMAGE_WIDTH),
                )
                status_text.value = record_summary(record, slider_idx, total_frames)
                pose_text.content = pose_markdown(pose_cache[slider_idx])
                update_3d_scene(scene_handles, pose_cache[slider_idx])
                apply_3d_visibility(
                    scene_handles,
                    show_box=bool(show_box_checkbox.value),
                    show_obj=bool(show_obj_checkbox.value),
                    show_axes=bool(show_axes_checkbox.value),
                    show_trajectory=bool(show_trajectory_checkbox.value),
                    show_samples=bool(show_samples_checkbox.value),
                    show_endpoints=bool(show_endpoints_checkbox.value),
                    show_grid=False,
                    show_camera=bool(show_camera_checkbox.value),
                )
                current_idx = slider_idx
            except Exception as exc:
                status_text.value = f"Failed to load frame {slider_idx}: {type(exc).__name__}: {exc}"
                print(f"[WARNING] {status_text.value}")
                current_idx = slider_idx
        time.sleep(0.03)


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit(
            "009 does not accept command-line arguments; edit the User macros "
            "at the top of this script."
        )
    main()
