#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

THIS_FILE = Path(__file__).resolve()
APRILCUBE_ROOT = THIS_FILE.parent.parent
SCRIPT_012_PATH = THIS_FILE.parent / "012_rs_aprilcube_detect.py"
DEEPTAG_ROOT = APRILCUBE_ROOT / "thirdparty" / "deeptag-pytorch"
DEFAULT_INPUT_PKL = APRILCUBE_ROOT / "recordings" / "016_deeptag_robust_cluster_012_rs_raw_frames_20260710_214336.pkl"
DEFAULT_OUTPUT_PKL = APRILCUBE_ROOT / "recordings" / "020_deeptag_dense_keypoints_pose_012_rs_raw_frames_20260710_214336.pkl"

if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))
if str(DEEPTAG_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEPTAG_ROOT))

import aprilcube  # noqa: E402
from stag_decode.pose_estimator import get_fine_grid_points_anno  # noqa: E402
from fiducial_marker.unit_arucotag import UnitArucoTag  # noqa: E402


CORNER_ORDER_TRANSFORMS = {
    "id": (0, 1, 2, 3),
    "rot90": (1, 2, 3, 0),
    "rev": (0, 3, 2, 1),
    "rot180": (2, 3, 0, 1),
    "rot270": (3, 0, 1, 2),
    "rev_rot90": (1, 0, 3, 2),
    "rev_rot180": (2, 1, 0, 3),
    "rev_rot270": (3, 2, 1, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute cube pose from saved DeepTag dense keypoints with no temporal filter."
    )
    parser.add_argument("deeptag_pkl", nargs="?", default=str(DEFAULT_INPUT_PKL))
    parser.add_argument("--output-pkl", type=Path, default=DEFAULT_OUTPUT_PKL)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-tags", type=int, default=2)
    parser.add_argument("--ransac-reproj", type=float, default=4.0)
    parser.add_argument("--max-reproj", type=float, default=6.0)
    parser.add_argument("--point-reject-px", type=float, default=8.0)
    parser.add_argument("--tag-reject-px", type=float, default=8.0)
    parser.add_argument("--min-inlier-tag-fraction", type=float, default=0.5)
    parser.add_argument("--coverage-check-min-raw-tags", type=int, default=3)
    parser.add_argument("--max-required-inlier-tags", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--no-source-overlay", action="store_true")
    return parser.parse_args()


def load_script012_module() -> Any:
    spec = importlib.util.spec_from_file_location("aprilcube_dense_deeptag_012", SCRIPT_012_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load 012 logic from {SCRIPT_012_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["aprilcube_dense_deeptag_012"] = module
    spec.loader.exec_module(module)
    return module


def build_stream_index(path: Path, expected_format: set[str] | None = None) -> tuple[dict[str, Any], list[int], dict[str, Any] | None]:
    offsets: list[int] = []
    footer: dict[str, Any] | None = None
    with path.open("rb") as f:
        header = pickle.load(f)
        if expected_format is not None and header.get("format") not in expected_format:
            raise ValueError(f"Unsupported pkl format in {path}: {header.get('format', None)}")
        while True:
            offset = f.tell()
            try:
                obj = pickle.load(f)
            except EOFError:
                break
            if not isinstance(obj, dict):
                continue
            if obj.get("type") == "frame":
                offsets.append(offset)
            elif obj.get("type") == "footer":
                footer = obj
                break
    if not offsets:
        raise ValueError(f"No frame records found in {path}")
    return header, offsets, footer


def load_at(path: Path, offset: int) -> dict[str, Any]:
    with path.open("rb") as f:
        f.seek(int(offset))
        obj = pickle.load(f)
    if not isinstance(obj, dict) or obj.get("type") != "frame":
        raise ValueError(f"Offset {offset} in {path} is not a frame")
    return obj


def encode_bgr_jpeg(image_bgr: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        np.asarray(image_bgr, dtype=np.uint8),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(max(1, min(int(quality), 100)))],
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return encoded.tobytes()


def decode_jpeg_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode JPEG")
    return image


def _jsonish(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonish(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonish(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _jsonish(item) for key, item in value.items()}
    return str(value)


def rotation_from_rvec(rvec: Any) -> np.ndarray:
    rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return rot


def visible_faces_for_ids(face_id_sets: dict[str, set[int]], tag_ids: list[int]) -> set[str]:
    faces: set[str] = set()
    for tag_id in tag_ids:
        for face_name, ids in face_id_sets.items():
            if int(tag_id) in ids:
                faces.add(str(face_name))
    return faces


def face_normals_ok(rvec: Any, visible_faces: set[str]) -> bool:
    rot = rotation_from_rvec(rvec)
    for face_name in visible_faces:
        for face_def in aprilcube.FACE_DEFS:
            if str(face_def[0]) != str(face_name):
                continue
            normal = np.zeros(3, dtype=np.float64)
            normal[int(face_def[1])] = float(face_def[2])
            if float((rot @ normal)[2]) > 0.0:
                return False
            break
    return True


def dense_local_annotations(num_points: int) -> np.ndarray:
    n = int(round(np.sqrt(int(num_points))))
    if n * n != int(num_points) or n < 3:
        raise ValueError(f"Unsupported dense keypoint count: {num_points}")
    grid_size = n - 2
    unit_tag = UnitArucoTag(grid_size, [0] * (grid_size * grid_size))
    anno = np.asarray(get_fine_grid_points_anno(unit_tag, step_elem_num=1), dtype=np.float64)
    return anno.reshape(-1, anno.shape[-1])[:, :2]


def local_to_cube_affine(tag_corners_3d: np.ndarray, corner_order: str) -> np.ndarray:
    # Stage-1 DeepTag corner annotation is [-0.5,-0.5] -> [0.5,-0.5] -> ...
    # Dense keypoint annotation applies the same x inversion as DeepTag's
    # PoseSolver.  Fit that dense-local plane to the ordered cube tag corners.
    stage1_corners = np.array(
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
        dtype=np.float64,
    )
    dense_corners = stage1_corners.copy()
    dense_corners[:, 0] *= -1.0
    order = np.asarray(CORNER_ORDER_TRANSFORMS[str(corner_order)], dtype=np.int64)
    local = np.c_[dense_corners[order], np.ones(4, dtype=np.float64)]
    target = np.asarray(tag_corners_3d, dtype=np.float64).reshape(4, 3)
    affine_t, *_ = np.linalg.lstsq(local, target, rcond=None)
    return affine_t


def dense_points_for_frame(
    frame: dict[str, Any],
    *,
    tag_corner_map: dict[int, np.ndarray],
    min_tags: int,
) -> tuple[np.ndarray, np.ndarray, list[int], dict[int, int], dict[str, Any]]:
    cluster_orders = frame.get("cluster_stats", {}).get("cluster_corner_orders", {}) or {}
    cluster_orders = {int(k): str(v) for k, v in cluster_orders.items()}
    order_votes: dict[str, int] = {}
    for order in cluster_orders.values():
        if order in CORNER_ORDER_TRANSFORMS:
            order_votes[order] = order_votes.get(order, 0) + 1
    dominant_order = max(order_votes.items(), key=lambda item: item[1])[0] if order_votes else "id"
    decoded_by_id: dict[int, dict[str, Any]] = {}
    for decoded in frame.get("decoded_tags", []) or []:
        if not decoded.get("is_valid", False):
            continue
        tag_id = int(decoded.get("tag_id", -1))
        if tag_id in tag_corner_map:
            decoded_by_id[tag_id] = decoded

    obj_chunks: list[np.ndarray] = []
    img_chunks: list[np.ndarray] = []
    tag_ids: list[int] = []
    point_counts: dict[int, int] = {}
    for tag_id in sorted(decoded_by_id):
        decoded = decoded_by_id[tag_id]
        image_points = np.asarray(decoded.get("keypoints_in_images", []), dtype=np.float64).reshape(-1, 2)
        if image_points.shape[0] < 4:
            continue
        local_xy = dense_local_annotations(image_points.shape[0])
        corner_order = cluster_orders.get(int(tag_id), dominant_order)
        affine_t = local_to_cube_affine(tag_corner_map[tag_id], corner_order)
        object_points = np.c_[local_xy, np.ones(local_xy.shape[0], dtype=np.float64)] @ affine_t
        obj_chunks.append(object_points.astype(np.float64))
        img_chunks.append(image_points.astype(np.float64))
        tag_ids.append(int(tag_id))
        point_counts[int(tag_id)] = int(image_points.shape[0])

    if len(tag_ids) < int(min_tags):
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            tag_ids,
            point_counts,
            {"reason": f"dense_tags_too_small:{len(tag_ids)}<{int(min_tags)}"},
        )
    return (
        np.vstack(obj_chunks),
        np.vstack(img_chunks),
        tag_ids,
        point_counts,
        {
            "cluster_corner_order_count": int(len(cluster_orders)),
            "corner_order_fallback": dominant_order,
            "used_fallback_order_tag_ids": [
                int(tag_id) for tag_id in tag_ids if int(tag_id) not in cluster_orders
            ],
        },
    )


def project_errors(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    return np.linalg.norm(image_points - projected.reshape(-1, 2), axis=1)


def face_def_by_name(face_name: str) -> tuple:
    for face_def in aprilcube.FACE_DEFS:
        if str(face_def[0]) == str(face_name):
            return face_def
    raise KeyError(f"Unknown face name: {face_name}")


def face_local_basis(cube_config: Any, face_name: str) -> tuple[np.ndarray, np.ndarray]:
    face_def = face_def_by_name(face_name)
    _name, normal_ax, normal_sign, right_ax, right_sign, down_ax, down_sign = face_def
    rot_cube_face = np.zeros((3, 3), dtype=np.float64)
    rot_cube_face[int(right_ax), 0] = float(right_sign)
    rot_cube_face[int(down_ax), 1] = float(down_sign)
    rot_cube_face[int(normal_ax), 2] = float(normal_sign)
    t_cube_face = np.zeros(3, dtype=np.float64)
    t_cube_face[int(normal_ax)] = float(normal_sign) * float(cube_config.box_dims[int(normal_ax)]) / 2.0
    return rot_cube_face, t_cube_face


def cube_points_to_face_points(cube_config: Any, face_name: str, cube_points: np.ndarray) -> np.ndarray:
    rot_cube_face, t_cube_face = face_local_basis(cube_config, face_name)
    points = np.asarray(cube_points, dtype=np.float64).reshape(-1, 3)
    face_points = (rot_cube_face.T @ (points - t_cube_face).T).T
    face_points[:, 2] = 0.0
    return face_points


def face_pose_to_cube_pose(
    cube_config: Any,
    face_name: str,
    face_rvec: np.ndarray,
    face_tvec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rot_cube_face, t_cube_face = face_local_basis(cube_config, face_name)
    rot_cam_face = rotation_from_rvec(face_rvec)
    rot_cam_cube = rot_cam_face @ rot_cube_face.T
    t_cam_cube = np.asarray(face_tvec, dtype=np.float64).reshape(3) - rot_cam_cube @ t_cube_face
    cube_rvec, _ = cv2.Rodrigues(rot_cam_cube)
    return cube_rvec.reshape(3, 1), t_cam_cube.reshape(3, 1)


def transform_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_from_rvec(rvec)
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def inlier_tag_coverage_failure(
    raw_tag_ids: list[int],
    used_tag_ids: list[int],
    *,
    min_tags: int,
    min_inlier_tag_fraction: float,
    coverage_check_min_raw_tags: int,
    max_required_inlier_tags: int,
) -> str:
    raw_count = len(set(int(tag_id) for tag_id in raw_tag_ids))
    used_count = len(set(int(tag_id) for tag_id in used_tag_ids))
    if raw_count < int(coverage_check_min_raw_tags):
        return ""
    required = int(np.ceil(raw_count * max(0.0, float(min_inlier_tag_fraction))))
    required = max(int(min_tags), required)
    required = min(max(required, 1), int(max_required_inlier_tags))
    if used_count < required:
        return f"dense_inlier_tags_low:{used_count}<{required}(raw={raw_count})"
    return ""


def best_single_face_ippe_pose(
    face_points: np.ndarray,
    cube_points: np.ndarray,
    image_points: np.ndarray,
    *,
    cube_config: Any,
    face_name: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, float, int]:
    try:
        retval, rvecs, tvecs, _errs = cv2.solvePnPGeneric(
            face_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        retval, rvecs, tvecs = 0, (), ()

    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    if retval:
        for face_rvec, face_tvec in zip(rvecs, tvecs):
            face_rvec = np.asarray(face_rvec, dtype=np.float64).reshape(3, 1)
            face_tvec = np.asarray(face_tvec, dtype=np.float64).reshape(3, 1)
            rot_cam_face = rotation_from_rvec(face_rvec)
            if float((rot_cam_face @ np.array([0.0, 0.0, 1.0], dtype=np.float64))[2]) > 0.0:
                continue
            cube_rvec, cube_tvec = face_pose_to_cube_pose(cube_config, face_name, face_rvec, face_tvec)
            if float(cube_tvec.reshape(3)[2]) <= 0.0:
                continue
            errors = project_errors(cube_points, image_points, cube_rvec, cube_tvec, camera_matrix, dist_coeffs)
            candidates.append((float(np.mean(errors)), cube_rvec, cube_tvec))

    if not candidates:
        try:
            ok, face_rvec, face_tvec = cv2.solvePnP(
                face_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            ok, face_rvec, face_tvec = False, None, None
        if ok and face_rvec is not None and face_tvec is not None:
            cube_rvec, cube_tvec = face_pose_to_cube_pose(cube_config, face_name, face_rvec, face_tvec)
            if float(cube_tvec.reshape(3)[2]) > 0.0:
                errors = project_errors(cube_points, image_points, cube_rvec, cube_tvec, camera_matrix, dist_coeffs)
                candidates.append((float(np.mean(errors)), cube_rvec, cube_tvec))

    if not candidates:
        return False, None, None, float("inf"), int(retval or 0)
    candidates.sort(key=lambda item: item[0])
    reproj, rvec, tvec = candidates[0]
    return True, rvec, tvec, reproj, len(candidates)


def solve_single_face_dense_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    tag_ids: list[int],
    point_counts: dict[int, int],
    *,
    cube_config: Any,
    face_name: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    max_reproj: float,
    point_reject_px: float,
    tag_reject_px: float,
    min_tags: int,
    min_inlier_tag_fraction: float,
    coverage_check_min_raw_tags: int,
    max_required_inlier_tags: int,
) -> dict[str, Any]:
    if object_points.shape[0] < 4:
        return {"success": False, "failure_reason": "dense_single_face_no_points", "reproj_error": float("inf")}
    face_points = cube_points_to_face_points(cube_config, face_name, object_points)
    active = np.ones(object_points.shape[0], dtype=bool)
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    candidate_count = 0
    rejected_points = 0
    rejected_tags: list[int] = []

    for _iteration in range(3):
        if int(active.sum()) < 4:
            break
        ok, next_rvec, next_tvec, _reproj, candidate_count = best_single_face_ippe_pose(
            face_points[active],
            object_points[active],
            image_points[active],
            cube_config=cube_config,
            face_name=face_name,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        if not ok or next_rvec is None or next_tvec is None:
            break
        rvec, tvec = next_rvec, next_tvec
        errors = project_errors(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs)
        active_errors = errors[active]
        if active_errors.size == 0:
            break
        point_thresh = min(max(float(np.median(active_errors)) * 3.0, 2.0), float(point_reject_px))
        point_keep = errors <= point_thresh

        tag_keep_ids: set[int] = set()
        start = 0
        for tag_id in tag_ids:
            count = int(point_counts[int(tag_id)])
            end = start + count
            tag_active = active[start:end] & point_keep[start:end]
            if int(tag_active.sum()) >= 4:
                mean_err = float(np.mean(errors[start:end][tag_active]))
                if mean_err <= float(tag_reject_px):
                    tag_keep_ids.add(int(tag_id))
            start = end

        next_active = active & point_keep
        start = 0
        for tag_id in tag_ids:
            count = int(point_counts[int(tag_id)])
            end = start + count
            if int(tag_id) not in tag_keep_ids:
                next_active[start:end] = False
            start = end
        if np.array_equal(next_active, active):
            break
        rejected_points += int(active.sum() - next_active.sum())
        rejected_tags = [int(tag_id) for tag_id in tag_ids if int(tag_id) not in tag_keep_ids]
        active = next_active

    if rvec is None or tvec is None or int(active.sum()) < 4:
        return {
            "success": False,
            "failure_reason": "dense_single_face_ippe_failed",
            "reproj_error": float("inf"),
            "rejected_points": int(rejected_points),
            "rejected_tags": rejected_tags,
        }

    ok, final_rvec, final_tvec, _final_reproj, candidate_count = best_single_face_ippe_pose(
        face_points[active],
        object_points[active],
        image_points[active],
        cube_config=cube_config,
        face_name=face_name,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )
    if ok and final_rvec is not None and final_tvec is not None:
        rvec, tvec = final_rvec, final_tvec
    errors = project_errors(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs)
    active_errors = errors[active]
    reproj = float(np.mean(active_errors))
    if not np.isfinite(reproj) or reproj > float(max_reproj):
        return {
            "success": False,
            "failure_reason": f"dense_single_face_reproj_too_high:{reproj:.2f}>{float(max_reproj):.2f}",
            "reproj_error": float("inf"),
            "raw_reproj_error": reproj,
            "rejected_points": int(rejected_points),
            "rejected_tags": rejected_tags,
        }

    used_ids: list[int] = []
    per_tag_reproj: dict[int, float] = {}
    per_tag_inliers: dict[int, int] = {}
    start = 0
    for tag_id in tag_ids:
        count = int(point_counts[int(tag_id)])
        end = start + count
        tag_active = active[start:end]
        if int(tag_active.sum()) > 0:
            used_ids.append(int(tag_id))
            per_tag_reproj[int(tag_id)] = float(np.mean(errors[start:end][tag_active]))
            per_tag_inliers[int(tag_id)] = int(tag_active.sum())
        start = end

    if len(used_ids) < int(min_tags):
        return {
            "success": False,
            "failure_reason": f"dense_single_face_final_tags_too_small:{len(used_ids)}<{int(min_tags)}",
            "reproj_error": float("inf"),
            "raw_reproj_error": reproj,
            "n_tags": int(len(used_ids)),
            "tag_ids": used_ids,
            "per_tag_reproj_error": per_tag_reproj,
            "per_tag_inlier_points": per_tag_inliers,
            "rejected_points": int(rejected_points),
            "rejected_tags": rejected_tags,
        }
    coverage_failure = inlier_tag_coverage_failure(
        tag_ids,
        used_ids,
        min_tags=min_tags,
        min_inlier_tag_fraction=min_inlier_tag_fraction,
        coverage_check_min_raw_tags=coverage_check_min_raw_tags,
        max_required_inlier_tags=max_required_inlier_tags,
    )
    if coverage_failure:
        return {
            "success": False,
            "failure_reason": coverage_failure,
            "reproj_error": float("inf"),
            "raw_reproj_error": reproj,
            "n_tags": int(len(used_ids)),
            "tag_ids": used_ids,
            "per_tag_reproj_error": per_tag_reproj,
            "per_tag_inlier_points": per_tag_inliers,
            "rejected_points": int(rejected_points),
            "rejected_tags": rejected_tags,
        }

    if not face_normals_ok(rvec, {face_name}):
        return {
            "success": False,
            "failure_reason": "dense_single_face_normal_away",
            "reproj_error": float("inf"),
            "raw_reproj_error": reproj,
        }

    if len(used_ids) >= 2:
        quality_level = "B"
        quality_reason = f"dense_singleface_face_frame:{len(used_ids)}tags"
    else:
        quality_level = "C"
        quality_reason = "dense_singletag_face_frame"

    return {
        "success": True,
        "failure_reason": "",
        "pose_source": "deeptag_dense_keypoints_single_face_ippe_cfg_transform",
        "quality_level": quality_level,
        "quality_reason": quality_reason,
        "pose_filled": False,
        "rvec": np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        "tvec": np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        "T": transform_from_rvec_tvec(rvec, tvec),
        "reproj_error": reproj,
        "n_points": int(active.sum()),
        "n_points_raw": int(object_points.shape[0]),
        "n_tags": int(len(used_ids)),
        "tag_ids": used_ids,
        "visible_faces": {face_name},
        "single_face_name": face_name,
        "single_face_ippe_candidates": int(candidate_count),
        "per_tag_reproj_error": per_tag_reproj,
        "per_tag_inlier_points": per_tag_inliers,
        "rejected_points": int(rejected_points),
        "rejected_tags": rejected_tags,
    }


def solve_dense_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    tag_ids: list[int],
    point_counts: dict[int, int],
    *,
    cube_config: Any,
    face_id_sets: dict[str, set[int]],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    ransac_reproj: float,
    max_reproj: float,
    point_reject_px: float,
    tag_reject_px: float,
    min_tags: int,
    min_inlier_tag_fraction: float,
    coverage_check_min_raw_tags: int,
    max_required_inlier_tags: int,
) -> dict[str, Any]:
    if object_points.shape[0] < 4:
        return {"success": False, "failure_reason": "dense_no_points", "reproj_error": float("inf")}

    raw_visible_faces = visible_faces_for_ids(face_id_sets, tag_ids)
    if len(raw_visible_faces) == 1:
        return solve_single_face_dense_pose(
            object_points,
            image_points,
            tag_ids,
            point_counts,
            cube_config=cube_config,
            face_name=next(iter(raw_visible_faces)),
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            max_reproj=max_reproj,
            point_reject_px=point_reject_px,
            tag_reject_px=tag_reject_px,
            min_tags=min_tags,
            min_inlier_tag_fraction=min_inlier_tag_fraction,
            coverage_check_min_raw_tags=coverage_check_min_raw_tags,
            max_required_inlier_tags=max_required_inlier_tags,
        )

    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            iterationsCount=300,
            reprojectionError=float(ransac_reproj),
            confidence=0.995,
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error:
        ok, rvec, tvec, inliers = False, None, None, None
    if not ok or rvec is None or tvec is None or float(np.asarray(tvec).reshape(3)[2]) <= 0.0:
        return {
            "success": False,
            "failure_reason": "dense_pnp_failed",
            "reproj_error": float("inf"),
        }

    active = np.ones(object_points.shape[0], dtype=bool)
    if inliers is not None and len(inliers) >= 4:
        active[:] = False
        active[np.asarray(inliers, dtype=np.int64).reshape(-1)] = True

    rejected_points = 0
    rejected_tags: list[int] = []
    for _iteration in range(2):
        if int(active.sum()) < 4:
            break
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points[active],
                image_points[active],
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
            )
        except cv2.error:
            pass
        errors = project_errors(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs)
        active_errors = errors[active]
        if active_errors.size == 0:
            break
        point_thresh = min(max(float(np.median(active_errors)) * 3.0, 2.0), float(point_reject_px))
        point_keep = errors <= point_thresh

        tag_keep_ids: set[int] = set()
        start = 0
        per_tag_mean: dict[int, float] = {}
        for tag_id in tag_ids:
            count = int(point_counts[int(tag_id)])
            end = start + count
            tag_active = active[start:end] & point_keep[start:end]
            if int(tag_active.sum()) >= 4:
                mean_err = float(np.mean(errors[start:end][tag_active]))
                per_tag_mean[int(tag_id)] = mean_err
                if mean_err <= float(tag_reject_px):
                    tag_keep_ids.add(int(tag_id))
            start = end

        next_active = active & point_keep
        start = 0
        for tag_id in tag_ids:
            count = int(point_counts[int(tag_id)])
            end = start + count
            if int(tag_id) not in tag_keep_ids:
                next_active[start:end] = False
            start = end
        if np.array_equal(next_active, active):
            break
        rejected_points += int(active.sum() - next_active.sum())
        rejected_tags = [int(tag_id) for tag_id in tag_ids if int(tag_id) not in tag_keep_ids]
        active = next_active

    if int(active.sum()) < 4:
        return {
            "success": False,
            "failure_reason": "dense_too_few_inlier_points",
            "reproj_error": float("inf"),
            "rejected_points": int(rejected_points),
            "rejected_tags": rejected_tags,
        }

    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points[active],
            image_points[active],
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
        )
    except cv2.error:
        pass
    errors = project_errors(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs)
    active_errors = errors[active]
    reproj = float(np.mean(active_errors))
    if not np.isfinite(reproj) or reproj > float(max_reproj):
        return {
            "success": False,
            "failure_reason": f"dense_reproj_too_high:{reproj:.2f}>{float(max_reproj):.2f}",
            "reproj_error": float("inf"),
            "raw_reproj_error": reproj,
            "rejected_points": int(rejected_points),
            "rejected_tags": rejected_tags,
        }

    used_ids: list[int] = []
    per_tag_reproj: dict[int, float] = {}
    per_tag_inliers: dict[int, int] = {}
    start = 0
    for tag_id in tag_ids:
        count = int(point_counts[int(tag_id)])
        end = start + count
        tag_active = active[start:end]
        if int(tag_active.sum()) > 0:
            used_ids.append(int(tag_id))
            per_tag_reproj[int(tag_id)] = float(np.mean(errors[start:end][tag_active]))
            per_tag_inliers[int(tag_id)] = int(tag_active.sum())
        start = end

    if len(used_ids) < int(min_tags):
        return {
            "success": False,
            "failure_reason": f"dense_final_tags_too_small:{len(used_ids)}<{int(min_tags)}",
            "reproj_error": float("inf"),
            "raw_reproj_error": reproj,
            "n_tags": int(len(used_ids)),
            "tag_ids": used_ids,
            "per_tag_reproj_error": per_tag_reproj,
            "per_tag_inlier_points": per_tag_inliers,
            "rejected_points": int(rejected_points),
            "rejected_tags": rejected_tags,
        }
    coverage_failure = inlier_tag_coverage_failure(
        tag_ids,
        used_ids,
        min_tags=min_tags,
        min_inlier_tag_fraction=min_inlier_tag_fraction,
        coverage_check_min_raw_tags=coverage_check_min_raw_tags,
        max_required_inlier_tags=max_required_inlier_tags,
    )
    if coverage_failure:
        return {
            "success": False,
            "failure_reason": coverage_failure,
            "reproj_error": float("inf"),
            "raw_reproj_error": reproj,
            "n_tags": int(len(used_ids)),
            "tag_ids": used_ids,
            "per_tag_reproj_error": per_tag_reproj,
            "per_tag_inlier_points": per_tag_inliers,
            "rejected_points": int(rejected_points),
            "rejected_tags": rejected_tags,
        }

    visible_faces = visible_faces_for_ids(face_id_sets, used_ids)
    if not face_normals_ok(rvec, visible_faces):
        return {
            "success": False,
            "failure_reason": "dense_face_normal_away",
            "reproj_error": float("inf"),
            "raw_reproj_error": reproj,
        }

    if len(visible_faces) >= 2:
        quality_level = "A"
        quality_reason = f"dense_multiface:{len(visible_faces)}faces/{len(used_ids)}tags"
    elif len(used_ids) >= 2:
        quality_level = "B"
        quality_reason = f"dense_multitag_singleface:{len(used_ids)}tags"
    else:
        quality_level = "C"
        quality_reason = "dense_single_tag_planar"

    return {
        "success": True,
        "failure_reason": "",
        "pose_source": "deeptag_dense_keypoints_all_point_pnp",
        "quality_level": quality_level,
        "quality_reason": quality_reason,
        "pose_filled": False,
        "rvec": np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        "tvec": np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        "T": transform_from_rvec_tvec(rvec, tvec),
        "reproj_error": reproj,
        "n_points": int(active.sum()),
        "n_points_raw": int(object_points.shape[0]),
        "n_tags": int(len(used_ids)),
        "tag_ids": used_ids,
        "visible_faces": visible_faces,
        "per_tag_reproj_error": per_tag_reproj,
        "per_tag_inlier_points": per_tag_inliers,
        "rejected_points": int(rejected_points),
        "rejected_tags": rejected_tags,
    }


def sanitize_pose(pose: dict[str, Any]) -> dict[str, Any]:
    return _jsonish(pose)


def make_runtime(header: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(header.get("metadata", {}) or {})
    cube_cfg = Path(metadata["cube_cfg"]).expanduser().resolve()
    cfg_path = cube_cfg / "config.json" if cube_cfg.is_dir() else cube_cfg
    cube_config, face_id_sets = aprilcube.load_cube_config(str(cfg_path))
    camera_matrix = np.asarray(metadata["camera_matrix"], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(metadata.get("dist_coeffs", np.zeros(5)), dtype=np.float64).reshape(-1)
    draw_detector = aprilcube.detector(
        cube_cfg,
        intrinsic_cfg={
            "fx": float(camera_matrix[0, 0]),
            "fy": float(camera_matrix[1, 1]),
            "cx": float(camera_matrix[0, 2]),
            "cy": float(camera_matrix[1, 2]),
        },
        dist_coeffs=dist_coeffs,
        enable_filter=False,
        fast=True,
    )
    return {
        "metadata": metadata,
        "cube_cfg": cube_cfg,
        "cube_config": cube_config,
        "face_id_sets": face_id_sets,
        "tag_corner_map": aprilcube.build_tag_corner_map(cube_config),
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "draw_detector": draw_detector,
    }


def make_source_frame_loader(header: dict[str, Any]) -> tuple[Path | None, dict[int, int], Any | None, tuple | None]:
    source = header.get("source_pkl", "")
    if not source:
        return None, {}, None, None
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        return None, {}, None, None
    source_header, source_offsets, _source_footer = build_stream_index(
        source_path,
        {"aprilcube_rs_raw_frame_stream_v1", "aprilcube_012_raw_with_pose_stream_v1"},
    )
    offset_set = {int(offset): int(offset) for offset in source_offsets}
    script012 = load_script012_module()
    metadata: dict[str, Any] = {}
    if source_header.get("format") == "aprilcube_012_raw_with_pose_stream_v1":
        metadata.update(source_header.get("raw_header", {}).get("metadata", {}) or {})
    metadata.update(source_header.get("metadata", {}) or {})
    try:
        intrinsics_yaml = Path(metadata.get("intrinsics_yaml")).expanduser().resolve()
        calib = script012.load_intrinsics_yaml(intrinsics_yaml)
        image_size = tuple(int(v) for v in metadata.get("image_size", calib["image_size"]))
        undistort_pack = None
        if bool(metadata.get("undistort_for_detection", True)):
            undistort_pack = script012.create_undistort_maps(calib, image_size)
        return source_path, offset_set, script012, undistort_pack
    except Exception:
        return source_path, offset_set, None, None


def source_detection_frame(
    source_path: Path | None,
    source_offsets: dict[int, int],
    script012: Any | None,
    undistort_pack: tuple | None,
    source_offset: int,
) -> np.ndarray | None:
    if source_path is None or int(source_offset) not in source_offsets:
        return None
    try:
        record = load_at(source_path, source_offsets[int(source_offset)])
        image = np.asarray(record["image_bgr"], dtype=np.uint8)
        if script012 is not None:
            return script012.undistort_frame(image, undistort_pack)
        return image
    except Exception:
        return None


def draw_overlay(
    base_bgr: np.ndarray,
    runtime: dict[str, Any],
    pose: dict[str, Any],
) -> np.ndarray:
    result = {
        "success": bool(pose.get("success", False)),
        "detections": [],
        "rvec": np.asarray(pose["rvec"], dtype=np.float64).reshape(3, 1) if pose.get("success", False) else None,
        "tvec": np.asarray(pose["tvec"], dtype=np.float64).reshape(3, 1) if pose.get("success", False) else None,
        "reproj_error": float(pose.get("reproj_error", float("inf"))),
        "n_tags": int(pose.get("n_tags", 0)),
        "visible_faces": set(pose.get("visible_faces", []) or []),
        "predicted": False,
    }
    vis = runtime["draw_detector"].draw_result(base_bgr.copy(), result)
    text = (
        f"DenseDeepTag success={pose.get('success', False)} "
        f"tags={pose.get('n_tags', 0)} pts={pose.get('n_points', 0)} "
        f"reproj={float(pose.get('reproj_error', float('inf'))):.2f}px"
    )
    cv2.rectangle(vis, (8, 8), (900, 42), (0, 0, 0), -1)
    cv2.putText(vis, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def main() -> None:
    args = parse_args()
    input_pkl = Path(args.deeptag_pkl).expanduser().resolve()
    header, all_offsets, footer = build_stream_index(input_pkl, {"deeptag_012_offline_stream_v1"})
    offsets = all_offsets[int(args.start_frame) :: max(int(args.stride), 1)]
    if args.max_frames > 0:
        offsets = offsets[: int(args.max_frames)]

    runtime = make_runtime(header)
    source_path, source_offsets, script012, undistort_pack = make_source_frame_loader(header)
    output_pkl = Path(args.output_pkl).expanduser().resolve()
    output_pkl.parent.mkdir(parents=True, exist_ok=True)

    success_count = 0
    total_points = 0
    t0 = time.perf_counter()
    with output_pkl.open("wb") as f:
        pickle.dump(
            {
                "type": "header",
                "format": "deeptag_012_offline_stream_v1",
                "source_pkl": str(input_pkl),
                "source_footer": footer,
                "metadata": {
                    "script": str(THIS_FILE),
                    "method": "DeepTag dense keypoints; single-face frames use cfg face-frame IPPE then fixed face-to-cube transform; multiface frames use cube-frame all-point PnP; no temporal filter",
                    "cube_cfg": str(runtime["cube_cfg"]),
                    "camera_matrix": runtime["camera_matrix"].tolist(),
                    "dist_coeffs": runtime["dist_coeffs"].tolist(),
                    "frame_count": int(len(offsets)),
                    "min_tags": int(args.min_tags),
                    "ransac_reproj": float(args.ransac_reproj),
                    "max_reproj": float(args.max_reproj),
                    "point_reject_px": float(args.point_reject_px),
                    "tag_reject_px": float(args.tag_reject_px),
                    "min_inlier_tag_fraction": float(args.min_inlier_tag_fraction),
                    "coverage_check_min_raw_tags": int(args.coverage_check_min_raw_tags),
                    "max_required_inlier_tags": int(args.max_required_inlier_tags),
                    "input_header": _jsonish(header),
                },
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        for out_idx, offset in enumerate(offsets):
            frame = load_at(input_pkl, offset)
            object_points, image_points, tag_ids, point_counts, dense_stats = dense_points_for_frame(
                frame,
                tag_corner_map=runtime["tag_corner_map"],
                min_tags=int(args.min_tags),
            )
            if object_points.shape[0] >= 4:
                pose = solve_dense_pose(
                    object_points,
                    image_points,
                    tag_ids,
                    point_counts,
                    cube_config=runtime["cube_config"],
                    face_id_sets=runtime["face_id_sets"],
                    camera_matrix=runtime["camera_matrix"],
                    dist_coeffs=runtime["dist_coeffs"],
                    ransac_reproj=float(args.ransac_reproj),
                    max_reproj=float(args.max_reproj),
                    point_reject_px=float(args.point_reject_px),
                    tag_reject_px=float(args.tag_reject_px),
                    min_tags=int(args.min_tags),
                    min_inlier_tag_fraction=float(args.min_inlier_tag_fraction),
                    coverage_check_min_raw_tags=int(args.coverage_check_min_raw_tags),
                    max_required_inlier_tags=int(args.max_required_inlier_tags),
                )
            else:
                pose = {
                    "success": False,
                    "failure_reason": str(dense_stats.get("reason", "dense_no_points")),
                    "reproj_error": float("inf"),
                    "n_tags": len(tag_ids),
                    "tag_ids": tag_ids,
                    "pose_source": "deeptag_dense_keypoints_all_point_pnp",
                    "pose_filled": False,
                }
            pose["dense_stats"] = {
                **dense_stats,
                "raw_tag_ids": tag_ids,
                "raw_point_counts": point_counts,
            }
            pose_sanitized = sanitize_pose(pose)
            success_count += int(bool(pose.get("success", False)))
            total_points += int(pose.get("n_points", 0) or 0)

            base = None
            if not args.no_source_overlay:
                base = source_detection_frame(
                    source_path,
                    source_offsets,
                    script012,
                    undistort_pack,
                    int(frame.get("source_offset", -1)),
                )
            if base is None:
                base = decode_jpeg_bgr(frame["overlay_jpeg"])
            overlay = draw_overlay(base, runtime, pose)
            frame_record = {
                "type": "frame",
                "frame_index": int(frame.get("frame_index", out_idx)),
                "source_offset": int(frame.get("source_offset", -1)),
                "loop_frame_idx": int(frame.get("loop_frame_idx", out_idx)),
                "capture_timestamp": frame.get("capture_timestamp", None),
                "pose": pose_sanitized,
                "dense_point_count": int(object_points.shape[0]),
                "overlay_jpeg": encode_bgr_jpeg(overlay, int(args.jpeg_quality)),
                "overlay_format": "jpeg_bgr",
                "cluster_stats": frame.get("cluster_stats", {}),
                "detection_stats": frame.get("detection_stats", {}),
            }
            pickle.dump(frame_record, f, protocol=pickle.HIGHEST_PROTOCOL)

            done = out_idx + 1
            if done == len(offsets) or done % 25 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"\r[INFO] dense pose {done}/{len(offsets)} "
                    f"success={success_count} fps={done / max(elapsed, 1e-9):.1f}",
                    end="",
                    flush=True,
                )

        pickle.dump(
            {
                "type": "footer",
                "frame_count": int(len(offsets)),
                "success_count": int(success_count),
                "avg_inlier_points": float(total_points / max(success_count, 1)),
                "stopped_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print()
    print(f"[INFO] Saved dense DeepTag pose pkl: {output_pkl}")
    print(f"[INFO] success={success_count}/{len(offsets)}")


if __name__ == "__main__":
    main()
