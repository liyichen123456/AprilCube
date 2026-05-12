from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aprilcube_pose_benchmark.common_pose import (
    cube_pose_from_tag_pose,
    detections_from_tags,
    is_valid_rotation_matrix,
    pnp_cube_pose_lm,
    reorder_pupil_corners_to_cube_order,
    reprojection_error,
    result_from_pose,
    tag_object_corners,
    visible_faces_for_ids,
)
from aprilcube_pose_benchmark.common_runner import BenchmarkConfig, result_from_pnp_tuple, run_benchmark


# =========================
# 中文参数区
# =========================

# 要评测的 002 录制文件。
PKL_PATH = "/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/logs_002/recording_20260511_162011.pkl"

# 结果保存根目录。脚本会自动创建 recording 名称和算法名称子目录。
OUTPUT_ROOT = "/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/outputs/aprilcube_pose_benchmark"

# 是否打开 viser 播放每帧结果。批量跑算法时建议设为 False。
ENABLE_VISER = True
VISER_HOST = "0.0.0.0"
VISER_PORT = 8098
PLAYBACK_FPS = 25.0
LOOP_PLAYBACK = True

# 检测前对灰度图做 CLAHE 增强，所有算法统一使用这组参数。
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# 单 tag IPPE 分支的 temporal disambiguation 评分：
# score = rotation_delta_deg + TRANSLATION_WEIGHT * translation_delta_mm + REPROJ_WEIGHT * reproj_px
TEMPORAL_TRANSLATION_WEIGHT_DEG_PER_MM = 1.0
TEMPORAL_REPROJ_WEIGHT = 1.0

ALGORITHM_NAME = "alg_08_hybrid_multiface_pnp_single_tag_ippe_temporal"


def rotation_angle_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    delta = np.asarray(rot_a, dtype=np.float64).reshape(3, 3).T @ np.asarray(rot_b, dtype=np.float64).reshape(3, 3)
    cos_angle = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def cube_delta_score(
    previous_rot: np.ndarray,
    previous_tvec: np.ndarray,
    candidate_rot: np.ndarray,
    candidate_tvec: np.ndarray,
    reproj_error_px: float | None,
) -> tuple[float, float, float]:
    rot_delta_deg = rotation_angle_deg(previous_rot, candidate_rot)
    trans_delta_mm = float(
        np.linalg.norm(
            np.asarray(candidate_tvec, dtype=np.float64).reshape(3)
            - np.asarray(previous_tvec, dtype=np.float64).reshape(3)
        )
    )
    reproj_term = 0.0 if reproj_error_px is None else float(reproj_error_px)
    score = (
        rot_delta_deg
        + float(TEMPORAL_TRANSLATION_WEIGHT_DEG_PER_MM) * trans_delta_mm
        + float(TEMPORAL_REPROJ_WEIGHT) * reproj_term
    )
    return score, rot_delta_deg, trans_delta_mm


def single_tag_ippe_cube_candidates(detector: Any, tag: Any) -> list[dict[str, Any]]:
    obj_pts = tag_object_corners(float(detector.config.tag_size_mm) / 1000.0)
    img_pts = reorder_pupil_corners_to_cube_order(np.asarray(tag.corners, dtype=np.float64).reshape(4, 2))
    k = np.asarray(detector.camera_matrix, dtype=np.float64)
    dist = np.asarray(detector.dist_coeffs, dtype=np.float64).reshape(-1, 1)
    try:
        retval, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
            obj_pts,
            img_pts,
            k,
            dist,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        retval, rvecs, tvecs, reproj_errs = 0, [], [], None
    if int(retval) <= 0:
        return []

    if reproj_errs is None:
        reproj_arr = np.full(len(rvecs), np.nan, dtype=np.float64)
    else:
        reproj_arr = np.asarray(reproj_errs, dtype=np.float64).reshape(-1)

    candidates: list[dict[str, Any]] = []
    tag_id = int(tag.tag_id)
    for idx, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        tag_rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        if not is_valid_rotation_matrix(tag_rot):
            continue
        tag_tvec_m = np.asarray(tvec, dtype=np.float64).reshape(3)
        if float(tag_tvec_m[2]) <= 0.0:
            continue
        tag_tvec_mm = tag_tvec_m.reshape(3, 1) * 1000.0
        cube_pose = cube_pose_from_tag_pose(detector, tag_id, tag_rot, tag_tvec_mm)
        if cube_pose is None:
            continue
        cube_rot, cube_tvec = cube_pose
        candidates.append(
            {
                "tag_id": tag_id,
                "candidate_index": int(idx),
                "rot_mat": cube_rot,
                "tvec": cube_tvec,
                "tag_rot_mat": tag_rot,
                "tag_tvec": tag_tvec_mm,
                "reproj_error": float(reproj_arr[idx])
                if idx < len(reproj_arr) and np.isfinite(reproj_arr[idx])
                else None,
            }
        )
    return candidates


def result_from_cube_candidate(
    detector: Any,
    detections: list[tuple[int, np.ndarray]],
    candidate: dict[str, Any],
    *,
    debug: dict[str, Any],
) -> dict[str, Any]:
    cube_rot = np.asarray(candidate["rot_mat"], dtype=np.float64).reshape(3, 3)
    cube_tvec = np.asarray(candidate["tvec"], dtype=np.float64).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(cube_rot)
    reproj = reprojection_error(detector, detections, rvec, cube_tvec)
    tag_id = int(candidate["tag_id"])
    tag_pose_by_id = {
        tag_id: {
            "rot_mat": np.asarray(candidate["tag_rot_mat"], dtype=np.float64).reshape(3, 3),
            "tvec": np.asarray(candidate["tag_tvec"], dtype=np.float64).reshape(3, 1),
            "reproj_error": candidate.get("reproj_error", None),
        }
    }
    return result_from_pose(
        detector,
        detections,
        rvec,
        cube_tvec,
        reproj_error=reproj,
        n_inliers=4,
        debug=debug,
        tag_pose_by_id=tag_pose_by_id,
    )


def store_last_pose(context: dict[str, Any], result: dict[str, Any]) -> None:
    if not result.get("success", False):
        return
    rvec = np.asarray(result["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(result["tvec"], dtype=np.float64).reshape(3, 1)
    rot, _ = cv2.Rodrigues(rvec)
    context["last_good_cube_pose"] = {
        "rot_mat": rot,
        "tvec": tvec,
        "mode": result.get("algorithm_debug", {}).get("mode", ""),
    }


def algorithm_fn(detector: Any, native_detector: Any, tags: list[Any], gray: np.ndarray, context: dict[str, Any]) -> dict[str, Any]:
    """多 face 用统一 cube PnP+LM；单 tag 用 IPPE 候选并通过上一帧 cube pose 消歧。"""
    detections = detections_from_tags(tags)
    tag_ids = [tag_id for tag_id, _corners in detections]
    faces = visible_faces_for_ids(detector, tag_ids)

    if len(faces) >= 2:
        pose = pnp_cube_pose_lm(detector, detections, use_ransac=False)
        result = result_from_pnp_tuple(detector, detections, pose)
        result["algorithm_debug"] = {
            "mode": "multiface_cube_pnp_lm",
            "visible_faces": sorted(faces),
            "num_tags": len(detections),
        }
        store_last_pose(context, result)
        return result

    if len(detections) != 1 or len(tags) != 1:
        result = result_from_pnp_tuple(detector, detections, None)
        result["visible_faces"] = faces
        result["algorithm_debug"] = {
            "mode": "no_pose_requires_multiface_or_single_tag",
            "visible_faces": sorted(faces),
            "num_tags": len(detections),
        }
        return result

    candidates = single_tag_ippe_cube_candidates(detector, tags[0])
    if not candidates:
        result = result_from_pnp_tuple(detector, detections, None)
        result["algorithm_debug"] = {
            "mode": "single_tag_no_ippe_candidates",
            "visible_faces": sorted(faces),
            "num_tags": len(detections),
        }
        return result

    previous = context.get("last_good_cube_pose", None)
    if previous is None:
        # No temporal reference: choose the best-reprojection branch, but keep it low-confidence
        # and do not seed last_good_cube_pose from it.
        best = min(
            candidates,
            key=lambda cand: float(cand["reproj_error"]) if cand.get("reproj_error", None) is not None else float("inf"),
        )
        debug = {
            "mode": "single_tag_ippe_no_temporal_reference",
            "confidence": "low",
            "raw_candidate_count": len(candidates),
            "selected_candidate_index": int(best.get("candidate_index", -1)),
            "visible_faces": sorted(faces),
        }
        return result_from_cube_candidate(detector, detections, best, debug=debug)

    previous_rot = np.asarray(previous["rot_mat"], dtype=np.float64).reshape(3, 3)
    previous_tvec = np.asarray(previous["tvec"], dtype=np.float64).reshape(3, 1)
    best = None
    best_score = float("inf")
    best_rot_delta = float("nan")
    best_trans_delta = float("nan")
    for candidate in candidates:
        score, rot_delta, trans_delta = cube_delta_score(
            previous_rot,
            previous_tvec,
            np.asarray(candidate["rot_mat"], dtype=np.float64).reshape(3, 3),
            np.asarray(candidate["tvec"], dtype=np.float64).reshape(3, 1),
            candidate.get("reproj_error", None),
        )
        if score < best_score:
            best = candidate
            best_score = score
            best_rot_delta = rot_delta
            best_trans_delta = trans_delta
    if best is None:
        result = result_from_pnp_tuple(detector, detections, None)
        result["algorithm_debug"] = {
            "mode": "single_tag_temporal_selection_failed",
            "visible_faces": sorted(faces),
            "num_tags": len(detections),
        }
        return result

    debug = {
        "mode": "single_tag_ippe_temporal_disambiguation",
        "confidence": "medium",
        "raw_candidate_count": len(candidates),
        "selected_candidate_index": int(best.get("candidate_index", -1)),
        "temporal_score": float(best_score),
        "temporal_rotation_delta_deg": float(best_rot_delta),
        "temporal_translation_delta_mm": float(best_trans_delta),
        "previous_mode": str(previous.get("mode", "")),
        "visible_faces": sorted(faces),
    }
    result = result_from_cube_candidate(detector, detections, best, debug=debug)
    store_last_pose(context, result)
    return result


CONFIG = BenchmarkConfig(
    algorithm_name=ALGORITHM_NAME,
    pkl_path=PKL_PATH,
    output_root=OUTPUT_ROOT,
    enable_viser=ENABLE_VISER,
    viser_host=VISER_HOST,
    viser_port=VISER_PORT,
    playback_fps=PLAYBACK_FPS,
    loop_playback=LOOP_PLAYBACK,
    clahe_clip_limit=CLAHE_CLIP_LIMIT,
    clahe_tile_grid_size=CLAHE_TILE_GRID_SIZE,
    estimate_tag_pose=False,
)


def main() -> None:
    run_benchmark(CONFIG, algorithm_fn)


if __name__ == "__main__":
    main()
