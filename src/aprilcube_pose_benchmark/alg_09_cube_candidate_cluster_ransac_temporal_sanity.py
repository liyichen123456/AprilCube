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
    average_rotations,
    cube_pose_from_tag_pose,
    detections_from_tags,
    empty_result,
    is_valid_rotation_matrix,
    pnp_cube_pose_lm,
    reorder_pupil_corners_to_cube_order,
    reprojection_error,
    result_from_pose,
    tag_object_corners,
    visible_faces_for_ids,
)
from aprilcube_pose_benchmark.common_runner import BenchmarkConfig, run_benchmark


# =========================
# 中文参数区
# =========================

PKL_PATH = "/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/logs_002/recording_20260511_162011.pkl"
OUTPUT_ROOT = "/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/outputs/aprilcube_pose_benchmark"

ENABLE_VISER = True
VISER_HOST = "0.0.0.0"
VISER_PORT = 8099
PLAYBACK_FPS = 25.0
LOOP_PLAYBACK = True

CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID_SIZE = (8, 8)

CLUSTER_ROTATION_THRESH_DEG = 35.0
CLUSTER_TRANSLATION_THRESH_MM = 25.0
TEMPORAL_ROTATION_GATE_DEG = 90.0
TEMPORAL_TRANSLATION_GATE_MM = 25.0
TEMPORAL_TRANSLATION_WEIGHT_DEG_PER_MM = 1.0
TEMPORAL_REPROJ_WEIGHT = 1.0

ALGORITHM_NAME = "alg_09_cube_candidate_cluster_ransac_temporal_sanity"


def rotation_angle_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    delta = np.asarray(rot_a, dtype=np.float64).reshape(3, 3).T @ np.asarray(rot_b, dtype=np.float64).reshape(3, 3)
    cos_angle = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def make_failure_result(
    detector: Any,
    detections: list[tuple[int, np.ndarray]],
    *,
    debug: dict[str, Any],
) -> dict[str, Any]:
    result = empty_result()
    result["detections"] = detections
    result["n_tags"] = len(detections)
    result["tag_ids"] = [int(tag_id) for tag_id, _corners in detections]
    result["visible_faces"] = visible_faces_for_ids(detector, result["tag_ids"])
    result["algorithm_debug"] = debug
    return result


def make_hold_result(
    detector: Any,
    detections: list[tuple[int, np.ndarray]],
    last_good: dict[str, Any] | None,
    *,
    debug: dict[str, Any],
) -> dict[str, Any]:
    if last_good is None:
        return make_failure_result(detector, detections, debug={**debug, "hold_available": False})

    rot = np.asarray(last_good["rot_mat"], dtype=np.float64).reshape(3, 3)
    tvec = np.asarray(last_good["tvec"], dtype=np.float64).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(rot)
    result = result_from_pose(
        detector,
        detections,
        rvec,
        tvec,
        reproj_error=float("nan"),
        n_inliers=0,
        debug={**debug, "hold_available": True, "held_previous_mode": last_good.get("mode", "")},
    )
    result["predicted"] = True
    return result


def store_last_good(context: dict[str, Any], result: dict[str, Any]) -> None:
    if not result.get("success", False) or result.get("predicted", False):
        return
    rot, _ = cv2.Rodrigues(np.asarray(result["rvec"], dtype=np.float64).reshape(3, 1))
    context["last_good_pose"] = {
        "rot_mat": rot,
        "tvec": np.asarray(result["tvec"], dtype=np.float64).reshape(3, 1),
        "mode": result.get("algorithm_debug", {}).get("mode", ""),
    }


def temporal_delta(candidate: dict[str, Any], last_good: dict[str, Any]) -> tuple[float, float, float]:
    rot = np.asarray(candidate["rot_mat"], dtype=np.float64).reshape(3, 3)
    tvec = np.asarray(candidate["tvec"], dtype=np.float64).reshape(3, 1)
    prev_rot = np.asarray(last_good["rot_mat"], dtype=np.float64).reshape(3, 3)
    prev_tvec = np.asarray(last_good["tvec"], dtype=np.float64).reshape(3, 1)
    rot_delta = rotation_angle_deg(prev_rot, rot)
    trans_delta = float(np.linalg.norm(tvec.reshape(3) - prev_tvec.reshape(3)))
    reproj = candidate.get("reproj_error", None)
    reproj_term = 0.0 if reproj is None or not np.isfinite(float(reproj)) else float(reproj)
    score = (
        rot_delta
        + float(TEMPORAL_TRANSLATION_WEIGHT_DEG_PER_MM) * trans_delta
        + float(TEMPORAL_REPROJ_WEIGHT) * reproj_term
    )
    return score, rot_delta, trans_delta


def passes_temporal_sanity(candidate: dict[str, Any], last_good: dict[str, Any]) -> bool:
    _score, rot_delta, trans_delta = temporal_delta(candidate, last_good)
    candidate["temporal_rotation_delta_deg"] = rot_delta
    candidate["temporal_translation_delta_mm"] = trans_delta
    return (
        rot_delta <= float(TEMPORAL_ROTATION_GATE_DEG)
        and trans_delta <= float(TEMPORAL_TRANSLATION_GATE_MM)
    )


def ippe_cube_candidates(detector: Any, tags: list[Any]) -> list[dict[str, Any]]:
    obj_pts = tag_object_corners(float(detector.config.tag_size_mm) / 1000.0)
    k = np.asarray(detector.camera_matrix, dtype=np.float64)
    dist = np.asarray(detector.dist_coeffs, dtype=np.float64).reshape(-1, 1)
    candidates: list[dict[str, Any]] = []

    for tag in tags:
        tag_id = int(tag.tag_id)
        img_pts = reorder_pupil_corners_to_cube_order(np.asarray(tag.corners, dtype=np.float64).reshape(4, 2))
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
            continue

        reproj_arr = (
            np.asarray(reproj_errs, dtype=np.float64).reshape(-1)
            if reproj_errs is not None
            else np.full(len(rvecs), np.nan, dtype=np.float64)
        )
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
                    "source": "ippe",
                    "tag_id": tag_id,
                    "support_tag_ids": {tag_id},
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


def ransac_cube_candidate(
    detector: Any,
    detections: list[tuple[int, np.ndarray]],
) -> dict[str, Any] | None:
    if len(detections) < 2:
        return None
    pose = pnp_cube_pose_lm(detector, detections, use_ransac=True)
    if pose is None:
        return None
    rvec, tvec, reproj, n_inliers = pose
    rot, _ = cv2.Rodrigues(rvec)
    return {
        "source": "cube_pnp_ransac_lm",
        "support_tag_ids": {int(tag_id) for tag_id, _corners in detections},
        "rot_mat": rot,
        "tvec": np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        "reproj_error": float(reproj),
        "n_inliers": int(n_inliers),
    }


def cluster_candidates(candidates: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    for anchor in candidates:
        anchor_rot = np.asarray(anchor["rot_mat"], dtype=np.float64).reshape(3, 3)
        anchor_tvec = np.asarray(anchor["tvec"], dtype=np.float64).reshape(3, 1)
        support = []
        for cand in candidates:
            rot_delta = rotation_angle_deg(anchor_rot, np.asarray(cand["rot_mat"], dtype=np.float64).reshape(3, 3))
            trans_delta = float(
                np.linalg.norm(
                    np.asarray(cand["tvec"], dtype=np.float64).reshape(3)
                    - anchor_tvec.reshape(3)
                )
            )
            if (
                rot_delta <= float(CLUSTER_ROTATION_THRESH_DEG)
                and trans_delta <= float(CLUSTER_TRANSLATION_THRESH_MM)
            ):
                support.append(cand)
        clusters.append(support)
    return clusters


def candidate_support_tags(candidate: dict[str, Any]) -> set[int]:
    return {int(tag_id) for tag_id in candidate.get("support_tag_ids", set())}


def cluster_support_tags(cluster: list[dict[str, Any]]) -> set[int]:
    tags: set[int] = set()
    for candidate in cluster:
        tags.update(candidate_support_tags(candidate))
    return tags


def select_largest_cluster(clusters: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if not clusters:
        return []

    def mean_reproj(cluster: list[dict[str, Any]]) -> float:
        values = [
            float(candidate["reproj_error"])
            for candidate in cluster
            if candidate.get("reproj_error", None) is not None
            and np.isfinite(float(candidate["reproj_error"]))
        ]
        return float(np.mean(values)) if values else float("inf")

    return max(
        clusters,
        key=lambda cluster: (
            len(cluster_support_tags(cluster)),
            len(cluster),
            -mean_reproj(cluster),
        ),
    )


def fuse_cluster(cluster: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not cluster:
        return None
    weights = []
    rot_mats = []
    tvecs = []
    for candidate in cluster:
        err = candidate.get("reproj_error", None)
        weight = 1.0 if err is None or not np.isfinite(float(err)) else 1.0 / max(float(err), 1e-3)
        weights.append(weight)
        rot_mats.append(np.asarray(candidate["rot_mat"], dtype=np.float64).reshape(3, 3))
        tvecs.append(np.asarray(candidate["tvec"], dtype=np.float64).reshape(3, 1))
    weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr /= max(float(np.sum(weights_arr)), 1e-12)
    rot_avg = average_rotations(rot_mats, weights_arr)
    if rot_avg is None:
        return None
    t_avg = np.zeros((3, 1), dtype=np.float64)
    for weight, tvec in zip(weights_arr, tvecs):
        t_avg += float(weight) * tvec
    reproj_values = [
        float(candidate["reproj_error"])
        for candidate in cluster
        if candidate.get("reproj_error", None) is not None
        and np.isfinite(float(candidate["reproj_error"]))
    ]
    return {
        "source": "cluster_fused",
        "support_tag_ids": cluster_support_tags(cluster),
        "rot_mat": rot_avg,
        "tvec": t_avg,
        "cluster_size": len(cluster),
        "cluster_sources": [str(candidate.get("source", "")) for candidate in cluster],
        "reproj_error": float(np.mean(reproj_values)) if reproj_values else None,
    }


def result_from_cube_candidate(
    detector: Any,
    detections: list[tuple[int, np.ndarray]],
    candidate: dict[str, Any],
    *,
    debug: dict[str, Any],
) -> dict[str, Any]:
    rot = np.asarray(candidate["rot_mat"], dtype=np.float64).reshape(3, 3)
    tvec = np.asarray(candidate["tvec"], dtype=np.float64).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(rot)
    reproj = reprojection_error(detector, detections, rvec, tvec)
    result = result_from_pose(
        detector,
        detections,
        rvec,
        tvec,
        reproj_error=reproj,
        n_inliers=4 * len(candidate_support_tags(candidate)),
        debug=debug,
    )
    return result


def choose_single_tag_candidate(
    candidates: list[dict[str, Any]],
    last_good: dict[str, Any],
) -> dict[str, Any] | None:
    best = None
    best_score = float("inf")
    for candidate in candidates:
        score, rot_delta, trans_delta = temporal_delta(candidate, last_good)
        candidate["temporal_score"] = score
        candidate["temporal_rotation_delta_deg"] = rot_delta
        candidate["temporal_translation_delta_mm"] = trans_delta
        if score < best_score:
            best = candidate
            best_score = score
    return best


def select_candidate(
    detector: Any,
    detections: list[tuple[int, np.ndarray]],
    raw_candidates: list[dict[str, Any]],
    last_good: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    distinct_tags = {int(tag_id) for tag_id, _corners in detections}
    debug: dict[str, Any] = {
        "raw_candidate_count": len(raw_candidates),
        "visible_faces": sorted(visible_faces_for_ids(detector, list(distinct_tags))),
        "num_tags": len(detections),
        "has_last_good": last_good is not None,
        "temporal_rotation_gate_deg": float(TEMPORAL_ROTATION_GATE_DEG),
        "temporal_translation_gate_mm": float(TEMPORAL_TRANSLATION_GATE_MM),
    }
    if not raw_candidates:
        debug["mode"] = "no_candidates"
        return None, debug

    candidates = raw_candidates
    if last_good is not None:
        candidates = [candidate for candidate in raw_candidates if passes_temporal_sanity(candidate, last_good)]
        debug["temporal_passed_candidate_count"] = len(candidates)
        if not candidates:
            debug["mode"] = "temporal_sanity_rejected_all"
            return None, debug

    if len(distinct_tags) <= 1:
        debug["mode"] = "single_tag_ippe_temporal_branch_select"
        if last_good is None:
            debug["reject_reason"] = "single_tag_requires_last_good"
            return None, debug
        best = choose_single_tag_candidate(candidates, last_good)
        if best is None:
            return None, debug
        debug.update(
            {
                "selected_source": str(best.get("source", "")),
                "selected_candidate_index": int(best.get("candidate_index", -1)),
                "temporal_score": float(best.get("temporal_score", float("nan"))),
                "temporal_rotation_delta_deg": float(best.get("temporal_rotation_delta_deg", float("nan"))),
                "temporal_translation_delta_mm": float(best.get("temporal_translation_delta_mm", float("nan"))),
            }
        )
        return best, debug

    clusters = cluster_candidates(candidates)
    selected_cluster = select_largest_cluster(clusters)
    fused = fuse_cluster(selected_cluster)
    debug.update(
        {
            "mode": "cluster_largest_consistent",
            "cluster_count": len(clusters),
            "selected_cluster_size": len(selected_cluster),
            "selected_cluster_support_tags": sorted(cluster_support_tags(selected_cluster)),
            "selected_cluster_sources": [str(candidate.get("source", "")) for candidate in selected_cluster],
        }
    )
    if fused is None:
        debug["mode"] = "cluster_fuse_failed"
        return None, debug

    if last_good is not None and not passes_temporal_sanity(fused, last_good):
        debug.update(
            {
                "mode": "cluster_fused_temporal_sanity_rejected",
                "temporal_rotation_delta_deg": float(fused.get("temporal_rotation_delta_deg", float("nan"))),
                "temporal_translation_delta_mm": float(fused.get("temporal_translation_delta_mm", float("nan"))),
            }
        )
        return None, debug

    if last_good is None and len(candidate_support_tags(fused)) < 2:
        debug["mode"] = "cluster_bootstrap_requires_two_tags"
        return None, debug

    if last_good is not None:
        score, rot_delta, trans_delta = temporal_delta(fused, last_good)
        debug.update(
            {
                "temporal_score": float(score),
                "temporal_rotation_delta_deg": float(rot_delta),
                "temporal_translation_delta_mm": float(trans_delta),
            }
        )
    return fused, debug


def algorithm_fn(detector: Any, native_detector: Any, tags: list[Any], gray: np.ndarray, context: dict[str, Any]) -> dict[str, Any]:
    detections = detections_from_tags(tags)
    last_good = context.get("last_good_pose", None)

    raw_candidates = ippe_cube_candidates(detector, tags)
    ransac_candidate = ransac_cube_candidate(detector, detections)
    if ransac_candidate is not None:
        raw_candidates.append(ransac_candidate)

    selected, debug = select_candidate(detector, detections, raw_candidates, last_good)
    if selected is None:
        result = make_hold_result(
            detector,
            detections,
            last_good,
            debug={**debug, "mode": f"hold_last_good_after_{debug.get('mode', 'unknown')}"},
        )
        return result

    result = result_from_cube_candidate(detector, detections, selected, debug=debug)
    store_last_good(context, result)
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
