#!/usr/bin/env python3
"""Benchmark DeepTag internal-grid cube pose estimation on a 021 recording."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


APRILCUBE_ROOT = Path(__file__).resolve().parent.parent
DEEPTAG_ROOT = APRILCUBE_ROOT / "thirdparty/deeptag-pytorch"
sys.path.insert(0, str(APRILCUBE_ROOT / "src"))
sys.path.insert(0, str(DEEPTAG_ROOT))

from aprilcube.detect import build_tag_corner_map, load_cube_config  # noqa: E402
from deeptag_model_setting import load_deeptag_models  # noqa: E402
from marker_dict_setting import load_marker_codebook  # noqa: E402
from stag_decode.detection_engine import DetectionEngine  # noqa: E402


CAMERA_NAME = "middle_finger_cam"
UNDISTORTED_IMAGE_FIELD = "undistorted_image_jpeg"
HOMOGRAPHY_RANSAC_PX = 2.5
TAG_SUPPORT_MEDIAN_PX = 6.0
MAX_ACCEPTED_MEDIAN_PX = 5.0
MAX_ACCEPTED_P90_PX = 9.0
MIN_DETECTED_POINTS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DeepTag on embedded undistorted frames, estimate cube poses "
            "from measured internal grid points, and compare with offline_pos."
        )
    )
    parser.add_argument("pkl_path", type=Path)
    parser.add_argument("--camera", default=CAMERA_NAME)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose-deeptag", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def pose_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def pose_vectors(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rvec = cv2.Rodrigues(transform[:3, :3])[0].reshape(3, 1)
    tvec = transform[:3, 3].reshape(3, 1)
    return rvec, tvec


def local_tag_to_cube_transform(tag_corners: np.ndarray) -> np.ndarray:
    """Return cube_from_tag for DeepTag's x-right, y-up local frame."""
    top_left, _top_right, bottom_right, bottom_left = np.asarray(
        tag_corners, dtype=np.float64
    )
    tag_size = float(np.linalg.norm(bottom_right - bottom_left))
    x_axis = (bottom_right - bottom_left) / tag_size
    y_axis = (top_left - bottom_left) / tag_size
    z_axis = np.cross(x_axis, y_axis)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    transform[:3, 3] = np.asarray(tag_corners, dtype=np.float64).mean(axis=0)
    return transform


def deep_points_to_cube(
    normalized_points: np.ndarray,
    tag_corners: np.ndarray,
) -> np.ndarray:
    """Map DeepTag normalized x-right/y-up points onto a cube tag face."""
    points = np.asarray(normalized_points, dtype=np.float64)
    u = points[:, 0] + 0.5
    v = 0.5 - points[:, 1]
    top_left, top_right, bottom_right, bottom_left = np.asarray(
        tag_corners, dtype=np.float64
    )
    return (
        ((1.0 - u) * (1.0 - v))[:, None] * top_left
        + (u * (1.0 - v))[:, None] * top_right
        + (u * v)[:, None] * bottom_right
        + ((1.0 - u) * v)[:, None] * bottom_left
    )


def reprojection_errors(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    projected = cv2.projectPoints(
        np.asarray(object_points, dtype=np.float64),
        rvec,
        tvec,
        camera_matrix,
        np.zeros(5, dtype=np.float64),
    )[0].reshape(-1, 2)
    return np.linalg.norm(projected - image_points, axis=1)


def robust_tag_candidates(
    decoded_tag: dict[str, Any],
    normalized_points: np.ndarray,
    tag_corners: np.ndarray,
    camera_matrix: np.ndarray,
) -> list[dict[str, Any]]:
    image_points_all = np.asarray(
        decoded_tag["keypoints_in_images"], dtype=np.float64
    ).reshape(-1, 2)
    detected_flags = np.asarray(
        decoded_tag.get("keypoints_detected_flags", [True] * len(image_points_all)),
        dtype=bool,
    )
    if int(detected_flags.sum()) < MIN_DETECTED_POINTS:
        return []

    normalized = np.asarray(normalized_points, dtype=np.float64)[detected_flags]
    image_points = image_points_all[detected_flags]
    _homography, homography_mask = cv2.findHomography(
        normalized,
        image_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=HOMOGRAPHY_RANSAC_PX,
        maxIters=2000,
        confidence=0.999,
    )
    if homography_mask is None:
        return []
    homography_inliers = homography_mask.reshape(-1).astype(bool)
    if int(homography_inliers.sum()) < MIN_DETECTED_POINTS:
        return []

    tag_size_mm = float(np.linalg.norm(tag_corners[1] - tag_corners[0]))
    local_object_points = np.column_stack(
        (normalized * tag_size_mm, np.zeros(len(normalized), dtype=np.float64))
    )
    inlier_object = local_object_points[homography_inliers]
    inlier_image = image_points[homography_inliers]
    result = cv2.solvePnPGeneric(
        inlier_object,
        inlier_image,
        camera_matrix,
        np.zeros(5, dtype=np.float64),
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not result[0]:
        return []

    cube_from_tag = local_tag_to_cube_transform(tag_corners)
    cube_points = deep_points_to_cube(normalized, tag_corners)
    candidates: list[dict[str, Any]] = []
    for initial_rvec, initial_tvec in zip(result[1], result[2]):
        if float(np.asarray(initial_tvec).reshape(3)[2]) <= 0.0:
            continue
        rvec, tvec = cv2.solvePnPRefineLM(
            inlier_object,
            inlier_image,
            camera_matrix,
            np.zeros(5, dtype=np.float64),
            np.asarray(initial_rvec, dtype=np.float64),
            np.asarray(initial_tvec, dtype=np.float64),
        )
        initial_errors = reprojection_errors(
            local_object_points,
            image_points,
            rvec,
            tvec,
            camera_matrix,
        )
        median = float(np.median(initial_errors[homography_inliers]))
        mad = float(
            np.median(np.abs(initial_errors[homography_inliers] - median))
        )
        trim_threshold = min(5.0, max(1.5, median + 3.0 * 1.4826 * mad))
        robust_inliers = homography_inliers & (initial_errors <= trim_threshold)
        if int(robust_inliers.sum()) < MIN_DETECTED_POINTS:
            continue
        rvec, tvec = cv2.solvePnPRefineLM(
            local_object_points[robust_inliers],
            image_points[robust_inliers],
            camera_matrix,
            np.zeros(5, dtype=np.float64),
            rvec,
            tvec,
        )
        if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
            continue

        errors = reprojection_errors(
            local_object_points,
            image_points,
            rvec,
            tvec,
            camera_matrix,
        )
        if not np.all(np.isfinite(errors[robust_inliers])):
            continue
        camera_from_tag = pose_matrix(rvec, tvec)
        camera_from_cube = camera_from_tag @ np.linalg.inv(cube_from_tag)
        cube_rvec, cube_tvec = pose_vectors(camera_from_cube)
        candidates.append(
            {
                "tag_id": int(decoded_tag["tag_id"]),
                "rvec": cube_rvec,
                "tvec": cube_tvec,
                "object_points": cube_points,
                "image_points": image_points,
                "point_inliers": robust_inliers,
                "detected_points": int(detected_flags.sum()),
                "inlier_points": int(robust_inliers.sum()),
                "median_px": float(np.median(errors[robust_inliers])),
                "p90_px": float(np.percentile(errors[robust_inliers], 90)),
                "mean_px": float(np.mean(errors[robust_inliers])),
                "score": float(decoded_tag.get("score", 0.0)),
            }
        )
    return candidates


def choose_and_refine_cube_pose(
    candidates_by_tag: dict[int, list[dict[str, Any]]],
    camera_matrix: np.ndarray,
    previous_transform: np.ndarray | None,
) -> dict[str, Any] | None:
    hypotheses = [
        candidate
        for candidates in candidates_by_tag.values()
        for candidate in candidates
        if candidate["median_px"] <= MAX_ACCEPTED_MEDIAN_PX
        and candidate["p90_px"] <= MAX_ACCEPTED_P90_PX
    ]
    if not hypotheses:
        return None

    scored: list[tuple[tuple[float, ...], dict[str, Any], list[int]]] = []
    for hypothesis in hypotheses:
        tag_medians: dict[int, float] = {}
        for tag_id, candidates in candidates_by_tag.items():
            best_tag_median = float("inf")
            for candidate in candidates:
                errors = reprojection_errors(
                    candidate["object_points"],
                    candidate["image_points"],
                    hypothesis["rvec"],
                    hypothesis["tvec"],
                    camera_matrix,
                )
                inlier_errors = errors[candidate["point_inliers"]]
                if len(inlier_errors):
                    best_tag_median = min(
                        best_tag_median,
                        float(np.median(inlier_errors)),
                    )
            tag_medians[tag_id] = best_tag_median
        supported_tags = [
            tag_id
            for tag_id, median in tag_medians.items()
            if median <= TAG_SUPPORT_MEDIAN_PX
        ]
        if not supported_tags:
            continue
        transform = pose_matrix(hypothesis["rvec"], hypothesis["tvec"])
        if previous_transform is None:
            motion_score = 0.0
        else:
            translation_delta = float(
                np.linalg.norm(transform[:3, 3] - previous_transform[:3, 3])
            )
            rotation_delta = rotation_angle_deg(
                transform[:3, :3] @ previous_transform[:3, :3].T
            )
            motion_score = translation_delta + 0.15 * rotation_delta
        support_median = float(
            np.median([tag_medians[tag_id] for tag_id in supported_tags])
        )
        if previous_transform is None:
            score = (
                -float(len(supported_tags)),
                -float(hypothesis["inlier_points"]),
                support_median,
                float(hypothesis["median_px"]),
            )
        else:
            # Planar IPPE candidates can have nearly indistinguishable pixel
            # errors while representing opposite poses. Once both pass the
            # quality gates, continuity is the stronger disambiguation signal.
            score = (
                -float(len(supported_tags)),
                motion_score,
                support_median,
                float(hypothesis["median_px"]),
            )
        scored.append((score, hypothesis, supported_tags))

    if not scored:
        return None
    _score, selected, supported_tags = min(scored, key=lambda item: item[0])
    object_blocks = []
    image_blocks = []
    for tag_id in supported_tags:
        candidate = min(candidates_by_tag[tag_id], key=lambda item: item["median_px"])
        mask = candidate["point_inliers"]
        object_blocks.append(candidate["object_points"][mask])
        image_blocks.append(candidate["image_points"][mask])

    object_points = np.concatenate(object_blocks, axis=0)
    image_points = np.concatenate(image_blocks, axis=0)
    rvec, tvec = cv2.solvePnPRefineLM(
        object_points,
        image_points,
        camera_matrix,
        np.zeros(5, dtype=np.float64),
        selected["rvec"],
        selected["tvec"],
    )
    errors = reprojection_errors(
        object_points,
        image_points,
        rvec,
        tvec,
        camera_matrix,
    )
    median = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median)))
    threshold = min(6.0, max(1.5, median + 3.0 * 1.4826 * mad))
    inliers = errors <= threshold
    if int(inliers.sum()) >= MIN_DETECTED_POINTS:
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points[inliers],
            image_points[inliers],
            camera_matrix,
            np.zeros(5, dtype=np.float64),
            rvec,
            tvec,
        )
        errors = reprojection_errors(
            object_points,
            image_points,
            rvec,
            tvec,
            camera_matrix,
        )

    return {
        "rvec": rvec,
        "tvec": tvec,
        "seed_tag_id": int(selected["tag_id"]),
        "supported_tag_ids": [int(tag_id) for tag_id in supported_tags],
        "detected_tag_ids": [int(tag_id) for tag_id in candidates_by_tag],
        "point_count": int(len(object_points)),
        "inlier_count": int(inliers.sum()),
        "median_px": float(np.median(errors[inliers])),
        "mean_px": float(np.mean(errors[inliers])),
        "p90_px": float(np.percentile(errors[inliers], 90)),
    }


def find_old_cube_result(
    camera_record: dict[str, Any],
    cube_name: str,
) -> dict[str, Any]:
    for cube in camera_record.get("offline_pos", {}).get("cube_results", []):
        if cube.get("cube_name") == cube_name:
            return cube.get("result", {})
    return {}


def main() -> None:
    args = parse_args()
    pkl_path = args.pkl_path.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else APRILCUBE_ROOT
        / "outputs/deeptag_internal_points"
        / f"{pkl_path.stem}_{args.camera}.json"
    )

    with pkl_path.open("rb") as file:
        header = pickle.load(file)
    cube_paths = header.get("metadata", {}).get("camera_cube_configs", {}).get(
        args.camera, []
    )
    if len(cube_paths) != 1:
        raise ValueError(
            f"Expected exactly one cube config for {args.camera}, got {cube_paths}"
        )
    cube_path = Path(cube_paths[0]).expanduser().resolve()
    cube_config, _face_ids = load_cube_config(str(cube_path / "config.json"))
    tag_corner_map = build_tag_corner_map(cube_config)
    target_tag_ids = set(tag_corner_map)

    previous_cwd = Path.cwd()
    try:
        os.chdir(DEEPTAG_ROOT)
        device = "cpu" if args.cpu else None
        model_detector, model_decoder, device, tag_type, grids = load_deeptag_models(
            "apriltag", device
        )
        codebook = load_marker_codebook(
            str(DEEPTAG_ROOT / "codebook/apriltag_codebook.txt"), tag_type
        )
    finally:
        os.chdir(previous_cwd)

    engine: DetectionEngine | None = None
    camera_matrix: np.ndarray | None = None
    previous_transform: np.ndarray | None = None
    frame_results: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    translation_differences: list[float] = []
    rotation_differences: list[float] = []
    started = time.perf_counter()

    with pkl_path.open("rb") as file:
        pickle.load(file)
        frame_index = 0
        while args.max_frames is None or frame_index < args.max_frames:
            try:
                record = pickle.load(file)
            except EOFError:
                break
            if not isinstance(record, dict) or record.get("type") != "frame_pair":
                continue
            camera_record = record.get("cameras", {}).get(args.camera)
            if not isinstance(camera_record, dict):
                frame_index += 1
                continue
            encoded = camera_record.get(UNDISTORTED_IMAGE_FIELD)
            visualization_meta = camera_record.get("undistorted_visualization_meta", {})
            if encoded is None or "detection_camera_matrix" not in visualization_meta:
                raise ValueError(
                    "PKL must contain embedded undistorted images and their camera matrix"
                )
            image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
            if camera_matrix is None:
                camera_matrix = np.asarray(
                    visualization_meta["detection_camera_matrix"], dtype=np.float64
                ).reshape(3, 3)
                engine = DetectionEngine(
                    model_detector,
                    model_decoder,
                    device,
                    tag_type,
                    grids,
                    stg2_iter_num=2,
                    min_center_score=0.2,
                    min_corner_score=0.2,
                    batch_size_stg2=4,
                    hamming_dist=8,
                    cameraMatrix=camera_matrix,
                    distCoeffs=np.zeros(8, dtype=np.float64),
                    codebook=codebook,
                    tag_real_size_in_meter_dict={-1: cube_config.tag_size_mm / 1000.0},
                )
            assert engine is not None and camera_matrix is not None

            if args.verbose_deeptag:
                decoded_tags = engine.process(image, detect_scale=None)
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    decoded_tags = engine.process(image, detect_scale=None)

            candidates_by_tag: dict[int, list[dict[str, Any]]] = {}
            raw_valid_ids: list[int] = []
            for decoded_tag in decoded_tags:
                if not decoded_tag.get("is_valid", False):
                    continue
                tag_id = int(decoded_tag.get("tag_id", -1))
                raw_valid_ids.append(tag_id)
                if tag_id not in target_tag_ids:
                    continue
                normalized_points = np.asarray(
                    engine.pose_solver_dict[
                        len(decoded_tag["keypoints_in_images"])
                    ].fine_grid_points_anno,
                    dtype=np.float64,
                )
                candidates = robust_tag_candidates(
                    decoded_tag,
                    normalized_points,
                    tag_corner_map[tag_id],
                    camera_matrix,
                )
                if candidates:
                    candidates_by_tag.setdefault(tag_id, []).extend(candidates)

            deep_result = choose_and_refine_cube_pose(
                candidates_by_tag,
                camera_matrix,
                previous_transform,
            )
            old_result = find_old_cube_result(camera_record, cube_path.name)
            old_success = bool(old_result.get("success", False))
            deep_success = deep_result is not None
            counters["frames"] += 1
            counters["old_success"] += int(old_success)
            counters["deeptag_success"] += int(deep_success)
            counters["both_success"] += int(old_success and deep_success)
            counters["deeptag_recovered_old_failure"] += int(
                deep_success and not old_success
            )
            counters["deeptag_missed_old_success"] += int(
                old_success and not deep_success
            )

            comparison: dict[str, float] = {}
            if deep_result is not None:
                previous_transform = pose_matrix(
                    deep_result["rvec"], deep_result["tvec"]
                )
                if old_success:
                    old_transform = pose_matrix(
                        old_result["rvec"], old_result["tvec"]
                    )
                    translation_difference = float(
                        np.linalg.norm(
                            previous_transform[:3, 3] - old_transform[:3, 3]
                        )
                    )
                    rotation_difference = rotation_angle_deg(
                        previous_transform[:3, :3]
                        @ old_transform[:3, :3].T
                    )
                    translation_differences.append(translation_difference)
                    rotation_differences.append(rotation_difference)
                    comparison = {
                        "translation_difference_mm": translation_difference,
                        "rotation_difference_deg": rotation_difference,
                    }

            frame_results.append(
                {
                    "frame_index": frame_index,
                    "old_success": old_success,
                    "old_failure_reason": old_result.get("failure_reason", ""),
                    "deeptag_success": deep_success,
                    "raw_deeptag_ids": raw_valid_ids,
                    "deeptag_result": (
                        None
                        if deep_result is None
                        else {
                            key: value
                            for key, value in deep_result.items()
                            if key not in {"rvec", "tvec"}
                        }
                        | {
                            "rvec": np.asarray(deep_result["rvec"])
                            .reshape(3)
                            .tolist(),
                            "tvec_mm": np.asarray(deep_result["tvec"])
                            .reshape(3)
                            .tolist(),
                        }
                    ),
                    "comparison": comparison,
                }
            )
            frame_index += 1
            if frame_index % 25 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"[INFO] frames={frame_index} "
                    f"deeptag={counters['deeptag_success']} "
                    f"recovered={counters['deeptag_recovered_old_failure']} "
                    f"speed={frame_index / max(elapsed, 1e-9):.2f} frame/s"
                )

    def distribution(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "p90": float(np.percentile(array, 90)),
            "max": float(np.max(array)),
        }

    successful_deep_frames = [
        frame for frame in frame_results if frame["deeptag_success"]
    ]
    recovered_frames = [
        frame
        for frame in successful_deep_frames
        if not frame["old_success"]
    ]
    both_successful_frames = [
        frame
        for frame in successful_deep_frames
        if frame["old_success"]
    ]
    recovered_reasons = Counter(
        str(frame["old_failure_reason"]).split(":", 1)[0]
        for frame in recovered_frames
    )
    agreement_thresholds = {}
    for translation_mm, rotation_deg in ((2.0, 5.0), (5.0, 10.0), (10.0, 20.0)):
        count = sum(
            frame["comparison"]["translation_difference_mm"] <= translation_mm
            and frame["comparison"]["rotation_difference_deg"] <= rotation_deg
            for frame in both_successful_frames
        )
        agreement_thresholds[f"within_{translation_mm:g}mm_{rotation_deg:g}deg"] = {
            "count": int(count),
            "fraction": count / max(len(both_successful_frames), 1),
        }
    summary = {
        **dict(counters),
        "deeptag_success_rate": counters["deeptag_success"]
        / max(counters["frames"], 1),
        "old_success_rate": counters["old_success"] / max(counters["frames"], 1),
        "translation_difference_mm": distribution(translation_differences),
        "rotation_difference_deg": distribution(rotation_differences),
        "agreement_with_old_pose": agreement_thresholds,
        "internal_reprojection_median_px": distribution(
            [frame["deeptag_result"]["median_px"] for frame in successful_deep_frames]
        ),
        "internal_reprojection_p90_px": distribution(
            [frame["deeptag_result"]["p90_px"] for frame in successful_deep_frames]
        ),
        "internal_inlier_count": distribution(
            [frame["deeptag_result"]["inlier_count"] for frame in successful_deep_frames]
        ),
        "recovered_old_failure_reasons": dict(recovered_reasons),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    report = {
        "pkl_path": str(pkl_path),
        "camera": args.camera,
        "cube_config": str(cube_path),
        "algorithm": "deeptag_measured_internal_grid_ransac_ippe_lm_v1",
        "comparison_note": (
            "Differences from offline_pos measure agreement only; neither pose is "
            "independent ground truth."
        ),
        "parameters": {
            "homography_ransac_px": HOMOGRAPHY_RANSAC_PX,
            "tag_support_median_px": TAG_SUPPORT_MEDIAN_PX,
            "max_accepted_median_px": MAX_ACCEPTED_MEDIAN_PX,
            "max_accepted_p90_px": MAX_ACCEPTED_P90_PX,
            "min_detected_points": MIN_DETECTED_POINTS,
        },
        "summary": summary,
        "frames": frame_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[INFO] Report: {output_path}")


if __name__ == "__main__":
    main()
