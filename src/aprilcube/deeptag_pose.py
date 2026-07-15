"""DeepTag internal-grid pose backend for AprilCube recordings."""

from __future__ import annotations

import contextlib
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from aprilcube.detect import build_tag_corner_map, load_cube_config


APRILCUBE_ROOT = Path(__file__).resolve().parents[2]
DEEPTAG_ROOT = APRILCUBE_ROOT / "thirdparty/deeptag-pytorch"
if str(DEEPTAG_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEPTAG_ROOT))

from deeptag_model_setting import load_deeptag_models  # noqa: E402
from marker_dict_setting import load_marker_codebook  # noqa: E402
from stag_decode.detection_engine import DetectionEngine  # noqa: E402


HOMOGRAPHY_RANSAC_PX = 2.5
TAG_SUPPORT_MEDIAN_PX = 6.0
MAX_ACCEPTED_MEDIAN_PX = 5.0
MAX_ACCEPTED_P90_PX = 9.0
MIN_DETECTED_POINTS = 8


_MODEL_BUNDLE: tuple[Any, Any, Any, str, list[int], dict] | None = None


def load_model_bundle(
    *,
    use_cpu: bool = False,
) -> tuple[Any, Any, Any, str, list[int], dict]:
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is not None:
        return _MODEL_BUNDLE

    previous_cwd = Path.cwd()
    try:
        os.chdir(DEEPTAG_ROOT)
        requested_device = "cpu" if use_cpu else None
        model_detector, model_decoder, device, tag_type, grids = load_deeptag_models(
            "apriltag", requested_device
        )
        codebook = load_marker_codebook(
            str(DEEPTAG_ROOT / "codebook/apriltag_codebook.txt"), tag_type
        )
    finally:
        os.chdir(previous_cwd)

    _MODEL_BUNDLE = (
        model_detector,
        model_decoder,
        device,
        tag_type,
        grids,
        codebook,
    )
    return _MODEL_BUNDLE


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def pose_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def pose_vectors(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        cv2.Rodrigues(transform[:3, :3])[0].reshape(3, 1),
        transform[:3, 3].reshape(3, 1),
    )


def local_tag_to_cube_transform(tag_corners: np.ndarray) -> np.ndarray:
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
    solutions = cv2.solvePnPGeneric(
        inlier_object,
        inlier_image,
        camera_matrix,
        np.zeros(5, dtype=np.float64),
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not solutions[0]:
        return []

    cube_from_tag = local_tag_to_cube_transform(tag_corners)
    cube_points = deep_points_to_cube(normalized, tag_corners)
    candidates: list[dict[str, Any]] = []
    for initial_rvec, initial_tvec in zip(solutions[1], solutions[2]):
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
        mad = float(np.median(np.abs(initial_errors[homography_inliers] - median)))
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
        camera_from_cube = pose_matrix(rvec, tvec) @ np.linalg.inv(cube_from_tag)
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
                        best_tag_median, float(np.median(inlier_errors))
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
        score = (
            (
                -float(len(supported_tags)),
                -float(hypothesis["inlier_points"]),
                support_median,
                hypothesis["median_px"],
            )
            if previous_transform is None
            else (
                -float(len(supported_tags)),
                motion_score,
                support_median,
                hypothesis["median_px"],
            )
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
        object_points, image_points, rvec, tvec, camera_matrix
    )
    median = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median)))
    threshold = min(6.0, max(1.5, median + 3.0 * 1.4826 * mad))
    inliers = errors <= threshold
    if int(inliers.sum()) < MIN_DETECTED_POINTS:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(
        object_points[inliers],
        image_points[inliers],
        camera_matrix,
        np.zeros(5, dtype=np.float64),
        rvec,
        tvec,
    )
    errors = reprojection_errors(
        object_points, image_points, rvec, tvec, camera_matrix
    )
    inlier_errors = errors[inliers]
    if not np.all(np.isfinite(inlier_errors)):
        return None
    return {
        "rvec": rvec,
        "tvec": tvec,
        "seed_tag_id": int(selected["tag_id"]),
        "supported_tag_ids": [int(tag_id) for tag_id in supported_tags],
        "detected_tag_ids": [int(tag_id) for tag_id in candidates_by_tag],
        "point_count": int(len(object_points)),
        "inlier_count": int(inliers.sum()),
        "mean_px": float(np.mean(inlier_errors)),
        "median_px": float(np.median(inlier_errors)),
        "p90_px": float(np.percentile(inlier_errors, 90)),
        "max_px": float(np.max(inlier_errors)),
        "rmse_px": float(np.sqrt(np.mean(np.square(inlier_errors)))),
    }


@dataclass
class CubeModel:
    name: str
    tag_corner_map: dict[int, np.ndarray]
    face_id_sets: dict[str, set[int]]

    def visible_faces(self, tag_ids: list[int]) -> set[str]:
        return {
            face_name
            for face_name, face_tags in self.face_id_sets.items()
            if any(tag_id in face_tags for tag_id in tag_ids)
        }


class DeepTagPoseBackend:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        cube_paths: list[Path],
        *,
        use_cpu: bool = False,
    ) -> None:
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        self.cube_models: list[CubeModel] = []
        marker_sizes_mm: list[float] = []
        for cube_path in cube_paths:
            config, face_id_sets = load_cube_config(str(cube_path / "config.json"))
            self.cube_models.append(
                CubeModel(
                    name=cube_path.name,
                    tag_corner_map=build_tag_corner_map(config),
                    face_id_sets=face_id_sets,
                )
            )
            marker_sizes_mm.append(float(config.tag_size_mm))
        if not marker_sizes_mm:
            raise ValueError("DeepTag backend requires at least one cube config")

        model_detector, model_decoder, device, tag_type, grids, codebook = (
            load_model_bundle(use_cpu=use_cpu)
        )
        self.device = str(device)
        self.engine = DetectionEngine(
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
            cameraMatrix=self.camera_matrix,
            distCoeffs=np.zeros(8, dtype=np.float64),
            codebook=codebook,
            tag_real_size_in_meter_dict={-1: marker_sizes_mm[0] / 1000.0},
        )
        self.previous_transform_by_cube: dict[str, np.ndarray] = {}

    def estimate(self, image_bgr: np.ndarray) -> dict[str, Any]:
        with contextlib.redirect_stdout(io.StringIO()):
            decoded_tags = self.engine.process(image_bgr, detect_scale=None)
        valid_tags = [tag for tag in decoded_tags if tag.get("is_valid", False)]
        raw_valid_ids = [int(tag.get("tag_id", -1)) for tag in valid_tags]
        cube_results = []
        for cube_model in self.cube_models:
            target_tags = [
                tag
                for tag in valid_tags
                if int(tag.get("tag_id", -1)) in cube_model.tag_corner_map
            ]
            candidates_by_tag: dict[int, list[dict[str, Any]]] = {}
            for decoded_tag in target_tags:
                tag_id = int(decoded_tag["tag_id"])
                normalized_points = np.asarray(
                    self.engine.pose_solver_dict[
                        len(decoded_tag["keypoints_in_images"])
                    ].fine_grid_points_anno,
                    dtype=np.float64,
                )
                candidates = robust_tag_candidates(
                    decoded_tag,
                    normalized_points,
                    cube_model.tag_corner_map[tag_id],
                    self.camera_matrix,
                )
                if candidates:
                    candidates_by_tag.setdefault(tag_id, []).extend(candidates)

            selected = choose_and_refine_cube_pose(
                candidates_by_tag,
                self.camera_matrix,
                self.previous_transform_by_cube.get(cube_model.name),
            )
            if selected is None:
                if not valid_tags:
                    failure_reason = "deeptag_no_valid_tags"
                elif not target_tags:
                    failure_reason = "deeptag_no_target_tags"
                elif not candidates_by_tag:
                    failure_reason = "deeptag_internal_points_or_pnp_rejected"
                else:
                    failure_reason = "deeptag_pose_quality_rejected"
                pose_result = {
                    "success": False,
                    "rvec": None,
                    "tvec": None,
                    "T": None,
                    "reproj_error": float("inf"),
                    "n_tags": len(target_tags),
                    "n_inliers": 0,
                    "detections": [],
                    "tag_ids": [],
                    "visible_faces": set(),
                    "predicted": False,
                    "failure_reason": failure_reason,
                    "deeptag_raw_valid_ids": raw_valid_ids,
                    "deeptag_raw_roi_count": len(decoded_tags),
                    "deeptag_target_tag_count": len(target_tags),
                }
            else:
                transform = pose_matrix(selected["rvec"], selected["tvec"])
                self.previous_transform_by_cube[cube_model.name] = transform
                supported_tag_ids = selected["supported_tag_ids"]
                pose_result = {
                    "success": True,
                    "rvec": selected["rvec"],
                    "tvec": selected["tvec"],
                    "T": transform,
                    "reproj_error": selected["mean_px"],
                    "n_tags": len(supported_tag_ids),
                    "n_inliers": selected["inlier_count"],
                    "detections": [],
                    "tag_ids": supported_tag_ids,
                    "visible_faces": cube_model.visible_faces(supported_tag_ids),
                    "predicted": False,
                    "failure_reason": "",
                    "deeptag_raw_valid_ids": raw_valid_ids,
                    "deeptag_raw_roi_count": len(decoded_tags),
                    "deeptag_target_tag_count": len(target_tags),
                    "deeptag_seed_tag_id": selected["seed_tag_id"],
                    "deeptag_detected_tag_ids": selected["detected_tag_ids"],
                    "deeptag_supported_tag_ids": supported_tag_ids,
                    "deeptag_internal_point_count": selected["point_count"],
                    "deeptag_internal_inlier_count": selected["inlier_count"],
                    "deeptag_internal_reproj_mean_px": selected["mean_px"],
                    "deeptag_internal_reproj_median_px": selected["median_px"],
                    "deeptag_internal_reproj_p90_px": selected["p90_px"],
                    "deeptag_internal_reproj_max_px": selected["max_px"],
                    "deeptag_internal_reproj_rmse_px": selected["rmse_px"],
                }
            cube_results.append({"cube_name": cube_model.name, "result": pose_result})

        return {
            "cube_results": cube_results,
            "decoded_tag_count": len(raw_valid_ids),
            "decoded_tag_ids": raw_valid_ids,
            "raw_roi_count": len(decoded_tags),
            "tag_detect_mode": "deeptag_once_per_camera",
            "pose_backend": "deeptag_internal_grid",
            "deeptag_device": self.device,
            "runtime_pose_filter": False,
            "temporal_candidate_selection": True,
        }


def backend_parameters() -> dict[str, Any]:
    return {
        "homography_ransac_px": HOMOGRAPHY_RANSAC_PX,
        "tag_support_median_px": TAG_SUPPORT_MEDIAN_PX,
        "max_accepted_median_px": MAX_ACCEPTED_MEDIAN_PX,
        "max_accepted_p90_px": MAX_ACCEPTED_P90_PX,
        "min_detected_points": MIN_DETECTED_POINTS,
        "uses_only_network_measured_points": True,
        "planar_solver": "IPPE_then_LM",
        "temporal_candidate_selection": True,
    }
