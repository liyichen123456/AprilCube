#!/usr/bin/env python3
"""Recover missing 023 cube poses using RGB flow and visible-edge alignment."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp

from aprilcube.detect import build_tag_corner_map, load_cube_config


APRILCUBE_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_ALGORITHM = "deeptag_internal_grid_primary_cv2_fallback_v2"
RECOVERY_ALGORITHM = "rgb_planar_flow_visible_edge_refine_v2"
UNDISTORTED_IMAGE_FIELD = "undistorted_image_jpeg"
UNDISTORTED_OVERLAY_FIELD = "undistorted_pose_overlay_jpeg"
MIN_FLOW_POINTS = 8
MIN_PLANAR_FLOW_POINTS = 5
MAX_FLOW_INITIAL_ERROR_PX = 24.0
MAX_FLOW_FINAL_MEDIAN_PX = 4.0
MAX_TEMPORAL_TRANSLATION_DELTA_MM = 15.0
MAX_TEMPORAL_ROTATION_DELTA_DEG = 10.0


CUBE_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
FACE_VERTEX_NORMALS = (
    ((0, 1, 2, 3), (0.0, 0.0, -1.0)),
    ((4, 5, 6, 7), (0.0, 0.0, 1.0)),
    ((0, 1, 5, 4), (0.0, -1.0, 0.0)),
    ((1, 2, 6, 5), (1.0, 0.0, 0.0)),
    ((2, 3, 7, 6), (0.0, 1.0, 0.0)),
    ((3, 0, 4, 7), (-1.0, 0.0, 0.0)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=APRILCUBE_ROOT / "outputs/026_rgb_pose_recovery",
    )
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def pose_matrix(rvec: Any, tvec: Any) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = cv2.Rodrigues(
        np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    )[0]
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def pose_vectors(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        cv2.Rodrigues(transform[:3, :3])[0].reshape(3, 1),
        transform[:3, 3].reshape(3, 1),
    )


def rotation_delta_deg(rvec_a: Any, rvec_b: Any) -> float:
    rotation_a = Rotation.from_rotvec(
        np.asarray(rvec_a, dtype=np.float64).reshape(3)
    )
    rotation_b = Rotation.from_rotvec(
        np.asarray(rvec_b, dtype=np.float64).reshape(3)
    )
    return float(np.degrees((rotation_b * rotation_a.inv()).magnitude()))


def interpolate_pose(
    before: dict[str, Any],
    after: dict[str, Any],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    t0 = np.asarray(before["tvec"], dtype=np.float64).reshape(3)
    t1 = np.asarray(after["tvec"], dtype=np.float64).reshape(3)
    r0 = Rotation.from_rotvec(
        np.asarray(before["rvec"], dtype=np.float64).reshape(3)
    )
    r1 = Rotation.from_rotvec(
        np.asarray(after["rvec"], dtype=np.float64).reshape(3)
    )
    rotation = Slerp(
        [0.0, 1.0], Rotation.from_quat([r0.as_quat(), r1.as_quat()])
    )([alpha])[0]
    return (
        rotation.as_rotvec().reshape(3, 1),
        ((1.0 - alpha) * t0 + alpha * t1).reshape(3, 1),
    )


def cube_vertices(box_dims: tuple[float, float, float]) -> np.ndarray:
    x, y, z = (float(value) * 0.5 for value in box_dims)
    return np.asarray(
        [
            [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
            [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z],
        ],
        dtype=np.float64,
    )


def visible_cube_edges(
    vertices: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> list[tuple[int, int]]:
    rotation = cv2.Rodrigues(rvec)[0]
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    visible_faces: list[set[int]] = []
    for face_vertices, normal_value in FACE_VERTEX_NORMALS:
        normal = np.asarray(normal_value, dtype=np.float64)
        center = vertices[np.asarray(face_vertices)].mean(axis=0)
        normal_camera = rotation @ normal
        center_camera = rotation @ center + translation
        if float(np.dot(normal_camera, center_camera)) < 0.0:
            visible_faces.append(set(face_vertices))
    output = []
    for edge in CUBE_EDGES:
        if any(edge[0] in face and edge[1] in face for face in visible_faces):
            output.append(edge)
    return output or list(CUBE_EDGES)


def sample_segment(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    values = np.linspace(0.02, 0.98, max(4, int(count)), dtype=np.float64)
    return (1.0 - values[:, None]) * start + values[:, None] * end


def model_edge_points(
    box_dims: tuple[float, float, float],
    tag_corner_map: dict[int, np.ndarray],
    tag_ids: list[int],
    init_rvec: np.ndarray,
    init_tvec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = cube_vertices(box_dims)
    cube_chunks = [
        sample_segment(vertices[a], vertices[b], 36)
        for a, b in visible_cube_edges(vertices, init_rvec, init_tvec)
    ]
    tag_chunks = []
    for tag_id in sorted(set(int(value) for value in tag_ids)):
        if tag_id not in tag_corner_map:
            continue
        corners = np.asarray(tag_corner_map[tag_id], dtype=np.float64).reshape(4, 3)
        for index in range(4):
            tag_chunks.append(
                sample_segment(corners[index], corners[(index + 1) % 4], 24)
            )
    return (
        np.vstack(cube_chunks),
        np.vstack(tag_chunks) if tag_chunks else np.empty((0, 3), dtype=np.float64),
    )


def make_edge_distance(gray: np.ndarray) -> np.ndarray:
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    edges = cv2.bitwise_or(
        cv2.Canny(blurred, 35, 105),
        cv2.Canny(blurred, 65, 180),
    )
    return cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)


def sampled_distance(
    distance: np.ndarray,
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    if len(object_points) == 0:
        return np.empty(0, dtype=np.float64)
    projected = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        np.zeros(5, dtype=np.float64),
    )[0].reshape(-1, 2)
    height, width = distance.shape[:2]
    x = projected[:, 0]
    y = projected[:, 1]
    valid = (x >= 0.0) & (x <= width - 1.0) & (y >= 0.0) & (y <= height - 1.0)
    values = np.full(len(projected), 12.0, dtype=np.float64)
    if np.any(valid):
        sampled = cv2.remap(
            distance,
            x[valid].astype(np.float32).reshape(-1, 1),
            y[valid].astype(np.float32).reshape(-1, 1),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=12.0,
        ).reshape(-1)
        values[valid] = np.minimum(sampled, 12.0)
    return values


def robust_distance_cost(values: np.ndarray) -> float:
    if len(values) == 0:
        return 12.0
    clipped = np.minimum(np.asarray(values, dtype=np.float64), 8.0)
    return float(0.65 * np.mean(clipped) + 0.35 * np.median(clipped))


def project_errors(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    if len(object_points) == 0:
        return np.empty(0, dtype=np.float64)
    projected = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        np.zeros(5, dtype=np.float64),
    )[0].reshape(-1, 2)
    return np.linalg.norm(projected - image_points, axis=1)


def marker_template_variants(dict_id: int, tag_id: int, size: int = 128) -> list[np.ndarray]:
    dictionary = cv2.aruco.getPredefinedDictionary(int(dict_id))
    marker = cv2.aruco.generateImageMarker(
        dictionary, int(tag_id), size, borderBits=1
    )
    variants: list[np.ndarray] = []
    for mirrored in (False, True):
        base = cv2.flip(marker, 1) if mirrored else marker
        for turns in range(4):
            candidate = np.rot90(base, turns).copy()
            if not any(np.array_equal(candidate, value) for value in variants):
                variants.append(candidate)
    return variants


def warped_tag_ncc(
    gray: np.ndarray,
    corners_3d: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    template: np.ndarray,
) -> float:
    corners_2d = cv2.projectPoints(
        corners_3d,
        rvec,
        tvec,
        camera_matrix,
        np.zeros(5, dtype=np.float64),
    )[0].reshape(4, 2)
    height, width = gray.shape[:2]
    if (
        np.any(corners_2d[:, 0] < 1.0)
        or np.any(corners_2d[:, 0] >= width - 1.0)
        or np.any(corners_2d[:, 1] < 1.0)
        or np.any(corners_2d[:, 1] >= height - 1.0)
    ):
        return -1.0
    size = int(template.shape[0])
    canonical = np.asarray(
        [[0.0, 0.0], [size - 1.0, 0.0], [size - 1.0, size - 1.0], [0.0, size - 1.0]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(corners_2d.astype(np.float32), canonical)
    patch = cv2.warpPerspective(
        gray,
        transform,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float32)
    patch = cv2.GaussianBlur(patch, (3, 3), 0.8)
    expected = cv2.GaussianBlur(template, (3, 3), 0.8).astype(np.float32)
    margin = max(2, size // 32)
    patch = patch[margin:-margin, margin:-margin]
    expected = expected[margin:-margin, margin:-margin]
    patch -= float(np.mean(patch))
    expected -= float(np.mean(expected))
    denominator = float(np.linalg.norm(patch) * np.linalg.norm(expected))
    if denominator < 1e-6:
        return -1.0
    return float(np.sum(patch * expected) / denominator)


def refine_pose_with_tag_templates(
    target_gray: np.ndarray,
    anchor_data: list[tuple[str, np.ndarray, dict[str, Any]]],
    config: Any,
    tag_corner_map: dict[int, np.ndarray],
    tag_ids: list[int],
    temporal_rvec: np.ndarray,
    temporal_tvec: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    init_rotation = Rotation.from_rotvec(temporal_rvec.reshape(3))
    init_translation = temporal_tvec.reshape(3)
    candidates: list[dict[str, Any]] = []
    for tag_id in tag_ids:
        corners = np.asarray(tag_corner_map[tag_id], dtype=np.float64).reshape(4, 3)
        variants = marker_template_variants(config.dict_id, tag_id)
        anchor_scores = []
        for variant_index, variant in enumerate(variants):
            scores = [
                warped_tag_ncc(
                    gray,
                    corners,
                    pose["rvec"],
                    pose["tvec"],
                    camera_matrix,
                    variant,
                )
                for _label, gray, pose in anchor_data
                if tag_id in pose.get("tag_ids", [])
            ]
            anchor_scores.append(max(scores) if scores else -1.0)
        variant_index = int(np.argmax(anchor_scores))
        anchor_ncc = float(anchor_scores[variant_index])
        if anchor_ncc < 0.20:
            continue
        template = variants[variant_index]
        initial_ncc = warped_tag_ncc(
            target_gray,
            corners,
            temporal_rvec,
            temporal_tvec,
            camera_matrix,
            template,
        )

        def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            rotation = Rotation.from_rotvec(parameters[:3]) * init_rotation
            translation = init_translation + parameters[3:]
            return rotation.as_rotvec().reshape(3, 1), translation.reshape(3, 1)

        def objective(parameters: np.ndarray) -> float:
            rvec, tvec = unpack(parameters)
            ncc = warped_tag_ncc(
                target_gray, corners, rvec, tvec, camera_matrix, template
            )
            rotation_prior = float(np.linalg.norm(parameters[:3]) / 0.06) ** 2
            translation_prior = float(
                np.linalg.norm(
                    parameters[3:] / np.asarray([3.0, 3.0, 2.0], dtype=np.float64)
                )
            ) ** 2
            return 1.0 - ncc + 0.025 * rotation_prior + 0.025 * translation_prior

        result = minimize(
            objective,
            np.zeros(6, dtype=np.float64),
            method="Powell",
            bounds=[
                (-0.10, 0.10), (-0.10, 0.10), (-0.10, 0.10),
                (-4.0, 4.0), (-4.0, 4.0), (-3.0, 3.0),
            ],
            options={"maxiter": 90, "xtol": 2e-4, "ftol": 2e-4, "disp": False},
        )
        rvec, tvec = unpack(np.asarray(result.x, dtype=np.float64))
        final_ncc = warped_tag_ncc(
            target_gray, corners, rvec, tvec, camera_matrix, template
        )
        translation_delta = float(
            np.linalg.norm(tvec.reshape(3) - temporal_tvec.reshape(3))
        )
        rotation_delta = rotation_delta_deg(temporal_rvec, rvec)
        vertices = cube_vertices(tuple(float(value) for value in config.box_dims))
        temporal_projection = cv2.projectPoints(
            vertices,
            temporal_rvec,
            temporal_tvec,
            camera_matrix,
            np.zeros(5, dtype=np.float64),
        )[0].reshape(-1, 2)
        candidate_projection = cv2.projectPoints(
            vertices,
            rvec,
            tvec,
            camera_matrix,
            np.zeros(5, dtype=np.float64),
        )[0].reshape(-1, 2)
        projection_delta = np.linalg.norm(
            candidate_projection - temporal_projection, axis=1
        )
        projection_delta_median = float(np.median(projection_delta))
        projection_delta_max = float(np.max(projection_delta))
        minimum_ncc = max(0.22, 0.45 * anchor_ncc)
        if (
            final_ncc < minimum_ncc
            or translation_delta > 10.0
            or rotation_delta > 8.0
            or projection_delta_median > 35.0
            or projection_delta_max > 70.0
        ):
            continue
        candidates.append(
            {
                "rvec": rvec,
                "tvec": tvec,
                "tag_id": int(tag_id),
                "variant_index": variant_index,
                "anchor_ncc": anchor_ncc,
                "initial_ncc": initial_ncc,
                "final_ncc": final_ncc,
                "translation_delta_mm": translation_delta,
                "rotation_delta_deg": rotation_delta,
                "projection_delta_median_px": projection_delta_median,
                "projection_delta_max_px": projection_delta_max,
                "score": (
                    -final_ncc
                    + 0.003 * translation_delta
                    + 0.002 * rotation_delta
                ),
            }
        )
    if not candidates:
        return None
    selected = min(candidates, key=lambda value: value["score"])
    details = {
        key: value
        for key, value in selected.items()
        if key not in {"rvec", "tvec", "score"}
    }
    details["candidate_count"] = len(candidates)
    return selected["rvec"], selected["tvec"], details


def tag_feature_points(
    gray: np.ndarray,
    pose: dict[str, Any],
    tag_corner_map: dict[int, np.ndarray],
    tag_ids: list[int],
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    image_chunks: list[np.ndarray] = []
    object_chunks: list[np.ndarray] = []
    for tag_id in sorted(set(int(value) for value in tag_ids)):
        if tag_id not in tag_corner_map:
            continue
        corners_3d = np.asarray(tag_corner_map[tag_id], dtype=np.float64).reshape(4, 3)
        corners_2d = cv2.projectPoints(
            corners_3d,
            pose["rvec"],
            pose["tvec"],
            camera_matrix,
            np.zeros(5, dtype=np.float64),
        )[0].reshape(4, 2)
        mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(corners_2d).astype(np.int32), 255)
        features = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=80,
            qualityLevel=0.008,
            minDistance=3.0,
            mask=mask,
            blockSize=5,
            useHarrisDetector=False,
        )
        if features is None:
            continue
        image_points = features.reshape(-1, 2).astype(np.float64)
        image_to_uv = cv2.getPerspectiveTransform(
            corners_2d.astype(np.float32),
            np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
        )
        uv = cv2.perspectiveTransform(
            image_points.astype(np.float32).reshape(-1, 1, 2), image_to_uv
        ).reshape(-1, 2).astype(np.float64)
        valid = np.all((uv >= -0.02) & (uv <= 1.02), axis=1)
        if not np.any(valid):
            continue
        uv = uv[valid]
        image_points = image_points[valid]
        u = uv[:, 0]
        v = uv[:, 1]
        top_left, top_right, bottom_right, bottom_left = corners_3d
        object_points = (
            ((1.0 - u) * (1.0 - v))[:, None] * top_left
            + (u * (1.0 - v))[:, None] * top_right
            + (u * v)[:, None] * bottom_right
            + ((1.0 - u) * v)[:, None] * bottom_left
        )
        image_chunks.append(image_points)
        object_chunks.append(object_points)
    if not image_chunks:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )
    return np.vstack(image_chunks), np.vstack(object_chunks)


def track_features(
    source_gray: np.ndarray,
    target_gray: np.ndarray,
    source_points: np.ndarray,
    object_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if len(source_points) == 0:
        return object_points[:0], source_points[:0], {"raw": 0, "fb_valid": 0}
    source = source_points.astype(np.float32).reshape(-1, 1, 2)
    lk_params = {
        "winSize": (31, 31),
        "maxLevel": 4,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            40,
            0.001,
        ),
    }
    target, status_forward, error_forward = cv2.calcOpticalFlowPyrLK(
        source_gray, target_gray, source, None, **lk_params
    )
    if target is None:
        return object_points[:0], source_points[:0], {"raw": len(source), "fb_valid": 0}
    reverse, status_reverse, _error_reverse = cv2.calcOpticalFlowPyrLK(
        target_gray, source_gray, target, None, **lk_params
    )
    if reverse is None:
        return object_points[:0], source_points[:0], {"raw": len(source), "fb_valid": 0}
    fb_error = np.linalg.norm(reverse.reshape(-1, 2) - source.reshape(-1, 2), axis=1)
    valid = (
        status_forward.reshape(-1).astype(bool)
        & status_reverse.reshape(-1).astype(bool)
        & (fb_error <= 3.0)
        & (error_forward.reshape(-1) <= 60.0)
    )
    tracked = target.reshape(-1, 2).astype(np.float64)
    height, width = target_gray.shape[:2]
    valid &= (
        (tracked[:, 0] >= 1.0)
        & (tracked[:, 0] < width - 1.0)
        & (tracked[:, 1] >= 1.0)
        & (tracked[:, 1] < height - 1.0)
    )
    return (
        object_points[valid],
        tracked[valid],
        {
            "raw": len(source),
            "fb_valid": int(valid.sum()),
            "fb_median_px": (
                float(np.median(fb_error[valid])) if np.any(valid) else float("inf")
            ),
        },
    )


def refine_pose_from_planar_flow_groups(
    flow_groups: list[dict[str, Any]],
    all_object_points: np.ndarray,
    all_image_points: np.ndarray,
    temporal_rvec: np.ndarray,
    temporal_tvec: np.ndarray,
    camera_matrix: np.ndarray,
    tag_corner_map: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]] | None:
    """Recover a pose from a tracked planar tag without trusting the temporal seed."""
    candidates: list[dict[str, Any]] = []
    canonical_corners = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    for group in flow_groups:
        object_points = np.asarray(group["object_points"], dtype=np.float64)
        image_points = np.asarray(group["image_points"], dtype=np.float64)
        if len(object_points) < MIN_PLANAR_FLOW_POINTS:
            continue
        corners = np.asarray(
            tag_corner_map[int(group["tag_id"])], dtype=np.float64
        ).reshape(4, 3)
        origin = corners[0]
        u_axis = corners[1] - origin
        v_axis = corners[3] - origin
        local_points = np.column_stack(
            (
                (object_points - origin) @ u_axis / float(u_axis @ u_axis),
                (object_points - origin) @ v_axis / float(v_axis @ v_axis),
            )
        )
        homography, mask = cv2.findHomography(
            local_points,
            image_points,
            cv2.RANSAC,
            3.0,
            maxIters=3000,
            confidence=0.999,
        )
        if homography is None or mask is None:
            continue
        homography_inliers = mask.reshape(-1).astype(bool)
        if int(homography_inliers.sum()) < MIN_PLANAR_FLOW_POINTS:
            continue
        target_corners = cv2.perspectiveTransform(
            canonical_corners.astype(np.float32).reshape(-1, 1, 2),
            homography,
        ).reshape(4, 2).astype(np.float64)
        try:
            solutions = cv2.solvePnPGeneric(
                corners,
                target_corners,
                camera_matrix,
                np.zeros(5, dtype=np.float64),
                flags=cv2.SOLVEPNP_IPPE,
            )
        except cv2.error:
            continue
        if not solutions[0]:
            continue
        for rvec, tvec in zip(solutions[1], solutions[2]):
            if float(np.asarray(tvec).reshape(3)[2]) <= 0.0:
                continue
            try:
                rvec, tvec = cv2.solvePnPRefineLM(
                    object_points[homography_inliers],
                    image_points[homography_inliers],
                    camera_matrix,
                    np.zeros(5, dtype=np.float64),
                    np.asarray(rvec, dtype=np.float64),
                    np.asarray(tvec, dtype=np.float64),
                )
            except cv2.error:
                continue
            own_errors = project_errors(
                object_points, image_points, rvec, tvec, camera_matrix
            )
            own_median = float(np.median(own_errors[homography_inliers]))
            if own_median > MAX_FLOW_FINAL_MEDIAN_PX:
                continue
            global_errors = project_errors(
                all_object_points, all_image_points, rvec, tvec, camera_matrix
            )
            median = float(np.median(global_errors))
            mad = float(np.median(np.abs(global_errors - median)))
            threshold = min(8.0, max(2.0, median + 3.0 * 1.4826 * mad))
            active = global_errors <= threshold
            if int(active.sum()) < MIN_PLANAR_FLOW_POINTS:
                continue
            try:
                rvec, tvec = cv2.solvePnPRefineLM(
                    all_object_points[active],
                    all_image_points[active],
                    camera_matrix,
                    np.zeros(5, dtype=np.float64),
                    rvec,
                    tvec,
                )
            except cv2.error:
                continue
            global_errors = project_errors(
                all_object_points, all_image_points, rvec, tvec, camera_matrix
            )
            active_median = float(np.median(global_errors[active]))
            translation_delta = float(
                np.linalg.norm(tvec.reshape(3) - temporal_tvec.reshape(3))
            )
            rotation_delta = rotation_delta_deg(temporal_rvec, rvec)
            if (
                active_median > MAX_FLOW_FINAL_MEDIAN_PX
                or translation_delta > MAX_TEMPORAL_TRANSLATION_DELTA_MM
                or rotation_delta > MAX_TEMPORAL_ROTATION_DELTA_DEG
            ):
                continue
            candidates.append(
                {
                    "rvec": rvec,
                    "tvec": tvec,
                    "active": active,
                    "tag_id": int(group["tag_id"]),
                    "anchor": str(group["anchor"]),
                    "homography_inliers": int(homography_inliers.sum()),
                    "flow_median_px": active_median,
                    "translation_delta_mm": translation_delta,
                    "rotation_delta_deg": rotation_delta,
                    "score": (
                        active_median
                        + 0.08 * translation_delta
                        + 0.05 * rotation_delta
                        - 0.003 * int(active.sum())
                    ),
                }
            )
    if not candidates:
        return None
    selected = min(candidates, key=lambda value: value["score"])
    details = {
        "candidate_count": len(candidates),
        "selected_tag_id": selected["tag_id"],
        "selected_anchor": selected["anchor"],
        "homography_inliers": selected["homography_inliers"],
    }
    return selected["rvec"], selected["tvec"], selected["active"], details


def refine_pose_with_flow(
    object_points: np.ndarray,
    image_points: np.ndarray,
    init_rvec: np.ndarray,
    init_tvec: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if len(object_points) < MIN_FLOW_POINTS:
        return None
    initial_errors = project_errors(
        object_points, image_points, init_rvec, init_tvec, camera_matrix
    )
    active = initial_errors <= MAX_FLOW_INITIAL_ERROR_PX
    if int(active.sum()) < MIN_FLOW_POINTS:
        return None
    rvec = np.asarray(init_rvec, dtype=np.float64).reshape(3, 1).copy()
    tvec = np.asarray(init_tvec, dtype=np.float64).reshape(3, 1).copy()
    for _iteration in range(4):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points[active],
                image_points[active],
                camera_matrix,
                np.zeros(5, dtype=np.float64),
                rvec,
                tvec,
            )
        except cv2.error:
            return None
        errors = project_errors(
            object_points, image_points, rvec, tvec, camera_matrix
        )
        active_errors = errors[active]
        median = float(np.median(active_errors))
        mad = float(np.median(np.abs(active_errors - median)))
        threshold = min(6.0, max(1.25, median + 3.0 * 1.4826 * mad))
        next_active = errors <= threshold
        if int(next_active.sum()) < MIN_FLOW_POINTS:
            break
        if np.array_equal(next_active, active):
            active = next_active
            break
        active = next_active
    if int(active.sum()) < MIN_FLOW_POINTS:
        return None
    errors = project_errors(object_points, image_points, rvec, tvec, camera_matrix)
    if float(np.median(errors[active])) > MAX_FLOW_FINAL_MEDIAN_PX:
        return None
    return rvec, tvec, active


def optimize_visible_edges(
    distance: np.ndarray,
    cube_edge_points: np.ndarray,
    tag_edge_points: np.ndarray,
    init_rvec: np.ndarray,
    init_tvec: np.ndarray,
    temporal_rvec: np.ndarray,
    temporal_tvec: np.ndarray,
    camera_matrix: np.ndarray,
    flow_object_points: np.ndarray,
    flow_image_points: np.ndarray,
) -> dict[str, Any]:
    init_rotation = Rotation.from_rotvec(init_rvec.reshape(3))
    init_translation = init_tvec.reshape(3)

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rotation = Rotation.from_rotvec(parameters[:3]) * init_rotation
        translation = init_translation + parameters[3:]
        return rotation.as_rotvec().reshape(3, 1), translation.reshape(3, 1)

    def metrics(rvec: np.ndarray, tvec: np.ndarray) -> tuple[float, float, float]:
        cube_cost = robust_distance_cost(
            sampled_distance(distance, cube_edge_points, rvec, tvec, camera_matrix)
        )
        tag_cost = robust_distance_cost(
            sampled_distance(distance, tag_edge_points, rvec, tvec, camera_matrix)
        ) if len(tag_edge_points) else cube_cost
        flow_errors = project_errors(
            flow_object_points,
            flow_image_points,
            rvec,
            tvec,
            camera_matrix,
        )
        flow_median = float(np.median(flow_errors)) if len(flow_errors) else float("inf")
        return cube_cost, tag_cost, flow_median

    temporal_rotation = Rotation.from_rotvec(temporal_rvec.reshape(3))
    temporal_translation = temporal_tvec.reshape(3)

    def objective(parameters: np.ndarray) -> float:
        rvec, tvec = unpack(parameters)
        if float(tvec.reshape(3)[2]) <= 0.0:
            return 1e4
        cube_cost, tag_cost, flow_median = metrics(rvec, tvec)
        current_rotation = Rotation.from_rotvec(rvec.reshape(3))
        rotation_prior = float(
            (current_rotation * temporal_rotation.inv()).magnitude() / 0.10
        ) ** 2
        translation_prior = float(
            np.linalg.norm(
                (tvec.reshape(3) - temporal_translation)
                / np.asarray([8.0, 8.0, 14.0], dtype=np.float64)
            )
        ) ** 2
        flow_cost = min(flow_median, 12.0) if np.isfinite(flow_median) else 0.0
        return (
            0.62 * cube_cost
            + 0.38 * tag_cost
            + 1.20 * flow_cost
            + 0.08 * rotation_prior
            + 0.08 * translation_prior
        )

    initial_parameters = np.zeros(6, dtype=np.float64)
    initial_value = objective(initial_parameters)
    result = minimize(
        objective,
        initial_parameters,
        method="Powell",
        bounds=[
            (-0.16, 0.16), (-0.16, 0.16), (-0.16, 0.16),
            (-10.0, 10.0), (-10.0, 10.0), (-15.0, 15.0),
        ],
        options={"maxiter": 80, "xtol": 5e-4, "ftol": 5e-4, "disp": False},
    )
    rvec, tvec = unpack(np.asarray(result.x, dtype=np.float64))
    cube_cost, tag_cost, flow_median = metrics(rvec, tvec)
    init_cube, init_tag, init_flow = metrics(init_rvec, init_tvec)
    return {
        "rvec": rvec,
        "tvec": tvec,
        "objective_before": float(initial_value),
        "objective_after": float(result.fun),
        "cube_edge_cost_before": init_cube,
        "cube_edge_cost_after": cube_cost,
        "tag_edge_cost_before": init_tag,
        "tag_edge_cost_after": tag_cost,
        "flow_median_before": init_flow,
        "flow_median_after": flow_median,
    }


def minimal_pose(result: dict[str, Any]) -> dict[str, Any] | None:
    if not result.get("success", False):
        return None
    return {
        "success": True,
        "rvec": np.asarray(result["rvec"], dtype=np.float64).reshape(3, 1),
        "tvec": np.asarray(result["tvec"], dtype=np.float64).reshape(3, 1),
        "tag_ids": [int(value) for value in result.get("tag_ids", [])],
        "visible_faces": set(result.get("visible_faces", set())),
        "pose_backend": str(result.get("pose_backend", "")),
    }


def build_index(
    pkl_path: Path,
) -> tuple[
    dict[str, Any],
    list[int],
    dict[str, list[dict[str, Any] | None]],
    dict[str, list[float]],
    dict[str, tuple[str, str]],
]:
    offsets: list[int] = []
    tracks: dict[str, list[dict[str, Any] | None]] = {}
    timestamps: dict[str, list[float]] = {}
    key_parts: dict[str, tuple[str, str]] = {}
    with pkl_path.open("rb") as file:
        header = pickle.load(file)
        algorithm = header.get("metadata", {}).get("offline_pos_estimation", {}).get(
            "algorithm"
        )
        if algorithm != EXPECTED_ALGORITHM:
            raise ValueError(f"Expected {EXPECTED_ALGORITHM}, got {algorithm}")
        while True:
            offset = file.tell()
            try:
                record = pickle.load(file)
            except EOFError:
                break
            if not isinstance(record, dict) or record.get("type") != "frame_pair":
                continue
            frame_index = len(offsets)
            offsets.append(offset)
            for camera_name, camera_record in record["cameras"].items():
                for cube in camera_record["offline_pos"]["cube_results"]:
                    cube_name = str(cube["cube_name"])
                    key = f"{camera_name}/{cube_name}"
                    key_parts[key] = (camera_name, cube_name)
                    tracks.setdefault(key, [])
                    timestamps.setdefault(key, [])
                    while len(tracks[key]) < frame_index:
                        tracks[key].append(None)
                        timestamps[key].append(float("nan"))
                    tracks[key].append(minimal_pose(cube["result"]))
                    timestamps[key].append(float(camera_record["capture_timestamp"]))
    for key in tracks:
        while len(tracks[key]) < len(offsets):
            tracks[key].append(None)
            timestamps[key].append(float("nan"))
    return header, offsets, tracks, timestamps, key_parts


class FrameLoader:
    def __init__(self, pkl_path: Path, offsets: list[int]) -> None:
        self.pkl_path = pkl_path
        self.offsets = offsets
        self.gray_cache: dict[tuple[int, str], np.ndarray] = {}

    def record(self, frame_index: int) -> dict[str, Any]:
        with self.pkl_path.open("rb") as file:
            file.seek(self.offsets[frame_index])
            return pickle.load(file)

    def image(self, frame_index: int, camera_name: str) -> np.ndarray:
        record = self.record(frame_index)
        encoded = record["cameras"][camera_name][UNDISTORTED_IMAGE_FIELD]
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode frame {frame_index} {camera_name}")
        return image

    def gray(self, frame_index: int, camera_name: str) -> np.ndarray:
        key = (frame_index, camera_name)
        if key not in self.gray_cache:
            image = self.image(frame_index, camera_name)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.gray_cache[key] = cv2.createCLAHE(
                clipLimit=2.0, tileGridSize=(8, 8)
            ).apply(gray)
        return self.gray_cache[key]

    def camera_matrix(self, frame_index: int, camera_name: str) -> np.ndarray:
        record = self.record(frame_index)
        return np.asarray(
            record["cameras"][camera_name]["undistorted_visualization_meta"][
                "detection_camera_matrix"
            ],
            dtype=np.float64,
        ).reshape(3, 3)


def bracketing_successes(
    track: list[dict[str, Any] | None], frame_index: int
) -> tuple[int, int] | None:
    before = next(
        (index for index in range(frame_index - 1, -1, -1) if track[index] is not None),
        None,
    )
    after = next(
        (
            index
            for index in range(frame_index + 1, len(track))
            if track[index] is not None
        ),
        None,
    )
    if before is None or after is None:
        return None
    return before, after


def draw_edges(
    image: np.ndarray,
    vertices: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    projected = cv2.projectPoints(
        vertices, rvec, tvec, camera_matrix, np.zeros(5, dtype=np.float64)
    )[0].reshape(-1, 2)
    for a, b in CUBE_EDGES:
        cv2.line(
            image,
            tuple(np.round(projected[a]).astype(int)),
            tuple(np.round(projected[b]).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )


def recover_one(
    loader: FrameLoader,
    frame_index: int,
    camera_name: str,
    cube_name: str,
    track: list[dict[str, Any] | None],
    timestamps: list[float],
    config: Any,
    tag_corner_map: dict[int, np.ndarray],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bracket = bracketing_successes(track, frame_index)
    if bracket is None:
        raise RuntimeError(f"Frame {frame_index} has no bracketing poses")
    before_index, after_index = bracket
    before_pose = track[before_index]
    after_pose = track[after_index]
    assert before_pose is not None and after_pose is not None
    denominator = timestamps[after_index] - timestamps[before_index]
    alpha = (
        (timestamps[frame_index] - timestamps[before_index]) / denominator
        if np.isfinite(denominator) and abs(denominator) > 1e-9
        else (frame_index - before_index) / (after_index - before_index)
    )
    temporal_rvec, temporal_tvec = interpolate_pose(before_pose, after_pose, alpha)
    camera_matrix = loader.camera_matrix(frame_index, camera_name)
    target_gray = loader.gray(frame_index, camera_name)
    tag_ids = sorted(
        set(before_pose.get("tag_ids", [])) | set(after_pose.get("tag_ids", []))
    )
    flow_object_chunks = []
    flow_image_chunks = []
    flow_groups: list[dict[str, Any]] = []
    anchor_data: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    flow_stats: dict[str, Any] = {}
    for label, anchor_index, anchor_pose in (
        ("before", before_index, before_pose),
        ("after", after_index, after_pose),
    ):
        anchor_gray = loader.gray(anchor_index, camera_name)
        anchor_data.append((label, anchor_gray, anchor_pose))
        flow_stats[label] = {}
        for tag_id in tag_ids:
            source_points, object_points = tag_feature_points(
                anchor_gray,
                anchor_pose,
                tag_corner_map,
                [tag_id],
                camera_matrix,
            )
            tracked_object, tracked_image, stats = track_features(
                anchor_gray, target_gray, source_points, object_points
            )
            flow_stats[label][str(tag_id)] = stats
            if len(tracked_object):
                flow_object_chunks.append(tracked_object)
                flow_image_chunks.append(tracked_image)
                flow_groups.append(
                    {
                        "anchor": label,
                        "anchor_index": anchor_index,
                        "tag_id": int(tag_id),
                        "object_points": tracked_object,
                        "image_points": tracked_image,
                    }
                )
    flow_object = (
        np.vstack(flow_object_chunks)
        if flow_object_chunks
        else np.empty((0, 3), dtype=np.float64)
    )
    flow_image = (
        np.vstack(flow_image_chunks)
        if flow_image_chunks
        else np.empty((0, 2), dtype=np.float64)
    )
    flow_solution = refine_pose_with_flow(
        flow_object,
        flow_image,
        temporal_rvec,
        temporal_tvec,
        camera_matrix,
    )
    planar_details: dict[str, Any] = {}
    photometric_details: dict[str, Any] = {}
    photometric_source = False
    if flow_solution is None:
        planar_solution = refine_pose_from_planar_flow_groups(
            flow_groups,
            flow_object,
            flow_image,
            temporal_rvec,
            temporal_tvec,
            camera_matrix,
            tag_corner_map,
        )
        if planar_solution is None:
            flow_rvec, flow_tvec = temporal_rvec, temporal_tvec
            flow_active = np.zeros(len(flow_object), dtype=bool)
            flow_source = False
            flow_method = "none"
        else:
            flow_rvec, flow_tvec, flow_active, planar_details = planar_solution
            flow_source = True
            flow_method = "planar_homography_ippe"
    else:
        flow_rvec, flow_tvec, flow_active = flow_solution
        flow_source = True
        flow_method = "sparse_lk_seeded_lm"
    if not flow_source:
        photometric_solution = refine_pose_with_tag_templates(
            target_gray,
            anchor_data,
            config,
            tag_corner_map,
            tag_ids,
            temporal_rvec,
            temporal_tvec,
            camera_matrix,
        )
        if photometric_solution is not None:
            flow_rvec, flow_tvec, photometric_details = photometric_solution
            photometric_source = True
            flow_method = "known_tag_template_ncc"

    distance = make_edge_distance(target_gray)
    cube_edge_points, tag_edge_points = model_edge_points(
        tuple(float(value) for value in config.box_dims),
        tag_corner_map,
        tag_ids,
        temporal_rvec,
        temporal_tvec,
    )
    candidates = []
    if flow_source:
        for name, seed_rvec, seed_tvec in (
            ("temporal", temporal_rvec, temporal_tvec),
            ("flow", flow_rvec, flow_tvec),
        ):
            candidate = optimize_visible_edges(
                distance,
                cube_edge_points,
                tag_edge_points,
                seed_rvec,
                seed_tvec,
                temporal_rvec,
                temporal_tvec,
                camera_matrix,
                flow_object[flow_active],
                flow_image[flow_active],
            )
            candidate["seed"] = name
            candidate["temporal_translation_delta_mm"] = float(
                np.linalg.norm(
                    candidate["tvec"].reshape(3) - temporal_tvec.reshape(3)
                )
            )
            candidate["temporal_rotation_delta_deg"] = rotation_delta_deg(
                temporal_rvec, candidate["rvec"]
            )
            if (
                candidate["temporal_translation_delta_mm"]
                <= MAX_TEMPORAL_TRANSLATION_DELTA_MM
                and candidate["temporal_rotation_delta_deg"]
                <= MAX_TEMPORAL_ROTATION_DELTA_DEG
                and candidate["flow_median_after"]
                <= max(
                    MAX_FLOW_FINAL_MEDIAN_PX,
                    candidate["flow_median_before"] + 0.75,
                )
            ):
                candidates.append(candidate)
    if not candidates:
        image_source = flow_source or photometric_source
        fallback_rvec = flow_rvec if image_source else temporal_rvec
        fallback_tvec = flow_tvec if image_source else temporal_tvec
        cube_cost = robust_distance_cost(
            sampled_distance(
                distance,
                cube_edge_points,
                fallback_rvec,
                fallback_tvec,
                camera_matrix,
            )
        )
        tag_cost = robust_distance_cost(
            sampled_distance(
                distance,
                tag_edge_points,
                fallback_rvec,
                fallback_tvec,
                camera_matrix,
            )
        ) if len(tag_edge_points) else cube_cost
        flow_errors = project_errors(
            flow_object[flow_active] if flow_source else flow_object[:0],
            flow_image[flow_active] if flow_source else flow_image[:0],
            fallback_rvec,
            fallback_tvec,
            camera_matrix,
        )
        flow_median = (
            float(np.median(flow_errors)) if len(flow_errors) else float("inf")
        )
        candidates.append(
            {
                "rvec": fallback_rvec,
                "tvec": fallback_tvec,
                "objective_before": cube_cost,
                "objective_after": cube_cost,
                "cube_edge_cost_before": cube_cost,
                "cube_edge_cost_after": cube_cost,
                "tag_edge_cost_before": tag_cost,
                "tag_edge_cost_after": tag_cost,
                "flow_median_before": flow_median,
                "flow_median_after": flow_median,
                "seed": (
                    "raw_flow"
                    if flow_source
                    else "tag_template_ncc"
                    if photometric_source
                    else "raw_temporal_prediction"
                ),
                "temporal_translation_delta_mm": float(
                    np.linalg.norm(fallback_tvec.reshape(3) - temporal_tvec.reshape(3))
                ),
                "temporal_rotation_delta_deg": rotation_delta_deg(
                    temporal_rvec, fallback_rvec
                ),
            }
        )
    selected = min(candidates, key=lambda value: value["objective_after"])
    measured = bool(
        photometric_source
        or (
            flow_source
            and np.isfinite(selected["flow_median_after"])
            and selected["flow_median_after"] <= MAX_FLOW_FINAL_MEDIAN_PX
        )
    )
    if measured and photometric_source:
        source = "rgb_known_tag_template_photometric_refine"
    elif measured and flow_method == "planar_homography_ippe":
        source = "rgb_planar_homography_flow_visible_edge_refine"
    elif measured:
        source = "rgb_bidirectional_flow_visible_edge_refine"
    else:
        source = "se3_temporal_prediction"
    transform = pose_matrix(selected["rvec"], selected["tvec"])
    result = {
        "success": True,
        "rvec": selected["rvec"],
        "tvec": selected["tvec"],
        "T": transform,
        "reproj_error": (
            float(selected["flow_median_after"])
            if np.isfinite(selected["flow_median_after"])
            else float("nan")
        ),
        "n_tags": 0,
        "n_inliers": int(flow_active.sum()),
        "detections": [],
        "tag_ids": tag_ids,
        "visible_faces": set(before_pose["visible_faces"]) | set(after_pose["visible_faces"]),
        "predicted": not measured,
        "measured": measured,
        "pose_filled": True,
        "pose_backend": source,
        "pose_source": source,
        "failure_reason": "",
        "recovery_algorithm": RECOVERY_ALGORITHM,
        "recovery_before_frame": before_index,
        "recovery_after_frame": after_index,
        "recovery_alpha": float(alpha),
        "flow_points_raw": int(len(flow_object)),
        "flow_inliers": int(flow_active.sum()),
        "flow_tracking_stats": flow_stats,
        "flow_method": flow_method,
        "planar_flow_details": planar_details,
        "photometric_details": photometric_details,
        "flow_reproj_median_px": selected["flow_median_after"],
        "cube_edge_cost_before": selected["cube_edge_cost_before"],
        "cube_edge_cost_after": selected["cube_edge_cost_after"],
        "tag_edge_cost_before": selected["tag_edge_cost_before"],
        "tag_edge_cost_after": selected["tag_edge_cost_after"],
        "edge_objective_before": selected["objective_before"],
        "edge_objective_after": selected["objective_after"],
        "temporal_translation_delta_mm": selected["temporal_translation_delta_mm"],
        "temporal_rotation_delta_deg": selected["temporal_rotation_delta_deg"],
        "recovery_seed": selected["seed"],
    }
    metrics = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "rvec", "tvec", "T", "detections", "visible_faces",
        }
    }
    metrics.update(
        {
            "frame_index": frame_index,
            "camera_name": camera_name,
            "cube_name": cube_name,
            "rvec": selected["rvec"].reshape(3).tolist(),
            "tvec": selected["tvec"].reshape(3).tolist(),
        }
    )

    image = loader.image(frame_index, camera_name)
    vertices = cube_vertices(tuple(float(value) for value in config.box_dims))
    draw_edges(image, vertices, temporal_rvec, temporal_tvec, camera_matrix, (0, 0, 255), 2)
    draw_edges(image, vertices, selected["rvec"], selected["tvec"], camera_matrix, (0, 255, 0), 3)
    cv2.putText(
        image,
        f"frame={frame_index} {cube_name} source={source}",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        (
            f"cube edge {selected['cube_edge_cost_before']:.2f}->{selected['cube_edge_cost_after']:.2f} "
            f"flow={selected['flow_median_after']:.2f}px inliers={int(flow_active.sum())}"
        ),
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    safe_key = f"{camera_name}__{cube_name}"
    output_path = output_dir / safe_key / f"frame_{frame_index:04d}.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    metrics["overlay_path"] = str(output_path.resolve())
    return result, metrics


def apply_recoveries(
    pkl_path: Path,
    recoveries: dict[str, dict[int, dict[str, Any]]],
    key_parts: dict[str, tuple[str, str]],
    model_by_key: dict[str, tuple[Any, dict[int, np.ndarray]]],
    recovery_summary: dict[str, Any],
) -> None:
    temporary = pkl_path.with_name(f".{pkl_path.name}.026-rewrite.tmp")
    temporary.unlink(missing_ok=True)
    source_size = pkl_path.stat().st_size
    required = source_size + 2 * 1024**3
    if shutil.disk_usage(pkl_path.parent).free < required:
        raise RuntimeError("Not enough free space for atomic 026 rewrite")
    key_by_parts = {value: key for key, value in key_parts.items()}
    frame_index = 0
    with pkl_path.open("rb") as source, temporary.open("wb") as destination:
        header = pickle.load(source)
        updated_header = copy.deepcopy(header)
        updated_header.setdefault("metadata", {})["rgb_pose_recovery"] = {
            "algorithm": RECOVERY_ALGORITHM,
            "completed": True,
            "written_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "quality_summary": copy.deepcopy(recovery_summary),
        }
        pickle.dump(updated_header, destination, protocol=pickle.HIGHEST_PROTOCOL)
        while True:
            try:
                record = pickle.load(source)
            except EOFError:
                break
            if isinstance(record, dict) and record.get("type") == "frame_pair":
                for camera_name, camera_record in record["cameras"].items():
                    changed = False
                    for cube in camera_record["offline_pos"]["cube_results"]:
                        parts = (camera_name, str(cube["cube_name"]))
                        key = key_by_parts[parts]
                        recovered = recoveries.get(key, {}).get(frame_index)
                        if recovered is not None:
                            cube["result"] = copy.deepcopy(recovered)
                            changed = True
                    if changed:
                        encoded = camera_record[UNDISTORTED_IMAGE_FIELD]
                        overlay = cv2.imdecode(
                            np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR
                        )
                        if overlay is None:
                            raise ValueError(
                                f"Could not decode {camera_name} frame {frame_index}"
                            )
                        camera_matrix = np.asarray(
                            camera_record["undistorted_visualization_meta"][
                                "detection_camera_matrix"
                            ],
                            dtype=np.float64,
                        ).reshape(3, 3)
                        colors = ((0, 255, 0), (255, 128, 0), (255, 0, 255))
                        for cube_index, cube in enumerate(
                            camera_record["offline_pos"]["cube_results"]
                        ):
                            result = cube["result"]
                            if not result.get("success", False):
                                continue
                            key = key_by_parts[(camera_name, str(cube["cube_name"]))]
                            config, _tag_corner_map = model_by_key[key]
                            draw_edges(
                                overlay,
                                cube_vertices(
                                    tuple(float(value) for value in config.box_dims)
                                ),
                                np.asarray(result["rvec"], dtype=np.float64),
                                np.asarray(result["tvec"], dtype=np.float64),
                                camera_matrix,
                                colors[cube_index % len(colors)],
                                3,
                            )
                            cv2.putText(
                                overlay,
                                (
                                    f"{Path(str(cube['cube_name'])).name}: "
                                    f"{result.get('pose_source', result.get('pose_backend', 'pose'))}"
                                ),
                                (18, 32 + 28 * cube_index),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.58,
                                colors[cube_index % len(colors)],
                                2,
                                cv2.LINE_AA,
                            )
                        ok, overlay_jpeg = cv2.imencode(
                            ".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 92]
                        )
                        if not ok:
                            raise RuntimeError(
                                f"Could not encode {camera_name} frame {frame_index}"
                            )
                        camera_record[UNDISTORTED_OVERLAY_FIELD] = overlay_jpeg.tobytes()
                frame_index += 1
            elif isinstance(record, dict) and record.get("type") == "footer":
                record["rgb_pose_recovery"] = {
                    "algorithm": RECOVERY_ALGORITHM,
                    "frame_pair_count": frame_index,
                    "recovered_counts": {
                        key: len(value) for key, value in recoveries.items()
                    },
                    "quality_summary": copy.deepcopy(recovery_summary),
                }
            pickle.dump(record, destination, protocol=pickle.HIGHEST_PROTOCOL)
        destination.flush()
        os.fsync(destination.fileno())
    temporary.replace(pkl_path)


def distribution(values: list[float]) -> dict[str, float | int] | None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(finite):
        return None
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(np.max(finite)),
    }


def main() -> None:
    args = parse_args()
    pkl_path = args.pkl_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    header, offsets, tracks, timestamps, key_parts = build_index(pkl_path)
    loader = FrameLoader(pkl_path, offsets)
    cube_paths_by_camera = header["metadata"]["camera_cube_configs"]
    model_by_key: dict[str, tuple[Any, dict[int, np.ndarray]]] = {}
    for key, (camera_name, cube_name) in key_parts.items():
        cube_path = next(
            Path(value)
            for value in cube_paths_by_camera[camera_name]
            if Path(value).name == cube_name
        )
        config, _face_sets = load_cube_config(str(cube_path / "config.json"))
        model_by_key[key] = (config, build_tag_corner_map(config))

    recoveries: dict[str, dict[int, dict[str, Any]]] = {}
    metrics: list[dict[str, Any]] = []
    started = time.perf_counter()
    for key, track in tracks.items():
        camera_name, cube_name = key_parts[key]
        config, tag_corner_map = model_by_key[key]
        for frame_index, pose in enumerate(track):
            if pose is not None:
                continue
            recovered, frame_metrics = recover_one(
                loader,
                frame_index,
                camera_name,
                cube_name,
                track,
                timestamps[key],
                config,
                tag_corner_map,
                output_dir,
            )
            recoveries.setdefault(key, {})[frame_index] = recovered
            track[frame_index] = minimal_pose(recovered)
            metrics.append(frame_metrics)
            print(
                f"[INFO] {key} frame={frame_index} "
                f"source={recovered['pose_source']} "
                f"edge={recovered['cube_edge_cost_before']:.2f}->"
                f"{recovered['cube_edge_cost_after']:.2f} "
                f"flow={recovered['flow_reproj_median_px']:.2f}px "
                f"inliers={recovered['flow_inliers']}"
            )

    source_counts = Counter(metric["pose_source"] for metric in metrics)
    summary = {
        "pkl_path": str(pkl_path),
        "algorithm": RECOVERY_ALGORITHM,
        "apply": bool(args.apply),
        "failure_pose_count_before": len(metrics),
        "recovered_pose_count": len(metrics),
        "source_counts": dict(source_counts),
        "cube_edge_cost_before": distribution(
            [float(metric["cube_edge_cost_before"]) for metric in metrics]
        ),
        "cube_edge_cost_after": distribution(
            [float(metric["cube_edge_cost_after"]) for metric in metrics]
        ),
        "flow_reproj_median_px": distribution(
            [float(metric["flow_reproj_median_px"]) for metric in metrics]
        ),
        "flow_inliers": distribution(
            [float(metric["flow_inliers"]) for metric in metrics]
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report = {"summary": summary, "frames": metrics}
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{pkl_path.stem}_{RECOVERY_ALGORITHM}.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[INFO] Report: {report_path}")
    if args.apply:
        apply_recoveries(
            pkl_path,
            recoveries,
            key_parts,
            model_by_key,
            summary,
        )
        print(f"[INFO] Applied recoveries atomically to {pkl_path}")


if __name__ == "__main__":
    main()
