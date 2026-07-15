#!/usr/bin/env python3
"""Refine middle-camera cube poses from precise tag borders and sequence motion.

The DeepTag internal grid is useful for detection, but points on one planar tag do
not constrain the cube depth direction well.  This pass re-detects the physical
AprilTag border with sub-pixel refinement, keeps weak observations from an
oblique second face, and jointly optimizes all frame poses with a smooth-motion
prior.  It writes reports by default and only modifies the source PKL with
``--apply``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
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
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

import aprilcube
from aprilcube.detect import build_tag_corner_map, load_cube_config
from aprilcube.deeptag_pose import DeepTagPoseBackend


APRILCUBE_ROOT = Path(__file__).resolve().parent.parent
CAMERA_NAME = "middle_finger_cam"
CUBE_NAME = "cube_april_36h11_0_5_1x1x1_15mm"
UNDISTORTED_IMAGE_FIELD = "undistorted_image_jpeg"
REFINEMENT_ALGORITHM = "empirical_multiface_tag_border_temporal_bundle_v3"

CORNER_LOSS_SCALE_PX = 3.0
ROTATION_ACCEL_WEIGHT = 180.0
TRANSLATION_ACCEL_WEIGHT = 2.5
MISSING_ROTATION_PRIOR_WEIGHT = 3.0
MISSING_TRANSLATION_PRIOR_WEIGHT = 0.25
MAX_LOCAL_ROTATION_DELTA_RAD = 0.55
MAX_LOCAL_TRANSLATION_DELTA_MM = 25.0
MAX_ACCEPTED_ROTATION_DELTA_DEG = 35.0
MAX_ACCEPTED_TRANSLATION_DELTA_MM = 30.0

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
    parser = argparse.ArgumentParser(
        description=(
            "Jointly refine middle-camera single-tag cube poses using sub-pixel "
            "tag borders, weak second-face observations, and temporal smoothness."
        )
    )
    parser.add_argument("pkl_path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=APRILCUBE_ROOT / "outputs/027_middle_single_face_refine",
    )
    parser.add_argument(
        "--reuse-detections",
        action="store_true",
        help="Reuse the high-quality detection cache when it matches this PKL.",
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


def rotation_delta_deg(rvec_a: Any, rvec_b: Any) -> float:
    rotation_a = Rotation.from_rotvec(np.asarray(rvec_a).reshape(3))
    rotation_b = Rotation.from_rotvec(np.asarray(rvec_b).reshape(3))
    return float(np.degrees((rotation_b * rotation_a.inv()).magnitude()))


def cube_vertices(box_dims: Any) -> np.ndarray:
    x, y, z = (float(value) * 0.5 for value in box_dims)
    return np.asarray(
        [
            [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
            [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z],
        ],
        dtype=np.float64,
    )


def camera_intrinsics(camera_matrix: np.ndarray) -> dict[str, float]:
    return {
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
    }


def decode_image(camera_record: dict[str, Any]) -> np.ndarray:
    encoded = camera_record[UNDISTORTED_IMAGE_FIELD]
    image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode embedded undistorted image")
    return image


def target_result(camera_record: dict[str, Any]) -> dict[str, Any]:
    for cube in camera_record["offline_pos"]["cube_results"]:
        if str(cube["cube_name"]) == CUBE_NAME:
            return cube["result"]
    raise KeyError(f"Missing {CUBE_NAME} in {CAMERA_NAME}")


def build_index(
    pkl_path: Path,
) -> tuple[dict[str, Any], list[int], np.ndarray, np.ndarray, np.ndarray, list[float]]:
    offsets: list[int] = []
    poses: list[np.ndarray] = []
    camera_matrix: np.ndarray | None = None
    timestamps: list[float] = []
    with pkl_path.open("rb") as file:
        header = pickle.load(file)
        while True:
            offset = file.tell()
            try:
                record = pickle.load(file)
            except EOFError:
                break
            if not isinstance(record, dict) or record.get("type") != "frame_pair":
                continue
            camera_record = record["cameras"][CAMERA_NAME]
            result = target_result(camera_record)
            if not result.get("success", False):
                raise ValueError(
                    f"Frame {len(offsets)} has no initial {CAMERA_NAME} pose; "
                    "run 026 recovery first"
                )
            offsets.append(offset)
            poses.append(
                np.r_[
                    np.asarray(result["rvec"], dtype=np.float64).reshape(3),
                    np.asarray(result["tvec"], dtype=np.float64).reshape(3),
                ]
            )
            timestamps.append(float(camera_record["capture_timestamp"]))
            frame_matrix = np.asarray(
                camera_record["undistorted_visualization_meta"][
                    "detection_camera_matrix"
                ],
                dtype=np.float64,
            ).reshape(3, 3)
            if camera_matrix is None:
                camera_matrix = frame_matrix
            elif not np.allclose(camera_matrix, frame_matrix):
                raise ValueError("Detection camera matrix changes within the recording")
    if camera_matrix is None or not offsets:
        raise ValueError(f"No frame pairs in {pkl_path}")
    return (
        header,
        offsets,
        np.asarray(poses, dtype=np.float64),
        camera_matrix,
        np.asarray(timestamps, dtype=np.float64),
        timestamps,
    )


class FrameLoader:
    def __init__(self, pkl_path: Path, offsets: list[int]) -> None:
        self.pkl_path = pkl_path
        self.offsets = offsets

    def record(self, frame_index: int) -> dict[str, Any]:
        with self.pkl_path.open("rb") as file:
            file.seek(self.offsets[frame_index])
            return pickle.load(file)

    def camera_record(self, frame_index: int) -> dict[str, Any]:
        return self.record(frame_index)["cameras"][CAMERA_NAME]

    def image(self, frame_index: int) -> np.ndarray:
        return decode_image(self.camera_record(frame_index))


def cache_fingerprint(pkl_path: Path, frame_count: int) -> dict[str, Any]:
    stat = pkl_path.stat()
    return {
        "pkl_path": str(pkl_path),
        "pkl_size": int(stat.st_size),
        "frame_count": int(frame_count),
        "camera": CAMERA_NAME,
        "cube": CUBE_NAME,
        "detector": "opencv_aruco_subpixel_adaptive_clahe",
    }


def load_detection_cache(
    cache_path: Path,
    fingerprint: dict[str, Any],
) -> list[list[dict[str, Any]]] | None:
    if not cache_path.exists():
        return None
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    if data.get("fingerprint") != fingerprint:
        return None
    frames = data.get("frames")
    if not isinstance(frames, list) or len(frames) != fingerprint["frame_count"]:
        return None
    return frames


def detect_tag_borders(
    loader: FrameLoader,
    frame_count: int,
    cube_path: Path,
    camera_matrix: np.ndarray,
    target_ids: set[int],
    cache_path: Path,
    fingerprint: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    detector = aprilcube.detector(
        cube_path,
        intrinsic_cfg=camera_intrinsics(camera_matrix),
        dist_coeffs=np.zeros(5, dtype=np.float64),
        enable_filter=False,
        fast=False,
    )
    frames: list[list[dict[str, Any]]] = []
    started = time.perf_counter()
    for frame_index in range(frame_count):
        image = loader.image(frame_index)
        detected = detector.detect_tags(image, adaptive_clahe=True)["detections"]
        by_id: dict[int, np.ndarray] = {}
        for tag_id_value, corners_value in detected:
            tag_id = int(tag_id_value)
            if tag_id in target_ids:
                by_id[tag_id] = np.asarray(corners_value, dtype=np.float64).reshape(4, 2)
        frame_detections = [
            {"tag_id": tag_id, "corners": by_id[tag_id].tolist()}
            for tag_id in sorted(by_id)
        ]
        frames.append(frame_detections)
        if frame_index % 20 == 0 or frame_index + 1 == frame_count:
            elapsed = time.perf_counter() - started
            print(
                f"[INFO] Detecting precise tag borders {frame_index + 1}/{frame_count} "
                f"ids={sorted(by_id)} elapsed={elapsed:.1f}s",
                flush=True,
            )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"fingerprint": fingerprint, "frames": frames}),
        encoding="utf-8",
    )
    return frames


def load_or_detect_deeptag_points(
    loader: FrameLoader,
    frame_count: int,
    cube_path: Path,
    camera_matrix: np.ndarray,
    target_ids: set[int],
    cache_path: Path,
    fingerprint: dict[str, Any],
    reuse: bool,
) -> list[list[dict[str, Any]]]:
    if reuse and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        frames = cached.get("frames")
        cached_fingerprint = cached.get("fingerprint")
        # Accept the cache produced during development before fingerprints were
        # added; frame count and source PKL are checked by the caller.
        if (
            isinstance(frames, list)
            and len(frames) == frame_count
            and (cached_fingerprint is None or cached_fingerprint == fingerprint)
        ):
            print(f"[INFO] Reused DeepTag point cache: {cache_path}")
            return frames

    backend = DeepTagPoseBackend(camera_matrix, [cube_path])
    frames: list[list[dict[str, Any]]] = []
    started = time.perf_counter()
    for frame_index in range(frame_count):
        with contextlib.redirect_stdout(io.StringIO()):
            decoded = backend.engine.process(loader.image(frame_index), detect_scale=None)
        frame_points = []
        for tag in decoded:
            tag_id = int(tag.get("tag_id", -1))
            if not tag.get("is_valid", False) or tag_id not in target_ids:
                continue
            image_points = np.asarray(
                tag["keypoints_in_images"], dtype=np.float64
            ).reshape(-1, 2)
            detected_flags = np.asarray(
                tag.get("keypoints_detected_flags", [True] * len(image_points)),
                dtype=bool,
            )
            normalized_points = np.asarray(
                backend.engine.pose_solver_dict[
                    len(image_points)
                ].fine_grid_points_anno,
                dtype=np.float64,
            ).reshape(-1, 2)
            frame_points.append(
                {
                    "tag_id": tag_id,
                    "image_points": image_points.tolist(),
                    "detected_flags": detected_flags.tolist(),
                    "normalized_points": normalized_points.tolist(),
                }
            )
        frames.append(frame_points)
        if frame_index % 25 == 0 or frame_index + 1 == frame_count:
            print(
                f"[INFO] DeepTag planar points {frame_index + 1}/{frame_count} "
                f"ids={[value['tag_id'] for value in frame_points]} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    cache_path.write_text(
        json.dumps({"fingerprint": fingerprint, "frames": frames}),
        encoding="utf-8",
    )
    return frames


def deeptag_outer_border(detection: dict[str, Any]) -> dict[str, Any] | None:
    normalized = np.asarray(
        detection["normalized_points"], dtype=np.float64
    ).reshape(-1, 2)
    image_points = np.asarray(
        detection["image_points"], dtype=np.float64
    ).reshape(-1, 2)
    detected = np.asarray(detection["detected_flags"], dtype=bool).reshape(-1)
    if int(detected.sum()) < 12:
        return None
    homography, mask = cv2.findHomography(
        normalized[detected],
        image_points[detected],
        method=cv2.RANSAC,
        ransacReprojThreshold=2.5,
        maxIters=2000,
        confidence=0.999,
    )
    if homography is None or mask is None or int(mask.sum()) < 12:
        return None
    predicted = cv2.perspectiveTransform(
        normalized[detected].reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    inliers = mask.reshape(-1).astype(bool)
    errors = np.linalg.norm(predicted - image_points[detected], axis=1)
    inlier_median = float(np.median(errors[inliers]))
    if not np.isfinite(inlier_median) or inlier_median > 3.5:
        return None
    # DeepTag coordinates are x-right/y-up.  Extrapolating the cell-center
    # grid to +/-0.5 recovers the physical outer black border of the tag.
    outer_normalized = np.asarray(
        [[-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]],
        dtype=np.float64,
    ).reshape(-1, 1, 2)
    corners = cv2.perspectiveTransform(outer_normalized, homography).reshape(4, 2)
    return {
        "tag_id": int(detection["tag_id"]),
        "corners": corners.tolist(),
        "source": "deeptag_homography_outer_border",
        "homography_inliers": int(inliers.sum()),
        "homography_median_px": inlier_median,
    }


def fuse_border_detections(
    cv2_frames: list[list[dict[str, Any]]],
    deeptag_frames: list[list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    fused_frames: list[list[dict[str, Any]]] = []
    for cv2_detections, deep_points in zip(cv2_frames, deeptag_frames):
        cv2_by_id = {int(value["tag_id"]): value for value in cv2_detections}
        deep_by_id: dict[int, dict[str, Any]] = {}
        for points in deep_points:
            border = deeptag_outer_border(points)
            if border is not None:
                deep_by_id[int(border["tag_id"])] = border
        frame = []
        for tag_id in sorted(set(cv2_by_id) | set(deep_by_id)):
            cv_value = cv2_by_id.get(tag_id)
            deep_value = deep_by_id.get(tag_id)
            if cv_value is not None and deep_value is not None:
                cv_corners = np.asarray(cv_value["corners"], dtype=np.float64)
                deep_corners = np.asarray(deep_value["corners"], dtype=np.float64)
                disagreement = np.linalg.norm(cv_corners - deep_corners, axis=1)
                median_disagreement = float(np.median(disagreement))
                if median_disagreement <= 12.0:
                    corners = 0.55 * cv_corners + 0.45 * deep_corners
                    source = "cv2_subpixel+deeptag_homography"
                else:
                    corners = cv_corners
                    source = "cv2_subpixel_deeptag_disagreed"
                frame.append(
                    {
                        "tag_id": tag_id,
                        "corners": corners.tolist(),
                        "source": source,
                        "detector_disagreement_median_px": median_disagreement,
                    }
                )
            elif cv_value is not None:
                frame.append(
                    {
                        "tag_id": tag_id,
                        "corners": cv_value["corners"],
                        "source": "cv2_subpixel",
                    }
                )
            elif deep_value is not None:
                frame.append(deep_value)
        fused_frames.append(frame)
    return fused_frames


def cube_from_tag_transform(tag_corners: np.ndarray) -> np.ndarray:
    top_left, _top_right, bottom_right, bottom_left = np.asarray(
        tag_corners, dtype=np.float64
    ).reshape(4, 3)
    tag_size = float(np.linalg.norm(bottom_right - bottom_left))
    x_axis = (bottom_right - bottom_left) / tag_size
    y_axis = (top_left - bottom_left) / tag_size
    z_axis = np.cross(x_axis, y_axis)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    transform[:3, 3] = np.asarray(tag_corners, dtype=np.float64).mean(axis=0)
    return transform


def planar_tag_pose_near_expected(
    corners: np.ndarray,
    tag_size_mm: float,
    camera_matrix: np.ndarray,
    expected_camera_from_tag: np.ndarray,
) -> np.ndarray | None:
    half = float(tag_size_mm) * 0.5
    local_corners = np.asarray(
        [[-half, half, 0.0], [half, half, 0.0],
         [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )
    try:
        solved = cv2.solvePnPGeneric(
            local_corners,
            np.asarray(corners, dtype=np.float64).reshape(4, 2),
            camera_matrix,
            np.zeros(5, dtype=np.float64),
            flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        return None
    if not solved[0]:
        return None
    expected_rotation = Rotation.from_matrix(expected_camera_from_tag[:3, :3])
    candidates = []
    for rvec, tvec in zip(solved[1], solved[2]):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                local_corners,
                np.asarray(corners, dtype=np.float64).reshape(4, 2),
                camera_matrix,
                np.zeros(5, dtype=np.float64),
                rvec,
                tvec,
            )
        except cv2.error:
            pass
        transform = pose_matrix(rvec, tvec)
        rotation_error = float(
            np.degrees(
                (
                    Rotation.from_matrix(transform[:3, :3])
                    * expected_rotation.inv()
                ).magnitude()
            )
        )
        translation_error = float(
            np.linalg.norm(transform[:3, 3] - expected_camera_from_tag[:3, 3])
        )
        candidates.append((rotation_error + 0.5 * translation_error, transform))
    return min(candidates, key=lambda value: value[0])[1] if candidates else None


def calibrate_observed_tag_geometry(
    detections: list[list[dict[str, Any]]],
    initial_poses: np.ndarray,
    tag_corner_map: dict[int, np.ndarray],
    tag_size_mm: float,
    camera_matrix: np.ndarray,
) -> tuple[dict[int, np.ndarray], set[int], dict[str, Any]]:
    counts = Counter(
        int(detection["tag_id"])
        for frame in detections
        for detection in frame
    )
    if not counts:
        return tag_corner_map, set(), {"failure": "no_tag_detections"}
    reference_tag = int(counts.most_common(1)[0][0])
    ideal_transforms = {
        tag_id: cube_from_tag_transform(corners)
        for tag_id, corners in tag_corner_map.items()
    }
    observations: dict[int, list[np.ndarray]] = {
        tag_id: [] for tag_id in tag_corner_map if tag_id != reference_tag
    }
    for frame_index, frame in enumerate(detections):
        by_id = {int(value["tag_id"]): value for value in frame}
        if reference_tag not in by_id:
            continue
        camera_from_cube_initial = pose_matrix(
            initial_poses[frame_index, :3], initial_poses[frame_index, 3:]
        )
        camera_from_reference = planar_tag_pose_near_expected(
            by_id[reference_tag]["corners"],
            tag_size_mm,
            camera_matrix,
            camera_from_cube_initial @ ideal_transforms[reference_tag],
        )
        if camera_from_reference is None:
            continue
        camera_from_cube = (
            camera_from_reference @ np.linalg.inv(ideal_transforms[reference_tag])
        )
        for tag_id, values in by_id.items():
            if tag_id == reference_tag or tag_id not in observations:
                continue
            camera_from_tag = planar_tag_pose_near_expected(
                values["corners"],
                tag_size_mm,
                camera_matrix,
                camera_from_cube_initial @ ideal_transforms[tag_id],
            )
            if camera_from_tag is not None:
                observations[tag_id].append(
                    np.linalg.inv(camera_from_cube) @ camera_from_tag
                )

    calibrated_map = dict(tag_corner_map)
    accepted_ids = {reference_tag}
    report: dict[str, Any] = {"reference_tag_id": reference_tag, "tags": {}}
    half = float(tag_size_mm) * 0.5
    local_homogeneous = np.column_stack(
        (
            np.asarray(
                [[-half, half, 0.0], [half, half, 0.0],
                 [half, -half, 0.0], [-half, -half, 0.0]],
                dtype=np.float64,
            ),
            np.ones(4, dtype=np.float64),
        )
    )
    for tag_id, transforms in observations.items():
        tag_report: dict[str, Any] = {"observation_count": len(transforms)}
        if len(transforms) < 12:
            tag_report["accepted"] = False
            tag_report["reason"] = "too_few_joint_observations"
            report["tags"][str(tag_id)] = tag_report
            continue
        matrices = np.asarray([value[:3, :3] for value in transforms])
        translations = np.asarray([value[:3, 3] for value in transforms])
        keep = np.ones(len(transforms), dtype=bool)
        for _iteration in range(4):
            mean_rotation = Rotation.from_matrix(matrices[keep]).mean()
            median_translation = np.median(translations[keep], axis=0)
            angle_errors = np.degrees(
                (Rotation.from_matrix(matrices) * mean_rotation.inv()).magnitude()
            )
            translation_errors = np.linalg.norm(
                translations - median_translation, axis=1
            )
            angle_limit = max(8.0, float(np.percentile(angle_errors, 60)))
            translation_limit = max(
                4.0, float(np.percentile(translation_errors, 60))
            )
            keep = (angle_errors <= angle_limit) & (
                translation_errors <= translation_limit
            )
        calibrated = np.eye(4, dtype=np.float64)
        calibrated[:3, :3] = Rotation.from_matrix(matrices[keep]).mean().as_matrix()
        calibrated[:3, 3] = np.median(translations[keep], axis=0)
        delta = calibrated @ np.linalg.inv(ideal_transforms[tag_id])
        delta_rotation_deg = float(
            np.degrees(Rotation.from_matrix(delta[:3, :3]).magnitude())
        )
        delta_translation_mm = float(np.linalg.norm(delta[:3, 3]))
        accepted = bool(
            int(keep.sum()) >= 12
            and delta_rotation_deg <= 25.0
            and delta_translation_mm <= 6.0
        )
        tag_report.update(
            {
                "inlier_count": int(keep.sum()),
                "accepted": accepted,
                "delta_rotation_deg": delta_rotation_deg,
                "delta_translation_mm": delta_translation_mm,
                "cube_from_tag": calibrated.tolist(),
            }
        )
        if accepted:
            calibrated_map[tag_id] = (
                calibrated @ local_homogeneous.T
            ).T[:, :3]
            accepted_ids.add(tag_id)
        else:
            tag_report["reason"] = "relative_transform_inconsistent_with_cube"
        report["tags"][str(tag_id)] = tag_report
    return calibrated_map, accepted_ids, report


def detection_arrays(
    detections: list[dict[str, Any]],
    tag_corner_map: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not detections:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    areas = []
    for detection in detections:
        corners = np.asarray(detection["corners"], dtype=np.float64).reshape(4, 2)
        areas.append(abs(float(cv2.contourArea(corners.astype(np.float32)))))
    max_area = max(max(areas), 1.0)
    object_blocks = []
    image_blocks = []
    weight_blocks = []
    for detection, area in zip(detections, areas):
        tag_id = int(detection["tag_id"])
        object_blocks.append(
            np.asarray(tag_corner_map[tag_id], dtype=np.float64).reshape(4, 3)
        )
        image_blocks.append(
            np.asarray(detection["corners"], dtype=np.float64).reshape(4, 2)
        )
        # Keep the oblique second face influential, but let robust loss reject a
        # bad corner instead of allowing it to dominate the frontal face.
        tag_weight = 0.55 + 0.45 * np.sqrt(max(area, 1.0) / max_area)
        weight_blocks.append(np.full(4, tag_weight, dtype=np.float64))
    return (
        np.vstack(object_blocks),
        np.vstack(image_blocks),
        np.concatenate(weight_blocks),
    )


def project_points(
    object_points: np.ndarray,
    pose: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    if not len(object_points):
        return np.empty((0, 2), dtype=np.float64)
    return cv2.projectPoints(
        object_points,
        pose[:3],
        pose[3:],
        camera_matrix,
        np.zeros(5, dtype=np.float64),
    )[0].reshape(-1, 2)


def local_pose_refinement(
    initial_pose: np.ndarray,
    object_points: np.ndarray,
    image_points: np.ndarray,
    point_weights: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    if not len(object_points):
        return initial_pose.copy()
    sqrt_weights = np.sqrt(point_weights)[:, None]

    def residual(pose: np.ndarray) -> np.ndarray:
        return (
            (project_points(object_points, pose, camera_matrix) - image_points)
            * sqrt_weights
        ).reshape(-1)

    lower = initial_pose - np.r_[
        np.full(3, MAX_LOCAL_ROTATION_DELTA_RAD),
        np.full(3, MAX_LOCAL_TRANSLATION_DELTA_MM),
    ]
    upper = initial_pose + np.r_[
        np.full(3, MAX_LOCAL_ROTATION_DELTA_RAD),
        np.full(3, MAX_LOCAL_TRANSLATION_DELTA_MM),
    ]
    result = least_squares(
        residual,
        initial_pose,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=CORNER_LOSS_SCALE_PX,
        x_scale=np.asarray([0.08, 0.08, 0.08, 4.0, 4.0, 6.0]),
        max_nfev=100,
    )
    return np.asarray(result.x, dtype=np.float64)


def global_residual_layout(
    observations: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[int, list[int]]:
    starts = []
    cursor = 0
    for object_points, _image_points, _weights in observations:
        starts.append(cursor)
        cursor += 2 * len(object_points)
    cursor += 6 * max(0, len(observations) - 2)
    cursor += 6 * sum(1 for points, _image, _weights in observations if not len(points))
    return cursor, starts


def build_jacobian_sparsity(
    observations: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> lil_matrix:
    frame_count = len(observations)
    residual_count, starts = global_residual_layout(observations)
    sparsity = lil_matrix((residual_count, frame_count * 6), dtype=np.int8)
    cursor = 0
    for frame_index, (object_points, _image_points, _weights) in enumerate(observations):
        rows = 2 * len(object_points)
        if rows:
            sparsity[cursor : cursor + rows, frame_index * 6 : (frame_index + 1) * 6] = 1
        cursor += rows
    for frame_index in range(1, frame_count - 1):
        for neighbor in (frame_index - 1, frame_index, frame_index + 1):
            sparsity[cursor : cursor + 6, neighbor * 6 : (neighbor + 1) * 6] = 1
        cursor += 6
    for frame_index, (object_points, _image_points, _weights) in enumerate(observations):
        if not len(object_points):
            sparsity[cursor : cursor + 6, frame_index * 6 : (frame_index + 1) * 6] = 1
            cursor += 6
    return sparsity


def joint_sequence_refinement(
    initial_poses: np.ndarray,
    observations: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray, Any]:
    frame_count = len(initial_poses)

    def pseudo_huber_residual(values: np.ndarray) -> np.ndarray:
        """Apply robust loss to image errors while leaving motion quadratic."""
        scaled = values / CORNER_LOSS_SCALE_PX
        magnitude = np.sqrt(
            2.0
            * CORNER_LOSS_SCALE_PX**2
            * (np.sqrt(1.0 + scaled * scaled) - 1.0)
        )
        return np.copysign(magnitude, values)

    def residual(flat_poses: np.ndarray) -> np.ndarray:
        poses = flat_poses.reshape(frame_count, 6)
        blocks: list[np.ndarray] = []
        for pose, (object_points, image_points, point_weights) in zip(
            poses, observations
        ):
            if len(object_points):
                error = project_points(object_points, pose, camera_matrix) - image_points
                weighted = error * np.sqrt(point_weights)[:, None]
                blocks.append(pseudo_huber_residual(weighted).reshape(-1))
        for frame_index in range(1, frame_count - 1):
            second_difference = (
                poses[frame_index - 1] - 2.0 * poses[frame_index] + poses[frame_index + 1]
            )
            blocks.append(
                np.r_[
                    second_difference[:3] * ROTATION_ACCEL_WEIGHT,
                    second_difference[3:] * TRANSLATION_ACCEL_WEIGHT,
                ]
            )
        for frame_index, (object_points, _image_points, _weights) in enumerate(observations):
            if not len(object_points):
                delta = poses[frame_index] - initial_poses[frame_index]
                blocks.append(
                    np.r_[
                        delta[:3] * MISSING_ROTATION_PRIOR_WEIGHT,
                        delta[3:] * MISSING_TRANSLATION_PRIOR_WEIGHT,
                    ]
                )
        return np.concatenate(blocks)

    sparsity = build_jacobian_sparsity(observations).tocsr()
    result = least_squares(
        residual,
        initial_poses.reshape(-1),
        jac_sparsity=sparsity,
        loss="linear",
        x_scale=np.tile(
            np.asarray([0.05, 0.05, 0.05, 3.0, 3.0, 4.0]), frame_count
        ),
        max_nfev=120,
        verbose=1,
    )
    return np.asarray(result.x, dtype=np.float64).reshape(frame_count, 6), result


def visible_edges(
    vertices: np.ndarray,
    pose: np.ndarray,
) -> list[tuple[int, int]]:
    rotation = cv2.Rodrigues(pose[:3])[0]
    translation = pose[3:]
    visible_faces: list[set[int]] = []
    for face_vertices, normal_value in FACE_VERTEX_NORMALS:
        normal = np.asarray(normal_value, dtype=np.float64)
        center = vertices[np.asarray(face_vertices)].mean(axis=0)
        normal_camera = rotation @ normal
        center_camera = rotation @ center + translation
        if float(np.dot(normal_camera, center_camera)) < 0.0:
            visible_faces.append(set(face_vertices))
    return [
        edge
        for edge in CUBE_EDGES
        if any(edge[0] in face and edge[1] in face for face in visible_faces)
    ]


def edge_distance_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    edges = cv2.bitwise_or(
        cv2.Canny(blurred, 35, 105),
        cv2.Canny(blurred, 70, 180),
    )
    return cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)


def visible_edge_cost(
    distance: np.ndarray,
    vertices: np.ndarray,
    pose: np.ndarray,
    camera_matrix: np.ndarray,
) -> float:
    projected = project_points(vertices, pose, camera_matrix)
    samples = []
    for first, second in visible_edges(vertices, pose):
        alpha = np.linspace(0.08, 0.92, 28)[:, None]
        samples.append((1.0 - alpha) * projected[first] + alpha * projected[second])
    if not samples:
        return float("nan")
    points = np.vstack(samples)
    height, width = distance.shape
    valid = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] <= width - 1.0)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= height - 1.0)
    )
    if not np.any(valid):
        return float("nan")
    values = cv2.remap(
        distance,
        points[valid, 0].astype(np.float32).reshape(-1, 1),
        points[valid, 1].astype(np.float32).reshape(-1, 1),
        cv2.INTER_LINEAR,
    ).reshape(-1)
    values = np.minimum(values, 15.0)
    return float(np.mean(np.minimum(values, 5.0) ** 2))


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
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def draw_cube(
    image: np.ndarray,
    vertices: np.ndarray,
    pose: np.ndarray,
    camera_matrix: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    projected = project_points(vertices, pose, camera_matrix)
    for first, second in CUBE_EDGES:
        cv2.line(
            image,
            tuple(np.round(projected[first]).astype(int)),
            tuple(np.round(projected[second]).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )


def save_debug_images(
    output_dir: Path,
    loader: FrameLoader,
    frame_metrics: list[dict[str, Any]],
    old_poses: np.ndarray,
    refined_poses: np.ndarray,
    detections: list[list[dict[str, Any]]],
    vertices: np.ndarray,
    camera_matrix: np.ndarray,
) -> Path:
    ranked = sorted(
        range(len(frame_metrics)),
        key=lambda index: float(frame_metrics[index]["corner_median_improvement_px"]),
        reverse=True,
    )
    selected = []
    for index in ranked[:8] + np.linspace(
        0, len(frame_metrics) - 1, 8, dtype=int
    ).tolist():
        if index not in selected:
            selected.append(index)
        if len(selected) == 12:
            break
    debug_dir = output_dir / "overlays"
    debug_dir.mkdir(parents=True, exist_ok=True)
    thumbnails = []
    for frame_index in selected:
        image = loader.image(frame_index)
        draw_cube(image, vertices, old_poses[frame_index], camera_matrix, (0, 0, 255), 4)
        draw_cube(
            image,
            vertices,
            refined_poses[frame_index],
            camera_matrix,
            (0, 255, 0),
            3,
        )
        for detection in detections[frame_index]:
            for point in np.asarray(detection["corners"]).reshape(4, 2):
                cv2.circle(image, tuple(np.round(point).astype(int)), 5, (255, 255, 0), -1)
        metric = frame_metrics[frame_index]
        cv2.putText(
            image,
            f"frame={frame_index} tags={metric['tag_ids']} red=old green=refined",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            (
                f"corner median {metric['old_corner_median_px']:.2f}->"
                f"{metric['refined_corner_median_px']:.2f}px "
                f"rot={metric['rotation_delta_deg']:.1f}deg"
            ),
            (18, 61),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        image_path = debug_dir / f"frame_{frame_index:04d}.jpg"
        cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        thumbnails.append(cv2.resize(image, (648, 486), interpolation=cv2.INTER_AREA))
    if len(thumbnails) % 2:
        thumbnails.append(np.zeros_like(thumbnails[0]))
    sheet = np.vstack(
        [np.hstack(thumbnails[index : index + 2]) for index in range(0, len(thumbnails), 2)]
    )
    sheet_path = output_dir / "comparison_contact_sheet.jpg"
    cv2.imwrite(str(sheet_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return sheet_path


def apply_refined_poses(
    pkl_path: Path,
    refined_poses: np.ndarray,
    detections: list[list[dict[str, Any]]],
    frame_metrics: list[dict[str, Any]],
    summary: dict[str, Any],
    face_id_sets: dict[str, set[int]],
) -> None:
    required = pkl_path.stat().st_size + 2 * 1024**3
    free = shutil.disk_usage(pkl_path.parent).free
    if free < required:
        raise RuntimeError(
            f"Atomic rewrite needs about {required / 1024**3:.1f} GiB free, "
            f"only {free / 1024**3:.1f} GiB is available"
        )
    temporary = pkl_path.with_suffix(pkl_path.suffix + ".027-rewrite.tmp")
    if temporary.exists():
        temporary.unlink()
    frame_index = 0
    try:
        with pkl_path.open("rb") as source, temporary.open("wb") as destination:
            header = pickle.load(source)
            if not isinstance(header, dict):
                raise ValueError("PKL header is not a dictionary")
            metadata = header.setdefault("metadata", {})
            offline = metadata.setdefault("offline_pos_estimation", {})
            offline["middle_single_tag_refinement"] = summary
            pickle.dump(header, destination, protocol=pickle.HIGHEST_PROTOCOL)
            while True:
                try:
                    record = pickle.load(source)
                except EOFError:
                    break
                if isinstance(record, dict) and record.get("type") == "frame_pair":
                    camera_record = record["cameras"][CAMERA_NAME]
                    result = target_result(camera_record)
                    pose = refined_poses[frame_index]
                    metric = frame_metrics[frame_index]
                    tag_ids = [int(value["tag_id"]) for value in detections[frame_index]]
                    old_rvec = np.asarray(result["rvec"], dtype=np.float64).reshape(3, 1)
                    old_tvec = np.asarray(result["tvec"], dtype=np.float64).reshape(3, 1)
                    result["pre_refine_rvec"] = old_rvec
                    result["pre_refine_tvec"] = old_tvec
                    result["pre_refine_pose_backend"] = str(result.get("pose_backend", ""))
                    result["rvec"] = pose[:3].reshape(3, 1)
                    result["tvec"] = pose[3:].reshape(3, 1)
                    result["T"] = pose_matrix(result["rvec"], result["tvec"])
                    result["reproj_error"] = float(metric["refined_corner_mean_px"])
                    result["n_tags"] = len(tag_ids)
                    result["tag_ids"] = tag_ids
                    result["visible_faces"] = {
                        face
                        for face, ids in face_id_sets.items()
                        if any(tag_id in ids for tag_id in tag_ids)
                    }
                    result["predicted"] = not bool(tag_ids)
                    result["measured"] = bool(tag_ids)
                    result["pose_backend"] = REFINEMENT_ALGORITHM
                    result["pose_source"] = REFINEMENT_ALGORITHM
                    result["middle_single_tag_refinement"] = metric
                    frame_index += 1
                pickle.dump(record, destination, protocol=pickle.HIGHEST_PROTOCOL)
            destination.flush()
            os.fsync(destination.fileno())
        if frame_index != len(refined_poses):
            raise RuntimeError(
                f"Rewrote {frame_index} frames, expected {len(refined_poses)}"
            )
        temporary.replace(pkl_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def main() -> None:
    args = parse_args()
    pkl_path = args.pkl_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    header, offsets, old_poses, camera_matrix, _timestamp_array, _timestamps = build_index(
        pkl_path
    )
    loader = FrameLoader(pkl_path, offsets)
    cube_path = next(
        Path(value).expanduser().resolve()
        for value in header["metadata"]["camera_cube_configs"][CAMERA_NAME]
        if Path(value).name == CUBE_NAME
    )
    config, face_id_sets = load_cube_config(str(cube_path / "config.json"))
    tag_corner_map = build_tag_corner_map(config)
    target_ids = set(int(value) for value in tag_corner_map)

    cache_path = output_dir / f"{pkl_path.stem}_subpixel_detections.json"
    fingerprint = cache_fingerprint(pkl_path, len(offsets))
    cv2_detections = (
        load_detection_cache(cache_path, fingerprint) if args.reuse_detections else None
    )
    if cv2_detections is None:
        cv2_detections = detect_tag_borders(
            loader,
            len(offsets),
            cube_path,
            camera_matrix,
            target_ids,
            cache_path,
            fingerprint,
        )
    else:
        print(f"[INFO] Reused precise tag-border cache: {cache_path}")

    deeptag_cache_path = output_dir / f"{pkl_path.stem}_deeptag_points.json"
    deeptag_points = load_or_detect_deeptag_points(
        loader,
        len(offsets),
        cube_path,
        camera_matrix,
        target_ids,
        deeptag_cache_path,
        fingerprint,
        args.reuse_detections,
    )
    detections = fuse_border_detections(cv2_detections, deeptag_points)

    calibrated_tag_corner_map, accepted_tag_ids, tag_geometry_calibration = (
        calibrate_observed_tag_geometry(
            detections,
            old_poses,
            tag_corner_map,
            float(config.tag_size_mm),
            camera_matrix,
        )
    )
    detections = [
        [
            value
            for value in frame
            if int(value["tag_id"]) in accepted_tag_ids
        ]
        for frame in detections
    ]
    print(
        f"[INFO] Empirically calibrated tag geometry; accepted ids="
        f"{sorted(accepted_tag_ids)}"
    )

    observations = [
        detection_arrays(frame_detections, calibrated_tag_corner_map)
        for frame_detections in detections
    ]
    local_poses = np.vstack(
        [
            local_pose_refinement(old_pose, *observation, camera_matrix)
            for old_pose, observation in zip(old_poses, observations)
        ]
    )
    print("[INFO] Running sparse whole-sequence pose refinement")
    refined_poses, optimizer_result = joint_sequence_refinement(
        local_poses, observations, camera_matrix
    )

    accepted_poses = refined_poses.copy()
    rejected = []
    for frame_index, (old_pose, refined_pose) in enumerate(
        zip(old_poses, refined_poses)
    ):
        rotation_delta = rotation_delta_deg(old_pose[:3], refined_pose[:3])
        translation_delta = float(np.linalg.norm(old_pose[3:] - refined_pose[3:]))
        if (
            rotation_delta > MAX_ACCEPTED_ROTATION_DELTA_DEG
            or translation_delta > MAX_ACCEPTED_TRANSLATION_DELTA_MM
            or refined_pose[5] <= 0.0
        ):
            accepted_poses[frame_index] = local_poses[frame_index]
            rejected.append(frame_index)

    vertices = cube_vertices(config.box_dims)
    frame_metrics: list[dict[str, Any]] = []
    old_edge_costs = []
    refined_edge_costs = []
    for frame_index, (old_pose, refined_pose, observation) in enumerate(
        zip(old_poses, accepted_poses, observations)
    ):
        object_points, image_points, _weights = observation
        old_errors = np.linalg.norm(
            project_points(object_points, old_pose, camera_matrix) - image_points,
            axis=1,
        ) if len(object_points) else np.empty(0)
        refined_errors = np.linalg.norm(
            project_points(object_points, refined_pose, camera_matrix) - image_points,
            axis=1,
        ) if len(object_points) else np.empty(0)
        image = loader.image(frame_index)
        distance = edge_distance_image(image)
        old_edge = visible_edge_cost(distance, vertices, old_pose, camera_matrix)
        refined_edge = visible_edge_cost(distance, vertices, refined_pose, camera_matrix)
        old_edge_costs.append(old_edge)
        refined_edge_costs.append(refined_edge)
        old_median = float(np.median(old_errors)) if len(old_errors) else float("nan")
        refined_median = (
            float(np.median(refined_errors)) if len(refined_errors) else float("nan")
        )
        frame_metrics.append(
            {
                "frame_index": frame_index,
                "tag_ids": [int(value["tag_id"]) for value in detections[frame_index]],
                "tag_count": len(detections[frame_index]),
                "old_corner_mean_px": float(np.mean(old_errors)) if len(old_errors) else float("nan"),
                "old_corner_median_px": old_median,
                "refined_corner_mean_px": float(np.mean(refined_errors)) if len(refined_errors) else float("nan"),
                "refined_corner_median_px": refined_median,
                "corner_median_improvement_px": old_median - refined_median,
                "old_visible_edge_cost": old_edge,
                "refined_visible_edge_cost": refined_edge,
                "rotation_delta_deg": rotation_delta_deg(old_pose[:3], refined_pose[:3]),
                "translation_delta_mm": float(np.linalg.norm(old_pose[3:] - refined_pose[3:])),
                "global_solution_rejected": frame_index in rejected,
                "rvec": refined_pose[:3].tolist(),
                "tvec": refined_pose[3:].tolist(),
            }
        )

    old_rotation_steps = [
        rotation_delta_deg(old_poses[index - 1, :3], old_poses[index, :3])
        for index in range(1, len(old_poses))
    ]
    refined_rotation_steps = [
        rotation_delta_deg(accepted_poses[index - 1, :3], accepted_poses[index, :3])
        for index in range(1, len(accepted_poses))
    ]
    tag_count_distribution = Counter(len(value) for value in detections)
    detection_source_counts = Counter(
        str(detection.get("source", "unknown"))
        for frame in detections
        for detection in frame
    )
    summary = {
        "algorithm": REFINEMENT_ALGORITHM,
        "pkl_path": str(pkl_path),
        "frame_count": len(offsets),
        "apply": bool(args.apply),
        "tag_count_frames": {str(key): value for key, value in sorted(tag_count_distribution.items())},
        "single_tag_frames": int(tag_count_distribution.get(1, 0)),
        "nonplanar_two_or_more_tag_frames": int(
            sum(value for key, value in tag_count_distribution.items() if key >= 2)
        ),
        "no_tag_frames": int(tag_count_distribution.get(0, 0)),
        "detection_source_counts": dict(detection_source_counts),
        "tag_geometry_calibration": tag_geometry_calibration,
        "optimizer_success": bool(optimizer_result.success),
        "optimizer_status": int(optimizer_result.status),
        "optimizer_message": str(optimizer_result.message),
        "optimizer_cost": float(optimizer_result.cost),
        "optimizer_nfev": int(optimizer_result.nfev),
        "rejected_global_frames": rejected,
        "old_corner_median_px": distribution(
            [float(value["old_corner_median_px"]) for value in frame_metrics]
        ),
        "refined_corner_median_px": distribution(
            [float(value["refined_corner_median_px"]) for value in frame_metrics]
        ),
        "corner_median_improvement_px": distribution(
            [float(value["corner_median_improvement_px"]) for value in frame_metrics]
        ),
        "old_visible_edge_cost": distribution(old_edge_costs),
        "refined_visible_edge_cost": distribution(refined_edge_costs),
        "old_interframe_rotation_deg": distribution(old_rotation_steps),
        "refined_interframe_rotation_deg": distribution(refined_rotation_steps),
        "old_rotation_jumps_over_10deg": int(sum(value > 10.0 for value in old_rotation_steps)),
        "refined_rotation_jumps_over_10deg": int(
            sum(value > 10.0 for value in refined_rotation_steps)
        ),
        "parameters": {
            "corner_loss_scale_px": CORNER_LOSS_SCALE_PX,
            "rotation_accel_weight": ROTATION_ACCEL_WEIGHT,
            "translation_accel_weight": TRANSLATION_ACCEL_WEIGHT,
        },
    }
    sheet_path = save_debug_images(
        output_dir,
        loader,
        frame_metrics,
        old_poses,
        accepted_poses,
        detections,
        vertices,
        camera_matrix,
    )
    report_path = output_dir / f"{pkl_path.stem}_{REFINEMENT_ALGORITHM}.json"
    report_path.write_text(
        json.dumps({"summary": summary, "frames": frame_metrics}, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / f"{pkl_path.stem}_{REFINEMENT_ALGORITHM}.npz",
        old_poses=old_poses,
        local_poses=local_poses,
        refined_poses=accepted_poses,
    )
    print(json.dumps(summary, indent=2))
    print(f"[INFO] Report: {report_path}")
    print(f"[INFO] Comparison sheet: {sheet_path}")
    if args.apply:
        apply_refined_poses(
            pkl_path,
            accepted_poses,
            detections,
            frame_metrics,
            summary,
            face_id_sets,
        )
        print(f"[INFO] Applied refined poses atomically to {pkl_path}")


if __name__ == "__main__":
    main()
