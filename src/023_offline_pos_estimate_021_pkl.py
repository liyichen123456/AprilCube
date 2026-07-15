#!/usr/bin/env python3
from __future__ import annotations

import copy
import argparse
import json
import os
import pickle
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

import aprilcube
from aprilcube.deeptag_pose import DeepTagPoseBackend, backend_parameters


APRILCUBE_ROOT = Path(__file__).resolve().parent.parent
PKL_PATH = (
    APRILCUBE_ROOT
    / "recordings/021_hand_back_sync_raw_frames_20260712_173546.pkl"
)
EXPECTED_PKL_FORMAT = "aprilcube_hand_back_software_synced_raw_v1"
CAMERA_NAMES = ("thumb_web_cam", "middle_finger_cam")

OFFLINE_POS_FIELD = "offline_pos"
OFFLINE_POS_KEY_FIELD = "offline_pos_key"
OFFLINE_POS_ALGORITHM = "deeptag_internal_grid_primary_cv2_fallback_v2"
POSE_BACKEND = "deeptag_internal_grid"
UNDISTORTED_IMAGE_JPEG_FIELD = "undistorted_image_jpeg"
UNDISTORTED_POSE_OVERLAY_JPEG_FIELD = "undistorted_pose_overlay_jpeg"
UNDISTORTED_VISUALIZATION_META_FIELD = "undistorted_visualization_meta"
UNDISTORTED_VISUALIZATION_VERSION = 1
UNDISTORTED_VISUALIZATION_JPEG_QUALITY = 92

ADAPTIVE_CLAHE_DETECTION = True
FAST_DETECTOR = True
UNDISTORT_BEFORE_DETECTION = True
PINHOLE_UNDISTORT_ALPHA = 0.0

SINGLE_TAG_FACE_FRAME_STRICT_MAX_REPROJ_PX = 5.0
SINGLE_TAG_FACE_FRAME_MAX_REPROJ_PX = 12.0
SINGLE_TAG_FACE_FRAME_MAX_OTHER_TAG_REPROJ_PX = 15.0
SINGLE_TAG_FACE_FRAME_REPROJ_TIE_PX = 1.0
SINGLE_TAG_FACE_FRAME_LM_REFINE = True

MINIMUM_REWRITE_FREE_SPACE_GIB = 2.0
PROGRESS_PRINT_INTERVAL_S = 0.5


def values_equal(left: Any, right: Any) -> bool:
    """Compare nested PKL metadata without ambiguous NumPy truth values."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return False
        return bool(np.array_equal(left, right, equal_nan=True))
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        return left.keys() == right.keys() and all(
            values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(values_equal(a, b) for a, b in zip(left, right))
    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(equal, np.ndarray):
        return bool(np.all(equal))
    return bool(equal)


def encode_jpeg_bgr(image_bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        np.asarray(image_bgr, dtype=np.uint8),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(UNDISTORTED_VISUALIZATION_JPEG_QUALITY)],
    )
    if not ok:
        raise RuntimeError("Failed to JPEG-encode visualization image")
    return encoded.tobytes()


def load_calibration(path: Path) -> dict[str, Any]:
    yaml_path = path.expanduser().resolve()
    with yaml_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    dist_coeffs = data.get("dist", data.get("D", np.zeros(5)))
    return {
        "path": str(yaml_path),
        "camera_model": str(data.get("camera_model", "")),
        "distortion_model": str(data.get("distortion_model", "")),
        "image_size": tuple(int(value) for value in data["image_size"]),
        "camera_matrix": np.asarray(data["K"], dtype=np.float64).reshape(3, 3),
        "dist_coeffs": np.asarray(dist_coeffs, dtype=np.float64).reshape(-1),
    }


def is_fisheye(calibration: dict[str, Any]) -> bool:
    return (
        calibration["camera_model"].lower() == "fisheye"
        or calibration["distortion_model"].lower() == "opencv_fisheye"
    )


def make_detection_camera_matrix(calibration: dict[str, Any]) -> np.ndarray:
    camera_matrix = calibration["camera_matrix"]
    dist_coeffs = calibration["dist_coeffs"]
    image_size = calibration["image_size"]
    if not UNDISTORT_BEFORE_DETECTION or np.allclose(dist_coeffs, 0.0):
        return camera_matrix.copy()

    if is_fisheye(calibration):
        width, height = image_size
        focal = float(camera_matrix[0, 0])
        return np.array(
            [
                [focal, 0.0, width / 2.0],
                [0.0, focal, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        PINHOLE_UNDISTORT_ALPHA,
        image_size,
    )
    return np.asarray(new_camera_matrix, dtype=np.float64).reshape(3, 3)


def make_undistort_maps(
    calibration: dict[str, Any],
    detection_camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not UNDISTORT_BEFORE_DETECTION:
        return None
    camera_matrix = calibration["camera_matrix"]
    dist_coeffs = calibration["dist_coeffs"]
    image_size = calibration["image_size"]
    if np.allclose(dist_coeffs, 0.0):
        return None

    if is_fisheye(calibration):
        if dist_coeffs.size != 4:
            raise ValueError(
                f"fisheye calibration requires 4 coefficients, got {dist_coeffs.size}"
            )
        return cv2.fisheye.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs.reshape(4, 1),
            np.eye(3, dtype=np.float64),
            detection_camera_matrix,
            image_size,
            cv2.CV_16SC2,
        )

    return cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        np.eye(3, dtype=np.float64),
        detection_camera_matrix,
        image_size,
        cv2.CV_16SC2,
    )


def camera_matrix_as_intrinsics(camera_matrix: np.ndarray) -> dict[str, float]:
    return {
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
    }


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
        if definition[0] == face_name:
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
    previous_rvec: np.ndarray | None,
    previous_tvec: np.ndarray | None,
) -> float:
    if previous_rvec is None or previous_tvec is None:
        return 0.0
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    previous_rotation, _ = cv2.Rodrigues(
        np.asarray(previous_rvec, dtype=np.float64).reshape(3, 1)
    )
    angle = np.arccos(
        np.clip((np.trace(previous_rotation.T @ rotation) - 1.0) / 2.0, -1.0, 1.0)
    )
    translation_delta = float(
        np.linalg.norm(
            np.asarray(tvec, dtype=np.float64).reshape(3)
            - np.asarray(previous_tvec, dtype=np.float64).reshape(3)
        )
    )
    return min(translation_delta / 20.0, 20.0) + min(
        float(np.degrees(angle)) / 10.0,
        20.0,
    )


def per_detected_tag_reprojection_errors(
    detections: list[tuple[int, np.ndarray]],
    tag_corner_map: dict[int, np.ndarray],
    selected_tag_id: int,
    selected_corner_rotation: int,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict[int, float]:
    errors: dict[int, float] = {}
    for tag_id_value, corners_2d_value in detections:
        tag_id = int(tag_id_value)
        cube_corners = tag_corner_map.get(tag_id)
        if cube_corners is None:
            continue
        cube_corners = np.asarray(cube_corners, dtype=np.float64).reshape(4, 3)
        if tag_id == selected_tag_id and selected_corner_rotation:
            cube_corners = np.roll(cube_corners, -int(selected_corner_rotation), axis=0)
        corners_2d = np.asarray(corners_2d_value, dtype=np.float64).reshape(4, 2)
        projected, _ = cv2.projectPoints(
            cube_corners,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        errors[tag_id] = float(
            np.mean(np.linalg.norm(corners_2d - projected.reshape(-1, 2), axis=1))
        )
    return errors


def estimate_single_tag_pose_from_face_frame(
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
                solved, face_rvecs, face_tvecs, _ = cv2.solvePnPGeneric(
                    corners_face,
                    corners_2d,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE,
                )
            except cv2.error:
                solved, face_rvecs, face_tvecs = 0, (), ()
            if not solved:
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
                reprojection_error = float(
                    np.mean(
                        np.linalg.norm(
                            corners_2d - projected.reshape(-1, 2),
                            axis=1,
                        )
                    )
                )
                if (
                    not np.isfinite(reprojection_error)
                    or reprojection_error > SINGLE_TAG_FACE_FRAME_MAX_REPROJ_PX
                ):
                    continue
                per_tag_errors = per_detected_tag_reprojection_errors(
                    detections,
                    tag_corner_map,
                    int(tag_id),
                    int(corner_rotation),
                    cube_rvec,
                    translation_camera_from_cube,
                    camera_matrix,
                    dist_coeffs,
                )
                all_tag_errors = [
                    value for value in per_tag_errors.values() if np.isfinite(value)
                ]
                other_tag_errors = [
                    value
                    for other_tag_id, value in per_tag_errors.items()
                    if int(other_tag_id) != int(tag_id) and np.isfinite(value)
                ]
                all_tag_mean_reproj = (
                    float(np.mean(all_tag_errors)) if all_tag_errors else float("inf")
                )
                all_tag_max_reproj = (
                    float(np.max(all_tag_errors)) if all_tag_errors else float("inf")
                )
                other_tag_max_reproj = (
                    float(np.max(other_tag_errors)) if other_tag_errors else 0.0
                )
                candidates.append(
                    {
                        "rvec": cube_rvec,
                        "tvec": translation_camera_from_cube,
                        "reproj_error": reprojection_error,
                        "per_tag_reproj_error": per_tag_errors,
                        "all_tag_mean_reproj_error": all_tag_mean_reproj,
                        "all_tag_max_reproj_error": all_tag_max_reproj,
                        "other_tag_max_reproj_error": other_tag_max_reproj,
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

    if len(detections) > 1:
        near_best = list(candidates)
        selected = min(
            candidates,
            key=lambda candidate: (
                float(candidate["other_tag_max_reproj_error"])
                > SINGLE_TAG_FACE_FRAME_MAX_OTHER_TAG_REPROJ_PX,
                float(candidate["other_tag_max_reproj_error"]),
                float(candidate["all_tag_mean_reproj_error"]),
                float(candidate["continuity_cost"]),
                float(candidate["reproj_error"]),
            ),
        )
        best_reprojection_error = float(selected["reproj_error"])
    else:
        best_reprojection_error = min(
            float(candidate["reproj_error"]) for candidate in candidates
        )
        near_best = [
            candidate
            for candidate in candidates
            if float(candidate["reproj_error"])
            <= best_reprojection_error + SINGLE_TAG_FACE_FRAME_REPROJ_TIE_PX
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

    return (
        True,
        selected["rvec"],
        selected["tvec"],
        float(selected["reproj_error"]),
        np.arange(4, dtype=np.int32).reshape(-1, 1),
        {
            "single_tag_id": int(selected["tag_id"]),
            "single_tag_face": selected["face_name"],
            "single_tag_candidate_count": len(candidates),
            "single_tag_corner_rotation_deg": int(selected["corner_rotation"]) * 90,
            "single_tag_face_frame_pose": True,
            "single_tag_face_frame_lm_refined": bool(selected["lm_refined"]),
            "single_tag_face_frame_relaxed": bool(
                float(selected["reproj_error"]) > SINGLE_TAG_FACE_FRAME_STRICT_MAX_REPROJ_PX
            ),
            "single_tag_face_frame_global_scored": bool(len(detections) > 1),
            "single_tag_face_frame_per_tag_reproj_error": {
                int(tag_id): float(error)
                for tag_id, error in selected["per_tag_reproj_error"].items()
            },
            "single_tag_face_frame_all_tag_mean_reproj_error": float(
                selected["all_tag_mean_reproj_error"]
            ),
            "single_tag_face_frame_all_tag_max_reproj_error": float(
                selected["all_tag_max_reproj_error"]
            ),
            "single_tag_face_frame_other_tag_max_reproj_error": float(
                selected["other_tag_max_reproj_error"]
            ),
            "single_tag_face_frame_strict_max_reproj_px": float(
                SINGLE_TAG_FACE_FRAME_STRICT_MAX_REPROJ_PX
            ),
            "single_tag_face_frame_max_reproj_px": float(
                SINGLE_TAG_FACE_FRAME_MAX_REPROJ_PX
            ),
            "single_tag_face_frame_max_other_tag_reproj_px": float(
                SINGLE_TAG_FACE_FRAME_MAX_OTHER_TAG_REPROJ_PX
            ),
            "single_tag_face_frame_raw_candidate_count": raw_candidate_count,
            "single_tag_face_frame_near_best_count": len(near_best),
            "single_tag_face_frame_best_reproj_error": best_reprojection_error,
        },
    )


def process_with_single_tag_face_solver(
    detector: Any,
    image: np.ndarray,
    tag_detections: list[tuple[int, np.ndarray]],
    **kwargs: Any,
) -> dict[str, Any]:
    from aprilcube import detect as detect_module

    original_solver = detect_module.estimate_single_tag_cube_pose
    detect_module.estimate_single_tag_cube_pose = estimate_single_tag_pose_from_face_frame
    try:
        return detector.process_detections(image, tag_detections, **kwargs)
    finally:
        detect_module.estimate_single_tag_cube_pose = original_solver


def clone_optional_array(value: np.ndarray | None) -> np.ndarray | None:
    return None if value is None else value.copy()


def snapshot_detector_state(detector: Any) -> dict[str, Any]:
    return {
        "prev_rvec": clone_optional_array(detector.prev_rvec),
        "prev_tvec": clone_optional_array(detector.prev_tvec),
        "pose_filter": copy.deepcopy(detector.pose_filter),
        "_prev_gray": clone_optional_array(detector._prev_gray),
        "_prev_corners_2d": clone_optional_array(detector._prev_corners_2d),
        "_prev_corners_3d": clone_optional_array(detector._prev_corners_3d),
    }


def restore_detector_state(detector: Any, state: dict[str, Any]) -> None:
    detector.prev_rvec = clone_optional_array(state["prev_rvec"])
    detector.prev_tvec = clone_optional_array(state["prev_tvec"])
    detector.pose_filter = copy.deepcopy(state["pose_filter"])
    detector._prev_gray = clone_optional_array(state["_prev_gray"])
    detector._prev_corners_2d = clone_optional_array(state["_prev_corners_2d"])
    detector._prev_corners_3d = clone_optional_array(state["_prev_corners_3d"])


def is_measured_pose(result: dict[str, Any]) -> bool:
    return bool(result.get("success", False)) and not bool(result.get("predicted", False))


def copy_pose_result(result: dict[str, Any]) -> dict[str, Any]:
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
        "single_tag_face_frame_relaxed",
        "single_tag_face_frame_global_scored",
        "single_tag_face_frame_per_tag_reproj_error",
        "single_tag_face_frame_all_tag_mean_reproj_error",
        "single_tag_face_frame_all_tag_max_reproj_error",
        "single_tag_face_frame_other_tag_max_reproj_error",
        "single_tag_face_frame_strict_max_reproj_px",
        "single_tag_face_frame_max_reproj_px",
        "single_tag_face_frame_max_other_tag_reproj_px",
        "single_tag_face_frame_raw_candidate_count",
        "single_tag_face_frame_near_best_count",
        "single_tag_face_frame_best_reproj_error",
        "failure_reason",
    ):
        value = result.get(key)
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


def adaptive_enhancement_variants() -> tuple[dict[str, Any], ...]:
    from aprilcube import detect as detect_module

    variants = getattr(detect_module, "_adaptive_image_enhancement_variants", ())
    if variants:
        return tuple(dict(variant) for variant in variants)
    return tuple(
        {
            "name": f"adaptive clip={float(clip_limit):.1f} tile={tuple(tile_size)}",
            "clahe": (float(clip_limit), tuple(tile_size)),
        }
        for clip_limit, tile_size in getattr(
            detect_module,
            "_adaptive_clahe_variants",
            (),
        )
    )


def normalize_pose_result_for_drawing(result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result or {})
    normalized.setdefault("success", False)
    normalized.setdefault("detections", [])
    normalized.setdefault("visible_faces", set())
    normalized.setdefault("n_tags", 0)
    normalized.setdefault("reproj_error", float("inf"))
    for key in ("rvec", "tvec"):
        if normalized.get(key) is not None:
            normalized[key] = np.asarray(normalized[key], dtype=np.float64).reshape(3, 1)
    if normalized.get("T") is not None:
        normalized["T"] = np.asarray(normalized["T"], dtype=np.float64).reshape(4, 4)
    return normalized


class CameraPoseEstimator:
    def __init__(
        self,
        camera_name: str,
        intrinsics_yaml: Path,
        cube_paths: list[Path],
    ) -> None:
        self.camera_name = camera_name
        self.intrinsics_yaml = intrinsics_yaml.expanduser().resolve()
        self.cube_paths = [path.expanduser().resolve() for path in cube_paths]
        self.calibration = load_calibration(self.intrinsics_yaml)
        self.detection_camera_matrix = make_detection_camera_matrix(self.calibration)
        self.undistort_maps = make_undistort_maps(
            self.calibration,
            self.detection_camera_matrix,
        )
        detector_distortion = (
            np.zeros(5, dtype=np.float64)
            if UNDISTORT_BEFORE_DETECTION
            else self.calibration["dist_coeffs"]
        )
        self.detectors: list[tuple[str, Any]] = []
        for cube_path in self.cube_paths:
            detector = aprilcube.detector(
                cube_path,
                intrinsic_cfg=camera_matrix_as_intrinsics(self.detection_camera_matrix),
                dist_coeffs=detector_distortion,
                enable_filter=False,
                fast=FAST_DETECTOR,
            )
            self.detectors.append((cube_path.name, detector))
        self.deeptag_backend = DeepTagPoseBackend(
            self.detection_camera_matrix,
            self.cube_paths,
        )

    def prepare_image(self, image_bgr: np.ndarray) -> np.ndarray:
        image = np.asarray(image_bgr, dtype=np.uint8)
        target_size = self.calibration["image_size"]
        height, width = image.shape[:2]
        if (width, height) != target_size:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        if self.undistort_maps is not None:
            image = cv2.remap(
                image,
                self.undistort_maps[0],
                self.undistort_maps[1],
                interpolation=cv2.INTER_LINEAR,
            )
        return image

    def estimate(self, camera_record: dict[str, Any]) -> dict[str, Any]:
        image = self.prepare_image(camera_record["image_bgr"])
        result = self.deeptag_backend.estimate(image)
        timestamp = float(camera_record["capture_timestamp"])
        detector_by_cube = dict(self.detectors)
        deep_primary_success_count = 0
        cv2_fallback_success_count = 0
        for cube in result["cube_results"]:
            deep_result = cube["result"]
            if deep_result.get("success", False):
                deep_result["pose_backend"] = "deeptag_internal_grid"
                deep_primary_success_count += 1
                continue

            deep_failure_reason = str(
                deep_result.get("failure_reason") or "unknown_deeptag_failure"
            )
            fallback_result, fallback_tags, recovery_mode = self._estimate_one_cube(
                detector_by_cube[cube["cube_name"]],
                image,
                timestamp,
            )
            if is_measured_pose(fallback_result):
                pose = copy_pose_result(fallback_result)
                pose["pose_backend"] = "cv2_fallback"
                pose["deeptag_primary_failure_reason"] = deep_failure_reason
                pose["cv2_fallback_recovery_mode"] = recovery_mode
                pose["cv2_fallback_decoded_tag_count"] = len(
                    fallback_tags["detections"]
                )
                cube["result"] = pose
                cv2_fallback_success_count += 1
            else:
                deep_result["pose_backend"] = "failed"
                deep_result["deeptag_primary_failure_reason"] = deep_failure_reason
                deep_result["cv2_fallback_failure_reason"] = str(
                    fallback_result.get("failure_reason") or "unknown_cv2_failure"
                )
                deep_result["cv2_fallback_recovery_mode"] = recovery_mode

        result["camera_name"] = self.camera_name
        result["source_image_field"] = "image_bgr"
        result["undistort_before_detection"] = bool(UNDISTORT_BEFORE_DETECTION)
        result["deeptag_primary_success_count"] = deep_primary_success_count
        result["cv2_fallback_success_count"] = cv2_fallback_success_count
        result["algorithm"] = OFFLINE_POS_ALGORITHM
        return result

    def add_visualization_images(
        self,
        camera_record: dict[str, Any],
        offline_pos: dict[str, Any],
    ) -> None:
        undistorted = self.prepare_image(camera_record["image_bgr"]).copy()
        overlay = undistorted.copy()
        results_by_cube = {
            str(cube.get("cube_name")): cube.get("result", {})
            for cube in offline_pos.get("cube_results", [])
        }
        lines = [
            f"{self.camera_name} sequence={camera_record.get('sequence', '?')}",
        ]
        missing_pose = False
        for cube_name, detector in self.detectors:
            result = normalize_pose_result_for_drawing(results_by_cube.get(cube_name, {}))
            overlay = detector.draw_result(overlay, result)
            if result.get("success", False):
                tvec = result["tvec"].reshape(3)
                relaxed = " relaxed" if result.get("single_tag_face_frame_relaxed", False) else ""
                lines.append(
                    f"{cube_name}: t=({tvec[0]:.1f},{tvec[1]:.1f},{tvec[2]:.1f})mm "
                    f"reproj={float(result.get('reproj_error', float('nan'))):.2f}px{relaxed}"
                )
            else:
                missing_pose = True
                lines.append(
                    f"{cube_name}: FAILED {result.get('failure_reason', '')}"
                )

        for line_index, line in enumerate(lines):
            cv2.putText(
                overlay,
                line,
                (18, 30 + line_index * 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if missing_pose:
            border = max(6, min(overlay.shape[:2]) // 120)
            cv2.rectangle(
                overlay,
                (0, 0),
                (overlay.shape[1] - 1, overlay.shape[0] - 1),
                (0, 0, 255),
                border,
            )

        camera_record[UNDISTORTED_IMAGE_JPEG_FIELD] = encode_jpeg_bgr(undistorted)
        camera_record[UNDISTORTED_POSE_OVERLAY_JPEG_FIELD] = encode_jpeg_bgr(overlay)
        camera_record[UNDISTORTED_VISUALIZATION_META_FIELD] = {
            "version": int(UNDISTORTED_VISUALIZATION_VERSION),
            "encoding": "jpeg",
            "color_order_after_decode": "bgr",
            "jpeg_quality": int(UNDISTORTED_VISUALIZATION_JPEG_QUALITY),
            "image_size": tuple(int(value) for value in self.calibration["image_size"]),
            "undistort_before_detection": bool(UNDISTORT_BEFORE_DETECTION),
            "intrinsics_yaml": str(self.intrinsics_yaml),
            "detection_camera_matrix": self.detection_camera_matrix.tolist(),
        }

    def _estimate_one_cube(
        self,
        detector: Any,
        image: np.ndarray,
        timestamp: float,
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        state_before = snapshot_detector_state(detector)
        base_tags = detector.detect_tags(image, adaptive_clahe=False)
        base_result = process_with_single_tag_face_solver(
            detector,
            image,
            base_tags["detections"],
            rejected_quads=base_tags["rejected"],
            gray=base_tags["gray"],
            enhanced=base_tags["enhanced"],
            timestamp=timestamp,
        )
        state_after_base = snapshot_detector_state(detector)
        if is_measured_pose(base_result) or not ADAPTIVE_CLAHE_DETECTION:
            return base_result, base_tags, "base"

        for variant in adaptive_enhancement_variants():
            restore_detector_state(detector, state_before)
            candidate_tags = detector.detect_tags(
                image,
                adaptive_clahe=True,
                enhancement_variants=(variant,),
            )
            candidate_result = process_with_single_tag_face_solver(
                detector,
                image,
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

        restore_detector_state(detector, state_after_base)
        return base_result, base_tags, "base_failed_enhancement_rejected"


def build_offline_pos_key(
    camera_name: str,
    estimator: CameraPoseEstimator,
) -> dict[str, Any]:
    return {
        "algorithm": OFFLINE_POS_ALGORITHM,
        "camera_name": camera_name,
        "intrinsics_yaml": str(estimator.intrinsics_yaml),
        "cube_paths": [str(path) for path in estimator.cube_paths],
        "pose_backend": POSE_BACKEND,
        "deeptag_parameters": backend_parameters(),
        "fallback_backend": "cv2_adaptive_clahe_single_tag_face_frame",
        "undistort_before_detection": bool(UNDISTORT_BEFORE_DETECTION),
        "source_image_field": "image_bgr",
        "runtime_pose_filter": False,
        "temporal_candidate_selection": True,
    }


def load_header(pkl_path: Path) -> dict[str, Any]:
    with pkl_path.open("rb") as file:
        header = pickle.load(file)
    if not isinstance(header, dict) or header.get("type") != "header":
        raise ValueError(f"Invalid 021 PKL header: {pkl_path}")
    if header.get("format") != EXPECTED_PKL_FORMAT:
        raise ValueError(
            f"Unsupported PKL format {header.get('format')!r}; "
            f"expected {EXPECTED_PKL_FORMAT!r}"
        )
    return header


def build_estimators_from_header(
    header: dict[str, Any],
) -> tuple[dict[str, CameraPoseEstimator], dict[str, dict[str, Any]]]:
    metadata = header.get("metadata", {})
    intrinsics_by_camera = metadata.get("camera_intrinsics_yaml", {})
    cube_paths_by_camera = metadata.get("camera_cube_configs", {})
    estimators: dict[str, CameraPoseEstimator] = {}
    keys: dict[str, dict[str, Any]] = {}
    for camera_name in CAMERA_NAMES:
        intrinsics_value = intrinsics_by_camera.get(camera_name)
        cube_values = cube_paths_by_camera.get(camera_name)
        if not intrinsics_value or not isinstance(cube_values, list) or not cube_values:
            raise ValueError(
                f"PKL header lacks intrinsics/cube configs for {camera_name}"
            )
        estimator = CameraPoseEstimator(
            camera_name,
            Path(intrinsics_value),
            [Path(value) for value in cube_values],
        )
        estimators[camera_name] = estimator
        keys[camera_name] = build_offline_pos_key(camera_name, estimator)
    return estimators, keys


def ensure_enough_space_for_safe_rewrite(pkl_path: Path) -> None:
    source_size = pkl_path.stat().st_size
    free_space = shutil.disk_usage(pkl_path.parent).free
    required_space = source_size + int(MINIMUM_REWRITE_FREE_SPACE_GIB * 1024**3)
    print(
        f"[INFO] Disk space: source={source_size / (1024**3):.2f}GiB "
        f"free={free_space / (1024**3):.2f}GiB "
        f"required={required_space / (1024**3):.2f}GiB"
    )
    if free_space < required_space:
        raise RuntimeError(
            "Not enough free disk space for an atomic PKL rewrite. "
            f"Need at least {required_space / (1024**3):.2f}GiB."
        )


def camera_record_has_current_pose(
    camera_record: dict[str, Any],
    expected_key: dict[str, Any],
) -> bool:
    return (
        isinstance(camera_record.get(OFFLINE_POS_FIELD), dict)
        and camera_record.get(OFFLINE_POS_KEY_FIELD) == expected_key
        and camera_record[OFFLINE_POS_FIELD].get("algorithm") == OFFLINE_POS_ALGORITHM
    )


def header_has_current_visualizations(header: dict[str, Any]) -> bool:
    metadata = header.get("metadata", {})
    visualization = metadata.get("offline_visualization_images", {})
    return (
        isinstance(visualization, dict)
        and visualization.get("version") == UNDISTORTED_VISUALIZATION_VERSION
        and visualization.get("pose_overlay_field") == UNDISTORTED_POSE_OVERLAY_JPEG_FIELD
        and visualization.get("undistorted_field") == UNDISTORTED_IMAGE_JPEG_FIELD
    )


def print_progress(
    completed_bytes: int,
    total_bytes: int,
    completed_pairs: int,
    elapsed_s: float,
    success_counts: dict[str, dict[str, int]],
) -> None:
    width = 30
    ratio = min(max(completed_bytes / max(total_bytes, 1), 0.0), 1.0)
    filled = int(round(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    pair_fps = completed_pairs / max(elapsed_s, 1e-9)
    summaries = []
    for camera_name in CAMERA_NAMES:
        camera_successes = success_counts.get(camera_name, {})
        summaries.append(
            f"{camera_name}="
            + ",".join(f"{cube}:{count}" for cube, count in camera_successes.items())
        )
    sys.stdout.write(
        f"\r[INFO] Offline pose [{bar}] {ratio * 100:5.1f}% "
        f"pairs={completed_pairs} speed={pair_fps:.2f}pair/s "
        + " ".join(summaries)
    )
    sys.stdout.flush()


def distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def make_pose_statistics(
    estimators: dict[str, CameraPoseEstimator],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        camera_name: {
            cube_name: {
                "total_frames": 0,
                "success_count": 0,
                "failure_reasons": Counter(),
                "backend_counts": Counter(),
                "deeptag_primary_failure_reasons": Counter(),
                "all_success_reproj_px": [],
                "cv2_fallback_reproj_px": [],
                "reproj_mean_px": [],
                "reproj_median_px": [],
                "reproj_p90_px": [],
                "reproj_rmse_px": [],
                "inlier_count": [],
            }
            for cube_name, _detector in estimator.detectors
        }
        for camera_name, estimator in estimators.items()
    }


def update_pose_statistics(
    accumulator: dict[str, dict[str, dict[str, Any]]],
    camera_name: str,
    cube_name: str,
    result: dict[str, Any],
) -> None:
    stats = accumulator[camera_name][cube_name]
    stats["total_frames"] += 1
    backend = str(result.get("pose_backend") or "unknown")
    stats["backend_counts"][backend] += 1
    deep_failure_reason = result.get("deeptag_primary_failure_reason")
    if deep_failure_reason:
        stats["deeptag_primary_failure_reasons"][str(deep_failure_reason)] += 1
    if not result.get("success", False):
        deep_reason = str(result.get("failure_reason") or "unknown_failure")
        cv2_reason = result.get("cv2_fallback_failure_reason")
        reason = (
            f"{deep_reason} | cv2:{cv2_reason}"
            if cv2_reason
            else deep_reason
        )
        stats["failure_reasons"][reason] += 1
        return
    stats["success_count"] += 1
    stats["all_success_reproj_px"].append(float(result["reproj_error"]))
    if backend == "cv2_fallback":
        stats["cv2_fallback_reproj_px"].append(float(result["reproj_error"]))
        return
    if backend != "deeptag_internal_grid":
        return
    stats["reproj_mean_px"].append(
        float(result["deeptag_internal_reproj_mean_px"])
    )
    stats["reproj_median_px"].append(
        float(result["deeptag_internal_reproj_median_px"])
    )
    stats["reproj_p90_px"].append(
        float(result["deeptag_internal_reproj_p90_px"])
    )
    stats["reproj_rmse_px"].append(
        float(result["deeptag_internal_reproj_rmse_px"])
    )
    stats["inlier_count"].append(
        float(result["deeptag_internal_inlier_count"])
    )


def finalize_pose_statistics(
    accumulator: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    finalized: dict[str, dict[str, dict[str, Any]]] = {}
    for camera_name, camera_stats in accumulator.items():
        finalized[camera_name] = {}
        for cube_name, stats in camera_stats.items():
            total = int(stats["total_frames"])
            successes = int(stats["success_count"])
            failures = total - successes
            failure_rows = []
            for reason, count in stats["failure_reasons"].most_common():
                failure_rows.append(
                    {
                        "reason": reason,
                        "count": int(count),
                        "percent_of_failures": 100.0 * count / max(failures, 1),
                        "percent_of_all_frames": 100.0 * count / max(total, 1),
                    }
                )
            primary_failure_count = sum(
                stats["deeptag_primary_failure_reasons"].values()
            )
            primary_failure_rows = [
                {
                    "reason": reason,
                    "count": int(count),
                    "percent_of_deeptag_primary_failures": (
                        100.0 * count / max(primary_failure_count, 1)
                    ),
                    "percent_of_all_frames": 100.0 * count / max(total, 1),
                }
                for reason, count in stats[
                    "deeptag_primary_failure_reasons"
                ].most_common()
            ]
            finalized[camera_name][cube_name] = {
                "total_frames": total,
                "success_count": successes,
                "failure_count": failures,
                "success_rate_percent": 100.0 * successes / max(total, 1),
                "failure_rate_percent": 100.0 * failures / max(total, 1),
                "backend_counts": {
                    key: int(value)
                    for key, value in stats["backend_counts"].items()
                },
                "deeptag_primary_success_count": int(
                    stats["backend_counts"].get("deeptag_internal_grid", 0)
                ),
                "cv2_fallback_success_count": int(
                    stats["backend_counts"].get("cv2_fallback", 0)
                ),
                "deeptag_primary_failure_count": int(primary_failure_count),
                "deeptag_primary_failure_reasons": primary_failure_rows,
                "failure_reasons": failure_rows,
                "reprojection_error_px": {
                    "all_success_per_frame": distribution(
                        stats["all_success_reproj_px"]
                    ),
                    "cv2_fallback_per_frame": distribution(
                        stats["cv2_fallback_reproj_px"]
                    ),
                    "per_frame_mean": distribution(stats["reproj_mean_px"]),
                    "per_frame_median": distribution(stats["reproj_median_px"]),
                    "per_frame_p90": distribution(stats["reproj_p90_px"]),
                    "per_frame_rmse": distribution(stats["reproj_rmse_px"]),
                },
                "internal_inlier_count": distribution(stats["inlier_count"]),
            }
    return finalized


def estimate_and_rewrite_original_pkl(
    pkl_path: Path,
    header: dict[str, Any],
    estimators: dict[str, CameraPoseEstimator],
    offline_pos_keys: dict[str, dict[str, Any]],
    source_header: dict[str, Any] | None = None,
) -> None:
    source_size = pkl_path.stat().st_size
    temporary_path = pkl_path.with_name(f".{pkl_path.name}.023-rewrite.tmp")
    temporary_path.unlink(missing_ok=True)
    success_counts = {
        camera_name: {
            cube_name: 0 for cube_name, _detector in estimator.detectors
        }
        for camera_name, estimator in estimators.items()
    }
    pose_statistics = make_pose_statistics(estimators)
    completed_pairs = 0
    started = time.perf_counter()
    last_progress = started

    updated_header = copy.deepcopy(header)
    updated_header.setdefault("metadata", {}).pop("rgb_pose_recovery", None)
    updated_header.setdefault("metadata", {})["offline_pos_estimation"] = {
        "algorithm": OFFLINE_POS_ALGORITHM,
        "pose_backend": POSE_BACKEND,
        "fallback_backend": "cv2_adaptive_clahe_single_tag_face_frame",
        "source_image_field": "image_bgr",
        "deeptag_parameters": backend_parameters(),
        "completed": True,
        "camera_keys": offline_pos_keys,
        "written_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    updated_header.setdefault("metadata", {})["offline_visualization_images"] = {
        "version": int(UNDISTORTED_VISUALIZATION_VERSION),
        "completed": True,
        "undistorted_field": UNDISTORTED_IMAGE_JPEG_FIELD,
        "pose_overlay_field": UNDISTORTED_POSE_OVERLAY_JPEG_FIELD,
        "meta_field": UNDISTORTED_VISUALIZATION_META_FIELD,
        "encoding": "jpeg",
        "color_order_after_decode": "bgr",
        "jpeg_quality": int(UNDISTORTED_VISUALIZATION_JPEG_QUALITY),
        "written_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        with pkl_path.open("rb") as source, temporary_path.open("wb") as destination:
            actual_source_header = pickle.load(source)
            expected_source_header = source_header if source_header is not None else header
            if not values_equal(actual_source_header, expected_source_header):
                raise RuntimeError("PKL header changed after estimator initialization")
            pickle.dump(updated_header, destination, protocol=pickle.HIGHEST_PROTOCOL)

            while True:
                try:
                    record = pickle.load(source)
                except EOFError:
                    break
                if isinstance(record, dict) and record.get("type") == "frame_pair":
                    cameras = record.get("cameras", {})
                    for camera_name in CAMERA_NAMES:
                        camera_record = cameras.get(camera_name)
                        if not isinstance(camera_record, dict):
                            raise ValueError(
                                f"Frame pair {completed_pairs} lacks {camera_name}"
                            )
                        if camera_record_has_current_pose(
                            camera_record,
                            offline_pos_keys[camera_name],
                        ):
                            offline_pos = camera_record[OFFLINE_POS_FIELD]
                        else:
                            offline_pos = estimators[camera_name].estimate(camera_record)
                        camera_record[OFFLINE_POS_FIELD] = offline_pos
                        camera_record[OFFLINE_POS_KEY_FIELD] = offline_pos_keys[camera_name]
                        estimators[camera_name].add_visualization_images(
                            camera_record,
                            offline_pos,
                        )
                        for cube in offline_pos["cube_results"]:
                            update_pose_statistics(
                                pose_statistics,
                                camera_name,
                                cube["cube_name"],
                                cube["result"],
                            )
                            if cube["result"].get("success", False):
                                success_counts[camera_name][cube["cube_name"]] += 1
                    record["offline_pos_algorithm"] = OFFLINE_POS_ALGORITHM
                    completed_pairs += 1
                elif isinstance(record, dict) and record.get("type") == "footer":
                    record.pop("rgb_pose_recovery", None)
                    finalized_statistics = finalize_pose_statistics(pose_statistics)
                    record["offline_pos_estimation"] = {
                        "algorithm": OFFLINE_POS_ALGORITHM,
                        "pose_backend": POSE_BACKEND,
                        "fallback_backend": "cv2_adaptive_clahe_single_tag_face_frame",
                        "source_image_field": "image_bgr",
                        "frame_pair_count": completed_pairs,
                        "success_counts": success_counts,
                        "statistics": finalized_statistics,
                    }
                    record["offline_visualization_images"] = {
                        "version": int(UNDISTORTED_VISUALIZATION_VERSION),
                        "frame_pair_count": completed_pairs,
                        "undistorted_field": UNDISTORTED_IMAGE_JPEG_FIELD,
                        "pose_overlay_field": UNDISTORTED_POSE_OVERLAY_JPEG_FIELD,
                    }

                pickle.dump(record, destination, protocol=pickle.HIGHEST_PROTOCOL)
                now = time.perf_counter()
                if now - last_progress >= PROGRESS_PRINT_INTERVAL_S:
                    print_progress(
                        source.tell(),
                        source_size,
                        completed_pairs,
                        now - started,
                        success_counts,
                    )
                    last_progress = now

            destination.flush()
            os.fsync(destination.fileno())

        print_progress(
            source_size,
            source_size,
            completed_pairs,
            time.perf_counter() - started,
            success_counts,
        )
        print()
        temporary_path.replace(pkl_path)
        print(
            f"[INFO] Replaced original PKL atomically: {pkl_path} "
            f"frame_pairs={completed_pairs}"
        )
        print(f"[INFO] Pose success counts: {success_counts}")
        print("[INFO] Pose statistics:")
        finalized_statistics = finalize_pose_statistics(pose_statistics)
        print(json.dumps(finalized_statistics, indent=2))
        report_path = (
            APRILCUBE_ROOT
            / "outputs/offline_pose_statistics"
            / f"{pkl_path.stem}_{OFFLINE_POS_ALGORITHM}.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as report_file:
            json.dump(
                {
                    "pkl_path": str(pkl_path),
                    "algorithm": OFFLINE_POS_ALGORITHM,
                    "pose_backend": POSE_BACKEND,
                    "fallback_backend": "cv2_adaptive_clahe_single_tag_face_frame",
                    "frame_pair_count": completed_pairs,
                    "success_counts": success_counts,
                    "statistics": finalized_statistics,
                },
                report_file,
                indent=2,
            )
        print(f"[INFO] Statistics report: {report_path}")
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate AprilCube poses offline for a 021 synchronized raw-frame PKL."
    )
    parser.add_argument(
        "pkl_path",
        nargs="?",
        type=Path,
        default=PKL_PATH,
        help="021_hand_back_sync_raw_frames_*.pkl to rewrite in place.",
    )
    parser.add_argument(
        "--thumb-web-intrinsics",
        type=Path,
        help="Override thumb_web_cam intrinsics in the PKL header before rewriting.",
    )
    parser.add_argument(
        "--middle-finger-intrinsics",
        type=Path,
        help="Override middle_finger_cam intrinsics in the PKL header before rewriting.",
    )
    args = parser.parse_args()

    pkl_path = args.pkl_path.expanduser().resolve()
    if not pkl_path.is_file():
        raise FileNotFoundError(f"021 PKL not found: {pkl_path}")
    source_header = load_header(pkl_path)
    header = copy.deepcopy(source_header)
    intrinsics_changed = False
    intrinsics_overrides = {
        "thumb_web_cam": args.thumb_web_intrinsics,
        "middle_finger_cam": args.middle_finger_intrinsics,
    }
    intrinsics_by_camera = header.setdefault("metadata", {}).setdefault(
        "camera_intrinsics_yaml", {}
    )
    for camera_name, override in intrinsics_overrides.items():
        if override is None:
            continue
        resolved = override.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"[{camera_name}] intrinsics YAML not found: {resolved}"
            )
        if str(intrinsics_by_camera.get(camera_name, "")) != str(resolved):
            intrinsics_by_camera[camera_name] = str(resolved)
            intrinsics_changed = True
    existing = header.get("metadata", {}).get("offline_pos_estimation")
    if (
        isinstance(existing, dict)
        and existing.get("algorithm") == OFFLINE_POS_ALGORITHM
        and header_has_current_visualizations(header)
        and not intrinsics_changed
    ):
        print(
            f"[INFO] PKL already contains {OFFLINE_POS_ALGORITHM} and "
            f"undistorted visualization images; no rewrite needed: {pkl_path}"
        )
        return

    ensure_enough_space_for_safe_rewrite(pkl_path)
    estimators, offline_pos_keys = build_estimators_from_header(header)
    print(f"[INFO] Input PKL: {pkl_path}")
    for camera_name, estimator in estimators.items():
        print(
            f"[INFO] [{camera_name}] intrinsics={estimator.intrinsics_yaml} "
            f"cubes={[path.name for path in estimator.cube_paths]}"
        )
    print(
        "[INFO] Default pose backend: DeepTag measured internal grid points, "
        "homography RANSAC, IPPE/LM, temporal planar-candidate selection; "
        "CV2/CLAHE fallback only when DeepTag fails."
    )
    estimate_and_rewrite_original_pkl(
        pkl_path,
        header,
        estimators,
        offline_pos_keys,
        source_header,
    )


if __name__ == "__main__":
    main()
