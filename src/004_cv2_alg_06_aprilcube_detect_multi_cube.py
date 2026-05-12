# OpenCV 相机系
# +x: image right
# +y: image down
# +z: camera forward
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from pupil_apriltags import Detector

THIS_FILE = Path(__file__).resolve()
THIRDPARTY_DIR = THIS_FILE.parent.parent.parent
PROJECT_ROOT = THIRDPARTY_DIR.parent
RECORDER_UTILS_DIR = PROJECT_ROOT / "scripts" / "utils"

if str(RECORDER_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(RECORDER_UTILS_DIR))

import aprilcube  # noqa: E402
from april_tag_detector import TemporalTagPoseEstimator  # noqa: E402
from aprilcube.detect import FACE_DEFS, _quad_quality, estimate_pose  # noqa: E402
from aprilcube_runtime import AprilCubeTemporalPoseRuntime  # noqa: E402
from recorder_cv2_cam import CV2CameraManager  # noqa: E402


def load_intrinsics_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path).expanduser().resolve()
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return {
        "path": str(yaml_path),
        "image_size": tuple(int(v) for v in data["image_size"]),
        "K": np.asarray(data["K"], dtype=np.float64).reshape(3, 3),
        "dist": np.asarray(data["dist"], dtype=np.float64).reshape(-1),
    }


# ============================================================
# User macros
# ============================================================

CAMERA_TO_PORT: dict[str, str] = {
    "cam0": "4-9.4.4.1:1.0",
}

CAMERA_TO_INTRINSICS_YAML: dict[str, str] = {
    "cam0": "/home/ps/RobotCamCalib1/outputs/intrinsics_0511_163437.yaml",
}

ACTIVE_CAMERA_NAMES: list[str] = ["cam0"]

CAPTURE_SIZE: tuple[int, int] = (1920, 1080)  # width, height
DETECT_IMG_SIZE: tuple[int, int] = (1920, 1080)  # width, height
FPS = 120
FOURCC = "MJPG"
WINDOW_PREFIX = "CV2 Multi-AprilCube"
PRINT_EVERY_N_FRAMES = 5
DRAW_TAG_FRAME_2D = True
TAG_AXIS_LENGTH_SCALE = 0.8
UNDISTORT_BEFORE_DETECTION = True

CUBE_CFG_DIRS: list[Path] = [
    THIRDPARTY_DIR / "aprilcube" / "cube_april_36h11_0_5_1x1x1_10mm",
    THIRDPARTY_DIR / "aprilcube" / "cube_april_36h11_6_11_1x1x1_10mm",
    THIRDPARTY_DIR / "aprilcube" / "cube_april_36h11_12_17_1x1x1_10mm",
    THIRDPARTY_DIR / "aprilcube" / "cube_april_36h11_18_23_1x1x1_10mm",
    THIRDPARTY_DIR / "aprilcube" / "cube_april_36h11_24_29_1x1x1_10mm",
    THIRDPARTY_DIR / "aprilcube" / "cube_april_36h11_30_35_1x1x1_10mm",
]

CALIB_BY_CAMERA = {name: load_intrinsics_yaml(path) for name, path in CAMERA_TO_INTRINSICS_YAML.items()}
K_BY_CAMERA: dict[str, np.ndarray] = {name: calib["K"] for name, calib in CALIB_BY_CAMERA.items()}
K_ORIGINAL_SIZE_BY_CAMERA: dict[str, tuple[int, int]] = {
    name: calib["image_size"] for name, calib in CALIB_BY_CAMERA.items()
}
DIST_COEFFS_BY_CAMERA: dict[str, np.ndarray | None] = {
    name: calib["dist"] for name, calib in CALIB_BY_CAMERA.items()
}

ENABLE_FILTER = True
FAST_DETECTOR = True
USE_TEMPORAL_TAG_POSE_ESTIMATOR = True
PUPIL_TO_OBJECT_CORNER_INDEX = [2, 1, 0, 3]
USE_SOLVEPNP_REFINE_LM = True
USE_TEMPORAL_CANDIDATE_SELECTION = True
SOLVEPNP_GENERIC_FLAG = cv2.SOLVEPNP_IPPE
SOLVEPNP_FLAG = cv2.SOLVEPNP_ITERATIVE
TRANSLATION_SCORE_WEIGHT_DEG_PER_MM = 1.0
REJECT_NEGATIVE_CAMERA_Z = True
ALGORITHM_NAME = "pupil_corners_to_aprilcube_cube_frame_pnp"

# Native pupil_apriltags detection settings. alg_06 only needs native-detected
# tag corners; pose candidates are generated below with OpenCV IPPE.
NATIVE_QUAD_DECIMATE = 1.0
NATIVE_QUAD_SIGMA = 0.0
NATIVE_REFINE_EDGES = 1
NATIVE_DECODE_SHARPENING = 0.25
NATIVE_DEBUG = 0


def scale_intrinsics(
    k: np.ndarray,
    old_size: tuple[int, int],
    new_size: tuple[int, int],
) -> np.ndarray:
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
        raise ValueError(f"Unsupported native AprilTag family: {dict_name}")
    return family_map[dict_name]


def create_detector_for_camera(cube_path: Path, camera_name: str) -> Any:
    if camera_name not in K_BY_CAMERA:
        raise KeyError(f"Missing intrinsics for camera '{camera_name}'.")

    k_scaled = scale_intrinsics(
        K_BY_CAMERA[camera_name],
        old_size=K_ORIGINAL_SIZE_BY_CAMERA[camera_name],
        new_size=DETECT_IMG_SIZE,
    )
    intrinsic_cfg = camera_matrix_to_intrinsic_dict(k_scaled)
    dist_coeffs = DIST_COEFFS_BY_CAMERA.get(camera_name)
    if dist_coeffs is not None:
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)

    detector_dist_coeffs = dist_coeffs
    if UNDISTORT_BEFORE_DETECTION:
        detector_dist_coeffs = np.zeros(5, dtype=np.float64)

    return aprilcube.detector(
        cube_path,
        intrinsic_cfg=intrinsic_cfg,
        dist_coeffs=detector_dist_coeffs,
        enable_filter=ENABLE_FILTER,
        fast=FAST_DETECTOR,
    )


def create_pose_estimator(detector: Any) -> TemporalTagPoseEstimator:
    return TemporalTagPoseEstimator(
        tag_size_m=float(detector.config.tag_size_mm) / 1000.0,
        pupil_to_object_corner_index=PUPIL_TO_OBJECT_CORNER_INDEX,
        solvepnp_generic_flag=SOLVEPNP_GENERIC_FLAG,
        solvepnp_flag=SOLVEPNP_FLAG,
        use_temporal_candidate_selection=USE_TEMPORAL_TAG_POSE_ESTIMATOR and USE_TEMPORAL_CANDIDATE_SELECTION,
        use_solvepnp_refine_lm=USE_SOLVEPNP_REFINE_LM,
        translation_score_weight_deg_per_mm=TRANSLATION_SCORE_WEIGHT_DEG_PER_MM,
        reject_negative_camera_z=REJECT_NEGATIVE_CAMERA_Z,
    )


class AprilCubePupilCornersPoseRuntime(AprilCubeTemporalPoseRuntime):
    """Use pupil_apriltags for corners, then AprilCube's cube-frame PnP logic."""

    def detect_native_apriltags_all(self, image: np.ndarray) -> list[Any]:
        gray = self.prepare_native_detection_gray(image)
        tags = self.native_detector.detect(
            np.asarray(gray, dtype=np.uint8),
            estimate_tag_pose=False,
        )
        return list(tags)

    def detect_native_apriltags(self, image: np.ndarray) -> list[Any]:
        return self.filter_native_tags_for_cube(self.detect_native_apriltags_all(image))

    def filter_native_tags_for_cube(self, native_tags: list[Any]) -> list[Any]:
        return [tag for tag in native_tags if int(tag.tag_id) in self.detector.valid_ids]

    @staticmethod
    def reorder_pupil_corners_to_cube_order(corners_xy: np.ndarray) -> np.ndarray:
        corners = np.asarray(corners_xy, dtype=np.float64).reshape(4, 2)
        return corners[np.asarray(PUPIL_TO_OBJECT_CORNER_INDEX, dtype=np.int64)]

    def detections_from_native_tags(self, native_tags: list[Any]) -> list[tuple[int, np.ndarray]]:
        detections = []
        for tag in native_tags:
            tag_id = int(tag.tag_id)
            corners = self.reorder_pupil_corners_to_cube_order(tag.corners)
            if _quad_quality(corners) > 0.15:
                detections.append((tag_id, corners))
        return detections

    def process_frame(
        self,
        camera_name: str,
        image: np.ndarray,
        native_tags: list[Any] | None = None,
    ) -> dict[str, Any]:
        del camera_name
        timestamp = time.monotonic()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

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
            "algorithm_debug": {"algorithm_name": ALGORITHM_NAME},
        }

        if native_tags is None:
            raw_tags = self.detect_native_apriltags_all(image)
            raw_count = len(raw_tags)
            native_tags = self.filter_native_tags_for_cube(raw_tags)
        else:
            raw_count = len(native_tags)
            native_tags = self.filter_native_tags_for_cube(native_tags)

        result["algorithm_debug"]["native_raw_tag_count"] = int(raw_count)
        result["algorithm_debug"]["native_valid_tag_count"] = int(len(native_tags))

        detections = self.detections_from_native_tags(native_tags)
        result["detections"] = detections
        result["n_tags"] = len(detections)
        result["tag_ids"] = [tag_id for tag_id, _corners in detections]

        for tag_id, _corners in detections:
            for face_name, id_set in self.detector.face_id_sets.items():
                if tag_id in id_set:
                    result["visible_faces"].add(face_name)

        if not detections:
            self.detector._prev_gray = np.asarray(gray, dtype=np.uint8).copy()
            return self.detector._store_latest(self.detector._try_predict(timestamp, result), image)

        object_points = np.vstack([
            np.asarray(self.detector.tag_corner_map[int(tag_id)], dtype=np.float64).reshape(4, 3)
            for tag_id, _corners in detections
        ])
        image_points = np.vstack([
            np.asarray(corners_2d, dtype=np.float64).reshape(4, 2)
            for _tag_id, corners_2d in detections
        ])

        pnp_rv_guess = self.detector.prev_rvec
        pnp_tv_guess = self.detector.prev_tvec
        if pnp_rv_guess is None and self.detector.pose_filter and self.detector.pose_filter.is_initialized:
            pred = self.detector.pose_filter.predict(timestamp)
            if pred is not None:
                pnp_rv_guess, pnp_tv_guess = pred

        success, rvec, tvec, reproj_err, inliers = estimate_pose(
            object_points,
            image_points,
            self.detector.camera_matrix,
            self.detector.dist_coeffs,
            pnp_rv_guess,
            pnp_tv_guess,
        )

        if success and len(detections) >= 3:
            detections, object_points, image_points, success, rvec, tvec, reproj_err, inliers = (
                self._drop_reprojection_outlier_tags_and_resolve(
                    detections=detections,
                    object_points=object_points,
                    image_points=image_points,
                    rvec=rvec,
                    tvec=tvec,
                    inliers=inliers,
                    pnp_rv_guess=pnp_rv_guess,
                    pnp_tv_guess=pnp_tv_guess,
                )
            )
            result["detections"] = detections
            result["n_tags"] = len(detections)
            result["tag_ids"] = [tag_id for tag_id, _corners in detections]
            result["visible_faces"] = self.visible_faces_for_detections(detections)

        max_reproj = 3.0
        if not success or rvec is None or tvec is None or reproj_err > max_reproj:
            result["algorithm_debug"]["reject_reason"] = "pnp_failed_or_reprojection_too_high"
            result["algorithm_debug"]["raw_reproj_error"] = float(reproj_err)
            return self.detector._store_latest(self.detector._try_predict(timestamp, result), image)

        cube_rot_mat, _ = cv2.Rodrigues(rvec)
        if not self._visible_face_normals_are_camera_facing(cube_rot_mat, result["visible_faces"]):
            result["algorithm_debug"]["reject_reason"] = "visible_face_normal_flipped"
            return self.detector._store_latest(self.detector._try_predict(timestamp, result), image)

        if self.detector.prev_rvec is not None and self.detector.prev_tvec is not None:
            if self._is_temporal_jump_too_large(timestamp, cube_rot_mat, tvec):
                result["algorithm_debug"]["reject_reason"] = "temporal_jump_too_large"
                return self.detector._store_latest(self.detector._try_predict(timestamp, result), image)
        elif reproj_err > 2.5:
            result["algorithm_debug"]["reject_reason"] = "cold_start_reprojection_too_high"
            return self.detector._store_latest(self.detector._try_predict(timestamp, result), image)

        n_inlier_count = len(inliers) if inliers is not None else 0
        if self.detector.pose_filter:
            rvec, tvec = self.detector.pose_filter.update(
                rvec,
                tvec,
                timestamp,
                reproj_error=reproj_err,
                n_tags=max(len(detections), 1),
                n_inliers=n_inlier_count,
            )
            cube_rot_mat, _ = cv2.Rodrigues(rvec)

        self.detector.prev_rvec = rvec.copy()
        self.detector.prev_tvec = tvec.copy()
        self.detector._save_corners_for_flow(np.asarray(gray, dtype=np.uint8), detections)

        result["success"] = True
        result["rvec"] = rvec
        result["tvec"] = tvec
        result["T"] = np.eye(4, dtype=np.float64)
        result["T"][:3, :3] = cube_rot_mat
        result["T"][:3, 3] = tvec.reshape(3)
        result["reproj_error"] = float(reproj_err)
        result["n_inliers"] = int(n_inlier_count)
        result["algorithm_debug"]["mode"] = "pupil_corners_aprilcube_estimate_pose"
        result["algorithm_debug"]["num_points"] = int(object_points.shape[0])
        return self.detector._store_latest(result, image)

    def visible_faces_for_detections(self, detections: list[tuple[int, np.ndarray]]) -> set[str]:
        faces: set[str] = set()
        for tag_id, _corners in detections:
            for face_name, id_set in self.detector.face_id_sets.items():
                if int(tag_id) in id_set:
                    faces.add(face_name)
        return faces

    def _drop_reprojection_outlier_tags_and_resolve(
        self,
        *,
        detections: list[tuple[int, np.ndarray]],
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        inliers: np.ndarray | None,
        pnp_rv_guess: np.ndarray | None,
        pnp_tv_guess: np.ndarray | None,
    ) -> tuple[list[tuple[int, np.ndarray]], np.ndarray, np.ndarray, bool, np.ndarray | None, np.ndarray | None, float, np.ndarray | None]:
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.detector.camera_matrix,
            self.detector.dist_coeffs,
        )
        projected = projected.reshape(-1, 2)
        per_tag_err = []
        for idx in range(len(detections)):
            start, end = idx * 4, (idx + 1) * 4
            per_tag_err.append(float(np.mean(np.linalg.norm(image_points[start:end] - projected[start:end], axis=1))))

        median_err = float(np.median(per_tag_err))
        tag_reproj_thresh = max(median_err * 3.0, 2.0)
        keep = [idx for idx, err in enumerate(per_tag_err) if err <= tag_reproj_thresh]
        if len(keep) == len(detections) or len(keep) < 1:
            return detections, object_points, image_points, True, rvec, tvec, float(
                np.mean(np.linalg.norm(image_points - projected, axis=1))
            ), inliers

        kept_detections = [detections[idx] for idx in keep]
        kept_object_points = np.vstack([
            np.asarray(self.detector.tag_corner_map[int(tag_id)], dtype=np.float64).reshape(4, 3)
            for tag_id, _corners in kept_detections
        ])
        kept_image_points = np.vstack([
            np.asarray(corners_2d, dtype=np.float64).reshape(4, 2)
            for _tag_id, corners_2d in kept_detections
        ])
        success, next_rvec, next_tvec, next_reproj, next_inliers = estimate_pose(
            kept_object_points,
            kept_image_points,
            self.detector.camera_matrix,
            self.detector.dist_coeffs,
            pnp_rv_guess,
            pnp_tv_guess,
        )
        return (
            kept_detections,
            kept_object_points,
            kept_image_points,
            success,
            next_rvec,
            next_tvec,
            next_reproj,
            next_inliers,
        )

    @staticmethod
    def _visible_face_normals_are_camera_facing(cube_rot_mat: np.ndarray, visible_faces: set[str]) -> bool:
        for face_name in visible_faces:
            for face_def in FACE_DEFS:
                if face_def[0] != face_name:
                    continue
                normal_obj = np.zeros(3, dtype=np.float64)
                normal_obj[face_def[1]] = face_def[2]
                normal_cam = np.asarray(cube_rot_mat, dtype=np.float64).reshape(3, 3) @ normal_obj
                if normal_cam[2] > 0.0:
                    return False
                break
        return True

    def _is_temporal_jump_too_large(
        self,
        timestamp: float,
        cube_rot_mat: np.ndarray,
        tvec: np.ndarray,
    ) -> bool:
        prev_rvec = self.detector.prev_rvec
        prev_tvec = self.detector.prev_tvec
        if prev_rvec is None or prev_tvec is None:
            return False

        jump_mm = float(np.linalg.norm(np.asarray(tvec, dtype=np.float64).reshape(3) - prev_tvec.reshape(3)))
        prev_rot_mat, _ = cv2.Rodrigues(prev_rvec)
        angle = np.arccos(np.clip((np.trace(prev_rot_mat.T @ cube_rot_mat) - 1.0) * 0.5, -1.0, 1.0))
        max_jump_mm = 100.0
        max_angle = np.radians(45.0)
        pose_filter = self.detector.pose_filter
        if pose_filter and pose_filter.is_initialized:
            speed = float(np.linalg.norm(pose_filter._x_t[3:6]))
            omega = float(np.linalg.norm(pose_filter._x_r[3:6]))
            dt = max(float(timestamp - pose_filter._last_ts), 1.0 / 60.0)
            max_jump_mm = max(100.0, speed * dt * 3.0)
            max_angle = max(np.radians(45.0), omega * dt * 3.0)
        return jump_mm > max_jump_mm or angle > max_angle


def validate_cube_path(cube_path: Path) -> Path:
    cube_path = cube_path.resolve()
    if cube_path.is_dir() and (cube_path / "config.json").is_file():
        return cube_path
    if cube_path.is_file() and cube_path.name == "config.json":
        return cube_path
    raise FileNotFoundError(f"Invalid AprilCube cfg path: {cube_path}")


def rotation_matrix_to_euler_xyz_deg(rot_mat: np.ndarray) -> np.ndarray:
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


def rotation_handedness_text(rot: np.ndarray | None) -> str:
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


def make_handedness_overlay_text(result: dict[str, Any] | None) -> str:
    if not result:
        return "handedness: no result"

    tag_pose_by_id = result.get("tag_pose_by_id", {})
    if not tag_pose_by_id:
        tag_ids = result.get("tag_ids", [])
        if tag_ids:
            return "hand no pose ids=" + ",".join(str(int(tag_id)) for tag_id in tag_ids)
        return "handedness: no tags"

    parts = []
    for tag_id in sorted(tag_pose_by_id):
        handed = rotation_handedness_text(tag_pose_by_id[tag_id].get("rot_mat", None))
        short = "?"
        if handed.startswith("right-handed"):
            short = "R"
        elif handed.startswith("left-handed"):
            short = "L"
        elif handed.startswith("invalid"):
            short = "I"
        elif handed.startswith("degenerate"):
            short = "D"
        parts.append(f"{tag_id}:{short}")
    return "hand " + " ".join(parts)


def result_to_text(camera_name: str, cube_name: str, result: dict[str, Any] | None) -> str:
    prefix = f"[{camera_name}][{cube_name}]"
    if not result:
        return f"{prefix} no result"
    if not result.get("success", False):
        text = f"{prefix} cube not detected"
        debug = result.get("algorithm_debug", {})
        raw_count = debug.get("native_raw_tag_count", None)
        valid_count = debug.get("native_valid_tag_count", None)
        reject_reason = debug.get("reject_reason", None)
        raw_reproj = debug.get("raw_reproj_error", None)
        tag_ids = result.get("tag_ids", [])
        faces = result.get("visible_faces", None)
        if valid_count is not None and raw_count is not None:
            text += f" native_tags={int(valid_count)}/{int(raw_count)}"
        if tag_ids:
            text += f" ids={list(tag_ids)}"
        if faces:
            text += f" faces={sorted(list(faces))}"
        if reject_reason:
            text += f" reject={reject_reason}"
        if raw_reproj is not None and np.isfinite(float(raw_reproj)):
            text += f" raw_reproj={float(raw_reproj):.2f}px"
        return text

    tvec = np.asarray(result["tvec"], dtype=np.float64).reshape(-1)
    text = f"{prefix} t=({tvec[0]:.1f},{tvec[1]:.1f},{tvec[2]:.1f})"

    if result.get("rvec", None) is not None:
        rot_mat, _ = cv2.Rodrigues(np.asarray(result["rvec"], dtype=np.float64).reshape(3, 1))
        euler = rotation_matrix_to_euler_xyz_deg(rot_mat)
        text += f" rot=({euler[0]:.1f},{euler[1]:.1f},{euler[2]:.1f})"

    error = result.get("reproj_error", None)
    if error is not None:
        text += f" reproj={float(error):.2f}px"

    faces = result.get("visible_faces", None)
    if faces is not None:
        text += f" faces={sorted(list(faces))}"

    inward = result.get("tag_z_inward_count", None)
    invalid = result.get("tag_z_invalid_count", None)
    if inward is not None and invalid is not None:
        text += f" z_in={int(inward)} z_out={int(invalid)}"

    debug = result.get("algorithm_debug", {})
    raw_count = debug.get("native_raw_tag_count", None)
    valid_count = debug.get("native_valid_tag_count", None)
    selected_count = debug.get("selected_candidate_count", None)
    if raw_count is not None and valid_count is not None:
        text += f" native_tags={int(valid_count)}/{int(raw_count)}"
    if selected_count is not None:
        text += f" selected={int(selected_count)}"

    return text


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
        y += 24
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect multiple AprilCube cfgs using CV2 camera input.")
    parser.add_argument(
        "--cameras",
        type=str,
        default=",".join(ACTIVE_CAMERA_NAMES),
        help="Comma-separated logical camera names.",
    )
    parser.add_argument(
        "--cube-dirs",
        type=str,
        default=",".join(str(path) for path in CUBE_CFG_DIRS),
        help="Comma-separated AprilCube cfg directories.",
    )
    args = parser.parse_args()

    active_camera_names = [x.strip() for x in args.cameras.split(",") if x.strip()]
    cube_paths = [validate_cube_path(Path(x.strip())) for x in args.cube_dirs.split(",") if x.strip()]
    if not active_camera_names:
        print("[ERROR] No active camera names specified.")
        sys.exit(1)
    if not cube_paths:
        print("[ERROR] No cube cfg paths specified.")
        sys.exit(1)

    missing_camera_cfg = [name for name in active_camera_names if name not in CAMERA_TO_PORT]
    if missing_camera_cfg:
        print(f"[ERROR] Missing CAMERA_TO_PORT entries for: {missing_camera_cfg}")
        sys.exit(1)
    missing_intrinsics_cfg = [name for name in active_camera_names if name not in CALIB_BY_CAMERA]
    if missing_intrinsics_cfg:
        print(f"[ERROR] Missing CAMERA_TO_INTRINSICS_YAML entries for: {missing_intrinsics_cfg}")
        sys.exit(1)

    for camera_name in active_camera_names:
        calib = CALIB_BY_CAMERA[camera_name]
        print(
            f"[INFO] [{camera_name}] intrinsics_yaml={calib['path']} "
            f"image_size={calib['image_size']}"
        )

    runtimes_by_camera: dict[str, list[dict[str, Any]]] = {name: [] for name in active_camera_names}
    shared_native_detectors: dict[str, Detector] = {}

    for cube_path in cube_paths:
        cube_name = cube_path.name if cube_path.is_dir() else cube_path.parent.name
        for camera_name in active_camera_names:
            detector = create_detector_for_camera(cube_path, camera_name)
            native_family = apriltag_family_from_dict_name(detector.config.dict_name)
            if native_family not in shared_native_detectors:
                shared_native_detectors[native_family] = Detector(
                    families=native_family,
                    quad_decimate=float(NATIVE_QUAD_DECIMATE),
                    quad_sigma=float(NATIVE_QUAD_SIGMA),
                    refine_edges=int(NATIVE_REFINE_EDGES),
                    decode_sharpening=float(NATIVE_DECODE_SHARPENING),
                    debug=int(NATIVE_DEBUG),
                )

            runtime = AprilCubePupilCornersPoseRuntime(
                detector=detector,
                native_detector=shared_native_detectors[native_family],
                pose_estimator=None,
            )
            runtimes_by_camera[camera_name].append(
                {
                    "cube_name": cube_name,
                    "runtime": runtime,
                    "detector": detector,
                }
            )
            print(f"[INFO] Loaded cube cfg for {camera_name}: {cube_name}")

    camera_manager = CV2CameraManager(
        camera_to_port={name: CAMERA_TO_PORT[name] for name in active_camera_names},
        capture_size=CAPTURE_SIZE,
        fps=FPS,
        fourcc=FOURCC,
    )

    try:
        opened = camera_manager.open_all_cameras()
        if opened == 0:
            print("[ERROR] No CV2 camera opened.")
            sys.exit(1)

        opened_names = camera_manager.get_active_camera_names()
        print(f"[INFO] Opened CV2 cameras: {opened_names}")
        print(f"[INFO] Pose algorithm: {ALGORITHM_NAME}")
        print("[INFO] Press 'q' or ESC to quit.")

        frame_idx = 0
        last_no_frame_print_time = 0.0
        while True:
            frame_idx += 1
            frames, _origin_frames, _timestamps = camera_manager.get_frames(
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
                runtime_entries = runtimes_by_camera[camera_name]
                detect_frame = frame
                if UNDISTORT_BEFORE_DETECTION:
                    raw_dist_coeffs = DIST_COEFFS_BY_CAMERA.get(camera_name)
                    if raw_dist_coeffs is not None:
                        raw_dist_coeffs = np.asarray(raw_dist_coeffs, dtype=np.float64)
                        detect_frame = cv2.undistort(
                            frame,
                            runtime_entries[0]["detector"].camera_matrix,
                            raw_dist_coeffs,
                        )

                vis = detect_frame.copy()
                fps_text = camera_manager.get_latest_fps(camera_name)
                status_lines = [
                    f"[{camera_name}] cubes={len(runtime_entries)} detect_size={DETECT_IMG_SIZE} "
                    f"capture_size={CAPTURE_SIZE} fps={fps_text:.1f}" if fps_text is not None else
                    f"[{camera_name}] cubes={len(runtime_entries)} detect_size={DETECT_IMG_SIZE} capture_size={CAPTURE_SIZE}"
                ]

                grouped_entries: dict[tuple[str, float], list[dict[str, Any]]] = {}
                for entry in runtime_entries:
                    runtime = entry["runtime"]
                    key = (runtime.native_family, round(runtime.tag_size_m, 6))
                    grouped_entries.setdefault(key, []).append(entry)

                for _group_key, group_entries in grouped_entries.items():
                    shared_tags = group_entries[0]["runtime"].detect_native_apriltags_all(detect_frame)
                    for entry in group_entries:
                        cube_name = entry["cube_name"]
                        detector = entry["detector"]
                        runtime = entry["runtime"]
                        result = runtime.process_frame(
                            camera_name=camera_name,
                            image=detect_frame,
                            native_tags=shared_tags,
                        )

                        try:
                            vis = detector.draw_result(vis, result)
                        except Exception as exc:
                            print(f"[WARNING] draw_result failed for {camera_name}/{cube_name}: {type(exc).__name__}: {exc}")

                        vis = runtime.draw_detected_tag_visuals(
                            img=vis,
                            result=result,
                            draw_tag_frame_2d=DRAW_TAG_FRAME_2D,
                            tag_axis_length_scale=TAG_AXIS_LENGTH_SCALE,
                        )

                        line = result_to_text(camera_name, cube_name, result)
                        status_lines.append(line)
                        status_lines.append(f"  {cube_name} {make_handedness_overlay_text(result)}")

                        if frame_idx % PRINT_EVERY_N_FRAMES == 0:
                            print(line)

                status_lines.append("press q or ESC to quit")
                vis = draw_text_panel(vis, status_lines)
                cv2.imshow(f"{WINDOW_PREFIX}: {camera_name}", vis)

            key = cv2.waitKey(1)
            if key == 27 or key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        camera_manager.release_all()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
