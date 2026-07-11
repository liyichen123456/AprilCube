from __future__ import annotations

import argparse
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
DEFAULT_RECORDING_DIR = THIS_FILE.parent.parent / "recordings"
VISER_HOST = "0.0.0.0"
VISER_PORT = 8091
DEMO_008_PATH = THIS_FILE.parent / "008_cv2_naive_aprilcube_detect_multi_cube.py"
ASSETS_DIR = THIS_FILE.parent.parent / "assets"
OBJ_MESH_SCALE = 0.001
POSE_CACHE_FORMAT = "aprilcube_008_pose_cache_v1"
IMAGE_RECOVERY_VERSION = 9
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


def resolve_pkl_path(path_str: str | None) -> Path:
    if path_str is None:
        candidates = sorted(DEFAULT_RECORDING_DIR.glob("008_raw_frames_*.pkl"))
        if not candidates:
            raise FileNotFoundError(f"No 008_raw_frames_*.pkl found in {DEFAULT_RECORDING_DIR}")
        return candidates[-1].resolve()

    path = Path(path_str).expanduser().resolve()
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
) -> tuple[dict[str, Any] | None, list[int], dict[str, Any] | None, dict[str, Any] | None]:
    header: dict[str, Any] | None = None
    footer: dict[str, Any] | None = None
    pose_cache_record: dict[str, Any] | None = None
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
            elif record_type == "footer":
                footer = record
            elif record_type == "pose_cache":
                pose_cache_record = record

            now = time.monotonic()
            if now - last_print > 0.5:
                print_index_progress(f.tell(), file_size)
                last_print = now

    print_index_progress(file_size, file_size, force_newline=True)
    return header, frame_offsets, footer, pose_cache_record


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
                result = detector.process_detections(
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
            status_lines.append(self.demo008.result_to_text(camera_name, cube_name, result))
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
        base_result = detector.process_detections(
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
            candidate_result = detector.process_detections(
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
            vis = detector.draw_result(vis, cube["result"])
        vis = self.demo008.draw_text_panel(vis, pose_frame["status_lines"])
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
    demo008: Any,
) -> dict[str, Any]:
    return {
        "format": POSE_CACHE_FORMAT,
        "frame_count": len(frame_offsets),
        "frame_offsets": [int(v) for v in frame_offsets],
        "active_camera_names": list(active_camera_names),
        "cube_paths": [str(path) for path in cube_paths],
        "intrinsics_yaml": {
            name: demo008.CAMERA_TO_INTRINSICS_YAML[name] for name in active_camera_names
        },
        "use_undistort": bool(use_undistort),
        "adaptive_clahe": bool(adaptive_clahe),
        "image_recovery_version": int(IMAGE_RECOVERY_VERSION),
        "shared_tag_detection": bool(shared_tag_detection),
        "enable_filter": bool(enable_filter),
        "fast": bool(fast),
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
) -> tuple[list[dict[str, Any]], bool] | None:
    if not isinstance(pose_cache_record, dict):
        return None
    if pose_cache_record.get("format") != POSE_CACHE_FORMAT:
        return None
    record_key = pose_cache_record.get("key")
    exact_match = record_key == expected_key
    compatible_without_temporal = False
    if not exact_match and isinstance(record_key, dict):
        temporal_keys = {
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


def append_pose_cache_to_pkl(
    pkl_path: Path,
    cache_key: dict[str, Any],
    pose_cache: list[dict[str, Any]],
) -> None:
    record = {
        "type": "pose_cache",
        "format": POSE_CACHE_FORMAT,
        "created_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "key": cache_key,
        "pose_cache": pose_cache,
    }
    with pkl_path.open("ab") as f:
        pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize 008 raw-frame PKL with a Viser sidebar frame slider."
    )
    parser.add_argument(
        "pkl_path",
        nargs="?",
        default=None,
        help="Path to 008_raw_frames_*.pkl, or directory containing such files. Defaults to latest recording.",
    )
    parser.add_argument("--host", type=str, default=VISER_HOST, help="Viser server host.")
    parser.add_argument("--port", type=int, default=VISER_PORT, help="Viser server port.")
    parser.add_argument(
        "--max-width",
        type=int,
        default=1600,
        help="Resize displayed image to this max width. Use 0 for original size.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality for Viser GUI image transport.",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help="Comma-separated logical camera names. Defaults to 008 ACTIVE_CAMERA_NAMES.",
    )
    parser.add_argument(
        "--cube-dirs",
        type=str,
        default=None,
        help="Comma-separated AprilCube cfg directories. Defaults to 008 CUBE_CFG_DIRS.",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Use 008 slow/high-accuracy detector parameters.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable 008 runtime temporal pose filter during sequential replay.",
    )
    parser.add_argument(
        "--with-filter",
        action="store_true",
        help=(
            "Enable the 008 runtime temporal pose filter during replay. "
            "The offline viewer disables it by default and applies its own postprocess."
        ),
    )
    parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="Do not run the 008 fisheye rectification path before detection.",
    )
    parser.add_argument(
        "--shared-detect-tags",
        action="store_true",
        help="Use the realtime 008 shared detect_tags() path. Default offline mode detects tags per cube.",
    )
    args = parser.parse_args()

    demo008 = load_demo008_module()
    pkl_path = resolve_pkl_path(args.pkl_path)
    print(f"[INFO] PKL: {pkl_path}")
    print("[INFO] Building lightweight frame index. This scans the file once without retaining images.")
    header, frame_offsets, footer, pose_cache_record = build_frame_index(pkl_path)
    if not frame_offsets:
        raise ValueError(f"No frame records found in {pkl_path}")

    total_frames = len(frame_offsets)
    metadata = header.get("metadata", {}) if isinstance(header, dict) else {}
    first_record = load_frame_at_offset(pkl_path, frame_offsets[0])
    first_record_camera_name = str(first_record.get("camera_name", ""))
    print(f"[INFO] Indexed frames: {total_frames}")
    if footer is not None:
        print(f"[INFO] Footer frame_count={footer.get('frame_count')} reason={footer.get('reason')}")

    if args.cameras:
        active_camera_names = [x.strip() for x in args.cameras.split(",") if x.strip()]
    else:
        active_camera_names = list(demo008.ACTIVE_CAMERA_NAMES)
        if (
            first_record_camera_name
            and len(active_camera_names) == 1
            and first_record_camera_name != active_camera_names[0]
        ):
            config_camera_name = active_camera_names[0]
            active_camera_names = [first_record_camera_name]
            demo008.CAMERA_TO_INTRINSICS_YAML[first_record_camera_name] = (
                demo008.CAMERA_TO_INTRINSICS_YAML[config_camera_name]
            )
            print(
                "[INFO] Historical PKL camera alias: "
                f"recorded camera '{first_record_camera_name}' uses current 008 "
                f"config '{config_camera_name}'."
            )
    missing_camera_configs = [
        name for name in active_camera_names if name not in demo008.CAMERA_TO_INTRINSICS_YAML
    ]
    if missing_camera_configs and len(demo008.ACTIVE_CAMERA_NAMES) == 1:
        config_camera_name = demo008.ACTIVE_CAMERA_NAMES[0]
        for camera_name in missing_camera_configs:
            demo008.CAMERA_TO_INTRINSICS_YAML[camera_name] = (
                demo008.CAMERA_TO_INTRINSICS_YAML[config_camera_name]
            )
        print(
            "[INFO] Historical PKL camera alias: "
            f"{missing_camera_configs} use current 008 config '{config_camera_name}'."
        )
    cube_paths = (
        [demo008.validate_cube_path(Path(x.strip())) for x in args.cube_dirs.split(",") if x.strip()]
        if args.cube_dirs
        else [demo008.validate_cube_path(Path(path)) for path in demo008.CUBE_CFG_DIRS]
    )
    use_undistort = bool(demo008.UNDISTORT_BEFORE_DETECTION) and not args.no_undistort
    adaptive_clahe = bool(getattr(demo008, "ADAPTIVE_CLAHE_DETECTION", False))
    enable_filter = bool(args.with_filter) and not args.no_filter
    fast = not args.slow
    estimator = ReplayPoseEstimator(
        demo008,
        active_camera_names=active_camera_names,
        cube_paths=cube_paths,
        use_undistort=use_undistort,
        adaptive_clahe=adaptive_clahe,
        shared_tag_detection=bool(args.shared_detect_tags),
        enable_filter=enable_filter,
        fast=fast,
    )
    pose_cache_key = make_pose_cache_key(
        frame_offsets=frame_offsets,
        active_camera_names=active_camera_names,
        cube_paths=cube_paths,
        use_undistort=use_undistort,
        adaptive_clahe=adaptive_clahe,
        shared_tag_detection=bool(args.shared_detect_tags),
        enable_filter=enable_filter,
        fast=fast,
        demo008=demo008,
    )
    print(
        "[INFO] 008 replay detection path: "
        f"{'shared' if args.shared_detect_tags else 'per-cube'} detect_tags(frame) "
        "+ per-cube process_detections(), sequential over PKL frames."
    )
    cached_pose = load_cached_pose_cache(pose_cache_record, pose_cache_key)
    pose_cache_needs_append = False
    if cached_pose is not None:
        pose_cache, cache_exact_match = cached_pose
        if cache_exact_match:
            print(f"[INFO] Loaded cached temporal-completed smoothed pose estimation from PKL: frames={len(pose_cache)}")
        else:
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
            pose_cache_needs_append = True
            print(
                "[INFO] Loaded cached pose estimation from PKL and applied "
                "single-tag gate + temporal completion+smoothing: "
                f"frames={len(pose_cache)} reset={reset_count} "
                f"rejected={rejected_count} filled={filled_count} smoothed={smoothed_count}"
            )
    else:
        pose_cache = precompute_pose_cache(pkl_path, frame_offsets, metadata, estimator)
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
        pose_cache_needs_append = True
        print(
            "[INFO] Applied single-tag gate + temporal completion+smoothing: "
            f"reset={reset_count} rejected={rejected_count} "
            f"filled={filled_count} smoothed={smoothed_count}"
        )

    if pose_cache_needs_append:
        append_pose_cache_to_pkl(pkl_path, pose_cache_key, pose_cache)
        print(f"[INFO] Appended temporal-completed smoothed pose estimation to PKL: frames={len(pose_cache)}")

    first_raw_rgb = bgr_to_rgb_for_viser(first_record["image_bgr"], int(args.max_width))
    first_detector_tagpose_bgr = estimator.draw_detector_input_pose_frame(first_record, pose_cache[0])
    first_detector_tagpose_rgb = bgr_to_rgb_for_viser(
        first_detector_tagpose_bgr,
        int(args.max_width),
    )
    first_undistorted_debug_bgr = estimator.draw_undistorted_debug_frame(first_record, pose_cache[0])
    first_undistorted_debug_rgb = bgr_to_rgb_for_viser(
        first_undistorted_debug_bgr,
        int(args.max_width),
    )

    server = viser.ViserServer(host=args.host, port=int(args.port))
    scene_handles = create_3d_scene_handles(server, estimator, pose_cache)
    update_3d_scene(scene_handles, pose_cache[0])

    with server.gui.add_folder("Detector Input TagPose"):
        detector_tagpose_handle = server.gui.add_image(
            first_detector_tagpose_rgb,
            label="",
            format="jpeg",
            jpeg_quality=int(args.jpeg_quality),
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
            jpeg_quality=int(args.jpeg_quality),
        )

    with server.gui.add_folder("Raw Image"):
        raw_image_handle = server.gui.add_image(
            first_raw_rgb,
            label="raw origin_frame_bgr",
            format="jpeg",
            jpeg_quality=int(args.jpeg_quality),
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

    print(f"[INFO] Viser: http://{args.host}:{int(args.port)}")
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
                    int(args.max_width),
                )
                undistorted_debug_bgr = estimator.draw_undistorted_debug_frame(
                    record,
                    pose_cache[slider_idx],
                )
                undistorted_debug_handle.image = bgr_to_rgb_for_viser(
                    undistorted_debug_bgr,
                    int(args.max_width),
                )
                raw_image_handle.image = bgr_to_rgb_for_viser(record["image_bgr"], int(args.max_width))
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
    main()
