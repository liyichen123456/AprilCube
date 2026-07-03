from __future__ import annotations

import argparse
import copy
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
POSE_CACHE_FORMAT = "aprilcube_008_pose_cache_v1"
TEMPORAL_FILL_MAX_GAP_FRAMES = 30
TEMPORAL_FILL_VERSION = 1


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
        "temporal_filled",
        "temporal_fill_source",
        "temporal_fill_alpha",
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


def quat_to_rvec(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    if q[0] < 0:
        q = -q
    sin_half = float(np.linalg.norm(q[1:]))
    if sin_half < 1e-12:
        return np.zeros((3, 1), dtype=np.float64)
    angle = 2.0 * np.arctan2(sin_half, q[0])
    axis = q[1:] / sin_half
    return (angle * axis).reshape(3, 1)


def slerp_quat(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64).reshape(4)
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    q0 = q0 / max(float(np.linalg.norm(q0)), 1e-12)
    q1 = q1 / max(float(np.linalg.norm(q1)), 1e-12)
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
                adaptive_clahe=False,
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
                recovery_mode = "shared_base"
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

        variants = getattr(detect_mod, "_adaptive_clahe_variants", ())
        for clip_limit, tile_grid_size in variants:
            restore_detector_tracking_state(detector, state_before)
            candidate_tags = detector.detect_tags(
                detect_frame,
                adaptive_clahe=True,
                clahe_variants=((float(clip_limit), tuple(tile_grid_size)),),
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
                    f"adaptive clip={float(clip_limit):.1f} tile={tuple(tile_grid_size)}",
                )

        restore_detector_tracking_state(detector, base_state_after)
        return base_result, base_tags, "base_failed_adaptive_rejected"

    def draw_pose_frame(self, record: dict[str, Any], pose_frame: dict[str, Any]) -> np.ndarray:
        camera_name = pose_frame["camera_name"]
        detect_frame = self.prepare_detect_frame(record["image_bgr"], camera_name)
        vis = self.demo008.make_tag_detection_vis_image(detect_frame)
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
        lines.append(
            f"- `{cube_name}`: t=({tvec[0]:.1f}, {tvec[1]:.1f}, {tvec[2]:.1f}) mm, "
            f"reproj={float(result.get('reproj_error', float('inf'))):.2f}px, "
            f"tags={int(result.get('n_tags', 0))}, faces={faces}{predicted}, "
            f"mode={result.get('clahe_recovery_mode', 'unknown')}{temporal_fill}"
        )
    return "\n".join(lines)


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
    rvec = quat_to_rvec(slerp_quat(q0, q1, alpha))

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
        f"temporal_filled={pose_frame.get('temporal_filled_count', 0)}"
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
        "shared_tag_detection": bool(shared_tag_detection),
        "enable_filter": bool(enable_filter),
        "fast": bool(fast),
        "temporal_fill_enabled": True,
        "temporal_fill_max_gap_frames": int(TEMPORAL_FILL_MAX_GAP_FRAMES),
        "temporal_fill_version": int(TEMPORAL_FILL_VERSION),
        "fisheye_rectified_horizontal_fov_deg": float(
            getattr(demo008, "FISHEYE_RECTIFIED_HORIZONTAL_FOV_DEG", 0.0)
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
            "temporal_fill_enabled",
            "temporal_fill_max_gap_frames",
            "temporal_fill_version",
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
        help="Disable 008 temporal pose filter during sequential replay.",
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
    print(f"[INFO] Indexed frames: {total_frames}")
    if footer is not None:
        print(f"[INFO] Footer frame_count={footer.get('frame_count')} reason={footer.get('reason')}")

    active_camera_names = (
        [x.strip() for x in args.cameras.split(",") if x.strip()]
        if args.cameras
        else list(demo008.ACTIVE_CAMERA_NAMES)
    )
    cube_paths = (
        [demo008.validate_cube_path(Path(x.strip())) for x in args.cube_dirs.split(",") if x.strip()]
        if args.cube_dirs
        else [demo008.validate_cube_path(Path(path)) for path in demo008.CUBE_CFG_DIRS]
    )
    use_undistort = bool(demo008.UNDISTORT_BEFORE_DETECTION) and not args.no_undistort
    adaptive_clahe = bool(getattr(demo008, "ADAPTIVE_CLAHE_DETECTION", False))
    enable_filter = not args.no_filter
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
            print(f"[INFO] Loaded cached temporal-completed pose estimation from PKL: frames={len(pose_cache)}")
        else:
            pose_cache, filled_count = complete_pose_cache_temporally(pose_cache, estimator)
            pose_cache_needs_append = True
            print(
                "[INFO] Loaded cached pose estimation from PKL and applied temporal completion: "
                f"frames={len(pose_cache)} filled={filled_count}"
            )
    else:
        pose_cache = precompute_pose_cache(pkl_path, frame_offsets, metadata, estimator)
        pose_cache, filled_count = complete_pose_cache_temporally(pose_cache, estimator)
        pose_cache_needs_append = True
        print(f"[INFO] Applied temporal completion: filled={filled_count}")

    if pose_cache_needs_append:
        append_pose_cache_to_pkl(pkl_path, pose_cache_key, pose_cache)
        print(f"[INFO] Appended temporal-completed pose estimation to PKL: frames={len(pose_cache)}")

    first_record = load_frame_at_offset(pkl_path, frame_offsets[0])
    first_raw_rgb = bgr_to_rgb_for_viser(first_record["image_bgr"], int(args.max_width))
    first_pose_bgr = estimator.draw_pose_frame(first_record, pose_cache[0])
    first_pose_rgb = bgr_to_rgb_for_viser(first_pose_bgr, int(args.max_width))
    first_undistorted_debug_bgr = estimator.draw_undistorted_debug_frame(first_record, pose_cache[0])
    first_undistorted_debug_rgb = bgr_to_rgb_for_viser(
        first_undistorted_debug_bgr,
        int(args.max_width),
    )

    server = viser.ViserServer(host=args.host, port=int(args.port))
    server.scene.world_axes.visible = False

    with server.gui.add_folder("TagPose Visualization"):
        pose_image_handle = server.gui.add_image(
            first_pose_rgb,
            label="008 pose estimation",
            format="jpeg",
            jpeg_quality=int(args.jpeg_quality),
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

    with server.gui.add_folder("Replay Control"):
        server.gui.add_markdown(
            "Drag `Frame`; raw frames and 008 TagPose overlays are shown in separate sidebar folders."
        )
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=total_frames - 1,
            step=1,
            initial_value=0,
        )
        status_text = server.gui.add_text(
            "Status",
            initial_value=record_summary(first_record, 0, total_frames),
            disabled=True,
        )
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
        "[INFO] Use the sidebar folders: TagPose Visualization, "
        "Undistorted Debug Image, Raw Image, Replay Control."
    )

    current_idx = -1
    while True:
        slider_idx = int(frame_slider.value)
        if slider_idx != current_idx:
            try:
                record = load_frame_at_offset(pkl_path, frame_offsets[slider_idx])
                pose_bgr = estimator.draw_pose_frame(record, pose_cache[slider_idx])
                pose_image_handle.image = bgr_to_rgb_for_viser(
                    pose_bgr,
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
                current_idx = slider_idx
            except Exception as exc:
                status_text.value = f"Failed to load frame {slider_idx}: {type(exc).__name__}: {exc}"
                print(f"[WARNING] {status_text.value}")
                current_idx = slider_idx
        time.sleep(0.03)


if __name__ == "__main__":
    main()
