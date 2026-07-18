#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import viser

THIS_FILE = Path(__file__).resolve()
APRILCUBE_ROOT = THIS_FILE.parent.parent
SCRIPT_012_PATH = THIS_FILE.parent / "012_rs_aprilcube_detect.py"
SCRIPT_015_PATH = THIS_FILE.parent / "015_deeptag_012_pkl.py"
DEFAULT_RECORDING_DIR = APRILCUBE_ROOT / "recordings"
DEFAULT_PORT = 8094
PLAYBACK_FPS = 15.0

if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))

import aprilcube  # noqa: E402


def load_script012_module() -> Any:
    spec = importlib.util.spec_from_file_location("aprilcube_replay_012", SCRIPT_012_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load 012 logic from {SCRIPT_012_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["aprilcube_replay_012"] = module
    spec.loader.exec_module(module)
    return module


def load_script015_module() -> Any:
    spec = importlib.util.spec_from_file_location("aprilcube_replay_015_deeptag", SCRIPT_015_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load DeepTag logic from {SCRIPT_015_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate offline AprilCube poses from a 012 raw-frame pkl, optionally "
            "recover failed measurements with DeepTag, and visualize with viser."
        )
    )
    parser.add_argument(
        "pkl_path",
        nargs="?",
        default=str(DEFAULT_RECORDING_DIR),
        help="012 raw-frame pkl path, or a directory containing 012_rs_raw_frames_*.pkl.",
    )
    parser.add_argument("--intrinsics-yaml", type=Path, default=None)
    parser.add_argument("--cube-cfg", type=Path, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--fps", type=float, default=PLAYBACK_FPS)
    parser.add_argument("--max-width", type=int, default=960, help="Max width for GUI images.")
    parser.add_argument("--max-frames", type=int, default=0, help="Only process the first N frames; 0 means all.")
    parser.add_argument("--no-undistort", action="store_true", help="Do not undistort frames before detection.")
    parser.add_argument("--slow", action="store_true", help="Use slower high-accuracy AprilTag detector settings.")
    parser.add_argument("--no-filter", action="store_true", help="Disable AprilCube temporal pose filter.")
    parser.add_argument(
        "--fallback-layout",
        choices=("off", "cfg", "printed-pdf"),
        default="cfg",
        help="Fallback PnP layout when the detector rejects pose; cfg uses config.json tag geometry.",
    )
    parser.add_argument(
        "--fallback-max-reproj",
        type=float,
        default=5.0,
        help="Accept fallback PnP poses up to this mean reprojection error in pixels.",
    )
    parser.add_argument(
        "--fallback-ransac-reproj",
        type=float,
        default=3.0,
        help="RANSAC reprojection threshold in pixels for fallback PnP.",
    )
    parser.add_argument(
        "--deeptag",
        action="store_true",
        help="Load the 015 DeepTag backend and use conservative robust-cluster recovery.",
    )
    parser.add_argument(
        "--deeptag-mode",
        choices=("fallback", "validate"),
        default="fallback",
        help=(
            "fallback runs DeepTag only when AprilCube fails; validate runs it on every "
            "frame for diagnostics but still keeps successful AprilCube measurements."
        ),
    )
    parser.add_argument("--deeptag-cpu", action="store_true", help="Force DeepTag onto CPU.")
    parser.add_argument(
        "--deeptag-verbose",
        action="store_true",
        help="Keep DeepTag per-frame stdout instead of suppressing it.",
    )
    parser.add_argument(
        "--deeptag-detect-scale",
        type=float,
        default=-1.0,
        help="DeepTag detection scale; negative uses its default.",
    )
    parser.add_argument("--deeptag-min-tags", type=int, default=2)
    parser.add_argument("--deeptag-max-reproj", type=float, default=6.0)
    parser.add_argument("--deeptag-single-tag-max-reproj", type=float, default=4.0)
    parser.add_argument("--deeptag-cluster-trans-mm", type=float, default=70.0)
    parser.add_argument("--deeptag-cluster-rot-deg", type=float, default=55.0)
    parser.add_argument(
        "--no-fill-missing-pose",
        action="store_true",
        help="Do not fill failed frames with the nearest previous successful pose.",
    )
    parser.add_argument("--precompute-only", action="store_true", help="Run offline detection and exit without viser.")
    parser.add_argument(
        "--output-pkl",
        type=Path,
        default=None,
        help="Write offline pose results and visualization JPEGs to this pkl, then exit unless --show-viser is set.",
    )
    parser.add_argument(
        "--show-viser",
        action="store_true",
        help="When --output-pkl is used, also start the viser replay UI after writing.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality for saved visualization images.")
    parser.add_argument(
        "--save-raw-jpeg",
        action="store_true",
        help="Also store raw input frames as JPEG bytes in the output pkl.",
    )
    return parser.parse_args()


def resolve_pkl_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(path.glob("012_rs_raw_frames_*.pkl"))
        if not candidates:
            raise FileNotFoundError(f"No 012_rs_raw_frames_*.pkl found in directory: {path}")
        return candidates[-1]
    raise FileNotFoundError(f"Invalid pkl path: {path}")


def build_stream_index(path: Path) -> tuple[dict[str, Any], list[int], dict[str, Any] | None]:
    frame_offsets: list[int] = []
    footer: dict[str, Any] | None = None
    supported_formats = {
        "aprilcube_rs_raw_frame_stream_v1",
        "aprilcube_012_raw_with_pose_stream_v1",
    }
    with path.open("rb") as f:
        header = pickle.load(f)
        if not isinstance(header, dict) or header.get("format") not in supported_formats:
            raise ValueError(f"Unsupported pkl format in {path}")

        while True:
            offset = f.tell()
            try:
                obj = pickle.load(f)
            except EOFError:
                break
            if not isinstance(obj, dict):
                continue
            obj_type = obj.get("type")
            if obj_type == "frame":
                frame_offsets.append(offset)
            elif obj_type == "footer":
                footer = obj
                break

    if not frame_offsets:
        raise ValueError(f"No frame records found in {path}")
    return header, frame_offsets, footer


def load_frame_at(path: Path, offset: int) -> dict[str, Any]:
    with path.open("rb") as f:
        f.seek(int(offset))
        record = pickle.load(f)
    if not isinstance(record, dict) or record.get("type") != "frame":
        raise ValueError(f"Offset {offset} in {path} is not a frame record")
    return record


def resize_bgr_if_needed(frame: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = image_size
    h, w = frame.shape[:2]
    if (w, h) == (target_w, target_h):
        return frame
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)


def scale_for_gui(image_rgb: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return image_rgb
    h, w = image_rgb.shape[:2]
    if w <= max_width:
        return image_rgb
    scale = float(max_width) / float(w)
    out_h = max(1, int(round(h * scale)))
    return cv2.resize(image_rgb, (max_width, out_h), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(image_bgr: np.ndarray, max_width: int = 0) -> np.ndarray:
    image_bgr = np.asarray(image_bgr, dtype=np.uint8)
    if image_bgr.ndim == 2:
        image_rgb = np.repeat(image_bgr[:, :, None], 3, axis=2)
    else:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return scale_for_gui(image_rgb, max_width)


def rotation_matrix_to_wxyz(rot: np.ndarray) -> tuple[float, float, float, float]:
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    return tuple(float(v) for v in quat)


def rvec_to_wxyz(rvec: Any) -> tuple[float, float, float, float]:
    rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return rotation_matrix_to_wxyz(rot)


def wxyz_to_rvec(wxyz: Any) -> np.ndarray:
    w, x, y, z = np.asarray(wxyz, dtype=np.float64).reshape(4)
    n = max(float(np.linalg.norm([w, x, y, z])), 1e-12)
    w, x, y, z = w / n, x / n, y / n, z / n
    rot = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    rvec, _ = cv2.Rodrigues(rot)
    return np.asarray(rvec, dtype=np.float64).reshape(3, 1)


def slerp_wxyz(q0: Any, q1: Any, alpha: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64).reshape(4)
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    q0 /= max(float(np.linalg.norm(q0)), 1e-12)
    q1 /= max(float(np.linalg.norm(q1)), 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        out = q0 + float(alpha) * (q1 - q0)
        out /= max(float(np.linalg.norm(out)), 1e-12)
        return out
    theta_0 = float(np.arccos(dot))
    theta = theta_0 * float(alpha)
    sin_theta = float(np.sin(theta))
    sin_theta_0 = float(np.sin(theta_0))
    s0 = float(np.cos(theta)) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


def ndarray_to_list(value: Any) -> Any:
    if value is None:
        return None
    return np.asarray(value).tolist()


def scalar_or_none(value: Any) -> float | int | bool | str | None:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def sanitize_result(result: dict[str, Any]) -> dict[str, Any]:
    detections = []
    for item in result.get("detections", []) or []:
        if len(item) != 2:
            continue
        tag_id, corners = item
        detections.append(
            {
                "tag_id": int(tag_id),
                "corners_xy": ndarray_to_list(np.asarray(corners, dtype=np.float64).reshape(4, 2)),
            }
        )

    per_tag = result.get("per_tag_reproj_error", {})
    if isinstance(per_tag, dict):
        per_tag_reproj_error = {int(k): float(v) for k, v in per_tag.items()}
    else:
        per_tag_reproj_error = {}

    return {
        "success": bool(result.get("success", False)),
        "failure_reason": str(result.get("failure_reason", "")),
        "n_tags": int(result.get("n_tags", 0)),
        "n_inliers": int(result.get("n_inliers", 0)),
        "reproj_error": float(result.get("reproj_error", float("inf"))),
        "tag_ids": [int(v) for v in result.get("tag_ids", [])],
        "visible_faces": sorted(str(v) for v in result.get("visible_faces", set())),
        "predicted": bool(result.get("predicted", False)),
        "pose_source": str(result.get("pose_source", "aprilcube_detector")),
        "pose_filled": bool(result.get("pose_filled", False)),
        "fill_original_failure_reason": str(result.get("fill_original_failure_reason", "")),
        "fallback_original_failure_reason": str(result.get("fallback_original_failure_reason", "")),
        "fallback_layout": str(result.get("fallback_layout", "")),
        "single_tag_cfg_pose": bool(result.get("single_tag_cfg_pose", False)),
        "single_tag_id": scalar_or_none(result.get("single_tag_id", None)),
        "single_tag_face": scalar_or_none(result.get("single_tag_face", None)),
        "rvec": ndarray_to_list(result.get("rvec", None)),
        "tvec": ndarray_to_list(result.get("tvec", None)),
        "T": ndarray_to_list(result.get("T", None)),
        "detections": detections,
        "per_tag_reproj_error": per_tag_reproj_error,
        "fallback_outlier_rejected_ids": [
            int(v) for v in result.get("fallback_outlier_rejected_ids", []) or []
        ],
        "deeptag_attempted": bool(result.get("deeptag_attempted", False)),
        "deeptag_selected": bool(result.get("deeptag_selected", False)),
        "deeptag_success": bool(result.get("deeptag_success", False)),
        "deeptag_failure_reason": str(result.get("deeptag_failure_reason", "")),
        "deeptag_corner_order": str(result.get("deeptag_corner_order", "")),
        "deeptag_n_tags": int(result.get("deeptag_n_tags", 0) or 0),
        "deeptag_reproj_error": float(result.get("deeptag_reproj_error", float("inf"))),
        "deeptag_elapsed_s": float(result.get("deeptag_elapsed_s", 0.0) or 0.0),
        "deeptag_detection_stats": dict(result.get("deeptag_detection_stats", {}) or {}),
        "deeptag_cluster_stats": dict(result.get("deeptag_cluster_stats", {}) or {}),
        "deeptag_translation_delta_mm": scalar_or_none(
            result.get("deeptag_translation_delta_mm", None)
        ),
        "deeptag_rotation_delta_deg": scalar_or_none(
            result.get("deeptag_rotation_delta_deg", None)
        ),
        "aprilcube_pose_source": str(result.get("aprilcube_pose_source", "")),
        "aprilcube_failure_reason": str(result.get("aprilcube_failure_reason", "")),
    }


def encode_bgr_jpeg(image_bgr: np.ndarray, quality: int) -> bytes:
    quality = int(max(1, min(int(quality), 100)))
    ok, encoded = cv2.imencode(
        ".jpg",
        np.asarray(image_bgr, dtype=np.uint8),
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg) failed")
    return encoded.tobytes()


def result_to_markdown(record: dict[str, Any], result: dict[str, Any], slider_idx: int) -> str:
    lines = [
        f"frame_index: `{slider_idx}`",
        f"loop_frame_idx: `{record.get('loop_frame_idx', '?')}`",
        f"camera: `{record.get('camera_name', record.get('device_name', '?'))}`",
        f"timestamp: `{record.get('capture_timestamp', None)}`",
    ]
    if result.get("deeptag_attempted", False):
        lines.extend(
            [
                f"deeptag_success: `{result.get('deeptag_success', False)}`",
                f"deeptag_selected: `{result.get('deeptag_selected', False)}`",
                f"deeptag_tags/reproj: `{result.get('deeptag_n_tags', 0)} / "
                f"{result.get('deeptag_reproj_error', None)}`",
            ]
        )
    if not result.get("success", False):
        tag_ids = result.get("tag_ids", [])
        lines.append(f"pose: `not detected`, tags=`{int(result.get('n_tags', 0))}`")
        lines.append(f"failure_reason: `{result.get('failure_reason', '')}`")
        lines.append(f"tag_ids: `{tag_ids}`")
        return "\n".join(lines)

    tvec = np.asarray(result["tvec"], dtype=np.float64).reshape(3)
    faces = sorted(list(result.get("visible_faces", set())))
    lines.extend(
        [
            "pose: `detected`",
            f"pose_source: `{result.get('pose_source', '')}`",
            f"t_mm: `({tvec[0]:.1f}, {tvec[1]:.1f}, {tvec[2]:.1f})`",
            f"reproj_px: `{float(result.get('reproj_error', float('nan'))):.3f}`",
            f"tags: `{int(result.get('n_tags', 0))}`",
            f"faces: `{faces}`",
        ]
    )
    if result.get("predicted", False):
        lines.append("predicted: `true`")
    if result.get("single_tag_cfg_pose", False):
        lines.append(
            f"single_tag_cfg_pose: `id={result.get('single_tag_id', '?')}, "
            f"face={result.get('single_tag_face', '?')}`"
        )
    return "\n".join(lines)


class OfflineEstimator:
    def __init__(self, script012: Any, metadata: dict[str, Any], args: argparse.Namespace) -> None:
        self.script012 = script012
        self.metadata = metadata
        self.fallback_layout = str(args.fallback_layout)
        self.fallback_max_reproj = float(args.fallback_max_reproj)
        self.fallback_ransac_reproj = float(args.fallback_ransac_reproj)
        self.intrinsics_yaml = Path(
            args.intrinsics_yaml or metadata.get("intrinsics_yaml") or script012.DEFAULT_INTRINSICS_YAML
        ).expanduser().resolve()
        self.cube_cfg = Path(
            args.cube_cfg or metadata.get("cube_cfg") or script012.DEFAULT_CUBE_CFG
        ).expanduser().resolve()

        calib = script012.load_intrinsics_yaml(self.intrinsics_yaml)
        self.image_size = tuple(int(v) for v in calib["image_size"])
        self.raw_camera_matrix = np.asarray(calib["K"], dtype=np.float64).reshape(3, 3)
        self.raw_dist_coeffs = np.asarray(calib["dist"], dtype=np.float64).reshape(-1)
        if args.intrinsics_yaml is None:
            if metadata.get("image_size", None) is not None:
                self.image_size = tuple(int(v) for v in metadata["image_size"])
            if metadata.get("raw_camera_matrix", None) is not None:
                self.raw_camera_matrix = np.asarray(
                    metadata["raw_camera_matrix"], dtype=np.float64
                ).reshape(3, 3)
            if metadata.get("raw_dist_coeffs", None) is not None:
                self.raw_dist_coeffs = np.asarray(
                    metadata["raw_dist_coeffs"], dtype=np.float64
                ).reshape(-1)
        self.undistort_pack = None
        self.detection_camera_matrix = self.raw_camera_matrix.copy()
        self.detector_dist_coeffs = self.raw_dist_coeffs

        should_undistort = bool(metadata.get("undistort_for_detection", True)) and not bool(args.no_undistort)
        if should_undistort:
            self.undistort_pack = script012.create_undistort_maps(calib, self.image_size)
            if self.undistort_pack is not None:
                self.detection_camera_matrix = self.undistort_pack[2]
                self.detector_dist_coeffs = np.zeros(5, dtype=np.float64)
            if args.intrinsics_yaml is None and metadata.get("detection_camera_matrix", None) is not None:
                self.detection_camera_matrix = np.asarray(
                    metadata["detection_camera_matrix"], dtype=np.float64
                ).reshape(3, 3)
            if args.intrinsics_yaml is None and metadata.get("detector_dist_coeffs", None) is not None:
                self.detector_dist_coeffs = np.asarray(
                    metadata["detector_dist_coeffs"], dtype=np.float64
                ).reshape(-1)

        self.detector = aprilcube.detector(
            self.cube_cfg,
            intrinsic_cfg=script012.camera_matrix_to_intrinsic_dict(self.detection_camera_matrix),
            dist_coeffs=self.detector_dist_coeffs,
            enable_filter=not bool(args.no_filter),
            fast=not bool(args.slow),
        )
        self.fallback_tag_corner_map, self.fallback_face_id_sets = self._build_fallback_geometry()

    @property
    def cube_name(self) -> str:
        return self.cube_cfg.name if self.cube_cfg.is_dir() else self.cube_cfg.parent.name

    def _build_fallback_geometry(self) -> tuple[dict[int, np.ndarray], dict[str, set[int]]]:
        if self.fallback_layout == "off":
            return {}, {}
        cfg_path = self.cube_cfg / "config.json" if self.cube_cfg.is_dir() else self.cube_cfg
        config, face_id_sets = aprilcube.load_cube_config(str(cfg_path))
        if self.fallback_layout == "printed-pdf":
            all_ids = sorted({int(tag_id) for ids in face_id_sets.values() for tag_id in ids})
            tag_ids: list[int] = []
            new_face_sets: dict[str, set[int]] = {}
            cursor = 0
            for face_def in aprilcube.FACE_DEFS:
                face_name = str(face_def[0])
                face_rows, face_cols, _down_cells, _right_cells = config.face_layout(face_def)
                count = int(face_rows * face_cols)
                face_ids = all_ids[cursor : cursor + count]
                tag_ids.extend(face_ids)
                new_face_sets[face_name] = set(face_ids)
                cursor += count
            config.tag_ids = tag_ids
            config.tag_pattern_mirrored = False
            config.compute()
            face_id_sets = new_face_sets
        return aprilcube.build_tag_corner_map(config), face_id_sets

    def detection_frame(self, image_bgr: np.ndarray) -> np.ndarray:
        color = resize_bgr_if_needed(image_bgr, self.image_size)
        return self.script012.undistort_frame(color, self.undistort_pack)

    def _visible_faces_for_ids(self, tag_ids: list[int]) -> set[str]:
        visible: set[str] = set()
        for tag_id in tag_ids:
            for face_name, face_ids in self.fallback_face_id_sets.items():
                if int(tag_id) in face_ids:
                    visible.add(str(face_name))
        return visible

    def _face_normals_ok(self, rvec: np.ndarray, visible_faces: set[str]) -> bool:
        rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        for face_name in visible_faces:
            for face_def in aprilcube.FACE_DEFS:
                if str(face_def[0]) != str(face_name):
                    continue
                normal = np.zeros(3, dtype=np.float64)
                normal[int(face_def[1])] = float(face_def[2])
                normal_cam = rot @ normal
                if float(normal_cam[2]) > 0.0:
                    return False
                break
        return True

    def _fallback_points_from_detections(
        self,
        detections: list[tuple[int, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        object_chunks: list[np.ndarray] = []
        image_chunks: list[np.ndarray] = []
        used_ids: list[int] = []
        for tag_id_raw, corners_raw in detections:
            tag_id = int(tag_id_raw)
            corners_3d = self.fallback_tag_corner_map.get(tag_id)
            if corners_3d is None:
                continue
            corners_2d = np.asarray(corners_raw, dtype=np.float64).reshape(4, 2)
            object_chunks.append(np.asarray(corners_3d, dtype=np.float64).reshape(4, 3))
            image_chunks.append(corners_2d)
            used_ids.append(tag_id)
        if not object_chunks:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 2), dtype=np.float64),
                [],
            )
        return (
            np.vstack(object_chunks).astype(np.float64),
            np.vstack(image_chunks).astype(np.float64),
            used_ids,
        )

    def _solve_fallback_global_pnp(
        self,
        detections: list[tuple[int, np.ndarray]],
    ) -> dict[str, Any] | None:
        object_points, image_points, used_ids = self._fallback_points_from_detections(detections)
        if object_points.shape[0] < 4:
            return None

        inliers = None
        if object_points.shape[0] >= 6:
            try:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objectPoints=object_points,
                    imagePoints=image_points,
                    cameraMatrix=self.detection_camera_matrix,
                    distCoeffs=self.detector_dist_coeffs,
                    iterationsCount=200,
                    reprojectionError=float(self.fallback_ransac_reproj),
                    confidence=0.99,
                    flags=cv2.SOLVEPNP_SQPNP,
                )
            except cv2.error:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objectPoints=object_points,
                    imagePoints=image_points,
                    cameraMatrix=self.detection_camera_matrix,
                    distCoeffs=self.detector_dist_coeffs,
                    iterationsCount=200,
                    reprojectionError=float(self.fallback_ransac_reproj),
                    confidence=0.99,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
        else:
            try:
                success, rvec, tvec = cv2.solvePnP(
                    objectPoints=object_points,
                    imagePoints=image_points,
                    cameraMatrix=self.detection_camera_matrix,
                    distCoeffs=self.detector_dist_coeffs,
                    flags=cv2.SOLVEPNP_SQPNP,
                )
            except cv2.error:
                success, rvec, tvec = cv2.solvePnP(
                    objectPoints=object_points,
                    imagePoints=image_points,
                    cameraMatrix=self.detection_camera_matrix,
                    distCoeffs=self.detector_dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            inliers = np.arange(object_points.shape[0], dtype=np.int32).reshape(-1, 1) if success else None

        if not success or rvec is None or tvec is None:
            return None
        if float(np.asarray(tvec).reshape(3)[2]) <= 0.0:
            return None

        if inliers is not None and len(inliers) >= 4:
            idx = np.asarray(inliers, dtype=np.int32).reshape(-1)
            try:
                rvec, tvec = cv2.solvePnPRefineLM(
                    objectPoints=object_points[idx],
                    imagePoints=image_points[idx],
                    cameraMatrix=self.detection_camera_matrix,
                    distCoeffs=self.detector_dist_coeffs,
                    rvec=rvec,
                    tvec=tvec,
                )
            except cv2.error:
                pass

        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.detection_camera_matrix,
            self.detector_dist_coeffs,
        )
        projected = projected.reshape(-1, 2)
        corner_errors = np.linalg.norm(image_points - projected, axis=1)
        per_tag_reproj_error: dict[int, float] = {}
        for idx, tag_id in enumerate(used_ids):
            start = idx * 4
            end = start + 4
            per_tag_reproj_error[int(tag_id)] = float(np.mean(corner_errors[start:end]))

        visible_faces = self._visible_faces_for_ids(used_ids)
        if not self._face_normals_ok(np.asarray(rvec, dtype=np.float64).reshape(3, 1), visible_faces):
            return None

        return {
            "rvec": np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            "tvec": np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            "reproj_error": float(np.mean(corner_errors)),
            "n_inliers": 0 if inliers is None else int(len(inliers)),
            "used_ids": used_ids,
            "visible_faces": visible_faces,
            "per_tag_reproj_error": per_tag_reproj_error,
        }

    def _fallback_pose(self, result: dict[str, Any]) -> dict[str, Any] | None:
        if not self.fallback_tag_corner_map:
            return None
        detections = [
            (int(tag_id), np.asarray(corners, dtype=np.float64).reshape(4, 2))
            for tag_id, corners in (result.get("detections", []) or [])
            if int(tag_id) in self.fallback_tag_corner_map
        ]
        if not detections:
            return None

        solved = self._solve_fallback_global_pnp(detections)
        if solved is None:
            return None
        outlier_rejected_ids: list[int] = []
        if len(detections) >= 3:
            per_tag = solved["per_tag_reproj_error"]
            per_tag_values = np.asarray([per_tag[int(tag_id)] for tag_id, _ in detections], dtype=np.float64)
            median_err = float(np.median(per_tag_values))
            tag_reproj_thresh = max(median_err * 3.0, 2.0)
            keep = [
                idx
                for idx, (tag_id, _corners) in enumerate(detections)
                if float(per_tag[int(tag_id)]) <= tag_reproj_thresh
            ]
            if len(keep) < len(detections) and len(keep) >= 1:
                outlier_rejected_ids = [
                    int(tag_id)
                    for idx, (tag_id, _corners) in enumerate(detections)
                    if idx not in keep
                ]
                detections = [detections[idx] for idx in keep]
                solved = self._solve_fallback_global_pnp(detections)
                if solved is None:
                    return None

        reproj_error = float(solved["reproj_error"])
        if reproj_error > self.fallback_max_reproj:
            return None

        rvec = np.asarray(solved["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(solved["tvec"], dtype=np.float64).reshape(3, 1)
        used_ids = [int(v) for v in solved["used_ids"]]

        rot, _ = cv2.Rodrigues(rvec)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rot
        transform[:3, 3] = tvec.reshape(3)

        fallback = dict(result)
        fallback.update(
            {
                "success": True,
                "failure_reason": "",
                "pose_source": f"fallback_pnp_{self.fallback_layout}_aprilcube_style",
                "rvec": rvec,
                "tvec": tvec,
                "T": transform,
                "reproj_error": reproj_error,
                "n_inliers": int(solved["n_inliers"]),
                "n_tags": len(used_ids),
                "visible_faces": solved["visible_faces"],
                "tag_ids": used_ids,
                "detections": detections,
                "per_tag_reproj_error": solved["per_tag_reproj_error"],
                "fallback_outlier_rejected_ids": outlier_rejected_ids,
                "fallback_original_failure_reason": str(result.get("failure_reason", "")),
                "fallback_layout": self.fallback_layout,
            }
        )
        return fallback

    def process_record(self, record: dict[str, Any]) -> dict[str, Any]:
        image_bgr = np.asarray(record["image_bgr"], dtype=np.uint8)
        detect_frame = self.detection_frame(image_bgr)
        timestamp = record.get("capture_timestamp", None)
        result = self.detector.process_frame(
            detect_frame,
            timestamp=None if timestamp is None else float(timestamp),
        )
        detector_success = bool(result.get("success", False))
        detector_reproj = float(result.get("reproj_error", float("inf")))
        detector_n_tags = int(result.get("n_tags", 0) or 0)
        detector_usable = (
            detector_success
            and detector_n_tags > 0
            and np.isfinite(detector_reproj)
            and detector_reproj <= self.fallback_max_reproj
        )
        if detector_usable:
            result = dict(result)
            result["pose_source"] = "aprilcube_detector"
        else:
            rejected_reason = (
                ""
                if not detector_success
                else "detector_no_tags"
                if detector_n_tags <= 0
                else f"detector_reproj_rejected:{detector_reproj:.2f}>{self.fallback_max_reproj:.2f}"
            )
            fallback_seed = dict(result)
            if rejected_reason:
                fallback_seed["failure_reason"] = rejected_reason
            fallback = self._fallback_pose(fallback_seed)
            if fallback is not None:
                result = fallback
            else:
                result = fallback_seed
                result["success"] = False
        return {
            "success": bool(result.get("success", False)),
            "n_tags": int(result.get("n_tags", 0)),
            "result": result,
        }

    def overlay_image(self, record: dict[str, Any], result: dict[str, Any]) -> np.ndarray:
        detect_frame = self.detection_frame(np.asarray(record["image_bgr"], dtype=np.uint8))
        vis = self.script012.make_detector_input_vis(detect_frame)
        return self.detector.draw_result(vis, result)


class DeepTagPoseBackend:
    """Conservative adapter around the standalone 015 DeepTag implementation."""

    def __init__(
        self,
        script012: Any,
        metadata: dict[str, Any],
        args: argparse.Namespace,
    ) -> None:
        if int(args.deeptag_min_tags) < 2:
            raise ValueError("--deeptag-min-tags must be at least 2 for conservative recovery.")
        if float(args.deeptag_max_reproj) <= 0.0:
            raise ValueError("--deeptag-max-reproj must be positive.")

        self.script012 = script012
        self.module = load_script015_module()
        self.args = argparse.Namespace(
            intrinsics_yaml=args.intrinsics_yaml,
            cube_cfg=args.cube_cfg,
            no_undistort=bool(args.no_undistort),
            cpu=bool(args.deeptag_cpu),
            detect_scale=float(args.deeptag_detect_scale),
            min_center_score=0.2,
            min_corner_score=0.2,
            hamming_dist=8,
            stg2_iter_num=2,
            batch_size_stg2=4,
            corner_order="rot180",
            pose_mode="robust-cluster",
            robust_min_tags=int(args.deeptag_min_tags),
            robust_cluster_trans_mm=float(args.deeptag_cluster_trans_mm),
            robust_cluster_rot_deg=float(args.deeptag_cluster_rot_deg),
            robust_max_reproj=float(args.deeptag_max_reproj),
            robust_single_tag_max_reproj=float(args.deeptag_single_tag_max_reproj),
        )
        self.quiet = not bool(args.deeptag_verbose)
        self.runtime = self.module.make_runtime(script012, metadata, self.args)
        tag_size_m = float(self.runtime["cube_config"].tag_size_mm) / 1000.0
        print(f"[INFO] Loading integrated DeepTag backend from {self.module.DEEPTAG_ROOT}")
        started = time.perf_counter()
        self.engine = self.module.load_deeptag_engine(
            camera_matrix=self.runtime["detection_camera_matrix"],
            dist_coeffs=self.runtime["detector_dist_coeffs"],
            tag_size_m=tag_size_m,
            args=self.args,
        )
        print(f"[INFO] Integrated DeepTag loaded in {time.perf_counter() - started:.2f}s")
        self.metadata = {
            "enabled": True,
            "backend_script": str(SCRIPT_015_PATH),
            "mode": str(args.deeptag_mode),
            "cpu": bool(args.deeptag_cpu),
            "detect_scale": float(args.deeptag_detect_scale),
            "min_tags": int(args.deeptag_min_tags),
            "max_reproj": float(args.deeptag_max_reproj),
            "single_tag_max_reproj": float(args.deeptag_single_tag_max_reproj),
            "cluster_trans_mm": float(args.deeptag_cluster_trans_mm),
            "cluster_rot_deg": float(args.deeptag_cluster_rot_deg),
        }

    def process_record(self, record: dict[str, Any]) -> dict[str, Any]:
        frame = self.module.detection_frame(
            self.script012,
            self.runtime,
            np.asarray(record["image_bgr"], dtype=np.uint8),
        )
        started = time.perf_counter()
        stream = io.StringIO()
        output_context = contextlib.redirect_stdout(stream) if self.quiet else contextlib.nullcontext()
        try:
            with output_context:
                decoded_tags = self.engine.process(
                    frame,
                    detect_scale=(
                        None
                        if float(self.args.detect_scale) < 0.0
                        else float(self.args.detect_scale)
                    ),
                )
            raw_detections, detection_stats = self.module.deeptag_detections_to_raw_corners(
                self.engine,
                decoded_tags,
                valid_ids=set(int(v) for v in self.runtime["cube_config"].tag_ids),
            )
            pose, _detections, cluster_stats = self.module.robust_cluster_pose(
                raw_detections,
                self.runtime,
                self.args,
            )
            pose = dict(pose)
            if not self.module.finite_pose_success(pose):
                pose["success"] = False
                pose["rvec"] = None
                pose["tvec"] = None
                pose["T"] = None
                pose["reproj_error"] = float("inf")
                if not pose.get("failure_reason", ""):
                    pose["failure_reason"] = "non_finite_or_failed_deeptag_pose"
        except Exception as exc:
            detection_stats = {}
            cluster_stats = {}
            pose = {
                "success": False,
                "failure_reason": f"deeptag_exception:{type(exc).__name__}:{exc}",
                "rvec": None,
                "tvec": None,
                "T": None,
                "reproj_error": float("inf"),
                "n_tags": 0,
                "tag_ids": [],
                "detections": [],
            }
        elapsed = time.perf_counter() - started
        pose["deeptag_elapsed_s"] = float(elapsed)
        pose["deeptag_detection_stats"] = self.module._jsonish(detection_stats)
        pose["deeptag_cluster_stats"] = self.module._jsonish(cluster_stats)
        return {
            "success": bool(pose.get("success", False)),
            "n_tags": int(pose.get("n_tags", 0) or 0),
            "result": pose,
        }


def pose_delta(
    first: dict[str, Any],
    second: dict[str, Any],
) -> tuple[float, float]:
    first_t = np.asarray(first["tvec"], dtype=np.float64).reshape(3)
    second_t = np.asarray(second["tvec"], dtype=np.float64).reshape(3)
    translation_mm = float(np.linalg.norm(first_t - second_t))
    first_rot, _ = cv2.Rodrigues(np.asarray(first["rvec"], dtype=np.float64).reshape(3, 1))
    second_rot, _ = cv2.Rodrigues(np.asarray(second["rvec"], dtype=np.float64).reshape(3, 1))
    cosine = float(np.clip((np.trace(first_rot.T @ second_rot) - 1.0) / 2.0, -1.0, 1.0))
    rotation_deg = float(np.degrees(np.arccos(cosine)))
    return translation_mm, rotation_deg


def combine_aprilcube_and_deeptag(
    april_item: dict[str, Any],
    deeptag_item: dict[str, Any],
) -> dict[str, Any]:
    april_result = dict(april_item["result"])
    deeptag_result = dict(deeptag_item["result"])
    april_success = bool(april_result.get("success", False))
    deeptag_success = bool(deeptag_result.get("success", False))

    diagnostics = {
        "deeptag_attempted": True,
        "deeptag_selected": bool(deeptag_success and not april_success),
        "deeptag_success": deeptag_success,
        "deeptag_failure_reason": str(deeptag_result.get("failure_reason", "")),
        "deeptag_n_tags": int(deeptag_result.get("n_tags", 0) or 0),
        "deeptag_reproj_error": float(
            deeptag_result.get("reproj_error", float("inf"))
        ),
        "deeptag_elapsed_s": float(deeptag_result.get("deeptag_elapsed_s", 0.0)),
        "deeptag_detection_stats": dict(
            deeptag_result.get("deeptag_detection_stats", {}) or {}
        ),
        "deeptag_cluster_stats": dict(
            deeptag_result.get("deeptag_cluster_stats", {}) or {}
        ),
        "aprilcube_pose_source": str(april_result.get("pose_source", "")),
        "aprilcube_failure_reason": str(april_result.get("failure_reason", "")),
    }
    if april_success and deeptag_success:
        translation_mm, rotation_deg = pose_delta(april_result, deeptag_result)
        diagnostics["deeptag_translation_delta_mm"] = translation_mm
        diagnostics["deeptag_rotation_delta_deg"] = rotation_deg

    selected = deeptag_result if deeptag_success and not april_success else april_result
    selected = dict(selected)
    selected.update(diagnostics)
    return {
        "success": bool(selected.get("success", False)),
        "n_tags": int(selected.get("n_tags", 0) or 0),
        "result": selected,
    }


def precompute_pose_cache(
    pkl_path: Path,
    offsets: list[int],
    estimator: OfflineEstimator,
    deeptag_backend: DeepTagPoseBackend | None = None,
    deeptag_mode: str = "fallback",
) -> list[dict[str, Any]]:
    cache: list[dict[str, Any]] = []
    total = len(offsets)
    t0 = time.perf_counter()
    for idx, offset in enumerate(offsets):
        record = load_frame_at(pkl_path, offset)
        pose = estimator.process_record(record)
        should_run_deeptag = (
            deeptag_backend is not None
            and (deeptag_mode == "validate" or not bool(pose.get("success", False)))
        )
        if should_run_deeptag:
            deeptag_pose = deeptag_backend.process_record(record)
            pose = combine_aprilcube_and_deeptag(pose, deeptag_pose)
        cache.append(pose)
        done = idx + 1
        if done == total or done % 10 == 0:
            elapsed = time.perf_counter() - t0
            fps = done / max(elapsed, 1e-9)
            print(
                f"\r[INFO] Offline pose detection {done}/{total} "
                f"success={sum(int(v['success']) for v in cache)} "
                f"deeptag_selected={sum(int(v['result'].get('deeptag_selected', False)) for v in cache)} "
                f"fps={fps:.1f}",
                end="",
                flush=True,
            )
    print()
    return cache


def fill_missing_pose_cache(pose_cache: list[dict[str, Any]]) -> int:
    good_indices = [
        idx
        for idx, item in enumerate(pose_cache)
        if item["result"].get("success", False)
        and item["result"].get("rvec", None) is not None
        and item["result"].get("tvec", None) is not None
    ]
    if not good_indices:
        return 0

    def make_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rot
        transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
        return transform

    def fill_one(idx: int, source: str, rvec: np.ndarray, tvec: np.ndarray) -> None:
        result = pose_cache[idx]["result"]
        filled = dict(result)
        filled.update(
            {
                "success": True,
                "pose_source": source,
                "pose_filled": True,
                "fill_original_failure_reason": str(result.get("failure_reason", "")),
                "failure_reason": "",
                "rvec": np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                "tvec": np.asarray(tvec, dtype=np.float64).reshape(3, 1),
                "T": make_transform(rvec, tvec),
                "reproj_error": float("inf"),
                "n_inliers": 0,
            }
        )
        pose_cache[idx]["result"] = filled
        pose_cache[idx]["success"] = True

    filled_count = 0
    first_good = good_indices[0]
    first_result = pose_cache[first_good]["result"]
    for idx in range(0, first_good):
        fill_one(
            idx,
            "filled_next_pose",
            np.asarray(first_result["rvec"], dtype=np.float64).reshape(3, 1),
            np.asarray(first_result["tvec"], dtype=np.float64).reshape(3, 1),
        )
        filled_count += 1

    for left_idx, right_idx in zip(good_indices[:-1], good_indices[1:]):
        if right_idx <= left_idx + 1:
            continue
        left = pose_cache[left_idx]["result"]
        right = pose_cache[right_idx]["result"]
        left_t = np.asarray(left["tvec"], dtype=np.float64).reshape(3)
        right_t = np.asarray(right["tvec"], dtype=np.float64).reshape(3)
        left_q = np.asarray(rvec_to_wxyz(left["rvec"]), dtype=np.float64)
        right_q = np.asarray(rvec_to_wxyz(right["rvec"]), dtype=np.float64)
        gap = right_idx - left_idx
        for idx in range(left_idx + 1, right_idx):
            alpha = float(idx - left_idx) / float(gap)
            tvec = ((1.0 - alpha) * left_t + alpha * right_t).reshape(3, 1)
            rvec = wxyz_to_rvec(slerp_wxyz(left_q, right_q, alpha))
            fill_one(idx, "filled_interpolated_pose", rvec, tvec)
            filled_count += 1

    last_good = good_indices[-1]
    last_result = pose_cache[last_good]["result"]
    for idx in range(last_good + 1, len(pose_cache)):
        fill_one(
            idx,
            "filled_previous_pose",
            np.asarray(last_result["rvec"], dtype=np.float64).reshape(3, 1),
            np.asarray(last_result["tvec"], dtype=np.float64).reshape(3, 1),
        )
        filled_count += 1
    return filled_count


def default_output_pkl_path(source_pkl: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return source_pkl.with_name(f"013_offline_pose_{source_pkl.stem}_{stamp}.pkl")


def write_processed_pkl(
    *,
    source_pkl: Path,
    output_pkl: Path,
    header: dict[str, Any],
    footer: dict[str, Any] | None,
    offsets: list[int],
    estimator: OfflineEstimator,
    pose_cache: list[dict[str, Any]],
    jpeg_quality: int,
    save_raw_jpeg: bool,
    deeptag_metadata: dict[str, Any] | None,
) -> None:
    output_pkl = output_pkl.expanduser().resolve()
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    total = len(offsets)
    t0 = time.perf_counter()
    success_count = sum(int(item["success"]) for item in pose_cache)
    with output_pkl.open("wb") as f:
        pickle.dump(
            {
                "type": "header",
                "format": "aprilcube_012_offline_pose_vis_stream_v1",
                "created_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_pkl": str(source_pkl),
                "source_format": header.get("format", ""),
                "source_metadata": header.get("metadata", {}),
                "source_footer": footer,
                "metadata": {
                    "script": str(THIS_FILE),
                    "intrinsics_yaml": str(estimator.intrinsics_yaml),
                    "cube_cfg": str(estimator.cube_cfg),
                    "image_size": tuple(int(v) for v in estimator.image_size),
                    "detection_camera_matrix": estimator.detection_camera_matrix.tolist(),
                    "detector_dist_coeffs": estimator.detector_dist_coeffs.tolist(),
                    "undistort_for_detection": estimator.undistort_pack is not None,
                    "jpeg_quality": int(jpeg_quality),
                    "contains_raw_jpeg": bool(save_raw_jpeg),
                    "fallback_layout": estimator.fallback_layout,
                    "fallback_max_reproj": float(estimator.fallback_max_reproj),
                    "fallback_ransac_reproj": float(estimator.fallback_ransac_reproj),
                    "deeptag": dict(deeptag_metadata or {"enabled": False}),
                    "deeptag_attempted_count": int(
                        sum(bool(item["result"].get("deeptag_attempted", False)) for item in pose_cache)
                    ),
                    "deeptag_selected_count": int(
                        sum(bool(item["result"].get("deeptag_selected", False)) for item in pose_cache)
                    ),
                    "fill_missing_pose": any(
                        bool(item["result"].get("pose_filled", False)) for item in pose_cache
                    ),
                    "filled_pose_count": int(
                        sum(bool(item["result"].get("pose_filled", False)) for item in pose_cache)
                    ),
                    "frame_count": int(total),
                    "success_count": int(success_count),
                },
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        for idx, offset in enumerate(offsets):
            record = load_frame_at(source_pkl, offset)
            result = pose_cache[idx]["result"]
            overlay_bgr = estimator.overlay_image(record, result)
            frame_record = {
                "type": "frame",
                "frame_index": int(idx),
                "source_offset": int(offset),
                "camera_name": str(record.get("camera_name", record.get("device_name", ""))),
                "device_name": str(record.get("device_name", "")),
                "loop_frame_idx": int(record.get("loop_frame_idx", idx)),
                "capture_timestamp": record.get("capture_timestamp", None),
                "source_shape": tuple(int(v) for v in np.asarray(record["image_bgr"]).shape),
                "overlay_shape": tuple(int(v) for v in overlay_bgr.shape),
                "overlay_format": "jpeg_bgr",
                "overlay_jpeg": encode_bgr_jpeg(overlay_bgr, jpeg_quality),
                "pose": sanitize_result(result),
            }
            if save_raw_jpeg:
                frame_record["raw_format"] = "jpeg_bgr"
                frame_record["raw_jpeg"] = encode_bgr_jpeg(record["image_bgr"], jpeg_quality)
            pickle.dump(frame_record, f, protocol=pickle.HIGHEST_PROTOCOL)

            done = idx + 1
            if done == total or done % 10 == 0:
                elapsed = time.perf_counter() - t0
                fps = done / max(elapsed, 1e-9)
                print(
                    f"\r[INFO] Writing processed pkl {done}/{total} "
                    f"success={success_count}/{total} fps={fps:.1f}",
                    end="",
                    flush=True,
                )

        pickle.dump(
            {
                "type": "footer",
                "frame_count": int(total),
                "success_count": int(success_count),
                "stopped_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print()
    print(f"[INFO] Saved processed pose visualization pkl: {output_pkl}")


def add_optional_cube_mesh(server: viser.ViserServer, cube_cfg: Path) -> None:
    cube_dir = cube_cfg if cube_cfg.is_dir() else cube_cfg.parent
    obj_path = cube_dir / "mujoco" / "cube.obj"
    if not obj_path.exists():
        return
    try:
        import trimesh

        mesh = trimesh.load(str(obj_path))
        server.scene.add_mesh_trimesh("/cube/mesh", mesh)
    except Exception as exc:
        print(f"[WARNING] Could not add cube mesh to viser: {type(exc).__name__}: {exc}")


def update_cube_handle(cube_handle: Any, result: dict[str, Any]) -> None:
    if not result.get("success", False):
        cube_handle.visible = False
        return
    cube_handle.visible = True
    cube_handle.position = tuple(float(v) for v in (np.asarray(result["tvec"], dtype=np.float64).reshape(3) / 1000.0))
    cube_handle.wxyz = rvec_to_wxyz(result["rvec"])


def main() -> None:
    args = parse_args()
    pkl_path = resolve_pkl_path(args.pkl_path)
    header, offsets, footer = build_stream_index(pkl_path)
    metadata = dict(header.get("metadata", {}))
    if header.get("format") == "aprilcube_012_raw_with_pose_stream_v1":
        metadata = dict(header.get("raw_header", {}).get("metadata", metadata))
    if args.max_frames > 0:
        offsets = offsets[: int(args.max_frames)]

    script012 = load_script012_module()
    estimator = OfflineEstimator(script012, metadata, args)
    deeptag_backend = (
        DeepTagPoseBackend(script012, metadata, args)
        if bool(args.deeptag)
        else None
    )
    pose_cache = precompute_pose_cache(
        pkl_path,
        offsets,
        estimator,
        deeptag_backend=deeptag_backend,
        deeptag_mode=str(args.deeptag_mode),
    )
    filled_count = 0
    if not args.no_fill_missing_pose:
        filled_count = fill_missing_pose_cache(pose_cache)
    success_count = sum(int(item["success"]) for item in pose_cache)

    print(f"[INFO] pkl={pkl_path}")
    print(f"[INFO] frames={len(offsets)} footer={footer}")
    print(f"[INFO] intrinsics_yaml={estimator.intrinsics_yaml}")
    print(f"[INFO] cube_cfg={estimator.cube_cfg}")
    print(
        f"[INFO] offline pose success={success_count}/{len(pose_cache)} "
        f"filled={filled_count} "
        f"deeptag_selected={sum(int(item['result'].get('deeptag_selected', False)) for item in pose_cache)}"
    )
    if args.output_pkl is not None:
        output_pkl = args.output_pkl
        if str(output_pkl) == "auto":
            output_pkl = default_output_pkl_path(pkl_path)
        write_processed_pkl(
            source_pkl=pkl_path,
            output_pkl=Path(output_pkl),
            header=header,
            footer=footer,
            offsets=offsets,
            estimator=estimator,
            pose_cache=pose_cache,
            jpeg_quality=int(args.jpeg_quality),
            save_raw_jpeg=bool(args.save_raw_jpeg),
            deeptag_metadata=(
                None if deeptag_backend is None else deeptag_backend.metadata
            ),
        )
        if not args.show_viser:
            return
    if args.precompute_only:
        return

    server = viser.ViserServer(host=args.host, port=int(args.port))
    server.scene.set_up_direction("-y")
    server.scene.world_axes.visible = False
    server.gui.set_panel_label("013 Offline AprilCube Pose")
    server.scene.add_frame(
        "/camera",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        axes_length=0.05,
        axes_radius=0.002,
        origin_radius=0.0,
    )
    cube_handle = server.scene.add_frame(
        "/cube",
        axes_length=0.04,
        axes_radius=0.0015,
        origin_radius=0.002,
        visible=False,
    )
    add_optional_cube_mesh(server, estimator.cube_cfg)

    frame_idx = 0
    is_playing = len(offsets) > 1
    loop_playback = True
    last_step_time = time.monotonic()

    with server.gui.add_folder("Replay Controls"):
        play_checkbox = server.gui.add_checkbox("Play", initial_value=is_playing)
        loop_checkbox = server.gui.add_checkbox("Loop", initial_value=loop_playback)
        frame_slider = server.gui.add_slider("Frame", min=0, max=len(offsets) - 1, step=1, initial_value=0)
        status_text = server.gui.add_text("Status", initial_value="", disabled=True)

    with server.gui.add_folder("Images"):
        raw_image_handle = server.gui.add_image(
            np.zeros((120, 160, 3), dtype=np.uint8),
            label="Raw BGR frame",
            format="jpeg",
            jpeg_quality=80,
        )
        overlay_image_handle = server.gui.add_image(
            np.zeros((120, 160, 3), dtype=np.uint8),
            label="Offline detection",
            format="jpeg",
            jpeg_quality=80,
        )

    pose_markdown = server.gui.add_markdown("")
    server.gui.add_markdown(
        "\n".join(
            [
                f"pkl: `{pkl_path}`",
                f"frames: `{len(offsets)}`",
                f"success: `{success_count}/{len(pose_cache)}`",
                f"intrinsics: `{estimator.intrinsics_yaml}`",
                f"cube_cfg: `{estimator.cube_cfg}`",
                f"undistort_for_detection: `{estimator.undistort_pack is not None}`",
            ]
        )
    )

    def clamp_index(value: int) -> int:
        return max(0, min(int(value), len(offsets) - 1))

    def render_frame(idx: int) -> None:
        record = load_frame_at(pkl_path, offsets[idx])
        result = pose_cache[idx]["result"]
        raw_image_handle.image = bgr_to_rgb(record["image_bgr"], args.max_width)
        overlay = estimator.overlay_image(record, result)
        overlay_image_handle.image = bgr_to_rgb(overlay, args.max_width)
        update_cube_handle(cube_handle, result)
        pose_markdown.content = result_to_markdown(record, result, idx)
        status_text.value = (
            f"{idx + 1}/{len(offsets)} "
            f"success={bool(result.get('success', False))} tags={int(result.get('n_tags', 0))}"
        )

    @play_checkbox.on_update
    def _on_play(_event: Any) -> None:
        nonlocal is_playing, last_step_time
        is_playing = bool(play_checkbox.value)
        last_step_time = time.monotonic()

    @loop_checkbox.on_update
    def _on_loop(_event: Any) -> None:
        nonlocal loop_playback
        loop_playback = bool(loop_checkbox.value)

    @frame_slider.on_update
    def _on_frame(_event: Any) -> None:
        nonlocal frame_idx, last_step_time
        frame_idx = clamp_index(int(frame_slider.value))
        last_step_time = time.monotonic()
        render_frame(frame_idx)

    render_frame(frame_idx)
    print(f"[INFO] Viser server started: http://{args.host}:{args.port}")

    while True:
        if is_playing and len(offsets) > 1:
            now = time.monotonic()
            if now - last_step_time >= 1.0 / max(float(args.fps), 1e-6):
                next_idx = frame_idx + 1
                if next_idx >= len(offsets):
                    if loop_playback:
                        next_idx = 0
                    else:
                        next_idx = len(offsets) - 1
                        is_playing = False
                        play_checkbox.value = False
                frame_idx = next_idx
                frame_slider.value = frame_idx
                render_frame(frame_idx)
                last_step_time = now
        time.sleep(0.005)


if __name__ == "__main__":
    main()
