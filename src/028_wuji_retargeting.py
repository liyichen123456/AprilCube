#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import trimesh
import viser
import yaml
import yourdfpy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from viser.extras import ViserUrdf


APRILCUBE_ROOT = Path(__file__).resolve().parent.parent
FINGEREYE_MESH_ROOT = (
    APRILCUBE_ROOT.parent / "xarm7_wuji_left_description/fingereye_mesh"
)
DEFAULT_PKL_PATH = (
    APRILCUBE_ROOT
    / "recordings/021_hand_back_sync_raw_frames_20260712_233831.pkl"
)
DEFAULT_URDF_PATH = Path(
    "/home/ps/project/ConSensV2Lab/thirdparty/wuji-description/"
    "hand/body-with-soft/urdf/left_simplified_w_fingereye.urdf"
)
DEFAULT_MIDDLE_EXTRINSICS = Path(
    "/home/ps/RobotCamCalib1/outputs/"
    "extrinsics_middle_finger_cam_cube_d435_charuco_multisession_joint_"
    "0713_012814_022818.yaml"
)
DEFAULT_THUMB_EXTRINSICS = Path(
    "/home/ps/RobotCamCalib1/outputs/"
    "extrinsics_wrist_Q_thumb_web_cam_middle_finger_cam_apriltag_grid_"
    "offline_2samples_0712_030212_0712_031300.yaml"
)
DEFAULT_OUTPUT_DIR = APRILCUBE_ROOT / "outputs/028_wuji_retargeting"

EXPECTED_PKL_FORMAT = "aprilcube_hand_back_software_synced_raw_v1"
WORLD_POSE_FIELD = "hand_back_cube_obj_poses"
WORLD_POSE_SCHEMA = "aprilcube.hand_back_cube_obj_poses.v1"
RETARGET_FIELD = "wuji_retargeting"
RETARGET_ALGORITHM = "028_wuji_contact_surface_global_translation_v13"

ROOT_LINK_NAME = "left_palm_link"
INITIAL_T_PALM_HAND_BACK = np.asarray(
    [
        [0.0, 0.0, -1.0, -0.071750],
        [0.0, -1.0, 0.0, 0.005444],
        [-1.0, 0.0, 0.0, 0.011613],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

VISER_HOST = "0.0.0.0"
VISER_PORT = 8104
AUTO_PLAY_FPS = 5.0
SURFACE_POINT_WEIGHT = 160.0
CONTACT_SURFACE_HALF_LINE_M = 0.004
CONTACT_SURFACE_INSET_M = 0.004
CONTACT_SURFACE_POINT_COUNT = 3
REST_WEIGHT = 0.08
THUMB_POSTURE_WEIGHT = 0.60
THUMB_DISTAL_CURVATURE_WEIGHT = 1.50
THUMB_DISTAL_BEND_DIFFERENCE_MAX_RAD = np.deg2rad(20.0)
SMOOTHNESS_WEIGHT = 0.2
INACTIVE_WEIGHT = 20.0
SOFT_LIMIT_MARGIN_RATIO = 0.08
SOFT_LIMIT_WEIGHT = 0.6
VELOCITY_LIMIT_SCALE = 1.0
HUMAN_TO_WUJI_SCALE_MIN = 1.00
HUMAN_TO_WUJI_SCALE_MAX = 1.40
HUMAN_TO_WUJI_SCALE_PRIOR_WEIGHT = 0.08
GLOBAL_TRANSLATION_DELTA_LIMIT_M = 0.050
GLOBAL_ROTATION_DELTA_LIMIT_RAD = np.deg2rad(30.0)
GLOBAL_TRANSLATION_PRIOR_SIGMA_M = 0.010
GLOBAL_ROTATION_PRIOR_SIGMA_RAD = np.deg2rad(10.0)
GLOBAL_OFFSET_PRIOR_WEIGHT = 0.25

THUMB_JOINT_NAMES = (
    "left_finger1_joint1",
    "left_finger1_joint2",
    "left_finger1_joint3",
    "left_finger1_joint4",
)
# This neutral arc comes from configs/robot/openarm_wuji.yaml and is used only
# as a retargeting prior. It does not alter the physical URDF joint limits.
THUMB_NATURAL_QPOS = np.asarray([0.826, 0.407, 0.550, 0.556], dtype=np.float32)
THUMB_RETARGET_LOWER = np.asarray([0.25, -0.12, 0.0, 0.0], dtype=np.float32)
THUMB_RETARGET_UPPER = np.asarray([1.25, 0.85, 1.45, 1.45], dtype=np.float32)
THUMB_CENTERLINE_LINK_NAMES = (
    "left_finger1_link1",
    "left_finger1_link2",
    "left_finger1_link3",
    "left_finger1_link4",
    "left_finger1_tip_link",
)


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    cube_name: str
    source_camera: str
    robot_link: str
    robot_tip_link: str
    mesh_path: Path
    color: tuple[int, int, int]
    contact_line_axis: tuple[float, float, float]


OBJECT_SPECS = (
    ObjectSpec(
        name="thumb",
        cube_name="cube_april_36h11_12_17_1x1x1_15mm",
        source_camera="thumb_web_cam",
        robot_link="left_finger1_link4",
        robot_tip_link="left_finger1_tip_link",
        mesh_path=FINGEREYE_MESH_ROOT / "thumb.obj",
        color=(255, 170, 55),
        contact_line_axis=(1.0, 0.0, 0.0),
    ),
    ObjectSpec(
        name="index",
        cube_name="cube_april_36h11_6_11_1x1x1_15mm",
        source_camera="thumb_web_cam",
        robot_link="left_finger2_link4",
        robot_tip_link="left_finger2_tip_link",
        mesh_path=FINGEREYE_MESH_ROOT / "index.obj",
        color=(70, 180, 255),
        contact_line_axis=(0.0, 1.0, 0.0),
    ),
    ObjectSpec(
        name="middle",
        cube_name="cube_april_36h11_0_5_1x1x1_15mm",
        source_camera="middle_finger_cam",
        robot_link="left_finger3_link4",
        robot_tip_link="left_finger3_tip_link",
        mesh_path=FINGEREYE_MESH_ROOT / "middle.obj",
        color=(110, 215, 120),
        contact_line_axis=(0.0, 1.0, 0.0),
    ),
)

# Semantic OBJ-to-fingertip correspondences for retargeting only. These transforms
# are not robot assembly parameters and must never be written into the Wuji URDF.
SEMANTIC_LINK_FROM_OBJ = {
    "thumb": ([0.0, 0.024865, 0.02], [0.0, 0.0, 0.0]),
    "index": ([0.0, -0.019365, 0.02], [3.14159, 0.0, 0.0]),
    "middle": ([-0.022681, 0.01599, -0.044], [-1.57, 0.0, 1.57]),
}


def matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).as_quat()
    return np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def wxyz_to_matrix(wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(wxyz, dtype=np.float64).reshape(4)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def matrix_to_wxyz_xyz(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    return np.concatenate([matrix_to_wxyz(transform[:3, :3]), transform[:3, 3]])


def wxyz_xyz_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = wxyz_to_matrix(pose[:4])
    transform[:3, 3] = pose[4:]
    return transform


def transforms_to_wxyz_xyz(transforms: np.ndarray) -> np.ndarray:
    transforms = np.asarray(transforms, dtype=np.float64)
    flat = transforms.reshape(-1, 4, 4)
    poses = np.stack([matrix_to_wxyz_xyz(value) for value in flat], axis=0)
    return poses.reshape(*transforms.shape[:-2], 7)


def poses_to_transforms(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    flat = poses.reshape(-1, 7)
    transforms = np.stack([wxyz_xyz_to_matrix(value) for value in flat], axis=0)
    return transforms.reshape(*poses.shape[:-1], 4, 4)


def validate_transform(transform: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError(f"{label} has an invalid homogeneous bottom row")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{label} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{label} rotation determinant is not +1")
    return value


def load_yaml_transform(path: Path, keys: tuple[str, ...]) -> np.ndarray:
    with path.open("r", encoding="utf-8") as file:
        node: Any = yaml.safe_load(file)
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"Missing {'.'.join(keys)} in {path}")
        node = node[key]
    return validate_transform(np.asarray(node, dtype=np.float64), f"{path}:{'.'.join(keys)}")


def load_world_from_camera(
    middle_extrinsics: Path,
    thumb_extrinsics: Path,
) -> dict[str, np.ndarray]:
    return {
        "middle_finger_cam": load_yaml_transform(
            middle_extrinsics,
            ("solution", "T_cube_middle_finger_cam"),
        ),
        "thumb_web_cam": load_yaml_transform(
            thumb_extrinsics,
            ("Q_T_thumb_web_cam",),
        ),
    }


def result_transform_m(result: dict[str, Any]) -> np.ndarray:
    transform = np.asarray(result["T"], dtype=np.float64).copy()
    transform[:3, 3] /= 1000.0
    return validate_transform(transform, "T_camera_obj")


def find_cube_result(camera_record: dict[str, Any], cube_name: str) -> dict[str, Any]:
    for item in camera_record.get("offline_pos", {}).get("cube_results", []):
        if str(item.get("cube_name")) == cube_name:
            return item.get("result", {})
    return {}


def build_world_pose_payload(
    frame_pair: dict[str, Any],
    world_from_camera: dict[str, np.ndarray],
) -> dict[str, Any]:
    objects: dict[str, Any] = {}
    for spec in OBJECT_SPECS:
        result = find_cube_result(
            frame_pair["cameras"][spec.source_camera],
            spec.cube_name,
        )
        success = bool(result.get("success", False)) and result.get("T") is not None
        entry: dict[str, Any] = {
            "success": success,
            "cube_name": spec.cube_name,
            "source_camera": spec.source_camera,
            "predicted": bool(result.get("predicted", False)),
            "pose_source": str(result.get("pose_source", result.get("pose_backend", ""))),
            "reproj_error_px": float(result.get("reproj_error", float("nan"))),
        }
        if success:
            transform = world_from_camera[spec.source_camera] @ result_transform_m(result)
            entry["T_hand_back_cube_obj"] = transform
        objects[spec.name] = entry
    return {
        "schema": WORLD_POSE_SCHEMA,
        "frame_convention": "A_T_B maps coordinates from frame B into frame A",
        "translation_unit": "m",
        "reference_frame": "hand_back_cube",
        "objects": objects,
    }


def print_rewrite_progress(label: str, done: int, total: int, finish: bool = False) -> None:
    width = 32
    ratio = min(max(done / max(total, 1), 0.0), 1.0)
    filled = int(round(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    end = "\n" if finish else ""
    print(
        f"\r[INFO] {label} [{bar}] {done / 1024**3:.2f}/{total / 1024**3:.2f} GiB",
        end=end,
        flush=True,
    )


def ensure_atomic_rewrite_space(pkl_path: Path) -> None:
    required = pkl_path.stat().st_size + 2 * 1024**3
    free = shutil.disk_usage(pkl_path.parent).free
    if free < required:
        raise RuntimeError(
            f"Atomic rewrite requires about {required / 1024**3:.1f} GiB free; "
            f"only {free / 1024**3:.1f} GiB is available"
        )


def embed_world_obj_poses(
    pkl_path: Path,
    world_from_camera: dict[str, np.ndarray],
    middle_extrinsics: Path,
    thumb_extrinsics: Path,
    *,
    force: bool,
) -> None:
    with pkl_path.open("rb") as source:
        header = pickle.load(source)
    existing = header.get("metadata", {}).get(WORLD_POSE_FIELD, {})
    if existing.get("schema") == WORLD_POSE_SCHEMA and not force:
        print(f"[INFO] PKL already contains {WORLD_POSE_SCHEMA}; skipping world-pose rewrite")
        return

    ensure_atomic_rewrite_space(pkl_path)
    temporary = pkl_path.with_suffix(pkl_path.suffix + ".028-world-rewrite.tmp")
    if temporary.exists():
        temporary.unlink()
    total_bytes = pkl_path.stat().st_size
    frame_count = 0
    success_counts = {spec.name: 0 for spec in OBJECT_SPECS}
    last_progress = 0.0
    try:
        with pkl_path.open("rb") as source, temporary.open("wb") as destination:
            header = pickle.load(source)
            if not isinstance(header, dict) or header.get("format") != EXPECTED_PKL_FORMAT:
                raise ValueError(f"Unsupported PKL format: {header.get('format')}")
            if force:
                # A forced world-pose refresh means any embedded IK/retargeting
                # result was derived from obsolete object poses.
                header.setdefault("metadata", {}).pop(RETARGET_FIELD, None)
            header.setdefault("metadata", {})[WORLD_POSE_FIELD] = {
                "schema": WORLD_POSE_SCHEMA,
                "frame_convention": "A_T_B maps coordinates from frame B into frame A",
                "translation_unit": "m",
                "middle_extrinsics": str(middle_extrinsics),
                "thumb_extrinsics": str(thumb_extrinsics),
                "objects": {
                    spec.name: {
                        "cube_name": spec.cube_name,
                        "source_camera": spec.source_camera,
                    }
                    for spec in OBJECT_SPECS
                },
            }
            pickle.dump(header, destination, protocol=pickle.HIGHEST_PROTOCOL)
            while True:
                try:
                    record = pickle.load(source)
                except EOFError:
                    break
                if isinstance(record, dict) and record.get("type") == "frame_pair":
                    if force:
                        record.pop(RETARGET_FIELD, None)
                    payload = build_world_pose_payload(record, world_from_camera)
                    record[WORLD_POSE_FIELD] = payload
                    frame_count += 1
                    for spec in OBJECT_SPECS:
                        success_counts[spec.name] += int(
                            payload["objects"][spec.name]["success"]
                        )
                pickle.dump(record, destination, protocol=pickle.HIGHEST_PROTOCOL)
                now = time.monotonic()
                if now - last_progress >= 0.5:
                    print_rewrite_progress("Embedding world OBJ poses", source.tell(), total_bytes)
                    last_progress = now
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(pkl_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    print_rewrite_progress("Embedding world OBJ poses", total_bytes, total_bytes, finish=True)
    print(f"[INFO] Embedded world OBJ poses in {frame_count} frames: {success_counts}")


@dataclass
class PoseDataset:
    header: dict[str, Any]
    observations: np.ndarray
    confidence: np.ndarray
    timestamps: np.ndarray
    predicted: np.ndarray
    reprojection_px: np.ndarray
    embedded_qpos: np.ndarray | None
    embedded_offset: np.ndarray | None
    embedded_scale: float | None
    embedded_wrist_delta: np.ndarray | None
    embedded_metrics: list[dict[str, Any]] | None


def confidence_from_entry(entry: dict[str, Any]) -> float:
    if not entry.get("success", False):
        return 0.0
    confidence = 0.35 if entry.get("predicted", False) else 1.0
    reproj = float(entry.get("reproj_error_px", float("nan")))
    if np.isfinite(reproj):
        confidence *= float(np.clip(1.0 / np.sqrt(1.0 + (reproj / 2.0) ** 2), 0.25, 1.0))
    return confidence


def balanced_retargeting_weights(confidence: np.ndarray) -> np.ndarray:
    confidence = np.asarray(confidence, dtype=np.float64)
    object_mean = np.maximum(np.mean(confidence, axis=0, keepdims=True), 1e-6)
    return np.clip(confidence / object_mean, 0.25, 2.0)


def repair_isolated_pose_spikes(
    observations: np.ndarray,
    confidence: np.ndarray,
    timestamps: np.ndarray,
    predicted: np.ndarray,
) -> list[tuple[int, str]]:
    repaired: list[tuple[int, str]] = []
    for object_index, spec in enumerate(OBJECT_SPECS):
        for frame_index in range(1, len(observations) - 1):
            if np.any(confidence[frame_index - 1 : frame_index + 2, object_index] <= 0.0):
                continue
            rotations = Rotation.from_matrix(
                observations[frame_index - 1 : frame_index + 2, object_index, :3, :3]
            )
            previous_jump = np.rad2deg((rotations[0].inv() * rotations[1]).magnitude())
            next_jump = np.rad2deg((rotations[1].inv() * rotations[2]).magnitude())
            bridge_rotation = np.rad2deg((rotations[0].inv() * rotations[2]).magnitude())
            bridge_translation = np.linalg.norm(
                observations[frame_index + 1, object_index, :3, 3]
                - observations[frame_index - 1, object_index, :3, 3]
            )
            if not (
                previous_jump > 45.0
                and next_jump > 45.0
                and bridge_rotation < 20.0
                and bridge_translation < 0.02
            ):
                continue
            denominator = timestamps[frame_index + 1] - timestamps[frame_index - 1]
            alpha = (
                (timestamps[frame_index] - timestamps[frame_index - 1]) / denominator
                if denominator > 1e-6
                else 0.5
            )
            alpha = float(np.clip(alpha, 0.0, 1.0))
            observations[frame_index, object_index, :3, 3] = (
                (1.0 - alpha) * observations[frame_index - 1, object_index, :3, 3]
                + alpha * observations[frame_index + 1, object_index, :3, 3]
            )
            observations[frame_index, object_index, :3, :3] = Slerp(
                [0.0, 1.0],
                Rotation.from_matrix(
                    observations[
                        [frame_index - 1, frame_index + 1], object_index, :3, :3
                    ]
                ),
            )([alpha]).as_matrix()[0]
            confidence[frame_index, object_index] = (
                min(
                    confidence[frame_index - 1, object_index],
                    confidence[frame_index + 1, object_index],
                )
                * 0.5
            )
            predicted[frame_index, object_index] = True
            repaired.append((frame_index, spec.name))
    return repaired


def load_pose_dataset(pkl_path: Path, max_frames: int | None = None) -> PoseDataset:
    observations: list[np.ndarray] = []
    confidence: list[np.ndarray] = []
    timestamps: list[float] = []
    predicted: list[np.ndarray] = []
    reprojection: list[np.ndarray] = []
    qpos: list[np.ndarray] = []
    wrist_delta: list[np.ndarray] = []
    frame_metrics: list[dict[str, Any]] = []
    all_embedded = True
    total_bytes = pkl_path.stat().st_size
    last_progress = 0.0
    with pkl_path.open("rb") as source:
        header = pickle.load(source)
        if not isinstance(header, dict) or header.get("format") != EXPECTED_PKL_FORMAT:
            raise ValueError(f"Unsupported PKL format: {header.get('format')}")
        if header.get("metadata", {}).get(WORLD_POSE_FIELD, {}).get("schema") != WORLD_POSE_SCHEMA:
            raise ValueError(f"PKL does not contain {WORLD_POSE_SCHEMA}")
        while max_frames is None or len(observations) < max_frames:
            try:
                record = pickle.load(source)
            except EOFError:
                break
            if not isinstance(record, dict) or record.get("type") != "frame_pair":
                continue
            payload = record.get(WORLD_POSE_FIELD, {})
            object_entries = payload.get("objects", {})
            frame_poses = np.tile(np.eye(4, dtype=np.float64), (len(OBJECT_SPECS), 1, 1))
            frame_confidence = np.zeros(len(OBJECT_SPECS), dtype=np.float64)
            frame_predicted = np.zeros(len(OBJECT_SPECS), dtype=bool)
            frame_reproj = np.full(len(OBJECT_SPECS), np.nan, dtype=np.float64)
            for object_index, spec in enumerate(OBJECT_SPECS):
                entry = object_entries.get(spec.name, {})
                if entry.get("success", False):
                    frame_poses[object_index] = validate_transform(
                        entry["T_hand_back_cube_obj"],
                        f"frame {len(observations)} {spec.name}",
                    )
                frame_confidence[object_index] = confidence_from_entry(entry)
                frame_predicted[object_index] = bool(entry.get("predicted", False))
                frame_reproj[object_index] = float(entry.get("reproj_error_px", float("nan")))
            observations.append(frame_poses)
            confidence.append(frame_confidence)
            predicted.append(frame_predicted)
            reprojection.append(frame_reproj)
            timestamps.append(float(record.get("pair_timestamp", len(timestamps))))
            embedded = record.get(RETARGET_FIELD)
            if isinstance(embedded, dict) and embedded.get("algorithm") == RETARGET_ALGORITHM:
                qpos.append(np.asarray(embedded["qpos"], dtype=np.float64))
                wrist_delta.append(
                    validate_transform(
                        embedded["T_left_palm_reference_left_palm_dynamic"],
                        f"frame {len(observations) - 1} wrist delta",
                    )
                )
                frame_metrics.append(embedded.get("metrics", {}))
            else:
                all_embedded = False
            now = time.monotonic()
            if now - last_progress >= 0.5:
                print_rewrite_progress("Loading target poses", source.tell(), total_bytes)
                last_progress = now
    print_rewrite_progress("Loading target poses", total_bytes, total_bytes, finish=True)
    if not observations:
        raise ValueError("No frame_pair records found")
    observations_array = np.asarray(observations, dtype=np.float64)
    confidence_array = np.asarray(confidence, dtype=np.float64)
    timestamps_array = np.asarray(timestamps, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=bool)
    repaired = repair_isolated_pose_spikes(
        observations_array,
        confidence_array,
        timestamps_array,
        predicted_array,
    )
    if repaired:
        print(f"[WARN] Repaired isolated pose spikes: {repaired}")
    missing = np.argwhere(confidence_array <= 0.0)
    if len(missing):
        print(f"[WARN] Missing object targets: {len(missing)} slots; zero-weighted in IK")
    retarget_metadata = header.get("metadata", {}).get(RETARGET_FIELD, {})
    embedded_offset = None
    embedded_scale = None
    if all_embedded and len(qpos) == len(observations):
        embedded_offset = validate_transform(
            retarget_metadata["T_left_palm_link_hand_back_cube"],
            "embedded T_left_palm_link_hand_back_cube",
        )
        embedded_scale = float(retarget_metadata.get("human_to_wuji_scale", 1.0))
    return PoseDataset(
        header=header,
        observations=observations_array,
        confidence=confidence_array,
        timestamps=timestamps_array,
        predicted=predicted_array,
        reprojection_px=np.asarray(reprojection, dtype=np.float64),
        embedded_qpos=np.asarray(qpos, dtype=np.float64) if embedded_offset is not None else None,
        embedded_offset=embedded_offset,
        embedded_scale=embedded_scale,
        embedded_wrist_delta=(
            np.asarray(wrist_delta, dtype=np.float64)
            if embedded_offset is not None
            else None
        ),
        embedded_metrics=frame_metrics if embedded_offset is not None else None,
    )


def load_urdf(urdf_path: Path) -> yourdfpy.URDF:
    return yourdfpy.URDF.load(
        str(urdf_path),
        filename_handler=partial(yourdfpy.filename_handler_magic, dir=urdf_path.parent),
    )


def rpy_transform(xyz: list[float], rpy: list[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def load_link_from_obj_transforms(urdf_path: Path) -> dict[str, np.ndarray]:
    root = ET.parse(urdf_path).getroot()
    transforms: dict[str, np.ndarray] = {}
    expected_mesh = {spec.mesh_path.name: spec for spec in OBJECT_SPECS}
    for link in root.findall("link"):
        for visual in link.findall("visual"):
            mesh = visual.find("./geometry/mesh")
            if mesh is None:
                continue
            mesh_name = Path(mesh.attrib.get("filename", "")).name
            spec = expected_mesh.get(mesh_name)
            if spec is None:
                continue
            if link.attrib["name"] != spec.robot_link:
                raise ValueError(
                    f"{mesh_name} is attached to {link.attrib['name']}, expected {spec.robot_link}"
                )
            origin = visual.find("origin")
            xyz = [float(value) for value in origin.attrib.get("xyz", "0 0 0").split()]
            rpy = [float(value) for value in origin.attrib.get("rpy", "0 0 0").split()]
            transforms[spec.name] = rpy_transform(xyz, rpy)
    if not transforms:
        print(
            "[INFO] Physical URDF contains only original Wuji meshes; "
            "using external semantic OBJ correspondences for retargeting"
        )
        return {
            name: rpy_transform(list(xyz), list(rpy))
            for name, (xyz, rpy) in SEMANTIC_LINK_FROM_OBJ.items()
        }
    missing = [spec.name for spec in OBJECT_SPECS if spec.name not in transforms]
    if missing:
        raise ValueError(f"URDF is missing OBJ visual transforms for: {missing}")
    return transforms


def load_obj_from_tip_transforms(
    urdf_path: Path,
    link_from_obj: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    root = ET.parse(urdf_path).getroot()
    child_to_joint = {
        joint.find("child").attrib["link"]: joint
        for joint in root.findall("joint")
        if joint.find("child") is not None
    }
    transforms: dict[str, np.ndarray] = {}
    for spec in OBJECT_SPECS:
        joint = child_to_joint.get(spec.robot_tip_link)
        if joint is None:
            raise ValueError(f"Missing fixed joint for {spec.robot_tip_link}")
        parent = joint.find("parent")
        if parent is None or parent.attrib["link"] != spec.robot_link:
            raise ValueError(
                f"{spec.robot_tip_link} must be fixed directly to {spec.robot_link}"
            )
        origin = joint.find("origin")
        xyz = [float(value) for value in origin.attrib.get("xyz", "0 0 0").split()]
        rpy = [float(value) for value in origin.attrib.get("rpy", "0 0 0").split()]
        link_from_tip = rpy_transform(xyz, rpy)
        transforms[spec.name] = np.linalg.inv(link_from_obj[spec.name]) @ link_from_tip

    return transforms


def load_obj_mesh_m(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    mesh = (
        trimesh.util.concatenate(tuple(loaded.geometry.values()))
        if isinstance(loaded, trimesh.Scene)
        else loaded
    )
    mesh = mesh.copy()
    mesh.apply_scale(0.001)
    return mesh


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    return np.einsum("ij,...j->...i", transform[:3, :3], points) + transform[:3, 3]


def load_contact_surface_points_obj(
    obj_from_tip: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Select a repeatable three-point patch on each OBJ contact surface."""
    surface_points: dict[str, np.ndarray] = {}
    for spec in OBJECT_SPECS:
        mesh = load_obj_mesh_m(spec.mesh_path)
        transform = np.asarray(obj_from_tip[spec.name], dtype=np.float64)
        anchor_query = transform[:3, 3]
        center, _, center_faces = trimesh.proximity.closest_point_naive(
            mesh, anchor_query[None, :]
        )
        center = center[0]
        normal = np.asarray(mesh.face_normals[int(center_faces[0])], dtype=np.float64)

        line_axis = transform[:3, :3] @ np.asarray(
            spec.contact_line_axis, dtype=np.float64
        )
        line_axis -= normal * np.dot(line_axis, normal)
        line_norm = np.linalg.norm(line_axis)
        if line_norm < 1e-8:
            raise ValueError(f"{spec.name} contact line is parallel to its surface normal")
        line_axis /= line_norm
        inward_axis = np.cross(normal, line_axis)
        inward_axis /= np.linalg.norm(inward_axis)

        inward_probes = np.stack(
            [
                center + CONTACT_SURFACE_INSET_M * inward_axis,
                center - CONTACT_SURFACE_INSET_M * inward_axis,
            ]
        )
        inward_points, inward_distances, inward_faces = (
            trimesh.proximity.closest_point_naive(mesh, inward_probes)
        )
        normal_alignment = np.abs(mesh.face_normals[inward_faces] @ normal)
        inward_scores = inward_distances + CONTACT_SURFACE_INSET_M * (
            1.0 - normal_alignment
        )
        inward_point = inward_points[int(np.argmin(inward_scores))]

        edge_probes = np.stack(
            [
                center - CONTACT_SURFACE_HALF_LINE_M * line_axis,
                center + CONTACT_SURFACE_HALF_LINE_M * line_axis,
            ]
        )
        edge_points, _, _ = trimesh.proximity.closest_point_naive(mesh, edge_probes)
        points = np.concatenate([edge_points, inward_point[None, :]], axis=0)
        doubled_area = np.linalg.norm(
            np.cross(points[1] - points[0], points[2] - points[0])
        )
        if doubled_area < 2e-6:
            raise ValueError(f"{spec.name} contact surface points are nearly collinear")
        surface_points[spec.name] = points
        print(
            f"[INFO] {spec.name} contact surface points (OBJ mm): "
            f"{np.round(points * 1000.0, 4).tolist()}"
        )
    return surface_points


def apply_retarget_posture_bounds(
    joint_names: list[str] | tuple[str, ...],
    physical_lower: np.ndarray,
    physical_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(physical_lower, dtype=np.float32).copy()
    upper = np.asarray(physical_upper, dtype=np.float32).copy()
    for joint_name, thumb_lower, thumb_upper in zip(
        THUMB_JOINT_NAMES,
        THUMB_RETARGET_LOWER,
        THUMB_RETARGET_UPPER,
    ):
        joint_index = list(joint_names).index(joint_name)
        lower[joint_index] = max(lower[joint_index], float(thumb_lower))
        upper[joint_index] = min(upper[joint_index], float(thumb_upper))
        if lower[joint_index] > upper[joint_index]:
            raise ValueError(f"Empty retargeting range for {joint_name}")
    return lower, upper


class SequentialWujiContactSolver:
    def __init__(
        self,
        robot: pk.Robot,
        tip_link_indices: np.ndarray,
        contact_surface_points_tip: np.ndarray,
        natural_qpos: np.ndarray,
        active_mask: np.ndarray,
        *,
        max_iterations: int,
    ) -> None:
        self.robot = robot
        self.natural_qpos = np.asarray(natural_qpos, dtype=np.float32)
        self.active_mask = np.asarray(active_mask, dtype=bool)
        self.lower, self.upper = apply_retarget_posture_bounds(
            tuple(robot.joints.names),
            np.asarray(robot.joints.lower_limits, dtype=np.float32),
            np.asarray(robot.joints.upper_limits, dtype=np.float32),
        )
        self.velocity = np.asarray(robot.joints.velocity_limits, dtype=np.float32)
        self.max_iterations = max_iterations
        num_objects = len(OBJECT_SPECS)
        num_joints = robot.joints.num_actuated_joints

        class TargetPointVar(
            jaxls.Var[jax.Array],
            default_factory=lambda: jnp.zeros(
                (num_objects, CONTACT_SURFACE_POINT_COUNT, 3),
                dtype=jnp.float32,
            ),
            tangent_dim=0,
            retract_fn=lambda value, _delta: value,
        ): ...

        class TargetWeightVar(
            jaxls.Var[jax.Array],
            default_factory=lambda: jnp.ones((num_objects,), dtype=jnp.float32),
            tangent_dim=0,
            retract_fn=lambda value, _delta: value,
        ): ...

        class PreviousCfgVar(
            jaxls.Var[jax.Array],
            default_factory=lambda: jnp.asarray(self.natural_qpos),
            tangent_dim=0,
            retract_fn=lambda value, _delta: value,
        ): ...

        class DtVar(
            jaxls.Var[jax.Array],
            default_factory=lambda: jnp.asarray(1.0 / 15.0, dtype=jnp.float32),
            tangent_dim=0,
            retract_fn=lambda value, _delta: value,
        ): ...

        class PreviousScaleVar(
            jaxls.Var[jax.Array],
            default_factory=lambda: jnp.asarray(1.0, dtype=jnp.float32),
            tangent_dim=0,
            retract_fn=lambda value, _delta: value,
        ): ...

        joint_var = robot.joint_var_cls(0)
        target_point_var = TargetPointVar(0)
        target_weight_var = TargetWeightVar(0)
        previous_cfg_var = PreviousCfgVar(0)
        dt_var = DtVar(0)
        previous_scale_var = PreviousScaleVar(0)
        tip_link_indices_jax = jnp.asarray(tip_link_indices, dtype=jnp.int32)
        surface_points_tip_jax = jnp.asarray(
            contact_surface_points_tip, dtype=jnp.float32
        )
        natural_jax = jnp.asarray(self.natural_qpos)
        active_jax = jnp.asarray(active_mask, dtype=jnp.float32)
        rest_weight_values = (
            np.asarray(active_mask, dtype=np.float32) * REST_WEIGHT
            + (1.0 - np.asarray(active_mask, dtype=np.float32)) * INACTIVE_WEIGHT
        )
        for joint_name in THUMB_JOINT_NAMES:
            rest_weight_values[list(robot.joints.names).index(joint_name)] = (
                THUMB_POSTURE_WEIGHT
            )
        thumb_joint_indices = jnp.asarray(
            [list(robot.joints.names).index(name) for name in THUMB_JOINT_NAMES],
            dtype=jnp.int32,
        )
        rest_weights = jnp.asarray(rest_weight_values)
        smooth_weights = active_jax * SMOOTHNESS_WEIGHT + (1.0 - active_jax) * INACTIVE_WEIGHT
        lower_jax = jnp.asarray(self.lower)
        upper_jax = jnp.asarray(self.upper)
        soft_limit_margin = jnp.maximum(
            (upper_jax - lower_jax) * SOFT_LIMIT_MARGIN_RATIO,
            1e-4,
        )
        velocity_limits = jnp.asarray(self.velocity) * VELOCITY_LIMIT_SCALE

        @jaxls.Cost.factory
        def contact_surface_cost(
            vals: jaxls.VarValues,
            var_q: jaxls.Var[jax.Array],
            var_target: jaxls.Var[jax.Array],
            var_weight: jaxls.Var[jax.Array],
        ) -> jax.Array:
            q = vals[var_q]
            target_points = vals[var_target]
            weights = jnp.sqrt(jnp.maximum(vals[var_weight], 0.0))
            predicted_tip = jaxlie.SE3(
                robot.forward_kinematics(q)[tip_link_indices_jax]
            )
            predicted_points = (
                jnp.einsum(
                    "nij,npj->npi",
                    predicted_tip.rotation().as_matrix(),
                    surface_points_tip_jax,
                )
                + predicted_tip.translation()[:, None, :]
            )
            return (
                (predicted_points - target_points)
                * SURFACE_POINT_WEIGHT
                * weights[:, None, None]
            ).reshape(-1)

        @jaxls.Cost.factory
        def previous_solution_cost(
            vals: jaxls.VarValues,
            var_q: jaxls.Var[jax.Array],
            var_previous: jaxls.Var[jax.Array],
            var_scale: jaxls.Var[jax.Array],
        ) -> jax.Array:
            return (
                (vals[var_q] - vals[var_previous])
                * smooth_weights
                * vals[var_scale]
            ).reshape(-1)

        @jaxls.Cost.factory
        def soft_joint_limit_cost(
            vals: jaxls.VarValues,
            var_q: jaxls.Var[jax.Array],
        ) -> jax.Array:
            qpos = vals[var_q]
            lower_residual = jnp.maximum(
                0.0,
                lower_jax + soft_limit_margin - qpos,
            ) / soft_limit_margin
            upper_residual = jnp.maximum(
                0.0,
                qpos - (upper_jax - soft_limit_margin),
            ) / soft_limit_margin
            return (
                jnp.concatenate([lower_residual, upper_residual])
                * jnp.concatenate([active_jax, active_jax])
                * SOFT_LIMIT_WEIGHT
            ).reshape(-1)

        @jaxls.Cost.factory
        def thumb_distal_curvature_cost(
            vals: jaxls.VarValues,
            var_q: jaxls.Var[jax.Array],
        ) -> jax.Array:
            thumb_qpos = vals[var_q][thumb_joint_indices]
            return jnp.asarray(
                [
                    (thumb_qpos[2] - thumb_qpos[3])
                    * THUMB_DISTAL_CURVATURE_WEIGHT
                ]
            )

        @jaxls.Cost.factory(kind="constraint_leq_zero")
        def retarget_range_constraint(
            vals: jaxls.VarValues,
            var_q: jaxls.Var[jax.Array],
        ) -> jax.Array:
            qpos = vals[var_q]
            thumb_qpos = qpos[thumb_joint_indices]
            distal_bend_difference = (
                jnp.abs(thumb_qpos[2] - thumb_qpos[3])
                - THUMB_DISTAL_BEND_DIFFERENCE_MAX_RAD
            )
            return jnp.concatenate(
                [
                    lower_jax - qpos,
                    qpos - upper_jax,
                    jnp.asarray([distal_bend_difference]),
                ]
            )

        @jaxls.Cost.factory(kind="constraint_leq_zero")
        def velocity_constraint(
            vals: jaxls.VarValues,
            var_q: jaxls.Var[jax.Array],
            var_previous: jaxls.Var[jax.Array],
            var_dt: jaxls.Var[jax.Array],
            var_scale: jaxls.Var[jax.Array],
        ) -> jax.Array:
            velocity = jnp.abs(vals[var_q] - vals[var_previous]) / jnp.maximum(
                vals[var_dt], 1e-4
            )
            return jnp.maximum(0.0, velocity - velocity_limits) * vals[var_scale]

        costs: list[jaxls.Cost] = [
            contact_surface_cost(joint_var, target_point_var, target_weight_var),
            pk.costs.rest_cost(joint_var, natural_jax, rest_weights),
            previous_solution_cost(joint_var, previous_cfg_var, previous_scale_var),
            soft_joint_limit_cost(joint_var),
            thumb_distal_curvature_cost(joint_var),
            pk.costs.limit_constraint(robot, joint_var),
            retarget_range_constraint(joint_var),
            velocity_constraint(
                joint_var,
                previous_cfg_var,
                dt_var,
                previous_scale_var,
            ),
        ]
        variables = [
            joint_var,
            target_point_var,
            target_weight_var,
            previous_cfg_var,
            dt_var,
            previous_scale_var,
        ]
        self.joint_var = joint_var
        self.target_point_var = target_point_var
        self.target_weight_var = target_weight_var
        self.previous_cfg_var = previous_cfg_var
        self.dt_var = dt_var
        self.previous_scale_var = previous_scale_var
        self.problem = jaxls.LeastSquaresProblem(costs=costs, variables=variables).analyze(
            use_onp=True
        )
        if self.natural_qpos.shape != (num_joints,):
            raise ValueError("Natural qpos dimension does not match the URDF")

    def solve_frame(
        self,
        target_points: np.ndarray,
        target_weights: np.ndarray,
        initial_qpos: np.ndarray,
        previous_qpos: np.ndarray,
        dt: float,
        previous_scale: float,
    ) -> np.ndarray:
        initial_values = jaxls.VarValues.make(
            [
                self.joint_var.with_value(jnp.asarray(initial_qpos, dtype=jnp.float32)),
                self.target_point_var.with_value(
                    jnp.asarray(target_points, dtype=jnp.float32)
                ),
                self.target_weight_var.with_value(
                    jnp.asarray(target_weights, dtype=jnp.float32)
                ),
                self.previous_cfg_var.with_value(
                    jnp.asarray(previous_qpos, dtype=jnp.float32)
                ),
                self.dt_var.with_value(jnp.asarray(dt, dtype=jnp.float32)),
                self.previous_scale_var.with_value(
                    jnp.asarray(previous_scale, dtype=jnp.float32)
                ),
            ]
        )
        solution = self.problem.solve(
            initial_vals=initial_values,
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            termination=jaxls.TerminationConfig(max_iterations=self.max_iterations),
        )
        qpos = np.asarray(solution[self.joint_var], dtype=np.float32)
        qpos = np.clip(qpos, self.lower, self.upper)
        qpos[~self.active_mask] = self.natural_qpos[~self.active_mask]
        if previous_scale > 0.0:
            max_step = self.velocity * VELOCITY_LIMIT_SCALE * max(float(dt), 1e-4)
            qpos = np.clip(qpos, previous_qpos - max_step, previous_qpos + max_step)
            qpos = np.clip(qpos, self.lower, self.upper)
        return qpos

    def solve_sequence(
        self,
        target_points: np.ndarray,
        target_weights: np.ndarray,
        timestamps: np.ndarray,
        initial_qpos: np.ndarray | None,
        *,
        progress_label: str,
    ) -> np.ndarray:
        frame_count = len(target_points)
        output = np.zeros((frame_count, len(self.natural_qpos)), dtype=np.float32)
        previous = self.natural_qpos.copy()
        for frame_index in range(frame_count):
            dt = (
                float(np.clip(timestamps[frame_index] - timestamps[frame_index - 1], 1e-3, 0.5))
                if frame_index > 0
                else 1.0
            )
            initial = (
                np.asarray(initial_qpos[frame_index], dtype=np.float32)
                if initial_qpos is not None
                else previous
            )
            output[frame_index] = self.solve_frame(
                target_points[frame_index],
                target_weights[frame_index],
                initial,
                previous,
                dt,
                0.0 if frame_index == 0 else 1.0,
            )
            previous = output[frame_index]
            if frame_index == 0 or (frame_index + 1) % 50 == 0 or frame_index + 1 == frame_count:
                print(f"[INFO] {progress_label}: {frame_index + 1}/{frame_count}")
        return output


def compose_target_poses(
    palm_from_hand_back: np.ndarray,
    observations: np.ndarray,
    human_to_wuji_scale: float = 1.0,
) -> np.ndarray:
    observations = np.asarray(observations, dtype=np.float64)
    target = np.matmul(palm_from_hand_back[None, None, :, :], observations)
    scaled_position = observations[..., :3, 3] * float(human_to_wuji_scale)
    target[..., :3, 3] = (
        np.einsum("ij,...j->...i", palm_from_hand_back[:3, :3], scaled_position)
        + palm_from_hand_back[:3, 3]
    )
    return target


def compose_target_tip_poses(
    palm_from_hand_back: np.ndarray,
    observations: np.ndarray,
    obj_from_tip: np.ndarray,
    human_to_wuji_scale: float = 1.0,
) -> np.ndarray:
    cube_from_tip = np.matmul(
        np.asarray(observations, dtype=np.float64),
        obj_from_tip[None, :, :, :],
    )
    target = np.matmul(palm_from_hand_back[None, None, :, :], cube_from_tip)
    scaled_position = (
        np.asarray(observations, dtype=np.float64)[..., :3, 3]
        * float(human_to_wuji_scale)
        + np.einsum(
            "...oij,oj->...oi",
            np.asarray(observations, dtype=np.float64)[..., :3, :3],
            obj_from_tip[:, :3, 3],
        )
    )
    target[..., :3, 3] = (
        np.einsum("ij,...j->...i", palm_from_hand_back[:3, :3], scaled_position)
        + palm_from_hand_back[:3, 3]
    )
    return target


def compose_target_surface_points(
    palm_from_hand_back: np.ndarray,
    observations: np.ndarray,
    surface_points_obj: np.ndarray,
    human_to_wuji_scale: float = 1.0,
) -> np.ndarray:
    observations = np.asarray(observations, dtype=np.float64)
    # Scale the human kinematic displacement from the hand-back cube to each
    # OBJ origin, but preserve the physical dimensions of the contact patch.
    points_hand_back = (
        np.einsum(
            "...oij,opj->...opi",
            observations[..., :3, :3],
            np.asarray(surface_points_obj, dtype=np.float64),
        )
        + observations[..., :3, 3][..., :, None, :] * float(human_to_wuji_scale)
    )
    return (
        np.einsum(
            "ij,...opj->...opi",
            np.asarray(palm_from_hand_back, dtype=np.float64)[:3, :3],
            points_hand_back,
        )
        + np.asarray(palm_from_hand_back, dtype=np.float64)[:3, 3]
    )


def compute_predicted_link_poses(
    robot: pk.Robot,
    qpos: np.ndarray,
    link_indices: np.ndarray,
) -> np.ndarray:
    fk_poses = np.asarray(
        robot.forward_kinematics(jnp.asarray(qpos, dtype=jnp.float32)),
        dtype=np.float64,
    )
    return poses_to_transforms(fk_poses[:, link_indices, :])


def compute_predicted_surface_points(
    robot: pk.Robot,
    qpos: np.ndarray,
    tip_link_indices: np.ndarray,
    surface_points_tip: np.ndarray,
) -> np.ndarray:
    predicted_tip = compute_predicted_link_poses(robot, qpos, tip_link_indices)
    return (
        np.einsum(
            "...oij,opj->...opi",
            predicted_tip[..., :3, :3],
            np.asarray(surface_points_tip, dtype=np.float64),
        )
        + predicted_tip[..., :3, 3][..., :, None, :]
    )


def compute_predicted_obj_poses(
    robot: pk.Robot,
    qpos: np.ndarray,
    link_indices: np.ndarray,
    link_from_obj: np.ndarray,
) -> np.ndarray:
    link_transforms = compute_predicted_link_poses(robot, qpos, link_indices)
    return np.matmul(link_transforms, link_from_obj[None, :, :, :])


def apply_wrist_delta(
    wrist_delta: np.ndarray,
    poses: np.ndarray,
) -> np.ndarray:
    return np.matmul(np.asarray(wrist_delta)[:, None, :, :], np.asarray(poses))


def tip_error_arrays(
    predicted: np.ndarray,
    target: np.ndarray,
    contact_line_axes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position_m = np.linalg.norm(predicted[..., :3, 3] - target[..., :3, 3], axis=-1)
    predicted_line = np.einsum(
        "...nij,nj->...ni", predicted[..., :3, :3], contact_line_axes
    )
    target_line = np.einsum(
        "...nij,nj->...ni", target[..., :3, :3], contact_line_axes
    )
    cosine = np.abs(np.sum(predicted_line * target_line, axis=-1))
    line_deg = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return position_m, line_deg


def summarize_tip_errors(
    predicted: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    contact_line_axes: np.ndarray,
) -> dict[str, Any]:
    position_m, line_deg = tip_error_arrays(predicted, target, contact_line_axes)
    valid = confidence > 0.0
    summary: dict[str, Any] = {
        "position_mm": {
            "mean": float(np.mean(position_m[valid]) * 1000.0),
            "median": float(np.median(position_m[valid]) * 1000.0),
            "p95": float(np.percentile(position_m[valid], 95) * 1000.0),
            "max": float(np.max(position_m[valid]) * 1000.0),
        },
        "line_deg": {
            "mean": float(np.mean(line_deg[valid])),
            "median": float(np.median(line_deg[valid])),
            "p95": float(np.percentile(line_deg[valid], 95)),
            "max": float(np.max(line_deg[valid])),
        },
        "per_object": {},
    }
    for object_index, spec in enumerate(OBJECT_SPECS):
        object_valid = valid[:, object_index]
        summary["per_object"][spec.name] = {
            "position_mean_mm": float(
                np.mean(position_m[object_valid, object_index]) * 1000.0
            ),
            "position_p95_mm": float(
                np.percentile(position_m[object_valid, object_index], 95) * 1000.0
            ),
            "line_mean_deg": float(np.mean(line_deg[object_valid, object_index])),
            "line_p95_deg": float(
                np.percentile(line_deg[object_valid, object_index], 95)
            ),
        }
    return summary


def pose_error_arrays(
    predicted: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position_m = np.linalg.norm(predicted[..., :3, 3] - target[..., :3, 3], axis=-1)
    relative_rotation = np.matmul(
        np.swapaxes(predicted[..., :3, :3], -1, -2),
        target[..., :3, :3],
    )
    rotation_deg = np.rad2deg(
        Rotation.from_matrix(relative_rotation.reshape(-1, 3, 3)).magnitude()
    ).reshape(position_m.shape)
    return position_m, rotation_deg


def summarize_errors(
    predicted: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, Any]:
    position_m, rotation_deg = pose_error_arrays(predicted, target)
    valid = confidence > 0.0
    summary: dict[str, Any] = {
        "position_mm": {
            "mean": float(np.mean(position_m[valid]) * 1000.0),
            "median": float(np.median(position_m[valid]) * 1000.0),
            "p95": float(np.percentile(position_m[valid], 95) * 1000.0),
            "max": float(np.max(position_m[valid]) * 1000.0),
        },
        "rotation_deg": {
            "mean": float(np.mean(rotation_deg[valid])),
            "median": float(np.median(rotation_deg[valid])),
            "p95": float(np.percentile(rotation_deg[valid], 95)),
            "max": float(np.max(rotation_deg[valid])),
        },
        "per_object": {},
    }
    for object_index, spec in enumerate(OBJECT_SPECS):
        object_valid = valid[:, object_index]
        summary["per_object"][spec.name] = {
            "position_mean_mm": float(
                np.mean(position_m[object_valid, object_index]) * 1000.0
            ),
            "position_p95_mm": float(
                np.percentile(position_m[object_valid, object_index], 95) * 1000.0
            ),
            "rotation_mean_deg": float(
                np.mean(rotation_deg[object_valid, object_index])
            ),
            "rotation_p95_deg": float(
                np.percentile(rotation_deg[object_valid, object_index], 95)
            ),
        }
    return summary


def contact_surface_residuals(
    predicted_points: np.ndarray,
    target_points: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    weights = np.sqrt(np.maximum(confidence, 0.0))[..., None, None]
    return (
        (np.asarray(predicted_points) - np.asarray(target_points))
        * SURFACE_POINT_WEIGHT
        * weights
    ).reshape(-1)


def surface_point_error_arrays(
    predicted_points: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    return np.linalg.norm(
        np.asarray(predicted_points) - np.asarray(target_points), axis=-1
    )


def summarize_surface_point_errors(
    predicted_points: np.ndarray,
    target_points: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, Any]:
    errors_m = surface_point_error_arrays(predicted_points, target_points)
    valid = np.broadcast_to(
        (np.asarray(confidence) > 0.0)[..., None],
        errors_m.shape,
    )
    summary: dict[str, Any] = {
        "point_mm": {
            "mean": float(np.mean(errors_m[valid]) * 1000.0),
            "median": float(np.median(errors_m[valid]) * 1000.0),
            "p95": float(np.percentile(errors_m[valid], 95) * 1000.0),
            "max": float(np.max(errors_m[valid]) * 1000.0),
        },
        "per_object": {},
    }
    for object_index, spec in enumerate(OBJECT_SPECS):
        object_valid = valid[..., object_index, :]
        object_errors = errors_m[..., object_index, :]
        summary["per_object"][spec.name] = {
            "point_mean_mm": float(np.mean(object_errors[object_valid]) * 1000.0),
            "point_p95_mm": float(
                np.percentile(object_errors[object_valid], 95) * 1000.0
            ),
            "point_max_mm": float(np.max(object_errors[object_valid]) * 1000.0),
        }
    return summary


def optimize_global_scale_translation(
    current_scale: float,
    current_palm_from_hand_back: np.ndarray,
    observations: np.ndarray,
    predicted_surface_points: np.ndarray,
    confidence: np.ndarray,
    surface_points_obj: np.ndarray,
) -> tuple[float, np.ndarray, Any]:
    valid_scalar_residuals = (
        np.count_nonzero(np.asarray(confidence) > 0.0)
        * CONTACT_SURFACE_POINT_COUNT
        * 3
    )
    data_normalization = np.sqrt(max(valid_scalar_residuals, 1))

    def residual(parameters: np.ndarray) -> np.ndarray:
        scale = float(parameters[0])
        translation_delta = np.asarray(parameters[1:4], dtype=np.float64)
        palm_from_hand_back = INITIAL_T_PALM_HAND_BACK.copy()
        palm_from_hand_back[:3, 3] += translation_delta
        target_points = compose_target_surface_points(
            palm_from_hand_back,
            observations,
            surface_points_obj,
            scale,
        )
        data_residual = contact_surface_residuals(
            predicted_surface_points,
            target_points,
            confidence,
        ) / data_normalization
        scale_prior = np.asarray(
            [(scale - 1.0) * HUMAN_TO_WUJI_SCALE_PRIOR_WEIGHT],
            dtype=np.float64,
        )
        translation_prior = (
            translation_delta
            / GLOBAL_TRANSLATION_PRIOR_SIGMA_M
            * GLOBAL_OFFSET_PRIOR_WEIGHT
        )
        return np.concatenate(
            [data_residual, scale_prior, translation_prior]
        )

    current_translation_delta = (
        np.asarray(current_palm_from_hand_back, dtype=np.float64)[:3, 3]
        - INITIAL_T_PALM_HAND_BACK[:3, 3]
    )
    initial = np.concatenate(
        [np.asarray([current_scale], dtype=np.float64), current_translation_delta]
    )
    lower = np.asarray(
        [
            HUMAN_TO_WUJI_SCALE_MIN,
            -GLOBAL_TRANSLATION_DELTA_LIMIT_M,
            -GLOBAL_TRANSLATION_DELTA_LIMIT_M,
            -GLOBAL_TRANSLATION_DELTA_LIMIT_M,
        ],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            HUMAN_TO_WUJI_SCALE_MAX,
            GLOBAL_TRANSLATION_DELTA_LIMIT_M,
            GLOBAL_TRANSLATION_DELTA_LIMIT_M,
            GLOBAL_TRANSLATION_DELTA_LIMIT_M,
        ],
        dtype=np.float64,
    )

    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        loss="linear",
        x_scale="jac",
        max_nfev=200,
    )
    optimized_palm_from_hand_back = INITIAL_T_PALM_HAND_BACK.copy()
    optimized_palm_from_hand_back[:3, 3] += result.x[1:4]
    return float(result.x[0]), optimized_palm_from_hand_back, result


def calibrate_initial_qpos_scale(
    urdf: yourdfpy.URDF,
    observations: np.ndarray,
    confidence: np.ndarray,
    surface_points_obj: np.ndarray,
    surface_points_tip: np.ndarray,
    natural_qpos: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    multistarts: int = 5,
) -> tuple[np.ndarray, float]:
    observations = np.asarray(observations, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    active_count = 12
    active_span = np.asarray(upper[:active_count] - lower[:active_count])
    soft_margin = np.maximum(active_span * SOFT_LIMIT_MARGIN_RATIO, 1e-4)
    calibration_rest_weights = np.full(active_count, REST_WEIGHT, dtype=np.float64)
    calibration_rest_weights[: len(THUMB_JOINT_NAMES)] = THUMB_POSTURE_WEIGHT

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, float]:
        qpos = np.asarray(natural_qpos, dtype=np.float64).copy()
        qpos[:active_count] = parameters[1:]
        return qpos, float(parameters[0])

    def evaluate(
        parameters: np.ndarray,
        *,
        return_points: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        qpos, scale = unpack(parameters)
        urdf.update_cfg(qpos)
        predicted_tip = np.stack(
            [
                np.asarray(
                    urdf.get_transform(spec.robot_tip_link, ROOT_LINK_NAME),
                    dtype=np.float64,
                )
                for spec in OBJECT_SPECS
            ],
            axis=0,
        )
        predicted_points = (
            np.einsum(
                "oij,opj->opi",
                predicted_tip[:, :3, :3],
                surface_points_tip,
            )
            + predicted_tip[:, :3, 3][:, None, :]
        )
        target_points = compose_target_surface_points(
            INITIAL_T_PALM_HAND_BACK,
            observations[None, ...],
            surface_points_obj,
            scale,
        )[0]
        residual = contact_surface_residuals(
            predicted_points,
            target_points,
            confidence,
        )
        active_qpos = qpos[:active_count]
        lower_soft_residual = np.maximum(
            0.0,
            lower[:active_count] + soft_margin - active_qpos,
        ) / soft_margin
        upper_soft_residual = np.maximum(
            0.0,
            active_qpos - (upper[:active_count] - soft_margin),
        ) / soft_margin
        residual = np.concatenate(
            [
                residual,
                (active_qpos - natural_qpos[:active_count])
                * calibration_rest_weights,
                np.asarray(
                    [
                        (active_qpos[2] - active_qpos[3])
                        * THUMB_DISTAL_CURVATURE_WEIGHT
                    ],
                    dtype=np.float64,
                ),
                lower_soft_residual * SOFT_LIMIT_WEIGHT,
                upper_soft_residual * SOFT_LIMIT_WEIGHT,
                np.asarray(
                    [(scale - 1.0) * HUMAN_TO_WUJI_SCALE_PRIOR_WEIGHT],
                    dtype=np.float64,
                ),
            ]
        )
        if return_points:
            return qpos, scale, predicted_points, target_points
        return residual

    bounds_lower = np.concatenate(
        [
            np.asarray([HUMAN_TO_WUJI_SCALE_MIN]),
            np.asarray(lower[:active_count], dtype=np.float64),
        ]
    )
    bounds_upper = np.concatenate(
        [
            np.asarray([HUMAN_TO_WUJI_SCALE_MAX]),
            np.asarray(upper[:active_count], dtype=np.float64),
        ]
    )
    rng = np.random.default_rng(20260713)
    best_result = None
    for start_index in range(max(multistarts, 1)):
        q_start = (
            np.asarray(natural_qpos[:active_count], dtype=np.float64)
            if start_index == 0
            else rng.uniform(
                lower[:active_count] + soft_margin,
                upper[:active_count] - soft_margin,
            )
        )
        initial = np.concatenate([np.ones(1), q_start])
        optimization = least_squares(
            evaluate,
            initial,
            bounds=(bounds_lower, bounds_upper),
            loss="linear",
            f_scale=0.2,
            x_scale="jac",
            max_nfev=800,
        )
        if best_result is None or optimization.cost < best_result.cost:
            best_result = optimization
    assert best_result is not None
    qpos, scale, predicted_points, target_points = evaluate(
        best_result.x,
        return_points=True,
    )
    point_errors_mm = surface_point_error_arrays(
        predicted_points, target_points
    ) * 1000.0
    print(
        "[INFO] Fixed-base initial joint calibration: "
        f"cost={best_result.cost:.6f}, nfev={best_result.nfev}, "
        f"surface_point_mean_mm={np.round(np.mean(point_errors_mm, axis=1), 2).tolist()}, "
        f"scale={scale:.6f}"
    )
    return qpos.astype(np.float32), scale


@dataclass
class RetargetResult:
    qpos: np.ndarray
    palm_from_hand_back: np.ndarray
    human_to_wuji_scale: float
    wrist_delta: np.ndarray
    predicted: np.ndarray
    target: np.ndarray
    position_error_m: np.ndarray
    rotation_error_deg: np.ndarray
    predicted_tip: np.ndarray
    target_tip: np.ndarray
    tip_position_error_m: np.ndarray
    tip_line_error_deg: np.ndarray
    predicted_surface_points: np.ndarray
    target_surface_points: np.ndarray
    surface_point_error_m: np.ndarray
    summary: dict[str, Any]


def run_retargeting(
    dataset: PoseDataset,
    urdf: yourdfpy.URDF,
    robot: pk.Robot,
    tip_link_indices: np.ndarray,
    mount_link_indices: np.ndarray,
    link_from_obj: np.ndarray,
    obj_from_tip: np.ndarray,
    surface_points_obj: np.ndarray,
    surface_points_tip: np.ndarray,
    natural_qpos: np.ndarray,
    active_mask: np.ndarray,
    *,
    outer_iterations: int,
    max_solver_iterations: int,
) -> RetargetResult:
    optimization_weights = balanced_retargeting_weights(dataset.confidence)
    contact_line_axes = np.asarray(
        [spec.contact_line_axis for spec in OBJECT_SPECS], dtype=np.float64
    )
    surface_solver = SequentialWujiContactSolver(
        robot,
        tip_link_indices,
        surface_points_tip,
        natural_qpos,
        active_mask,
        max_iterations=max_solver_iterations,
    )
    reference_qpos, human_to_wuji_scale = calibrate_initial_qpos_scale(
        urdf,
        dataset.observations[0],
        optimization_weights[0],
        surface_points_obj,
        surface_points_tip,
        natural_qpos,
        surface_solver.lower,
        surface_solver.upper,
    )
    palm_from_hand_back = INITIAL_T_PALM_HAND_BACK.copy()
    qpos: np.ndarray | None = np.tile(reference_qpos, (len(dataset.observations), 1))
    wrist_delta = np.tile(
        np.eye(4, dtype=np.float64),
        (len(dataset.observations), 1, 1),
    )
    for iteration in range(max(int(outer_iterations), 1)):
        target_surface_points = compose_target_surface_points(
            palm_from_hand_back,
            dataset.observations,
            surface_points_obj,
            human_to_wuji_scale,
        )
        qpos = surface_solver.solve_sequence(
            target_surface_points,
            optimization_weights,
            dataset.timestamps,
            qpos,
            progress_label=f"Contact-surface IK {iteration + 1}/{outer_iterations}",
        )
        predicted_surface_points = compute_predicted_surface_points(
            robot, qpos, tip_link_indices, surface_points_tip
        )
        before = summarize_surface_point_errors(
            predicted_surface_points,
            target_surface_points,
            optimization_weights,
        )
        (
            human_to_wuji_scale,
            palm_from_hand_back,
            global_result,
        ) = optimize_global_scale_translation(
            human_to_wuji_scale,
            palm_from_hand_back,
            dataset.observations,
            predicted_surface_points,
            optimization_weights,
            surface_points_obj,
        )
        updated_target_surface_points = compose_target_surface_points(
            palm_from_hand_back,
            dataset.observations,
            surface_points_obj,
            human_to_wuji_scale,
        )
        after = summarize_surface_point_errors(
            predicted_surface_points,
            updated_target_surface_points,
            optimization_weights,
        )
        print(
            f"[INFO] Surface stage {iteration + 1}: mean point error "
            f"{before['point_mm']['mean']:.2f} -> "
            f"{after['point_mm']['mean']:.2f} mm, "
            f"scale={human_to_wuji_scale:.6f}, "
            "translation_delta_mm="
            f"{np.round((palm_from_hand_back[:3, 3] - INITIAL_T_PALM_HAND_BACK[:3, 3]) * 1000.0, 3).tolist()}, "
            f"global_nfev={global_result.nfev}"
        )

    final_target_surface_points = compose_target_surface_points(
        palm_from_hand_back,
        dataset.observations,
        surface_points_obj,
        human_to_wuji_scale,
    )
    qpos = surface_solver.solve_sequence(
        final_target_surface_points,
        optimization_weights,
        dataset.timestamps,
        qpos,
        progress_label="Final contact-surface IK",
    )
    predicted_surface_points = compute_predicted_surface_points(
        robot, qpos, tip_link_indices, surface_points_tip
    )
    surface_point_error_m = surface_point_error_arrays(
        predicted_surface_points, final_target_surface_points
    )
    final_targets_tip = compose_target_tip_poses(
        palm_from_hand_back,
        dataset.observations,
        obj_from_tip,
        human_to_wuji_scale,
    )
    predicted_tip = compute_predicted_link_poses(robot, qpos, tip_link_indices)
    predicted = compute_predicted_obj_poses(
        robot, qpos, mount_link_indices, link_from_obj
    )
    final_targets = compose_target_poses(
        palm_from_hand_back,
        dataset.observations,
        human_to_wuji_scale,
    )
    position_error_m, rotation_error_deg = pose_error_arrays(predicted, final_targets)
    tip_position_error_m, tip_line_error_deg = tip_error_arrays(
        predicted_tip, final_targets_tip, contact_line_axes
    )
    offset_delta_translation = (
        palm_from_hand_back[:3, 3] - INITIAL_T_PALM_HAND_BACK[:3, 3]
    )
    offset_delta_rotation = Rotation.from_matrix(
        palm_from_hand_back[:3, :3] @ INITIAL_T_PALM_HAND_BACK[:3, :3].T
    )
    summary = {
        "retargeting_objective": "three_corresponding_points_on_each_obj_contact_surface",
        "weighting": "per-frame confidence normalized to equal mean weight per finger",
        "human_to_wuji_scale": float(human_to_wuji_scale),
        "dynamic_wrist_enabled": False,
        "global_offset": {
            "initial_T_left_palm_link_hand_back_cube": INITIAL_T_PALM_HAND_BACK.tolist(),
            "translation_delta_mm": (offset_delta_translation * 1000.0).tolist(),
            "translation_delta_norm_mm": float(
                np.linalg.norm(offset_delta_translation) * 1000.0
            ),
            "rotation_delta_deg": float(np.rad2deg(offset_delta_rotation.magnitude())),
            "translation_component_limit_mm": float(
                GLOBAL_TRANSLATION_DELTA_LIMIT_M * 1000.0
            ),
            "translation_prior_sigma_mm": float(
                GLOBAL_TRANSLATION_PRIOR_SIGMA_M * 1000.0
            ),
            "translation_prior_weight": GLOBAL_OFFSET_PRIOR_WEIGHT,
            "rotation_component_limit_deg": 0.0,
            "optimized": True,
            "translation_optimized": True,
            "rotation_optimized": False,
            "fixed_from_prior": False,
            "time_varying": False,
        },
        "contact_surface_points_obj_mm": {
            spec.name: (surface_points_obj[index] * 1000.0).tolist()
            for index, spec in enumerate(OBJECT_SPECS)
        },
        "object_pose": summarize_errors(
            predicted,
            final_targets,
            dataset.confidence,
        ),
        "fingertip": summarize_tip_errors(
            predicted_tip,
            final_targets_tip,
            dataset.confidence,
            contact_line_axes,
        ),
        "contact_surface": summarize_surface_point_errors(
            predicted_surface_points,
            final_target_surface_points,
            dataset.confidence,
        ),
    }
    lower = np.asarray(robot.joints.lower_limits)
    upper = np.asarray(robot.joints.upper_limits)
    summary["joint_limits"] = {
        "minimum_margin_rad": float(np.min(np.minimum(qpos - lower, upper - qpos))),
        "violation_count": int(np.sum((qpos < lower - 1e-6) | (qpos > upper + 1e-6))),
    }
    if len(qpos) > 1:
        dt = np.clip(np.diff(dataset.timestamps), 1e-3, 0.5)[:, None]
        velocity_ratio = (
            np.abs(np.diff(qpos, axis=0))
            / dt
            / np.asarray(robot.joints.velocity_limits)[None, :]
        )
        summary["velocity_limits"] = {
            "maximum_ratio": float(np.max(velocity_ratio)),
            "violation_count": int(np.sum(velocity_ratio > 1.0 + 1e-5)),
            "optimization_guard_scale": VELOCITY_LIMIT_SCALE,
            "note": "The sequence is clipped to the URDF velocity limits at recorded timestamps",
        }
    summary["dynamic_wrist"] = {
        "enabled": False,
        "translation_mean_mm": 0.0,
        "translation_p95_mm": 0.0,
        "rotation_mean_deg": 0.0,
        "rotation_p95_deg": 0.0,
    }
    return RetargetResult(
        qpos=qpos,
        palm_from_hand_back=palm_from_hand_back,
        human_to_wuji_scale=human_to_wuji_scale,
        wrist_delta=wrist_delta,
        predicted=predicted,
        target=final_targets,
        position_error_m=position_error_m,
        rotation_error_deg=rotation_error_deg,
        predicted_tip=predicted_tip,
        target_tip=final_targets_tip,
        tip_position_error_m=tip_position_error_m,
        tip_line_error_deg=tip_line_error_deg,
        predicted_surface_points=predicted_surface_points,
        target_surface_points=final_target_surface_points,
        surface_point_error_m=surface_point_error_m,
        summary=summary,
    )


def apply_retargeting_to_pkl(
    pkl_path: Path,
    result: RetargetResult,
    urdf_path: Path,
    joint_names: list[str],
    active_joint_names: list[str],
) -> None:
    ensure_atomic_rewrite_space(pkl_path)
    temporary = pkl_path.with_suffix(pkl_path.suffix + ".028-retarget-rewrite.tmp")
    if temporary.exists():
        temporary.unlink()
    total_bytes = pkl_path.stat().st_size
    frame_index = 0
    last_progress = 0.0
    metadata = {
        "algorithm": RETARGET_ALGORITHM,
        "urdf_path": str(urdf_path),
        "root_link_name": ROOT_LINK_NAME,
        "frame_convention": "A_T_B maps coordinates from frame B into frame A",
        "translation_unit": "m",
        "joint_unit": "rad",
        "joint_names": joint_names,
        "active_joint_names": active_joint_names,
        "T_left_palm_link_hand_back_cube": result.palm_from_hand_back,
        "human_to_wuji_scale": result.human_to_wuji_scale,
        "dynamic_wrist_enabled": result.summary["dynamic_wrist"]["enabled"],
        "initial_T_left_palm_link_hand_back_cube": INITIAL_T_PALM_HAND_BACK,
        "summary": result.summary,
    }
    try:
        with pkl_path.open("rb") as source, temporary.open("wb") as destination:
            header = pickle.load(source)
            header.setdefault("metadata", {})[RETARGET_FIELD] = metadata
            pickle.dump(header, destination, protocol=pickle.HIGHEST_PROTOCOL)
            while True:
                try:
                    record = pickle.load(source)
                except EOFError:
                    break
                if isinstance(record, dict) and record.get("type") == "frame_pair":
                    if frame_index >= len(result.qpos):
                        raise RuntimeError("PKL has more frames than the retarget result")
                    per_object = {
                        spec.name: {
                            "object_origin_error_mm": float(
                                result.position_error_m[frame_index, object_index] * 1000.0
                            ),
                            "object_rotation_error_deg": float(
                                result.rotation_error_deg[frame_index, object_index]
                            ),
                            "fingertip_position_error_mm": float(
                                result.tip_position_error_m[frame_index, object_index]
                                * 1000.0
                            ),
                            "fingertip_line_error_deg": float(
                                result.tip_line_error_deg[frame_index, object_index]
                            ),
                            "contact_surface_point_errors_mm": (
                                result.surface_point_error_m[
                                    frame_index, object_index
                                ]
                                * 1000.0
                            ).tolist(),
                            "contact_surface_point_mean_error_mm": float(
                                np.mean(
                                    result.surface_point_error_m[
                                        frame_index, object_index
                                    ]
                                )
                                * 1000.0
                            ),
                        }
                        for object_index, spec in enumerate(OBJECT_SPECS)
                    }
                    record[RETARGET_FIELD] = {
                        "algorithm": RETARGET_ALGORITHM,
                        "root_link_name": ROOT_LINK_NAME,
                        "qpos": result.qpos[frame_index].astype(np.float32),
                        "T_left_palm_reference_left_palm_dynamic": result.wrist_delta[
                            frame_index
                        ],
                        "T_left_palm_link_hand_back_cube": result.palm_from_hand_back,
                        "human_to_wuji_scale": result.human_to_wuji_scale,
                        "metrics": {"per_object": per_object},
                    }
                    frame_index += 1
                pickle.dump(record, destination, protocol=pickle.HIGHEST_PROTOCOL)
                now = time.monotonic()
                if now - last_progress >= 0.5:
                    print_rewrite_progress("Embedding Wuji qpos", source.tell(), total_bytes)
                    last_progress = now
            destination.flush()
            os.fsync(destination.fileno())
        if frame_index != len(result.qpos):
            raise RuntimeError(
                f"Rewrote {frame_index} frames, expected {len(result.qpos)}"
            )
        temporary.replace(pkl_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    print_rewrite_progress("Embedding Wuji qpos", total_bytes, total_bytes, finish=True)
    print(f"[INFO] Embedded Wuji qpos in {frame_index} frames")


def result_from_embedded(
    dataset: PoseDataset,
    robot: pk.Robot,
    tip_link_indices: np.ndarray,
    mount_link_indices: np.ndarray,
    link_from_obj: np.ndarray,
    obj_from_tip: np.ndarray,
    surface_points_obj: np.ndarray,
    surface_points_tip: np.ndarray,
) -> RetargetResult:
    if (
        dataset.embedded_qpos is None
        or dataset.embedded_offset is None
        or dataset.embedded_scale is None
        or dataset.embedded_wrist_delta is None
    ):
        raise ValueError("Embedded Wuji retargeting is incomplete")
    target = compose_target_poses(
        dataset.embedded_offset,
        dataset.observations,
        dataset.embedded_scale,
    )
    predicted_local = compute_predicted_obj_poses(
        robot,
        dataset.embedded_qpos,
        mount_link_indices,
        link_from_obj,
    )
    predicted = apply_wrist_delta(dataset.embedded_wrist_delta, predicted_local)
    target_tip = compose_target_tip_poses(
        dataset.embedded_offset,
        dataset.observations,
        obj_from_tip,
        dataset.embedded_scale,
    )
    predicted_tip_local = compute_predicted_link_poses(
        robot,
        dataset.embedded_qpos,
        tip_link_indices,
    )
    predicted_tip = apply_wrist_delta(
        dataset.embedded_wrist_delta,
        predicted_tip_local,
    )
    position_error_m, rotation_error_deg = pose_error_arrays(predicted, target)
    contact_line_axes = np.asarray(
        [spec.contact_line_axis for spec in OBJECT_SPECS], dtype=np.float64
    )
    tip_position_error_m, tip_line_error_deg = tip_error_arrays(
        predicted_tip,
        target_tip,
        contact_line_axes,
    )
    target_surface_points = compose_target_surface_points(
        dataset.embedded_offset,
        dataset.observations,
        surface_points_obj,
        dataset.embedded_scale,
    )
    predicted_surface_points = (
        np.einsum(
            "...oij,opj->...opi",
            predicted_tip[..., :3, :3],
            surface_points_tip,
        )
        + predicted_tip[..., :3, 3][..., :, None, :]
    )
    surface_point_error_m = surface_point_error_arrays(
        predicted_surface_points, target_surface_points
    )
    summary = dataset.header.get("metadata", {}).get(RETARGET_FIELD, {}).get("summary")
    if not isinstance(summary, dict):
        summary = {
            "retargeting_objective": "three_corresponding_points_on_each_obj_contact_surface",
            "human_to_wuji_scale": dataset.embedded_scale,
            "object_pose": summarize_errors(
                predicted,
                target,
                dataset.confidence,
            ),
            "fingertip": summarize_tip_errors(
                predicted_tip,
                target_tip,
                dataset.confidence,
                contact_line_axes,
            ),
            "contact_surface": summarize_surface_point_errors(
                predicted_surface_points,
                target_surface_points,
                dataset.confidence,
            ),
        }
    return RetargetResult(
        qpos=dataset.embedded_qpos,
        palm_from_hand_back=dataset.embedded_offset,
        human_to_wuji_scale=dataset.embedded_scale,
        wrist_delta=dataset.embedded_wrist_delta,
        predicted=predicted,
        target=target,
        position_error_m=position_error_m,
        rotation_error_deg=rotation_error_deg,
        predicted_tip=predicted_tip,
        target_tip=target_tip,
        tip_position_error_m=tip_position_error_m,
        tip_line_error_deg=tip_line_error_deg,
        predicted_surface_points=predicted_surface_points,
        target_surface_points=target_surface_points,
        surface_point_error_m=surface_point_error_m,
        summary=summary,
    )


def save_result_artifacts(
    output_dir: Path,
    pkl_path: Path,
    result: RetargetResult,
    joint_names: list[str],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pkl_path.stem
    npz_path = output_dir / f"{stem}_{RETARGET_ALGORITHM}.npz"
    json_path = output_dir / f"{stem}_{RETARGET_ALGORITHM}.json"
    np.savez_compressed(
        npz_path,
        qpos=result.qpos.astype(np.float32),
        joint_names=np.asarray(joint_names),
        T_left_palm_link_hand_back_cube=result.palm_from_hand_back,
        human_to_wuji_scale=np.asarray(result.human_to_wuji_scale),
        wrist_delta=result.wrist_delta,
        target_poses=result.target,
        predicted_poses=result.predicted,
        position_error_m=result.position_error_m,
        rotation_error_deg=result.rotation_error_deg,
        target_tip_poses=result.target_tip,
        predicted_tip_poses=result.predicted_tip,
        tip_position_error_m=result.tip_position_error_m,
        tip_line_error_deg=result.tip_line_error_deg,
        target_surface_points=result.target_surface_points,
        predicted_surface_points=result.predicted_surface_points,
        surface_point_error_m=result.surface_point_error_m,
    )
    json_path.write_text(
        json.dumps(
            {
                "algorithm": RETARGET_ALGORITHM,
                "joint_names": joint_names,
                "T_left_palm_link_hand_back_cube": result.palm_from_hand_back.tolist(),
                "human_to_wuji_scale": result.human_to_wuji_scale,
                "dynamic_wrist": result.summary["dynamic_wrist"],
                "summary": result.summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return npz_path, json_path


def load_colored_mesh(path: Path, color: tuple[int, int, int]) -> trimesh.Trimesh:
    mesh = load_obj_mesh_m(path)
    rgba = np.asarray([*color, 175], dtype=np.uint8)
    mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))
    return mesh


def transform_from_handle(handle: Any) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = wxyz_to_matrix(np.asarray(handle.wxyz))
    transform[:3, 3] = np.asarray(handle.position, dtype=np.float64)
    return transform


def format_frame_metadata(
    frame_index: int,
    result: RetargetResult,
    joint_names: list[str],
    current_offset: np.ndarray,
    observations: np.ndarray,
    predicted_flag: np.ndarray,
) -> str:
    current_target = compose_target_poses(
        current_offset,
        observations[frame_index : frame_index + 1],
        result.human_to_wuji_scale,
    )[0]
    position_m, rotation_deg = pose_error_arrays(
        result.predicted[frame_index : frame_index + 1],
        current_target[None, ...],
    )
    lines = [f"**Frame:** `{frame_index}`", "", "**Contact-surface / object residuals**"]
    for object_index, spec in enumerate(OBJECT_SPECS):
        source = "predicted pose" if predicted_flag[frame_index, object_index] else "measured pose"
        surface_errors_mm = result.surface_point_error_m[
            frame_index, object_index
        ] * 1000.0
        lines.append(
            f"- `{spec.name}`: surface points "
            f"{np.round(surface_errors_mm, 2).tolist()} mm "
            f"(mean {np.mean(surface_errors_mm):.2f} mm); obj "
            f"{position_m[0, object_index] * 1000.0:.2f} mm / "
            f"{rotation_deg[0, object_index]:.2f} deg, {source}"
        )
    lines.extend(["", "**Active qpos (rad)**"])
    for joint_index, joint_name in enumerate(joint_names[:12]):
        lines.append(f"- `{joint_name}`: {result.qpos[frame_index, joint_index]:+.4f}")
    return "\n".join(lines)


def run_viser(
    host: str,
    port: int,
    display_mode: str,
    urdf: yourdfpy.URDF,
    dataset: PoseDataset,
    result: RetargetResult,
    joint_names: list[str],
) -> None:
    server = viser.ViserServer(host=host, port=port)
    print(f"[INFO] Viser: http://localhost:{server.get_port()}")
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = False
    server.initial_camera.position = (0.21, -0.24, 0.24)
    server.initial_camera.look_at = (0.035, -0.025, 0.145)
    server.initial_camera.up = (0.0, 0.0, 1.0)
    server.initial_camera.near = 0.001
    server.gui.set_panel_label(f"028 Wuji Contact Retargeting ({display_mode})")

    server.scene.add_frame(
        "/left_palm_link",
        axes_length=0.08,
        axes_radius=0.0025,
        origin_radius=0.004,
    )
    server.scene.add_grid(
        "/left_palm_link/grid",
        width=0.5,
        height=0.5,
        plane="xy",
        cell_size=0.02,
        section_size=0.1,
    )
    wuji_root = server.scene.add_frame("/wuji", show_axes=False)
    urdf_vis = ViserUrdf(
        server,
        urdf,
        root_node_name="/wuji",
        mesh_color_override=(0.79, 0.82, 0.93, 0.50),
    )
    thumb_centerline = server.scene.add_line_segments(
        "/wuji/diagnostics/thumb_centerline",
        points=np.zeros((len(THUMB_CENTERLINE_LINK_NAMES) - 1, 2, 3), dtype=np.float32),
        colors=(25, 185, 75),
        line_width=8.0,
    )

    offset_control = server.scene.add_transform_controls(
        "/targets/hand_back_cube",
        scale=0.08,
        position=result.palm_from_hand_back[:3, 3],
        wxyz=matrix_to_wxyz(result.palm_from_hand_back[:3, :3]),
    )
    server.scene.add_box(
        "/targets/hand_back_cube/reference_cube",
        dimensions=(0.0625, 0.0625, 0.0625),
        color=(85, 115, 255),
        opacity=0.12,
        side="double",
    )

    target_handles: dict[str, dict[str, Any]] = {}
    predicted_frames: dict[str, Any] = {}
    predicted_tip_frames: dict[str, Any] = {}
    target_tip_frames: dict[str, Any] = {}
    predicted_contact_points: dict[str, list[Any]] = {}
    target_contact_points: dict[str, list[Any]] = {}
    predicted_surface_edges: dict[str, Any] = {}
    target_surface_edges: dict[str, Any] = {}
    error_lines: dict[str, Any] = {}
    for spec in OBJECT_SPECS:
        root = f"/targets/hand_back_cube/objects/{spec.name}"
        frame = server.scene.add_frame(
            root,
            show_axes=False,
        )
        axes = server.scene.add_frame(
            f"{root}/axes",
            axes_length=0.035,
            axes_radius=0.0012,
        )
        mesh = server.scene.add_mesh_trimesh(
            f"{root}/{spec.name}_target_obj",
            load_colored_mesh(spec.mesh_path, spec.color),
            scale=1.0,
            cast_shadow=False,
            receive_shadow=False,
        )
        box = server.scene.add_box(
            f"{root}/cube_frame_box",
            dimensions=(0.01875, 0.01875, 0.01875),
            color=spec.color,
            opacity=0.18,
            side="double",
        )
        target_handles[spec.name] = {
            "frame": frame,
            "axes": axes,
            "mesh": mesh,
            "box": box,
        }
        predicted_frames[spec.name] = server.scene.add_frame(
            f"/predicted_obj_frames/{spec.name}",
            axes_length=0.028,
            axes_radius=0.001,
        )
        predicted_tip_frames[spec.name] = server.scene.add_frame(
            f"/predicted_tip_frames/{spec.name}",
            show_axes=False,
        )
        target_tip_frames[spec.name] = server.scene.add_frame(
            f"/target_tip_frames/{spec.name}",
            show_axes=False,
        )
        target_contact_points[spec.name] = [
            server.scene.add_icosphere(
                f"/contact_surface/target/{spec.name}/point_{point_index}",
                radius=0.0025,
                color=spec.color,
                subdivisions=2,
            )
            for point_index in range(CONTACT_SURFACE_POINT_COUNT)
        ]
        predicted_contact_points[spec.name] = [
            server.scene.add_icosphere(
                f"/contact_surface/predicted/{spec.name}/point_{point_index}",
                radius=0.0017,
                color=(20, 20, 20),
                subdivisions=2,
            )
            for point_index in range(CONTACT_SURFACE_POINT_COUNT)
        ]
        target_surface_edges[spec.name] = server.scene.add_line_segments(
            f"/contact_surface/target/{spec.name}/triangle",
            points=np.zeros((3, 2, 3), dtype=np.float32),
            colors=spec.color,
            line_width=6.0,
        )
        predicted_surface_edges[spec.name] = server.scene.add_line_segments(
            f"/contact_surface/predicted/{spec.name}/triangle",
            points=np.zeros((3, 2, 3), dtype=np.float32),
            colors=(25, 25, 25),
            line_width=9.0,
        )
        error_lines[spec.name] = server.scene.add_line_segments(
            f"/errors/{spec.name}",
            points=np.zeros((CONTACT_SURFACE_POINT_COUNT, 2, 3), dtype=np.float32),
            colors=spec.color,
            line_width=3.0,
        )

    frame_count = len(result.qpos)
    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=frame_count - 1,
            step=1,
            initial_value=0,
        )
        autoplay = server.gui.add_checkbox("Auto play", initial_value=False)
        loop = server.gui.add_checkbox("Loop", initial_value=True)
        status = server.gui.add_text("Status", initial_value=f"1/{frame_count}", disabled=True)
    with server.gui.add_folder("Visibility"):
        show_target_mesh = server.gui.add_checkbox(
            "Target OBJ", initial_value=False
        )
        show_target_box = server.gui.add_checkbox("Target cube box", initial_value=False)
        show_target_axes = server.gui.add_checkbox("Target axes", initial_value=False)
        show_robot_axes = server.gui.add_checkbox("Robot OBJ axes", initial_value=False)
        show_contact_points = server.gui.add_checkbox(
            "Surface points", initial_value=display_mode in {"point", "point-line"}
        )
        show_contact_lines = server.gui.add_checkbox(
            "Surface triangles", initial_value=display_mode in {"line", "point-line"}
        )
        show_error = server.gui.add_checkbox(
            "Three-point residuals", initial_value=True
        )
        show_thumb_centerline = server.gui.add_checkbox(
            "Thumb centerline", initial_value=True
        )
    with server.gui.add_folder("Fixed Base Offset"):
        reset_offset = server.gui.add_button("Reset to fixed prior")
        offset_text = server.gui.add_markdown("")
    with server.gui.add_folder("Frame Metrics"):
        frame_text = server.gui.add_markdown("")
    with server.gui.add_folder("Sequence Summary"):
        server.gui.add_markdown(
            "\n".join(
                [
                    f"**Algorithm:** `{RETARGET_ALGORITHM}`",
                    f"**Frames:** `{frame_count}`",
                    f"**Human to Wuji scale:** `{result.human_to_wuji_scale:.6f}`",
                    f"**Surface point mean:** `{result.summary['contact_surface']['point_mm']['mean']:.2f} mm`",
                    f"**Surface point p95:** `{result.summary['contact_surface']['point_mm']['p95']:.2f} mm`",
                    f"**Offset delta:** `{result.summary['global_offset']['translation_delta_norm_mm']:.2f} mm / "
                    f"{result.summary['global_offset']['rotation_delta_deg']:.2f} deg`",
                    f"**OBJ position mean:** `{result.summary['object_pose']['position_mm']['mean']:.2f} mm`",
                    f"**OBJ position p95:** `{result.summary['object_pose']['position_mm']['p95']:.2f} mm`",
                    f"**OBJ rotation mean:** `{result.summary['object_pose']['rotation_deg']['mean']:.2f} deg`",
                    f"**OBJ rotation p95:** `{result.summary['object_pose']['rotation_deg']['p95']:.2f} deg`",
                    f"**Joint-limit violations:** `{result.summary.get('joint_limits', {}).get('violation_count', 0)}`",
                    f"**Velocity-limit violations:** `{result.summary.get('velocity_limits', {}).get('violation_count', 0)}`",
                    f"**Wrist translation mean:** `{result.summary['dynamic_wrist']['translation_mean_mm']:.2f} mm`",
                    f"**Wrist rotation mean:** `{result.summary['dynamic_wrist']['rotation_mean_deg']:.2f} deg`",
                ]
            )
        )

    @reset_offset.on_click
    def _(_event: Any) -> None:
        offset_control.position = result.palm_from_hand_back[:3, 3]
        offset_control.wxyz = matrix_to_wxyz(result.palm_from_hand_back[:3, :3])

    rendered_frame = -1
    last_offset = np.full((4, 4), np.nan)
    last_playback_step = time.monotonic()

    def render(frame_index: int, current_offset: np.ndarray) -> None:
        nonlocal rendered_frame, last_offset
        wrist = result.wrist_delta[frame_index]
        wuji_root.position = wrist[:3, 3]
        wuji_root.wxyz = matrix_to_wxyz(wrist[:3, :3])
        urdf_vis.update_cfg(result.qpos[frame_index])
        urdf.update_cfg(result.qpos[frame_index])
        thumb_centerline_points = np.stack(
            [
                np.asarray(
                    urdf.get_transform(link_name, ROOT_LINK_NAME),
                    dtype=np.float64,
                )[:3, 3]
                for link_name in THUMB_CENTERLINE_LINK_NAMES
            ],
            axis=0,
        )
        thumb_centerline.points = np.asarray(
            np.stack(
                [thumb_centerline_points[:-1], thumb_centerline_points[1:]],
                axis=1,
            ),
            dtype=np.float32,
        )
        for object_index, spec in enumerate(OBJECT_SPECS):
            observed = dataset.observations[frame_index, object_index]
            target_handles[spec.name]["frame"].position = (
                observed[:3, 3] * result.human_to_wuji_scale
            )
            target_handles[spec.name]["frame"].wxyz = matrix_to_wxyz(observed[:3, :3])
            predicted_pose = result.predicted[frame_index, object_index]
            predicted_frames[spec.name].position = predicted_pose[:3, 3]
            predicted_frames[spec.name].wxyz = matrix_to_wxyz(predicted_pose[:3, :3])
            predicted_tip = result.predicted_tip[frame_index, object_index]
            predicted_tip_frames[spec.name].position = predicted_tip[:3, 3]
            predicted_tip_frames[spec.name].wxyz = matrix_to_wxyz(predicted_tip[:3, :3])
            target_tip = (
                current_offset
                @ np.linalg.inv(result.palm_from_hand_back)
                @ result.target_tip[frame_index, object_index]
            )
            target_tip_frames[spec.name].position = target_tip[:3, 3]
            target_tip_frames[spec.name].wxyz = matrix_to_wxyz(target_tip[:3, :3])
            offset_adjustment = current_offset @ np.linalg.inv(
                result.palm_from_hand_back
            )
            target_points = transform_points(
                offset_adjustment,
                result.target_surface_points[frame_index, object_index],
            )
            predicted_points = result.predicted_surface_points[
                frame_index, object_index
            ]
            for point_index in range(CONTACT_SURFACE_POINT_COUNT):
                target_contact_points[spec.name][point_index].position = target_points[
                    point_index
                ]
                predicted_contact_points[spec.name][point_index].position = (
                    predicted_points[point_index]
                )
            triangle_indices = np.asarray([[0, 1], [1, 2], [2, 0]])
            target_surface_edges[spec.name].points = np.asarray(
                target_points[triangle_indices], dtype=np.float32
            )
            predicted_surface_edges[spec.name].points = np.asarray(
                predicted_points[triangle_indices], dtype=np.float32
            )
            error_lines[spec.name].points = np.asarray(
                np.stack([predicted_points, target_points], axis=1),
                dtype=np.float32,
            )
        frame_text.content = format_frame_metadata(
            frame_index,
            result,
            joint_names,
            current_offset,
            dataset.observations,
            dataset.predicted,
        )
        offset_text.content = (
            f"**Human to Wuji scale:** `{result.human_to_wuji_scale:.6f}`\n\n"
            "**T_left_palm_link_hand_back_cube**\n\n```text\n"
            + np.array2string(current_offset, precision=6, suppress_small=True)
            + "\n```"
        )
        status.value = f"{frame_index + 1}/{frame_count}"
        rendered_frame = frame_index
        last_offset = current_offset.copy()

    while True:
        current_offset = transform_from_handle(offset_control)
        selected_frame = int(frame_slider.value)
        if selected_frame != rendered_frame or not np.allclose(current_offset, last_offset):
            render(selected_frame, current_offset)
        for handles in target_handles.values():
            handles["mesh"].visible = bool(show_target_mesh.value)
            handles["box"].visible = bool(show_target_box.value)
            handles["axes"].visible = bool(show_target_axes.value)
        thumb_centerline.visible = bool(show_thumb_centerline.value)
        for handle in predicted_frames.values():
            handle.visible = bool(show_robot_axes.value)
        for handles in predicted_contact_points.values():
            for handle in handles:
                handle.visible = bool(show_contact_points.value)
        for handles in target_contact_points.values():
            for handle in handles:
                handle.visible = bool(show_contact_points.value)
        for handle in predicted_surface_edges.values():
            handle.visible = bool(show_contact_lines.value)
        for handle in target_surface_edges.values():
            handle.visible = bool(show_contact_lines.value)
        for handle in error_lines.values():
            handle.visible = bool(show_error.value)

        if bool(autoplay.value):
            now = time.monotonic()
            if now - last_playback_step >= 1.0 / AUTO_PLAY_FPS:
                next_frame = rendered_frame + 1
                if next_frame >= frame_count:
                    if bool(loop.value):
                        next_frame = 0
                    else:
                        next_frame = frame_count - 1
                        autoplay.value = False
                frame_slider.value = next_frame
                last_playback_step = now
        else:
            last_playback_step = time.monotonic()
        time.sleep(0.01)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Embed hand-back-frame OBJ poses, retarget thumb/index/middle to the "
            "left WujiHand, and visualize the result in Viser."
        )
    )
    parser.add_argument("pkl_path", nargs="?", type=Path, default=DEFAULT_PKL_PATH)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--middle-extrinsics", type=Path, default=DEFAULT_MIDDLE_EXTRINSICS)
    parser.add_argument("--thumb-extrinsics", type=Path, default=DEFAULT_THUMB_EXTRINSICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--host", default=VISER_HOST)
    parser.add_argument("--port", type=int, default=VISER_PORT)
    parser.add_argument(
        "--viser-mode",
        choices=("point", "line", "point-line"),
        default="point-line",
        help="Initial contact geometry shown in Viser.",
    )
    parser.add_argument("--outer-iterations", type=int, default=10)
    parser.add_argument("--solver-iterations", type=int, default=60)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--force-world-pose-rewrite", action="store_true")
    parser.add_argument("--embed-world-poses-only", action="store_true")
    parser.add_argument("--force-optimize", action="store_true")
    parser.add_argument(
        "--apply",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Atomically write qpos and retarget metrics into the original PKL.",
    )
    parser.add_argument(
        "--viser",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pkl_path = args.pkl_path.expanduser().resolve()
    urdf_path = args.urdf.expanduser().resolve()
    middle_extrinsics = args.middle_extrinsics.expanduser().resolve()
    thumb_extrinsics = args.thumb_extrinsics.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (pkl_path, urdf_path, middle_extrinsics, thumb_extrinsics):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.max_frames is not None and args.apply:
        raise ValueError("--max-frames requires --no-apply")
    world_from_camera = load_world_from_camera(middle_extrinsics, thumb_extrinsics)
    embed_world_obj_poses(
        pkl_path,
        world_from_camera,
        middle_extrinsics,
        thumb_extrinsics,
        force=bool(args.force_world_pose_rewrite),
    )
    if args.embed_world_poses_only:
        return

    dataset = load_pose_dataset(pkl_path, max_frames=args.max_frames)
    urdf = load_urdf(urdf_path)
    link_from_obj_map = load_link_from_obj_transforms(urdf_path)
    obj_from_tip_map = load_obj_from_tip_transforms(urdf_path, link_from_obj_map)
    link_from_obj = np.stack(
        [link_from_obj_map[spec.name] for spec in OBJECT_SPECS],
        axis=0,
    )
    obj_from_tip = np.stack(
        [obj_from_tip_map[spec.name] for spec in OBJECT_SPECS],
        axis=0,
    )
    surface_points_obj_map = load_contact_surface_points_obj(obj_from_tip_map)
    surface_points_obj = np.stack(
        [surface_points_obj_map[spec.name] for spec in OBJECT_SPECS],
        axis=0,
    )
    surface_points_tip = np.stack(
        [
            transform_points(
                np.linalg.inv(obj_from_tip_map[spec.name]),
                surface_points_obj_map[spec.name],
            )
            for spec in OBJECT_SPECS
        ],
        axis=0,
    )
    lower = np.asarray(
        [float(urdf.joint_map[name].limit.lower) for name in urdf.actuated_joint_names],
        dtype=np.float32,
    )
    upper = np.asarray(
        [float(urdf.joint_map[name].limit.upper) for name in urdf.actuated_joint_names],
        dtype=np.float32,
    )
    retarget_lower, retarget_upper = apply_retarget_posture_bounds(
        tuple(urdf.actuated_joint_names), lower, upper
    )
    natural_qpos = np.zeros(len(urdf.actuated_joint_names), dtype=np.float32)
    for joint_name, value in zip(THUMB_JOINT_NAMES, THUMB_NATURAL_QPOS):
        natural_qpos[urdf.actuated_joint_names.index(joint_name)] = value
    natural_qpos = np.clip(natural_qpos, retarget_lower, retarget_upper)
    active_mask = np.asarray(
        [
            name.startswith("left_finger1_")
            or name.startswith("left_finger2_")
            or name.startswith("left_finger3_")
            for name in urdf.actuated_joint_names
        ],
        dtype=bool,
    )
    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=jnp.asarray(natural_qpos))
    mount_link_indices = np.asarray(
        [robot.links.names.index(spec.robot_link) for spec in OBJECT_SPECS],
        dtype=np.int32,
    )
    tip_link_indices = np.asarray(
        [robot.links.names.index(spec.robot_tip_link) for spec in OBJECT_SPECS],
        dtype=np.int32,
    )

    if dataset.embedded_qpos is not None and not args.force_optimize:
        print(f"[INFO] Reusing embedded {RETARGET_ALGORITHM} qpos")
        result = result_from_embedded(
            dataset,
            robot,
            tip_link_indices,
            mount_link_indices,
            link_from_obj,
            obj_from_tip,
            surface_points_obj,
            surface_points_tip,
        )
    else:
        result = run_retargeting(
            dataset,
            urdf,
            robot,
            tip_link_indices,
            mount_link_indices,
            link_from_obj,
            obj_from_tip,
            surface_points_obj,
            surface_points_tip,
            natural_qpos,
            active_mask,
            outer_iterations=max(int(args.outer_iterations), 1),
            max_solver_iterations=max(int(args.solver_iterations), 1),
        )

    joint_names = list(urdf.actuated_joint_names)
    active_joint_names = [
        name for name, active in zip(joint_names, active_mask.tolist()) if active
    ]
    npz_path, json_path = save_result_artifacts(
        output_dir,
        pkl_path,
        result,
        joint_names,
    )
    print(json.dumps(result.summary, indent=2))
    print("[INFO] T_left_palm_link_hand_back_cube:")
    print(np.array2string(result.palm_from_hand_back, precision=8, suppress_small=True))
    print(f"[INFO] Human-to-Wuji isotropic scale: {result.human_to_wuji_scale:.8f}")
    print(f"[INFO] Result NPZ: {npz_path}")
    print(f"[INFO] Report JSON: {json_path}")

    if args.apply and (dataset.embedded_qpos is None or args.force_optimize):
        apply_retargeting_to_pkl(
            pkl_path,
            result,
            urdf_path,
            joint_names,
            active_joint_names,
        )
    if args.viser:
        run_viser(
            args.host,
            int(args.port),
            str(args.viser_mode),
            urdf,
            dataset,
            result,
            joint_names,
        )


if __name__ == "__main__":
    main()
