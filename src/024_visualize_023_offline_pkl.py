#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import viser
import yaml

import aprilcube


APRILCUBE_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = APRILCUBE_ROOT / "assets"
PKL_PATH = (
    APRILCUBE_ROOT
    / "recordings/021_hand_back_sync_raw_frames_20260712_173546.pkl"
)
EXPECTED_PKL_FORMAT = "aprilcube_hand_back_software_synced_raw_v1"
EXPECTED_POSE_ALGORITHMS = {
    "011_fast_single_frame_pose_v1",
    "011_fast_single_frame_relaxed_single_face_pose_v2",
    "011_fast_single_frame_global_single_face_pose_v3",
    "deeptag_internal_grid_ransac_ippe_lm_temporal_v1",
    "deeptag_internal_grid_primary_cv2_fallback_v2",
}
UNDISTORTED_IMAGE_JPEG_FIELD = "undistorted_image_jpeg"

CAMERA_NAMES = ("thumb_web_cam", "middle_finger_cam")
WORLD_FRAME_NAME = "hand_back_cube"
HAND_BACK_CUBE_SIZE_M = 0.0625
DEFAULT_MIDDLE_FINGER_EXTRINSICS = Path(
    "/home/ps/RobotCamCalib1/outputs/"
    "extrinsics_middle_finger_cam_cube_d435_charuco_multisession_joint_"
    "0713_012814_022818.yaml"
)
DEFAULT_THUMB_WEB_EXTRINSICS = Path(
    "/home/ps/RobotCamCalib1/outputs/"
    "extrinsics_wrist_Q_thumb_web_cam_middle_finger_cam_apriltag_grid_"
    "offline_2samples_0712_030212_0712_031300.yaml"
)
CUBE_TO_OBJ = {
    "cube_april_36h11_0_5_1x1x1_15mm": "middle",
    "cube_april_36h11_6_11_1x1x1_15mm": "index",
    "cube_april_36h11_12_17_1x1x1_15mm": "thumb",
}
CUBE_COLORS = {
    "cube_april_36h11_0_5_1x1x1_15mm": (120, 220, 120),
    "cube_april_36h11_6_11_1x1x1_15mm": (255, 150, 40),
    "cube_april_36h11_12_17_1x1x1_15mm": (80, 180, 255),
}
CAMERA_COLORS = {
    "thumb_web_cam": (30, 170, 255),
    "middle_finger_cam": (255, 80, 130),
}

VISER_HOST = "0.0.0.0"
VISER_PORT = 8094
VISER_MAX_IMAGE_WIDTH = 960
VISER_JPEG_QUALITY = 85
AUTO_PLAY_FPS = 3.0
INITIAL_VIEW_POSITION = (0.22, -0.24, 0.18)
INITIAL_VIEW_LOOK_AT = (-0.06, 0.03, -0.06)

OBJ_MESH_SCALE = 0.001
UNDISTORT_IMAGE = True
PINHOLE_UNDISTORT_ALPHA = 0.0


def print_index_progress(done_bytes: int, total_bytes: int, *, finish: bool = False) -> None:
    width = 36
    ratio = min(max(done_bytes / max(total_bytes, 1), 0.0), 1.0)
    filled = int(round(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(
        f"\r[INFO] Indexing poses [{bar}] "
        f"{done_bytes / (1024**3):.2f}/{total_bytes / (1024**3):.2f} GiB"
    )
    if finish:
        sys.stdout.write("\n")
    sys.stdout.flush()


def build_frame_index_and_tracks(
    pkl_path: Path,
) -> tuple[
    dict[str, Any],
    list[int],
    dict[tuple[str, str], list[tuple[int, np.ndarray]]],
    dict[str, Any] | None,
]:
    file_size = pkl_path.stat().st_size
    frame_offsets: list[int] = []
    tracks: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = {}
    footer: dict[str, Any] | None = None
    last_progress = 0.0

    with pkl_path.open("rb") as file:
        header = pickle.load(file)
        if not isinstance(header, dict) or header.get("format") != EXPECTED_PKL_FORMAT:
            raise ValueError(f"Unsupported 021 PKL format: {header.get('format', None)}")
        pose_metadata = header.get("metadata", {}).get("offline_pos_estimation", {})
        if pose_metadata.get("algorithm") not in EXPECTED_POSE_ALGORITHMS:
            raise ValueError(
                f"PKL does not contain a supported offline pose algorithm "
                f"{sorted(EXPECTED_POSE_ALGORITHMS)}: {pkl_path}"
            )

        while True:
            record_offset = file.tell()
            try:
                record = pickle.load(file)
            except EOFError:
                break
            if not isinstance(record, dict):
                continue
            if record.get("type") == "frame_pair":
                frame_index = len(frame_offsets)
                frame_offsets.append(record_offset)
                for camera_name in CAMERA_NAMES:
                    camera_record = record["cameras"][camera_name]
                    offline_pos = camera_record.get("offline_pos", {})
                    for cube in offline_pos.get("cube_results", []):
                        result = cube.get("result", {})
                        if not result.get("success", False) or result.get("tvec") is None:
                            continue
                        key = (camera_name, str(cube["cube_name"]))
                        position = (
                            np.asarray(result["tvec"], dtype=np.float64).reshape(3)
                            / 1000.0
                        )
                        tracks.setdefault(key, []).append((frame_index, position))
            elif record.get("type") == "footer":
                footer = record
                break

            now = time.monotonic()
            if now - last_progress >= 0.5:
                print_index_progress(file.tell(), file_size)
                last_progress = now

    print_index_progress(file_size, file_size, finish=True)
    if not frame_offsets:
        raise ValueError(f"No frame_pair records found in {pkl_path}")
    return header, frame_offsets, tracks, footer


def load_frame_pair(pkl_path: Path, offset: int) -> dict[str, Any]:
    with pkl_path.open("rb") as file:
        file.seek(offset)
        record = pickle.load(file)
    if not isinstance(record, dict) or record.get("type") != "frame_pair":
        raise ValueError(f"Offset {offset} is not a frame_pair record")
    return record


def load_calibration(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    dist_coeffs = data.get("dist", data.get("D", np.zeros(5)))
    return {
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
    if not UNDISTORT_IMAGE or np.allclose(dist_coeffs, 0.0):
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
    if not UNDISTORT_IMAGE:
        return None
    camera_matrix = calibration["camera_matrix"]
    dist_coeffs = calibration["dist_coeffs"]
    image_size = calibration["image_size"]
    if np.allclose(dist_coeffs, 0.0):
        return None
    if is_fisheye(calibration):
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


def decode_jpeg_bgr(encoded: bytes | bytearray | memoryview) -> np.ndarray:
    buffer = np.frombuffer(encoded, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode JPEG image from PKL")
    return image


def normalize_result_for_drawing(result: dict[str, Any]) -> dict[str, Any]:
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


class ReprojectionRenderer:
    def __init__(
        self,
        camera_name: str,
        intrinsics_yaml: Path,
        cube_paths: list[Path],
    ) -> None:
        self.camera_name = camera_name
        self.calibration = load_calibration(intrinsics_yaml)
        self.detection_camera_matrix = make_detection_camera_matrix(self.calibration)
        self.undistort_maps = make_undistort_maps(
            self.calibration,
            self.detection_camera_matrix,
        )
        detector_distortion = (
            np.zeros(5, dtype=np.float64)
            if UNDISTORT_IMAGE
            else self.calibration["dist_coeffs"]
        )
        self.detectors: dict[str, Any] = {}
        for cube_path in cube_paths:
            resolved_path = cube_path.expanduser().resolve()
            self.detectors[resolved_path.name] = aprilcube.detector(
                resolved_path,
                intrinsic_cfg=camera_matrix_as_intrinsics(self.detection_camera_matrix),
                dist_coeffs=detector_distortion,
                enable_filter=False,
                fast=True,
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

    def draw(self, camera_record: dict[str, Any]) -> np.ndarray:
        encoded_undistorted = camera_record.get(UNDISTORTED_IMAGE_JPEG_FIELD)
        if isinstance(encoded_undistorted, (bytes, bytearray, memoryview)):
            visualization = decode_jpeg_bgr(encoded_undistorted)
        else:
            visualization = self.prepare_image(camera_record["image_bgr"])
        visualization = visualization.copy()
        offline_pos = camera_record["offline_pos"]
        lines = [
            f"{self.camera_name} sequence={camera_record.get('sequence', '?')}",
        ]
        missing_pose = False
        for cube in offline_pos.get("cube_results", []):
            cube_name = str(cube["cube_name"])
            result = normalize_result_for_drawing(cube.get("result", {}))
            detector = self.detectors[cube_name]
            visualization = detector.draw_result(visualization, result)
            if result.get("success", False):
                tvec = result["tvec"].reshape(3)
                lines.append(
                    f"{cube_name}: t=({tvec[0]:.1f},{tvec[1]:.1f},{tvec[2]:.1f})mm "
                    f"reproj={float(result.get('reproj_error', float('nan'))):.2f}px"
                )
            else:
                missing_pose = True
                lines.append(
                    f"{cube_name}: FAILED {result.get('failure_reason', '')}"
                )

        for line_index, line in enumerate(lines):
            cv2.putText(
                visualization,
                line,
                (18, 30 + line_index * 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if missing_pose:
            border = max(6, min(visualization.shape[:2]) // 120)
            cv2.rectangle(
                visualization,
                (0, 0),
                (visualization.shape[1] - 1, visualization.shape[0] - 1),
                (0, 0, 255),
                border,
            )
        return visualization

def bgr_to_resized_rgb(image_bgr: np.ndarray) -> np.ndarray:
    image = np.asarray(image_bgr, dtype=np.uint8)
    height, width = image.shape[:2]
    if VISER_MAX_IMAGE_WIDTH > 0 and width > VISER_MAX_IMAGE_WIDTH:
        scale = VISER_MAX_IMAGE_WIDTH / width
        image = cv2.resize(
            image,
            (VISER_MAX_IMAGE_WIDTH, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rotation_matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        quaternion = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            ]
        )
    elif rotation[1, 1] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        quaternion = np.array(
            [
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            ]
        )
    else:
        scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        quaternion = np.array(
            [
                (rotation[1, 0] - rotation[0, 1]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
            ]
        )
    return quaternion / max(float(np.linalg.norm(quaternion)), 1e-12)


def rvec_to_wxyz(rvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return rotation_matrix_to_wxyz(rotation)


def load_yaml_transform(path: Path, keys: tuple[str, ...]) -> np.ndarray:
    with path.open("r", encoding="utf-8") as file:
        node: Any = yaml.safe_load(file)
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"Missing transform {'.'.join(keys)} in {path}")
        node = node[key]

    transform = np.asarray(node, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(
            f"Transform {'.'.join(keys)} in {path} must be a finite 4x4 matrix"
        )
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError(f"Invalid homogeneous transform bottom row in {path}")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"Transform rotation is not orthonormal in {path}")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"Transform rotation determinant is not +1 in {path}")
    return transform


def load_world_from_camera_transforms(
    middle_finger_extrinsics: Path,
    thumb_web_extrinsics: Path,
) -> dict[str, np.ndarray]:
    return {
        "middle_finger_cam": load_yaml_transform(
            middle_finger_extrinsics,
            ("solution", "T_cube_middle_finger_cam"),
        ),
        "thumb_web_cam": load_yaml_transform(
            thumb_web_extrinsics,
            ("Q_T_thumb_web_cam",),
        ),
    }


def load_colored_obj(obj_name: str, color: tuple[int, int, int]) -> Any:
    import trimesh

    obj_path = ASSETS_DIR / f"{obj_name}.obj"
    loaded = trimesh.load(obj_path, process=False)
    mesh = (
        trimesh.util.concatenate(tuple(loaded.geometry.values()))
        if isinstance(loaded, trimesh.Scene)
        else loaded
    )
    rgba = np.asarray([*color, 210], dtype=np.uint8)
    mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))
    return mesh


def trajectory_segments(track: list[tuple[int, np.ndarray]]) -> np.ndarray:
    if len(track) < 2:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.asarray(
        [[track[index][1], track[index + 1][1]] for index in range(len(track) - 1)],
        dtype=np.float32,
    )


def create_scene(
    server: viser.ViserServer,
    renderers: dict[str, ReprojectionRenderer],
    tracks: dict[tuple[str, str], list[tuple[int, np.ndarray]]],
    world_from_camera: dict[str, np.ndarray],
) -> dict[tuple[str, str], dict[str, Any]]:
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = False
    handles: dict[tuple[str, str], dict[str, Any]] = {}
    mesh_cache: dict[str, Any] = {}

    world_root = f"/{WORLD_FRAME_NAME}"
    server.scene.add_frame(
        world_root,
        axes_length=0.08,
        axes_radius=0.0025,
        origin_radius=0.004,
    )
    server.scene.add_box(
        f"{world_root}/reference_cube",
        dimensions=(HAND_BACK_CUBE_SIZE_M,) * 3,
        color=(80, 120, 255),
        opacity=0.16,
        side="double",
    )
    for camera_name in CAMERA_NAMES:
        root = f"{world_root}/cameras/{camera_name}"
        transform = world_from_camera[camera_name]
        server.scene.add_frame(
            root,
            position=transform[:3, 3],
            wxyz=rotation_matrix_to_wxyz(transform[:3, :3]),
            axes_length=0.05,
            axes_radius=0.002,
            origin_radius=0.003,
        )
        renderer = renderers[camera_name]
        image_width, image_height = renderer.calibration["image_size"]
        fy = float(renderer.detection_camera_matrix[1, 1])
        fov_y = float(2.0 * np.arctan(image_height / max(2.0 * fy, 1e-12)))
        server.scene.add_camera_frustum(
            f"{root}/camera_frustum",
            fov=fov_y,
            aspect=image_width / image_height,
            scale=0.06,
            line_width=1.5,
            color=CAMERA_COLORS[camera_name],
        )

        for cube_name, detector in renderer.detectors.items():
            key = (camera_name, cube_name)
            color = CUBE_COLORS[cube_name]
            cube_root = f"{root}/{cube_name}"
            dimensions = tuple(float(value) / 1000.0 for value in detector.config.box_dims)
            pose_frame = server.scene.add_frame(
                cube_root,
                axes_length=max(dimensions) * 0.8,
                axes_radius=max(dimensions) * 0.035,
                origin_radius=0.0,
                visible=False,
            )
            cube_box = server.scene.add_box(
                f"{cube_root}/cube_box",
                dimensions=dimensions,
                color=color,
                opacity=0.30,
                side="double",
                visible=False,
            )
            obj_name = CUBE_TO_OBJ[cube_name]
            if obj_name not in mesh_cache:
                mesh_cache[obj_name] = load_colored_obj(obj_name, color)
            obj_mesh = server.scene.add_mesh_trimesh(
                f"{cube_root}/{obj_name}_obj",
                mesh_cache[obj_name].copy(),
                scale=OBJ_MESH_SCALE,
                visible=False,
                cast_shadow=False,
                receive_shadow=False,
            )

            track = tracks.get(key, [])
            segments = trajectory_segments(track)
            trajectory = server.scene.add_line_segments(
                f"{root}/tracks/{cube_name}/trajectory",
                points=segments,
                colors=color,
                line_width=2.0,
                visible=len(segments) > 0,
            )
            samples = server.scene.add_point_cloud(
                f"{root}/tracks/{cube_name}/samples",
                points=np.asarray([position for _index, position in track], dtype=np.float32),
                colors=np.tile(np.asarray(color, dtype=np.uint8), (len(track), 1)),
                point_size=0.003,
                point_shape="circle",
                visible=len(track) > 0,
            )
            current_position = server.scene.add_icosphere(
                f"{root}/tracks/{cube_name}/current",
                radius=max(max(dimensions) * 0.08, 0.0015),
                color=(255, 255, 255),
                subdivisions=2,
                visible=False,
            )
            handles[key] = {
                "pose_frame": pose_frame,
                "cube_box": cube_box,
                "obj_mesh": obj_mesh,
                "trajectory": trajectory,
                "samples": samples,
                "current_position": current_position,
                "pose_visible": False,
            }
    return handles


def update_scene(
    scene_handles: dict[tuple[str, str], dict[str, Any]],
    frame_pair: dict[str, Any],
) -> None:
    for camera_name in CAMERA_NAMES:
        offline_pos = frame_pair["cameras"][camera_name]["offline_pos"]
        results_by_cube = {
            str(cube["cube_name"]): cube.get("result", {})
            for cube in offline_pos.get("cube_results", [])
        }
        for cube_name in CUBE_TO_OBJ:
            key = (camera_name, cube_name)
            handles = scene_handles.get(key)
            if handles is None:
                continue
            result = results_by_cube.get(cube_name, {})
            success = bool(result.get("success", False))
            handles["pose_visible"] = success
            if not success:
                for name in ("pose_frame", "cube_box", "obj_mesh", "current_position"):
                    handles[name].visible = False
                continue
            position = np.asarray(result["tvec"], dtype=np.float64).reshape(3) / 1000.0
            handles["pose_frame"].position = position
            handles["pose_frame"].wxyz = rvec_to_wxyz(result["rvec"])
            handles["current_position"].position = position


def apply_scene_visibility(
    scene_handles: dict[tuple[str, str], dict[str, Any]],
    *,
    show_obj: bool,
    show_box: bool,
    show_axes: bool,
    show_trajectory: bool,
    show_samples: bool,
) -> None:
    for handles in scene_handles.values():
        pose_visible = bool(handles["pose_visible"])
        handles["obj_mesh"].visible = show_obj and pose_visible
        handles["cube_box"].visible = show_box and pose_visible
        handles["pose_frame"].visible = show_axes and pose_visible
        handles["current_position"].visible = show_trajectory and pose_visible
        handles["trajectory"].visible = show_trajectory
        handles["samples"].visible = show_samples


def frame_metadata_markdown(
    frame_index: int,
    frame_pair: dict[str, Any],
    world_from_camera: dict[str, np.ndarray],
) -> str:
    lines = [
        f"**Frame pair:** `{frame_index}`",
        f"**Thumb - middle skew:** `{float(frame_pair.get('signed_skew_ms', 0.0)):+.3f} ms`",
    ]
    for camera_name in CAMERA_NAMES:
        lines.extend(["", f"**{camera_name}**"])
        offline_pos = frame_pair["cameras"][camera_name]["offline_pos"]
        for cube in offline_pos.get("cube_results", []):
            result = cube.get("result", {})
            cube_name = cube["cube_name"]
            if not result.get("success", False):
                lines.append(
                    f"- `{cube_name}`: failed, `{result.get('failure_reason', '')}`"
                )
                continue
            tvec = np.asarray(result["tvec"], dtype=np.float64).reshape(3)
            world_position_m = (
                world_from_camera[camera_name]
                @ np.asarray([*(tvec / 1000.0), 1.0], dtype=np.float64)
            )[:3]
            lines.append(
                f"- `{cube_name}`: world_t=({world_position_m[0] * 1000.0:.1f}, "
                f"{world_position_m[1] * 1000.0:.1f}, "
                f"{world_position_m[2] * 1000.0:.1f}) mm, "
                f"camera_t=({tvec[0]:.1f}, {tvec[1]:.1f}, {tvec[2]:.1f}) mm, "
                f"reproj={float(result.get('reproj_error', float('nan'))):.2f}px, "
                f"tags={int(result.get('n_tags', 0))}"
            )
    return "\n".join(lines)


def build_renderers(header: dict[str, Any]) -> dict[str, ReprojectionRenderer]:
    metadata = header["metadata"]
    return {
        camera_name: ReprojectionRenderer(
            camera_name,
            Path(metadata["camera_intrinsics_yaml"][camera_name]),
            [Path(path) for path in metadata["camera_cube_configs"][camera_name]],
        )
        for camera_name in CAMERA_NAMES
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize 023 offline poses embedded in a 021 synchronized PKL."
    )
    parser.add_argument(
        "pkl_path",
        nargs="?",
        type=Path,
        default=PKL_PATH,
        help="021_hand_back_sync_raw_frames_*.pkl containing embedded offline_pos records.",
    )
    parser.add_argument("--host", type=str, default=VISER_HOST, help="Viser server host.")
    parser.add_argument("--port", type=int, default=VISER_PORT, help="Viser server port.")
    parser.add_argument(
        "--middle-finger-extrinsics",
        type=Path,
        default=DEFAULT_MIDDLE_FINGER_EXTRINSICS,
        help="YAML containing solution.T_cube_middle_finger_cam.",
    )
    parser.add_argument(
        "--thumb-web-extrinsics",
        type=Path,
        default=DEFAULT_THUMB_WEB_EXTRINSICS,
        help="YAML containing Q_T_thumb_web_cam.",
    )
    args = parser.parse_args()

    pkl_path = args.pkl_path.expanduser().resolve()
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Offline 021 PKL not found: {pkl_path}")
    middle_finger_extrinsics = args.middle_finger_extrinsics.expanduser().resolve()
    thumb_web_extrinsics = args.thumb_web_extrinsics.expanduser().resolve()
    for extrinsics_path in (middle_finger_extrinsics, thumb_web_extrinsics):
        if not extrinsics_path.is_file():
            raise FileNotFoundError(f"Extrinsics YAML not found: {extrinsics_path}")
    world_from_camera = load_world_from_camera_transforms(
        middle_finger_extrinsics,
        thumb_web_extrinsics,
    )

    print(f"[INFO] PKL: {pkl_path}")
    for camera_name in CAMERA_NAMES:
        position_mm = world_from_camera[camera_name][:3, 3] * 1000.0
        print(
            f"[INFO] T_{WORLD_FRAME_NAME}_{camera_name}: "
            f"t=({position_mm[0]:.2f}, {position_mm[1]:.2f}, "
            f"{position_mm[2]:.2f}) mm"
        )
    header, frame_offsets, tracks, footer = build_frame_index_and_tracks(pkl_path)
    renderers = build_renderers(header)
    total_frames = len(frame_offsets)
    first_pair = load_frame_pair(pkl_path, frame_offsets[0])

    server = viser.ViserServer(host=args.host, port=int(args.port))
    server.initial_camera.position = INITIAL_VIEW_POSITION
    server.initial_camera.look_at = INITIAL_VIEW_LOOK_AT
    server.initial_camera.up = (0.0, 0.0, 1.0)
    server.initial_camera.fov = float(np.deg2rad(55.0))
    server.initial_camera.near = 0.001
    server.gui.set_panel_label("023 Offline Cube Pose")
    scene_handles = create_scene(server, renderers, tracks, world_from_camera)
    update_scene(scene_handles, first_pair)

    first_thumb_image = renderers["thumb_web_cam"].draw(
        first_pair["cameras"]["thumb_web_cam"]
    )
    first_middle_image = renderers["middle_finger_cam"].draw(
        first_pair["cameras"]["middle_finger_cam"]
    )
    first_thumb_raw = first_pair["cameras"]["thumb_web_cam"]["image_bgr"]
    first_middle_raw = first_pair["cameras"]["middle_finger_cam"]["image_bgr"]

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame pair",
            min=0,
            max=total_frames - 1,
            step=1,
            initial_value=0,
        )
        auto_play = server.gui.add_checkbox("Auto play", initial_value=False)
        loop_playback = server.gui.add_checkbox("Loop", initial_value=True)
        status_text = server.gui.add_text(
            "Status",
            initial_value=f"1/{total_frames}",
            disabled=True,
        )

    with server.gui.add_folder("Middle Finger Undistorted + Cube Pose"):
        middle_image = server.gui.add_image(
            bgr_to_resized_rgb(first_middle_image),
            label="Cube pose reprojected on the undistorted image",
            format="jpeg",
            jpeg_quality=VISER_JPEG_QUALITY,
        )

    with server.gui.add_folder("Thumb Web Undistorted + Cube Pose"):
        thumb_image = server.gui.add_image(
            bgr_to_resized_rgb(first_thumb_image),
            label="Cube pose reprojected on the undistorted image",
            format="jpeg",
            jpeg_quality=VISER_JPEG_QUALITY,
        )

    with server.gui.add_folder("3D Visibility"):
        show_obj = server.gui.add_checkbox("Finger OBJ", initial_value=True)
        show_box = server.gui.add_checkbox("Cube box", initial_value=True)
        show_axes = server.gui.add_checkbox("Cube axes", initial_value=True)
        show_trajectory = server.gui.add_checkbox("Trajectory", initial_value=True)
        show_samples = server.gui.add_checkbox("Pose samples", initial_value=False)

    with server.gui.add_folder("Pose Metadata"):
        pose_metadata = server.gui.add_markdown(
            frame_metadata_markdown(0, first_pair, world_from_camera)
        )
        server.gui.add_markdown(
            "\n".join(
                [
                    f"**PKL:** `{pkl_path}`",
                    f"**Algorithm:** `{header.get('metadata', {}).get('offline_pos_estimation', {}).get('algorithm')}`",
                    f"**Frame pairs:** `{total_frames}`",
                    f"**3D world:** `{WORLD_FRAME_NAME}`",
                    f"**Middle extrinsics:** `{middle_finger_extrinsics}`",
                    f"**Thumb extrinsics:** `{thumb_web_extrinsics}`",
                    "**Transform chain:** `T_world_cube = T_world_camera @ T_camera_cube`",
                    f"**Footer:** `{footer}`",
                ]
            )
        )

    # Keep the unmodified camera frames at the bottom of the sidebar. Pose
    # overlays above are always rendered in the rectified camera geometry.
    with server.gui.add_folder("Middle Finger Raw Image"):
        middle_raw_image = server.gui.add_image(
            bgr_to_resized_rgb(first_middle_raw),
            label="Original image_bgr from PKL",
            format="jpeg",
            jpeg_quality=VISER_JPEG_QUALITY,
        )

    with server.gui.add_folder("Thumb Web Raw Image"):
        thumb_raw_image = server.gui.add_image(
            bgr_to_resized_rgb(first_thumb_raw),
            label="Original image_bgr from PKL",
            format="jpeg",
            jpeg_quality=VISER_JPEG_QUALITY,
        )

    render_lock = threading.Lock()

    def render(frame_index: int) -> None:
        with render_lock:
            frame_pair = load_frame_pair(pkl_path, frame_offsets[frame_index])
            thumb_overlay = renderers["thumb_web_cam"].draw(
                frame_pair["cameras"]["thumb_web_cam"]
            )
            middle_overlay = renderers["middle_finger_cam"].draw(
                frame_pair["cameras"]["middle_finger_cam"]
            )
            thumb_image.image = bgr_to_resized_rgb(thumb_overlay)
            middle_image.image = bgr_to_resized_rgb(middle_overlay)
            middle_raw_image.image = bgr_to_resized_rgb(
                frame_pair["cameras"]["middle_finger_cam"]["image_bgr"]
            )
            thumb_raw_image.image = bgr_to_resized_rgb(
                frame_pair["cameras"]["thumb_web_cam"]["image_bgr"]
            )
            update_scene(scene_handles, frame_pair)
            pose_metadata.content = frame_metadata_markdown(
                frame_index,
                frame_pair,
                world_from_camera,
            )
            status_text.value = (
                f"{frame_index + 1}/{total_frames} | "
                f"skew={float(frame_pair.get('signed_skew_ms', 0.0)):+.3f} ms"
            )

    print(f"[INFO] Indexed frame pairs: {total_frames}")
    print(f"[INFO] Viser: http://localhost:{int(args.port)}")
    rendered_frame = 0
    last_playback_step = time.monotonic()
    while True:
        apply_scene_visibility(
            scene_handles,
            show_obj=bool(show_obj.value),
            show_box=bool(show_box.value),
            show_axes=bool(show_axes.value),
            show_trajectory=bool(show_trajectory.value),
            show_samples=bool(show_samples.value),
        )
        selected_frame = int(frame_slider.value)
        if selected_frame != rendered_frame:
            render(selected_frame)
            rendered_frame = selected_frame

        if bool(auto_play.value):
            now = time.monotonic()
            if now - last_playback_step >= 1.0 / max(AUTO_PLAY_FPS, 1e-6):
                next_frame = rendered_frame + 1
                if next_frame >= total_frames:
                    if bool(loop_playback.value):
                        next_frame = 0
                    else:
                        next_frame = total_frames - 1
                        auto_play.value = False
                frame_slider.value = next_frame
                last_playback_step = now
        else:
            last_playback_step = time.monotonic()
        time.sleep(0.01)


if __name__ == "__main__":
    main()
