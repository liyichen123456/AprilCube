from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import aprilcube
import cv2
import numpy as np
import viser
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

DEFAULT_PKL_PATH = "/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/logs_002/recording_20260510_174908.pkl"
VISER_HOST = "0.0.0.0"
VISER_PORT = 8080
CAMERA_AXES_LENGTH_M = 0.04
CAMERA_AXES_RADIUS_M = 0.0015
CUBE_AXES_LENGTH_M = 0.025
CUBE_AXES_RADIUS_M = 0.001
TAG_AXES_LENGTH_M = 0.018
TAG_AXES_RADIUS_M = 0.0009
PLAYBACK_FPS = 25.0
LOOP_PLAYBACK = True
GRAY_CUBE_AXIS_LENGTH_SCALE = 0.9
GRAY_TAG_AXIS_LENGTH_SCALE = 0.8
GRAY_CUBE_AXIS_THICKNESS = 1
GRAY_TAG_AXIS_THICKNESS = 3
GRAY_CORNER_INDEX_FONT_SCALE = 0.45
GRAY_CORNER_INDEX_THICKNESS = 1

THIS_FILE = Path(__file__).resolve()
SCRIPT_002_PATH = THIS_FILE.parent / "002_oak_aprilcube_detect_tag_visual_temporal_pose.py"


def load_logic_module() -> Any:
    spec = importlib.util.spec_from_file_location("replay_002_logic", SCRIPT_002_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load logic module from {SCRIPT_002_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["replay_002_logic"] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize saved 002 snapshot/recording pkl.")
    parser.add_argument(
        "pkl_path",
        nargs="?",
        default=DEFAULT_PKL_PATH,
        help="Path to pkl or a directory containing recording_*.pkl / snapshot_*.pkl.",
    )
    return parser.parse_args()


def resolve_pkl_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(list(path.glob("recording_*.pkl")) + list(path.glob("snapshot_*.pkl")))
        if not candidates:
            raise FileNotFoundError(f"No recording_*.pkl or snapshot_*.pkl found in directory: {path}")
        return candidates[-1]
    raise FileNotFoundError(f"Invalid pkl path: {path}")


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict payload, got {type(data).__name__}")
    return data


def normalize_frames(payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta = payload.get("meta", {})
    if "frames" in payload:
        frames_raw = payload["frames"]
        if isinstance(frames_raw, list):
            return meta, list(frames_raw)
        if isinstance(frames_raw, np.ndarray):
            return meta, [item for item in frames_raw.tolist()]
        raise ValueError(f"Unsupported frames type: {type(frames_raw).__name__}")
    if "cameras" in payload:
        single_frame = {
            "frame_idx": int(meta.get("frame_idx", 0)),
            "timestamp_epoch_s": float(meta.get("timestamp_epoch_s", 0.0)),
            "cameras": payload["cameras"],
        }
        return meta, [single_frame]
    raise ValueError("Pkl payload must contain either 'frames' or 'cameras'.")


def clamp_frame_index(frame_idx: int, num_frames: int) -> int:
    return max(0, min(int(frame_idx), num_frames - 1))


def rotation_matrix_to_wxyz(rot: np.ndarray) -> tuple[float, float, float, float]:
    quat_xyzw = R.from_matrix(np.asarray(rot, dtype=np.float64)).as_quat()
    x, y, z, w = quat_xyzw
    return (float(w), float(x), float(y), float(z))


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    img_bgr = np.asarray(img_bgr, dtype=np.uint8)
    if img_bgr.ndim == 2:
        return np.repeat(img_bgr[:, :, None], 3, axis=2)
    return img_bgr[:, :, ::-1]


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.uint8)
    if gray.ndim == 2:
        return np.repeat(gray[:, :, None], 3, axis=2)
    return gray


def project_axes_points(
    pose_R: np.ndarray,
    pose_t: np.ndarray,
    axis_length_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    pose_R = np.asarray(pose_R, dtype=np.float64).reshape(3, 3)
    pose_t = np.asarray(pose_t, dtype=np.float64).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(pose_R)
    obj_pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length_m, 0.0, 0.0],
            [0.0, axis_length_m, 0.0],
            [0.0, 0.0, axis_length_m],
        ],
        dtype=np.float64,
    )
    img_pts, _ = cv2.projectPoints(
        obj_pts,
        rvec,
        pose_t,
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1),
    )
    return np.round(img_pts.reshape(-1, 2)).astype(np.int32)


def draw_axes_overlay(
    img: np.ndarray,
    pose_R: np.ndarray,
    pose_t: np.ndarray,
    axis_length_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    thickness: int,
) -> None:
    pts = project_axes_points(
        pose_R=pose_R,
        pose_t=pose_t,
        axis_length_m=axis_length_m,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )
    origin = tuple(pts[0])
    pt_x = tuple(pts[1])
    pt_y = tuple(pts[2])
    pt_z = tuple(pts[3])
    cv2.arrowedLine(img, origin, pt_x, (0, 0, 255), thickness, tipLength=0.22)
    cv2.arrowedLine(img, origin, pt_y, (0, 255, 0), thickness, tipLength=0.22)
    cv2.arrowedLine(img, origin, pt_z, (255, 0, 0), thickness, tipLength=0.22)


def draw_corner_indices(
    img: np.ndarray,
    detections: list[Any],
) -> None:
    for _tag_id, corners_xy in detections:
        corners = np.asarray(corners_xy, dtype=np.float64).reshape(4, 2)
        for corner_idx, corner_xy in enumerate(corners):
            x = int(round(float(corner_xy[0])))
            y = int(round(float(corner_xy[1])))
            cv2.putText(
                img,
                str(corner_idx),
                (x + 4, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                GRAY_CORNER_INDEX_FONT_SCALE,
                (0, 255, 255),
                GRAY_CORNER_INDEX_THICKNESS,
                cv2.LINE_AA,
            )
            cv2.circle(img, (x, y), 2, (255, 255, 0), -1)


def make_gray_overlay_image(camera_record: dict[str, Any]) -> np.ndarray:
    detector_gray = np.asarray(camera_record.get("detector_gray"), dtype=np.uint8)
    overlay = gray_to_rgb(detector_gray).copy()

    camera_matrix = camera_record.get("camera_matrix", None)
    if camera_matrix is None:
        return overlay
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(camera_record.get("dist_coeffs", np.zeros(5)), dtype=np.float64).reshape(-1)
    result = camera_record.get("result", {})
    if not isinstance(result, dict):
        return overlay

    detections = result.get("detections", [])
    if isinstance(detections, list):
        draw_corner_indices(overlay, detections)

    if result.get("success", False) and result.get("rvec", None) is not None and result.get("tvec", None) is not None:
        cube_rot, _ = cv2.Rodrigues(np.asarray(result["rvec"], dtype=np.float64).reshape(3, 1))
        cube_tvec_m = np.asarray(result["tvec"], dtype=np.float64).reshape(3) / 1000.0
        box_dims_mm = np.asarray(camera_record.get("box_dims_mm", [10.0, 10.0, 10.0]), dtype=np.float64).reshape(-1)
        cube_axis_length_m = float(np.max(box_dims_mm)) / 1000.0 * GRAY_CUBE_AXIS_LENGTH_SCALE
        draw_axes_overlay(
            overlay,
            pose_R=cube_rot,
            pose_t=cube_tvec_m,
            axis_length_m=cube_axis_length_m,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            thickness=GRAY_CUBE_AXIS_THICKNESS,
        )

    tag_pose_by_id = result.get("tag_pose_by_id", {})
    if isinstance(tag_pose_by_id, dict):
        tag_size_mm = float(camera_record.get("tag_size_mm", 10.0))
        tag_axis_length_m = tag_size_mm / 1000.0 * GRAY_TAG_AXIS_LENGTH_SCALE
        for tag_id, tag_pose in tag_pose_by_id.items():
            if not isinstance(tag_pose, dict):
                continue
            pose_R = tag_pose.get("rot_mat", None)
            pose_t_mm = tag_pose.get("tvec", None)
            if pose_R is None or pose_t_mm is None:
                continue
            pose_R = np.asarray(pose_R, dtype=np.float64).reshape(3, 3)
            pose_t_m = np.asarray(pose_t_mm, dtype=np.float64).reshape(3) / 1000.0
            draw_axes_overlay(
                overlay,
                pose_R=pose_R,
                pose_t=pose_t_m,
                axis_length_m=tag_axis_length_m,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                thickness=GRAY_TAG_AXIS_THICKNESS,
            )
    return overlay


def camera_names_from_frames(frames: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for frame in frames:
        names.update(str(name) for name in frame.get("cameras", {}).keys())
    return sorted(names)


def camera_matrix_to_intrinsic_dict(k: np.ndarray) -> dict[str, float]:
    k = np.asarray(k, dtype=np.float64).reshape(3, 3)
    return {
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
    }


def resolve_camera_matrix_from_record(
    logic_module: Any,
    camera_name: str,
    record: dict[str, Any],
) -> np.ndarray:
    camera_matrix = record.get("camera_matrix", None)
    if camera_matrix is not None:
        arr = np.asarray(camera_matrix, dtype=np.float64)
        if arr.size == 9:
            return arr.reshape(3, 3)

    if camera_name not in logic_module.K_BY_CAMERA:
        raise KeyError(f"Missing fallback K_BY_CAMERA for camera '{camera_name}'")
    return logic_module.scale_intrinsics(
        np.asarray(logic_module.K_BY_CAMERA[camera_name], dtype=np.float64),
        old_size=tuple(int(v) for v in logic_module.K_ORIGINAL_SIZE),
        new_size=tuple(int(v) for v in logic_module.DETECT_IMG_SIZE),
    )


def resolve_dist_coeffs_from_record(
    logic_module: Any,
    camera_name: str,
    record: dict[str, Any],
) -> np.ndarray:
    dist_coeffs = record.get("dist_coeffs", None)
    if dist_coeffs is not None:
        arr = np.asarray(dist_coeffs, dtype=np.float64)
        if arr.size >= 4:
            return arr.reshape(-1)

    fallback = logic_module.DIST_COEFFS_BY_CAMERA.get(camera_name, None)
    if fallback is None:
        return np.zeros(5, dtype=np.float64)
    return np.asarray(fallback, dtype=np.float64).reshape(-1)


def build_replay_processors(
    logic_module: Any,
    meta: dict[str, Any],
    frames: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not frames:
        return {}

    cube_path = Path(str(meta.get("cube_path", ""))).expanduser().resolve()
    if not cube_path.exists():
        raise FileNotFoundError(f"Cube path from pkl does not exist: {cube_path}")

    first_cameras = frames[0].get("cameras", {})
    if not isinstance(first_cameras, dict):
        raise ValueError("Invalid frames payload: first frame has no camera dict")

    processors: dict[str, dict[str, Any]] = {}
    shared_native_detectors: dict[str, Detector] = {}
    for camera_name, record in first_cameras.items():
        camera_matrix = resolve_camera_matrix_from_record(logic_module, camera_name, record)
        dist_coeffs = resolve_dist_coeffs_from_record(logic_module, camera_name, record)
        detector = aprilcube.detector(
            cube_path,
            intrinsic_cfg=camera_matrix_to_intrinsic_dict(camera_matrix),
            dist_coeffs=dist_coeffs,
            enable_filter=bool(getattr(logic_module, "ENABLE_FILTER", True)),
            fast=bool(getattr(logic_module, "FAST_DETECTOR", True)),
        )
        native_family = logic_module.apriltag_family_from_dict_name(detector.config.dict_name)
        if native_family not in shared_native_detectors:
            shared_native_detectors[native_family] = Detector(
                families=native_family,
                quad_decimate=1.0,
            )
        pose_estimator = logic_module.TemporalTagPoseEstimator(
            tag_size_m=float(detector.config.tag_size_mm) / 1000.0,
            pupil_to_object_corner_index=list(getattr(logic_module, "PUPIL_TO_OBJECT_CORNER_INDEX")),
            solvepnp_generic_flag=int(getattr(logic_module, "SOLVEPNP_GENERIC_FLAG")),
            solvepnp_flag=int(getattr(logic_module, "SOLVEPNP_FLAG")),
            use_temporal_candidate_selection=bool(getattr(logic_module, "USE_TEMPORAL_TAG_POSE_ESTIMATOR"))
            and bool(getattr(logic_module, "USE_TEMPORAL_CANDIDATE_SELECTION")),
            use_solvepnp_refine_lm=bool(getattr(logic_module, "USE_SOLVEPNP_REFINE_LM")),
            translation_score_weight_deg_per_mm=float(getattr(logic_module, "TRANSLATION_SCORE_WEIGHT_DEG_PER_MM")),
            reject_negative_camera_z=bool(getattr(logic_module, "REJECT_NEGATIVE_CAMERA_Z")),
        )
        processors[camera_name] = {
            "detector": detector,
            "native_detector": shared_native_detectors[native_family],
            "pose_estimator": pose_estimator,
        }
    return processors


def recompute_frame_results(
    logic_module: Any,
    processors: dict[str, dict[str, Any]],
    frame: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    recomputed: dict[str, dict[str, Any]] = {}
    for camera_name, camera_record in frame.get("cameras", {}).items():
        processor = processors.get(camera_name)
        if processor is None:
            continue
        detect_frame_bgr = np.asarray(camera_record.get("detect_frame_bgr"), dtype=np.uint8)
        result = logic_module.process_frame_from_tag_poses(
            camera_name=camera_name,
            detector=processor["detector"],
            native_detector=processor["native_detector"],
            pose_estimator=processor["pose_estimator"],
            image=detect_frame_bgr,
        )
        gray = cv2.cvtColor(detect_frame_bgr, cv2.COLOR_BGR2GRAY)
        if bool(getattr(logic_module, "USE_CLAHE_FOR_TAG_DETECTION", False)):
            clahe = cv2.createCLAHE(
                clipLimit=float(getattr(logic_module, "CLAHE_CLIP_LIMIT", 2.0)),
                tileGridSize=tuple(int(v) for v in getattr(logic_module, "CLAHE_TILE_GRID_SIZE", (8, 8))),
            )
            gray = clahe.apply(np.asarray(gray, dtype=np.uint8))

        vis = processor["detector"].draw_result(detect_frame_bgr.copy(), result)
        vis = logic_module.draw_detected_tag_visuals(
            img=vis,
            detector=processor["detector"],
            result=result,
        )
        recomputed[camera_name] = {
            "result": result,
            "detector_gray": gray,
            "vis_bgr": vis,
            "camera_matrix": np.array(processor["detector"].camera_matrix, copy=True),
            "dist_coeffs": np.array(processor["detector"].dist_coeffs, copy=True)
            if processor["detector"].dist_coeffs is not None
            else np.zeros(5, dtype=np.float64),
            "tag_size_mm": float(processor["detector"].config.tag_size_mm),
            "box_dims_mm": np.array(processor["detector"].config.box_dims, copy=True),
        }
    return recomputed


def update_scene_for_frame(
    server: viser.ViserServer,
    frame: dict[str, Any],
    camera_frame_handles: dict[str, Any],
    cube_frame_handles: dict[str, Any],
    tag_frame_handles: dict[str, Any],
) -> None:
    visible_cubes: set[str] = set()
    visible_tags: set[str] = set()
    visible_cameras: set[str] = set()

    for camera_name, record in frame.get("cameras", {}).items():
        visible_cameras.add(camera_name)
        if camera_name not in camera_frame_handles:
            camera_frame_handles[camera_name] = server.scene.add_frame(
                f"/camera/{camera_name}",
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
                axes_length=CAMERA_AXES_LENGTH_M,
                axes_radius=CAMERA_AXES_RADIUS_M,
                origin_radius=0.0,
            )
        camera_frame_handles[camera_name].visible = True

        result = record.get("result", {})
        if isinstance(result, dict) and result.get("success", False):
            rvec = np.asarray(result["rvec"], dtype=np.float64).reshape(3, 1)
            tvec_m = np.asarray(result["tvec"], dtype=np.float64).reshape(3) / 1000.0
            cube_rot, _ = cv2.Rodrigues(rvec)
            cube_name = f"{camera_name}/cube"
            visible_cubes.add(cube_name)
            if cube_name not in cube_frame_handles:
                cube_frame_handles[cube_name] = server.scene.add_frame(
                    f"/cube/{cube_name}",
                    wxyz=rotation_matrix_to_wxyz(cube_rot),
                    position=(float(tvec_m[0]), float(tvec_m[1]), float(tvec_m[2])),
                    axes_length=CUBE_AXES_LENGTH_M,
                    axes_radius=CUBE_AXES_RADIUS_M,
                    origin_radius=0.0,
                )
            else:
                handle = cube_frame_handles[cube_name]
                handle.wxyz = rotation_matrix_to_wxyz(cube_rot)
                handle.position = (float(tvec_m[0]), float(tvec_m[1]), float(tvec_m[2]))
                handle.visible = True

        tag_pose_by_id = result.get("tag_pose_by_id", {}) if isinstance(result, dict) else {}
        if isinstance(tag_pose_by_id, dict):
            for tag_id, tag_pose in tag_pose_by_id.items():
                if not isinstance(tag_pose, dict):
                    continue
                rot_mat = tag_pose.get("rot_mat", None)
                tvec = tag_pose.get("tvec", None)
                if rot_mat is None or tvec is None:
                    continue
                rot_mat = np.asarray(rot_mat, dtype=np.float64).reshape(3, 3)
                tvec_m = np.asarray(tvec, dtype=np.float64).reshape(3) / 1000.0
                tag_name = f"{camera_name}/tag_{int(tag_id)}"
                visible_tags.add(tag_name)
                if tag_name not in tag_frame_handles:
                    tag_frame_handles[tag_name] = server.scene.add_frame(
                        f"/tag/{tag_name}",
                        wxyz=rotation_matrix_to_wxyz(rot_mat),
                        position=(float(tvec_m[0]), float(tvec_m[1]), float(tvec_m[2])),
                        axes_length=TAG_AXES_LENGTH_M,
                        axes_radius=TAG_AXES_RADIUS_M,
                        origin_radius=0.0,
                    )
                else:
                    handle = tag_frame_handles[tag_name]
                    handle.wxyz = rotation_matrix_to_wxyz(rot_mat)
                    handle.position = (float(tvec_m[0]), float(tvec_m[1]), float(tvec_m[2]))
                    handle.visible = True

    for name, handle in camera_frame_handles.items():
        if name not in visible_cameras:
            handle.visible = False
    for name, handle in cube_frame_handles.items():
        if name not in visible_cubes:
            handle.visible = False
    for name, handle in tag_frame_handles.items():
        if name not in visible_tags:
            handle.visible = False


def main() -> None:
    args = parse_args()
    pkl_path = resolve_pkl_path(args.pkl_path)
    payload = load_payload(pkl_path)
    meta, frames = normalize_frames(payload)
    if not frames:
        raise ValueError(f"No frames found in {pkl_path}")

    camera_names = camera_names_from_frames(frames)
    selected_camera_name = camera_names[0] if camera_names else None
    frame_idx = 0
    is_playing = len(frames) > 1
    loop_playback = bool(LOOP_PLAYBACK)
    last_step_time = time.monotonic()
    logic_module = load_logic_module()
    processors = build_replay_processors(logic_module, meta, frames)

    server = viser.ViserServer(host=VISER_HOST, port=VISER_PORT)
    server.scene.set_up_direction("-y")
    server.scene.world_axes.visible = False
    server.scene.add_frame(
        "/world",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        axes_length=0.05,
        axes_radius=0.002,
        origin_radius=0.0,
    )

    with server.gui.add_folder("Replay Controls"):
        play_checkbox = server.gui.add_checkbox("Play", initial_value=is_playing)
        loop_checkbox = server.gui.add_checkbox("Loop", initial_value=loop_playback)
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=len(frames) - 1,
            step=1,
            initial_value=0,
        )
        status_text = server.gui.add_text("Status", initial_value="", disabled=True)
        server.gui.add_text("Recording", initial_value=str(pkl_path), disabled=True)

    with server.gui.add_folder("Images"):
        camera_dropdown = server.gui.add_dropdown(
            "Camera",
            options=camera_names if camera_names else ["<none>"],
            initial_value=selected_camera_name if selected_camera_name is not None else "<none>",
        )
        vis_image_handle = server.gui.add_image(
            np.zeros((120, 160, 3), dtype=np.uint8),
            label="Vis",
            format="jpeg",
            jpeg_quality=80,
        )
        gray_image_handle = server.gui.add_image(
            np.zeros((120, 160, 3), dtype=np.uint8),
            label="Gray / CLAHE",
            format="jpeg",
            jpeg_quality=80,
        )

    summary_lines = [
        f"cube_path: `{meta.get('cube_path', '')}`",
        f"frames: `{len(frames)}`",
        f"detect_img_size: `{meta.get('detect_img_size', None)}`",
        f"clahe: `{meta.get('use_clahe_for_tag_detection', False)}` "
        f"clip={meta.get('clahe_clip_limit', None)} tile={meta.get('clahe_tile_grid_size', None)}",
    ]
    server.gui.add_markdown("\n".join(summary_lines))

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
        frame_idx = clamp_frame_index(int(frame_slider.value), len(frames))
        last_step_time = time.monotonic()

    @camera_dropdown.on_update
    def _on_camera(_event: Any) -> None:
        nonlocal selected_camera_name
        selected_camera_name = None if camera_dropdown.value == "<none>" else str(camera_dropdown.value)

    camera_frame_handles: dict[str, Any] = {}
    cube_frame_handles: dict[str, Any] = {}
    tag_frame_handles: dict[str, Any] = {}

    print(f"[INFO] Loaded pkl: {pkl_path}")
    print(f"[INFO] Viser server started on http://{VISER_HOST}:{VISER_PORT}")

    while True:
        curr_frame = frames[frame_idx]
        recomputed_records = recompute_frame_results(logic_module, processors, curr_frame)
        display_frame = {
            "frame_idx": curr_frame.get("frame_idx", frame_idx),
            "timestamp_epoch_s": curr_frame.get("timestamp_epoch_s", 0.0),
            "cameras": {
                camera_name: {
                    **camera_record,
                    **recomputed_records.get(camera_name, {}),
                }
                for camera_name, camera_record in curr_frame.get("cameras", {}).items()
            },
        }
        update_scene_for_frame(
            server,
            display_frame,
            camera_frame_handles,
            cube_frame_handles,
            tag_frame_handles,
        )

        camera_record = None if selected_camera_name is None else display_frame.get("cameras", {}).get(selected_camera_name)
        if camera_record is not None:
            vis_bgr = np.asarray(camera_record.get("vis_bgr"), dtype=np.uint8)
            vis_image_handle.image = bgr_to_rgb(vis_bgr)
            gray_image_handle.image = make_gray_overlay_image(camera_record)

        status_text.value = (
            f"frame={frame_idx + 1}/{len(frames)} "
            f"frame_idx_saved={curr_frame.get('frame_idx', -1)} "
            f"ts={float(curr_frame.get('timestamp_epoch_s', 0.0)):.3f}"
        )

        now = time.monotonic()
        step_s = 1.0 / max(float(meta.get("fps", PLAYBACK_FPS)), 1e-6)
        if is_playing and now - last_step_time >= step_s:
            next_idx = frame_idx + 1
            if next_idx >= len(frames):
                if loop_playback:
                    next_idx = 0
                else:
                    next_idx = len(frames) - 1
                    is_playing = False
                    play_checkbox.value = False
            frame_idx = next_idx
            frame_slider.value = frame_idx
            last_step_time = now

        time.sleep(0.01)


if __name__ == "__main__":
    main()
