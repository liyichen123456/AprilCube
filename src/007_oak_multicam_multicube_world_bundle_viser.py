# OpenCV / AprilTag camera convention
# +x: image right
# +y: image down
# +z: camera forward
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import viser
import yaml
from pupil_apriltags import Detector
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

THIS_FILE = Path(__file__).resolve()
THIRDPARTY_DIR = THIS_FILE.parent.parent.parent
PROJECT_ROOT = THIRDPARTY_DIR.parent
UTILS_DIR = PROJECT_ROOT / "scripts" / "utils"

if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

import aprilcube  # noqa: E402
from april_tag_detector import TemporalTagPoseEstimator  # noqa: E402
from aprilcube_runtime import AprilCubeTemporalPoseRuntime, is_valid_rotation_matrix  # noqa: E402
from recorder_oak_cam import OAK1WCameraManager, list_oak_devices  # noqa: E402


# ============================================================
# User macros
# ============================================================

PRINT_AVAILABLE_DEVICES = True

CAMERA_TO_DEVICE: dict[str, str] = {
    "cam0": "3.10.4.4.2",
    "cam1": "3.10.4.4.3",
    "cam2": "3.10.4.4.4",
}

CAMERA_TO_INTRINSICS_YAML: dict[str, str] = {
    "cam0": "/home/ps/RobotCamCalib1/outputs/intrinsics_cam0.yaml",
    "cam1": "/home/ps/RobotCamCalib1/outputs/intrinsics_cam1.yaml",
    "cam2": "/home/ps/RobotCamCalib1/outputs/intrinsics_cam2.yaml",
}

ACTIVE_CAMERA_NAMES: list[str] = ["cam0", "cam1", "cam2"]

DETECT_IMG_SIZE: tuple[int, int] = (1280, 960)
ISP_SCALE: tuple[int, int] = (1, 3)
FPS = 25
QUEUE_SIZE = 4
QUEUE_BLOCKING = False
ROTATE_180_NAMES: set[str] = set()

INIT_TAG_FAMILY = "tagCustom48h12"
INIT_TAG_ID = 0
INIT_TAG_SIZE_M = 0.1
INIT_REQUIRED_SAMPLES_PER_CAMERA = 10

CUBE_CFG_DIRS: list[Path] = [
    THIRDPARTY_DIR / "aprilcube" / "cubes" / "cube_april_36h11_6_11_1x1x1_10mm",
    THIRDPARTY_DIR / "aprilcube" / "cubes" / "cube_april_36h11_12_17_1x1x1_10mm",
    THIRDPARTY_DIR / "aprilcube" / "cubes" / "cube_april_36h11_18_23_1x1x1_10mm",
]

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

PRINT_EVERY_N_FRAMES = 5
UNDISTORT_BEFORE_DETECTION = True
SHOW_CV2_WINDOWS = False
WINDOW_PREFIX = "OAK Multi-AprilCube World Bundle"
PROCESSING_FPS_EMA_ALPHA = 0.2

VISER_HOST = "0.0.0.0"
VISER_PORT = 8080
WORLD_AXES_LENGTH_M = 0.15
WORLD_AXES_RADIUS_M = 0.003
WORLD_ORIGIN_RADIUS_M = 0.004
CAMERA_AXES_LENGTH_M = 0.08
CAMERA_AXES_RADIUS_M = 0.002
CAMERA_ORIGIN_RADIUS_M = 0.003
CUBE_AXES_LENGTH_M = 0.04
CUBE_AXES_RADIUS_M = 0.0015
CUBE_ORIGIN_RADIUS_M = 0.0025
TAG_AXES_LENGTH_M = 0.02
TAG_AXES_RADIUS_M = 0.001
TAG_ORIGIN_RADIUS_M = 0.0015
TEMPORAL_TRANSLATION_GATE_M = 0.01
TEMPORAL_ROTATION_GATE_DEG = 90.0
BUNDLE_LOSS = "huber"
BUNDLE_F_SCALE_PX = 3.0
PRINT_TIMING = True


# ============================================================
# Utilities
# ============================================================

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


def scale_intrinsics(k: np.ndarray, old_size: tuple[int, int], new_size: tuple[int, int]) -> np.ndarray:
    old_w, old_h = old_size
    new_w, new_h = new_size
    sx = new_w / old_w
    sy = new_h / old_h
    k_new = np.asarray(k, dtype=np.float64).copy()
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


def k_to_camera_params(k: np.ndarray) -> tuple[float, float, float, float]:
    return (float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]))


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


def rotation_matrix_to_wxyz(rot: np.ndarray) -> tuple[float, float, float, float]:
    quat_xyzw = R.from_matrix(np.asarray(rot, dtype=np.float64)).as_quat()
    x, y, z, w = quat_xyzw
    return (float(w), float(x), float(y), float(z))


def invert_pose(pose_R: np.ndarray, pose_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose_R = np.asarray(pose_R, dtype=np.float64).reshape(3, 3)
    pose_t = np.asarray(pose_t, dtype=np.float64).reshape(3)
    rot_inv = pose_R.T
    t_inv = -rot_inv @ pose_t
    return rot_inv, t_inv


def compose_pose(
    a_R_b: np.ndarray,
    a_t_b: np.ndarray,
    b_R_c: np.ndarray,
    b_t_c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    a_R_b = np.asarray(a_R_b, dtype=np.float64).reshape(3, 3)
    a_t_b = np.asarray(a_t_b, dtype=np.float64).reshape(3)
    b_R_c = np.asarray(b_R_c, dtype=np.float64).reshape(3, 3)
    b_t_c = np.asarray(b_t_c, dtype=np.float64).reshape(3)
    a_R_c = a_R_b @ b_R_c
    a_t_c = a_R_b @ b_t_c + a_t_b
    return a_R_c, a_t_c


def average_pose_candidates(candidates: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray] | None:
    if not candidates:
        return None
    weights = []
    rot_mats = []
    tvecs = []
    for cand in candidates:
        err = cand.get("reproj_error", None)
        weight = 1.0 if err is None else 1.0 / max(float(err), 1e-3)
        weights.append(weight)
        rot_mats.append(np.asarray(cand["rot_mat"], dtype=np.float64).reshape(3, 3))
        tvecs.append(np.asarray(cand["tvec"], dtype=np.float64).reshape(3))
    weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr /= np.sum(weights_arr)
    rot_avg = AprilCubeTemporalPoseRuntime.average_rotations(rot_mats, weights_arr)
    if rot_avg is None:
        return None
    t_avg = np.zeros(3, dtype=np.float64)
    for weight, tvec in zip(weights_arr, tvecs):
        t_avg += float(weight) * tvec
    return rot_avg, t_avg


def rotation_angle_deg_between(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rot_a = np.asarray(rot_a, dtype=np.float64).reshape(3, 3)
    rot_b = np.asarray(rot_b, dtype=np.float64).reshape(3, 3)
    rel = rot_a.T @ rot_b
    trace = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def add_timing_ms(timing: dict[str, float], key: str, dt_s: float) -> None:
    timing[key] = timing.get(key, 0.0) + float(dt_s) * 1000.0


def format_timing_ms(timing: dict[str, float], keys: list[str]) -> str:
    return " ".join(f"{key}={timing[key]:.1f}ms" for key in keys if key in timing)


def draw_text_panel(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    y = 28
    for line in lines:
        cv2.putText(out, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        y += 24
    return out


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    img_bgr = np.asarray(img_bgr, dtype=np.uint8)
    if img_bgr.ndim == 2:
        return np.repeat(img_bgr[:, :, None], 3, axis=2)
    return img_bgr[:, :, ::-1]


def draw_tag_pose_overlay(img: np.ndarray, tag: Any, k: np.ndarray, axis_length_m: float = 0.05) -> np.ndarray:
    out = img.copy()
    corners = np.round(np.asarray(tag.corners, dtype=np.float64)).astype(np.int32)
    cv2.polylines(out, [corners], True, (0, 255, 0), 2)
    cv2.putText(out, f"ID:{int(tag.tag_id)}", (int(tag.center[0]) - 12, int(tag.center[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    pose_R = np.asarray(tag.pose_R, dtype=np.float64).reshape(3, 3)
    pose_t = np.asarray(tag.pose_t, dtype=np.float64).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(pose_R)
    obj_pts = np.array([[0.0, 0.0, 0.0], [axis_length_m, 0.0, 0.0], [0.0, axis_length_m, 0.0], [0.0, 0.0, axis_length_m]], dtype=np.float64)
    img_pts, _ = cv2.projectPoints(obj_pts, rvec, pose_t, np.asarray(k, dtype=np.float64), np.zeros(5))
    img_pts = np.round(img_pts.reshape(-1, 2)).astype(np.int32)
    cv2.arrowedLine(out, tuple(img_pts[0]), tuple(img_pts[1]), (0, 0, 255), 3, tipLength=0.2)
    cv2.arrowedLine(out, tuple(img_pts[0]), tuple(img_pts[2]), (0, 255, 0), 3, tipLength=0.2)
    cv2.arrowedLine(out, tuple(img_pts[0]), tuple(img_pts[3]), (255, 0, 0), 3, tipLength=0.2)
    return out


def detect_target_tag(detector: Detector, image_bgr: np.ndarray, camera_params: tuple[float, float, float, float]) -> Any | None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=INIT_TAG_SIZE_M)
    matches = [tag for tag in tags if int(tag.tag_id) == INIT_TAG_ID]
    if len(matches) != 1:
        return None
    return matches[0]


def validate_cube_path(cube_path: Path) -> Path:
    cube_path = cube_path.resolve()
    if cube_path.is_dir() and (cube_path / "config.json").is_file():
        return cube_path
    if cube_path.is_file() and cube_path.name == "config.json":
        return cube_path
    raise FileNotFoundError(f"Invalid AprilCube cfg path: {cube_path}")


def create_detector_for_camera(cube_path: Path, camera_name: str, calib_by_camera: dict[str, dict[str, Any]]) -> Any:
    calib = calib_by_camera[camera_name]
    k_scaled = scale_intrinsics(calib["K"], old_size=tuple(calib["image_size"]), new_size=DETECT_IMG_SIZE)
    intrinsic_cfg = camera_matrix_to_intrinsic_dict(k_scaled)
    dist_coeffs = np.asarray(calib["dist"], dtype=np.float64)
    detector_dist_coeffs = np.zeros(5, dtype=np.float64) if UNDISTORT_BEFORE_DETECTION else dist_coeffs
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


def build_cube_runtimes(
    opened_names: list[str],
    calib_by_camera: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[int]]]:
    runtimes_by_camera: dict[str, list[dict[str, Any]]] = {name: [] for name in opened_names}
    shared_native_detectors: dict[str, Detector] = {}
    cube_tag_ids_by_name: dict[str, list[int]] = {}

    for cube_path_raw in CUBE_CFG_DIRS:
        cube_path = validate_cube_path(cube_path_raw)
        cube_name = cube_path.name if cube_path.is_dir() else cube_path.parent.name
        for camera_name in opened_names:
            detector = create_detector_for_camera(cube_path, camera_name, calib_by_camera)
            native_family = apriltag_family_from_dict_name(detector.config.dict_name)
            if native_family not in shared_native_detectors:
                shared_native_detectors[native_family] = Detector(families=native_family, quad_decimate=1.0)
            runtime = AprilCubeTemporalPoseRuntime(
                detector=detector,
                native_detector=shared_native_detectors[native_family],
                pose_estimator=create_pose_estimator(detector),
            )
            runtimes_by_camera[camera_name].append({"cube_name": cube_name, "runtime": runtime, "detector": detector})
            cube_tag_ids_by_name[cube_name] = sorted(int(v) for v in detector.valid_ids)
            print(f"[INFO] Loaded cube cfg for {camera_name}: {cube_name}")

    return runtimes_by_camera, cube_tag_ids_by_name


def create_viser_server(
    cube_tag_ids_by_name: dict[str, list[int]],
) -> tuple[viser.ViserServer, dict[str, Any], dict[str, Any], dict[str, dict[int, Any]], dict[str, Any]]:
    server = viser.ViserServer(host=VISER_HOST, port=VISER_PORT)
    server.scene.set_up_direction("-y")
    server.scene.world_axes.visible = False

    camera_frame_handles: dict[str, Any] = {}
    cube_frame_handles: dict[str, Any] = {}
    cube_tag_frame_handles: dict[str, dict[int, Any]] = {}
    image_handles: dict[str, Any] = {}

    server.scene.add_frame("/world", wxyz=(1.0, 0.0, 0.0, 0.0), position=(0.0, 0.0, 0.0), axes_length=WORLD_AXES_LENGTH_M, axes_radius=WORLD_AXES_RADIUS_M, origin_radius=WORLD_ORIGIN_RADIUS_M)
    server.scene.add_frame("/tag_world", wxyz=(1.0, 0.0, 0.0, 0.0), position=(0.0, 0.0, 0.0), axes_length=WORLD_AXES_LENGTH_M, axes_radius=WORLD_AXES_RADIUS_M, origin_radius=WORLD_ORIGIN_RADIUS_M)

    for camera_name in ACTIVE_CAMERA_NAMES:
        camera_frame_handles[camera_name] = server.scene.add_frame(
            f"/camera/{camera_name}",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            axes_length=CAMERA_AXES_LENGTH_M,
            axes_radius=CAMERA_AXES_RADIUS_M,
            origin_radius=CAMERA_ORIGIN_RADIUS_M,
            visible=False,
        )
        image_handles[camera_name] = server.gui.add_image(np.zeros((120, 160, 3), dtype=np.uint8), label=camera_name)

    for cube_name, tag_ids in cube_tag_ids_by_name.items():
        cube_frame_handles[cube_name] = server.scene.add_frame(
            f"/cube/{cube_name}",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            axes_length=CUBE_AXES_LENGTH_M,
            axes_radius=CUBE_AXES_RADIUS_M,
            origin_radius=CUBE_ORIGIN_RADIUS_M,
            visible=False,
        )
        cube_tag_frame_handles[cube_name] = {}
        for tag_id in tag_ids:
            cube_tag_frame_handles[cube_name][int(tag_id)] = server.scene.add_frame(
                f"/cube/{cube_name}/tag_{int(tag_id)}",
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
                axes_length=TAG_AXES_LENGTH_M,
                axes_radius=TAG_AXES_RADIUS_M,
                origin_radius=TAG_ORIGIN_RADIUS_M,
                visible=False,
            )

    print(f"[INFO] Viser server started on http://{VISER_HOST}:{VISER_PORT}")
    print("[INFO] World frame is fixed from the initialization AprilTag.")
    return server, camera_frame_handles, cube_frame_handles, cube_tag_frame_handles, image_handles


def initialize_world_from_tag(
    camera_manager: OAK1WCameraManager,
    opened_names: list[str],
    calib_by_camera: dict[str, dict[str, Any]],
    tag_detector: Detector,
    camera_frame_handles: dict[str, Any],
    image_handles: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    world_from_camera_samples: dict[str, list[dict[str, np.ndarray]]] = {name: [] for name in opened_names}
    last_print_time = 0.0
    print("[INFO] Initialization phase started: waiting for the fixed world AprilTag.")

    while True:
        frames, _origin_frames = camera_manager.get_frames(camera_names=opened_names, img_size=DETECT_IMG_SIZE)
        if not frames:
            time.sleep(0.02)
            continue

        completed = True
        for camera_name in opened_names:
            frame = frames.get(camera_name)
            if frame is None:
                completed = False
                continue

            calib = calib_by_camera[camera_name]
            k_scaled = scale_intrinsics(calib["K"], old_size=tuple(calib["image_size"]), new_size=DETECT_IMG_SIZE)
            dist = np.asarray(calib["dist"], dtype=np.float64).reshape(-1)
            detect_frame = cv2.undistort(frame, k_scaled, dist) if UNDISTORT_BEFORE_DETECTION else frame
            tag = detect_target_tag(detector=tag_detector, image_bgr=detect_frame, camera_params=k_to_camera_params(k_scaled))

            if tag is not None:
                pose_R = np.asarray(tag.pose_R, dtype=np.float64).reshape(3, 3)
                pose_t = np.asarray(tag.pose_t, dtype=np.float64).reshape(3)
                if is_valid_rotation_matrix(pose_R):
                    world_R_cam, world_t_cam = invert_pose(pose_R, pose_t)
                    world_from_camera_samples[camera_name].append({"rot_mat": world_R_cam, "tvec": world_t_cam})
                    camera_frame_handles[camera_name].wxyz = rotation_matrix_to_wxyz(world_R_cam)
                    camera_frame_handles[camera_name].position = (float(world_t_cam[0]), float(world_t_cam[1]), float(world_t_cam[2]))
                    camera_frame_handles[camera_name].visible = True
                    detect_frame = draw_tag_pose_overlay(detect_frame, tag, k_scaled)
            else:
                completed = False

            samples = len(world_from_camera_samples[camera_name])
            if samples < INIT_REQUIRED_SAMPLES_PER_CAMERA:
                completed = False

            overlay = draw_text_panel(detect_frame, [f"[init][{camera_name}] samples={samples}/{INIT_REQUIRED_SAMPLES_PER_CAMERA}", f"family={INIT_TAG_FAMILY} id={INIT_TAG_ID} size_m={INIT_TAG_SIZE_M}"])
            image_handles[camera_name].image = bgr_to_rgb(overlay)
            if SHOW_CV2_WINDOWS:
                cv2.imshow(f"{WINDOW_PREFIX} init: {camera_name}", overlay)

        now = time.time()
        if now - last_print_time > 1.0:
            progress = ", ".join(f"{name}:{len(world_from_camera_samples[name])}/{INIT_REQUIRED_SAMPLES_PER_CAMERA}" for name in opened_names)
            print(f"[INFO] init progress: {progress}")
            last_print_time = now

        if completed:
            break

        key = cv2.waitKey(1)
        if key in (27, ord("q")):
            raise KeyboardInterrupt

    world_from_camera: dict[str, dict[str, np.ndarray]] = {}
    for camera_name in opened_names:
        fused = average_pose_candidates(world_from_camera_samples[camera_name])
        if fused is None:
            raise RuntimeError(f"Failed to fuse initialization poses for {camera_name}")
        rot_mat, tvec = fused
        world_from_camera[camera_name] = {"rot_mat": rot_mat, "tvec": tvec}
        camera_frame_handles[camera_name].wxyz = rotation_matrix_to_wxyz(rot_mat)
        camera_frame_handles[camera_name].position = (float(tvec[0]), float(tvec[1]), float(tvec[2]))
        camera_frame_handles[camera_name].visible = True
        print(f"[INFO] [{camera_name}] fixed world pose from init: t_m=({tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f})")

    print("[INFO] Initialization finished. World/camera extrinsics are now treated as time-invariant.")
    return world_from_camera


def pack_pose_params(world_R_cube: np.ndarray, world_t_cube: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.asarray(world_R_cube, dtype=np.float64).reshape(3, 3))
    tvec = np.asarray(world_t_cube, dtype=np.float64).reshape(3)
    return np.concatenate([rvec.reshape(3), tvec.reshape(3)], axis=0)


def unpack_pose_params(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    params = np.asarray(params, dtype=np.float64).reshape(6)
    rvec = params[:3].reshape(3, 1)
    tvec = params[3:].reshape(3)
    rot_mat, _ = cv2.Rodrigues(rvec)
    return rot_mat, tvec


def world_cube_residuals(
    params: np.ndarray,
    observations: list[dict[str, Any]],
    detector: Any,
    world_from_camera: dict[str, dict[str, np.ndarray]],
    scaled_k_by_camera: dict[str, np.ndarray],
    pnp_dist_by_camera: dict[str, np.ndarray],
) -> np.ndarray:
    world_R_cube, world_t_cube = unpack_pose_params(params)
    if not is_valid_rotation_matrix(world_R_cube):
        return np.full((len(observations) * 8,), 1e6, dtype=np.float64)

    residuals: list[np.ndarray] = []
    for obs in observations:
        camera_name = str(obs["camera_name"])
        tag_id = int(obs["tag_id"])
        corners_2d = np.asarray(obs["corners_2d"], dtype=np.float64).reshape(4, 2)
        obj_pts_cube_m = np.asarray(detector.tag_corner_map[tag_id], dtype=np.float64).reshape(4, 3) / 1000.0

        world_R_cam = world_from_camera[camera_name]["rot_mat"]
        world_t_cam = world_from_camera[camera_name]["tvec"]
        cam_R_world, cam_t_world = invert_pose(world_R_cam, world_t_cam)
        cam_R_cube, cam_t_cube = compose_pose(cam_R_world, cam_t_world, world_R_cube, world_t_cube)
        cam_rvec_cube, _ = cv2.Rodrigues(np.asarray(cam_R_cube, dtype=np.float64).reshape(3, 3))
        proj, _ = cv2.projectPoints(
            objectPoints=obj_pts_cube_m,
            rvec=cam_rvec_cube,
            tvec=np.asarray(cam_t_cube, dtype=np.float64).reshape(3, 1),
            cameraMatrix=np.asarray(scaled_k_by_camera[camera_name], dtype=np.float64),
            distCoeffs=np.asarray(pnp_dist_by_camera[camera_name], dtype=np.float64).reshape(-1, 1),
        )
        proj = proj.reshape(-1, 2)
        residuals.append((proj - corners_2d).reshape(-1))

    if not residuals:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(residuals, axis=0).astype(np.float64)


def optimize_world_cube_pose(
    cube_name: str,
    detector: Any,
    observations: list[dict[str, Any]],
    world_from_camera: dict[str, dict[str, np.ndarray]],
    scaled_k_by_camera: dict[str, np.ndarray],
    pnp_dist_by_camera: dict[str, np.ndarray],
    init_pose: dict[str, np.ndarray] | None,
    previous_pose: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    if not observations:
        return None
    if init_pose is None:
        init_pose = previous_pose
    if init_pose is None:
        return None

    init_params = pack_pose_params(init_pose["rot_mat"], init_pose["tvec"])
    try:
        opt = least_squares(
            world_cube_residuals,
            init_params,
            method="trf",
            loss=BUNDLE_LOSS,
            f_scale=float(BUNDLE_F_SCALE_PX),
            args=(observations, detector, world_from_camera, scaled_k_by_camera, pnp_dist_by_camera),
            max_nfev=100,
        )
    except Exception as exc:
        print(f"[WARNING] [{cube_name}] bundle optimization failed: {type(exc).__name__}: {exc}")
        return None

    world_R_cube, world_t_cube = unpack_pose_params(opt.x)
    if not is_valid_rotation_matrix(world_R_cube):
        return None

    residual = world_cube_residuals(opt.x, observations, detector, world_from_camera, scaled_k_by_camera, pnp_dist_by_camera)
    reproj_error = float(np.mean(np.linalg.norm(residual.reshape(-1, 2), axis=1))) if residual.size else float("inf")

    if previous_pose is not None:
        trans_delta_m = float(np.linalg.norm(world_t_cube - np.asarray(previous_pose["tvec"], dtype=np.float64).reshape(3)))
        rot_delta_deg = rotation_angle_deg_between(previous_pose["rot_mat"], world_R_cube)
        if trans_delta_m > float(TEMPORAL_TRANSLATION_GATE_M) or rot_delta_deg > float(TEMPORAL_ROTATION_GATE_DEG):
            return (
                np.asarray(previous_pose["rot_mat"], dtype=np.float64).reshape(3, 3),
                np.asarray(previous_pose["tvec"], dtype=np.float64).reshape(3),
                {
                    "reproj_error": reproj_error,
                    "fallback_to_previous": True,
                    "trans_delta_m": trans_delta_m,
                    "rot_delta_deg": rot_delta_deg,
                    "num_observations": len(observations),
                    "num_residual_terms": int(residual.size),
                    "optimizer_cost": float(opt.cost),
                    "optimizer_success": bool(opt.success),
                    "optimizer_message": str(opt.message),
                },
            )

    return (
        world_R_cube,
        world_t_cube,
        {
            "reproj_error": reproj_error,
            "fallback_to_previous": False,
            "num_observations": len(observations),
            "num_residual_terms": int(residual.size),
            "optimizer_cost": float(opt.cost),
            "optimizer_success": bool(opt.success),
            "optimizer_message": str(opt.message),
        },
    )


def main() -> None:
    if PRINT_AVAILABLE_DEVICES:
        list_oak_devices()

    missing_devices = [name for name in ACTIVE_CAMERA_NAMES if name not in CAMERA_TO_DEVICE]
    if missing_devices:
        print(f"[ERROR] Missing CAMERA_TO_DEVICE entries for: {missing_devices}")
        sys.exit(1)
    missing_intrinsics = [name for name in ACTIVE_CAMERA_NAMES if name not in CAMERA_TO_INTRINSICS_YAML]
    if missing_intrinsics:
        print(f"[ERROR] Missing CAMERA_TO_INTRINSICS_YAML entries for: {missing_intrinsics}")
        sys.exit(1)

    calib_by_camera = {name: load_intrinsics_yaml(CAMERA_TO_INTRINSICS_YAML[name]) for name in ACTIVE_CAMERA_NAMES}
    for name, calib in calib_by_camera.items():
        print(f"[INFO] [{name}] intrinsics_yaml={calib['path']} image_size={calib['image_size']}")

    init_tag_detector = Detector(families=INIT_TAG_FAMILY, quad_decimate=1.0)
    print(f"[INFO] Init AprilTag detector: family={INIT_TAG_FAMILY}, id={INIT_TAG_ID}, size_m={INIT_TAG_SIZE_M}")

    camera_manager = OAK1WCameraManager(
        camera_to_device={name: CAMERA_TO_DEVICE[name] for name in ACTIVE_CAMERA_NAMES},
        isp_scale=ISP_SCALE,
        fps=FPS,
        rotate_180_names=ROTATE_180_NAMES,
        queue_size=QUEUE_SIZE,
        queue_blocking=QUEUE_BLOCKING,
    )

    try:
        opened = camera_manager.open_all_cameras()
        if opened != len(ACTIVE_CAMERA_NAMES):
            print(f"[ERROR] Need all cameras opened for initialization, got {opened}/{len(ACTIVE_CAMERA_NAMES)}.")
            sys.exit(1)

        opened_names = camera_manager.get_active_camera_names()
        print(f"[INFO] Opened OAK cameras: {opened_names}")
        camera_manager.wait_for_first_frames(camera_names=opened_names, timeout_s=5.0)

        runtimes_by_camera, cube_tag_ids_by_name = build_cube_runtimes(opened_names, calib_by_camera)
        cube_runtime_by_name: dict[str, AprilCubeTemporalPoseRuntime] = {entry["cube_name"]: entry["runtime"] for entry in runtimes_by_camera[opened_names[0]]}
        detector_by_cube_name: dict[str, Any] = {entry["cube_name"]: entry["detector"] for entry in runtimes_by_camera[opened_names[0]]}

        server, camera_frame_handles, cube_frame_handles, cube_tag_frame_handles, image_handles = create_viser_server(cube_tag_ids_by_name)

        world_from_camera = initialize_world_from_tag(
            camera_manager=camera_manager,
            opened_names=opened_names,
            calib_by_camera=calib_by_camera,
            tag_detector=init_tag_detector,
            camera_frame_handles=camera_frame_handles,
            image_handles=image_handles,
        )

        scaled_k_by_camera: dict[str, np.ndarray] = {}
        undistort_dist_by_camera: dict[str, np.ndarray] = {}
        pnp_dist_by_camera: dict[str, np.ndarray] = {}
        for camera_name in opened_names:
            calib = calib_by_camera[camera_name]
            scaled_k_by_camera[camera_name] = scale_intrinsics(calib["K"], old_size=tuple(calib["image_size"]), new_size=DETECT_IMG_SIZE)
            undistort_dist_by_camera[camera_name] = np.asarray(calib["dist"], dtype=np.float64).reshape(-1)
            pnp_dist_by_camera[camera_name] = (
                np.zeros(5, dtype=np.float64)
                if UNDISTORT_BEFORE_DETECTION
                else np.asarray(calib["dist"], dtype=np.float64).reshape(-1)
            )

        print("[INFO] Switched to multi-cube world bundle detection.")
        print("[INFO] Press q or ESC to quit.")
        last_world_cube_pose_by_name: dict[str, dict[str, np.ndarray]] = {}
        processing_fps_ema: float | None = None
        frame_idx = 0
        last_no_frame_print_time = 0.0

        while True:
            frame_start_time = time.perf_counter()
            timing_ms: dict[str, float] = {}
            frame_idx += 1

            t0 = time.perf_counter()
            frames, _origin_frames = camera_manager.get_frames(camera_names=opened_names, img_size=DETECT_IMG_SIZE)
            add_timing_ms(timing_ms, "get_frames", time.perf_counter() - t0)

            if not frames:
                now = time.time()
                if now - last_no_frame_print_time > 1.0:
                    print("[INFO] No frames received yet.")
                    last_no_frame_print_time = now
                if cv2.waitKey(1) in (27, ord("q")):
                    break
                continue

            observations_by_cube: dict[str, list[dict[str, Any]]] = {cube_name: [] for cube_name in cube_runtime_by_name}
            init_candidates_by_cube: dict[str, list[dict[str, Any]]] = {cube_name: [] for cube_name in cube_runtime_by_name}
            observed_tag_ids_by_cube: dict[str, set[int]] = {cube_name: set() for cube_name in cube_runtime_by_name}

            for camera_name in opened_names:
                camera_loop_start = time.perf_counter()
                runtime_entries = runtimes_by_camera[camera_name]
                frame = frames.get(camera_name)
                if frame is None:
                    image_handles[camera_name].image = np.zeros((120, 160, 3), dtype=np.uint8)
                    continue

                detect_frame = frame
                if UNDISTORT_BEFORE_DETECTION:
                    t0 = time.perf_counter()
                    detect_frame = cv2.undistort(frame, scaled_k_by_camera[camera_name], undistort_dist_by_camera[camera_name])
                    add_timing_ms(timing_ms, f"{camera_name}.undistort", time.perf_counter() - t0)

                vis = detect_frame.copy()
                status_lines = [f"[{camera_name}] cubes={len(runtime_entries)} detect_size={DETECT_IMG_SIZE} fps={FPS}"]

                grouped_entries: dict[tuple[str, float], list[dict[str, Any]]] = {}
                for entry in runtime_entries:
                    runtime = entry["runtime"]
                    grouped_entries.setdefault((runtime.native_family, round(runtime.tag_size_m, 6)), []).append(entry)

                for _group_key, group_entries in grouped_entries.items():
                    t0 = time.perf_counter()
                    shared_tags = group_entries[0]["runtime"].detect_native_apriltags_all(detect_frame)
                    add_timing_ms(timing_ms, f"{camera_name}.shared_detect", time.perf_counter() - t0)

                    for entry in group_entries:
                        cube_name = entry["cube_name"]
                        detector = entry["detector"]
                        runtime = entry["runtime"]

                        t0 = time.perf_counter()
                        result = runtime.process_frame(camera_name=camera_name, image=detect_frame, native_tags=shared_tags)
                        add_timing_ms(timing_ms, f"{camera_name}.process_frame", time.perf_counter() - t0)

                        t0 = time.perf_counter()
                        vis = detector.draw_result(vis, result)
                        add_timing_ms(timing_ms, f"{camera_name}.draw_cube", time.perf_counter() - t0)

                        t0 = time.perf_counter()
                        vis = runtime.draw_detected_tag_visuals(img=vis, result=result, draw_tag_frame_2d=True, tag_axis_length_scale=0.8)
                        add_timing_ms(timing_ms, f"{camera_name}.draw_tags", time.perf_counter() - t0)

                        if result.get("success", False):
                            cam_R_cube, _ = cv2.Rodrigues(np.asarray(result["rvec"], dtype=np.float64).reshape(3, 1))
                            cam_t_cube = np.asarray(result["tvec"], dtype=np.float64).reshape(3) / 1000.0
                            world_R_cam = world_from_camera[camera_name]["rot_mat"]
                            world_t_cam = world_from_camera[camera_name]["tvec"]
                            world_R_cube, world_t_cube = compose_pose(world_R_cam, world_t_cam, cam_R_cube, cam_t_cube)
                            init_candidates_by_cube[cube_name].append(
                                {
                                    "rot_mat": world_R_cube,
                                    "tvec": world_t_cube,
                                    "reproj_error": result.get("reproj_error", None),
                                    "camera_name": camera_name,
                                }
                            )
                            line = (
                                f"[{camera_name}][{cube_name}] success init_t_world_m="
                                f"({world_t_cube[0]:.3f}, {world_t_cube[1]:.3f}, {world_t_cube[2]:.3f}) "
                                f"reproj={float(result.get('reproj_error', float('nan'))):.2f}px"
                            )
                        else:
                            line = f"[{camera_name}][{cube_name}] cube not detected"

                        for tag in shared_tags:
                            tag_id = int(tag.tag_id)
                            if tag_id not in detector.valid_ids:
                                continue
                            observations_by_cube[cube_name].append(
                                {
                                    "camera_name": camera_name,
                                    "tag_id": tag_id,
                                    "corners_2d": np.asarray(tag.corners, dtype=np.float64).reshape(4, 2),
                                }
                            )
                            observed_tag_ids_by_cube[cube_name].add(tag_id)

                        status_lines.append(line)
                        if frame_idx % PRINT_EVERY_N_FRAMES == 0:
                            print(line)

                status_lines.append("press q or ESC to quit")
                t0 = time.perf_counter()
                vis = draw_text_panel(vis, status_lines)
                image_handles[camera_name].image = bgr_to_rgb(vis)
                add_timing_ms(timing_ms, f"{camera_name}.overlay", time.perf_counter() - t0)

                if SHOW_CV2_WINDOWS:
                    cv2.imshow(f"{WINDOW_PREFIX}: {camera_name}", vis)
                add_timing_ms(timing_ms, f"{camera_name}.total", time.perf_counter() - camera_loop_start)

            t0 = time.perf_counter()
            for cube_name, observations in observations_by_cube.items():
                cube_handle = cube_frame_handles[cube_name]
                detector = detector_by_cube_name[cube_name]
                for tag_handle in cube_tag_frame_handles[cube_name].values():
                    tag_handle.visible = False

                init_pose = average_pose_candidates(init_candidates_by_cube[cube_name]) if init_candidates_by_cube[cube_name] else None
                previous_pose = last_world_cube_pose_by_name.get(cube_name)
                optimized = optimize_world_cube_pose(
                    cube_name=cube_name,
                    detector=detector,
                    observations=observations,
                    world_from_camera=world_from_camera,
                    scaled_k_by_camera=scaled_k_by_camera,
                    pnp_dist_by_camera=pnp_dist_by_camera,
                    init_pose=init_pose,
                    previous_pose=previous_pose,
                )

                if frame_idx % PRINT_EVERY_N_FRAMES == 0:
                    obs_sources = [f"{obs['camera_name']}/tag{obs['tag_id']}" for obs in observations]
                    print(f"[world][{cube_name}] observations={len(observations)} sources={obs_sources}")

                if optimized is None:
                    cube_handle.visible = False
                    if frame_idx % PRINT_EVERY_N_FRAMES == 0:
                        print(f"[world][{cube_name}] bundle failed: no valid optimized world pose")
                    continue

                world_R_cube, world_t_cube, debug = optimized
                last_world_cube_pose_by_name[cube_name] = {"rot_mat": world_R_cube, "tvec": world_t_cube}
                cube_handle.wxyz = rotation_matrix_to_wxyz(world_R_cube)
                cube_handle.position = (float(world_t_cube[0]), float(world_t_cube[1]), float(world_t_cube[2]))
                cube_handle.visible = True

                if frame_idx % PRINT_EVERY_N_FRAMES == 0:
                    print(
                        f"[world][{cube_name}] optimized_t_m=({world_t_cube[0]:.3f}, {world_t_cube[1]:.3f}, {world_t_cube[2]:.3f}) "
                        f"reproj={float(debug['reproj_error']):.2f}px fallback_prev={bool(debug['fallback_to_previous'])} "
                        f"obs={debug['num_observations']} cost={debug['optimizer_cost']:.3f}"
                    )

                for tag_id in sorted(observed_tag_ids_by_cube[cube_name]):
                    tag_pose = cube_runtime_by_name[cube_name].build_tag_to_cube_transform(int(tag_id))
                    if tag_pose is None:
                        continue
                    rot_cube_tag, center_cube_mm = tag_pose
                    world_R_tag = world_R_cube @ rot_cube_tag
                    world_t_tag = world_R_cube @ (center_cube_mm.reshape(3) / 1000.0) + world_t_cube
                    tag_handle = cube_tag_frame_handles[cube_name][int(tag_id)]
                    tag_handle.wxyz = rotation_matrix_to_wxyz(world_R_tag)
                    tag_handle.position = (float(world_t_tag[0]), float(world_t_tag[1]), float(world_t_tag[2]))
                    tag_handle.visible = True

            add_timing_ms(timing_ms, "world_bundle", time.perf_counter() - t0)

            t0 = time.perf_counter()
            key = cv2.waitKey(1)
            add_timing_ms(timing_ms, "wait_key", time.perf_counter() - t0)
            add_timing_ms(timing_ms, "frame_total", time.perf_counter() - frame_start_time)

            frame_total_ms = timing_ms.get("frame_total", 0.0)
            if frame_total_ms > 1e-6:
                proc_fps = 1000.0 / frame_total_ms
                if processing_fps_ema is None:
                    processing_fps_ema = proc_fps
                else:
                    alpha = float(PROCESSING_FPS_EMA_ALPHA)
                    processing_fps_ema = (1.0 - alpha) * processing_fps_ema + alpha * proc_fps

            if PRINT_TIMING and frame_idx % PRINT_EVERY_N_FRAMES == 0:
                summary_keys = ["get_frames", "world_bundle", "wait_key", "frame_total"]
                proc_fps_str = f" proc_fps={1000.0 / frame_total_ms:.2f} ema_fps={processing_fps_ema:.2f}" if frame_total_ms > 1e-6 else ""
                print(f"[timing][frame={frame_idx}] {format_timing_ms(timing_ms, summary_keys)}{proc_fps_str}")
                for camera_name in opened_names:
                    camera_keys = [
                        f"{camera_name}.undistort",
                        f"{camera_name}.shared_detect",
                        f"{camera_name}.process_frame",
                        f"{camera_name}.draw_cube",
                        f"{camera_name}.draw_tags",
                        f"{camera_name}.overlay",
                        f"{camera_name}.total",
                    ]
                    camera_line = format_timing_ms(timing_ms, camera_keys)
                    if camera_line:
                        print(f"[timing][frame={frame_idx}][{camera_name}] {camera_line}")

            if key in (27, ord("q")):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        camera_manager.release_all()
        cv2.destroyAllWindows()
        print("[INFO] Finished.")


if __name__ == "__main__":
    main()
