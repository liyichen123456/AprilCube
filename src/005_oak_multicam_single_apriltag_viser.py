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
from scipy.spatial.transform import Rotation as R

THIS_FILE = Path(__file__).resolve()
THIRDPARTY_DIR = THIS_FILE.parent.parent.parent
PROJECT_ROOT = THIRDPARTY_DIR.parent
UTILS_DIR = PROJECT_ROOT / "scripts" / "utils"

if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

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

# Keep per-camera yaml path interface, but currently all three share one file.
CAMERA_TO_INTRINSICS_YAML: dict[str, str] = {
    "cam0": "/home/ps/RobotCamCalib1/outputs/intrinsics_r_wrist.yaml",
    "cam1": "/home/ps/RobotCamCalib1/outputs/intrinsics_r_wrist.yaml",
    "cam2": "/home/ps/RobotCamCalib1/outputs/intrinsics_r_wrist.yaml",
}

ACTIVE_CAMERA_NAMES: list[str] = ["cam0", "cam1", "cam2"]

DETECT_IMG_SIZE: tuple[int, int] = (1280, 960)  # width, height
ISP_SCALE: tuple[int, int] = (1, 3)
FPS = 25
QUEUE_SIZE = 4
QUEUE_BLOCKING = False
ROTATE_180_NAMES: set[str] = set()

TAG_FAMILY = "tagCustom48h12"
TAG_ID = 0
TAG_SIZE_M = 0.1

PRINT_EVERY_N_FRAMES = 5
SHOW_CV2_WINDOWS = False
WINDOW_PREFIX = "OAK AprilTag World"
UNDISTORT_BEFORE_DETECTION = True

VISER_HOST = "0.0.0.0"
VISER_PORT = 8080
WORLD_AXES_LENGTH_M = 0.12
WORLD_AXES_RADIUS_M = 0.003
WORLD_ORIGIN_RADIUS_M = 0.004
CAMERA_AXES_LENGTH_M = 0.08
CAMERA_AXES_RADIUS_M = 0.002
CAMERA_ORIGIN_RADIUS_M = 0.003


# ============================================================
# Utilities
# ============================================================

def load_intrinsics_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path).expanduser().resolve()
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    image_size = tuple(int(v) for v in data["image_size"])
    k = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(data["dist"], dtype=np.float64).reshape(-1)
    return {
        "path": str(yaml_path),
        "image_size": image_size,
        "K": k,
        "dist": dist,
    }


def scale_intrinsics(
    k: np.ndarray,
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> np.ndarray:
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)

    k_scaled = np.asarray(k, dtype=np.float64).copy()
    k_scaled[0, 0] *= sx
    k_scaled[1, 1] *= sy
    k_scaled[0, 2] *= sx
    k_scaled[1, 2] *= sy
    return k_scaled


def k_to_camera_params(k: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(k[0, 0]),
        float(k[1, 1]),
        float(k[0, 2]),
        float(k[1, 2]),
    )


def is_valid_rotation_matrix(rot: np.ndarray, det_tol: float = 0.2) -> bool:
    rot = np.asarray(rot, dtype=np.float64)
    if rot.shape != (3, 3):
        return False
    if not np.all(np.isfinite(rot)):
        return False

    det = float(np.linalg.det(rot))
    if det <= 0.0 or abs(det - 1.0) > det_tol:
        return False

    ortho_err = float(np.linalg.norm(rot.T @ rot - np.eye(3)))
    return ortho_err <= 0.2


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


def make_status_text(camera_name: str, pose_ok: bool, pose_t_world_cam: np.ndarray | None) -> str:
    if not pose_ok or pose_t_world_cam is None:
        return f"[{camera_name}] tag not detected"

    t = np.asarray(pose_t_world_cam, dtype=np.float64).reshape(3)
    return f"[{camera_name}] world_cam_t_m=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})"


def create_viser_server() -> tuple[viser.ViserServer, dict[str, Any]]:
    server = viser.ViserServer(host=VISER_HOST, port=VISER_PORT)
    server.scene.set_up_direction("-y")
    server.scene.world_axes.visible = False

    handles: dict[str, Any] = {}
    handles["tag_world"] = server.scene.add_frame(
        "/tag_world",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        axes_length=WORLD_AXES_LENGTH_M,
        axes_radius=WORLD_AXES_RADIUS_M,
        origin_radius=WORLD_ORIGIN_RADIUS_M,
    )

    for camera_name in ACTIVE_CAMERA_NAMES:
        handles[camera_name] = server.scene.add_frame(
            f"/camera/{camera_name}",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            axes_length=CAMERA_AXES_LENGTH_M,
            axes_radius=CAMERA_AXES_RADIUS_M,
            origin_radius=CAMERA_ORIGIN_RADIUS_M,
            visible=False,
        )

    print(f"[INFO] Viser server started on http://{VISER_HOST}:{VISER_PORT}")
    print("[INFO] World frame is the observed AprilTag frame.")
    return server, handles


def detect_target_tag(
    detector: Detector,
    image_bgr: np.ndarray,
    camera_params: tuple[float, float, float, float],
) -> Any | None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=TAG_SIZE_M,
    )

    matches = [tag for tag in tags if int(tag.tag_id) == TAG_ID]
    if len(matches) != 1:
        return None
    return matches[0]


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

    calib_by_camera = {
        name: load_intrinsics_yaml(CAMERA_TO_INTRINSICS_YAML[name])
        for name in ACTIVE_CAMERA_NAMES
    }
    for name, calib in calib_by_camera.items():
        print(f"[INFO] [{name}] intrinsics_yaml={calib['path']} image_size={calib['image_size']}")

    tag_detector = Detector(
        families=TAG_FAMILY,
        quad_decimate=1.0,
    )
    print(f"[INFO] AprilTag detector initialized with family={TAG_FAMILY}, id={TAG_ID}, size_m={TAG_SIZE_M}")

    server, frame_handles = create_viser_server()

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
        if opened == 0:
            print("[ERROR] No OAK camera opened.")
            sys.exit(1)

        active_camera_names = camera_manager.get_active_camera_names()
        print(f"[INFO] Opened OAK cameras: {active_camera_names}")
        print("[INFO] Waiting for first frames...")
        camera_manager.wait_for_first_frames(camera_names=active_camera_names, timeout_s=5.0)
        print("[INFO] Press q or ESC to quit.")

        frame_idx = 0
        last_no_frame_print_time = 0.0

        while True:
            frame_idx += 1
            frames, _origin_frames = camera_manager.get_frames(
                camera_names=active_camera_names,
                img_size=DETECT_IMG_SIZE,
            )

            if not frames:
                now = time.time()
                if now - last_no_frame_print_time > 1.0:
                    print("[INFO] No frames received yet.")
                    last_no_frame_print_time = now
                if cv2.waitKey(1) in (27, ord("q")):
                    break
                continue

            visible_count = 0
            status_lines: list[str] = []

            for camera_name in active_camera_names:
                frame_handle = frame_handles[camera_name]
                frame = frames.get(camera_name)
                if frame is None:
                    frame_handle.visible = False
                    status_lines.append(f"[{camera_name}] no frame")
                    continue

                calib = calib_by_camera[camera_name]
                k_scaled = scale_intrinsics(
                    calib["K"],
                    src_size=tuple(calib["image_size"]),
                    dst_size=DETECT_IMG_SIZE,
                )
                dist = np.asarray(calib["dist"], dtype=np.float64).reshape(-1)

                detect_frame = frame
                if UNDISTORT_BEFORE_DETECTION:
                    detect_frame = cv2.undistort(frame, k_scaled, dist)
                    detect_dist = np.zeros(5, dtype=np.float64)
                else:
                    detect_dist = dist

                del detect_dist  # pose is estimated on undistorted image through pupil_apriltags
                tag = detect_target_tag(
                    detector=tag_detector,
                    image_bgr=detect_frame,
                    camera_params=k_to_camera_params(k_scaled),
                )

                if tag is None:
                    frame_handle.visible = False
                    status_lines.append(f"[{camera_name}] tag not detected")
                    if SHOW_CV2_WINDOWS:
                        cv2.imshow(f"{WINDOW_PREFIX}: {camera_name}", detect_frame)
                    continue

                pose_R = np.asarray(tag.pose_R, dtype=np.float64).reshape(3, 3)
                pose_t = np.asarray(tag.pose_t, dtype=np.float64).reshape(3)
                if not is_valid_rotation_matrix(pose_R):
                    frame_handle.visible = False
                    status_lines.append(f"[{camera_name}] invalid pose_R")
                    continue

                world_R_cam, world_t_cam = invert_pose(pose_R, pose_t)
                frame_handle.wxyz = rotation_matrix_to_wxyz(world_R_cam)
                frame_handle.position = (
                    float(world_t_cam[0]),
                    float(world_t_cam[1]),
                    float(world_t_cam[2]),
                )
                frame_handle.visible = True
                visible_count += 1

                status_lines.append(make_status_text(camera_name, True, world_t_cam))

                if SHOW_CV2_WINDOWS:
                    corners = np.round(np.asarray(tag.corners, dtype=np.float64)).astype(np.int32)
                    vis = detect_frame.copy()
                    cv2.polylines(vis, [corners], True, (0, 255, 0), 2)
                    cv2.putText(
                        vis,
                        f"ID:{TAG_ID}",
                        (int(tag.center[0]) - 12, int(tag.center[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )
                    cv2.imshow(f"{WINDOW_PREFIX}: {camera_name}", vis)

            if frame_idx % PRINT_EVERY_N_FRAMES == 0:
                print(f"[frame {frame_idx}] visible_cameras={visible_count}")
                for line in status_lines:
                    print(line)

            if SHOW_CV2_WINDOWS:
                key = cv2.waitKey(1)
            else:
                key = cv2.waitKey(1)
            if key == 27 or key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        camera_manager.release_all()
        cv2.destroyAllWindows()
        print("[INFO] Finished.")


if __name__ == "__main__":
    main()
