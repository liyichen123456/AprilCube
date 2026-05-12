# OpenCV 相机系
# +x: image right
# +y: image down
# +z: camera forward
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

THIS_FILE = Path(__file__).resolve()
THIRDPARTY_DIR = THIS_FILE.parent.parent.parent
APRILCUBE_SRC_DIR = THIRDPARTY_DIR / "aprilcube" / "cube_april_25h9_0_5_1x1x1_10mm"

PROJECT_ROOT = THIRDPARTY_DIR.parent

RECORDER_UTILS_DIR = PROJECT_ROOT / "scripts" / "utils"

if str(RECORDER_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(RECORDER_UTILS_DIR))

import aprilcube  # noqa: E402
from recorder_oak_cam import OAK1WCameraManager, list_oak_devices  # noqa: E402

# ============================================================
# User macros
# ============================================================

PRINT_AVAILABLE_DEVICES = True

# Use the device name you used before.
# If you want l_wrist, add it here and add intrinsics below.
CAMERA_TO_DEVICE: dict[str, str] = {
    "r_wrist": "3.10.1",
    # "l_wrist": "3.10.4.4.1",
}

ACTIVE_CAMERA_NAMES: list[str] = ["r_wrist"]

# OAK-side output resolution.
# THE_12_MP is about 4056 x 3040.
# ISP_SCALE=(1, 3) gives about 1352 x 1013.
# ISP_SCALE=(1, 4) gives about 1014 x 760.
ISP_SCALE: tuple[int, int] = (1, 3)

# Lower FPS reduces USB bandwidth and camera-board EMI.
FPS = 25

QUEUE_SIZE = 4
QUEUE_BLOCKING = False

ROTATE_180_NAMES: set[str] = set()

# Host-side image size used by AprilCube detector.
# Must match the intrinsics used by the detector.
# DETECT_IMG_SIZE: tuple[int, int] = (4056, 3040)  # width, height
DETECT_IMG_SIZE: tuple[int, int] = (1280, 960)  # width, height

WINDOW_PREFIX = "OAK AprilCube"
ENABLE_VISER = True
VISER_BASE_PORT = 8080

PRINT_EVERY_N_FRAMES = 5

K_ORIGINAL_SIZE: tuple[int, int] = (1280, 960)

# Intrinsics calibrated at K_ORIGINAL_SIZE.
K_BY_CAMERA: dict[str, np.ndarray] = {
    "r_wrist": np.array(
        [
            [734.4013634654287, 0.0, 646.5993494171163],
            [0.0, 736.4711869706985, 474.326864306542],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    "l_wrist": np.array(
        [
            [734.4013634654287, 0.0, 646.5993494171163],
            [0.0, 736.4711869706985, 474.326864306542],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
}

# Optional distortion coefficients.
# For OAK-1-W wide-angle camera, real distortion coefficients are strongly recommended.
# If unknown, zeros are used. Detection may work, but pose accuracy is worse near image edges.
DIST_COEFFS_BY_CAMERA: dict[str, np.ndarray | None] = {
    "r_wrist": np.array(
        [
            -0.2515969903932863,
            0.10485302574350926,
            0.0002484901366271755,
            0.0004693307634594187,
            -0.026532680288426286,
        ],
        dtype=np.float64,
    ),
    "l_wrist": np.array(
        [
            -0.2515969903932863,
            0.10485302574350926,
            0.0002484901366271755,
            0.0004693307634594187,
            -0.026532680288426286,
        ],
        dtype=np.float64,
    ),
}

# Detector settings.
ENABLE_FILTER = True
FAST_DETECTOR = True


# ============================================================
# Utilities
# ============================================================

def scale_intrinsics(
    k: np.ndarray,
    old_size: tuple[int, int],
    new_size: tuple[int, int],
) -> np.ndarray:
    """Scale camera matrix K when resizing images.

    Args:
        k: 3x3 camera matrix for old_size.
        old_size: (width, height) of the image used during calibration.
        new_size: (width, height) of the image used for detection.
    """
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


def create_detectors(
    cube_path: Path,
    camera_names: list[str],
) -> dict[str, Any]:
    """Create one AprilCube detector per camera."""
    detectors: dict[str, Any] = {}

    for camera_name in camera_names:
        if camera_name not in K_BY_CAMERA:
            raise KeyError(
                f"Missing intrinsics for camera '{camera_name}'. "
                f"Please add K_BY_CAMERA['{camera_name}']."
            )

        k_scaled = scale_intrinsics(
            K_BY_CAMERA[camera_name],
            old_size=K_ORIGINAL_SIZE,
            new_size=DETECT_IMG_SIZE,
        )

        intrinsic_cfg = camera_matrix_to_intrinsic_dict(k_scaled)

        dist_coeffs = DIST_COEFFS_BY_CAMERA.get(camera_name)
        if dist_coeffs is not None:
            dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)

        detector = aprilcube.detector(
            cube_path,
            intrinsic_cfg=intrinsic_cfg,
            dist_coeffs=dist_coeffs,
            enable_filter=ENABLE_FILTER,
            fast=FAST_DETECTOR,
        )

        detectors[camera_name] = detector

        print(f"[INFO] Created AprilCube detector for {camera_name}")
        print(f"[INFO]   cube_path = {cube_path}")
        print(f"[INFO]   detect_size = {DETECT_IMG_SIZE}")
        print(f"[INFO]   K_scaled =\n{k_scaled}")
        print(f"[INFO]   dist_coeffs = {dist_coeffs}")

    return detectors


def build_viser_servers(
    detectors: dict[str, Any],
    camera_names: list[str],
) -> dict[str, Any]:
    """Start one aprilcube viser server per camera/detector."""
    viser_servers: dict[str, Any] = {}

    if not ENABLE_VISER:
        return viser_servers

    for idx, camera_name in enumerate(camera_names):
        detector = detectors[camera_name]
        port = int(VISER_BASE_PORT) + idx
        try:
            viser_servers[camera_name] = detector.build_viser(port=port)
            print(f"[INFO] AprilCube viser for {camera_name}: http://0.0.0.0:{port}")
        except Exception as exc:
            print(
                f"[WARNING] Failed to start AprilCube viser for {camera_name}: "
                f"{type(exc).__name__}: {exc}"
            )

    return viser_servers


def result_to_text(camera_name: str, result: dict[str, Any] | None) -> str:
    if not result:
        return f"[{camera_name}] no result"

    if not result.get("success", False):
        return f"[{camera_name}] cube not detected"

    tvec = result.get("tvec", None)
    rvec = result.get("rvec", None)
    error = result.get("reproj_error", None)
    visible_faces = result.get("visible_faces", None)

    if tvec is None:
        return f"[{camera_name}] success, but no tvec"

    t = np.asarray(tvec, dtype=np.float64).reshape(-1)

    text = (
        f"[{camera_name}] success "
        f"t_mm=({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f})"
    )

    if rvec is not None:
        r = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        rot_mat, _ = cv2.Rodrigues(r)
        euler_xyz_deg = rotation_matrix_to_euler_xyz_deg(rot_mat)
        text += (
            f" rot_xyz_deg=({euler_xyz_deg[0]:.1f}, "
            f"{euler_xyz_deg[1]:.1f}, {euler_xyz_deg[2]:.1f})"
        )

    if error is not None:
        text += f" reproj={float(error):.2f}px"

    if visible_faces is not None:
        text += f" faces={sorted(list(visible_faces))}"

    return text


def rotation_matrix_to_euler_xyz_deg(rot_mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to xyz Euler angles in degrees.

    Avoids requiring scipy.
    """
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
        y += 26

    return out


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Detect AprilCube using OAK camera manager.")
    parser.add_argument(
        "--cameras",
        type=str,
        default=",".join(ACTIVE_CAMERA_NAMES),
        help="Comma-separated logical camera names, e.g. r_wrist or r_wrist,l_wrist.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable AprilCube temporal filter.",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Use slower but more accurate AprilCube detector.",
    )
    args = parser.parse_args()

    global ENABLE_FILTER
    global FAST_DETECTOR

    if args.no_filter:
        ENABLE_FILTER = False

    if args.slow:
        FAST_DETECTOR = False

    active_camera_names = [x.strip() for x in args.cameras.split(",") if x.strip()]

    if not active_camera_names:
        print("[ERROR] No active camera names specified.")
        sys.exit(1)

    missing_camera_cfg = [name for name in active_camera_names if name not in CAMERA_TO_DEVICE]
    if missing_camera_cfg:
        print(f"[ERROR] Missing CAMERA_TO_DEVICE entries for: {missing_camera_cfg}")
        print("[ERROR] Please edit CAMERA_TO_DEVICE at the top of this script.")
        sys.exit(1)

    if PRINT_AVAILABLE_DEVICES:
        list_oak_devices()

    cube_path = APRILCUBE_SRC_DIR.resolve()
    if cube_path.is_dir():
        cube_config_path = cube_path / "config.json"
        if not cube_config_path.is_file():
            print(f"[ERROR] APRILCUBE_SRC_DIR has no config.json: {cube_path}")
            sys.exit(1)
    elif cube_path.is_file() and cube_path.name == "config.json":
        cube_config_path = cube_path
    else:
        print(f"[ERROR] APRILCUBE_SRC_DIR does not exist or is invalid: {cube_path}")
        sys.exit(1)

    print(f"[INFO] Using AprilCube model/config: {cube_path}")

    detectors = create_detectors(
        cube_path=cube_path,
        camera_names=active_camera_names,
    )
    _viser_servers = build_viser_servers(
        detectors=detectors,
        camera_names=active_camera_names,
    )

    camera_manager = OAK1WCameraManager(
        camera_to_device={
            name: CAMERA_TO_DEVICE[name]
            for name in active_camera_names
        },
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

        opened_names = camera_manager.get_active_camera_names()
        print(f"[INFO] Opened OAK cameras: {opened_names}")
        print("[INFO] Waiting for first frames...")

        camera_manager.wait_for_first_frames(
            camera_names=opened_names,
            timeout_s=5.0,
        )

        print("[INFO] Press 'q' or ESC to quit.")

        frame_idx = 0
        last_no_frame_print_time = 0.0

        while True:
            frame_idx += 1

            frames, _origin_frames = camera_manager.get_frames(
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
                detector = detectors[camera_name]

                result = detector.process_frame(frame)

                try:
                    vis = detector.draw_result(frame.copy(), result)
                except Exception as exc:
                    print(f"[WARNING] draw_result failed for {camera_name}: {type(exc).__name__}: {exc}")
                    vis = frame.copy()

                status = result_to_text(camera_name, result)

                if frame_idx % PRINT_EVERY_N_FRAMES == 0:
                    print(status)

                panel_lines = [
                    status,
                    f"detect_size={DETECT_IMG_SIZE}, isp_scale={ISP_SCALE}, fps={FPS}",
                    "press q or ESC to quit",
                ]
                vis = draw_text_panel(vis, panel_lines)

                cv2.imshow(f"{WINDOW_PREFIX}: {camera_name}", vis)

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
