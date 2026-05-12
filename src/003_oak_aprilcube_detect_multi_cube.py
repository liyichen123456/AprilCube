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
from pupil_apriltags import Detector

THIS_FILE = Path(__file__).resolve()
THIRDPARTY_DIR = THIS_FILE.parent.parent.parent
PROJECT_ROOT = THIRDPARTY_DIR.parent
RECORDER_UTILS_DIR = PROJECT_ROOT / "scripts" / "utils"

if str(RECORDER_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(RECORDER_UTILS_DIR))

import aprilcube  # noqa: E402
from april_tag_detector import TemporalTagPoseEstimator  # noqa: E402
from aprilcube_runtime import AprilCubeTemporalPoseRuntime  # noqa: E402
from recorder_oak_cam import OAK1WCameraManager, list_oak_devices  # noqa: E402


PRINT_AVAILABLE_DEVICES = True
CAMERA_TO_DEVICE: dict[str, str] = {
    "r_wrist": "3.10.4.3",
}
ACTIVE_CAMERA_NAMES: list[str] = ["r_wrist"]
ISP_SCALE: tuple[int, int] = (1, 3)
FPS = 25
QUEUE_SIZE = 4
QUEUE_BLOCKING = False
ROTATE_180_NAMES: set[str] = set()
DETECT_IMG_SIZE: tuple[int, int] = (1280, 960)
WINDOW_PREFIX = "OAK Multi-AprilCube"
PRINT_EVERY_N_FRAMES = 5
DRAW_TAG_FRAME_2D = True
TAG_AXIS_LENGTH_SCALE = 0.8
UNDISTORT_BEFORE_DETECTION = True
K_ORIGINAL_SIZE: tuple[int, int] = (1280, 960)

CUBE_CFG_DIRS: list[Path] = [
    THIRDPARTY_DIR / "aprilcube" / "cube_april_36h11_0_5_1x1x1_10mm",
    THIRDPARTY_DIR / "aprilcube" / "cube_april_36h11_6_11_1x1x1_10mm",
    THIRDPARTY_DIR / "aprilcube" / "cube_april_36h11_12_17_1x1x1_10mm",
    THIRDPARTY_DIR / "aprilcube" / "cube_april_36h11_18_23_1x1x1_10mm",
    THIRDPARTY_DIR / "aprilcube" / "cube_april_36h11_24_29_1x1x1_10mm",
    THIRDPARTY_DIR / "aprilcube" / "cube_april_36h11_30_35_1x1x1_10mm",
]

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
        old_size=K_ORIGINAL_SIZE,
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
        return f"{prefix} cube not detected"

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
    parser = argparse.ArgumentParser(description="Detect multiple AprilCube cfgs in one OAK camera stream.")
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

    missing_camera_cfg = [name for name in active_camera_names if name not in CAMERA_TO_DEVICE]
    if missing_camera_cfg:
        print(f"[ERROR] Missing CAMERA_TO_DEVICE entries for: {missing_camera_cfg}")
        sys.exit(1)

    if PRINT_AVAILABLE_DEVICES:
        list_oak_devices()

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
                    quad_decimate=1.0,
                )

            runtime = AprilCubeTemporalPoseRuntime(
                detector=detector,
                native_detector=shared_native_detectors[native_family],
                pose_estimator=create_pose_estimator(detector),
            )
            runtimes_by_camera[camera_name].append(
                {
                    "cube_name": cube_name,
                    "runtime": runtime,
                    "detector": detector,
                }
            )
            print(f"[INFO] Loaded cube cfg for {camera_name}: {cube_name}")

    camera_manager = OAK1WCameraManager(
        camera_to_device={name: CAMERA_TO_DEVICE[name] for name in active_camera_names},
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
        camera_manager.wait_for_first_frames(camera_names=opened_names, timeout_s=5.0)
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
                status_lines = [f"[{camera_name}] cubes={len(runtime_entries)} detect_size={DETECT_IMG_SIZE} fps={FPS}"]

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
        camera_manager.close_all_cameras()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
