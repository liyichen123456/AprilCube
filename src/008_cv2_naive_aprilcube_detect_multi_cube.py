# OpenCV camera frame:
# +x: image right
# +y: image down
# +z: camera forward
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

THIS_FILE = Path(__file__).resolve()
THIRDPARTY_DIR = THIS_FILE.parent.parent.parent
PROJECT_ROOT = THIRDPARTY_DIR.parent
RECORDER_UTILS_DIR = PROJECT_ROOT / "scripts" / "utils"

if str(RECORDER_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(RECORDER_UTILS_DIR))

import aprilcube  # noqa: E402
from aprilcube.detect import _preprocess as preprocess_tag_image  # noqa: E402
from recorder_cv2_cam import CV2CameraManager  # noqa: E402


def load_intrinsics_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path).expanduser().resolve()
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    dist = data.get("dist", data.get("D", None))
    if dist is None:
        dist = np.zeros(5, dtype=np.float64)

    return {
        "path": str(yaml_path),
        "camera_model": str(data.get("camera_model", "")),
        "distortion_model": str(data.get("distortion_model", "")),
        "image_size": tuple(int(v) for v in data["image_size"]),
        "K": np.asarray(data["K"], dtype=np.float64).reshape(3, 3),
        "dist": np.asarray(dist, dtype=np.float64).reshape(-1),
    }


# ============================================================
# User macros
# ============================================================

CAMERA_TO_PORT: dict[str, str] = {
    # "cam0": "4-9:1.0",
    "cam1": "3-5.4.3.4.4:1.0",
}

CAMERA_TO_INTRINSICS_YAML: dict[str, str] = {
    # "cam0": "/home/ps/RobotCamCalib1/outputs/intrinsics_cam0_fisheye_2592x1944_0618_181728.yaml",
    # "cam0": "/home/ps/RobotCamCalib1/outputs/intrinsics_cam0_fisheye_2592x1944_0703_210450.yaml",
    # "cam1": "/home/ps/RobotCamCalib1/outputs/intrinsics_cam0_fisheye_2592x1944_0703_230535.yaml",  # 180 degree

    # fisheye middle finger yaml
    # "cam0": "/home/ps/RobotCamCalib1/outputs/intrinsics_middle_finger_charuco_2592x1944_0705_180038.yaml",

    # pinehole middle finger yaml
    # "cam0": "/home/ps/RobotCamCalib1/outputs/intrinsics_cv2_apriltag_grid_1920x1080_0706_182725.yaml",

    # pinehole middle finger yaml 0707 0145 update
    "cam1": "/home/ps/RobotCamCalib1/outputs/intrinsics_middle_finger_1_1920x1200_0707_013313.yaml",
}

ACTIVE_CAMERA_NAMES: list[str] = ["cam1"]

FPS = 120
FOURCC = "MJPG"
WINDOW_PREFIX = "CV2 Native AprilCube"
PRINT_EVERY_N_FRAMES = 5
UNDISTORT_BEFORE_DETECTION = True
FISHEYE_RECTIFIED_HORIZONTAL_FOV_DEG = 120.0
PINHOLE_UNDISTORT_ALPHA = 0.0
RECORD_OUTPUT_DIR = THIS_FILE.parent.parent / "recordings"
ADAPTIVE_CLAHE_DETECTION = True

CUBE_CFG_DIRS: list[Path] = [
    # THIRDPARTY_DIR / "aprilcube" / "cubes" / "cube_april_36h11_0_5_1x1x1_10mm",
    # THIRDPARTY_DIR / "aprilcube" / "cubes" / "cube_april_36h11_6_11_1x1x1_15mm",  # test
    THIRDPARTY_DIR / "aprilcube" / "cubes" / "cube_april_36h11_6_11_1x1x1_10mm",
    THIRDPARTY_DIR / "aprilcube" / "cubes" / "cube_april_36h11_12_17_1x1x1_10mm",
    # THIRDPARTY_DIR / "aprilcube" / "cubes" / "cube_april_36h11_18_23_1x1x1_10mm",
    # THIRDPARTY_DIR / "aprilcube" / "cubes" / "cube_april_36h11_24_29_1x1x1_10mm",
    # THIRDPARTY_DIR / "aprilcube" / "cubes" / "cube_april_36h11_30_35_1x1x1_10mm",
]

ENABLE_FILTER = True
FAST_DETECTOR = True


def camera_matrix_to_intrinsic_dict(k: np.ndarray) -> dict[str, float]:
    return {
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
    }


def validate_cube_path(cube_path: Path) -> Path:
    cube_path = cube_path.expanduser().resolve()
    if cube_path.is_dir() and (cube_path / "config.json").is_file():
        return cube_path
    if cube_path.is_file() and cube_path.name == "config.json":
        return cube_path
    raise FileNotFoundError(f"Invalid AprilCube cfg path: {cube_path}")


def resolve_common_image_size(calib_by_camera: dict[str, dict[str, Any]]) -> tuple[int, int]:
    image_sizes = {
        camera_name: tuple(int(v) for v in calib["image_size"])
        for camera_name, calib in calib_by_camera.items()
    }
    unique_sizes = set(image_sizes.values())
    if len(unique_sizes) != 1:
        raise ValueError(
            "CV2CameraManager accepts one capture size for this script, "
            f"but active cameras use different YAML image_size values: {image_sizes}"
        )
    return next(iter(unique_sizes))


def is_fisheye_calib(calib: dict[str, Any]) -> bool:
    camera_model = str(calib.get("camera_model", "")).lower()
    distortion_model = str(calib.get("distortion_model", "")).lower()
    return camera_model == "fisheye" or distortion_model == "opencv_fisheye"


def make_centered_pinhole_camera_matrix(
    image_size: tuple[int, int],
    horizontal_fov_deg: float,
) -> np.ndarray:
    width, height = image_size
    half_fov_rad = np.radians(horizontal_fov_deg) / 2.0
    if not 0.0 < half_fov_rad < (np.pi / 2.0):
        raise ValueError(f"horizontal_fov_deg must be in (0, 180), got {horizontal_fov_deg}.")

    focal = width / (2.0 * np.tan(half_fov_rad))
    return np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def compute_detection_camera_matrix(
    calib: dict[str, Any],
    image_size: tuple[int, int],
    *,
    undistort_before_detection: bool,
) -> np.ndarray:
    camera_matrix = np.asarray(calib["K"], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(calib.get("dist", np.zeros(5)), dtype=np.float64).reshape(-1)
    if (
        not undistort_before_detection
        or dist_coeffs.size == 0
        or np.allclose(dist_coeffs, 0.0)
    ):
        return camera_matrix.copy()

    if is_fisheye_calib(calib):
        if dist_coeffs.size != 4:
            raise ValueError(f"OpenCV fisheye calibration expects 4 coeffs, got {dist_coeffs.size}.")
        return make_centered_pinhole_camera_matrix(
            image_size,
            FISHEYE_RECTIFIED_HORIZONTAL_FOV_DEG,
        )

    new_camera_matrix, _roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        PINHOLE_UNDISTORT_ALPHA,
        image_size,
    )
    return np.asarray(new_camera_matrix, dtype=np.float64).reshape(3, 3)


def create_undistort_maps(
    calib: dict[str, Any],
    image_size: tuple[int, int],
    detection_camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    camera_matrix = np.asarray(calib["K"], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(calib.get("dist", np.zeros(5)), dtype=np.float64).reshape(-1)
    if dist_coeffs.size == 0 or np.allclose(dist_coeffs, 0.0):
        return None

    detection_camera_matrix = np.asarray(detection_camera_matrix, dtype=np.float64).reshape(3, 3)
    if is_fisheye_calib(calib):
        if dist_coeffs.size != 4:
            raise ValueError(f"OpenCV fisheye calibration expects 4 coeffs, got {dist_coeffs.size}.")
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


def create_detector_for_camera(
    cube_path: Path,
    camera_name: str,
    calib_by_camera: dict[str, dict[str, Any]],
    detection_camera_matrix_by_camera: dict[str, np.ndarray],
    *,
    enable_filter: bool,
    fast: bool,
    undistort_before_detection: bool,
) -> Any:
    if camera_name not in calib_by_camera:
        raise KeyError(f"Missing intrinsics for camera '{camera_name}'.")

    calib = calib_by_camera[camera_name]
    detection_camera_matrix = detection_camera_matrix_by_camera[camera_name]
    intrinsic_cfg = camera_matrix_to_intrinsic_dict(detection_camera_matrix)
    dist_coeffs = calib.get("dist", None)
    if dist_coeffs is not None:
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)

    detector_dist_coeffs = dist_coeffs
    if undistort_before_detection:
        detector_dist_coeffs = np.zeros(5, dtype=np.float64)

    return aprilcube.detector(
        cube_path,
        intrinsic_cfg=intrinsic_cfg,
        dist_coeffs=detector_dist_coeffs,
        enable_filter=enable_filter,
        fast=fast,
    )


def undistort_frame(
    frame: np.ndarray,
    undistort_maps: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    if undistort_maps is None:
        return frame
    map1, map2 = undistort_maps
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)


def make_tag_detection_vis_image(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    enhanced = preprocess_tag_image(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


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


def result_to_text(camera_name: str, cube_name: str, result: dict[str, Any] | None) -> str:
    prefix = f"[{camera_name}][{cube_name}]"
    if not result:
        return f"{prefix} no result"
    if not result.get("success", False):
        n_tags = int(result.get("n_tags", 0))
        return f"{prefix} cube not detected tags={n_tags}"

    tvec = np.asarray(result["tvec"], dtype=np.float64).reshape(-1)
    text = f"{prefix} t=({tvec[0]:.1f},{tvec[1]:.1f},{tvec[2]:.1f})"

    if result.get("rvec", None) is not None:
        rot_mat, _ = cv2.Rodrigues(np.asarray(result["rvec"], dtype=np.float64).reshape(3, 1))
        euler = rotation_matrix_to_euler_xyz_deg(rot_mat)
        text += f" rot=({euler[0]:.1f},{euler[1]:.1f},{euler[2]:.1f})"

    error = result.get("reproj_error", None)
    if error is not None:
        text += f" reproj={float(error):.2f}px"

    text += f" tags={int(result.get('n_tags', 0))}"

    faces = result.get("visible_faces", None)
    if faces is not None:
        text += f" faces={sorted(list(faces))}"

    if result.get("predicted", False):
        text += " predicted"
    if result.get("single_tag_cfg_pose", False):
        text += (
            f" single_tag_cfg_pose"
            f"(id={result.get('single_tag_id', '?')},face={result.get('single_tag_face', '?')})"
        )

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


def count_adaptive_new_tag_ids(shared_tags: dict[str, Any]) -> int:
    attempts = shared_tags.get("adaptive_attempts", [])
    new_ids: set[int] = set()
    for attempt in attempts:
        if attempt.get("base", False):
            continue
        for tag_id in attempt.get("new_ids", []):
            new_ids.add(int(tag_id))
    return len(new_ids)


class RawFramePklRecorder:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.path: Path | None = None
        self._metadata: dict[str, Any] | None = None
        self._frames: list[dict[str, Any]] = []
        self.started_wall_time: str | None = None
        self.started_monotonic: float | None = None

    @property
    def is_recording(self) -> bool:
        return self._metadata is not None

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def buffered_bytes(self) -> int:
        return int(sum(frame["image_bgr"].nbytes for frame in self._frames))

    def start(self, metadata: dict[str, Any]) -> None:
        if self.is_recording:
            print(f"[INFO] Recording already active: {self.path}")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = self.output_dir / f"008_raw_frames_{stamp}.pkl"
        self.started_wall_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.started_monotonic = time.perf_counter()
        self._frames = []
        self._metadata = dict(metadata)
        print(f"[INFO] Started raw-frame memory buffering: {self.path}")

    def write(
        self,
        *,
        camera_name: str,
        loop_frame_idx: int,
        image_bgr: np.ndarray | None,
        capture_timestamp: float | None,
    ) -> None:
        if not self.is_recording or image_bgr is None:
            return

        self._frames.append(
            {
                "type": "frame",
                "camera_name": camera_name,
                "loop_frame_idx": int(loop_frame_idx),
                "capture_timestamp": None
                if capture_timestamp is None
                else float(capture_timestamp),
                "write_monotonic": float(time.perf_counter()),
                "shape": tuple(int(v) for v in image_bgr.shape),
                "dtype": str(image_bgr.dtype),
                "image_bgr": image_bgr,
            }
        )

    def _print_save_progress(self, done: int, total: int) -> None:
        width = 36
        ratio = 1.0 if total <= 0 else done / total
        filled = int(round(width * ratio))
        bar = "#" * filled + "-" * (width - filled)
        sys.stdout.write(f"\r[INFO] Saving PKL [{bar}] {done}/{total} frames")
        sys.stdout.flush()
        if done >= total:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def stop(self, reason: str = "user_stop") -> None:
        if not self.is_recording:
            print("[INFO] Recording is not active.")
            return

        path = self.path
        assert path is not None
        assert self._metadata is not None

        total_frames = self.frame_count
        buffered_gb = self.buffered_bytes / (1024**3)
        elapsed = (
            time.perf_counter() - self.started_monotonic
            if self.started_monotonic is not None
            else 0.0
        )
        print(
            f"[INFO] Stopped raw-frame buffering: frames={total_frames} "
            f"buffered={buffered_gb:.2f} GiB duration={elapsed:.2f}s"
        )
        print(f"[INFO] Writing PKL: {path}")

        with path.open("wb") as f:
            pickle.dump(
                {
                    "type": "header",
                    "format": "aprilcube_raw_frame_stream_v1",
                    "created_wall_time": self.started_wall_time,
                    "metadata": self._metadata,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            self._print_save_progress(0, total_frames)
            for idx, frame_record in enumerate(self._frames, start=1):
                pickle.dump(frame_record, f, protocol=pickle.HIGHEST_PROTOCOL)
                if idx == total_frames or idx % 10 == 0:
                    self._print_save_progress(idx, total_frames)
            pickle.dump(
                {
                    "type": "footer",
                    "reason": reason,
                    "frame_count": int(total_frames),
                    "stopped_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        self._frames = []
        self._metadata = None
        self.started_monotonic = None
        print(f"[INFO] Saved raw-frame PKL recording: {path} frames={total_frames}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect multiple AprilCube cfgs using one shared AprilTag detection pass per CV2 frame."
    )
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
        help="Comma-separated AprilCube cfg directories or config.json files.",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Use native AprilCube slow/high-accuracy detector parameters.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable native AprilCube temporal pose filter.",
    )
    parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="Do not undistort images before native AprilCube detection.",
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        default=str(RECORD_OUTPUT_DIR),
        help="Directory for raw-frame PKL recordings triggered by s/p.",
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

    missing_camera_cfg = [name for name in active_camera_names if name not in CAMERA_TO_PORT]
    if missing_camera_cfg:
        print(f"[ERROR] Missing CAMERA_TO_PORT entries for: {missing_camera_cfg}")
        sys.exit(1)
    missing_intrinsics_cfg = [
        name for name in active_camera_names if name not in CAMERA_TO_INTRINSICS_YAML
    ]
    if missing_intrinsics_cfg:
        print(f"[ERROR] Missing CAMERA_TO_INTRINSICS_YAML entries for: {missing_intrinsics_cfg}")
        sys.exit(1)

    calib_by_camera = {
        name: load_intrinsics_yaml(CAMERA_TO_INTRINSICS_YAML[name])
        for name in active_camera_names
    }
    image_size = resolve_common_image_size(calib_by_camera)
    capture_size = image_size
    detect_img_size = image_size
    vis_img_size = (max(1, detect_img_size[0] // 2), max(1, detect_img_size[1] // 2))
    use_undistort = UNDISTORT_BEFORE_DETECTION and not args.no_undistort
    detection_camera_matrix_by_camera = {
        camera_name: compute_detection_camera_matrix(
            calib,
            detect_img_size,
            undistort_before_detection=use_undistort,
        )
        for camera_name, calib in calib_by_camera.items()
    }
    undistort_maps_by_camera = {
        camera_name: create_undistort_maps(
            calib,
            detect_img_size,
            detection_camera_matrix_by_camera[camera_name],
        )
        if use_undistort
        else None
        for camera_name, calib in calib_by_camera.items()
    }

    for camera_name in active_camera_names:
        calib = calib_by_camera[camera_name]
        raw_k = np.asarray(calib["K"], dtype=np.float64).reshape(3, 3)
        detect_k = detection_camera_matrix_by_camera[camera_name]
        print(
            f"[INFO] [{camera_name}] intrinsics_yaml={calib['path']} "
            f"image_size={calib['image_size']} "
            f"camera_model={calib['camera_model'] or 'unknown'} "
            f"distortion_model={calib['distortion_model'] or 'unknown'} "
            f"undistort={use_undistort}"
        )
        print(
            f"[INFO] [{camera_name}] raw_K="
            f"fx={raw_k[0, 0]:.3f} fy={raw_k[1, 1]:.3f} "
            f"cx={raw_k[0, 2]:.3f} cy={raw_k[1, 2]:.3f}"
        )
        print(
            f"[INFO] [{camera_name}] detection_K="
            f"fx={detect_k[0, 0]:.3f} fy={detect_k[1, 1]:.3f} "
            f"cx={detect_k[0, 2]:.3f} cy={detect_k[1, 2]:.3f}"
        )
    print(
        f"[INFO] capture_size={capture_size} "
        f"detect_img_size={detect_img_size} vis_img_size={vis_img_size}"
    )

    detector_entries_by_camera: dict[str, list[dict[str, Any]]] = {
        name: [] for name in active_camera_names
    }

    for cube_path in cube_paths:
        cube_name = cube_path.name if cube_path.is_dir() else cube_path.parent.name
        for camera_name in active_camera_names:
            detector = create_detector_for_camera(
                cube_path,
                camera_name,
                calib_by_camera,
                detection_camera_matrix_by_camera,
                enable_filter=not args.no_filter,
                fast=not args.slow,
                undistort_before_detection=use_undistort,
            )
            detector_entries_by_camera[camera_name].append(
                {
                    "cube_name": cube_name,
                    "detector": detector,
                }
            )
            print(f"[INFO] Loaded native AprilCube detector for {camera_name}: {cube_name}")

    camera_manager = CV2CameraManager(
        camera_to_port={name: CAMERA_TO_PORT[name] for name in active_camera_names},
        capture_size=capture_size,
        fps=FPS,
        fourcc=FOURCC,
    )
    recorder = RawFramePklRecorder(Path(args.record_dir))

    try:
        opened = camera_manager.open_all_cameras()
        if opened == 0:
            print("[ERROR] No CV2 camera opened.")
            sys.exit(1)

        opened_names = camera_manager.get_active_camera_names()
        print(f"[INFO] Opened CV2 cameras: {opened_names}")
        print(
            "[INFO] Native detection path: shared detect_tags(frame) "
            "+ per-cube process_detections()."
        )
        print(f"[INFO] Adaptive CLAHE tag recovery: {ADAPTIVE_CLAHE_DETECTION}")
        print("[INFO] Press 's' to start raw-frame PKL recording, 'p' to stop, 'q' or ESC to quit.")

        recording_metadata = {
            "script": str(THIS_FILE),
            "recorded_image": "origin_frame_raw_bgr",
            "camera_to_port": {name: CAMERA_TO_PORT[name] for name in active_camera_names},
            "intrinsics_yaml": {
                name: CAMERA_TO_INTRINSICS_YAML[name] for name in active_camera_names
            },
            "opened_cameras": list(opened_names),
            "capture_size": tuple(int(v) for v in capture_size),
            "detect_img_size": tuple(int(v) for v in detect_img_size),
            "fps": int(FPS),
            "fourcc": str(FOURCC),
            "undistort_before_detection": bool(use_undistort),
            "fisheye_rectified_horizontal_fov_deg": float(
                FISHEYE_RECTIFIED_HORIZONTAL_FOV_DEG
            ),
            "cube_paths": [str(path) for path in cube_paths],
        }

        def handle_key(key: int) -> bool:
            if key == 27 or key == ord("q"):
                return False
            if key == ord("s"):
                recorder.start(recording_metadata)
            elif key == ord("p"):
                recorder.stop("user_stop")
            return True

        frame_idx = 0
        last_no_frame_print_time = 0.0
        while True:
            frame_idx += 1
            frames, _origin_frames, _timestamps = camera_manager.get_frames(
                camera_names=opened_names,
                img_size=detect_img_size,
            )
            if not frames:
                now = time.time()
                if now - last_no_frame_print_time > 1.0:
                    print("[INFO] No frames received yet.")
                    last_no_frame_print_time = now
                key = cv2.waitKey(1)
                if not handle_key(key):
                    break
                continue

            for camera_name, frame in frames.items():
                origin_frame = _origin_frames.get(camera_name)
                recorder.write(
                    camera_name=camera_name,
                    loop_frame_idx=frame_idx,
                    image_bgr=origin_frame,
                    capture_timestamp=_timestamps.get(camera_name),
                )
                if origin_frame is not None and frame_idx % PRINT_EVERY_N_FRAMES == 0:
                    origin_h, origin_w = origin_frame.shape[:2]
                    detect_h, detect_w = frame.shape[:2]
                    print(
                        f"[{camera_name}] origin_size=({origin_w}, {origin_h}) "
                        f"detect_frame_size=({detect_w}, {detect_h})"
                    )

                detector_entries = detector_entries_by_camera[camera_name]
                detect_frame = frame
                if use_undistort:
                    detect_frame = undistort_frame(
                        frame,
                        undistort_maps_by_camera[camera_name],
                    )

                vis = make_tag_detection_vis_image(detect_frame)
                fps_text = camera_manager.get_latest_fps(camera_name)
                shared_timestamp = time.monotonic()
                shared_tags = detector_entries[0]["detector"].detect_tags(
                    detect_frame,
                    adaptive_clahe=ADAPTIVE_CLAHE_DETECTION,
                )
                adaptive_new_tags = count_adaptive_new_tag_ids(shared_tags)
                status_lines = [
                    f"[{camera_name}] native_aprilcube cubes={len(detector_entries)} "
                    f"detect_size={detect_img_size} vis_size={vis_img_size} capture_size={capture_size} "
                    f"fps={fps_text:.1f}" if fps_text is not None else
                    f"[{camera_name}] native_aprilcube cubes={len(detector_entries)} "
                    f"detect_size={detect_img_size} vis_size={vis_img_size} capture_size={capture_size}"
                ]
                status_lines.append(
                    f"tags_decoded={len(shared_tags['detections'])} "
                    f"adaptive_clahe={ADAPTIVE_CLAHE_DETECTION} "
                    f"clahe_extra_tags={adaptive_new_tags}"
                )
                if recorder.is_recording:
                    buffered_gb = recorder.buffered_bytes / (1024**3)
                    status_lines.append(
                        f"REC buffering frames={recorder.frame_count} mem={buffered_gb:.2f}GiB"
                    )
                else:
                    status_lines.append("REC off: press s to start, p to stop")

                for entry in detector_entries:
                    cube_name = entry["cube_name"]
                    detector = entry["detector"]
                    result = detector.process_detections(
                        detect_frame,
                        shared_tags["detections"],
                        rejected_quads=shared_tags["rejected"],
                        gray=shared_tags["gray"],
                        enhanced=shared_tags["enhanced"],
                        timestamp=shared_timestamp,
                    )

                    try:
                        vis = detector.draw_result(vis, result)
                    except Exception as exc:
                        print(
                            f"[WARNING] draw_result failed for {camera_name}/{cube_name}: "
                            f"{type(exc).__name__}: {exc}"
                        )

                    line = result_to_text(camera_name, cube_name, result)
                    status_lines.append(line)

                    if frame_idx % PRINT_EVERY_N_FRAMES == 0:
                        print(line)

                status_lines.append("press s start rec, p stop rec, q or ESC quit")
                vis = draw_text_panel(vis, status_lines)
                vis = cv2.resize(vis, vis_img_size, interpolation=cv2.INTER_AREA)
                cv2.imshow(f"{WINDOW_PREFIX}: {camera_name}", vis)

            key = cv2.waitKey(1)
            if not handle_key(key):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        if recorder.is_recording:
            recorder.stop("shutdown")
        camera_manager.release_all()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
