#!/usr/bin/env python3
from __future__ import annotations

import pickle
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pyudev
import yaml

import aprilcube


THIS_FILE = Path(__file__).resolve()
APRILCUBE_ROOT = THIS_FILE.parent.parent
RECORDINGS_DIR = APRILCUBE_ROOT / "recordings"

CAMERA_PORTS = {
    "thumb_web_cam": "3-9:1.0",
    "middle_finger_cam": "3-8:1.0",
}
CAMERA_INTRINSICS = {
    "thumb_web_cam": Path(
        "/home/ps/RobotCamCalib1/outputs/"
        "intrinsics_thumb_web_cam_fisheye_charuco_2592x1944_0708_020331.yaml"
    ),
    "middle_finger_cam": Path(
        "/home/ps/RobotCamCalib1/outputs/"
        "intrinsics_charuco_offline_eval_0708_150154_0708_150928/"
        "intrinsics_None_charuco_2592x1944_0708_150154_offline_filtered.yaml"
    ),
}
CAMERA_CUBE_CONFIGS = {
    "thumb_web_cam": [
        APRILCUBE_ROOT / "cubes/cube_april_36h11_6_11_1x1x1_15mm",
        APRILCUBE_ROOT / "cubes/cube_april_36h11_12_17_1x1x1_15mm",
    ],
    "middle_finger_cam": [
        APRILCUBE_ROOT / "cubes/cube_april_36h11_0_5_1x1x1_15mm",
    ],
}

CAPTURE_SIZE = (2592, 1944)
REQUESTED_FPS = 120
FOURCC = "MJPG"
MEASURED_FPS = 25.0
MAX_PAIR_SKEW_MS = 1000.0 / (2.0 * MEASURED_FPS)
MAX_PENDING_FRAMES_PER_CAMERA = 8

PREVIEW_WINDOW = "021 Hand Back Software-Synchronized Capture"
PREVIEW_CAMERA_WIDTH = 720
PREVIEW_DETECTION_INTERVAL_S = 0.5
UNDISTORT_PREVIEW = True
PINHOLE_UNDISTORT_ALPHA = 0.0

PKL_FORMAT = "aprilcube_hand_back_software_synced_raw_v1"
SAVE_PROGRESS_INTERVAL = 5
STATUS_PRINT_INTERVAL_S = 1.0


@dataclass(frozen=True)
class CapturedFrame:
    camera_name: str
    sequence: int
    read_started_monotonic: float
    capture_timestamp: float
    image_bgr: np.ndarray


@dataclass(frozen=True)
class SynchronizedFramePair:
    pair_sequence: int
    pair_timestamp: float
    signed_skew_ms: float
    thumb_web: CapturedFrame
    middle_finger: CapturedFrame


def video_node_number(device_node: str) -> int:
    try:
        return int(device_node.replace("/dev/video", ""))
    except ValueError:
        return sys.maxsize


def find_video_nodes_on_usb_port(usb_port: str) -> list[str]:
    context = pyudev.Context()
    matches: list[str] = []
    for device in context.list_devices(subsystem="video4linux"):
        if device.device_node is None:
            continue
        if any(usb_port in ancestor.device_path for ancestor in device.ancestors):
            matches.append(str(device.device_node))
    return sorted(set(matches), key=video_node_number)


def configure_capture(cap: cv2.VideoCapture) -> None:
    width, height = CAPTURE_SIZE
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(REQUESTED_FPS))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def open_capture_device(camera_name: str, usb_port: str) -> tuple[cv2.VideoCapture, str]:
    candidates = find_video_nodes_on_usb_port(usb_port)
    if not candidates:
        raise RuntimeError(f"[{camera_name}] no video node found on USB port {usb_port}")

    errors: list[str] = []
    for device_node in candidates:
        cap = cv2.VideoCapture(device_node, cv2.CAP_V4L2)
        if not cap.isOpened():
            errors.append(f"{device_node}: open failed")
            cap.release()
            continue
        configure_capture(cap)
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            errors.append(f"{device_node}: no image frame")
            cap.release()
            continue

        height, width = frame.shape[:2]
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc_text = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))
        print(
            f"[INFO] [{camera_name}] opened {device_node} via {usb_port}: "
            f"size={width}x{height} reported_fps={actual_fps:.1f} "
            f"fourcc={actual_fourcc_text!r}"
        )
        return cap, device_node

    raise RuntimeError(f"[{camera_name}] failed to open capture stream: {errors}")


class CameraCaptureWorker:
    def __init__(
        self,
        camera_name: str,
        usb_port: str,
        start_barrier: threading.Barrier,
        on_frame: Callable[[CapturedFrame], None],
    ) -> None:
        self.camera_name = camera_name
        self.usb_port = usb_port
        self.start_barrier = start_barrier
        self.on_frame = on_frame
        self.cap: cv2.VideoCapture | None = None
        self.device_node: str | None = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.stats_lock = threading.Lock()
        self.latest_fps = 0.0
        self.total_frames = 0

    def open(self) -> None:
        self.cap, self.device_node = open_capture_device(self.camera_name, self.usb_port)

    def start(self) -> None:
        if self.cap is None:
            raise RuntimeError(f"[{self.camera_name}] camera must be opened before start")
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._capture_loop,
            name=f"capture-{self.camera_name}",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
        self.thread = None
        self.cap = None

    def fps(self) -> float:
        with self.stats_lock:
            return float(self.latest_fps)

    def frame_count(self) -> int:
        with self.stats_lock:
            return int(self.total_frames)

    def _capture_loop(self) -> None:
        assert self.cap is not None
        self.start_barrier.wait()
        sequence = 0
        window_start = time.perf_counter()
        window_frames = 0

        while not self.stop_event.is_set():
            read_started = time.perf_counter()
            ok, image_bgr = self.cap.read()
            received = time.perf_counter()
            if not ok or image_bgr is None:
                time.sleep(0.001)
                continue

            frame = CapturedFrame(
                camera_name=self.camera_name,
                sequence=sequence,
                read_started_monotonic=read_started,
                capture_timestamp=received,
                image_bgr=image_bgr,
            )
            sequence += 1
            window_frames += 1
            self.on_frame(frame)

            elapsed = received - window_start
            with self.stats_lock:
                self.total_frames = sequence
                if elapsed >= 1.0:
                    self.latest_fps = window_frames / elapsed
                    window_start = received
                    window_frames = 0


class ApproximateFrameSynchronizer:
    def __init__(
        self,
        max_skew_ms: float,
        on_pair: Callable[[SynchronizedFramePair], None],
    ) -> None:
        self.max_skew_s = float(max_skew_ms) / 1000.0
        self.on_pair = on_pair
        self.lock = threading.Lock()
        self.pending = {camera_name: deque() for camera_name in CAMERA_PORTS}
        self.pair_count = 0
        self.dropped = {camera_name: 0 for camera_name in CAMERA_PORTS}
        self.latest_signed_skew_ms: float | None = None
        self.skew_abs_sum_ms = 0.0
        self.skew_abs_max_ms = 0.0

    def submit(self, frame: CapturedFrame) -> None:
        ready_pairs: list[SynchronizedFramePair] = []
        with self.lock:
            queue = self.pending[frame.camera_name]
            queue.append(frame)
            while len(queue) > MAX_PENDING_FRAMES_PER_CAMERA:
                queue.popleft()
                self.dropped[frame.camera_name] += 1
            ready_pairs.extend(self._match_available_frames())

        for pair in ready_pairs:
            self.on_pair(pair)

    def _match_available_frames(self) -> list[SynchronizedFramePair]:
        thumb_queue = self.pending["thumb_web_cam"]
        middle_queue = self.pending["middle_finger_cam"]
        pairs: list[SynchronizedFramePair] = []

        while thumb_queue and middle_queue:
            thumb = thumb_queue[0]
            middle = middle_queue[0]
            signed_skew_s = thumb.capture_timestamp - middle.capture_timestamp
            if abs(signed_skew_s) <= self.max_skew_s:
                thumb_queue.popleft()
                middle_queue.popleft()
                signed_skew_ms = signed_skew_s * 1000.0
                pair = SynchronizedFramePair(
                    pair_sequence=self.pair_count,
                    pair_timestamp=(thumb.capture_timestamp + middle.capture_timestamp) / 2.0,
                    signed_skew_ms=signed_skew_ms,
                    thumb_web=thumb,
                    middle_finger=middle,
                )
                self.pair_count += 1
                self.latest_signed_skew_ms = signed_skew_ms
                abs_skew_ms = abs(signed_skew_ms)
                self.skew_abs_sum_ms += abs_skew_ms
                self.skew_abs_max_ms = max(self.skew_abs_max_ms, abs_skew_ms)
                pairs.append(pair)
                continue

            if thumb.capture_timestamp < middle.capture_timestamp:
                thumb_queue.popleft()
                self.dropped["thumb_web_cam"] += 1
            else:
                middle_queue.popleft()
                self.dropped["middle_finger_cam"] += 1

        return pairs

    def stats(self) -> dict[str, Any]:
        with self.lock:
            mean_skew_ms = (
                self.skew_abs_sum_ms / self.pair_count if self.pair_count > 0 else None
            )
            return {
                "pair_count": int(self.pair_count),
                "latest_signed_skew_ms": self.latest_signed_skew_ms,
                "mean_absolute_skew_ms": mean_skew_ms,
                "max_absolute_skew_ms": float(self.skew_abs_max_ms),
                "dropped_unmatched": dict(self.dropped),
                "pending": {name: len(queue) for name, queue in self.pending.items()},
            }


class SynchronizedPairRecorder:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.lock = threading.Lock()
        self.recording = False
        self.records: list[dict[str, Any]] = []
        self.buffered_bytes = 0
        self.started_wall_time: str | None = None
        self.started_monotonic: float | None = None
        self.metadata: dict[str, Any] | None = None
        self.output_path: Path | None = None
        self.save_thread: threading.Thread | None = None

    def start(self, metadata: dict[str, Any]) -> None:
        with self.lock:
            if self.recording:
                print("[INFO] Recording is already active.")
                return
            if self.save_thread is not None and self.save_thread.is_alive():
                print("[INFO] Previous recording is still being saved; wait before starting again.")
                return

            self.output_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.output_path = self.output_dir / f"021_hand_back_sync_raw_frames_{stamp}.pkl"
            self.records = []
            self.buffered_bytes = 0
            self.started_wall_time = time.strftime("%Y-%m-%d %H:%M:%S")
            self.started_monotonic = time.perf_counter()
            self.metadata = dict(metadata)
            self.recording = True
            print(f"[INFO] Started synchronized frame-pair buffering: {self.output_path}")

    def append_pair(self, pair: SynchronizedFramePair) -> None:
        with self.lock:
            if not self.recording:
                return
            record = {
                "type": "frame_pair",
                "pair_index": len(self.records),
                "pair_sequence": int(pair.pair_sequence),
                "pair_timestamp": float(pair.pair_timestamp),
                "signed_skew_ms": float(pair.signed_skew_ms),
                "absolute_skew_ms": abs(float(pair.signed_skew_ms)),
                "cameras": {
                    "thumb_web_cam": self._camera_record(pair.thumb_web),
                    "middle_finger_cam": self._camera_record(pair.middle_finger),
                },
            }
            self.records.append(record)
            self.buffered_bytes += pair.thumb_web.image_bgr.nbytes
            self.buffered_bytes += pair.middle_finger.image_bgr.nbytes

    @staticmethod
    def _camera_record(frame: CapturedFrame) -> dict[str, Any]:
        return {
            "camera_name": frame.camera_name,
            "sequence": int(frame.sequence),
            "read_started_monotonic": float(frame.read_started_monotonic),
            "capture_timestamp": float(frame.capture_timestamp),
            "shape": tuple(int(value) for value in frame.image_bgr.shape),
            "dtype": str(frame.image_bgr.dtype),
            "image_bgr": frame.image_bgr,
        }

    def status(self) -> dict[str, Any]:
        with self.lock:
            saving = self.save_thread is not None and self.save_thread.is_alive()
            return {
                "recording": bool(self.recording),
                "saving": bool(saving),
                "pair_count": len(self.records),
                "buffered_bytes": int(self.buffered_bytes),
                "output_path": None if self.output_path is None else str(self.output_path),
            }

    def stop_and_save(self, reason: str) -> None:
        with self.lock:
            if not self.recording:
                print("[INFO] Recording is not active.")
                return
            self.recording = False
            records = self.records
            metadata = self.metadata
            output_path = self.output_path
            started_wall_time = self.started_wall_time
            started_monotonic = self.started_monotonic
            buffered_bytes = self.buffered_bytes
            self.records = []
            self.metadata = None
            self.buffered_bytes = 0

        assert metadata is not None
        assert output_path is not None
        duration_s = time.perf_counter() - float(started_monotonic or time.perf_counter())
        print(
            f"[INFO] Stopped buffering: pairs={len(records)} "
            f"buffered={buffered_bytes / (1024**3):.2f} GiB duration={duration_s:.2f}s"
        )
        self.save_thread = threading.Thread(
            target=self._save_records,
            args=(
                output_path,
                metadata,
                records,
                started_wall_time,
                duration_s,
                reason,
            ),
            name="save-021-pkl",
            daemon=False,
        )
        self.save_thread.start()

    @staticmethod
    def _save_records(
        output_path: Path,
        metadata: dict[str, Any],
        records: list[dict[str, Any]],
        started_wall_time: str | None,
        duration_s: float,
        reason: str,
    ) -> None:
        total = len(records)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        try:
            with temporary_path.open("wb") as file:
                pickle.dump(
                    {
                        "type": "header",
                        "format": PKL_FORMAT,
                        "created_wall_time": started_wall_time,
                        "metadata": metadata,
                    },
                    file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                for index, record in enumerate(records, start=1):
                    pickle.dump(record, file, protocol=pickle.HIGHEST_PROTOCOL)
                    if index == total or index % SAVE_PROGRESS_INTERVAL == 0:
                        SynchronizedPairRecorder._print_save_progress(index, total)
                pickle.dump(
                    {
                        "type": "footer",
                        "reason": reason,
                        "frame_pair_count": total,
                        "recording_duration_s": float(duration_s),
                        "stopped_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            temporary_path.replace(output_path)
            print(f"[INFO] Saved synchronized raw-frame PKL: {output_path} pairs={total}")
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _print_save_progress(done: int, total: int) -> None:
        width = 36
        ratio = 1.0 if total == 0 else done / total
        filled = int(round(width * ratio))
        bar = "#" * filled + "-" * (width - filled)
        sys.stdout.write(f"\r[INFO] Saving PKL [{bar}] {done}/{total} pairs")
        if done >= total:
            sys.stdout.write("\n")
        sys.stdout.flush()

    def wait_for_save(self) -> None:
        thread = self.save_thread
        if thread is not None and thread.is_alive():
            print("[INFO] Waiting for PKL save to finish...")
            thread.join()


def load_calibration(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    dist = data.get("dist", data.get("D", np.zeros(5)))
    return {
        "path": str(path.expanduser().resolve()),
        "camera_model": str(data.get("camera_model", "")),
        "distortion_model": str(data.get("distortion_model", "")),
        "image_size": tuple(int(value) for value in data["image_size"]),
        "camera_matrix": np.asarray(data["K"], dtype=np.float64).reshape(3, 3),
        "dist_coeffs": np.asarray(dist, dtype=np.float64).reshape(-1),
    }


def is_fisheye(calibration: dict[str, Any]) -> bool:
    return (
        calibration["camera_model"].lower() == "fisheye"
        or calibration["distortion_model"].lower() == "opencv_fisheye"
    )


def make_rectified_camera_matrix(calibration: dict[str, Any]) -> np.ndarray:
    camera_matrix = calibration["camera_matrix"]
    dist_coeffs = calibration["dist_coeffs"]
    image_size = calibration["image_size"]
    if np.allclose(dist_coeffs, 0.0):
        return camera_matrix.copy()
    if is_fisheye(calibration):
        width, height = image_size
        focal = float(camera_matrix[0, 0])
        return np.array(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    rectified, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        PINHOLE_UNDISTORT_ALPHA,
        image_size,
    )
    return np.asarray(rectified, dtype=np.float64).reshape(3, 3)


def make_undistort_maps(
    calibration: dict[str, Any],
    rectified_camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    dist_coeffs = calibration["dist_coeffs"]
    if np.allclose(dist_coeffs, 0.0):
        return None
    camera_matrix = calibration["camera_matrix"]
    image_size = calibration["image_size"]
    if is_fisheye(calibration):
        if dist_coeffs.size != 4:
            raise ValueError(
                f"fisheye calibration requires 4 coefficients, got {dist_coeffs.size}"
            )
        return cv2.fisheye.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs.reshape(4, 1),
            np.eye(3),
            rectified_camera_matrix,
            image_size,
            cv2.CV_16SC2,
        )
    return cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        np.eye(3),
        rectified_camera_matrix,
        image_size,
        cv2.CV_16SC2,
    )


def camera_matrix_as_dict(camera_matrix: np.ndarray) -> dict[str, float]:
    return {
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
    }


class CameraCubePreview:
    def __init__(self, camera_name: str) -> None:
        self.camera_name = camera_name
        self.calibration = load_calibration(CAMERA_INTRINSICS[camera_name])
        if self.calibration["image_size"] != CAPTURE_SIZE:
            raise ValueError(
                f"[{camera_name}] calibration size {self.calibration['image_size']} "
                f"does not match capture size {CAPTURE_SIZE}"
            )
        self.rectified_camera_matrix = make_rectified_camera_matrix(self.calibration)
        self.undistort_maps = make_undistort_maps(
            self.calibration,
            self.rectified_camera_matrix,
        )
        detector_distortion = (
            np.zeros(5, dtype=np.float64)
            if UNDISTORT_PREVIEW
            else self.calibration["dist_coeffs"]
        )
        detector_camera_matrix = (
            self.rectified_camera_matrix
            if UNDISTORT_PREVIEW
            else self.calibration["camera_matrix"]
        )
        self.detectors: list[tuple[str, Any]] = []
        for cube_path in CAMERA_CUBE_CONFIGS[camera_name]:
            detector = aprilcube.detector(
                cube_path,
                intrinsic_cfg=camera_matrix_as_dict(detector_camera_matrix),
                dist_coeffs=detector_distortion,
                enable_filter=True,
                fast=True,
            )
            self.detectors.append((cube_path.name, detector))

    def draw(self, frame: CapturedFrame, capture_fps: float, pair_skew_ms: float) -> np.ndarray:
        image = frame.image_bgr
        if UNDISTORT_PREVIEW and self.undistort_maps is not None:
            image = cv2.remap(
                image,
                self.undistort_maps[0],
                self.undistort_maps[1],
                interpolation=cv2.INTER_LINEAR,
            )
        visualization = image.copy()
        status = [
            f"{self.camera_name} frame={frame.sequence} capture_fps={capture_fps:.1f}",
            f"software_pair_skew={pair_skew_ms:+.2f} ms",
        ]

        if self.detectors:
            shared = self.detectors[0][1].detect_tags(image, adaptive_clahe=True)
            for cube_name, detector in self.detectors:
                result = detector.process_detections(
                    image,
                    shared["detections"],
                    rejected_quads=shared["rejected"],
                    gray=shared["gray"],
                    enhanced=shared["enhanced"],
                    timestamp=frame.capture_timestamp,
                )
                visualization = detector.draw_result(visualization, result)
                if result.get("success", False):
                    tvec = np.asarray(result["tvec"]).reshape(3)
                    status.append(
                        f"{cube_name}: t=({tvec[0]:.1f},{tvec[1]:.1f},{tvec[2]:.1f})mm "
                        f"reproj={float(result.get('reproj_error', float('nan'))):.2f}px"
                    )
                else:
                    status.append(
                        f"{cube_name}: no pose tags={int(result.get('n_tags', 0))}"
                    )

        for line_index, line in enumerate(status):
            y = 30 + line_index * 27
            cv2.putText(
                visualization,
                line,
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return resize_to_width(visualization, PREVIEW_CAMERA_WIDTH)


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    height = max(1, int(round(image.shape[0] * width / image.shape[1])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


class NonBlockingPreviewWorker:
    def __init__(self, capture_workers: dict[str, CameraCaptureWorker]) -> None:
        self.capture_workers = capture_workers
        self.previews = {
            camera_name: CameraCubePreview(camera_name) for camera_name in CAMERA_PORTS
        }
        self.lock = threading.Lock()
        self.pending_pair: SynchronizedFramePair | None = None
        self.latest_canvas: np.ndarray | None = None
        self.stop_event = threading.Event()
        self.new_pair_event = threading.Event()
        self.thread = threading.Thread(
            target=self._worker_loop,
            name="021-preview",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def submit(self, pair: SynchronizedFramePair) -> None:
        with self.lock:
            self.pending_pair = pair
        self.new_pair_event.set()

    def latest_image(self) -> np.ndarray | None:
        with self.lock:
            return self.latest_canvas

    def stop(self) -> None:
        self.stop_event.set()
        self.new_pair_event.set()
        self.thread.join(timeout=3.0)

    def _worker_loop(self) -> None:
        last_detection_time = 0.0
        while not self.stop_event.is_set():
            self.new_pair_event.wait()
            if self.stop_event.is_set():
                return
            self.new_pair_event.clear()
            remaining_interval = PREVIEW_DETECTION_INTERVAL_S - (
                time.perf_counter() - last_detection_time
            )
            if remaining_interval > 0.0:
                time.sleep(remaining_interval)
            if self.stop_event.is_set():
                return

            with self.lock:
                pair = self.pending_pair
                self.pending_pair = None
            if pair is None:
                continue

            try:
                thumb_image = self.previews["thumb_web_cam"].draw(
                    pair.thumb_web,
                    self.capture_workers["thumb_web_cam"].fps(),
                    pair.signed_skew_ms,
                )
                middle_image = self.previews["middle_finger_cam"].draw(
                    pair.middle_finger,
                    self.capture_workers["middle_finger_cam"].fps(),
                    pair.signed_skew_ms,
                )
                target_height = min(thumb_image.shape[0], middle_image.shape[0])
                thumb_image = thumb_image[:target_height]
                middle_image = middle_image[:target_height]
                canvas = np.hstack([thumb_image, middle_image])
                with self.lock:
                    self.latest_canvas = canvas
            except Exception as exc:
                print(f"[WARNING] preview detection failed: {type(exc).__name__}: {exc}")
            last_detection_time = time.perf_counter()


def validate_configuration() -> None:
    if set(CAMERA_PORTS) != {"thumb_web_cam", "middle_finger_cam"}:
        raise ValueError("021 requires exactly thumb_web_cam and middle_finger_cam")
    for camera_name, path in CAMERA_INTRINSICS.items():
        if not path.is_file():
            raise FileNotFoundError(f"[{camera_name}] intrinsics YAML not found: {path}")
    for camera_name, cube_paths in CAMERA_CUBE_CONFIGS.items():
        for cube_path in cube_paths:
            if not (cube_path / "config.json").is_file():
                raise FileNotFoundError(f"[{camera_name}] cube config not found: {cube_path}")


def recording_metadata(
    capture_workers: dict[str, CameraCaptureWorker],
) -> dict[str, Any]:
    return {
        "script": str(THIS_FILE),
        "recorded_data": "software_synchronized_raw_bgr_frame_pairs",
        "camera_ports": dict(CAMERA_PORTS),
        "camera_device_nodes": {
            name: worker.device_node for name, worker in capture_workers.items()
        },
        "camera_intrinsics_yaml": {
            name: str(path.resolve()) for name, path in CAMERA_INTRINSICS.items()
        },
        "camera_cube_configs": {
            name: [str(path.resolve()) for path in paths]
            for name, paths in CAMERA_CUBE_CONFIGS.items()
        },
        "capture_size": tuple(CAPTURE_SIZE),
        "requested_fps": int(REQUESTED_FPS),
        "expected_measured_fps": float(MEASURED_FPS),
        "fourcc": FOURCC,
        "timestamp_clock": "time.perf_counter_after_cv2_cap_read",
        "pair_timestamp": "mean_of_camera_capture_timestamps",
        "max_pair_skew_ms": float(MAX_PAIR_SKEW_MS),
        "preview_detection_saved": False,
    }


def format_optional_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}ms"


def main() -> None:
    validate_configuration()
    recorder = SynchronizedPairRecorder(RECORDINGS_DIR)
    start_barrier = threading.Barrier(len(CAMERA_PORTS))
    capture_workers: dict[str, CameraCaptureWorker] = {}
    preview_holder: dict[str, NonBlockingPreviewWorker] = {}

    def on_pair(pair: SynchronizedFramePair) -> None:
        recorder.append_pair(pair)
        preview = preview_holder.get("preview")
        if preview is not None:
            preview.submit(pair)

    synchronizer = ApproximateFrameSynchronizer(MAX_PAIR_SKEW_MS, on_pair)
    for camera_name, usb_port in CAMERA_PORTS.items():
        capture_workers[camera_name] = CameraCaptureWorker(
            camera_name,
            usb_port,
            start_barrier,
            synchronizer.submit,
        )

    preview: NonBlockingPreviewWorker | None = None
    try:
        for worker in capture_workers.values():
            worker.open()

        preview = NonBlockingPreviewWorker(capture_workers)
        preview_holder["preview"] = preview
        preview.start()
        for worker in capture_workers.values():
            worker.start()

        print(
            f"[INFO] Software synchronization started: max_pair_skew={MAX_PAIR_SKEW_MS:.1f}ms"
        )
        bytes_per_pair = CAPTURE_SIZE[0] * CAPTURE_SIZE[1] * 3 * len(CAMERA_PORTS)
        estimated_mib_per_second = bytes_per_pair * MEASURED_FPS / (1024**2)
        print(
            "[WARNING] Raw BGR memory buffering grows by approximately "
            f"{estimated_mib_per_second:.0f} MiB/s at {MEASURED_FPS:.0f} paired FPS."
        )
        print("[INFO] Press 's' to start buffering, 'p' to stop/save, 'q' or ESC to quit.")

        last_status_print = 0.0
        while True:
            canvas = preview.latest_image()
            if canvas is not None:
                cv2.imshow(PREVIEW_WINDOW, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                recorder.start(recording_metadata(capture_workers))
            elif key == ord("p"):
                recorder.stop_and_save("user_stop")

            now = time.perf_counter()
            if now - last_status_print >= STATUS_PRINT_INTERVAL_S:
                sync_stats = synchronizer.stats()
                record_stats = recorder.status()
                print(
                    "[STATUS] "
                    f"thumb_fps={capture_workers['thumb_web_cam'].fps():.1f} "
                    f"middle_fps={capture_workers['middle_finger_cam'].fps():.1f} "
                    f"pairs={sync_stats['pair_count']} "
                    f"latest_skew={format_optional_ms(sync_stats['latest_signed_skew_ms'])} "
                    f"mean_abs_skew={format_optional_ms(sync_stats['mean_absolute_skew_ms'])} "
                    f"max_abs_skew={sync_stats['max_absolute_skew_ms']:.2f}ms "
                    f"dropped={sync_stats['dropped_unmatched']} "
                    f"recording={record_stats['recording']} "
                    f"buffered_pairs={record_stats['pair_count']} "
                    f"memory={record_stats['buffered_bytes'] / (1024**3):.2f}GiB"
                )
                last_status_print = now
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        if recorder.status()["recording"]:
            recorder.stop_and_save("shutdown")
        for worker in capture_workers.values():
            worker.stop()
        if preview is not None:
            preview.stop()
        cv2.destroyAllWindows()
        recorder.wait_for_save()
        final_stats = synchronizer.stats()
        print(
            "[INFO] Capture finished: "
            f"pairs={final_stats['pair_count']} "
            f"mean_abs_skew={format_optional_ms(final_stats['mean_absolute_skew_ms'])} "
            f"max_abs_skew={final_stats['max_absolute_skew_ms']:.2f}ms "
            f"dropped={final_stats['dropped_unmatched']}"
        )


if __name__ == "__main__":
    main()
