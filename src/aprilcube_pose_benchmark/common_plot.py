from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PLOT_COLORS_BGR = [
    (30, 80, 220),
    (20, 160, 40),
    (220, 80, 30),
    (180, 80, 180),
    (0, 150, 180),
    (120, 120, 20),
    (80, 80, 80),
    (0, 0, 0),
]


def unwrap_rpy_deg(rpy_deg: np.ndarray) -> np.ndarray:
    arr = np.asarray(rpy_deg, dtype=np.float64)
    out = np.array(arr, copy=True)
    valid = np.all(np.isfinite(out), axis=1)
    if np.count_nonzero(valid) >= 2:
        out[valid] = np.degrees(np.unwrap(np.radians(out[valid]), axis=0))
    return out


def _draw_panel(
    img: np.ndarray,
    rect: tuple[int, int, int, int],
    x: np.ndarray,
    series: list[np.ndarray],
    labels: list[str],
    title: str,
) -> None:
    x0, y0, w, h = rect
    pad_l, pad_r, pad_t, pad_b = 58, 16, 34, 36
    plot_x0, plot_y0 = x0 + pad_l, y0 + pad_t
    plot_x1, plot_y1 = x0 + w - pad_r, y0 + h - pad_b
    cv2.rectangle(img, (x0, y0), (x0 + w - 1, y0 + h - 1), (230, 230, 230), 1)
    cv2.putText(img, title, (x0 + 10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.line(img, (plot_x0, plot_y1), (plot_x1, plot_y1), (120, 120, 120), 1)
    cv2.line(img, (plot_x0, plot_y0), (plot_x0, plot_y1), (120, 120, 120), 1)

    finite_values = []
    for values in series:
        arr = np.asarray(values, dtype=np.float64)
        finite_values.extend(arr[np.isfinite(arr)].tolist())
    if not finite_values:
        cv2.putText(img, "no valid pose", (plot_x0 + 20, plot_y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
        return
    y_min = float(np.min(finite_values))
    y_max = float(np.max(finite_values))
    if abs(y_max - y_min) < 1e-9:
        y_min -= 1.0
        y_max += 1.0
    margin = 0.08 * (y_max - y_min)
    y_min -= margin
    y_max += margin
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if abs(x_max - x_min) < 1e-9:
        x_max = x_min + 1.0

    for i in range(5):
        gy = int(round(plot_y0 + i * (plot_y1 - plot_y0) / 4.0))
        cv2.line(img, (plot_x0, gy), (plot_x1, gy), (235, 235, 235), 1)
    cv2.putText(img, f"{y_max:.1f}", (x0 + 4, plot_y0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1)
    cv2.putText(img, f"{y_min:.1f}", (x0 + 4, plot_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1)

    for series_idx, values in enumerate(series):
        arr = np.asarray(values, dtype=np.float64)
        pts = []
        for xi, yi in zip(x, arr):
            if not np.isfinite(xi) or not np.isfinite(yi):
                if len(pts) >= 2:
                    cv2.polylines(img, [np.asarray(pts, dtype=np.int32)], False, PLOT_COLORS_BGR[series_idx % len(PLOT_COLORS_BGR)], 2)
                pts = []
                continue
            px = int(round(plot_x0 + (float(xi) - x_min) / (x_max - x_min) * (plot_x1 - plot_x0)))
            py = int(round(plot_y1 - (float(yi) - y_min) / (y_max - y_min) * (plot_y1 - plot_y0)))
            pts.append((px, py))
        if len(pts) >= 2:
            cv2.polylines(img, [np.asarray(pts, dtype=np.int32)], False, PLOT_COLORS_BGR[series_idx % len(PLOT_COLORS_BGR)], 2)

    legend_x = plot_x0 + 8
    legend_y = plot_y0 + 18
    for idx, label in enumerate(labels[:8]):
        color = PLOT_COLORS_BGR[idx % len(PLOT_COLORS_BGR)]
        cv2.line(img, (legend_x, legend_y + idx * 16), (legend_x + 18, legend_y + idx * 16), color, 2)
        cv2.putText(
            img,
            label[:42],
            (legend_x + 24, legend_y + idx * 16 + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )


def save_pose_curve(
    output_dir: str | Path,
    algorithm_name: str,
    frame_indices: np.ndarray,
    xyz_mm: np.ndarray,
    rpy_deg: np.ndarray,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = 1400, 820
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    rpy_plot = unwrap_rpy_deg(rpy_deg)
    _draw_panel(
        img,
        (20, 20, width - 40, 370),
        frame_indices,
        [xyz_mm[:, 0], xyz_mm[:, 1], xyz_mm[:, 2]],
        ["x", "y", "z"],
        f"{algorithm_name} translation (mm)",
    )
    _draw_panel(
        img,
        (20, 420, width - 40, 370),
        frame_indices,
        [rpy_plot[:, 0], rpy_plot[:, 1], rpy_plot[:, 2]],
        ["roll", "pitch", "yaw"],
        f"{algorithm_name} rotation xyz euler (deg)",
    )
    path = out_dir / "pose_xyz_rpy.png"
    cv2.imwrite(str(path), img)
    return path


def save_reprojection_curve(
    output_dir: str | Path,
    algorithm_name: str,
    frame_indices: np.ndarray,
    reproj_error_px: np.ndarray,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = np.full((420, 1400, 3), 255, dtype=np.uint8)
    _draw_panel(
        img,
        (20, 20, 1360, 370),
        frame_indices,
        [reproj_error_px],
        ["reproj"],
        f"{algorithm_name} reprojection error (px)",
    )
    path = out_dir / "reproj_error.png"
    cv2.imwrite(str(path), img)
    return path


def save_compare_pose_curve(
    output_path: str | Path,
    algorithm_outputs: dict[str, dict[str, np.ndarray]],
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1800, 1100
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    panel_w = width // 3
    panel_h = height // 2
    panel_specs = [
        ("xyz_mm", 0, "x mm"),
        ("xyz_mm", 1, "y mm"),
        ("xyz_mm", 2, "z mm"),
        ("rpy_deg", 0, "roll deg"),
        ("rpy_deg", 1, "pitch deg"),
        ("rpy_deg", 2, "yaw deg"),
    ]
    alg_names = list(algorithm_outputs.keys())
    for panel_idx, (key, axis_idx, title) in enumerate(panel_specs):
        row = panel_idx // 3
        col = panel_idx % 3
        series = []
        x_ref = None
        for alg_name in alg_names:
            data = algorithm_outputs[alg_name]
            x_ref = data["frame_indices"] if x_ref is None else x_ref
            values = np.asarray(data[key], dtype=np.float64)
            if key == "rpy_deg":
                values = unwrap_rpy_deg(values)
            series.append(values[:, axis_idx])
        _draw_panel(
            img,
            (col * panel_w + 10, row * panel_h + 10, panel_w - 20, panel_h - 20),
            np.asarray(x_ref, dtype=np.float64),
            series,
            alg_names,
            title,
        )
    cv2.imwrite(str(path), img)
    return path


def compute_metrics(
    success: np.ndarray,
    xyz_mm: np.ndarray,
    rpy_deg: np.ndarray,
    reproj_error_px: np.ndarray,
) -> dict[str, Any]:
    valid = np.asarray(success, dtype=bool)
    metrics: dict[str, Any] = {
        "num_frames": int(len(success)),
        "num_success": int(np.count_nonzero(valid)),
        "success_rate": float(np.mean(valid)) if len(valid) else 0.0,
    }
    if np.count_nonzero(valid) >= 2:
        xyz_valid = xyz_mm[valid]
        rpy_valid = unwrap_rpy_deg(rpy_deg[valid])
        d_xyz = np.linalg.norm(np.diff(xyz_valid, axis=0), axis=1)
        d_rpy = np.linalg.norm(np.diff(rpy_valid, axis=0), axis=1)
        metrics.update(
            {
                "translation_step_mean_mm": float(np.nanmean(d_xyz)),
                "translation_step_std_mm": float(np.nanstd(d_xyz)),
                "rotation_step_mean_deg": float(np.nanmean(d_rpy)),
                "rotation_step_std_deg": float(np.nanstd(d_rpy)),
                "reproj_mean_px": float(np.nanmean(reproj_error_px[valid])),
                "reproj_std_px": float(np.nanstd(reproj_error_px[valid])),
            }
        )
    return metrics


def save_metrics(output_dir: str | Path, metrics: dict[str, Any]) -> Path:
    path = Path(output_dir) / "metrics.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return path


def save_npz(
    output_dir: str | Path,
    *,
    frame_indices: np.ndarray,
    success: np.ndarray,
    xyz_mm: np.ndarray,
    rpy_deg: np.ndarray,
    reproj_error_px: np.ndarray,
    visible_face_count: np.ndarray,
    visible_tag_count: np.ndarray,
) -> Path:
    path = Path(output_dir) / "poses.npz"
    np.savez(
        path,
        frame_indices=frame_indices,
        success=success,
        xyz_mm=xyz_mm,
        rpy_deg=rpy_deg,
        reproj_error_px=reproj_error_px,
        visible_face_count=visible_face_count,
        visible_tag_count=visible_tag_count,
    )
    return path

