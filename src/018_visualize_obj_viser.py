#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import viser


DEFAULT_MESH = Path(
    "/home/ps/project/ConSensV2Lab/thirdparty/simplify_wuji_xarm_adapter.stl"
)
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8098
DEFAULT_SCALE = 0.001
DEFAULT_ORIGIN_MM = np.asarray([32.5, 26.0, 77.5], dtype=np.float64)
DEFAULT_FRAME_YAW_DEG = 90.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a mesh relative to a selected local coordinate origin."
    )
    parser.add_argument("mesh", nargs="?", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help="Mesh-unit to meter scale; use 0.001 for millimeter STL files.",
    )
    parser.add_argument(
        "--origin",
        type=float,
        nargs=3,
        default=DEFAULT_ORIGIN_MM,
        metavar=("X", "Y", "Z"),
        help="Local-frame origin in the unscaled mesh coordinates.",
    )
    parser.add_argument(
        "--frame-yaw-deg",
        type=float,
        default=DEFAULT_FRAME_YAW_DEG,
        help="New local frame's positive rotation about the original +z axis.",
    )
    parser.add_argument(
        "--mesh-yaw-deg",
        type=float,
        default=0.0,
        help="Active mesh rotation about the displayed local +z axis.",
    )
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"Mesh scene contains no geometry: {path}")
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported mesh type {type(loaded)!r}: {path}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Mesh is empty: {path}")
    mesh.visual.vertex_colors = np.tile(
        np.asarray([70, 155, 225, 230], dtype=np.uint8),
        (len(mesh.vertices), 1),
    )
    return mesh


def summary_markdown(
    path: Path,
    mesh: trimesh.Trimesh,
    scale: float,
    origin: np.ndarray,
    frame_yaw_deg: float,
    mesh_yaw_deg: float,
) -> str:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = np.asarray(mesh.extents, dtype=np.float64)
    return "\n".join(
        [
            f"**Mesh:** `{path}`",
            f"**Vertices:** `{len(mesh.vertices)}`",
            f"**Faces:** `{len(mesh.faces)}`",
            f"**Unit scale:** `{scale:g}`",
            (
                "**Local origin (mesh units):** "
                f"`({origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f})`"
            ),
            "**Local frame:** `+x/+y` lie on the flange plane; `+z` is its normal",
            f"**Frame yaw about original +z:** `{frame_yaw_deg:.3f} deg`",
            f"**Active mesh yaw about displayed +z:** `{mesh_yaw_deg:.3f} deg`",
            (
                "**Bounds (mesh units):** "
                f"`({bounds[0, 0]:.3f}, {bounds[0, 1]:.3f}, {bounds[0, 2]:.3f})` "
                f"to `({bounds[1, 0]:.3f}, {bounds[1, 1]:.3f}, {bounds[1, 2]:.3f})`"
            ),
            (
                "**Extents (mesh units):** "
                f"`({extents[0]:.3f}, {extents[1]:.3f}, {extents[2]:.3f})`"
            ),
        ]
    )


def main() -> None:
    args = parse_args()
    mesh_path = args.mesh.expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    scale = float(args.scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")
    mesh = load_mesh(mesh_path)
    origin = np.asarray(args.origin, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError(f"origin must contain three finite values, got {origin}")
    frame_yaw_deg = float(args.frame_yaw_deg)
    if not np.isfinite(frame_yaw_deg):
        raise ValueError(f"frame yaw must be finite, got {frame_yaw_deg}")
    mesh_yaw_deg = float(args.mesh_yaw_deg)
    if not np.isfinite(mesh_yaw_deg):
        raise ValueError(f"mesh yaw must be finite, got {mesh_yaw_deg}")

    # Express the mesh in a frame rotated positively about the original +z axis.
    # Coordinate conversion therefore applies the inverse frame rotation.
    inverse_yaw = np.deg2rad(-frame_yaw_deg)
    cos_yaw = np.cos(inverse_yaw)
    sin_yaw = np.sin(inverse_yaw)
    rotation_new_from_old = np.asarray(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    display_mesh = mesh.copy()
    centered_vertices = np.asarray(display_mesh.vertices) - origin
    display_mesh.vertices = centered_vertices @ rotation_new_from_old.T

    mesh_yaw_rad = np.deg2rad(mesh_yaw_deg)
    mesh_wxyz = np.asarray(
        [np.cos(mesh_yaw_rad / 2.0), 0.0, 0.0, np.sin(mesh_yaw_rad / 2.0)],
        dtype=np.float64,
    )
    mesh_rotation = np.asarray(
        [
            [np.cos(mesh_yaw_rad), -np.sin(mesh_yaw_rad), 0.0],
            [np.sin(mesh_yaw_rad), np.cos(mesh_yaw_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rotated_vertices = np.asarray(display_mesh.vertices) @ mesh_rotation.T
    bounds_m = np.stack(
        [rotated_vertices.min(axis=0), rotated_vertices.max(axis=0)], axis=0
    ) * scale
    center_m = np.mean(bounds_m, axis=0)
    extents_m = np.asarray(display_mesh.extents, dtype=np.float64) * scale
    scene_size = max(float(np.max(extents_m)), 0.01)

    server = viser.ViserServer(host=str(args.host), port=int(args.port))
    server.gui.set_panel_label("STL Coordinate Viewer")
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = False
    server.initial_camera.position = tuple(
        center_m + np.asarray([2.2, -2.5, 1.8]) * scene_size
    )
    server.initial_camera.look_at = tuple(center_m)
    server.initial_camera.up = (0.0, 0.0, 1.0)
    server.initial_camera.near = max(scene_size * 0.002, 1e-5)

    root = "/mesh_local_frame"
    frame = server.scene.add_frame(
        root,
        axes_length=scene_size * 0.75,
        axes_radius=scene_size * 0.018,
        origin_radius=scene_size * 0.04,
    )
    mesh_handle = server.scene.add_mesh_trimesh(
        f"{root}/mesh",
        display_mesh,
        scale=scale,
        wxyz=mesh_wxyz,
        cast_shadow=False,
        receive_shadow=False,
    )
    bounds_handle = server.scene.add_box(
        f"{root}/bounds",
        dimensions=tuple(extents_m),
        position=tuple(center_m),
        color=(245, 185, 45),
        opacity=0.12,
        side="double",
    )
    grid_handle = server.scene.add_grid(
        "/reference_grid",
        width=max(scene_size * 4.0, 0.2),
        height=max(scene_size * 4.0, 0.2),
        plane="xy",
        cell_size=0.01,
        section_size=0.05,
    )

    with server.gui.add_folder("Visibility"):
        show_mesh = server.gui.add_checkbox("Mesh", initial_value=True)
        show_frame = server.gui.add_checkbox("Local axes", initial_value=True)
        show_bounds = server.gui.add_checkbox("Bounds", initial_value=True)
        show_grid = server.gui.add_checkbox("Grid", initial_value=True)

    with server.gui.add_folder("Transform"):
        mesh_yaw = server.gui.add_slider(
            "Mesh yaw (deg)",
            min=-180.0,
            max=180.0,
            step=1.0,
            initial_value=mesh_yaw_deg,
        )

    @show_mesh.on_update
    def _on_mesh(_event: Any) -> None:
        mesh_handle.visible = bool(show_mesh.value)

    @show_frame.on_update
    def _on_frame(_event: Any) -> None:
        frame.visible = bool(show_frame.value)

    @show_bounds.on_update
    def _on_bounds(_event: Any) -> None:
        bounds_handle.visible = bool(show_bounds.value)

    @show_grid.on_update
    def _on_grid(_event: Any) -> None:
        grid_handle.visible = bool(show_grid.value)

    @mesh_yaw.on_update
    def _on_mesh_yaw(_event: Any) -> None:
        yaw_rad = np.deg2rad(float(mesh_yaw.value))
        mesh_handle.wxyz = np.asarray(
            [np.cos(yaw_rad / 2.0), 0.0, 0.0, np.sin(yaw_rad / 2.0)],
            dtype=np.float64,
        )

    server.gui.add_markdown(
        summary_markdown(
            mesh_path, mesh, scale, origin, frame_yaw_deg, mesh_yaw_deg
        )
    )
    print(f"[INFO] Mesh: {mesh_path}")
    print(f"[INFO] Local origin (mesh units): {origin.tolist()}")
    print(f"[INFO] Frame yaw about original +z: {frame_yaw_deg:g} deg")
    print(f"[INFO] Active mesh yaw about displayed +z: {mesh_yaw_deg:g} deg")
    print(f"[INFO] Bounds: {mesh.bounds.tolist()}")
    print(f"[INFO] Extents: {mesh.extents.tolist()}")
    print(f"[INFO] Viser: http://localhost:{int(args.port)}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
