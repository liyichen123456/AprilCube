#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import viser


APRILCUBE_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = APRILCUBE_ROOT / "assets"
DEFAULT_MESH = Path(
    "/home/ps/project/ConSensV2Lab/thirdparty/simplify_wuji_xarm_adapter.stl"
)
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8098
DEFAULT_SCALE = 0.001
DEFAULT_ORIGIN_MM = np.asarray([32.5, 26.0, 77.5], dtype=np.float64)
DEFAULT_FRAME_YAW_DEG = 90.0
DEFAULT_CUBE_EDGE_MM = 18.75


@dataclass(frozen=True)
class ObjSpec:
    name: str
    path: Path
    color: tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one mesh in a configurable local frame, or inspect multiple "
            "finger/custom meshes with reference cubes."
        )
    )
    parser.add_argument(
        "mesh",
        nargs="?",
        type=Path,
        default=DEFAULT_MESH,
        help="Single mesh for local-frame mode (default: the Wuji adapter STL).",
    )
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
    parser.add_argument(
        "--finger-objs",
        action="store_true",
        help="Use multi-mesh mode with the default index/middle/thumb assets.",
    )
    parser.add_argument(
        "--obj",
        action="append",
        type=Path,
        default=None,
        help="Custom mesh for multi-mesh mode; may be repeated.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help=(
            "With --finger-objs, force assets/*.obj; otherwise prefer "
            "*_cube_frame.obj and fall back to the available *.obj files."
        ),
    )
    parser.add_argument(
        "--layout",
        choices=("row", "origin"),
        default="row",
        help="Multi-mesh layout: separate objects in a row or overlay at the origin.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=0.09,
        help="Object spacing in meters for the multi-mesh row layout.",
    )
    parser.add_argument(
        "--cube-edge-mm",
        type=float,
        default=DEFAULT_CUBE_EDGE_MM,
        help="Reference-cube edge length in millimeters for multi-mesh mode.",
    )
    parser.add_argument("--no-cube", action="store_true", help="Hide reference cubes.")
    parser.add_argument("--no-grid", action="store_true", help="Hide the reference grid.")
    return parser.parse_args()


def load_mesh(
    path: Path,
    color: tuple[int, int, int] = (70, 155, 225),
) -> trimesh.Trimesh:
    if not path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {path}")
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
    rgba = np.asarray([color[0], color[1], color[2], 230], dtype=np.uint8)
    mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))
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


def default_obj_specs(*, raw: bool) -> list[ObjSpec]:
    colors = {
        "index": (255, 145, 60),
        "middle": (80, 180, 255),
        "thumb": (120, 220, 120),
    }
    specs: list[ObjSpec] = []
    for name, color in colors.items():
        raw_path = ASSETS_DIR / f"{name}.obj"
        cube_frame_path = ASSETS_DIR / f"{name}_cube_frame.obj"
        selected_path = raw_path if raw or not cube_frame_path.is_file() else cube_frame_path
        specs.append(ObjSpec(name, selected_path, color))
    return specs


def custom_obj_specs(paths: list[Path]) -> list[ObjSpec]:
    palette = [
        (255, 145, 60),
        (80, 180, 255),
        (120, 220, 120),
        (220, 120, 255),
        (255, 220, 80),
        (180, 180, 180),
    ]
    specs: list[ObjSpec] = []
    name_counts: dict[str, int] = {}
    for idx, path in enumerate(paths):
        resolved = path.expanduser().resolve()
        base_name = resolved.stem
        count = name_counts.get(base_name, 0)
        name_counts[base_name] = count + 1
        name = base_name if count == 0 else f"{base_name}_{count + 1}"
        specs.append(ObjSpec(name, resolved, palette[idx % len(palette)]))
    return specs


def object_position(
    index: int,
    count: int,
    layout: str,
    spacing: float,
) -> tuple[float, float, float]:
    if layout == "origin":
        return (0.0, 0.0, 0.0)
    x = (float(index) - (float(count) - 1.0) * 0.5) * float(spacing)
    return (x, 0.0, 0.0)


def cube_wireframe_points(edge_m: float) -> np.ndarray:
    half = edge_m * 0.5
    corners = np.asarray(
        [
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
        ],
        dtype=np.float32,
    )
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    return np.asarray([[corners[a], corners[b]] for a, b in edges], dtype=np.float32)


def multi_mesh_summary(
    spec: ObjSpec,
    mesh: trimesh.Trimesh,
    scale: float,
    position: tuple[float, float, float],
) -> str:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = np.asarray(mesh.extents, dtype=np.float64)
    return "\n".join(
        [
            f"### {spec.name}",
            f"- path: `{spec.path}`",
            f"- vertices/faces: `{len(mesh.vertices)}/{len(mesh.faces)}`",
            f"- position_m: `({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})`",
            f"- scale: `{scale:g}`",
            (
                "- bounds: "
                f"`[{bounds[0, 0]:.3f}, {bounds[0, 1]:.3f}, {bounds[0, 2]:.3f}] -> "
                f"[{bounds[1, 0]:.3f}, {bounds[1, 1]:.3f}, {bounds[1, 2]:.3f}]`"
            ),
            f"- extents: `({extents[0]:.3f}, {extents[1]:.3f}, {extents[2]:.3f})`",
        ]
    )


def add_multi_object(
    server: viser.ViserServer,
    spec: ObjSpec,
    mesh: trimesh.Trimesh,
    *,
    position: tuple[float, float, float],
    scale: float,
    cube_edge_m: float,
    show_cube: bool,
) -> dict[str, Any]:
    root = f"/objs/{spec.name}"
    frame = server.scene.add_frame(
        root,
        position=position,
        axes_length=0.03,
        axes_radius=0.0012,
        origin_radius=0.002,
    )
    mesh_handle = server.scene.add_mesh_trimesh(
        f"{root}/mesh",
        mesh,
        scale=scale,
        position=(0.0, 0.0, 0.0),
        cast_shadow=False,
        receive_shadow=False,
    )
    cube_handle = server.scene.add_line_segments(
        f"{root}/reference_cube",
        points=cube_wireframe_points(cube_edge_m),
        colors=np.asarray(spec.color, dtype=np.uint8),
        line_width=2.0,
        position=(0.0, 0.0, 0.0),
        visible=show_cube,
    )
    return {"frame": frame, "mesh": mesh_handle, "cube": cube_handle}


def run_multi_mesh_viewer(args: argparse.Namespace) -> None:
    if args.obj and args.finger_objs:
        raise ValueError("Use either --obj or --finger-objs, not both.")
    if args.obj and args.raw:
        raise ValueError("--raw only applies to the default --finger-objs assets.")

    specs = custom_obj_specs(args.obj) if args.obj else default_obj_specs(raw=bool(args.raw))
    if not specs:
        raise ValueError("No meshes were provided.")
    if not args.obj and not args.raw:
        fallback_names = [spec.name for spec in specs if not spec.path.name.endswith("_cube_frame.obj")]
        if fallback_names:
            print(
                "[WARNING] Cube-frame filenames are unavailable; using assets/*.obj for: "
                + ", ".join(fallback_names)
            )

    scale = float(args.scale)
    spacing = float(args.spacing)
    cube_edge_mm = float(args.cube_edge_mm)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")
    if not np.isfinite(spacing) or spacing < 0.0:
        raise ValueError(f"spacing must be non-negative, got {spacing}")
    if not np.isfinite(cube_edge_mm) or cube_edge_mm <= 0.0:
        raise ValueError(f"cube edge must be positive, got {cube_edge_mm}")

    meshes = [load_mesh(spec.path, spec.color) for spec in specs]
    server = viser.ViserServer(host=str(args.host), port=int(args.port))
    server.scene.set_up_direction("-y")
    server.scene.world_axes.visible = False
    server.gui.set_panel_label("Multi-Mesh Viewer")
    server.scene.add_frame(
        "/world",
        axes_length=0.05,
        axes_radius=0.0015,
        origin_radius=0.002,
    )
    grid_handle = server.scene.add_grid(
        "/world/reference_grid",
        width=0.36,
        height=0.36,
        plane="xy",
        cell_size=0.02,
        section_size=0.1,
        visible=not bool(args.no_grid),
    )

    handles: dict[str, dict[str, Any]] = {}
    summaries: list[str] = []
    cube_edge_m = cube_edge_mm * 0.001
    for idx, (spec, mesh) in enumerate(zip(specs, meshes, strict=True)):
        position = object_position(idx, len(specs), str(args.layout), spacing)
        handles[spec.name] = add_multi_object(
            server,
            spec,
            mesh,
            position=position,
            scale=scale,
            cube_edge_m=cube_edge_m,
            show_cube=not bool(args.no_cube),
        )
        summaries.append(multi_mesh_summary(spec, mesh, scale, position))

    with server.gui.add_folder("Visibility"):
        grid_checkbox = server.gui.add_checkbox("Grid", initial_value=not bool(args.no_grid))
        cube_checkbox = server.gui.add_checkbox(
            f"{cube_edge_mm:g}mm cubes",
            initial_value=not bool(args.no_cube),
        )
        object_checkboxes = {
            spec.name: server.gui.add_checkbox(spec.name, initial_value=True)
            for spec in specs
        }

    @grid_checkbox.on_update
    def _on_grid(_event: Any) -> None:
        grid_handle.visible = bool(grid_checkbox.value)

    @cube_checkbox.on_update
    def _on_cube(_event: Any) -> None:
        for object_handles in handles.values():
            object_handles["cube"].visible = bool(cube_checkbox.value)

    for name, checkbox in object_checkboxes.items():
        @checkbox.on_update
        def _on_object(_event: Any, object_name: str = name) -> None:
            visible = bool(object_checkboxes[object_name].value)
            for handle in handles[object_name].values():
                handle.visible = visible

    server.gui.add_markdown(
        "\n\n".join(
            [
                f"layout: `{args.layout}`",
                f"unit scale: `{scale:g}`",
                f"cube edge: `{cube_edge_mm:g} mm`",
                *summaries,
            ]
        )
    )
    print(f"[INFO] Multi-mesh viser: http://localhost:{int(args.port)}")
    for spec in specs:
        print(f"[INFO] {spec.name}: {spec.path}")
    while True:
        time.sleep(1.0)


def run_local_frame_viewer(args: argparse.Namespace) -> None:
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
        visible=not bool(args.no_grid),
    )

    with server.gui.add_folder("Visibility"):
        show_mesh = server.gui.add_checkbox("Mesh", initial_value=True)
        show_frame = server.gui.add_checkbox("Local axes", initial_value=True)
        show_bounds = server.gui.add_checkbox("Bounds", initial_value=True)
        show_grid = server.gui.add_checkbox("Grid", initial_value=not bool(args.no_grid))

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


def main() -> None:
    args = parse_args()
    multi_mesh_mode = bool(args.finger_objs or args.obj or args.raw)
    if multi_mesh_mode:
        run_multi_mesh_viewer(args)
    else:
        run_local_frame_viewer(args)


if __name__ == "__main__":
    main()
