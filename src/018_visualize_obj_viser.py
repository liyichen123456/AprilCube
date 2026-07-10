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
DEFAULT_PORT = 8098
MM_TO_M = 0.001
CUBE_EDGE_MM = 18.75


@dataclass(frozen=True)
class ObjSpec:
    name: str
    path: Path
    color: tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize finger OBJ meshes in viser. Default inputs use cube-frame OBJ files."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--layout",
        choices=("row", "origin"),
        default="row",
        help="row separates objects for inspection; origin overlays all objects in their own cube frame.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=0.09,
        help="Object spacing in meters for --layout row.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=MM_TO_M,
        help="Mesh scale. Use 0.001 for OBJ coordinates in millimeters.",
    )
    parser.add_argument(
        "--obj",
        action="append",
        type=Path,
        default=None,
        help="Custom OBJ path. Can be repeated. Defaults to assets/*_cube_frame.obj.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Load assets/index.obj, middle.obj, thumb.obj instead of *_cube_frame.obj.",
    )
    parser.add_argument("--no-cube", action="store_true", help="Hide the 18.75mm reference cube wireframes.")
    parser.add_argument("--no-grid", action="store_true", help="Hide the ground/reference grid.")
    return parser.parse_args()


def default_obj_specs(*, raw: bool) -> list[ObjSpec]:
    suffix = ".obj" if raw else "_cube_frame.obj"
    return [
        ObjSpec("index", ASSETS_DIR / f"index{suffix}", (255, 145, 60)),
        ObjSpec("middle", ASSETS_DIR / f"middle{suffix}", (80, 180, 255)),
        ObjSpec("thumb", ASSETS_DIR / f"thumb{suffix}", (120, 220, 120)),
    ]


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
    for idx, path in enumerate(paths):
        resolved = path.expanduser().resolve()
        specs.append(ObjSpec(resolved.stem, resolved, palette[idx % len(palette)]))
    return specs


def load_mesh(path: Path, color: tuple[int, int, int]) -> trimesh.Trimesh:
    if not path.is_file():
        raise FileNotFoundError(f"OBJ file not found: {path}")

    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported trimesh load result for {path}: {type(loaded)!r}")

    rgba = np.asarray([color[0], color[1], color[2], 220], dtype=np.uint8)
    mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))
    return mesh


def object_position(index: int, count: int, layout: str, spacing: float) -> tuple[float, float, float]:
    if layout == "origin":
        return (0.0, 0.0, 0.0)
    x = (float(index) - (float(count) - 1.0) * 0.5) * float(spacing)
    return (x, 0.0, 0.0)


def cube_wireframe_points(edge_m: float) -> np.ndarray:
    h = edge_m * 0.5
    corners = np.asarray(
        [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    return np.asarray([[corners[a], corners[b]] for a, b in edges], dtype=np.float32)


def grid_points(half_size: float = 0.18, step: float = 0.02) -> np.ndarray:
    lines = []
    n = int(round(half_size / step))
    for i in range(-n, n + 1):
        p = float(i) * step
        lines.append([[p, -half_size, 0.0], [p, half_size, 0.0]])
        lines.append([[-half_size, p, 0.0], [half_size, p, 0.0]])
    return np.asarray(lines, dtype=np.float32)


def mesh_summary(name: str, mesh: trimesh.Trimesh, scale: float, position: tuple[float, float, float]) -> str:
    bounds_mm = np.asarray(mesh.bounds, dtype=np.float64)
    extents_mm = np.asarray(mesh.extents, dtype=np.float64)
    return "\n".join(
        [
            f"### {name}",
            f"- vertices: `{len(mesh.vertices)}`",
            f"- faces: `{len(mesh.faces)}`",
            f"- position_m: `({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})`",
            f"- scale: `{scale:g}`",
            (
                "- bounds_mm: "
                f"`[{bounds_mm[0, 0]:.3f}, {bounds_mm[0, 1]:.3f}, {bounds_mm[0, 2]:.3f}] -> "
                f"[{bounds_mm[1, 0]:.3f}, {bounds_mm[1, 1]:.3f}, {bounds_mm[1, 2]:.3f}]`"
            ),
            f"- extents_mm: `({extents_mm[0]:.3f}, {extents_mm[1]:.3f}, {extents_mm[2]:.3f})`",
        ]
    )


def add_object(
    server: viser.ViserServer,
    spec: ObjSpec,
    mesh: trimesh.Trimesh,
    *,
    position: tuple[float, float, float],
    scale: float,
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
    cube_handle = None
    if show_cube:
        cube_handle = server.scene.add_line_segments(
            f"{root}/cube_18p75mm",
            points=cube_wireframe_points(CUBE_EDGE_MM * scale),
            colors=np.asarray(spec.color, dtype=np.uint8),
            line_width=2.0,
            position=(0.0, 0.0, 0.0),
        )
    return {"frame": frame, "mesh": mesh_handle, "cube": cube_handle}


def main() -> None:
    args = parse_args()
    specs = custom_obj_specs(args.obj) if args.obj else default_obj_specs(raw=bool(args.raw))
    if not specs:
        raise ValueError("No OBJ files were provided.")

    meshes = [load_mesh(spec.path, spec.color) for spec in specs]

    server = viser.ViserServer(host=args.host, port=int(args.port))
    server.scene.set_up_direction("-y")
    server.scene.world_axes.visible = False
    server.gui.set_panel_label("OBJ Mesh Viewer")

    server.scene.add_frame(
        "/world",
        axes_length=0.05,
        axes_radius=0.0015,
        origin_radius=0.002,
    )
    grid_handle = server.scene.add_line_segments(
        "/world/xy_grid_z0",
        points=grid_points(),
        colors=(90, 90, 90),
        line_width=1.0,
        visible=not bool(args.no_grid),
    )

    handles = {}
    summaries = []
    for idx, (spec, mesh) in enumerate(zip(specs, meshes, strict=True)):
        position = object_position(idx, len(specs), args.layout, float(args.spacing))
        handles[spec.name] = add_object(
            server,
            spec,
            mesh,
            position=position,
            scale=float(args.scale),
            show_cube=not bool(args.no_cube),
        )
        summaries.append(mesh_summary(spec.name, mesh, float(args.scale), position))

    with server.gui.add_folder("Visibility"):
        grid_checkbox = server.gui.add_checkbox("Grid", initial_value=not bool(args.no_grid))
        cube_checkbox = server.gui.add_checkbox("18.75mm cubes", initial_value=not bool(args.no_cube))
        object_checkboxes = {
            spec.name: server.gui.add_checkbox(spec.name, initial_value=True)
            for spec in specs
        }

    @grid_checkbox.on_update
    def _on_grid(_event: Any) -> None:
        grid_handle.visible = bool(grid_checkbox.value)

    @cube_checkbox.on_update
    def _on_cube(_event: Any) -> None:
        for obj_handles in handles.values():
            cube = obj_handles.get("cube")
            if cube is not None:
                cube.visible = bool(cube_checkbox.value)

    for name, checkbox in object_checkboxes.items():
        @checkbox.on_update
        def _on_object(_event: Any, object_name: str = name) -> None:
            visible = bool(object_checkboxes[object_name].value)
            for handle in handles[object_name].values():
                if handle is not None:
                    handle.visible = visible

    server.gui.add_markdown(
        "\n\n".join(
            [
                f"layout: `{args.layout}`",
                f"unit scale: `{float(args.scale):g}`",
                f"cube edge: `{CUBE_EDGE_MM} mm`",
                *summaries,
            ]
        )
    )

    print(f"[INFO] viser server started at http://{args.host}:{int(args.port)}")
    print("[INFO] Loaded OBJ files:")
    for spec in specs:
        print(f"  - {spec.name}: {spec.path}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
