# ArUco Cube — 1x1x1

![Cube preview](thumbnail.png)

## Parameters

| Parameter | Value |
|-----------|-------|
| Dictionary | `apriltag_36h11` |
| Grid | 1x1x1 (X x Y x Z tags) |
| Box dimensions | 12.5 x 12.5 x 12.5 mm |
| Tag size | 10 mm (8x8 cells) |
| Cell size | 1.25 mm |
| Margin | 1 cell (1.25 mm) |
| Border | 1 cell (1.25 mm) |
| Total tags | 6 |
| Tag IDs | 54–59 |

## Face Layout

| Face | Tag IDs |
|------|---------|
| +X | 54 |
| -X | 55 |
| +Y | 56 |
| -Y | 57 |
| +Z | 58 |
| -Z | 59 |

## Files

| File | Description |
|------|-------------|
| `cube.3mf` | Multi-color 3MF for Bambu Studio |
| `config.json` | Detector config (used by `detect_cube.py`) |
| `thumbnail.png` | 6-view preview |
| `mujoco/cube.xml` | MuJoCo MJCF model |
| `mujoco/cube.obj` | Wavefront OBJ mesh (UV-mapped) |
| `mujoco/cube.mtl` | OBJ material file |
| `mujoco/cube_atlas.png` | Texture atlas |

## Config JSON

```json
{
  "schema_version": 1,
  "target": {
    "type": "cuboid",
    "grid": "1x1x1"
  },
  "dict": "apriltag_36h11",
  "grid": "1x1x1",
  "tag_ids": [
    54,
    55,
    56,
    57,
    58,
    59
  ],
  "faces": {
    "+X": [
      54
    ],
    "-X": [
      55
    ],
    "+Y": [
      56
    ],
    "-Y": [
      57
    ],
    "+Z": [
      58
    ],
    "-Z": [
      59
    ]
  },
  "tag_size_mm": 10.0,
  "cell_size_mm": 1.25,
  "margin_cells": 1,
  "border_cells": 1,
  "marker_pixels": 8,
  "box_dims": [
    12.5,
    12.5,
    12.5
  ]
}
```

## Regenerate

```bash
aprilcube generate --grid 1x1x1 --dict apriltag_36h11 --tag-size 10 --margin-cell 1 --border-cell 1 -o cube_april_36h11_54_59_1x1x1_10mm
```
