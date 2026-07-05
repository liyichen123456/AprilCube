# ArUco Cube — 1x1x1

![Cube preview](thumbnail.png)

## Parameters

| Parameter | Value |
|-----------|-------|
| Dictionary | `apriltag_36h11` |
| Grid | 1x1x1 (X x Y x Z tags) |
| Box dimensions | 18.75 x 18.75 x 18.75 mm |
| Tag size | 15 mm (8x8 cells) |
| Cell size | 1.875 mm |
| Margin | 1 cell (1.875 mm) |
| Border | 1 cell (1.875 mm) |
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
  "tag_size_mm": 15.0,
  "cell_size_mm": 1.875,
  "margin_cells": 1,
  "border_cells": 1,
  "marker_pixels": 8,
  "box_dims": [
    18.75,
    18.75,
    18.75
  ]
}
```

## Regenerate

```bash
python generate_cube.py --grid 1x1x1 --dict apriltag_36h11 --tag-size 15 --margin-cell 1 --border-cell 1 -o cube_april_36h11_54_59_1x1x1_15mm
```
