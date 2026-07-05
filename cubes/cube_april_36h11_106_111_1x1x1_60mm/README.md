# ArUco Cube — 1x1x1

![Cube preview](thumbnail.png)

## Parameters

| Parameter | Value |
|-----------|-------|
| Dictionary | `apriltag_36h11` |
| Grid | 1x1x1 (X x Y x Z tags) |
| Box dimensions | 75 x 75 x 75 mm |
| Tag size | 60 mm (8x8 cells) |
| Cell size | 7.5 mm |
| Margin | 1 cell (7.5 mm) |
| Border | 1 cell (7.5 mm) |
| Total tags | 6 |
| Tag IDs | 106–111 |

## Face Layout

| Face | Tag IDs |
|------|---------|
| +X | 106 |
| -X | 107 |
| +Y | 108 |
| -Y | 109 |
| +Z | 110 |
| -Z | 111 |

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
    106,
    107,
    108,
    109,
    110,
    111
  ],
  "faces": {
    "+X": [
      106
    ],
    "-X": [
      107
    ],
    "+Y": [
      108
    ],
    "-Y": [
      109
    ],
    "+Z": [
      110
    ],
    "-Z": [
      111
    ]
  },
  "tag_size_mm": 60.0,
  "cell_size_mm": 7.5,
  "margin_cells": 1,
  "border_cells": 1,
  "marker_pixels": 8,
  "box_dims": [
    75.0,
    75.0,
    75.0
  ]
}
```

## Regenerate

```bash
python generate_cube.py --grid 1x1x1 --dict apriltag_36h11 --tag-size 60 --margin-cell 1 --border-cell 1 -o cube_april_36h11_106_111_1x1x1_60mm
```
