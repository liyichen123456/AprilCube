# ArUco Cube — 1x1x1

![Cube preview](thumbnail.png)

## Parameters

| Parameter | Value |
|-----------|-------|
| Dictionary | `apriltag_36h11` |
| Grid | 1x1x1 (X x Y x Z tags) |
| Box dimensions | 25 x 25 x 25 mm |
| Tag size | 20 mm (8x8 cells) |
| Cell size | 2.5 mm |
| Margin | 1 cell (2.5 mm) |
| Border | 1 cell (2.5 mm) |
| Total tags | 6 |
| Tag IDs | 36–41 |

## Face Layout

| Face | Tag IDs |
|------|---------|
| +X | 36 |
| -X | 37 |
| +Y | 38 |
| -Y | 39 |
| +Z | 40 |
| -Z | 41 |

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
    36,
    37,
    38,
    39,
    40,
    41
  ],
  "faces": {
    "+X": [
      36
    ],
    "-X": [
      37
    ],
    "+Y": [
      38
    ],
    "-Y": [
      39
    ],
    "+Z": [
      40
    ],
    "-Z": [
      41
    ]
  },
  "tag_size_mm": 20.0,
  "cell_size_mm": 2.5,
  "margin_cells": 1,
  "border_cells": 1,
  "marker_pixels": 8,
  "box_dims": [
    25.0,
    25.0,
    25.0
  ]
}
```

## Regenerate

```bash
aprilcube generate --grid 1x1x1 --dict apriltag_36h11 --tag-size 20 --margin-cell 1 --border-cell 1 -o cube_april_36h11_36_41_1x1x1_20mm
```
