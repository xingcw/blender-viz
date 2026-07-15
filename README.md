# Drone Racing Blender Viz

A cinematic Blender renderer for drone-racing rollouts. It reads the tracks in
`envs/mjx`, accepts common rollout formats, and generates the gates, racing
drone, animated flight, speed trail, direction markers, arena, lighting, and
cameras. It includes the MIT-licensed UZH Flightmare autonomous-racing gate.

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.10+, and Blender 4.x
on your `PATH`.

```bash
uv sync

# Render a built-in smooth demo around the lemniscate course
uv run drone-viz --track lemniscate --output renders/lemniscate.png \
  --blend renders/lemniscate.blend

# Render a rollout (the repository includes this small example)
uv run drone-viz --track lemniscate --trajectory examples/rollout.csv \
  --output renders/rollout.png --resolution 2560x1440

# Cinematic MP4; the output extension should be .mp4
uv run drone-viz --track powerloop --trajectory rollout.npz \
  --animation --camera chase --output renders/powerloop.mp4
```

The launcher also discovers Blender at `/opt/blender/blender` automatically. For
other installations not on `PATH`, pass `--blender /path/to/blender`. To build and
inspect the scene interactively without rendering, add `--open`. A render can
use `--engine CYCLES --samples 128` for final-quality ray tracing; Eevee is the
fast default.

PNG renders use a transparent background by default. Add `--opaque-background`
to restore the dark arena floor and world used by the cinematic presentation.

## Rollout formats

- CSV: `x,y,z` (also `px,py,pz` or `pos_x,pos_y,pos_z`), with optional
  `qw,qx,qy,qz`, `time`/`t`, and `speed` columns.
- JSON: `{ "positions": [[x,y,z], ...], "quaternions": [[w,x,y,z], ...] }`,
  or simply a list of positions.
- NPY: an `N x 3` position array or `N x 7` MuJoCo `qpos` array.
- NPZ: `positions`, `position`, `pos`, or `qpos`; optional `quaternions`,
  `time`, and `speed`. Run `uv sync --extra numpy` to enable NumPy input.

MuJoCo and Blender both use scalar-first quaternions here: `[w, x, y, z]`.
Positions and gate transforms remain in the simulator's metric world frame.

## Tracks and camera styles

`--track` selects a generated `envs/mjx/racing_simple_<track>.xml`; currently
`straight`, `straight2`, `circle4`, `lemniscate`, `complex`, and `powerloop` are
available. You can instead pass any compatible file with `--mjcf scene.xml`.

Camera choices are `hero` (course overview), `top` (diagram-like presentation),
and `chase` (drone-mounted animation). The saved `.blend` contains named scene
objects and editable materials, so art direction can continue normally.

## Validation and tests

The data layer does not import Blender and works on training machines:

```bash
uv run drone-viz --track circle4 --trajectory rollout.csv --dry-run
uv run python -m unittest discover -s tests
```

The generated visual language intentionally echoes the supplied reference while
using a richer 3D treatment: a cyan-to-magenta velocity ribbon, amber direction
chevrons, realistic competition gates, reflective arena markings, colored rim
lights, and bloom.
