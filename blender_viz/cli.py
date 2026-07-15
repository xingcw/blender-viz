from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from .data import demo_trajectory, load_gates, load_trajectory

ROOT = Path(__file__).resolve().parents[1]


def find_blender(requested: str) -> str | None:
    """Resolve Blender from PATH or common Linux/macOS install locations."""
    resolved = shutil.which(requested)
    if resolved:
        return resolved
    # Only apply fallbacks for the default name; an explicit custom path should
    # fail clearly instead of silently selecting a different installation.
    if requested != "blender":
        return None
    candidates = (
        Path("/opt/blender/blender"),
        Path("/usr/local/bin/blender"),
        Path("/Applications/Blender.app/Contents/MacOS/Blender"),
    )
    return next((str(path) for path in candidates if path.is_file() and path.stat().st_mode & 0o111), None)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render drone-racing trajectories in Blender")
    p.add_argument("--trajectory", "-t", type=Path, help="CSV, JSON, NPY, or NPZ rollout (omit for demo line)")
    p.add_argument("--track", default="lemniscate", help="env track name (default: lemniscate)")
    p.add_argument("--mjcf", type=Path, help="explicit MJCF file; overrides --track")
    p.add_argument("--output", "-o", type=Path, default=Path("renders/trajectory.png"))
    p.add_argument("--blend", type=Path, help="also save an editable .blend scene")
    p.add_argument("--engine", choices=("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"), default="BLENDER_EEVEE_NEXT")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--resolution", default="1920x1080", help="WIDTHxHEIGHT")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--animation", action="store_true", help="render MP4 animation instead of a still")
    p.add_argument("--camera", choices=("hero", "top", "chase"), default="hero")
    p.add_argument(
        "--opaque-background",
        action="store_true",
        help="render the arena floor and black world instead of transparent RGBA",
    )
    p.add_argument("--blender", default="blender", help="Blender executable")
    p.add_argument("--open", action="store_true", dest="open_ui", help="open the scene in Blender without rendering")
    p.add_argument("--dry-run", action="store_true", help="validate and print scene specification")
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    mjcf = (args.mjcf or ROOT / "envs" / "mjx" / f"racing_simple_{args.track}.xml").resolve()
    if not mjcf.exists():
        available = ", ".join(
            p.stem.removeprefix("racing_simple_") for p in sorted((ROOT / "envs/mjx").glob("racing_simple_*.xml"))
        )
        raise SystemExit(f"Track MJCF not found: {mjcf}\nAvailable tracks: {available}")
    gates = load_gates(mjcf)
    trajectory = load_trajectory(args.trajectory) if args.trajectory else demo_trajectory(gates)
    try:
        width, height = (int(v) for v in args.resolution.lower().split("x", 1))
    except ValueError:
        raise SystemExit("--resolution must look like 1920x1080")
    spec = {
        "trajectory": trajectory,
        "gates": gates,
        "track": args.track,
        "output": str(args.output.resolve()),
        "blend": str(args.blend.resolve()) if args.blend else None,
        "assets": str((ROOT / "envs/mjx/assets").resolve()),
        "gate_asset": str((ROOT / "assets/third_party/flightmare_gate/rpg_gate.blend").resolve()),
        "gate_texture": str((ROOT / "assets/third_party/flightmare_gate/RPGGate.png").resolve()),
        "transparent": not args.opaque_background,
        "engine": args.engine,
        "samples": args.samples,
        "width": width,
        "height": height,
        "fps": args.fps,
        "animation": args.animation,
        "camera": args.camera,
        "open_ui": args.open_ui,
    }
    if args.dry_run:
        summary = {**spec, "trajectory": {k: (len(v) if isinstance(v, list) else v) for k, v in trajectory.items()}}
        print(json.dumps(summary, indent=2))
        return
    blender = find_blender(args.blender)
    if not blender:
        raise SystemExit(
            "Blender was not found. Install Blender 4.x or pass --blender /path/to/blender. "
            "Use --dry-run to validate without it."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.blend:
        args.blend.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(spec, handle)
        spec_path = Path(handle.name)
    script = Path(__file__).with_name("scene.py")
    command = [blender]
    if not args.open_ui:
        command.append("--background")
    command += ["--python", str(script), "--", str(spec_path)]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
    finally:
        spec_path.unlink(missing_ok=True)
