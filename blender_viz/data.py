"""Rollout and MJCF loaders. This module deliberately has no Blender dependency."""

from __future__ import annotations

import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _float_row(row: dict[str, str], names: tuple[str, ...]) -> list[float] | None:
    if not all(name in row and row[name] != "" for name in names):
        return None
    return [float(row[name]) for name in names]


def load_trajectory(path: Path) -> dict[str, Any]:
    """Load CSV, JSON, NPY, or NPZ into a canonical trajectory dictionary.

    Positions are ``N x 3``. Optional quaternions use Blender's/MuJoCo's
    ``[w, x, y, z]`` convention. CSV accepts x/y/z, px/py/pz, or pos_x/y/z.
    NPZ looks for position/positions/pos/qpos and quaternion/orientation/quat.
    """
    path = path.expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"trajectory is empty: {path}")
        positions, quaternions, times, speeds = [], [], [], []
        for row in rows:
            pos = (
                _float_row(row, ("x", "y", "z"))
                or _float_row(row, ("px", "py", "pz"))
                or _float_row(row, ("pos_x", "pos_y", "pos_z"))
            )
            if pos is None:
                raise ValueError("CSV needs x,y,z (or px,py,pz / pos_x,pos_y,pos_z) columns")
            positions.append(pos)
            quat = _float_row(row, ("qw", "qx", "qy", "qz")) or _float_row(
                row, ("quat_w", "quat_x", "quat_y", "quat_z")
            )
            if quat is not None:
                quaternions.append(quat)
            if row.get("time", row.get("t", "")) != "":
                times.append(float(row.get("time", row.get("t", "0"))))
            if row.get("speed", "") != "":
                speeds.append(float(row["speed"]))
        result: dict[str, Any] = {"positions": positions}
        if len(quaternions) == len(positions):
            result["quaternions"] = quaternions
        if len(times) == len(positions):
            result["times"] = times
        if len(speeds) == len(positions):
            result["speeds"] = speeds
        return validate_trajectory(result)

    if suffix == ".json":
        raw = json.loads(path.read_text())
        if isinstance(raw, list):
            raw = {"positions": raw}
        return validate_trajectory(raw)

    if suffix in {".npy", ".npz"}:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("NPY/NPZ input requires: pip install 'drone-racing-viz[numpy]'") from exc
        raw = np.load(path, allow_pickle=False)
        if suffix == ".npy":
            array = raw
            result = {"positions": array[:, :3].tolist()}
            if array.ndim == 2 and array.shape[1] >= 7:
                result["quaternions"] = array[:, 3:7].tolist()
        else:
            keys = set(raw.files)
            pos_key = next((k for k in ("positions", "position", "pos", "qpos") if k in keys), None)
            if pos_key is None:
                raise ValueError(f"NPZ has no position array; found {sorted(keys)}")
            arr = raw[pos_key]
            result = {"positions": arr[:, :3].tolist()}
            quat_key = next((k for k in ("quaternions", "quaternion", "orientation", "quat") if k in keys), None)
            if quat_key:
                result["quaternions"] = raw[quat_key].tolist()
            elif pos_key == "qpos" and arr.shape[1] >= 7:
                result["quaternions"] = arr[:, 3:7].tolist()
            for source, target in (("time", "times"), ("times", "times"), ("speed", "speeds"), ("speeds", "speeds")):
                if source in keys:
                    result[target] = raw[source].tolist()
                    break
            raw.close()
        return validate_trajectory(result)
    raise ValueError(f"unsupported trajectory format '{suffix}'; use CSV, JSON, NPY, or NPZ")


def validate_trajectory(data: dict[str, Any]) -> dict[str, Any]:
    positions = data.get("positions")
    if not isinstance(positions, list) or len(positions) < 2:
        raise ValueError("trajectory needs at least two positions")
    clean = [[float(v) for v in point[:3]] for point in positions]
    if any(len(point) != 3 or not all(math.isfinite(v) for v in point) for point in clean):
        raise ValueError("every position must contain three finite values")
    result: dict[str, Any] = {"positions": clean}
    for key in ("quaternions", "times", "speeds"):
        if key in data:
            values = data[key]
            if len(values) != len(clean):
                raise ValueError(f"{key} length must match positions")
            result[key] = values
    return result


def load_gates(path: Path) -> list[dict[str, Any]]:
    """Extract gate transforms from a generated racing MJCF file."""
    root = ET.parse(path).getroot()
    gates = []
    for body in root.findall(".//body"):
        if not body.get("name", "").startswith("gate_"):
            continue
        pos = [float(v) for v in body.get("pos", "0 0 0").split()]
        euler_deg = [float(v) for v in body.get("euler", "0 0 0").split()]
        gates.append({"name": body.get("name"), "position": pos, "rotation": [math.radians(v) for v in euler_deg]})
    if not gates:
        raise ValueError(f"no gate_* bodies found in {path}")
    return gates


def demo_trajectory(gates: list[dict[str, Any]], samples_per_leg: int = 28) -> dict[str, Any]:
    """Create a smooth closed demonstration route through all gate centers."""
    points = [gate["position"] for gate in gates]
    if len(points) == 1:
        p = points[0]
        points = [[p[0], p[1] - 2, p[2]], p, [p[0], p[1] + 2, p[2]]]
    closed = len(points) > 2
    result = []
    count = len(points) if closed else len(points) - 1
    for i in range(count):
        p0 = points[(i - 1) % len(points)] if closed else points[max(0, i - 1)]
        p1, p2 = points[i], points[(i + 1) % len(points)]
        p3 = points[(i + 2) % len(points)] if closed else points[min(len(points) - 1, i + 2)]
        for j in range(samples_per_leg):
            t = j / samples_per_leg
            t2, t3 = t * t, t * t * t
            result.append(
                [
                    0.5
                    * (
                        (2 * p1[k])
                        + (-p0[k] + p2[k]) * t
                        + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2
                        + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3
                    )
                    for k in range(3)
                ]
            )
    result.append(points[0] if closed else points[-1])
    return {"positions": result}
