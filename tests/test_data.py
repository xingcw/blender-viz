import json
import unittest
from pathlib import Path

from blender_viz.cli import find_blender
from blender_viz.data import demo_trajectory, load_gates, load_trajectory

ROOT = Path(__file__).parents[1]


class DataTests(unittest.TestCase):
    def test_explicit_missing_blender_does_not_fallback(self):
        self.assertIsNone(find_blender("/definitely/missing/blender"))

    def test_load_repo_track(self):
        gates = load_gates(ROOT / "envs/mjx/racing_simple_lemniscate.xml")
        self.assertEqual(len(gates), 6)
        self.assertEqual(gates[0]["position"], [1.5, 3.5, 0.75])

    def test_demo_passes_every_gate_center(self):
        gates = load_gates(ROOT / "envs/mjx/racing_simple_circle4.xml")
        trajectory = demo_trajectory(gates, samples_per_leg=10)
        self.assertEqual(len(trajectory["positions"]), 41)
        for i, gate in enumerate(gates):
            for actual, expected in zip(trajectory["positions"][i * 10], gate["position"]):
                self.assertAlmostEqual(actual, expected)

    def test_csv_aliases_and_quaternion(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.csv"
            path.write_text("px,py,pz,qw,qx,qy,qz,t\n0,1,2,1,0,0,0,0\n1,2,3,1,0,0,0,.1\n")
            result = load_trajectory(path)
        self.assertEqual(result["positions"][-1], [1.0, 2.0, 3.0])
        self.assertEqual(result["quaternions"][0], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(result["times"], [0.0, 0.1])

    def test_json_list(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.json"
            path.write_text(json.dumps([[0, 0, 1], [1, 0, 1]]))
            self.assertEqual(load_trajectory(path)["positions"][1][0], 1.0)
