"""Emit racing_simple_<track>.xml for each track in RACING_TRACKS.

Re-run after editing the track list in crazyflie_racing_simple_env.py.

    python -m racing.envs.mjx.generate_racing_scene            # all tracks
    python -m racing.envs.mjx.generate_racing_scene straight   # one track
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from racing.envs.mjx.crazyflie_racing_simple_env import RACING_TRACKS


HEADER = """<mujoco model="CF2 Racing simple - {track}">
  <include file="cf2_asset.xml" />

  <statistic center="0 0 1" extent="5.0" meansize=".1" />

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
    <rgba haze="0.15 0.25 0.35 1" />
    <global azimuth="-20" elevation="-20" ellipsoidinertia="true" offwidth="1080" offheight="720" />
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
      rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2" />
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true" />
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"
          contype="0" conaffinity="0" />

    <body name="cf_body" pos="0 0 1" childclass="cf2">
      <freejoint />
      <inertial pos="0 0 0" mass="0.0282" diaginertia="1.657199937e-5 1.665600030e-5 2.926200250e-5" />
      <camera name="cf_track" pos="-1 0 .5" xyaxes="0 -1 0 1 0 2" mode="trackcom" />

      <geom mesh="cf2_0" material="blue_propeller_plastic" class="visual" />
      <geom mesh="cf2_1" material="medium_gloss_plastic"   class="visual" />
      <geom mesh="cf2_2" material="polished_gold"          class="visual" />
      <geom mesh="cf2_3" material="polished_plastic"       class="visual" />
      <geom mesh="cf2_4" material="burnished_chrome"       class="visual" />
      <geom mesh="cf2_5" material="body_frame_plastic"     class="visual" />
      <geom mesh="cf2_6" material="blue"                   class="visual" />

      <geom mesh="cf2_collision_0"  class="visual" />
      <geom mesh="cf2_collision_1"  class="visual" />
      <geom mesh="cf2_collision_2"  class="visual" />
      <geom mesh="cf2_collision_3"  class="visual" />
      <geom mesh="cf2_collision_4"  class="visual" />
      <geom mesh="cf2_collision_5"  class="visual" />
      <geom mesh="cf2_collision_6"  class="visual" />
      <geom mesh="cf2_collision_7"  class="visual" />
      <geom mesh="cf2_collision_8"  class="visual" />
      <geom mesh="cf2_collision_9"  class="visual" />
      <geom mesh="cf2_collision_10" class="visual" />
      <geom mesh="cf2_collision_11" class="visual" />
      <geom mesh="cf2_collision_12" class="visual" />
      <geom mesh="cf2_collision_13" class="visual" />
      <geom mesh="cf2_collision_14" class="visual" />
      <geom mesh="cf2_collision_15" class="visual" />
      <geom mesh="cf2_collision_16" class="visual" />
      <geom mesh="cf2_collision_17" class="visual" />
      <geom mesh="cf2_collision_18" class="visual" />
      <geom mesh="cf2_collision_19" class="visual" />
      <geom mesh="cf2_collision_20" class="visual" />
      <geom mesh="cf2_collision_21" class="visual" />
      <geom mesh="cf2_collision_22" class="visual" />
      <geom mesh="cf2_collision_23" class="visual" />
      <geom mesh="cf2_collision_24" class="visual" />
      <geom mesh="cf2_collision_25" class="visual" />
      <geom mesh="cf2_collision_26" class="visual" />
      <geom mesh="cf2_collision_27" class="visual" />
      <geom mesh="cf2_collision_28" class="visual" />
      <geom mesh="cf2_collision_29" class="visual" />
      <geom mesh="cf2_collision_30" class="visual" />
      <geom mesh="cf2_collision_31" class="visual" />

      <!-- Sphere kept for visualisation only; contacts are disabled
           (contype=0, conaffinity=0) and collision is checked analytically
           in _check_gate_collisions for speed. -->
      <geom name="cf_col" type="sphere" size="0.06" rgba="0 0 1 0"
            contype="0" conaffinity="0" class="collision" />

      <site name="cf_imu"       group="5" />
      <site name="cf_actuation" group="5" />
    </body>

    <body name="racingtarget_body" mocap="true">
      <geom name="racingtarget" type="sphere" size="0.05"
            rgba="0 1 0 0.5" contype="0" conaffinity="0" />
    </body>

"""

FOOTER = """
    <camera name="overview" mode="fixed" pos="5.0 0.0 4.0" xyaxes="0 1 0 -1 0 1.2" fovy="60"/>
  </worldbody>

  <sensor>
    <!-- Contact sensor removed — collision is now checked analytically
         via _check_gate_collisions (faster, no MJX contact overhead). -->
    <gyro          name="cf_body_gyro"    site="cf_imu" />
    <accelerometer name="cf_body_linacc"  site="cf_imu" />
    <framequat     name="cf_body_quat"    objtype="site" objname="cf_imu" />
  </sensor>

  <actuator>
    <motor class="cf2" ctrlrange="0 0.35"  gear="0 0 1 0 0 0"       site="cf_actuation" name="cf_body_thrust" />
    <motor class="cf2" ctrlrange="-1 1"    gear="0 0 0 -0.00001 0 0" site="cf_actuation" name="cf_x_moment" />
    <motor class="cf2" ctrlrange="-1 1"    gear="0 0 0 0 -0.00001 0" site="cf_actuation" name="cf_y_moment" />
    <motor class="cf2" ctrlrange="-1 1"    gear="0 0 0 0 0 -0.00001" site="cf_actuation" name="cf_z_moment" />
  </actuator>
</mujoco>
"""

GATE_TEMPLATE = """    <body name="gate_{idx}" pos="{x} {y} {z}" euler="{roll_deg:.4f} {pitch_deg:.4f} {yaw_deg:.4f}">
      <!-- Bars are visual-only (contype=0, conaffinity=0); collision is
           checked analytically in _check_gate_collisions for speed. -->
      <geom type="box" size="0.05 0.5 0.05" pos="0 0 0.5"  rgba="1 0.5 0 1" contype="0" conaffinity="0"/>
      <geom type="box" size="0.05 0.5 0.05" pos="0 0 -0.5" rgba="1 0.5 0 1" contype="0" conaffinity="0"/>
      <geom type="box" size="0.05 0.05 0.5" pos="0 0.5 0"  rgba="1 0.5 0 1" contype="0" conaffinity="0"/>
      <geom type="box" size="0.05 0.05 0.5" pos="0 -0.5 0" rgba="1 0.5 0 1" contype="0" conaffinity="0"/>
    </body>
"""


def build_scene_xml(track: str, gates: list[list[float]]) -> str:
    body = HEADER.format(track=track)
    for i, (x, y, z, roll, pitch, yaw) in enumerate(gates):
        body += GATE_TEMPLATE.format(
            idx=i, x=x, y=y, z=z,
            roll_deg=math.degrees(roll),
            pitch_deg=math.degrees(pitch),
            yaw_deg=math.degrees(yaw),
        )
    body += FOOTER
    return body


def main(argv: list[str]) -> None:
    out_dir = Path(__file__).resolve().parent
    tracks = argv[1:] if len(argv) > 1 else list(RACING_TRACKS.keys())

    for name in tracks:
        if name not in RACING_TRACKS:
            print(f"[skip] unknown track: {name}")
            continue
        xml = build_scene_xml(name, RACING_TRACKS[name])
        path = out_dir / f"racing_simple_{name}.xml"
        path.write_text(xml)
        print(f"[wrote] {path}  ({len(RACING_TRACKS[name])} gates)")


if __name__ == "__main__":
    main(sys.argv)
