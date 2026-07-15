"""
Crazyflie gate-racing environment — soft-collision variant.

Differences from crazyflie_racing_env.py (the "full" variant):
  - Drone collisions go through a single sphere proxy (`cf_col`, radius
    `drone_radius`) instead of the 32 cf2_collision_X meshes — much faster
    in MJX while still giving real, force-based contact.
  - Gate bars are collision-active (default contype=1/conaffinity=1) so
    cf_col contacts them. Floor + racingtarget marker remain visual-only.
  - Contact is checked analytically via `_check_gate_collisions` (sphere vs
    gate bars) — MJX contacts are fully disabled for speed:
      * a per-step `crash_scale * in_contact` penalty fires while in contact,
      * `state.info["contact_steps"]` accumulates contact frames (with a
        `contact_grace_steps` spawn mask), and the drone "dies" once that
        cumulative count exceeds `num_contact_steps` — mirrors IsaacLab's
        `self._crashed > 100`. Death replaces the reward with `death_cost`.
  - Floor death stays analytical: `z < low_z_threshold AND step >= self.num_lowz_steps`
    where `num_lowz_steps = ceil(low_z_grace_s / step_dt)` is derived in
    __init__ (no contact dynamics on the floor).
  - Scene XML files are pre-generated (racing_simple_<track>.xml); regenerate
    via `python -m racing.envs.mjx.generate_racing_scene` after editing
    RACING_TRACKS.
"""

import math
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, List, NamedTuple, Tuple

import mujoco
from mujoco import mjx

from brax.envs.base import PipelineEnv, State
from brax.base import Motion, System, Transform
from brax.io import mjcf
from etils import epath

import chex

from racing.envs.mjx.motor_dynamics import MotorDynamics, RandomizedDynamicsParams


# Multi-host TPU shim: brax.io.mjcf.load_model -> mjx.put_model(mj) passes
# device=None, which falls through to jax.devices()[0] (TPU_0, owned by
# process 0). Non-zero workers cannot address it and crash with
# INVALID_ARGUMENT before training begins. Default to jax.local_devices()[0]
# instead; explicit device args still win.
_orig_mjx_put_model = mjx.put_model
def _mjx_put_model_local_default(m, device=None, **kwargs):
    if device is None:
        device = jax.local_devices()[0]
    return _orig_mjx_put_model(m, device=device, **kwargs)
mjx.put_model = _mjx_put_model_local_default

SCENE_XML_PATH = epath.Path(epath.resource_path("racing")) / "envs/mjx/"

# ── Track definitions (x, y, z, roll, pitch, yaw in radians) ──────────────────
# For debug tracks, the drone spawns at gate_pos + gate_R @ (x_local, y_local, z_local)
# with x_local ∈ [0.5, 2.0] > 0. Gate local-X is the *approach* axis, so a yaw
# that points the gate's +X toward the drone's start region makes the drone
# spawn behind the gate and need to fly through in the -X direction of the gate.
RACING_TRACKS: Dict[str, List[List[float]]] = {
    # Single gate 3 m ahead along +y; gate faces -y (yaw = -π/2) so drone spawns
    # behind it and flies forward.
    'straight': [
        [ 0.0, 3.00, 1.00, 0, 0, -1.5708],
    ],
    # Two gates in a straight line along +y, both facing -y.
    'straight2': [
        [ 0.0, 3.00, 1.00, 0, 0, -1.5708],
        [ 0.0, 6.00, 1.00, 0, 0, -1.5708],
    ],
    # 4 gates evenly spaced on a CCW circle of radius 2.5m at z=1.0m.
    # Each gate's local +x axis (the approach side) points opposite the CCW
    # tangent, so the drone always flies forward through the next gate.
    # yaw_i = theta_i - pi/2 with theta_i = 2*pi*i/4.
    'circle4': [
        [ 2.5,  0.0, 1.00, 0, 0, -1.5708],   # theta=0
        [ 0.0,  2.5, 1.00, 0, 0,  0.0000],   # theta=pi/2
        [-2.5,  0.0, 1.00, 0, 0,  1.5708],   # theta=pi
        [ 0.0, -2.5, 1.00, 0, 0,  3.1416],   # theta=3pi/2
    ],
    'lemniscate': [
        [ 1.5, 3.50, 0.75, 0, 0, -1.57],
        [ 0.0, 5.25, 1.50, 0, 0,  0.00],
        [-2.0, 7.00, 0.75, 0, 0, -1.57],
        [ 1.5, 7.00, 0.75, 0, 0,  1.57],
        [ 0.0, 5.25, 1.50, 0, 0,  0.00],
        [-2.0, 3.50, 0.75, 0, 0,  1.57],
    ],
    'complex': [
        [ 1.5,  3.5, 0.75, 0, 0, -0.7854],
        [-1.5,  3.5, 0.75, 0, 0,  0.7854],
        [-2.0, -3.5, 2.00, 0, 0,  1.5708],
        [-2.0, -3.5, 0.75, 0, 0, -1.5708],
        [ 1.0, -1.0, 2.00, 0, 0,  3.1415],
        [ 1.0, -3.5, 0.75, 0, 0,  0.0000],
    ],
    'powerloop': [
        [ 2.0,   3.5, 0.75, 0, 0, -1.5708],
        [-1.5,   3.5, 2.00, 0, 0,  0.7854],
        [-0.625, 0.0, 0.75, 0, 0,  1.5708],
        [ 0.625, 0.0, 0.75, 0, 0,  1.5708],
        [-1.5,  -3.5, 2.00, 0, 0,  2.3560],
        [ 2.0,  -3.5, 0.75, 0, 0, -1.5708],
        [ 0.625, 0.0, 0.75, 0, 0, -1.5708],
    ],
}


# ── Config NamedTuples ─────────────────────────────────────────────────────────

class WorldConfig(NamedTuple):
    # motor dynamics
    arm_length: float = 0.043
    k_eta: float = 2.3e-08
    k_m: float = 7.8e-10
    tau_m: float = 0.005
    motor_speed_min: float = 0.0
    motor_speed_max: float = 2500.0
    hover_motor_speed: float = 1700.0
    thrust_to_weight: float = 3.15
    k_aero_xy: float = 9.1785e-7
    k_aero_z: float = 10.311e-7
    k_aero: chex.Array = jnp.array([k_aero_xy, k_aero_xy, k_aero_z])
    kp_omega_rp: float = 250.0
    ki_omega_rp: float = 500.0
    kd_omega_rp: float = 2.5
    i_limit_rp: float = 33.3
    kp_omega_y: float = 120.0
    ki_omega_y: float = 16.70
    kd_omega_y: float = 0.0
    i_limit_y: float = 166.7
    body_rate_scale_xy: float = 100 * np.pi / 180.0
    body_rate_scale_z: float = 200 * np.pi / 180.0
    kp_omega = jnp.array([kp_omega_rp, kp_omega_rp, kp_omega_y])
    ki_omega = jnp.array([ki_omega_rp, ki_omega_rp, ki_omega_y])
    kd_omega = jnp.array([kd_omega_rp, kd_omega_rp, kd_omega_y])

    # Episode timing — defined in seconds (matches IsaacLab convention) and
    # converted to step counts at __init__ time using the actual step_dt
    # (= timestep * n_frames). Resilient to n_frames / timestep changes.
    episode_length_s: float = 30.0   # matches IsaacLab episode_length_s
    low_z_grace_s:    float = 1.5    # matches IsaacLab max_time_on_ground
    low_z_threshold:  float = 0.1    # matches IsaacLab min_altitude
    max_altitude:     float = 3.0    # matches IsaacLab max_altitude; instant death on crossing
    timestep:         float = 0.002

    # Initial-state randomization (training only; eval starts at rest with
    # zero tilt for deterministic scoring). Yaw is always noised toward the
    # current gate in reset().
    init_x_local_min: float = 0.5    # m in front of gate; uniform [min, max]
    init_x_local_max: float = 2.0
    init_y_local_max: float = 0.5    # lateral offset; uniform [-max, max]
    init_z_local_max: float = 0.5    # vertical  offset; uniform [-max, max]
    init_lin_vel_max: float = 1.0    # world-frame lin vel ~ U[-x, x] per axis [m/s]
    init_ang_vel_max: float = 1.0    # body-frame ang vel ~ U[-x, x] per axis [rad/s]
    init_tilt_max:    float = 0.2    # roll/pitch ~ U[-x, x] [rad] (~11°)

    # Domain randomization scale bounds (aligned with quadrotor_racing_env.py)
    k_aero_scale_min: float = 0.5
    k_aero_scale_max: float = 2.0
    kp_omega_rp_scale_min: float = 0.85
    kp_omega_rp_scale_max: float = 1.15
    ki_omega_rp_scale_min: float = 0.85
    ki_omega_rp_scale_max: float = 1.15
    kd_omega_rp_scale_min: float = 0.7
    kd_omega_rp_scale_max: float = 1.2
    kp_omega_y_scale_min: float = 0.85
    kp_omega_y_scale_max: float = 1.15
    ki_omega_y_scale_min: float = 0.85
    ki_omega_y_scale_max: float = 1.15
    kd_omega_y_scale_min: float = 0.7
    kd_omega_y_scale_max: float = 1.2
    tau_m_scale_min: float = 0.8
    tau_m_scale_max: float = 1.2
    t2w_scale_min: float = 0.8
    t2w_scale_max: float = 1.2

    # Rotor failure (per-rotor independent power loss). Specialists pin a
    # 4-vector of losses (one per rotor); the DR baseline samples each
    # rotor's loss independently from [loss_min, loss_max] per episode.
    # loss == 0 -> healthy rotor, loss == 1 -> fully dead rotor; the env
    # applies rotor_efficiency = 1 - loss to each rotor's thrust.
    rotor_failure_dr_enabled: bool = False
    rotor_failure_loss_min: float = 0.0
    rotor_failure_loss_max: float = 0.0        # max==min collapses DR draw to a constant
    rotor_failure_losses: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    use_motor_dynamics: bool = True


class RacingConfig(NamedTuple):
    track_name: str = 'lemniscate'
    gate_side: float = 1.0
    gate_bar_thickness: float = 0.05
    # Half-side margin for the strict gate-pass check.
    # 0.0 matches IsaacLab's reward `gate_passed` (|y|,|z| < gate_side/2, no margin).
    gate_margin: float = 0.0
    # radius used for analytical gate-bar collision
    drone_radius: float = 0.06
    max_laps: int = 3
    is_train: bool = True
    domain_randomization: bool = True
    progress_scale: float = 50.0
    gate_bonus: float = 0.0
    rp_rate_scale: float = 0.1         # multiplied by step_dt inside get_rewards
    yaw_rate_scale: float = 0.1        # multiplied by step_dt inside get_rewards
    lap_bonus: float = 0.0             # paid once on the gate-pass that starts a new lap
    perception_scale: float = 0.0      # exp(-4 dyaw^4) toward current gate; * step_dt
    max_tilt_thresh: float = 2.6179939  # 150 deg in rad -- penalty kicks in past this
    max_tilt_scale: float = 0.0        # tilt penalty magnitude; * step_dt
    crash_scale: float = -1.0          # per-step penalty while in contact (no step_dt; matches IsaacLab dense)
    death_cost: float = -10.0          # replaces total reward when the drone "dies"
    # Soft-collision parameters
    num_contact_steps: int = 100
    contact_grace_steps: int = 100     # matches IsaacLab's `episode_length_buf > 100` mask on the death counter


# ── Gate geometry helpers ──────────────────────────────────────────────────────

def _build_gate_arrays(
    track_data: List[List[float]],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build pre-computed gate position/rotation arrays from track data."""
    from scipy.spatial.transform import Rotation

    positions, rotmats, normals = [], [], []
    for gate in track_data:
        x, y, z, roll, pitch, yaw = gate
        pos = np.array([x, y, z], dtype=np.float32)
        # Extrinsic xyz (uppercase) — matches the quaternion built in reset()
        # from (roll, pitch, yaw). Lowercase 'xyz' would be intrinsic, which
        # disagrees for non-yaw-only Euler tuples.
        R = Rotation.from_euler('XYZ', [roll, pitch, yaw]).as_matrix().astype(np.float32)
        positions.append(pos)
        rotmats.append(R)
        normals.append(R[:, 0])  # gate local x-axis = approach direction

    return (
        jnp.array(np.stack(positions)),  # (N, 3)
        jnp.array(np.stack(rotmats)),    # (N, 3, 3)
        jnp.array(np.stack(normals)),    # (N, 3)
    )


# ── Environment ────────────────────────────────────────────────────────────────

class CrazyflieRacingSimple(PipelineEnv):
    """
    JAX/MJX Crazyflie gate-racing environment with analytical collision.

    All MJX contacts are disabled for speed. Collision is detected
    analytically via `_check_gate_collisions` (sphere vs gate-bar geometry),
    OR-ed across all physics substeps to catch glancing hits.
    Termination conditions (env emits ``done`` for each — episode time-out is
    handled by Brax's ``EpisodeWrapper``, not here):
      - Floor:    qpos[2] < low_z_threshold AND step >= num_lowz_steps
                  (num_lowz_steps = ceil(cfg.low_z_grace_s / step_dt))
      - Contact:  cumulative contact frames > num_contact_steps (after grace)
      - Z-high:   qpos[2] > max_altitude
      - Below 0:  qpos[2] < 0 (instant; floor is non-colliding)
      - Lap done: only at eval (n_gates_passed >= max_laps * num_gates)

    Observation (33 dims):
        drone_pos_w          (3)
        drone_lin_vel_w      (3)
        drone_R_flat         (9)
        curr_gate_pos_w      (3)
        curr_gate_R[:2,:2]   (4)   horizontal yaw block only
        next_gate_pos_w      (3)
        next_gate_R[:2,:2]   (4)
        prev_action          (4)
        Total: 33.

    Action (4 dims, CTBR):
        [collective_thrust, roll_rate, pitch_rate, yaw_rate] ∈ [-1, 1]
    """

    def __init__(
        self,
        world_cfg: WorldConfig = WorldConfig(),
        racing_cfg: RacingConfig = RacingConfig(),
        **kwargs,
    ):
        self.cfg = world_cfg
        self.racing_cfg = racing_cfg

        track_name = racing_cfg.track_name
        if track_name not in RACING_TRACKS:
            raise ValueError(
                f"Unknown track '{track_name}'. "
                f"Available: {list(RACING_TRACKS.keys())}"
            )
        track_data = RACING_TRACKS[track_name]
        self.num_gates = len(track_data)

        self.gate_positions, self.gate_rotmats, self.gate_normals = _build_gate_arrays(track_data)

        # Load pre-generated scene XML (racing_simple_<track>.xml)
        xml_path = (SCENE_XML_PATH / f"racing_simple_{track_name}.xml").as_posix()
        mj_model = mujoco.MjModel.from_xml_path(xml_path)

        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        # With all contacts disabled (analytical collision only), the CG
        # solver has nothing to solve — 1 iteration is sufficient.
        mj_model.opt.iterations = 1
        mj_model.opt.ls_iterations = 1
        mj_model.opt.timestep = self.cfg.timestep

        sys = mjcf.load_model(mj_model)

        kwargs["n_frames"] = kwargs.get("n_frames", 10)
        kwargs["backend"]  = "mjx"
        self._n_frames_env = int(kwargs["n_frames"])
        self._step_dt = float(self.cfg.timestep) * self._n_frames_env

        self.max_episode_length = int(
            math.ceil(self.cfg.episode_length_s / self._step_dt)
        )
        self.num_lowz_steps = int(
            math.ceil(self.cfg.low_z_grace_s / self._step_dt)
        )

        self.body_com_id = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "cf_body"
        )

        super().__init__(sys, **kwargs)

        self.motor_dynamics = MotorDynamics(world_cfg, mj_model, [self.body_com_id])

        # Four corners of a gate, in the gate-local frame (gate +x is the
        # approach axis, so all corners lie at x=0 on the gate plane).
        # Used by `_get_obs` to encode each gate as 4 body-frame vertex
        # positions (matches IsaacLab's `_local_square`,
        # `quadcopter_env.py:742-747`).
        d = self.racing_cfg.gate_side / 2.0
        self._local_square = jnp.array([
            [0.0,  d,  d],
            [0.0, -d,  d],
            [0.0, -d, -d],
            [0.0,  d, -d],
        ], dtype=jnp.float32)

    @property
    def observation_size(self) -> int:
        # drone_pos_w(3) + drone_lin_vel_w(3) + R_flat(9)
        # + curr_gate_pos_w(3) + curr_gate_R[:2,:2]_flat(4)
        # + next_gate_pos_w(3) + next_gate_R[:2,:2]_flat(4) + prev_action(4) = 33
        return 3 + 3 + 9 + 3 + 4 + 3 + 4 + 4

    # ── Analytical gate-bar collision ──────────────────────────────────────────

    @partial(jax.jit, static_argnums=0)
    def _check_gate_collisions(self, pos: jnp.ndarray) -> jnp.ndarray:
        """
        Returns True if the drone sphere intersects any gate bar.

        The check is done in each gate's local frame:
          - near_plane: drone is within (drone_radius + bar_thickness) of the gate plane
          - in_bbox:    drone centre is within the gate's outer bounding square (+drone_radius)
          - outside:    drone centre is outside the gate opening minus bar_thickness
                        (i.e. it's in the bar region, not the open hole)
        """
        r    = self.racing_cfg.drone_radius
        bt   = self.racing_cfg.gate_bar_thickness
        half = self.racing_cfg.gate_side / 2.0

        def _hit_one_gate(gate_pos, gate_R):
            p = gate_R.T @ (pos - gate_pos)        # gate-local coords
            near_plane = jnp.abs(p[0]) < (r + bt)
            in_bbox    = (jnp.abs(p[1]) < half + r) & (jnp.abs(p[2]) < half + r)
            # "outside" means in the bar region (not through the opening)
            outside    = (jnp.abs(p[1]) > half - bt) | (jnp.abs(p[2]) > half - bt)
            return near_plane & in_bbox & outside

        hits = jax.vmap(_hit_one_gate)(self.gate_positions, self.gate_rotmats)
        return jnp.any(hits)

    # ── Gate passing detection ─────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=0)
    def _update_gate_target(
        self,
        pos: jnp.ndarray,
        prev_x_wrt_gate: jnp.ndarray,
        current_idx: jnp.ndarray,
        next_idx: jnp.ndarray,
    ) -> Tuple:
        gate_pos = jnp.take(self.gate_positions, current_idx, axis=0)
        gate_R   = jnp.take(self.gate_rotmats,   current_idx, axis=0)

        pos_local = gate_R.T @ (pos - gate_pos)
        x         = pos_local[0]
        # Strict pass window: the drone centre must be within
        # `half - gate_margin` of the gate axis on both in-plane axes.
        # Mirrors IsaacLab's `cond_gate_inside` (quadcopter_env.py:1057).
        # The default margin (0.1) excludes the bar region, so a drone
        # passing through the bar itself doesn't count as a pass.
        inside_half = self.racing_cfg.gate_side / 2.0 - self.racing_cfg.gate_margin
        inside_y  = jnp.abs(pos_local[1]) < inside_half
        inside_z  = jnp.abs(pos_local[2]) < inside_half
        # Plane crossing in the +x → −x direction (drone enters the gate
        # from the approach side and emerges on the back side).
        crossed   = (x < 0.0) & (prev_x_wrt_gate > 0.0)
        # A pass requires: actually pierced the plane, did so through the
        # inner opening, *and* the drone centre is within 1.0 m of the gate
        # centre. Bypassing the gate (crossing the plane outside the opening)
        # leaves `passed=False` so the target gate does NOT advance — the
        # drone has to come back around to the approach side and try again.
        # This prevents bypass exploits. The `< 1.0` distance gate mirrors
        # IsaacLab's `dist_to_gate < 1.0` in `_get_rewards`.
        dist_to_gate = jnp.linalg.norm(pos_local)
        passed = crossed & inside_y & inside_z & (dist_to_gate < 1.0)
        gate_passed = jnp.asarray(passed, dtype=jnp.float32)

        new_current = jax.lax.select(passed, next_idx, current_idx)
        new_next    = jax.lax.select(
            passed, jnp.mod(next_idx + 1, self.num_gates), next_idx
        )

        new_current_pos = jnp.take(self.gate_positions, new_current, axis=0)
        new_current_R   = jnp.take(self.gate_rotmats,   new_current, axis=0)
        new_next_pos    = jnp.take(self.gate_positions,  new_next,    axis=0)
        new_next_R      = jnp.take(self.gate_rotmats,    new_next,    axis=0)

        return (
            new_current, new_next,
            new_current_pos, new_current_R,
            new_next_pos,    new_next_R,
            gate_passed, x,
        )

    # ── Observation ────────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=0)
    def _get_obs(
        self,
        data: mjx.Data,
        current_gate_pos: jnp.ndarray,
        current_gate_R:   jnp.ndarray,
        next_gate_pos:    jnp.ndarray,
        next_gate_R:      jnp.ndarray,
        prev_action:      jnp.ndarray,
    ) -> jnp.ndarray:
        """33-dim obs: drone pose + velocity + (current, next) gate pose +
        previous action. Gate rotation is the 2x2 horizontal yaw block —
        the dropped row/col entries are constant for upright gates on a
        flat track and add no information.
        """
        pos = data.qpos[:3]
        R   = data.xmat[self.body_com_id]
        lin_vel_w = data.qvel[:3]

        return jnp.concatenate([
            pos,                                    # drone position (world)        (3)
            lin_vel_w,                              # drone linear velocity (world) (3)
            R.flatten(),                            # drone rotation matrix         (9)
            current_gate_pos,                       # current gate position (world) (3)
            current_gate_R[:2, :2].flatten(),       # current gate yaw block        (4)
            next_gate_pos,                          # next gate position (world)    (3)
            next_gate_R[:2, :2].flatten(),          # next gate yaw block           (4)
            prev_action,                            # last applied action           (4)
        ])

    # ── Reward ─────────────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=0)
    def get_rewards(
        self,
        data: mjx.Data,
        action: jnp.ndarray,
        current_gate_pos: jnp.ndarray,
        gate_passed: jnp.ndarray,
        lap_completed: jnp.ndarray,
        prev_dist: jnp.ndarray,
        in_contact: jnp.ndarray,
        died: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, dict]:
        """Reward shaping — mirrors quadrotor_racing_env._get_rewards.

        The crash term is a per-step fee (no step_dt multiplier) that fires
        whenever the contact sensor is active — same as IsaacLab. Death
        replaces the *entire* reward with `death_cost`, so the crash fee
        only contributes during the buildup before death is triggered.
        """
        pos  = data.qpos[:3]
        R    = data.xmat[self.body_com_id]   # drone rotation matrix (world ← body)
        dist = jnp.linalg.norm(pos - current_gate_pos)
        step_dt = self._step_dt

        progress    = (prev_dist - dist) * self.racing_cfg.progress_scale
        gate_reward = gate_passed   * self.racing_cfg.gate_bonus
        lap_reward  = lap_completed * self.racing_cfg.lap_bonus

        # Action rate penalties — multiplied by step_dt so the magnitude is invariant to physics rate.
        rp_penalty  = -jnp.sum(jnp.square(action[1:3])) * self.racing_cfg.rp_rate_scale  * step_dt
        yaw_penalty = -jnp.square(action[3])            * self.racing_cfg.yaw_rate_scale * step_dt

        # Perception: encourages the drone to face the current target gate.
        yaw_drone   = jnp.arctan2(R[1, 0], R[0, 0])
        to_gate     = current_gate_pos - pos
        yaw_to_gate = jnp.arctan2(to_gate[1], to_gate[0])
        delta_yaw   = jnp.mod(yaw_drone - yaw_to_gate + jnp.pi, 2.0 * jnp.pi) - jnp.pi
        perception  = jnp.exp(-4.0 * delta_yaw ** 4) * self.racing_cfg.perception_scale * step_dt

        # Tilt penalty — fires only past max_tilt_thresh, exponential beyond.
        cos_tilt    = R[2, 2]
        tilt_angle  = jnp.arccos(jnp.clip(cos_tilt, -1.0, 1.0))
        thresh      = self.racing_cfg.max_tilt_thresh
        tilt_arg    = self.racing_cfg.max_tilt_scale * (jnp.exp((tilt_angle - thresh) / thresh) - 1.0)
        tilt_pen    = -jnp.maximum(tilt_arg, 0.0) * step_dt

        in_contact_f = jnp.asarray(in_contact, dtype=jnp.float32)
        crash_pen    = self.racing_cfg.crash_scale * in_contact_f

        live_total = (
            progress + gate_reward + lap_reward
            + rp_penalty + yaw_penalty
            + perception + tilt_pen + crash_pen
        )

        died_f = jnp.asarray(died, dtype=jnp.float32)
        total  = jnp.where(died, jnp.float32(self.racing_cfg.death_cost), live_total)

        components = {
            "reward_progress":    progress,
            "reward_gate_bonus":  gate_reward,
            "reward_lap_bonus":   lap_reward,
            "reward_rp_penalty":  rp_penalty,
            "reward_yaw_penalty": yaw_penalty,
            "reward_perception":  perception,
            "reward_tilt":        tilt_pen,
            "reward_crash":       crash_pen,
            "reward_death":       died_f * jnp.float32(self.racing_cfg.death_cost),
        }
        return total, components

    # ── Reset ──────────────────────────────────────────────────────────────────

    @property
    def _default_proposal(self) -> Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
        RandomizedDynamicsParams, jnp.ndarray,
    ]:
        return (
            jnp.zeros(7, dtype=jnp.float32),                           # qpos:           [pos(3), quat(4)]
            jnp.zeros(6, dtype=jnp.float32),                           # qvel:           [lin_vel(3), ang_vel(3)]
            jnp.full((4,), self.cfg.hover_motor_speed, dtype=jnp.float32),  # motor_speeds
            jnp.int32(0),                                              # wp_idx
            jnp.bool_(False),                                          # use
            self.motor_dynamics.get_nominal_params(),                  # dyn_params (no-op when dyn_use=False)
            jnp.bool_(False),                                          # dyn_use
        )

    @partial(jax.jit, static_argnums=0)
    def reset(self, key: chex.PRNGKey) -> State:
        """Public reset — always uses warm-up randomization (no proposal).

        Auto-reset inside `env.step` calls `_reset_impl` directly with
        proposal fields read from the previous frame's `state.info`.
        """
        d_qpos, d_qvel, d_motor, d_wp, d_use, d_dyn, d_dyn_use = self._default_proposal
        return self._reset_impl(
            key, d_qpos, d_qvel, d_motor, d_wp, d_use, d_dyn, d_dyn_use,
        )

    @partial(jax.jit, static_argnums=0)
    def _reset_impl(
        self,
        key: chex.PRNGKey,
        proposal_qpos:             jnp.ndarray,   # (7,)  [pos(3), quat(4)]
        proposal_qvel:             jnp.ndarray,   # (6,)  [lin_vel(3), ang_vel(3)]
        proposal_motor_speeds:     jnp.ndarray,   # (4,)
        proposal_wp_idx:           jnp.ndarray,   # ()    int32
        use_proposal:              jnp.ndarray,   # ()    bool
        proposal_dynamics_params:  RandomizedDynamicsParams,  # full per-env dyn params
        use_dyn_proposal:          jnp.ndarray,   # ()    bool — gates the dyn-params override
    ) -> State:
        """Reset that merges warm-up randomization with two optional proposals.

        Spawn-state proposal (`use_proposal`):
          When True, `proposal_qpos / qvel / motor_speeds / wp_idx` replace
          the warm-up randomization. Single-shot: the proposal fields in
          `state_info` are reset to zeros after consumption.

        Dynamics-params proposal (`use_dyn_proposal`):
          When True, every leaf of `proposal_dynamics_params` replaces the
          sampled or nominal `dyn_params` for the freshly reset episode.
          Persistent: `state_info["reset_proposal_dynamics_params"]` and
          `state_info["reset_proposal_dyn_use"]` are written through from
          the args, so the override survives every auto-reset.
        """
        k_next, k0, k1, k2, k3, k4, k5, k_tilt, k_lvel, k_avel, k_dr = jax.random.split(key, 11)

        # 10% of TRAINING resets launch from the ground takeoff area; eval ALWAYS
        # does. `reset_launching` is a traced bool, so both spawn variants are
        # computed and selected with jnp.where — a Python `if` on a tracer would
        # raise under jit. Mirrors IsaacLab's percent_ground ground spawn (minus
        # the iteration curriculum gate, which is dead in Isaac anyway).
        reset_launching = jax.random.bernoulli(k0, p=0.1)
        use_ground = reset_launching if self.racing_cfg.is_train else jnp.bool_(True)

        # ── Candidate A: airborne, random gate, in front (normal training) ──
        gate_idx_air = jax.random.randint(k1, (), 0, self.num_gates)
        x_air = jax.random.uniform(
            k2, (), minval=self.cfg.init_x_local_min, maxval=self.cfg.init_x_local_max,
        )
        y_air = jax.random.uniform(
            k3, (), minval=-self.cfg.init_y_local_max, maxval=self.cfg.init_y_local_max,
        )
        z_air = jax.random.uniform(
            k4, (), minval=-self.cfg.init_z_local_max, maxval=self.cfg.init_z_local_max,
        )
        gate_pos_air = jnp.take(self.gate_positions, gate_idx_air, axis=0)
        gate_R_air   = jnp.take(self.gate_rotmats,   gate_idx_air, axis=0)
        spawn_air = gate_pos_air + gate_R_air @ jnp.array([x_air, y_air, z_air])
        to_gate_air = gate_pos_air - spawn_air
        yaw_air = jnp.arctan2(to_gate_air[1], to_gate_air[0]) + jax.random.uniform(
            k5, (), minval=-0.15, maxval=0.15,
        )

        # ── Candidate B: ground takeoff behind gate 0 at z=0.05, at rest ──
        # (= IsaacSim play-mode init = IsaacLab percent_ground ground spawn).
        gate_pos_0 = jnp.take(self.gate_positions, 0, axis=0)
        gate_R_0   = jnp.take(self.gate_rotmats,   0, axis=0)
        x_gnd = jax.random.uniform(k2, (), minval=-3.0, maxval=-0.5)
        y_gnd = jax.random.uniform(k3, (), minval=-1.0, maxval=1.0)
        theta0 = jnp.arctan2(gate_R_0[1, 0], gate_R_0[0, 0])   # gate-0 yaw
        cos0, sin0 = jnp.cos(theta0), jnp.sin(theta0)
        x0 = gate_pos_0[0] - (cos0 * x_gnd - sin0 * y_gnd)
        y0 = gate_pos_0[1] - (sin0 * x_gnd + cos0 * y_gnd)
        spawn_gnd = jnp.array([x0, y0, jnp.float32(0.05)])
        yaw_gnd = jnp.arctan2(-y0, -x0) + jax.random.uniform(
            k5, (), minval=-0.15, maxval=0.15,
        )

        # ── Select ground vs airborne ──
        gate_idx_wu  = jnp.where(use_ground, jnp.int32(0), gate_idx_air)
        spawn_pos_wu = jnp.where(use_ground, spawn_gnd, spawn_air)
        yaw          = jnp.where(use_ground, yaw_gnd, yaw_air)

        # Velocity/tilt: random for airborne training spawns; the ground takeoff
        # (and all eval) starts AT REST with zero tilt — matches IsaacLab and
        # avoids an instant z<0 death from a downward draw at z=0.05.
        if self.racing_cfg.is_train:
            lin_vel_rand = jax.random.uniform(
                k_lvel, (3,),
                minval=-self.cfg.init_lin_vel_max, maxval=self.cfg.init_lin_vel_max,
            )
            ang_vel_rand = jax.random.uniform(
                k_avel, (3,),
                minval=-self.cfg.init_ang_vel_max, maxval=self.cfg.init_ang_vel_max,
            )
            tilt = jax.random.uniform(
                k_tilt, (2,),
                minval=-self.cfg.init_tilt_max, maxval=self.cfg.init_tilt_max,
            )
            lin_vel_wu = jnp.where(use_ground, jnp.zeros(3), lin_vel_rand)
            ang_vel_wu = jnp.where(use_ground, jnp.zeros(3), ang_vel_rand)
            roll  = jnp.where(use_ground, jnp.float32(0.0), tilt[0])
            pitch = jnp.where(use_ground, jnp.float32(0.0), tilt[1])
        else:
            del k_lvel, k_avel, k_tilt
            lin_vel_wu = jnp.zeros(3)
            ang_vel_wu = jnp.zeros(3)
            roll = jnp.float32(0.0)
            pitch = jnp.float32(0.0)

        # Compose wxyz quaternion from (roll, pitch, yaw) — extrinsic xyz Euler,
        # matches the convention used by _build_gate_arrays.
        cy, sy = jnp.cos(yaw / 2),   jnp.sin(yaw / 2)
        cp, sp = jnp.cos(pitch / 2), jnp.sin(pitch / 2)
        cr, sr = jnp.cos(roll / 2),  jnp.sin(roll / 2)
        quat_wu = jnp.array([
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        ])

        motor_speeds_wu = jnp.full((4,), self.cfg.hover_motor_speed)

        # ── Merge warm-up with proposal via jnp.where ──────────────────────
        gate_idx     = jnp.where(use_proposal, proposal_wp_idx,        gate_idx_wu)
        spawn_pos    = jnp.where(use_proposal, proposal_qpos[:3],      spawn_pos_wu)
        quat         = jnp.where(use_proposal, proposal_qpos[3:7],     quat_wu)
        lin_vel      = jnp.where(use_proposal, proposal_qvel[:3],      lin_vel_wu)
        ang_vel      = jnp.where(use_proposal, proposal_qvel[3:6],     ang_vel_wu)
        motor_speeds = jnp.where(use_proposal, proposal_motor_speeds,  motor_speeds_wu)

        # Recompute gate-frame derivations from the merged values so they
        # stay consistent regardless of which path produced them.
        current_gate_pos = jnp.take(self.gate_positions, gate_idx, axis=0)
        current_gate_R   = jnp.take(self.gate_rotmats,   gate_idx, axis=0)
        pos_local        = current_gate_R.T @ (spawn_pos - current_gate_pos)

        qpos = self.sys.qpos0.at[:3].set(spawn_pos).at[3:7].set(quat)
        qvel = jnp.zeros(self.sys.nv).at[:3].set(lin_vel).at[3:6].set(ang_vel)

        data = self.pipeline_init(qpos, qvel)

        next_idx      = jnp.mod(gate_idx + 1, self.num_gates)
        next_gate_pos = jnp.take(self.gate_positions, next_idx, axis=0)
        next_gate_R   = jnp.take(self.gate_rotmats,   next_idx, axis=0)

        obs = self._get_obs(
            data, current_gate_pos, current_gate_R, next_gate_pos, next_gate_R,
            jnp.zeros(4, dtype=jnp.float32),
        )

        # Domain randomization: sample per-episode physics params (motor /
        # PID / aero scales) only during training. Eval uses nominal
        # WorldConfig values so the eval reward signal isn't muddied by
        # physics variation -- but a pinned rotor failure
        # (rotor_failure_dr_enabled=False, non-zero rotor_failure_losses) is
        # morphology, not DR, so `get_nominal_params()` itself applies it on
        # this path (see MotorDynamics.get_nominal_params).
        if self.racing_cfg.domain_randomization and self.racing_cfg.is_train:
            dyn_params = MotorDynamics.sample_randomized_params(k_dr, self.cfg)
        else:
            dyn_params = self.motor_dynamics.get_nominal_params()

        # Per-leaf dyn-params override. When `use_dyn_proposal` is True the
        # caller-provided proposal replaces every field
        dyn_params = jax.tree.map(
            lambda p, d: jnp.where(use_dyn_proposal, p, d),
            proposal_dynamics_params, dyn_params,
        )

        init_dist = jnp.linalg.norm(spawn_pos - current_gate_pos)

        state_info = {
            "rng":                      k_next,  # autoreset rng (consumed in step_env)
            "step":                     0,
            "in_contact":               False,
            "contact_steps":            jnp.array(0, dtype=jnp.int32),
            "z_too_low":                jnp.bool_(False),
            "died":                     jnp.bool_(False),
            "current_motor_speeds":     motor_speeds,
            "prev_omega_err_integral":  jnp.zeros(3),
            "prev_omega_meas":          jnp.zeros(3),
            "prev_action":              jnp.zeros(4),
            "current_gate_idx":         gate_idx,
            "next_gate_idx":            next_idx,
            "prev_x_wrt_gate":          pos_local[0],
            "n_gates_passed":           jnp.array(0, dtype=jnp.int32),
            "last_lap_step":            jnp.int32(0),
            "prev_dist_to_gate":        init_dist,
            "dynamics_params":          dyn_params,
            "reset_proposal_qpos":         jnp.zeros(7,  dtype=jnp.float32),
            "reset_proposal_qvel":         jnp.zeros(6,  dtype=jnp.float32),
            "reset_proposal_motor_speeds": jnp.zeros(4,  dtype=jnp.float32),
            "reset_proposal_wp_idx":       jnp.int32(0),
            "reset_proposal_use":          jnp.bool_(False),
            # Dyn-params proposal: PERSISTENT (unlike the spawn-state proposal
            # above, which is single-shot). Writing the args back unchanged
            # means the next auto-reset re-applies the same per-env
            # (T2W / PID / k_aero / tau_m) override -- the racing analogue
            # of brax's per-env vmapped System for ant.
            "reset_proposal_dynamics_params": proposal_dynamics_params,
            "reset_proposal_dyn_use":         use_dyn_proposal,
        }

        # Body-frame target-gate positions (debug telemetry; same R the obs uses).
        init_R = data.xmat[self.body_com_id]
        init_curr_b = init_R.T @ (current_gate_pos - spawn_pos)
        init_next_b = init_R.T @ (next_gate_pos    - spawn_pos)

        reward, reward_components = self.get_rewards(
            data,
            action=jnp.zeros(4),
            current_gate_pos=current_gate_pos,
            gate_passed=jnp.float32(0.0),
            lap_completed=jnp.float32(0.0),
            prev_dist=init_dist,
            in_contact=jnp.bool_(False),
            died=jnp.bool_(False),
        )

        metrics = {
            **reward_components,
            "gate_passed":      jnp.float32(0.0),
            "lap_completed":    jnp.float32(0.0),
            "lap_time_s":       jnp.float32(0.0),
            **{f"lap_time_{j}": jnp.float32(0.0)
               for j in range(1, self.racing_cfg.max_laps + 1)},
            "in_contact":       jnp.float32(0.0),
            "contact_steps":    jnp.float32(0.0),
            "z_too_low":        jnp.float32(0.0),
            "z_too_high":       jnp.float32(0.0),
            "z_below_ground":   jnp.float32(0.0),
            "died":             jnp.float32(0.0),
            "n_gates_passed":   jnp.float32(0.0),
            "current_gate_idx": gate_idx.astype(jnp.float32),
            "progress_raw":     jnp.float32(0.0),
            "dist_to_gate":     init_dist.astype(jnp.float32),
            "curr_gate_b_x":    init_curr_b[0].astype(jnp.float32),
            "curr_gate_b_y":    init_curr_b[1].astype(jnp.float32),
            "curr_gate_b_z":    init_curr_b[2].astype(jnp.float32),
            "next_gate_b_x":    init_next_b[0].astype(jnp.float32),
            "next_gate_b_y":    init_next_b[1].astype(jnp.float32),
            "next_gate_b_z":    init_next_b[2].astype(jnp.float32),
        }

        data = data.replace(mocap_pos=current_gate_pos[None, :])
        return State(data, obs, reward, jnp.float32(0.0), metrics, state_info)

    # ── Step ───────────────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=0)
    def _pre_physics_step(self, action: jnp.ndarray) -> jnp.ndarray:
        """Once-per-policy-step action preprocessing.

        IsaacLab analogue: ``DirectRLEnv._pre_physics_step(actions)`` in
        ``quadcopter_env.py:1009``. The IsaacLab version also applies a
        beta low-pass filter and pushes through a delay queue; those are
        intentionally left out here while debugging — slot them in at this
        point when re-enabling sim-to-real preprocessing.
        """
        return jnp.clip(action, -1.0, 1.0)

    @partial(jax.jit, static_argnums=0)
    def _apply_action(
        self,
        data: mjx.Data,
        action: jnp.ndarray,
        motor_state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        dyn_params: RandomizedDynamicsParams,
    ) -> Tuple[mjx.Data, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """One physics-rate substep: PID + motor dynamics + force application + ``mjx.step``.

        IsaacLab analogue: one iteration of the ``for _ in range(decimation):
        _apply_action(); sim.step()`` loop in ``DirectRLEnv.step``
        (``quadcopter_env.py:1019``). This helper is the JAX equivalent of
        ``_apply_action()`` + the immediately following ``sim.step()``: it
        recomputes the CTBR moment, advances motor first-order dynamics,
        rebuilds the wrench from motor speeds, adds drag, writes
        ``xfrc_applied`` (in world frame, since MuJoCo expects world-frame
        external wrenches), and steps the simulator by ``cfg.timestep``.

        Args:
            data:        Current ``mjx.Data`` for this substep.
            action:      Already preprocessed action (constant across the
                         ``decimation`` substeps, zero-order hold).
            motor_state: ``(current_motor_speeds, prev_omega_err_integral,
                         prev_omega_meas)`` carried across substeps.
            dyn_params:  Per-episode randomized dynamics parameters.
        Returns:
            (next data after ``mjx.step``, updated motor state).
        """
        motor_speeds, omega_err_integral, omega_meas = motor_state

        # Body-frame velocities for the PID + drag model (matches IsaacLab's
        # `root_ang_vel_b` / `root_com_lin_vel_b`).
        xmat = data.xmat[self.body_com_id, :, :]
        body_ang_vel_b = data.qvel[3:6]
        body_lin_vel_b = xmat.T @ data.qvel[:3]

        body_thrust, body_moment, motor_speeds, omega_err_integral, omega_meas = (
            self.motor_dynamics.forces_and_torques_from_ctbr(
                action, body_ang_vel_b, body_lin_vel_b,
                motor_speeds, omega_err_integral, omega_meas,
                dynamics_params=dyn_params,
            )
        )

        world_thrust = xmat @ body_thrust
        world_moment = xmat @ body_moment
        xfrc = jnp.zeros((self.sys.nbody, 6))
        xfrc = xfrc.at[self.body_com_id, :3].set(world_thrust)
        xfrc = xfrc.at[self.body_com_id, 3:].set(world_moment)

        data = data.replace(xfrc_applied=xfrc)
        data = mjx.step(self.sys, data)

        return data, (motor_speeds, omega_err_integral, omega_meas)

    @partial(jax.jit, static_argnums=0)
    def step_env(self, state: State, action: jnp.ndarray) -> State:
        # Advance the autoreset rng each step so consecutive autoresets see a
        # fresh trajectory. (No in-step stochasticity uses `rng` itself yet.)
        rng, _rng = jax.random.split(state.info["rng"])
        data0 = state.pipeline_state

        dyn_params  = state.info["dynamics_params"]

        if self.cfg.use_motor_dynamics:
            # Action is held constant (zero-order hold) across the
            # `_n_frames_env` substeps
            action_p = self._pre_physics_step(action)

            def _physics_substep(_, carry):
                data_c, motor_state_c = carry
                return self._apply_action(data_c, action_p, motor_state_c, dyn_params)

            init_carry = (
                data0,
                (
                    state.info["current_motor_speeds"],
                    state.info["prev_omega_err_integral"],
                    state.info["prev_omega_meas"],
                ),
            )
            data, (motor_speeds, omega_err_integral, omega_meas) = jax.lax.fori_loop(
                0, self._n_frames_env, _physics_substep, init_carry,
            )

            state.info["current_motor_speeds"] = motor_speeds
            state.info["prev_omega_err_integral"] = omega_err_integral
            state.info["prev_omega_meas"] = omega_meas

            # post processing to match pipeline_step
            q, qd = data.qpos, data.qvel
            x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
            cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
            offset = data.xpos[1:, :] - data.subtree_com[self.sys.body_rootid[1:]]
            offset = Transform.create(pos=offset)
            xd = offset.vmap().do(cvel)

            data = data.replace(q=q, qd=qd, x=x, xd=xd)
        else:
            action_clipped = jnp.clip(
                action,
                self.sys.actuator_ctrlrange[:, 0],
                self.sys.actuator_ctrlrange[:, 1],
            )
            data = self.pipeline_step(data0, action_clipped)
            motor_speeds = state.info["current_motor_speeds"]
            omega_err_integral    = state.info["prev_omega_err_integral"]
            omega_meas   = state.info["prev_omega_meas"]

        # Gate passing
        (
            current_idx, next_idx,
            current_gate_pos, current_gate_R,
            next_gate_pos,    next_gate_R,
            gate_passed, new_x_wrt_gate,
        ) = self._update_gate_target(
            data.qpos[:3],
            state.info["prev_x_wrt_gate"],
            state.info["current_gate_idx"],
            state.info["next_gate_idx"],
        )

        obs = self._get_obs(
            data, current_gate_pos, current_gate_R, next_gate_pos, next_gate_R,
            action,
        )

        # Analytical crash detection: `_check_gate_collisions` tests the
        # drone sphere against all gate bars once per policy step (on the
        # final substep position). MJX contacts are fully disabled for speed.
        # IsaacLab's contact sensor also fires on ground contact, so we add an
        # analytical ground hit (drone sphere touching the floor at z=0). The
        # track has no walls, so gate bars + floor cover all contact surfaces.
        gate_contact   = self._check_gate_collisions(data.qpos[:3])
        ground_contact = data.qpos[2] < self.racing_cfg.drone_radius
        in_contact     = gate_contact | ground_contact
        z_too_low      = data.qpos[2] < self.cfg.low_z_threshold
        z_too_high     = data.qpos[2] > self.cfg.max_altitude
        z_below_ground = data.qpos[2] < 0.0

        state.info["rng"]                      = _rng
        state.info["current_motor_speeds"]     = motor_speeds
        state.info["prev_omega_err_integral"]  = omega_err_integral
        state.info["prev_omega_meas"]          = omega_meas
        state.info["prev_action"]              = action
        state.info["step"]                    += 1

        # Spawn-frame mask: contacts only count toward the death counter after
        # `contact_grace_steps` (matches IsaacLab's `episode_length_buf > 100`
        # mask on `self._crashed`). Default 100.
        contact_counted = in_contact & (state.info["step"] >= self.racing_cfg.contact_grace_steps)
        new_contact_steps = state.info["contact_steps"] + jnp.asarray(contact_counted, dtype=jnp.int32)
        state.info["in_contact"]    = in_contact
        state.info["contact_steps"] = new_contact_steps

        n_gates   = state.info["n_gates_passed"] + jnp.asarray(gate_passed, dtype=jnp.int32)
        new_dist  = jnp.linalg.norm(data.qpos[:3] - current_gate_pos)
        gate_switched = current_idx != state.info["current_gate_idx"]
        # On a gate switch, seed prev_dist at 1.05 * dist-to-next-gate so the
        # switch step pays a +0.05 * dist progress bonus — matches IsaacLab's
        # `_last_distance_to_goal = 1.05 * distance_to_goal` reset on gate pass.
        prev_dist     = jnp.where(gate_switched, 1.05 * new_dist, state.info["prev_dist_to_gate"])

        lap_boundary = jnp.asarray(
            (gate_passed > 0.0)
            & (jnp.mod(n_gates, self.num_gates) == 0)
            & (n_gates > 0),
            dtype=jnp.bool_,
        )

        # Lap completion fires only at second-and-onward boundaries
        lap_completed = jnp.asarray(
            lap_boundary & (n_gates > self.num_gates),
            dtype=jnp.float32,
        )

        # Lap timing: lap_time_s is non-zero only on the step where a
        # CLEAN lap completes (lap_completed=1, i.e. a non-spawn
        # boundary). We use the previous `last_lap_step` (read before
        # the update below) so the duration is (cur_step - prev_lap_boundary_step) * step_dt.
        is_lap_done   = lap_completed > 0.0
        last_lap_step = state.info["last_lap_step"]
        cur_step      = state.info["step"]
        # Duration of the lap that just closed, at ANY boundary (the first lap
        # is the spawn -> gate-0 leg). Non-zero only on a boundary step.
        this_lap_time = jnp.where(
            lap_boundary,
            (cur_step - last_lap_step).astype(jnp.float32) * jnp.float32(self._step_dt),
            jnp.float32(0.0),
        )
        # `lap_time_s`: legacy scalar that excludes the first (spawn) lap.
        lap_time_s = jnp.where(is_lap_done, this_lap_time, jnp.float32(0.0))
        # Per-lap times INCLUDING the first lap. lap_index is the 1-based lap
        # that closes this step (n_gates is an exact multiple of num_gates at a
        # boundary). Each slot fires its lap's real duration once, on the step it
        # closes; a lap that never finishes (crash or time-out) stays 0, left for
        # downstream post-processing. The EvalWrapper sum recovers the value.
        lap_index = jnp.where(lap_boundary, n_gates // self.num_gates, jnp.int32(0))
        per_lap_times = {
            f"lap_time_{j}": jnp.where(lap_index == j, this_lap_time, jnp.float32(0.0))
            for j in range(1, self.racing_cfg.max_laps + 1)
        }
        # Update `last_lap_step` at every lap boundary
        state.info["last_lap_step"] = jnp.where(
            lap_boundary, cur_step, last_lap_step
        )

        state.info["current_gate_idx"]  = current_idx
        state.info["next_gate_idx"]     = next_idx
        state.info["prev_x_wrt_gate"]   = new_x_wrt_gate
        state.info["n_gates_passed"]    = n_gates
        state.info["prev_dist_to_gate"] = new_dist

        # Floor death: after a `self.num_lowz_steps` grace window (derived from cfg.low_z_grace_s) from episode start, any low-z frame is fatal.
        low      = z_too_low & (state.info["step"] >= self.num_lowz_steps)
        # Soft-collision death: cumulative contact frames over threshold.
        crashed  = new_contact_steps > self.racing_cfg.num_contact_steps
        died     = low | z_too_high | crashed

        # Surface terminal flags through state.info
        state.info["z_too_low"] = z_too_low
        state.info["died"]      = died

        reward, reward_components = self.get_rewards(
            data, action, current_gate_pos, gate_passed, lap_completed,
            prev_dist, in_contact, died,
        )

        # Episode time-out is *not* an env-side terminal
        max_gates_reached = n_gates >= self.racing_cfg.max_laps * self.num_gates
        # During training, don't terminate on max_gates_reached
        lap_done = jnp.where(self.racing_cfg.is_train, jnp.bool_(False), max_gates_reached)
        done = jnp.float32(died | lap_done)

        # Debug telemetry: pre-scale progress signal and target gate positions in body frame (same R the obs uses, post gate-pass switch).
        progress_raw = prev_dist - new_dist
        body_R       = data.xmat[self.body_com_id]
        drone_pos    = data.qpos[:3]
        curr_gate_b  = body_R.T @ (current_gate_pos - drone_pos)
        next_gate_b  = body_R.T @ (next_gate_pos    - drone_pos)

        metrics = {
            **state.metrics,  # preserve any keys injected by Brax wrappers (e.g. 'reward')
            **reward_components,
            "gate_passed":    jnp.asarray(gate_passed,         dtype=jnp.float32),
            "lap_completed":  lap_completed,
            "lap_time_s":     lap_time_s,
            **per_lap_times,
            "in_contact":     jnp.asarray(in_contact,          dtype=jnp.float32),
            "contact_steps":  new_contact_steps.astype(jnp.float32),
            "z_too_low":      jnp.asarray(z_too_low,           dtype=jnp.float32),
            "z_too_high":     jnp.asarray(z_too_high,          dtype=jnp.float32),
            "z_below_ground": jnp.asarray(z_below_ground,      dtype=jnp.float32),
            "died":           jnp.asarray(died,                dtype=jnp.float32),
            "n_gates_passed": jnp.asarray(n_gates,             dtype=jnp.float32),
            "current_gate_idx": current_idx.astype(jnp.float32),
            "progress_raw":   progress_raw.astype(jnp.float32),
            "dist_to_gate":   new_dist.astype(jnp.float32),
            "curr_gate_b_x":  curr_gate_b[0].astype(jnp.float32),
            "curr_gate_b_y":  curr_gate_b[1].astype(jnp.float32),
            "curr_gate_b_z":  curr_gate_b[2].astype(jnp.float32),
            "next_gate_b_x":  next_gate_b[0].astype(jnp.float32),
            "next_gate_b_y":  next_gate_b[1].astype(jnp.float32),
            "next_gate_b_z":  next_gate_b[2].astype(jnp.float32),
        }

        data = data.replace(mocap_pos=current_gate_pos[None, :])

        return state.replace(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
        )

    @partial(jax.jit, static_argnums=0)
    def step(self, state: State, action: jnp.ndarray) -> State:
        step_state = self.step_env(state, action)
        re_state = self._reset_impl(
            state.info["rng"],
            proposal_qpos             = state.info["reset_proposal_qpos"],
            proposal_qvel             = state.info["reset_proposal_qvel"],
            proposal_motor_speeds     = state.info["reset_proposal_motor_speeds"],
            proposal_wp_idx           = state.info["reset_proposal_wp_idx"],
            use_proposal              = state.info["reset_proposal_use"],
            proposal_dynamics_params  = state.info["reset_proposal_dynamics_params"],
            use_dyn_proposal          = state.info["reset_proposal_dyn_use"],
        )
        re_state = re_state.replace(info=step_state.info | re_state.info)
        re_state = re_state.replace(metrics=step_state.metrics | re_state.metrics)
        return jax.tree.map(
            lambda a, b: jax.lax.select(state.done > 0.0, a, b),
            re_state, step_state,
        )
