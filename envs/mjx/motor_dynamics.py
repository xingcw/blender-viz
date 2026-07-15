import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import chex

from typing import NamedTuple, Tuple, Optional
from functools import partial


class RandomizedDynamicsParams(NamedTuple):
    """Parameters that can be randomized per episode for domain randomization."""
    k_aero: chex.Array            # (3,) aerodynamic drag coefficients
    kp_omega: chex.Array          # (3,) proportional gains [rp, rp, yaw]
    ki_omega: chex.Array          # (3,) integral gains [rp, rp, yaw]
    kd_omega: chex.Array          # (3,) derivative gains [rp, rp, yaw]
    tau_m: chex.Array             # scalar motor time constant [s]
    thrust_to_weight: chex.Array  # scalar thrust-to-weight ratio (max_thrust / weight)
    rotor_efficiency: chex.Array  # (4,) per-rotor force multiplier; [1,1,1,1] = healthy


class MotorDynamics:
    def __init__(
        self,
        cfg: NamedTuple,
        m: mujoco.MjModel,
        cf_body_ids: list[int],
        effective_dt: Optional[float] = None,
    ):
        # ``effective_dt`` is the interval at which this motor/PID module ticks.
        # Defaults to the physics timestep (original behavior). With n_frames > 1
        # the env calls this once per policy step, so pass n_frames * timestep
        # to keep motor time constants and PID integrals correct in real time.
        self.cfg = cfg
        self.cf_body_ids = cf_body_ids
        self._effective_dt = float(effective_dt) if effective_dt is not None else float(cfg.timestep)

        arm_length = self.cfg.arm_length
        r2o2 = np.sqrt(2.0) / 2.0

        rotor_base = jnp.array(
            [
                [r2o2, r2o2, 0.0],
                [r2o2, -r2o2, 0.0],
                [-r2o2, -r2o2, 0.0],
                [-r2o2, r2o2, 0.0],
            ],
        )
        self._rotor_positions = arm_length * rotor_base
        self._rotor_directions = jnp.array([1.0, -1.0, 1.0, -1.0])

        self.k = self.cfg.k_m / self.cfg.k_eta  # moment/thrust ratio

        cross_terms = jnp.cross(self._rotor_positions, jnp.array([0.0, 0.0, 1.0]))
        thrust_row = jnp.ones((1, 4))
        torque_rows = cross_terms[:, :2].T
        yaw_row = (self.k * self._rotor_directions)[None, :]
        self.f_to_TM = jnp.concatenate([thrust_row, torque_rows, yaw_row], axis=0)
        self.TM_to_f = jnp.linalg.inv(self.f_to_TM)
        self._TM_to_f_T = self.TM_to_f.T
        self._f_to_TM_T = self.f_to_TM.T

        # inertia tensor and mass properties
        inertia = jnp.asarray(m.body_inertia[self.cf_body_ids[0]])
        self.mass = m.body_mass[self.cf_body_ids[0]]
        gravity_mag = np.abs(m.opt.gravity[2])
        self.weight = self.mass * gravity_mag
        q = jnp.asarray(m.body_iquat[self.cf_body_ids[0]])
        w, x, y, z = q
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        R = jnp.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ],
        )
        self.inertia_tensor = jnp.asarray(R.T @ jnp.diag(inertia) @ R)

        # cached PID and motor constants
        self._kp_omega = jnp.asarray(self.cfg.kp_omega)
        self._ki_omega = jnp.asarray(self.cfg.ki_omega)
        self._kd_omega = jnp.asarray(self.cfg.kd_omega)
        self._body_rate_scale = jnp.array(
            [
                self.cfg.body_rate_scale_xy,
                self.cfg.body_rate_scale_xy,
                self.cfg.body_rate_scale_z,
            ],
        )
        # self._pid_loop_rate = self.cfg.pid_loop_rate_hz
        self._pid_loop_dt = self._effective_dt
        self._pid_loop_rate = 1.0 / self._pid_loop_dt
        self._i_limits = jnp.array(
            [self.cfg.i_limit_rp, self.cfg.i_limit_rp, self.cfg.i_limit_y],
        )
        self._use_i_limits = bool((self.cfg.i_limit_rp > 0) or (self.cfg.i_limit_y > 0))
        self._zero_tol = 1e-4

        self._k_eta = self.cfg.k_eta
        self._inv_k_eta = 1.0 / self.cfg.k_eta
        self._k_aero = jnp.asarray(self.cfg.k_aero)
        self._motor_speed_min = self.cfg.motor_speed_min
        self._motor_speed_max = self.cfg.motor_speed_max
        # Explicit-Euler step factor for the motor first-order filter, clamped
        # to ≤1 so that effective_dt > tau_m doesn't cause overshoot/oscillation.
        # At dt/tau >= 1 the motor effectively snaps to the commanded speed.
        self._motor_speed_step = min(self._effective_dt / self.cfg.tau_m, 1.0)
        self._thrust_scale = 0.5 * self.weight * self.cfg.thrust_to_weight
        self._z_axis = jnp.array([0.0, 0.0, 1.0])

        # Store nominal values for randomization
        self._nominal_k_aero = jnp.asarray(self.cfg.k_aero)
        self._nominal_kp_omega = jnp.asarray(self.cfg.kp_omega)
        self._nominal_ki_omega = jnp.asarray(self.cfg.ki_omega)
        self._nominal_kd_omega = jnp.asarray(self.cfg.kd_omega)

    def get_nominal_params(self) -> RandomizedDynamicsParams:
        """Get the nominal (non-randomized) dynamics parameters.

        Rotor failure is a DETERMINISTIC specialist parameter (pinned via
        cfg.rotor_failure_losses at env-build time), not a per-episode DR
        draw. So when an eval env runs with is_train=False (DR off) we
        still want to apply the pinned failure -- otherwise the eval
        reports the healthy-quad reward instead of the failed-rotor reward
        the specialist was trained for. The DR baseline path
        (rotor_failure_dr_enabled=True) leaves rotor_failure_losses at its
        all-zero default and rotor_failure_loss_min/max drives the
        per-episode per-rotor draw inside sample_randomized_params; on the
        is_train=False branch this returns ones(4) as before for that case.
        """
        losses = jnp.asarray(self.cfg.rotor_failure_losses, dtype=jnp.float32)
        if bool(self.cfg.rotor_failure_dr_enabled):
            rotor_efficiency = jnp.ones(4, dtype=jnp.float32)
        else:
            rotor_efficiency = 1.0 - losses
        return RandomizedDynamicsParams(
            k_aero=self._nominal_k_aero,
            kp_omega=self._nominal_kp_omega,
            ki_omega=self._nominal_ki_omega,
            kd_omega=self._nominal_kd_omega,
            tau_m=jnp.array(self.cfg.tau_m),
            thrust_to_weight=jnp.array(self.cfg.thrust_to_weight),
            rotor_efficiency=rotor_efficiency,
        )

    @staticmethod
    def sample_randomized_params(
        key: chex.PRNGKey,
        cfg: NamedTuple,
    ) -> RandomizedDynamicsParams:
        """Sample randomized dynamics parameters from uniform distributions.

        Args:
            key: JAX PRNG key
            cfg: WorldConfig with randomization scale bounds

        Returns:
            RandomizedDynamicsParams with sampled values
        """
        keys = jax.random.split(key, 10)  # ..., t2w, rotor_loss_vec

        # k_aero: independent scale per axis (xy, xy, z)
        k_aero_scale = jax.random.uniform(
            keys[0], (3,), minval=cfg.k_aero_scale_min, maxval=cfg.k_aero_scale_max
        )
        k_aero = jnp.asarray(cfg.k_aero) * k_aero_scale

        # Roll/pitch PID scales
        kp_rp_scale = jax.random.uniform(
            keys[1], (), minval=cfg.kp_omega_rp_scale_min, maxval=cfg.kp_omega_rp_scale_max
        )
        ki_rp_scale = jax.random.uniform(
            keys[2], (), minval=cfg.ki_omega_rp_scale_min, maxval=cfg.ki_omega_rp_scale_max
        )
        kd_rp_scale = jax.random.uniform(
            keys[3], (), minval=cfg.kd_omega_rp_scale_min, maxval=cfg.kd_omega_rp_scale_max
        )

        # Yaw PID scales
        kp_y_scale = jax.random.uniform(
            keys[4], (), minval=cfg.kp_omega_y_scale_min, maxval=cfg.kp_omega_y_scale_max
        )
        ki_y_scale = jax.random.uniform(
            keys[5], (), minval=cfg.ki_omega_y_scale_min, maxval=cfg.ki_omega_y_scale_max
        )
        kd_y_scale = jax.random.uniform(
            keys[6], (), minval=cfg.kd_omega_y_scale_min, maxval=cfg.kd_omega_y_scale_max
        )

        # Motor time constant
        tau_m_scale = jax.random.uniform(
            keys[7], (), minval=cfg.tau_m_scale_min, maxval=cfg.tau_m_scale_max
        )
        tau_m = jnp.array(cfg.tau_m * tau_m_scale)

        # Thrust-to-weight ratio. Stored absolute (cfg.thrust_to_weight * scale)
        # so consumers can read it directly without re-applying the cfg base.
        t2w_scale = jax.random.uniform(
            keys[8], (), minval=cfg.t2w_scale_min, maxval=cfg.t2w_scale_max
        )
        thrust_to_weight = jnp.array(cfg.thrust_to_weight * t2w_scale)

        # Rotor failure DR. cfg.rotor_failure_dr_enabled toggles:
        #   dr_enabled=True   -> sample 4 independent per-rotor losses uniformly
        #                        from [rotor_failure_loss_min, rotor_failure_loss_max];
        #                        cfg.rotor_failure_losses is ignored (DR draw governs).
        #   dr_enabled=False  -> use cfg.rotor_failure_losses (length-4) as the
        #                        deterministic pinned per-rotor loss vector
        #                        (specialist morphology).
        sampled_losses = jax.random.uniform(
            keys[9], (4,),
            minval=cfg.rotor_failure_loss_min,
            maxval=cfg.rotor_failure_loss_max,
        )
        pinned_losses = jnp.asarray(cfg.rotor_failure_losses, dtype=jnp.float32)
        if bool(cfg.rotor_failure_dr_enabled):
            losses = sampled_losses
        else:
            losses = pinned_losses
        rotor_efficiency = 1.0 - losses

        # Build PID gain arrays [rp, rp, yaw]
        kp_omega = jnp.array([
            cfg.kp_omega_rp * kp_rp_scale,
            cfg.kp_omega_rp * kp_rp_scale,
            cfg.kp_omega_y  * kp_y_scale,
        ])
        ki_omega = jnp.array([
            cfg.ki_omega_rp * ki_rp_scale,
            cfg.ki_omega_rp * ki_rp_scale,
            cfg.ki_omega_y  * ki_y_scale,
        ])
        kd_omega = jnp.array([
            cfg.kd_omega_rp * kd_rp_scale,
            cfg.kd_omega_rp * kd_rp_scale,
            cfg.kd_omega_y  * kd_y_scale,
        ])

        return RandomizedDynamicsParams(
            k_aero=k_aero,
            kp_omega=kp_omega,
            ki_omega=ki_omega,
            kd_omega=kd_omega,
            tau_m=tau_m,
            thrust_to_weight=thrust_to_weight,
            rotor_efficiency=rotor_efficiency,
        )

    @partial(jax.jit, static_argnums=[0])
    def _compute_motor_speeds(self, wrench_des: chex.Array) -> chex.Array:
        """Compute desired motor speeds from desired wrench.
        Args:
            wrench_des: (4,) array of desired wrench [thrust, roll, pitch, yaw] in body frame.
        Returns:
            motor_speeds_des: (4,) array of desired motor speeds [rad/s].
        """
        # OPTIMIZED: Use einsum for better GPU utilization (JAX optimizes einsum well)
        f_des = jnp.einsum("i,ij->j", wrench_des, self._TM_to_f_T)
        motor_speed_squared = f_des * self._inv_k_eta
        # OPTIMIZED: Use sign * sqrt(abs) pattern (more efficient than where)
        motor_speeds_des = jnp.sign(motor_speed_squared) * jnp.sqrt(jnp.abs(motor_speed_squared))
        motor_speeds_des = jnp.clip(
            motor_speeds_des, self._motor_speed_min, self._motor_speed_max
        )
        return motor_speeds_des

    @partial(jax.jit, static_argnums=[0])
    def _get_moment_from_ctbr(
        self,
        action: chex.Array,
        root_com_ang_vel_b: chex.Array,
        previous_omega_err_integral: chex.Array,
        previous_omega_meas: chex.Array,
        kp_omega: Optional[chex.Array] = None,
        ki_omega: Optional[chex.Array] = None,
        kd_omega: Optional[chex.Array] = None,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Compute desired moment from ctbr action using PID controller.
        Args:
            action: (4,) array of [thrust, roll_rate, pitch_rate, yaw_rate] in
                normalized [-1, 1] range.
            root_com_ang_vel_b: (3,) array of body angular velocity in body frame [rad/s].
            previous_omega_err_integral: (3,) array of previous angular velocity error integral.
            previous_omega_meas: (3,) array of previous measured angular velocity.
            kp_omega: Optional (3,) array of proportional gains (uses nominal if None).
            ki_omega: Optional (3,) array of integral gains (uses nominal if None).
            kd_omega: Optional (3,) array of derivative gains (uses nominal if None).
        Returns:
            cmd_moment: (3,) array of desired moment in body frame [Nm].
            omega_err_integral: (3,) array of updated angular velocity error integral.
            omega_meas: (3,) array of current measured angular velocity.
        """
        # Use provided or default PID gains
        kp = kp_omega if kp_omega is not None else self._kp_omega
        ki = ki_omega if ki_omega is not None else self._ki_omega
        kd = kd_omega if kd_omega is not None else self._kd_omega

        # desired angular velocity
        omega_des = action[1:4] * self._body_rate_scale

        # pid integral term
        omega_err = omega_des - root_com_ang_vel_b
        omega_err_integral = previous_omega_err_integral + omega_err * self._pid_loop_dt

        # Apply separate I-limits for roll/pitch and yaw
        if self._use_i_limits:
            omega_err_integral = jnp.clip(
                omega_err_integral, -self._i_limits, self._i_limits
            )

        # OPTIMIZED: PID derivative term - combine operations
        # Check if previous measurement is near zero (first iteration)
        is_first_iter = jnp.abs(previous_omega_meas) < self._zero_tol
        omega_meas_prev = jnp.where(is_first_iter, root_com_ang_vel_b, previous_omega_meas)
        omega_meas_dot = (root_com_ang_vel_b - omega_meas_prev) * self._pid_loop_rate

        # OPTIMIZED: Compute PID terms more efficiently
        # Pre-compute PID contributions
        p_term = kp * omega_err
        i_term = ki * omega_err_integral
        d_term = kd * omega_meas_dot
        omega_dot = p_term + i_term - d_term

        # OPTIMIZED: Use einsum for matrix-vector product (better GPU utilization)
        cmd_moment = jnp.einsum("ij,j->i", self.inertia_tensor, omega_dot)
        return cmd_moment, omega_err_integral, root_com_ang_vel_b

    @partial(jax.jit, static_argnums=[0])
    def forces_and_torques_from_ctbr(
        self,
        action: chex.Array,
        root_com_ang_vel_b: chex.Array,
        root_com_lin_vel_b: chex.Array,
        current_motor_speeds: chex.Array,
        previous_omega_err_integral: chex.Array,
        previous_omega_meas: chex.Array,
        dynamics_params: Optional[RandomizedDynamicsParams] = None,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """Compute thrust, moment, and motor speeds from ctbr action.
        Args:
            action: (4,) array of [thrust, roll_rate, pitch_rate, yaw_rate] in
                normalized [-1, 1] range.
            root_com_ang_vel_b: (3,) array of body angular velocity in body frame [rad/s].
            root_com_lin_vel_b: (3,) array of body linear velocity in body frame [m/s].
            current_motor_speeds: (4,) array of current motor speeds [rad/s].
            previous_omega_err_integral: (3,) array of previous angular velocity error integral.
            previous_omega_meas: (3,) array of previous measured angular velocity.
            dynamics_params: Optional RandomizedDynamicsParams for domain randomization.
                If None, uses nominal parameters.
        Returns:
            thrust: (1,) array of thrust force in body frame [N].
            moment: (3,) array of moment in body frame [Nm].
            motor_speeds: (4,) array of updated motor speeds [rad/s].
            omega_err_integral: (3,) array of updated angular velocity error integral.
            omega_meas: (3,) array of current measured angular velocity.
        """
        # Get parameters (use randomized if provided, else nominal)
        if dynamics_params is not None:
            k_aero       = dynamics_params.k_aero
            kp_omega     = dynamics_params.kp_omega
            ki_omega     = dynamics_params.ki_omega
            kd_omega     = dynamics_params.kd_omega
            motor_step   = jnp.minimum(self._effective_dt / dynamics_params.tau_m, 1.0)
            # Per-episode thrust scale derived from the randomized T2W. Same
            # formula as the cached `self._thrust_scale` (0.5 * weight * T2W),
            # but T2W now varies per env.
            thrust_scale = 0.5 * self.weight * dynamics_params.thrust_to_weight
            rotor_efficiency = dynamics_params.rotor_efficiency
        else:
            k_aero       = self._k_aero
            kp_omega     = self._kp_omega
            ki_omega     = self._ki_omega
            kd_omega     = self._kd_omega
            motor_step   = self._motor_speed_step
            thrust_scale = self._thrust_scale
            rotor_efficiency = jnp.ones(4, dtype=jnp.float32)

        # total thrust to hover
        thrust_hover = thrust_scale * (action[0] + 1.0)
        # get desired moment from ctbr (action) with potentially randomized PID gains
        moment_des, omega_err_integral, omega_meas = self._get_moment_from_ctbr(
            action, root_com_ang_vel_b, previous_omega_err_integral, previous_omega_meas,
            kp_omega=kp_omega, ki_omega=ki_omega, kd_omega=kd_omega,
        )
        # OPTIMIZED: Build wrench_des directly without intermediate operations
        wrench_des = jnp.concatenate([jnp.array([thrust_hover]), moment_des])
        # get desired motor speeds from desired wrench
        motor_speeds_des = self._compute_motor_speeds(wrench_des)

        # --- ran every loop
        # motor forces and updated motor speeds
        motor_delta = motor_speeds_des - current_motor_speeds
        motor_speeds = current_motor_speeds + motor_delta * motor_step
        motor_speeds = jnp.clip(
            motor_speeds, self._motor_speed_min, self._motor_speed_max
        )
        motor_forces = self._k_eta * motor_speeds**2
        # Per-rotor failure: scale each rotor's force by its efficiency
        # factor. The PID controller is unchanged and still commands all
        # 4 rotors as if healthy; the broken rotor just delivers less.
        # f_to_TM_T couples this to yaw torque automatically (a half-
        # thrust rotor produces half the yaw-reaction torque).
        motor_forces = motor_forces * rotor_efficiency

        # OPTIMIZED: Use einsum for potentially better GPU utilization
        # wrench from motor forces: (4,) @ (4, 4) -> (4,)
        wrench = jnp.einsum("i,ij->j", motor_forces, self._f_to_TM_T)

        # OPTIMIZED: Compute theta_dot and drag more efficiently
        # thrust and moments
        theta_dot = jnp.sum(motor_speeds)
        # Pre-compute drag coefficient using potentially randomized k_aero
        drag_coeff = -theta_dot * k_aero
        drag = drag_coeff * root_com_lin_vel_b

        thrust = drag + self._z_axis * wrench[0]
        moment = wrench[1:]

        return thrust, moment, motor_speeds, omega_err_integral, omega_meas
