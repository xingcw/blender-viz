# Copyright 2026 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import logging as std_logging
import os
import tempfile
import time
from typing import Any, Callable, Mapping, Optional, Tuple, Union

# Suppress noisy XLA/PJRt C++ warnings (e.g. pjrt_executable.cc "Assume version compatibility").
# Must run before importing JAX (directly or via brax).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import logger as metric_logger
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import optimizer as ppo_optimizer
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from utils.gcp_utils import make_dir, save_pickle, save_json, save_txt


def _device_put_replicated(x, devices):
  """Modern-API replacement for the deprecated jax.device_put_replicated.

  Produces a pytree with a leading device axis sharded one-slice-per-device,
  matching the layout the pmap'd training loop expects (see _unpmap).
  """
  n = len(devices)
  mesh = Mesh(np.array(devices), axis_names=("d",))
  sharded = NamedSharding(mesh, P("d"))
  return jax.tree_util.tree_map(
      lambda v: jax.device_put(jnp.stack([v] * n), sharded), x
  )

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""

  optimizer_state: optax.OptState
  params: ppo_losses.PPONetworkParams
  normalizer_params: running_statistics.RunningStatisticsState
  env_steps: types.UInt64


def _unpmap(v):
  # Avoid degraded performance under the new jax.pmap.
  return jax.tree_util.tree_map(
      lambda x: x.addressable_shards[0].data.squeeze(0), v
  )


def _strip_weak_type(tree):
  # brax user code is sometimes ambiguous about weak_type.  in order to
  # avoid extra jit recompilations we strip all weak types from user input
  def f(leaf):
    leaf = jnp.asarray(leaf)
    return jnp.astype(leaf, leaf.dtype)

  return jax.tree_util.tree_map(f, tree)


def _maybe_wrap_env(
    env: envs.Env,
    wrap_env: bool,
    num_envs: int,
    episode_length: Optional[int],
    action_repeat: int,
    device_count: int,
    key_env: PRNGKey,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
):
  """Wraps the environment for training/eval if wrap_env is True."""
  if not wrap_env:
    return env
  if episode_length is None:
    raise ValueError('episode_length must be specified in ppo.train')
  v_randomization_fn = None
  if randomization_fn is not None:
    randomization_batch_size = num_envs // device_count
    # all devices gets the same randomization rng
    randomization_rng = jax.random.split(key_env, randomization_batch_size)
    v_randomization_fn = functools.partial(
        randomization_fn, rng=randomization_rng
    )
  if wrap_env_fn is not None:
    wrap_for_training = wrap_env_fn
  else:
    wrap_for_training = envs.training.wrap
  env = wrap_for_training(
      env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=v_randomization_fn,
  )  # pytype: disable=wrong-keyword-args
  return env


def _random_translate_pixels(
    obs: Mapping[str, jax.Array], key: PRNGKey
) -> Mapping[str, jax.Array]:
  """Apply random translations to B x T x ... pixel observations.

  The same shift is applied across the unroll_length (T) dimension.

  Args:
    obs: a dictionary of observations
    key: a PRNGKey

  Returns:
    A dictionary of observations with translated pixels
  """

  @jax.vmap
  def rt_all_views(
      ub_obs: Mapping[str, jax.Array], key: PRNGKey
  ) -> Mapping[str, jax.Array]:
    # Expects dictionary of unbatched observations.
    def rt_view(
        img: jax.Array, padding: int, key: PRNGKey
    ) -> jax.Array:  # TxHxWxC
      # Randomly translates a set of pixel inputs.
      # Adapted from
      # https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/agents/drq/augmentations.py
      crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
      zero = jnp.zeros((1,), dtype=jnp.int32)
      crop_from = jnp.concatenate([zero, crop_from, zero])
      padded_img = jnp.pad(
          img,
          ((0, 0), (padding, padding), (padding, padding), (0, 0)),
          mode='edge',
      )
      return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)

    out = {}
    for k_view, v_view in ub_obs.items():
      if k_view.startswith('pixels/'):
        key, key_shift = jax.random.split(key)
        out[k_view] = rt_view(v_view, 4, key_shift)
    return {**ub_obs, **out}

  bdim = next(iter(obs.items()), None)[1].shape[0]
  keys = jax.random.split(key, bdim)
  obs = rt_all_views(obs, keys)
  return obs


def _remove_pixels(
    obs: Union[jnp.ndarray, Mapping[str, jax.Array]],
) -> Union[jnp.ndarray, Mapping[str, jax.Array]]:
  """Removes pixel observations from the observation dict."""
  if not isinstance(obs, Mapping):
    return obs
  return {k: v for k, v in obs.items() if not k.startswith('pixels/')}


def train(
    environment: envs.Env,
    num_timesteps: int,
    max_devices_per_host: Optional[int] = None,
    # high-level control flow
    wrap_env: bool = True,
    vision: bool = False,
    augment_pixels: bool = False,
    # environment wrapper
    num_envs: int = 1,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    # ppo params
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    normalize_observations_std_eps: float = 0.0,
    normalize_observations_mode: str = "welford",
    normalize_until_count: Optional[int] = None,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    clipping_epsilon_value: float | None = None,
    gae_lambda: float = 0.95,
    max_grad_norm: Optional[float] = None,
    normalize_advantage: bool = True,
    vf_loss_coefficient: float = 0.5,
    bootstrap_on_timeout: bool = False,
    use_distributional_critic: bool = False,
    desired_kl: float = 0.01,
    learning_rate_schedule: Optional[
        Union[str, ppo_optimizer.LRSchedule]
    ] = None,
    learning_rate_schedule_min_lr: float = 1e-5,
    learning_rate_schedule_max_lr: float = 1e-2,
    network_factory: types.NetworkFactory[
        ppo_networks.PPONetworks
    ] = ppo_networks.make_ppo_networks,
    seed: int = 0,
    use_pmap_on_reset: bool = True,
    # eval
    num_evals: int = 1,
    eval_env: Optional[envs.Env] = None,
    num_eval_envs: int = 128,
    deterministic_eval: bool = False,
    # training metrics
    log_training_metrics: bool = False,
    training_metrics_steps: Optional[int] = None,
    # callbacks
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    # checkpointing
    save_checkpoint_path: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    restore_params: Optional[Any] = None,
    restore_value_fn: bool = True,
    run_evals: bool = True,
    checkpoint_logdir: Optional[str] = None,
    num_checkpoints: int = 20,
    ckpt_save_threshold: dict = {'eval/episode_reward': 0.0},
    save_q_network: bool = False,
):
  """PPO training.

  Args:
    environment: the environment to train
    num_timesteps: the total number of environment steps to use during training
    max_devices_per_host: maximum number of chips to use per host process
    wrap_env: If True, wrap the environment for training. Otherwise use the
      environment as is.
    vision: whether to use vision observations.
    augment_pixels: whether to add image augmentation to pixel inputs
    num_envs: the number of parallel environments to use for rollouts
      NOTE `num_envs` must be divisible by the total number of chips since each
        chip gets `num_envs // total_number_of_chips` environments to roll out
      NOTE `batch_size * num_minibatches` must be divisible by `num_envs` since
        data generated by `num_envs` parallel envs gets used for gradient
        updates over `num_minibatches` of data, where each minibatch has a
        leading dimension of `batch_size`
    episode_length: the length of an environment episode
    action_repeat: the number of timesteps to repeat an action
    wrap_env_fn: a custom function that wraps the environment for training. If
      not specified, the environment is wrapped with the default training
      wrapper.
    randomization_fn: a user-defined callback function that generates randomized
      environments
    learning_rate: learning rate for ppo loss
    entropy_cost: entropy reward for ppo loss, higher values increase entropy of
      the policy
    discounting: discounting rate
    unroll_length: the number of timesteps to unroll in each environment. The
      PPO loss is computed over `unroll_length` timesteps
    batch_size: the batch size for each minibatch SGD step
    num_minibatches: the number of times to run the SGD step, each with a
      different minibatch with leading dimension of `batch_size`
    num_updates_per_batch: the number of times to run the gradient update over
      all minibatches before doing a new environment rollout
    num_resets_per_eval: the number of environment resets to run between each
      eval. The environment resets occur on the host
    normalize_observations: whether to normalize observations
    normalize_observations_std_eps: small value added to the standard deviation
      for obs normalization to improve numerical stability
    normalize_observations_mode: method to use for running statistics, welford
      is the default, but ema is more numerically stable for long training runs
    normalize_until_count: the number of environment steps to normalize
      observations until
    reward_scaling: float scaling for reward
    clipping_epsilon: clipping epsilon for PPO loss
    clipping_epsilon_value: Value function loss clipping epsilon
    gae_lambda: General advantage estimation lambda
    max_grad_norm: gradient clipping norm value. If None, no clipping is done
    normalize_advantage: whether to normalize advantage estimate
    vf_loss_coefficient: Coefficient for value function loss.
    bootstrap_on_timeout: if True, bootstrap value on time_out steps using
      reward += gamma * V(s) * time_out. Environments should set
      state.info['time_out'] = 1.0 and done=True for steps where the episode
      ends due to a time_out.
    use_distributional_critic: whether to use a distributional critic
    desired_kl: Desired KL divergence for adaptive KL divergence learning rate
      schedule.
    learning_rate_schedule: Learning rate schedule for the optimizer.
    learning_rate_schedule_min_lr: Minimum learning rate for adaptive KL
      learning rate schedule.
    learning_rate_schedule_max_lr: Maximum learning rate for adaptive KL
      learning rate schedule.
    network_factory: function that generates networks for policy and value
      functions
    seed: random seed
    num_evals: the number of evals to run during the entire training run.
      Increasing the number of evals increases total training time
    eval_env: an optional environment for eval only, defaults to `environment`
    num_eval_envs: the number of envs to use for evluation. Each env will run 1
      episode, and all envs run in parallel during eval.
    deterministic_eval: whether to run the eval with a deterministic policy
    log_training_metrics: whether to log training metrics and callback to
      progress_fn
    training_metrics_steps: the number of environment steps between logging
      training metrics
    progress_fn: a user-defined callback function for reporting/plotting metrics
    policy_params_fn: a user-defined callback function that can be used for
      saving custom policy checkpoints or creating policy rollouts and videos
    save_checkpoint_path: the path used to save checkpoints. If None, no
      checkpoints are saved.
    restore_checkpoint_path: the path used to restore previous model params
    restore_params: raw network parameters to restore the TrainingState from.
      These override `restore_checkpoint_path`. These paramaters can be obtained
      from the return values of ppo.train().
    restore_value_fn: whether to restore the value function from the checkpoint
      or use a random initialization
    run_evals: if True, use the evaluator num_eval times to collect distinct
      eval rollouts. If False, num_eval_envs and eval_env are ignored.
      progress_fn is then expected to use training_metrics.
    use_pmap_on_reset: default to True. if True, use pmap instead of vmap for
      env.reset across devices.

  Returns:
    Tuple of (make_policy function, network params, metrics)
  """
  assert batch_size * num_minibatches % num_envs == 0

  if vision and action_repeat != 1:
    raise ValueError(
        "Implement action_repeat using PipelineEnv's _n_frames to avoid"
        ' unnecessary rendering!'
    )

  xt = time.time()

  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count
  if max_devices_per_host:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  logging.info(
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d',
      jax.device_count(),
      process_count,
      process_id,
      local_device_count,
      local_devices_to_use,
  )
  device_count = local_devices_to_use * process_count

  # Set up file logging to the checkpoint directory (process 0 only). When the
  # directory is on GCS the handler writes to a local temp file that we sync to
  # GCS on cleanup (mirrors data_gen.train_sac_brax).
  log_file_handler = None
  log_path_gcs = None
  if checkpoint_logdir and process_id == 0:
    make_dir(checkpoint_logdir)
    log_path = os.path.join(checkpoint_logdir, 'log.txt')
    if checkpoint_logdir.startswith('gs://'):
      fd, log_local_path = tempfile.mkstemp(suffix='.log.txt')
      os.close(fd)
      log_path_gcs = log_path
      log_path = log_local_path
    log_file_handler = std_logging.FileHandler(
        log_path, mode='a', encoding='utf-8'
    )
    log_file_handler.setLevel(std_logging.INFO)
    log_file_handler.setFormatter(
        std_logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    # This module logs via `absl.logging`; those records go to the `absl`
    # logger and do NOT propagate to the root logger, so attach the handler
    # there. The absl logger defaults to level WARNING (its own stderr handler
    # filters by verbosity, not level), so lower it to INFO -- otherwise our
    # handler never sees the INFO records this module emits.
    logging.get_absl_logger().addHandler(log_file_handler)
    logging.get_absl_logger().setLevel(std_logging.INFO)

  def _finalize_log_handler():
    if log_file_handler is None:
      return
    logging.get_absl_logger().removeHandler(log_file_handler)
    log_file_handler.flush()
    if log_path_gcs is not None:
      with open(log_file_handler.baseFilename, 'r', encoding='utf-8') as f:
        save_txt(f.read(), log_path_gcs)
      try:
        os.remove(log_file_handler.baseFilename)
      except OSError:
        pass
    log_file_handler.close()

  logging.info('training config: %s', {
      'num_timesteps': num_timesteps,
      'num_evals': num_evals,
      'episode_length': episode_length,
      'action_repeat': action_repeat,
      'num_envs': num_envs,
      'batch_size': batch_size,
      'num_minibatches': num_minibatches,
      'num_updates_per_batch': num_updates_per_batch,
      'unroll_length': unroll_length,
      'learning_rate': learning_rate,
      'entropy_cost': entropy_cost,
      'discounting': discounting,
      'reward_scaling': reward_scaling,
      'gae_lambda': gae_lambda,
      'clipping_epsilon': clipping_epsilon,
      'normalize_observations': normalize_observations,
      'normalize_advantage': normalize_advantage,
      'max_grad_norm': max_grad_norm,
      'max_devices_per_host': max_devices_per_host,
      'seed': seed,
      'use_distributional_critic': use_distributional_critic,
      'learning_rate_schedule': learning_rate_schedule,
      'save_q_network': save_q_network,
      'checkpoint_logdir': checkpoint_logdir,
      'restore_checkpoint_path': restore_checkpoint_path,
      'num_checkpoints': num_checkpoints,
      'ckpt_save_threshold': ckpt_save_threshold,
  })

  # The number of environment steps executed for every training step.
  env_step_per_training_step = (
      batch_size * unroll_length * num_minibatches * action_repeat
  )
  num_evals_after_init = max(num_evals - 1, 1)
  # The number of training_step calls per training_epoch call.
  # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
  #                                 num_resets_per_eval))
  num_training_steps_per_epoch = np.ceil(
      num_timesteps
      / (
          num_evals_after_init
          * env_step_per_training_step
          * max(num_resets_per_eval, 1)
      )
  ).astype(int)

  logging.info(
      'num_timesteps: %s, num_evals_after_init: %s, env_step_per_training_step: '
      '%s, num_resets_per_eval: %s, num_training_steps_per_epoch: %s',
      num_timesteps,
      num_evals_after_init,
      env_step_per_training_step,
      num_resets_per_eval,
      num_training_steps_per_epoch,
  )

  # Total ckpt-buffer slots: one entry per training step across every epoch
  # (one epoch per reset, num_resets_per_eval resets per eval iteration), plus
  # the prepended random-init ckpt at index 0. Mirrors data_gen.train_sac_brax.
  max_slots = (
      1
      + num_evals_after_init
      * max(num_resets_per_eval, 1)
      * int(num_training_steps_per_epoch)
  )
  if num_checkpoints > max_slots:
    logging.warning(
        'Capping num_checkpoints from %d to %d (max_slots = 1 + '
        'num_evals_after_init * max(num_resets_per_eval, 1) * '
        'num_training_steps_per_epoch = %d)',
        num_checkpoints, max_slots, max_slots,
    )
    num_checkpoints = max_slots

  key = jax.random.PRNGKey(seed)
  global_key, local_key = jax.random.split(key)
  del key
  local_key = jax.random.fold_in(local_key, process_id)
  local_key, key_env, eval_key = jax.random.split(local_key, 3)
  # key_networks should be global, so that networks are initialized the same
  # way for different processes.
  key_policy, key_value = jax.random.split(global_key)
  del global_key

  assert num_envs % device_count == 0

  env = _maybe_wrap_env(
      environment,
      wrap_env,
      num_envs,
      episode_length,
      action_repeat,
      device_count,
      key_env,
      wrap_env_fn,
      randomization_fn,
  )

  def reset_fn_donated_env_state(env_state_donated, key_envs):
    return env.reset(key_envs)

  key_envs = jax.random.split(key_env, num_envs // process_count)
  key_envs = jnp.reshape(
      key_envs, (local_devices_to_use, -1) + key_envs.shape[1:]
  )
  if local_devices_to_use > 1 or use_pmap_on_reset:
    reset_fn_ = jax.pmap(env.reset, axis_name=_PMAP_AXIS_NAME)
    env_state = reset_fn_(key_envs)
    reset_fn = jax.pmap(
        reset_fn_donated_env_state,
        axis_name=_PMAP_AXIS_NAME,
        donate_argnums=(0,),
    )
  else:
    reset_fn_ = jax.jit(jax.vmap(env.reset))
    env_state = reset_fn_(key_envs)
    reset_fn = jax.jit(
        reset_fn_donated_env_state, donate_argnums=(0,), keep_unused=True
    )

  # Discard the batch axes over devices and envs.
  obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)

  normalize = lambda x, y: x
  if normalize_observations:
    normalize = running_statistics.normalize
  if use_distributional_critic and clipping_epsilon_value is None:
    raise AssertionError(
        'clipping_epsilon_value must not be None when '
        'use_distributional_critic=True (it serves as kappa for quantile '
        'Huber loss)'
    )

  ppo_network = network_factory(
      obs_shape, env.action_size, preprocess_observations_fn=normalize
  )
  make_policy = ppo_networks.make_inference_fn(
      ppo_network,
      compute_value=bootstrap_on_timeout or clipping_epsilon_value is not None,
      use_distributional_critic=use_distributional_critic,
  )

  # Optimizer.
  base_optimizer = optax.adam(learning_rate=learning_rate)
  lr_schedule = learning_rate_schedule or ppo_optimizer.LRSchedule.NONE
  lr_schedule = ppo_optimizer.LRSchedule(lr_schedule)
  lr_is_adaptive_kl = lr_schedule == ppo_optimizer.LRSchedule.ADAPTIVE_KL
  if lr_is_adaptive_kl:
    base_optimizer = optax.inject_hyperparams(optax.adam)(
        learning_rate=learning_rate
    )
  if max_grad_norm is not None:
    # TODO(btaba): Move gradient clipping to `training/gradients.py`.
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        base_optimizer,
    )
  else:
    optimizer = base_optimizer

  loss_fn = functools.partial(
      ppo_losses.compute_ppo_loss,
      ppo_network=ppo_network,
      entropy_cost=entropy_cost,
      discounting=discounting,
      reward_scaling=reward_scaling,
      gae_lambda=gae_lambda,
      clipping_epsilon=clipping_epsilon,
      normalize_advantage=normalize_advantage,
      vf_coefficient=vf_loss_coefficient,
      clipping_epsilon_value=clipping_epsilon_value,
      use_distributional_critic=use_distributional_critic,
  )

  loss_and_pgrad_fn = gradients.loss_and_pgrad(
      loss_fn, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
  )

  steps_between_logging = training_metrics_steps or env_step_per_training_step
  metrics_aggregator = metric_logger.EpisodeMetricsLogger(
      steps_between_logging=steps_between_logging,
      progress_fn=progress_fn,
  )

  def minibatch_step(
      carry,
      data: types.Transition,
      normalizer_params: running_statistics.RunningStatisticsState,
  ):
    optimizer_state, params, key = carry
    key, key_loss = jax.random.split(key)
    (_, metrics), grads = loss_and_pgrad_fn(
        params, normalizer_params, data, key_loss
    )

    if lr_is_adaptive_kl:
      kl_mean = metrics['kl_mean']
      kl_mean = jax.lax.pmean(kl_mean, axis_name=_PMAP_AXIS_NAME)
      optimizer_state, lr = ppo_optimizer.adaptive_kl_learning_rate(
          optimizer_state, kl_mean, desired_kl,
          min_learning_rate=learning_rate_schedule_min_lr,
          max_learning_rate=learning_rate_schedule_max_lr,
      )
    else:
      lr = jnp.array(learning_rate)
    metrics['learning_rate'] = lr

    # apply gradients
    params_update, optimizer_state = optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(params, params_update)

    return (optimizer_state, params, key), metrics

  def sgd_step(
      carry,
      unused_t,
      data: types.Transition,
      normalizer_params: running_statistics.RunningStatisticsState,
  ):
    optimizer_state, params, key = carry
    key, key_perm, key_grad = jax.random.split(key, 3)

    if augment_pixels:
      key, key_rt = jax.random.split(key)
      r_translate = functools.partial(_random_translate_pixels, key=key_rt)
      data = types.Transition(
          observation=r_translate(data.observation),  # pytype: disable=wrong-arg-types
          action=data.action,
          reward=data.reward,
          discount=data.discount,
          next_observation=r_translate(data.next_observation),  # pytype: disable=wrong-arg-types
          extras=data.extras,
      )

    def convert_data(x: jnp.ndarray):
      x = jax.random.permutation(key_perm, x)
      x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
      return x

    shuffled_data = jax.tree_util.tree_map(convert_data, data)
    (optimizer_state, params, _), metrics = jax.lax.scan(
        functools.partial(minibatch_step, normalizer_params=normalizer_params),
        (optimizer_state, params, key_grad),
        shuffled_data,
        length=num_minibatches,
    )

    return (optimizer_state, params, key), metrics

  def training_step(
      carry: Tuple[TrainingState, envs.State, PRNGKey], unused_t
  ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
    training_state, state, key = carry
    key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

    policy = make_policy((
        training_state.normalizer_params,
        training_state.params.policy,
        training_state.params.value,
    ))

    def f(carry, unused_t):
      current_state, current_key = carry
      current_key, next_key = jax.random.split(current_key)
      extra_fields = ['truncation', 'episode_metrics', 'episode_done']
      if bootstrap_on_timeout:
        extra_fields.append('time_out')
      next_state, data = acting.generate_unroll(
          env,
          current_state,
          policy,
          current_key,
          unroll_length,
          extra_fields=tuple(extra_fields),
      )
      return (next_state, next_key), data

    (state, _), data = jax.lax.scan(
        f,
        (state, key_generate_unroll),
        (),
        length=batch_size * num_minibatches // num_envs,
    )
    # Have leading dimensions (batch_size * num_minibatches, unroll_length)
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
    data = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
    )
    assert data.discount.shape[1:] == (unroll_length,)

    if bootstrap_on_timeout:  # bootstrap reward on timeout
      time_out = data.extras['state_extras']['time_out']
      value = data.extras['policy_extras']['value']
      data = types.Transition(
          observation=data.observation,
          action=data.action,
          reward=data.reward + discounting * time_out * value,
          discount=data.discount,
          next_observation=data.next_observation,
          extras=data.extras,
      )

    normalizer_params = training_state.normalizer_params
    if not lr_is_adaptive_kl:
      # Update normalization params before SGD for backwards compatibility.
      normalizer_params = running_statistics.update(
          normalizer_params,
          _remove_pixels(data.observation),
          pmap_axis_name=_PMAP_AXIS_NAME,
          until_count=normalize_until_count,
      )

    (optimizer_state, params, _), metrics = jax.lax.scan(
        functools.partial(
            sgd_step, data=data, normalizer_params=normalizer_params
        ),
        (training_state.optimizer_state, training_state.params, key_sgd),
        (),
        length=num_updates_per_batch,
    )

    if lr_is_adaptive_kl:
      # For adaptive KL, normalization params should be updated after SGD s.t.
      # old distribution outputs are valid for KL computation.
      normalizer_params = running_statistics.update(
          normalizer_params,
          _remove_pixels(data.observation),
          pmap_axis_name=_PMAP_AXIS_NAME,
          until_count=normalize_until_count,
      )

    new_training_state = TrainingState(
        optimizer_state=optimizer_state,
        params=params,
        normalizer_params=normalizer_params,
        env_steps=training_state.env_steps + env_step_per_training_step,
    )

    # ── Buffer stats ───────────────────────────────────────────────────────────
    # Fold into `metrics` so training_epoch's scan averages them over all
    # training steps in the epoch and they reach progress_fn at eval cadence —
    # no per-step host callbacks needed.
    #
    # episode_metrics / episode_done shape: (B*M, unroll_length).
    # We average each metric over completed-episode steps (episode_done==True);
    # fall back to the full-buffer mean if no episode finished this batch.
    ep_done_f = data.extras['state_extras']['episode_done'].astype(jnp.float32)
    n_done    = jnp.sum(ep_done_f)
    ep_means  = jax.tree_util.tree_map(
        lambda v: jnp.where(
            n_done > 0,
            jnp.sum(v * ep_done_f) / jnp.maximum(n_done, 1.0),
            jnp.mean(v),
        ),
        data.extras['state_extras']['episode_metrics'],
    )
    metrics = {
        **metrics,
        'buffer_mean_reward': jnp.mean(data.reward),
        'buffer_reward_std':  jnp.std(data.reward),
        **{f'buffer_{k}': v for k, v in ep_means.items()},
    }

    if log_training_metrics:  # log unroll episode metrics via EpisodeMetricsLogger
      jax.debug.callback(
          metrics_aggregator.update_episode_metrics,
          data.extras['state_extras']['episode_metrics'],
          data.extras['state_extras']['episode_done'],
          metrics,
      )

    step_outputs = (
        metrics,
        new_training_state.normalizer_params,
        new_training_state.params.policy,
    )
    if save_q_network:
      step_outputs = step_outputs + (new_training_state.params.value,)
    return (new_training_state, state, new_key), step_outputs

  def training_epoch(
      training_state: TrainingState, state: envs.State, key: PRNGKey
  ) -> Tuple[TrainingState, envs.State, Metrics, Any, Any, Any]:
    (training_state, state, _), step_outputs = jax.lax.scan(
        training_step,
        (training_state, state, key),
        (),
        length=num_training_steps_per_epoch,
    )
    if save_q_network:
      loss_metrics, norm_buf, policy_buf, value_buf = step_outputs
    else:
      loss_metrics, norm_buf, policy_buf = step_outputs
      value_buf = None
    loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
    return training_state, state, loss_metrics, norm_buf, policy_buf, value_buf

  training_epoch = jax.pmap(
      training_epoch,
      axis_name=_PMAP_AXIS_NAME,
      donate_argnums=(
          0,
          1,
      ),
  )

  # Note that this is NOT a pure jittable method.
  def training_epoch_with_timing(
      training_state: TrainingState, env_state: envs.State, key: PRNGKey
  ) -> Tuple[TrainingState, envs.State, Metrics, Any, Any, Any]:
    nonlocal training_walltime
    t = time.time()
    training_state, env_state = _strip_weak_type((training_state, env_state))
    result = training_epoch(training_state, env_state, key)
    training_state, env_state, metrics, norm_buf, policy_buf, value_buf = (
        _strip_weak_type(result)
    )

    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (
        num_training_steps_per_epoch
        * env_step_per_training_step
        * max(num_resets_per_eval, 1)
    ) / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()},
    }
    return training_state, env_state, metrics, norm_buf, policy_buf, value_buf  # pytype: disable=bad-return-type  # py311-upgrade

  # Initialize model params and training state.
  init_params = ppo_losses.PPONetworkParams(
      policy=ppo_network.policy_network.init(key_policy),
      value=ppo_network.value_network.init(key_value),
  )

  obs_shape = jax.tree_util.tree_map(
      lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs
  )
  training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
      optimizer_state=optimizer.init(init_params),  # pytype: disable=wrong-arg-types  # numpy-scalars
      params=init_params,
      normalizer_params=running_statistics.init_state(
          _remove_pixels(obs_shape),
          std_eps=normalize_observations_std_eps,
          mode=normalize_observations_mode,
      ),
      env_steps=types.UInt64(hi=0, lo=0),
  )

  if restore_checkpoint_path is not None:
    params = checkpoint.load(restore_checkpoint_path)
    value_params = params[2] if restore_value_fn else init_params.value
    training_state = training_state.replace(
        normalizer_params=params[0],
        params=training_state.params.replace(
            policy=params[1], value=value_params
        ),
    )

  if restore_params is not None:
    logging.info('Restoring TrainingState from `restore_params`.')
    value_params = restore_params[2] if restore_value_fn else init_params.value
    training_state = training_state.replace(
        normalizer_params=restore_params[0],
        params=training_state.params.replace(
            policy=restore_params[1], value=value_params
        ),
    )

  if num_timesteps == 0:
    _finalize_log_handler()
    return (
        make_policy,
        (
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        ),
        {},
    )

  training_state = _device_put_replicated(
      training_state, jax.local_devices()[:local_devices_to_use]
  )

  eval_env = _maybe_wrap_env(
      eval_env or environment,
      wrap_env,
      num_eval_envs,
      episode_length,
      action_repeat,
      device_count=1,  # eval on the host only
      key_env=eval_key,
      wrap_env_fn=wrap_env_fn,
      randomization_fn=randomization_fn,
  )
  evaluator = acting.Evaluator(
      eval_env,
      functools.partial(make_policy, deterministic=deterministic_eval),
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=eval_key,
  )

  training_metrics = {}
  training_walltime = 0
  current_step = 0

  # Capture initial (random) params so we can prepend them to the ckpt
  # buffer at end of training -- that way `ckpt[0]` is always the
  # untrained policy and `ckpt[K-1]` is the final trained policy.
  init_norm_params = jax.device_get(_unpmap(training_state.normalizer_params))
  init_policy_params = jax.device_get(_unpmap(training_state.params.policy))
  init_value_params = jax.device_get(_unpmap(training_state.params.value))

  # Per-epoch ckpt buffers (host-side). Each `training_epoch_with_timing`
  # returns a scan-stacked buffer of shape (num_training_steps_per_epoch, ...);
  # we concatenate buffers across epochs into a single dense list, then
  # subsample K evenly at the end. Lives on host because it can be
  # several hundred MB at default racing settings.
  norm_buffers: list = []
  policy_buffers: list = []
  value_buffers: list = []
  step_buffers: list = []

  # Run initial eval
  metrics = {}
  if process_id == 0 and num_evals > 1 and run_evals:
    metrics = evaluator.run_evaluation(
        _unpmap((
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        )),
        training_metrics={},
    )
    logging.info(metrics)
    progress_fn(0, metrics)

  # Run initial policy_params_fn (host 0 only — every host would otherwise
  # race on shared dashboard/checkpoint artifacts for step 0 on multi-host runs).
  if process_id == 0:
    params = _unpmap((
        training_state.normalizer_params,
        training_state.params.policy,
        training_state.params.value,
    ))
    policy_params_fn(current_step, make_policy, params)

  for it in range(num_evals_after_init):
    logging.info('starting iteration %s %s', it, time.time() - xt)

    epoch_start_step = current_step
    for _ in range(max(num_resets_per_eval, 1)):
      # optimization
      epoch_key, local_key = jax.random.split(local_key)
      epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
      (training_state, env_state, training_metrics, norm_buf, policy_buf,
       value_buf) = training_epoch_with_timing(
           training_state, env_state, epoch_keys
       )
      current_step = int(_unpmap(training_state.env_steps))

      # Pull this epoch's per-training-step ckpt buffer to host. Shape after
      # _unpmap and reshape: (num_training_steps_per_epoch, ...) per leaf.
      # Skip on non-host processes -- only host 0 saves ckpts.
      if process_id == 0 and checkpoint_logdir is not None:
        norm_buffers.append(jax.device_get(_unpmap(norm_buf)))
        policy_buffers.append(jax.device_get(_unpmap(policy_buf)))
        if save_q_network:
          value_buffers.append(jax.device_get(_unpmap(value_buf)))
        steps_this_epoch = (
            epoch_start_step
            + (jnp.arange(num_training_steps_per_epoch, dtype=jnp.int64) + 1)
            * env_step_per_training_step
        )
        step_buffers.append(np.asarray(steps_this_epoch))

      key_envs = jax.vmap(
          lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
      )(key_envs, key_envs.shape[1])
      # TODO(brax-team): move extra reset logic to the AutoResetWrapper.
      if num_resets_per_eval > 0:
        env_state = reset_fn(env_state, key_envs)

    if process_id != 0:
      continue

    # Process id == 0.
    params = _unpmap((
        training_state.normalizer_params,
        training_state.params.policy,
        training_state.params.value,
    ))

    policy_params_fn(current_step, make_policy, params)

    if save_checkpoint_path is not None:
      ckpt_config = checkpoint.network_config(
          observation_size=obs_shape,
          action_size=env.action_size,
          normalize_observations=normalize_observations,
          network_factory=network_factory,
      )
      checkpoint.save(
          save_checkpoint_path, current_step, params, ckpt_config
      )

    if num_evals > 0:
      metrics = training_metrics
      if run_evals:
        metrics = evaluator.run_evaluation(
            params,
            training_metrics,
        )
      logging.info(metrics)
      progress_fn(current_step, metrics)

  total_steps = current_step
  if not total_steps >= num_timesteps:
    _finalize_log_handler()
    raise AssertionError(
        f'Total steps {total_steps} is less than `num_timesteps`='
        f' {num_timesteps}.'
    )

  if process_id == 0 and checkpoint_logdir is not None and norm_buffers:
    import json as _json
    save_ckpt = True
    for _thr_k, _thr_v in ckpt_save_threshold.items():
      if _thr_k in metrics and metrics[_thr_k] < _thr_v:
        save_ckpt = False
        logging.info(
            'eval metric %s (%s) is below threshold %s, not saving checkpoints',
            _thr_k, metrics[_thr_k], _thr_v,
        )
        break

    if save_ckpt:
      # Concatenate per-epoch buffers (each leaf of shape
      # (num_training_steps_per_epoch, ...)) along the leading axis.
      norm_full = jax.tree_util.tree_map(
          lambda *xs: np.concatenate(xs, axis=0), *norm_buffers,
      )
      policy_full = jax.tree_util.tree_map(
          lambda *xs: np.concatenate(xs, axis=0), *policy_buffers,
      )
      steps_full = np.concatenate(step_buffers, axis=0)

      # Prepend the random-init ckpt at index 0 so subsampling can include it.
      norm_full = jax.tree_util.tree_map(
          lambda buf, init: np.concatenate([np.expand_dims(init, axis=0), buf], axis=0),
          norm_full, init_norm_params,
      )
      policy_full = jax.tree_util.tree_map(
          lambda buf, init: np.concatenate([np.expand_dims(init, axis=0), buf], axis=0),
          policy_full, init_policy_params,
      )
      steps_full = np.concatenate([np.array([0], dtype=np.int64), steps_full], axis=0)

      num_total_ckpts = int(steps_full.shape[0])
      K = int(min(num_checkpoints, num_total_ckpts))
      sample_indices = np.linspace(0, num_total_ckpts, K, dtype=np.int64, endpoint=False)
      norm_sampled = jax.tree_util.tree_map(lambda x: x[sample_indices], norm_full)
      policy_sampled = jax.tree_util.tree_map(lambda x: x[sample_indices], policy_full)
      steps_sampled = steps_full[sample_indices]

      make_dir(checkpoint_logdir)
      ckpt_path = os.path.join(checkpoint_logdir, 'ppo_params.pkl')
      save_pickle((norm_sampled, policy_sampled), ckpt_path)
      save_pickle(jax.device_get(metrics),
                  os.path.join(checkpoint_logdir, 'ppo_metrics.pkl'))
      save_json(
          {'num_ckpts': K, 'steps': [int(s) for s in steps_sampled.tolist()]},
          os.path.join(checkpoint_logdir, 'ckpt_steps.json'),
      )
      logging.info(
          'saved %d ckpts (sampled from %d) to %s',
          K, num_total_ckpts, ckpt_path,
      )

      if save_q_network and value_buffers:
        value_full = jax.tree_util.tree_map(
            lambda *xs: np.concatenate(xs, axis=0), *value_buffers,
        )
        value_full = jax.tree_util.tree_map(
            lambda buf, init: np.concatenate([np.expand_dims(init, axis=0), buf], axis=0),
            value_full, init_value_params,
        )
        value_sampled = jax.tree_util.tree_map(lambda x: x[sample_indices], value_full)
        value_path = os.path.join(checkpoint_logdir, 'ppo_value_params.pkl')
        save_pickle(value_sampled, value_path)
        logging.info('saved value network checkpoints to %s', value_path)

  # If there was no mistakes the training_state should still be identical on all
  # devices.
  pmap.assert_is_replicated(training_state)
  params = _unpmap((
      training_state.normalizer_params,
      training_state.params.policy,
      training_state.params.value,
  ))
  logging.info('total steps: %s', total_steps)
  pmap.synchronize_hosts()
  _finalize_log_handler()
  return (make_policy, params, metrics)
