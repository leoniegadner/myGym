import copy
import json
import os
from typing import Optional
import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from mbrl import util as mbrl_util
import mbrl.algorithms as mbrl_algorithms
from mbrl.algorithms import pets, mbpo, planet
import mbrl.planning as mbrl_planning
import mbrl.models as mbrl_models
from myGym.mbrl_mygym.policy_mimic import PetsPolicyMimic


# This wrapper mimics the SB3 API used in train/test: learn, predict, save, load.
class MBRLWrapper:
    def __init__(self, _policy_unused, env, model_logdir, arg_dict, algo_name=None, cfg_builder=None, **_unused_kwargs):
        self.env = env
        self.model_logdir = model_logdir
        self.arg_dict = copy.deepcopy(arg_dict)
        self.arg_dict.setdefault("pets_policy_mimic", False)
        self.arg_dict.setdefault("pets_mimic_use_policy", self.arg_dict.get("pets_policy_mimic", False))
        self.arg_dict.setdefault("pets_mimic_hidden_sizes", [256, 256])
        self.arg_dict.setdefault("pets_mimic_learning_rate", 1e-3)
        self.arg_dict.setdefault("pets_mimic_batch_size", 512)
        self.arg_dict.setdefault("pets_mimic_epochs", 10)
        self.arg_dict.setdefault("pets_mimic_val_split", 0.1)
        self.algo_name = algo_name or self.arg_dict.get("algo")
        self.cfg_builder = cfg_builder
        self.num_timesteps = 0
        self.replay_buffer = None
        self.dynamics_model = None  # set when train runs
        self.agent = None  # populated when learn() runs
        self.model_env = None
        self._agent_needs_reset = False
        self.model_train_metrics = {}
        self.policy_mimic: Optional[PetsPolicyMimic] = None
        self.policy_mimic_path = os.path.join(self.model_logdir, "pets_policy_mimic.pth")
        self.mimic_stats_path = os.path.join(self.model_logdir, "pets_policy_mimic_stats.json")
        self.use_mimic_policy = bool(self.arg_dict.get("pets_mimic_use_policy"))
        self.mimic_train_stats = {}
        self._parse_flat_obs = self._build_obs_parser()
        self._policy_source_logged = None  # remember which policy we reported using

    @staticmethod
    def _find_train_root(path: str) -> str:
        """Return directory containing train.json (current path or its parent)."""
        candidate = path or "."

        def has_train_json(p: str) -> bool:
            return os.path.isfile(os.path.join(p, "train.json"))

        if has_train_json(candidate):
            return candidate
        parent = os.path.dirname(candidate)
        if parent and has_train_json(parent):
            return parent
        return candidate

    @staticmethod
    def _find_checkpoint_dir(path: str) -> str:
        """Resolve which directory actually stores the saved MBRL artifacts."""
        candidate = path or "."
        base_name = os.path.basename(os.path.normpath(candidate))
        if os.path.isfile(os.path.join(candidate, "mbrl_status.json")):
            return candidate
        if base_name in ("best_model",) or base_name.startswith("steps_"):
            return candidate

        best_dir = os.path.join(candidate, "best_model")
        if os.path.isdir(best_dir):
            return best_dir

        steps_file = os.path.join(candidate, "trained_steps.txt")
        if os.path.isfile(steps_file):
            try:
                with open(steps_file, "r") as f:
                    step_lines = [ln.strip() for ln in f if ln.strip()]
                for ln in reversed(step_lines):
                    try:
                        step_val = int(ln)
                    except Exception:
                        continue
                    step_dir = os.path.join(candidate, f"steps_{step_val}")
                    if os.path.isdir(step_dir):
                        return step_dir
            except Exception:
                pass
        return candidate

    @classmethod
    def load(cls, path, env=None, device="cpu", algo_name=None, cfg_builder=None, **_unused_kwargs):
        if env is None:
            raise ValueError("Environment is required to load an MBRL model.")
        base_path = path if os.path.isdir(path) else os.path.dirname(path)
        train_root = cls._find_train_root(base_path)
        ckpt_dir = cls._find_checkpoint_dir(base_path)
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = train_root
        cfg_path = os.path.join(train_root, "train.json")
        loaded_args = {}
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r") as f:
                loaded_args = json.load(f)
        instance = cls(None, env, train_root, loaded_args or {}, algo_name, cfg_builder)
        instance._restore_trained_components(load_dir=ckpt_dir)
        return instance

    def predict(self, observation, deterministic=False):
        # Use the trained MBRL agent when available; otherwise default to random.
        agent = getattr(self, "agent", None)
        if agent is None:
            self._log_policy_source("random", "no agent available")
            return self.env.action_space.sample(), None

        try:
            obs_np = observation
            if isinstance(obs_np, torch.Tensor):
                obs_np = obs_np.detach().cpu().numpy()

            if self.use_mimic_policy and self.policy_mimic is not None:
                try:
                    action = self.policy_mimic.predict(obs_np)
                    action = np.asarray(action)
                    if action.ndim > 1:
                        action = action.squeeze(0)
                    self._log_policy_source("mimic", f"using {os.path.basename(self.policy_mimic_path)}")
                    return action, None
                except Exception:
                    # fall back to planner if mimic forward fails
                    self._log_policy_source("planner", "mimic inference failed, falling back")

            # Reset planner once after training so it starts from a clean state for eval.
            if self._agent_needs_reset and hasattr(agent, "reset"):
                try:
                    agent.reset()
                finally:
                    self._agent_needs_reset = False

            # MBPO's SAC agent accepts sample/batched flags; PETS/Planet trajectory optimizer ignores them.
            if hasattr(agent, "sac_agent"):  # SACAgent wrapper
                action = agent.act(obs_np, sample=not deterministic, batched=False)
            else:
                action = agent.act(obs_np)
            self._log_policy_source("planner", "MBRL planner policy")
            return action, None
        except Exception:
            # If anything goes wrong, avoid breaking the run and fall back to random.
            self._log_policy_source("random", "error in predict, sampling action space")
            return self.env.action_space.sample(), None

    def save(self, path, steps=None, best=False, **kwargs):
        base_dir = path
        os.makedirs(base_dir, exist_ok=True)
        if steps is not None:
            try:
                with open(os.path.join(base_dir, "trained_steps.txt"), "a") as f:
                    f.write(f"{steps}\n")
            except Exception:
                pass

        if best:
            target_dir = os.path.join(base_dir, "best_model")
        elif steps is not None:
            target_dir = os.path.join(base_dir, f"steps_{steps}")
        else:
            target_dir = base_dir

        os.makedirs(target_dir, exist_ok=True)
        if self.dynamics_model is not None:
            try:
                self.dynamics_model.save(target_dir)
            except Exception:
                pass
        if self.replay_buffer is not None:
            try:
                self.replay_buffer.save(target_dir)
            except Exception:
                pass
        if self.policy_mimic is not None:
            try:
                mimic_path = os.path.join(target_dir, os.path.basename(self.policy_mimic_path))
                self.policy_mimic.save(mimic_path)
            except Exception:
                pass
        if self.mimic_train_stats:
            try:
                with open(os.path.join(target_dir, os.path.basename(self.mimic_stats_path)), "w") as f:
                    json.dump(self.mimic_train_stats, f, indent=2)
            except Exception:
                pass
        with open(os.path.join(target_dir, "mbrl_status.json"), "w") as f:
            json.dump({"algo": self.algo_name, "num_timesteps": steps if steps is not None else self.num_timesteps}, f)

    def _setup_callback(self, callback):
        """Align callback handling with SB3: always operate on a CallbackList."""
        if callback is None:
            return None
        if isinstance(callback, list):
            callback = CallbackList(callback)
        elif not isinstance(callback, BaseCallback):
            callback = CallbackList([callback])
        callback.init_callback(self)
        return callback

    def _model_train_epoch_cb(self, model, train_iter, epoch, train_loss, eval_score, best_val_score):
        """Capture latest dynamics model metrics so eval callback can report them."""
        # eval_score and best_val_score are tensors per ensemble member; reduce to mean scalars.
        def _to_float(val):
            if val is None:
                return None
            try:
                return float(val.mean().item() if hasattr(val, "mean") else val)
            except Exception:
                try:
                    return float(np.mean(val))
                except Exception:
                    return None

        self.model_train_metrics = {
            "train_iteration": int(train_iter),
            "epoch": int(epoch),
            "train_loss": _to_float(train_loss),
            "val_score": _to_float(eval_score),
            "best_val_score": _to_float(best_val_score),
        }

    def _build_obs_parser(self):
        """Creates a parser that converts flattened observations back to the env-style dict."""
        obs_cfg = self.arg_dict.get("observation", {}) or {}
        actual_key = obs_cfg.get("actual_state")
        goal_key = obs_cfg.get("goal_state")
        additional_keys = obs_cfg.get("additional_obs") or []

        # Minimal length map matching flatten_obs order: actual -> goal -> additional_obs (in list order).
        length_map = {"obj_xyz": 3, "obj_6D": 7, "endeff_xyz": 3, "endeff_6D": 7}
        try:
            actual_len = length_map[actual_key]
            goal_len = length_map[goal_key]
            additional_lens = [length_map[k] for k in additional_keys]
        except Exception as exc:
            raise ValueError(
                f"MBRLWrapper: unsupported observation component for reward parsing: {exc}"
            ) from exc

        def parser(flat):
            data = flat
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            data = np.asarray(data)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            idx = 0
            actual = data[:, idx : idx + actual_len]
            idx += actual_len
            goal = data[:, idx : idx + goal_len]
            idx += goal_len
            additional = {}
            for key, length in zip(additional_keys, additional_lens):
                additional[key] = data[:, idx : idx + length]
                idx += length

            obs_list = []
            for i in range(data.shape[0]):
                obs_list.append(
                    {
                        "actual_state": actual[i].tolist(),
                        "goal_state": goal[i].tolist(),
                        "additional_obs": {k: v[i].tolist() for k, v in additional.items()},
                    }
                )
            return obs_list

        return parser

    def _build_model_fns(self):
        """Build termination and reward functions used both for training and loading."""
        try:
            from mbrl.env import termination_fns
        except Exception as exc:
            raise ImportError("mbrl library is required for model-based algorithms.") from exc

        base_term_fn = termination_fns.no_termination

        def term_fn(act, next_obs):
            # PETS expects batched obs; guard against unbatched or extra dims
            if isinstance(next_obs, torch.Tensor):
                if next_obs.ndim != 2:
                    next_obs = next_obs.reshape(-1, next_obs.shape[-1])
            else:
                try:
                    if isinstance(next_obs, np.ndarray) and next_obs.ndim != 2:
                        next_obs = next_obs.reshape(-1, next_obs.shape[-1])
                except Exception:
                    pass
            return base_term_fn(act, next_obs)

        def env_reward_fn(act, next_obs):
            obs_dicts = self._parse_flat_obs(next_obs)
            rewards = []
            for obs_d in obs_dicts:
                if hasattr(self.env.unwrapped.reward, "compute_planning"):
                    rew = self.env.unwrapped.reward.compute_planning(observation=obs_d)
                else:
                    # Mark that we are evaluating on model-predicted observations to avoid mutating real env state
                    self.env.in_model_rollout = True
                    try:
                        rew = self.env.unwrapped.reward.compute(observation=obs_d)
                    finally:
                        self.env.in_model_rollout = False
                rewards.append(float(rew))
            device = act.device if isinstance(act, torch.Tensor) else "cpu"
            return torch.tensor(rewards, device=device, dtype=torch.float32).view(-1, 1)

        return term_fn, env_reward_fn

    def _get_mimic_hparams(self):
        hidden_sizes = self.arg_dict.get("pets_mimic_hidden_sizes", [256, 256])
        if isinstance(hidden_sizes, (int, float)):
            hidden_sizes = [int(hidden_sizes)]
        elif isinstance(hidden_sizes, str):
            try:
                hidden_sizes = [int(x.strip()) for x in hidden_sizes.split(",") if x.strip()]
            except Exception:
                hidden_sizes = [256, 256]

        return {
            "hidden_sizes": hidden_sizes,
            "lr": float(self.arg_dict.get("pets_mimic_learning_rate", 1e-3)),
            "batch_size": int(self.arg_dict.get("pets_mimic_batch_size", 512)),
            "epochs": int(self.arg_dict.get("pets_mimic_epochs", 10)),
            "val_split": float(self.arg_dict.get("pets_mimic_val_split", 0.1)),
            "device": self.arg_dict.get("device", "cpu"),
        }

    def _train_policy_mimic(self):
        if self.algo_name != "pets" or not self.arg_dict.get("pets_policy_mimic"):
            return None
        if self.replay_buffer is None:
            return None

        try:
            obs_shape = getattr(self.replay_buffer, "obs", None).shape[1:]
            act_shape = getattr(self.replay_buffer, "action", None).shape[1:]
            obs_dim = int(np.prod(obs_shape))
            act_dim = int(np.prod(act_shape))
        except Exception:
            return None

        hparams = self._get_mimic_hparams()
        self.policy_mimic = PetsPolicyMimic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hparams["hidden_sizes"],
            lr=hparams["lr"],
            val_split=hparams["val_split"],
            device=hparams["device"],
        )
        stats = self.policy_mimic.train_from_replay(
            self.replay_buffer,
            batch_size=hparams["batch_size"],
            epochs=hparams["epochs"],
        )
        self.mimic_train_stats = stats or {}
        try:
            self.policy_mimic.save(self.policy_mimic_path)
        except Exception:
            pass
        try:
            with open(self.mimic_stats_path, "w") as f:
                json.dump(self.mimic_train_stats, f, indent=2)
        except Exception:
            pass
        # Enable mimic usage only when explicitly requested.
        self.use_mimic_policy = bool(self.arg_dict.get("pets_mimic_use_policy"))
        return stats

    def _maybe_load_policy_mimic(self, obs_shape, act_shape, load_dir=None):
        if not (self.arg_dict.get("pets_policy_mimic") or self.arg_dict.get("pets_mimic_use_policy")):
            return
        load_dir = load_dir or self.model_logdir
        mimic_path = os.path.join(load_dir, os.path.basename(self.policy_mimic_path))
        stats_path = os.path.join(load_dir, os.path.basename(self.mimic_stats_path))
        if not os.path.isfile(mimic_path):
            return
        try:
            obs_dim = int(np.prod(obs_shape))
            act_dim = int(np.prod(act_shape))
            hparams = self._get_mimic_hparams()
            self.policy_mimic = PetsPolicyMimic(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_sizes=hparams["hidden_sizes"],
                lr=hparams["lr"],
                val_split=hparams["val_split"],
                device=hparams["device"],
            )
            self.policy_mimic.load(mimic_path)
            if os.path.isfile(stats_path):
                try:
                    with open(stats_path, "r") as f:
                        self.mimic_train_stats = json.load(f)
                except Exception:
                    self.mimic_train_stats = {}
        except Exception:
            self.policy_mimic = None

    def _restore_trained_components(self, load_dir=None):
        """Load dynamics model, replay buffer, and planner from a saved logdir."""
        if self.cfg_builder is None:
            return

        load_dir = load_dir or self.model_logdir
        cfg = self.cfg_builder(self.arg_dict)
        # honor saved status if present
        status_path = os.path.join(load_dir, "mbrl_status.json")
        if os.path.isfile(status_path):
            try:
                with open(status_path, "r") as f:
                    status = json.load(f)
                self.num_timesteps = status.get("num_timesteps", cfg.overrides.num_steps)
            except Exception:
                self.num_timesteps = cfg.overrides.num_steps
        else:
            self.num_timesteps = cfg.overrides.num_steps

        term_fn, env_reward_fn = self._build_model_fns()
        obs_shape = self.env.observation_space.shape
        act_shape = self.env.action_space.shape
        rng = np.random.default_rng(cfg.get("seed", None))

        # Restore replay buffer if available.
        try:
            self.replay_buffer = mbrl_util.common.create_replay_buffer(
                cfg, obs_shape, act_shape, load_dir=load_dir, rng=rng
            )
        except Exception:
            self.replay_buffer = None

        # Restore dynamics model.
        try:
            self.dynamics_model = mbrl_util.common.create_one_dim_tr_model(
                cfg, obs_shape, act_shape, model_dir=load_dir
            )
            self.dynamics_model.load(load_dir)
        except Exception:
            self.dynamics_model = None

        if self.dynamics_model is None:
            return

        torch_generator = torch.Generator(device=cfg.get("device", "cpu"))
        if cfg.get("seed", None) is not None:
            torch_generator.manual_seed(cfg.seed)

        reward_fn = env_reward_fn if self.algo_name != "mbpo" else None
        self.model_env = mbrl_models.ModelEnv(
            self.env, self.dynamics_model, term_fn, reward_fn, generator=torch_generator
        )

        # Recreate planner/agent
        try:
            if self.algo_name in ("pets", "planet"):
                self.agent = mbrl_planning.create_trajectory_optim_agent_for_model(
                    self.model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles
                )
            elif self.algo_name == "mbpo":
                sac_ckpt = os.path.join(load_dir, "sac.pth")
                if os.path.isfile(sac_ckpt):
                    # Instantiate the SAC agent and load checkpoint.
                    from mbrl.planning.sac_wrapper import SACAgent
                    import hydra.utils

                    sac = hydra.utils.instantiate(cfg.algorithm.agent)
                    sac.load_checkpoint(sac_ckpt, evaluate=True)
                    self.agent = SACAgent(sac)
        except Exception:
            self.agent = None

        self._maybe_load_policy_mimic(obs_shape, act_shape, load_dir=load_dir)
        self._agent_needs_reset = bool(self.agent)

    def learn(self, total_timesteps=None, callback=None):
        if total_timesteps is not None:
            self.arg_dict["steps"] = total_timesteps
        elif "steps" not in self.arg_dict:
            self.arg_dict["steps"] = 0

        cfg = self.cfg_builder(self.arg_dict)
        self.num_timesteps = cfg.overrides.num_steps

        try:
            from mbrl.algorithms import pets, mbpo, planet
        except Exception as exc:
            raise ImportError("mbrl library is required for model-based algorithms.") from exc

        term_fn, env_reward_fn = self._build_model_fns()

        # SB3-style callback integration: wrap env.step to call callback.on_step()
        wrapped_env = self.env
        callback = self._setup_callback(callback)
        if callback is not None:
            self.num_timesteps = 0

            class _StepCallbackEnv:
                def __init__(self, env, cb, model):
                    self.env = env
                    self.cb = cb
                    self.model = model

                def reset(self, *args, **kwargs):
                    return self.env.reset(*args, **kwargs)

                def step(self, action):
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    # Keep SB3 counters in sync
                    self.model.num_timesteps += 1
                    if hasattr(self.cb, "num_timesteps"):
                        self.cb.num_timesteps = self.model.num_timesteps

                    # Trigger SB3-style callback (__call__ handles n_calls and eval_freq)
                    stop_training = False
                    if callable(self.cb):
                        stop_training = not self.cb()
                    elif hasattr(self.cb, "on_step"):
                        stop_training = not self.cb.on_step()
                    # PETS has no built-in early stop; ignore stop_training but could be logged
                    return obs, reward, terminated, truncated, info

                def __getattr__(self, name):
                    return getattr(self.env, name)

            wrapped_env = _StepCallbackEnv(self.env, callback, self)
            if hasattr(callback, "on_training_start"):
                callback.on_training_start(locals(), globals())

        orig_create_rb = mbrl_util.common.create_replay_buffer
        orig_create_model = mbrl_util.common.create_one_dim_tr_model
        orig_rollout_fn = getattr(mbrl_algorithms.mbpo, "rollout_model_and_populate_sac_buffer", None)
        orig_train_model_and_save = mbrl_util.common.train_model_and_save_model_and_data
        orig_agent_factory = mbrl_planning.create_trajectory_optim_agent_for_model
        orig_mbpo_sac_agent = getattr(mbrl_algorithms.mbpo, "SACAgent", None)
        self.sac_buffer = None  # if you care about SAC buffer (MBPO)
        self.agent = None
        self.model_env = None
        self._agent_needs_reset = False

        def capture_rb(*args, **kwargs):
            self.replay_buffer = orig_create_rb(*args, **kwargs)
            return self.replay_buffer

        def capture_model(cfg, obs_shape, act_shape, model_dir=None):
            self.dynamics_model = orig_create_model(cfg, obs_shape, act_shape, model_dir)
            return self.dynamics_model

        def capture_agent_for_model(model_env, agent_cfg, num_particles=1):
            agent = orig_agent_factory(model_env, agent_cfg, num_particles)
            self.agent = agent
            self.model_env = model_env
            self._agent_needs_reset = True
            return agent

        # For MBPO's SAC agent, hook the constructor to keep a reference.
        if orig_mbpo_sac_agent is not None:
            outer_self = self

            class _CapturedSACAgent(orig_mbpo_sac_agent):  # type: ignore[misc, valid-type]
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    outer_self.agent = self
                    outer_self._agent_needs_reset = True

            mbrl_algorithms.mbpo.SACAgent = _CapturedSACAgent

        def capture_sac_buffer(model_env, replay_buffer, agent, sac_buffer, *args, **kwargs):
            self.sac_buffer = sac_buffer
            return orig_rollout_fn(model_env, replay_buffer, agent, sac_buffer, *args, **kwargs)

        def capture_train_model_and_save(model, model_trainer, cfg, replay_buffer, work_dir=None, callback=None):
            # Always record epoch metrics, while respecting any user callback passed.
            def chained_cb(*cb_args):
                self._model_train_epoch_cb(*cb_args)
                if callback is not None:
                    callback(*cb_args)

            return orig_train_model_and_save(
                model,
                model_trainer,
                cfg,
                replay_buffer,
                work_dir=work_dir,
                callback=chained_cb,
            )

        mbrl_util.common.create_replay_buffer = capture_rb
        mbrl_util.common.create_one_dim_tr_model = capture_model
        mbrl_planning.create_trajectory_optim_agent_for_model = capture_agent_for_model
        mbrl_util.common.train_model_and_save_model_and_data = capture_train_model_and_save
        if orig_rollout_fn:
            mbrl_algorithms.mbpo.rollout_model_and_populate_sac_buffer = capture_sac_buffer
        try:
            if self.algo_name == "pets":
                out = pets.train(wrapped_env, term_fn, env_reward_fn, cfg, silent=True, work_dir=self.model_logdir)
            elif self.algo_name == "mbpo":
                # Use wrapped_env so callbacks (eval/save) are triggered the same way as PPO.
                out = mbpo.train(wrapped_env, wrapped_env, term_fn, cfg, silent=True, work_dir=self.model_logdir)
            else:
                out = planet.train(self.env, cfg, silent=True, work_dir=self.model_logdir)
        finally:
            mbrl_util.common.create_replay_buffer = orig_create_rb
            mbrl_util.common.create_one_dim_tr_model = orig_create_model
            mbrl_planning.create_trajectory_optim_agent_for_model = orig_agent_factory
            mbrl_util.common.train_model_and_save_model_and_data = orig_train_model_and_save
            if orig_rollout_fn:
                mbrl_algorithms.mbpo.rollout_model_and_populate_sac_buffer = orig_rollout_fn
            if orig_mbpo_sac_agent is not None:
                mbrl_algorithms.mbpo.SACAgent = orig_mbpo_sac_agent

            if callback is not None and hasattr(callback, "on_training_end"):
                callback.on_training_end()

        if self.algo_name == "pets" and self.arg_dict.get("pets_policy_mimic"):
            self._train_policy_mimic()

        return out

    def _log_policy_source(self, source: str, detail: str = ""):
        """Log once which policy source is being used for actions."""
        if self._policy_source_logged == source:
            return
        self._policy_source_logged = source
        msg = f"[MBRL] action source: {source}"
        if detail:
            msg += f" ({detail})"
        print(msg)
