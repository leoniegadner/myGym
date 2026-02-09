import json
import os
import time

import numpy as np
from stable_baselines3.common.callbacks import EventCallback

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

class MBRLEvalCallback(EventCallback):
    """
    Evaluates an MBRL agent (policy rollout + dynamics model error).
    - Computes mean/model-free return over n_eval_episodes.
    - Computes one-step model prediction MSE on a replay buffer batch if available.
    """
    def __init__(
        self,
        eval_env,
        log_path,
        eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        replay_buffer=None,
        verbose=0,
        starting_steps=0,
        num_cpu=1,
    ):
        super().__init__(callback=None, verbose=verbose)
        self.eval_env = eval_env
        self.log_path = log_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.replay_buffer = replay_buffer
        self.starting_steps = starting_steps
        self.evaluations_results = {}
        self.num_cpu = num_cpu or 1
        self.num_evals = 0
        self.tb_writer = None
        self.tb_log_dir = None

        if self.log_path is not None and SummaryWriter is not None:
            # Keep all MBRL scalar summaries alongside other run artifacts.
            self.tb_log_dir = os.path.join(self.log_path, "mbrl_tb")
            self.tb_writer = SummaryWriter(log_dir=self.tb_log_dir)

    def _log_tb_scalars(self, metrics: dict, step: int):
        """Log numeric metrics to TensorBoard if available."""
        if self.tb_writer is None:
            return
        for key, val in metrics.items():
            if val is None:
                continue
            try:
                self.tb_writer.add_scalar(key, float(val), step)
            except Exception:
                # Skip values that cannot be logged as scalars
                continue
        try:
            self.tb_writer.flush()
        except Exception:
            pass

    def _on_training_end(self):
        if self.tb_writer is not None:
            try:
                self.tb_writer.flush()
                self.tb_writer.close()
            except Exception:
                pass
        return super()._on_training_end()

    def _on_step(self) -> bool:
        # Run every eval_freq steps
        actual_calls = getattr(self.model, "num_timesteps", self.n_calls)
        try:
            actual_calls = int(actual_calls)
        except Exception:
            actual_calls = self.n_calls
        actual_calls = actual_calls * self.num_cpu + self.starting_steps

        if self.eval_freq < 1 or actual_calls < self.eval_freq * self.num_evals:
            return True

        self.num_evals += 1
        start_timer = time.time()
        if self.verbose > 0:
            print("---Evaluation----")
        step_count = actual_calls
        returns = []
        lengths = []
        success_episodes_num = 0
        distance_error_sum = 0
        steps_sum = 0
        episode_rewards = []
        subrewards = []
        subrewsteps = []
        subrewsuccess = []

        def _get_env_attr(name):
            # Prefer wrapper-friendly access without using deprecated get_attr.
            if hasattr(self.eval_env, "get_wrapper_attr"):
                try:
                    return self.eval_env.get_wrapper_attr(name)
                except Exception:
                    pass
            base = getattr(self.eval_env, "unwrapped", None)
            if base is not None and hasattr(base, name):
                return getattr(base, name)
            return getattr(self.eval_env, name, None)

        env_reward = _get_env_attr("reward")
        env_task = _get_env_attr("task")

        for _ in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            ep_len = 0
            steps = 0
            last_info = {}
            last_terminated = False
            last_truncated = False
            distance_error = 0
            last_network = 0
            last_steps = 0
            srewardsteps = np.zeros(env_reward.num_networks)
            srewardsuccess = np.zeros(env_reward.num_networks)
            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1
                steps_sum += 1
                ep_len += 1
                last_info = info if isinstance(info, dict) else {}
                last_terminated = terminated
                last_truncated = truncated

                current_network = getattr(env_reward, "current_network", 0)
                if current_network != last_network:
                    if not done:
                        srewardsteps.put([last_network], steps - last_steps)
                        srewardsuccess.put([last_network], 1)
                        last_network = current_network
                        last_steps = steps

                distance_error = env_reward.get_distance_error(info.get("o", {})) if isinstance(info, dict) and "o" in info else 0

            srewardsteps.put([last_network], steps - last_steps)
            is_successful = int(last_terminated and not last_truncated and not last_info.get("f", False))
            if is_successful:
                srewardsuccess.put([last_network], 1)
            subrewards.append(getattr(env_reward, "eval_network_rewards", env_reward.network_rewards))
            subrewsteps.append(srewardsteps)
            subrewsuccess.append(srewardsuccess)
            episode_rewards.append(episode_reward)
            success_episodes_num += is_successful
            distance_error_sum += distance_error
            returns.append(episode_reward)
            lengths.append(ep_len)

        meansr = np.mean(subrewards, axis=0) if len(subrewards) > 0 else 0
        meansrs = np.mean(subrewsteps, axis=0) if len(subrewsteps) > 0 else 0
        srsu = np.array(subrewsuccess) if len(subrewsuccess) > 0 else np.array([])
        meansgoals = np.count_nonzero(srsu) / env_reward.num_networks / self.n_eval_episodes * 100 if srsu.size > 0 else 0

        success_rate = success_episodes_num / self.n_eval_episodes * 100 if self.n_eval_episodes else 0
        mean_distance_error = distance_error_sum / self.n_eval_episodes if self.n_eval_episodes else 0
        mean_steps_num = steps_sum // self.n_eval_episodes if self.n_eval_episodes else 0
        mean_subgoal_reward = float(np.mean(meansr)) if np.size(meansr) > 0 else 0.0
        mean_subgoal_steps = float(np.mean(meansrs)) if np.size(meansrs) > 0 else 0.0

        # Use eval/ prefix for standardized TensorBoard metrics (comparable across algorithms)
        metrics = {
            "eval/mean_reward": float(np.mean(returns)),
            "eval/std_reward": float(np.std(returns)),
            "eval/mean_ep_length": float(np.mean(lengths)),
            "eval/success_rate": float(success_rate),
            "eval/mean_distance_to_goal": float(mean_distance_error),
            "eval/mean_subgoals_finished": float(meansgoals),
        }

        results = {
            "episode": f"{step_count}",
            "n_eval_episodes": f"{self.n_eval_episodes}",
            "success_episodes_num": f"{success_episodes_num}",
            "success_rate": f"{success_rate}",
            "mean_distance_error": f"{mean_distance_error:.2f}",
            "mean_steps_num": f"{mean_steps_num}",
            "mean_reward": f"{np.mean(episode_rewards):.2f}" if len(episode_rewards) > 0 else "0.00",
            "std_reward": f"{np.std(episode_rewards):.2f}" if len(episode_rewards) > 0 else "0.00",
            "number of tasks": f"{env_task.number_tasks}",
            "number of networks": f"{env_reward.num_networks}",
            "mean subgoals finished": f"{meansgoals}",
            "mean subgoal reward": f"{meansr}",
            "mean subgoal steps": f"{meansrs}",
        }

        # Optional: model prediction error if replay buffer + model_eval available
        model_err = None
        if self.replay_buffer is not None and hasattr(self.model, "dynamics_model"):
            batch = self.replay_buffer.sample(min(256, len(self.replay_buffer)))
            # MBRL TransitionBatch uses 'obs', 'action', 'next_obs' attribute names
            obs = getattr(batch, 'obs', getattr(batch, 'observations', None))
            act = getattr(batch, 'action', getattr(batch, 'actions', None))
            next_obs = getattr(batch, 'next_obs', getattr(batch, 'next_observations', None))
            if obs is not None and act is not None and next_obs is not None:
                with np.errstate(all="ignore"):
                    pred_next, _ = self.model.dynamics_model.predict(obs, act, deterministic=True)
                    model_err = float(np.mean((pred_next - next_obs) ** 2))
                    metrics["train/model_mse"] = model_err

        model_train_metrics = getattr(self.model, "model_train_metrics", {}) or {}
        if model_train_metrics:
            results.update({
                "dynamics_train_loss": model_train_metrics.get("train_loss"),
                "dynamics_val_score": model_train_metrics.get("val_score"),
                "dynamics_best_val_score": model_train_metrics.get("best_val_score"),
                "dynamics_train_iteration": model_train_metrics.get("train_iteration"),
                "dynamics_epoch": model_train_metrics.get("epoch"),
            })
            metrics.update({
                "train/dynamics_train_loss": model_train_metrics.get("train_loss"),
                "train/dynamics_val_score": model_train_metrics.get("val_score"),
                "train/dynamics_best_val_score": model_train_metrics.get("best_val_score"),
            })

        self._log_tb_scalars(metrics, step_count)

        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)
            eval_key = f"evaluation_after_{step_count}_steps"
            self.evaluations_results[eval_key] = results
            with open(os.path.join(self.log_path, "evaluation_results.json"), "w") as f:
                json.dump(self.evaluations_results, f, indent=4)

            serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in {**results, **metrics}.items()}
            log_payload = {"step": step_count, **serializable}
            with open(os.path.join(self.log_path, "mbrl_eval.txt"), "a") as f:
                f.write(json.dumps(log_payload) + "\n")

        if self.verbose > 0:
            for k, v in results.items():
                print(k, ":", v)
            if model_err is not None:
                print("dynamics_model_mse :", model_err)
            if model_train_metrics:
                print("Dynamics model stats:")
                dyn_items = {k: v for k, v in results.items() if k.startswith("dynamics_")}
                for k, v in dyn_items.items():
                    label = k.replace("dynamics_", "", 1)
                    print(" ", label, ":", v)
            print("Evaluation finished successfully")
            print("evaluation time:", time.time() - start_timer)

        self.eval_env.reset()

        return True
