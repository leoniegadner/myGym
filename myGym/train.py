import multiprocessing
import subprocess
import sys
import warnings

import commentjson
import copy
import json
import os
import random
import time
from typing import Any, Callable, Dict, Optional


import numpy as np
import importlib.resources as pkg_resources
import os, sys, time, yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json, commentjson
import gymnasium as gym
from myGym.active_inference import ActiveInferenceSB3, MetaAIFSB3, MetaAIFActionWrapper
from sklearn.model_selection import ParameterGrid
from myGym.envs.gym_env import GymEnv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

try:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.callbacks import BaseCallback
except Exception as e:
    print(e)

try:
    from stable_baselines3 import A2C as A2C_P, SAC as SAC_P, TD3 as TD3_P, PPO
except:
    print("Torch isn't probably installed correctly")

from myGym.mbrl_mygym.mbrl_wrapper import MBRLWrapper
from myGym.utils.callbacksMBRL import MBRLEvalCallback
from omegaconf import OmegaConf


# Import helper classes and functions for monitoring
from myGym.utils.callbacksSB3 import SaveOnBestTrainingRewardCallback, MultiPPOEvalCallback, PPOEvalCallback
from myGym.envs.natural_language import NaturalLanguage
from myGym.stable_baselines_mygym.multi_ppo_SB3 import MultiPPOSB3
from myGym.stable_baselines_mygym.ppoSB3 import PPO as PPO_P
from myGym.stable_baselines_mygym.Subproc_vec_envSB3 import SubprocVecEnv
from myGym.utils.save_utils import uses_sb3_style

# This is a global variable for the type of engine we are working with
AVAILABLE_SIMULATION_ENGINES = ["pybullet"]
AVAILABLE_TRAINING_FRAMEWORKS = ["pytorch"]


def log_action_and_joint_info(env):
    """
    Print action space bounds and joint limits to help verify scaling.
    """
    try:
        base_env = env
        if hasattr(base_env, "envs"):  # VecEnv (e.g., SubprocVecEnv/DummyVecEnv)
            base_env = base_env.envs[0]
        if hasattr(base_env, "env"):  # Monitor wrapper
            base_env = base_env.env
        unwrapped = getattr(base_env, "unwrapped", base_env)
        robot = getattr(unwrapped, "robot", None)

        print("Action space low:", unwrapped.action_space.low)
        print("Action space high:", unwrapped.action_space.high)
        if robot is not None and hasattr(robot, "joints_limits"):
            print("Joint limits lower:", robot.joints_limits[0])
            print("Joint limits upper:", robot.joints_limits[1])
        if robot is not None and hasattr(robot, "gjoints_limits"):
            print("Gripper limits lower:", robot.gjoints_limits[0])
            print("Gripper limits upper:", robot.gjoints_limits[1])
    except Exception as exc:
        print(f"Could not log action/joint info: {exc}")


class ActionLoggerCallback(BaseCallback):
    """
    Periodically logs action statistics to the SB3 logger so we can verify scaling.
    """
    def __init__(self, log_every=1000, to_stdout=False, **kwargs):
        super().__init__(**kwargs)
        self.log_every = max(1, log_every)
        self.to_stdout = to_stdout
        self.joint_dim = None  # number of non-gripper action dims

    def _compute_joint_dim(self):
        """
        Derive how many action dims belong to joints (exclude gripper dims).
        """
        try:
            env = getattr(self, "training_env", None)
            if env is None:
                return
            base_env = env
            if hasattr(base_env, "envs"):
                base_env = base_env.envs[0]
            if hasattr(base_env, "env"):
                base_env = base_env.env
            unwrapped = getattr(base_env, "unwrapped", base_env)
            robot = getattr(unwrapped, "robot", None)

            gripper_dim = 0
            if robot is not None:
                if hasattr(robot, "gjoints_num"):
                    gripper_dim = robot.gjoints_num
                elif hasattr(robot, "gjoints_limits"):
                    gripper_dim = len(robot.gjoints_limits[0])

            action_dim = unwrapped.action_space.shape[0]
            self.joint_dim = max(action_dim - gripper_dim, 0)
        except Exception:
            self.joint_dim = None

    def _on_step(self) -> bool:
        if self.n_calls % self.log_every != 0:
            return True

        actions = self.locals.get("actions")
        if actions is None:
            return True

        if self.joint_dim is None:
            self._compute_joint_dim()

        actions_np = actions.detach().cpu().numpy() if hasattr(actions, "detach") else np.asarray(actions)
        joint_actions = actions_np[..., :self.joint_dim] if self.joint_dim else actions_np

        self.logger.record("debug/action_mean", joint_actions.mean())
        self.logger.record("debug/action_min", joint_actions.min())
        self.logger.record("debug/action_max", joint_actions.max())
        if self.to_stdout:
            print(
                f"[ActionLogger] joints-only mean={joint_actions.mean():.3f} "
                f"min={joint_actions.min():.3f} max={joint_actions.max():.3f}"
            )
        return True


def save_results(arg_dict, model_name, env, model_logdir=None, show=False):
    if model_logdir is None:
        model_logdir = arg_dict["logdir"]
    print(f"model_logdir: {model_logdir}")
    print("Congratulations! Training with {} timesteps succeed!".format(arg_dict["steps"]))


def configure_env(arg_dict, model_logdir=None, for_train=True):
    env_arguments = {"render_on": True, "visualize": arg_dict["visualize"], "workspace": arg_dict["workspace"],
                     "robot": arg_dict["robot"], "robot_init_joint_poses": arg_dict["robot_init"],
                     "robot_action": arg_dict["robot_action"], "max_velocity": arg_dict["max_velocity"],
                     "max_force": arg_dict["max_force"], "task_type": arg_dict["task_type"],
                     "action_repeat": arg_dict["action_repeat"],
                     "task_objects": arg_dict["task_objects"], "observation": arg_dict["observation"],
                     "framework": "SB3",
                     "distractors": arg_dict["distractors"],
                     "moving_target": arg_dict.get("moving_target", None),
                     "num_networks": arg_dict.get("num_networks", 1),
                     "network_switcher": arg_dict.get("network_switcher", "gt"),
                     "distance_type": arg_dict["distance_type"], "used_objects": arg_dict["used_objects"],
                     "active_cameras": arg_dict["camera"], "color_dict": arg_dict.get("color_dict", {}),
                     "visgym": arg_dict["visgym"],
                     "reward": arg_dict["reward"], "logdir": arg_dict["logdir"], "vae_path": arg_dict["vae_path"],
                     "yolact_path": arg_dict["yolact_path"], "yolact_config": arg_dict["yolact_config"],
                     "natural_language": bool(arg_dict["natural_language"]),
                     "training": bool(for_train), "top_grasp": arg_dict["top_grasp"],
                     "max_ep_steps": arg_dict["max_episode_steps"],
                     "gui_on": arg_dict["gui"],                     
                     }

    if "network_switcher" in arg_dict.keys():
        env_arguments["network_switcher"] = arg_dict["network_switcher"]
    if arg_dict["algo"] == "her":
        env = gym.make(arg_dict["env_name"], **env_arguments, obs_space="dict")  # her needs obs as a dict
    else:
        env = gym.make(arg_dict["env_name"], **env_arguments)
        env.spec.max_episode_steps = 512

    if for_train:
        if arg_dict["engine"] == "mujoco":
            env = VecMonitor(env, model_logdir) if arg_dict["multiprocessing"] else Monitor(env, model_logdir)
        elif arg_dict["engine"] == "pybullet" and not arg_dict["multiprocessing"]:
            env = Monitor(env, filename=model_logdir, info_keywords=tuple('d'))

    if arg_dict["algo"] == "her":
        env = HERGoalEnvWrapper(env)
    return env


def make_env(arg_dict: dict, rank: int, seed: int = 0, model_logdir=None) -> Callable:
    """
        Utility function for multiprocessed env.

        :param arg_dict: (dict) the environment ID
        :param seed: (int) the initial seed for RNG
        :param rank: (int) index of the subprocess
        :return: (Callable)
        """

    def _init():
        if "Gym-v0" not in gym.registry:
            gym.register("Gym-v0", GymEnv)
        arg_dict["seed"] = seed + rank
        env = configure_env(arg_dict, for_train=True, model_logdir=model_logdir)
        env.reset()
        return env

    set_random_seed(seed)
    return _init


def configure_implemented_combos(env, model_logdir, arg_dict):
    implemented_combos = {
        "ppo": {},
        "sac": {},
        "td3": {},
        "a2c": {},
        "multippo": {},
        "pets": {},
        "mbpo": {},
        "planet": {},
        "aif": {},
        "meta_aif": {},
    }  # mbrl + active inference

    # populate action bounds for MBRL agents if available
    # TODO: move this somewhere less hacky.
    if env and hasattr(env, "action_space") and hasattr(env.action_space, "low"):
        arg_dict.setdefault("action_lb", env.action_space.low.tolist())
        arg_dict.setdefault("action_ub", env.action_space.high.tolist())

    sac_policy = arg_dict.get("sac_policy", "MlpPolicy")
    auto_ent = bool(arg_dict.get("sac_automatic_entropy_tuning", True))
    sac_kwargs = {
        "verbose": 1,
        "tensorboard_log": model_logdir,
        "learning_rate": arg_dict.get("sac_lr", 3e-4),
        "buffer_size": arg_dict.get("sac_buffer_size", 1_000_000),
        "learning_starts": arg_dict.get("sac_learning_starts", 100),
        "batch_size": arg_dict.get("sac_batch_size", 256),
        "tau": arg_dict.get("sac_tau", 0.005),
        "gamma": arg_dict.get("sac_gamma", 0.99),
        "train_freq": arg_dict.get("sac_train_freq", 1),
        "gradient_steps": arg_dict.get("sac_gradient_steps", 1),
        "ent_coef": "auto" if auto_ent else arg_dict.get("sac_alpha", 0.2),
        "target_update_interval": arg_dict.get("sac_target_update_interval", 1),
        "use_sde": bool(arg_dict.get("sac_use_sde", False)),
        "sde_sample_freq": arg_dict.get("sac_sde_sample_freq", -1),
        "use_sde_at_warmup": bool(arg_dict.get("sac_use_sde_at_warmup", False)),
        "optimize_memory_usage": bool(arg_dict.get("sac_optimize_memory_usage", False)),
        "policy_kwargs": arg_dict.get("sac_policy_kwargs", None),
        "device": arg_dict.get("device", "auto"),
    }
    if auto_ent:
        sac_kwargs["target_entropy"] = arg_dict.get("sac_target_entropy", "auto")
    implemented_combos["sac"]["pytorch"] = [SAC_P, (sac_policy, env), sac_kwargs]

    mbrl_args = (None, env, model_logdir, arg_dict)
    implemented_combos["pets"]["pytorch"] = [MBRLWrapper, mbrl_args, {"algo_name": "pets", "cfg_builder": build_mbrl_cfg}]
    implemented_combos["mbpo"]["pytorch"] = [MBRLWrapper, mbrl_args, {"algo_name": "mbpo", "cfg_builder": build_mbrl_cfg}]
    implemented_combos["planet"]["pytorch"] = [MBRLWrapper, mbrl_args, {"algo_name": "planet", "cfg_builder": build_mbrl_cfg}]

    implemented_combos["ppo"]["pytorch"] = [PPO_P, ('MlpPolicy', env),
                                            {"n_steps": arg_dict["algo_steps"], "verbose": 1, "tensorboard_log": model_logdir,
                                             "device": "cpu"}]
    implemented_combos["td3"]["pytorch"] = [TD3_P, ('MlpPolicy', env), {"verbose": 1, "tensorboard_log": model_logdir}]
    implemented_combos["a2c"]["pytorch"] = [A2C_P, ('MlpPolicy', env), {"n_steps": arg_dict["algo_steps"], "verbose": 1,
                                                                        "tensorboard_log": model_logdir}]
    implemented_combos["multippo"]["pytorch"] = [MultiPPOSB3, ("MlpPolicy", env),
                                                 {"n_steps": arg_dict["algo_steps"], "verbose": 1, "tensorboard_log": model_logdir,
                                                  "device": "cpu", "n_models": arg_dict["num_networks"]}]
    implemented_combos["aif"]["pytorch"] = [ActiveInferenceSB3, (env,), {"arg_dict": arg_dict}]
    meta_env = env if isinstance(env, MetaAIFActionWrapper) else MetaAIFActionWrapper(env)
    implemented_combos["meta_aif"]["pytorch"] = [MetaAIFSB3, (meta_env,), {"arg_dict": arg_dict}]
    return implemented_combos

def train(env, implemented_combos, model_logdir, arg_dict, pretrained_model=None):
    model_name = arg_dict["algo"] + '_' + str(arg_dict["steps"])
    conf_pth = os.path.join(model_logdir, "train.json")
    arg_dict["logdir"] = model_logdir #TODO: figure out whether logdir is needed (as there is a duplicity with pretrained model)
    arg_dict["pretrained_model"] = model_logdir
    seed = arg_dict.get("seed", None)
    steps = 0
    if not pretrained_model:
        #creating train.json when training from scratch
        with open(conf_pth, "w") as f:
            json.dump(arg_dict, f, indent=4)
        with open(os.path.join(model_logdir,"trained_steps.txt"), "a+") as f:
            f.write(f"model {model_name} has been saved at steps:" + "\n")
    else:
        #when loading pretrained model, figure out how many steps have already been trained
        with open(os.path.join(pretrained_model,"trained_steps.txt"), "r") as f:
            lines = f.readlines()
            line = lines[-1]
            steps = int(line)
        model_logdir = pretrained_model
    model_args = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][1]
    model_kwargs = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][2]
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        model_kwargs["seed"] = seed
    if pretrained_model:
        if not os.path.isabs(pretrained_model):
            pretrained_model = str(pkg_resources.files("myGym") / pretrained_model)
        env_for_load = model_args[0] if isinstance(model_args, tuple) and len(model_args) == 1 else model_args[1]
        if arg_dict["algo"] in ["aif", "meta_aif"]:
            vec_env = env_for_load
        elif not arg_dict["multiprocessing"]:
            vec_env = DummyVecEnv([lambda: env_for_load])
        else:
            vec_env = env_for_load

        model = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][0].load(
                pretrained_model, vec_env, device="cpu", **model_kwargs)
    else:
        model = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][0](*model_args, **model_kwargs)

    if arg_dict["algo"] == "gail":
        # Multi processing: (using MPI)
        if arg_dict["train_framework"] == 'tensorflow':
            # Generate expert trajectories (train expert)
            generate_expert_traj(model, model_name, n_timesteps=3000, n_episodes=100)
            # Load the expert dataset
            dataset = ExpertDataset(expert_path=model_name + '.npz', traj_limitation=10, verbose=1)
            kwargs = {"verbose": 1}
            if seed is not None:
                kwargs["seed"] = seed
            model = GAIL_T('MlpPolicy', model_name, dataset, **kwargs)

    start_time = time.time()
    callbacks_list = []
    if arg_dict["algo"] in ["sac", "td3", "ppo", "a2c", "multippo"]:
        action_log_freq = arg_dict.get("action_log_freq", 1000)
        callbacks_list.append(ActionLoggerCallback(log_every=action_log_freq))
    auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=1024, logdir=model_logdir, env=env,
                                                              engine=arg_dict["engine"],
                                                              multiprocessing=arg_dict["multiprocessing"],
                                                              save_model_every_steps=arg_dict["eval_freq"], starting_steps = steps,
                                                          algo = arg_dict["algo"])
    callbacks_list.append(auto_save_callback)
    if arg_dict["eval_freq"]:
        eval_env = env
        if arg_dict["multiprocessing"] is not None:
            NUM_CPU = int(arg_dict["multiprocessing"])
        else:
            NUM_CPU = 1
        if arg_dict["algo"] == "multippo":
            eval_callback = MultiPPOEvalCallback(eval_env, log_path=model_logdir,
                                               eval_freq=arg_dict["eval_freq"],
                                               algo_steps=arg_dict["algo_steps"],
                                               n_eval_episodes=arg_dict["eval_episodes"],
                                               record=arg_dict["record"],
                                               camera_id=arg_dict["camera"], num_cpu=NUM_CPU, starting_steps = steps)
    
        
        elif arg_dict["algo"] in ["pets", "mbpo", "planet", "aif", "meta_aif"]:
            eval_callback = MBRLEvalCallback(
                eval_env, log_path=model_logdir,
                eval_freq=arg_dict["eval_freq"],
                n_eval_episodes=arg_dict["eval_episodes"],
                deterministic=True,
                replay_buffer=getattr(model, "replay_buffer", None),
                verbose=1,
                starting_steps=steps,
                num_cpu=NUM_CPU,
            )
        
        else: 
            eval_callback = PPOEvalCallback(eval_env, log_path=model_logdir,
                                           eval_freq=arg_dict["eval_freq"],
                                           algo_steps=arg_dict["algo_steps"],
                                           n_eval_episodes=arg_dict["eval_episodes"],
                                           record=arg_dict["record"],
                                           camera_id=arg_dict["camera"], num_cpu=NUM_CPU, starting_steps = steps)
        callbacks_list.append(eval_callback)
    print("learn started")
    model.learn(total_timesteps=arg_dict["steps"], callback=callbacks_list)
    print("learn ended")
    if uses_sb3_style(arg_dict["algo"]):
        model.save(model_logdir, steps = steps + model.num_timesteps)
    else:
        model.save(f"{model_logdir}_{steps + model.num_timesteps}")

    env.close()
    print("Training time: {:.2f} s".format(time.time() - start_time))
    print("Training steps: {:} s".format(model.num_timesteps))

    # info_keywords in monitor class above is necessary for pybullet to save_results
    # when using the info_keywords for mujoco we get an error
    if arg_dict["engine"] == "pybullet":
        save_results(arg_dict, model_name, env, model_logdir)
    return model

def build_mbrl_cfg(arg_dict):
    batch_size = arg_dict.get("mbrl_batch_size", 256)
    val_ratio = arg_dict.get("validation_ratio", 0.05)
    cem_num_samples = arg_dict.get("cem_num_samples", 400)
    cem_num_elites = arg_dict.get("cem_num_elites", 40)
    cem_num_iters = arg_dict.get("cem_num_iters", 5)
    cem_alpha = arg_dict.get("cem_alpha", 0.1)
    cem_clipped_normal = arg_dict.get("cem_clipped_normal", False)
    default_learned_rewards = arg_dict.get("mbrl_learned_rewards", arg_dict.get("algo") == "mbpo")
    real_ratio = arg_dict.get("real_ratio", 0.0) or 0.0

    device = arg_dict.get("device", "cpu")
    cem_elite_ratio = (cem_num_elites / float(cem_num_samples)) if cem_num_samples else 0.1
    optimizer_choice = str(arg_dict.get("mbrl_optimizer", "cem")).lower()

    if optimizer_choice == "cem":
        optimizer_cfg = {
            "_target_": "mbrl.planning.CEMOptimizer",
            "num_iterations": cem_num_iters,
            "population_size": cem_num_samples,
            "elite_ratio": cem_elite_ratio,
            "alpha": cem_alpha,
            "return_mean_elites": True,
            "clipped_normal": cem_clipped_normal,
            "device": device,
        }
    elif optimizer_choice == "icem":
        icem_num_iters = arg_dict.get("icem_num_iters", cem_num_iters)
        icem_population_size = arg_dict.get("icem_population_size", cem_num_samples)
        icem_elite_ratio = arg_dict.get("icem_elite_ratio", cem_elite_ratio)
        icem_population_decay_factor = arg_dict.get("icem_population_decay_factor", 1.25)
        icem_colored_noise_exponent = arg_dict.get("icem_colored_noise_exponent", 2.0)
        icem_keep_elite_frac = arg_dict.get("icem_keep_elite_frac", 0.1)
        icem_alpha = arg_dict.get("icem_alpha", cem_alpha)
        icem_return_mean_elites = arg_dict.get("icem_return_mean_elites", True)
        icem_population_size_module = arg_dict.get("icem_population_size_module", None)

        optimizer_cfg = {
            "_target_": "mbrl.planning.ICEMOptimizer",
            "num_iterations": icem_num_iters,
            "population_size": icem_population_size,
            "elite_ratio": icem_elite_ratio,
            "population_decay_factor": icem_population_decay_factor,
            "colored_noise_exponent": icem_colored_noise_exponent,
            "keep_elite_frac": icem_keep_elite_frac,
            "alpha": icem_alpha,
            "return_mean_elites": icem_return_mean_elites,
            "device": device,
        }
        if icem_population_size_module is not None:
            optimizer_cfg["population_size_module"] = icem_population_size_module
    elif optimizer_choice == "mppi":
        mppi_num_iters = arg_dict.get("mppi_num_iters", 5)
        mppi_population_size = arg_dict.get("mppi_num_samples", cem_num_samples)
        mppi_gamma = arg_dict.get("mppi_gamma", arg_dict.get("mppi_lambda", 1.0))
        mppi_sigma = arg_dict.get("mppi_sigma", 1.0)
        mppi_beta = arg_dict.get("mppi_beta", 0.1)

        optimizer_cfg = {
            "_target_": "mbrl.planning.MPPIOptimizer",
            "num_iterations": mppi_num_iters,
            "population_size": mppi_population_size,
            "gamma": mppi_gamma,
            "sigma": mppi_sigma,
            "beta": mppi_beta,
            "device": device,
        }
    else:
        raise ValueError(
            f"Unknown mbrl_optimizer '{optimizer_choice}'. Use 'cem', 'icem', or 'mppi'."
        )

    base = {
        "seed": arg_dict.get("seed", 0),
        "device": "cpu",
        "algorithm": {
            "initial_exploration_steps": arg_dict.get("mbrl_init_random", 1000),
            "freq_train_model": arg_dict.get("mbrl_train_freq", 250),
            "num_particles": arg_dict.get("mbrl_num_particles", 20),
            "learned_rewards": bool(default_learned_rewards),
            "target_is_delta": True,
            "normalize": True,
            "num_eval_episodes": int(arg_dict.get("num_eval_episodes", 1)),
            "agent": {
                "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
                "action_lb": arg_dict.get("action_lb", []),
                "action_ub": arg_dict.get("action_ub", []),
                "planning_horizon": arg_dict.get("mbrl_planning_horizon", 20),
                "replan_freq": 1,
                "verbose": False,
                "optimizer_cfg": optimizer_cfg,
            },
        },
        "log_frequency_agent": arg_dict.get("log_frequency_agent", 1000),
        "dynamics_model": {
            "_target_": "mbrl.models.GaussianMLP",
            "ensemble_size": arg_dict.get("ensemble_size", 5),
            "hid_size": arg_dict.get("mbrl_hid_size", 200),
            "num_layers": arg_dict.get("mbrl_num_layers", 4),
            "propagation_method": arg_dict.get("mbrl_propagation_method", "fixed_model"),
            "device": arg_dict.get("device", "cpu"),
        },
        "overrides": {
            "num_steps": arg_dict["steps"],
            "batch_size": batch_size,
            "model_batch_size": batch_size,
            "model_lr": arg_dict.get("mbrl_model_lr", 1e-3),
            "model_wd": arg_dict.get("mbrl_model_wd", 1e-4),
            "validation_ratio": val_ratio,
        },
    }
    if arg_dict["algo"] == "mbpo":
        # SAC agent config (shapes and bounds will be completed at runtime).
        base["algorithm"]["agent"] = {
            "_target_": "mbrl.third_party.pytorch_sac_pranz24.sac.SAC",
            "num_inputs": "???",
            "action_space": {
                "_target_": "gymnasium.spaces.Box",
                "low": "???",
                "high": "???",
                "shape": "???",
                "dtype": "float32",
            },
            "args": {
                "gamma": arg_dict.get("sac_gamma", 0.99),
                "tau": arg_dict.get("sac_tau", 0.005),
                "alpha": arg_dict.get("sac_alpha", 0.2),
                "policy": arg_dict.get("sac_policy", "Gaussian"),
                "target_update_interval": arg_dict.get("sac_target_update_interval", 4),
                "automatic_entropy_tuning": bool(arg_dict.get("sac_automatic_entropy_tuning", True)),
                "target_entropy": arg_dict.get("sac_target_entropy", -0.05),
                "hidden_size": arg_dict.get("sac_hidden_size", 256),
                "device": arg_dict.get("device", "cpu"),
                "lr": arg_dict.get("sac_lr", 3e-4),
            },
        }
        base["algorithm"]["rollout_schedule"] = arg_dict.get("rollout_schedule", [1, 15, 1, 1])
        base["algorithm"]["real_ratio"] = real_ratio
        base["algorithm"]["real_data_ratio"] = real_ratio
        base["algorithm"]["learned_rewards"] = True  # MBPO requires model-predicted rewards
        base["overrides"].update({
            "freq_train_model": arg_dict.get("mbrl_train_freq", 250),
            "effective_model_rollouts_per_step": arg_dict.get("effective_model_rollouts_per_step", 400),
            "rollout_schedule": arg_dict.get("rollout_schedule", [1, 15, 1, 1]),
            "epoch_length": arg_dict.get("epoch_length", 1000),
            "num_sac_updates_per_step": arg_dict.get("num_sac_updates_per_step", 20),
            "sac_updates_every_steps": arg_dict.get("sac_updates_every_steps", 1),
            "num_epochs_to_retain_sac_buffer": arg_dict.get("num_epochs_to_retain_sac_buffer", 1),
            "sac_batch_size": arg_dict.get("sac_batch_size", batch_size),
            "sac_gamma": arg_dict.get("sac_gamma", 0.99),
            "sac_tau": arg_dict.get("sac_tau", 0.005),
            "sac_alpha": arg_dict.get("sac_alpha", 0.2),
            "sac_policy": arg_dict.get("sac_policy", "Gaussian"),
            "sac_target_update_interval": arg_dict.get("sac_target_update_interval", 4),
            "sac_automatic_entropy_tuning": bool(arg_dict.get("sac_automatic_entropy_tuning", True)),
            "sac_target_entropy": arg_dict.get("sac_target_entropy", -0.05),
            "sac_hidden_size": arg_dict.get("sac_hidden_size", 256),
            "sac_lr": arg_dict.get("sac_lr", 3e-4),
        })
    return OmegaConf.create(base)


def get_parser():
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument("-cfg", "--config", type=str, default = "./configs/temp.json", help="Config file path") #./trained_models/AG/AG_table_tiago_tiago_dual_joints_gripper_multippo/train.json
    parser.add_argument("-n", "--env_name", type=str, help="Environment name")
    parser.add_argument("-ws", "--workspace", type=str, help="Workspace name")
    parser.add_argument("-p", "--engine", type=str, help="Simulation engine name")
    parser.add_argument("-sd", "--seed", type=int, default=1, help="Seed number")
    parser.add_argument("-d", "--render", type=str, help="Rendering type: opengl, opencv")
    parser.add_argument("-c", "--camera", type=int, help="Number of cameras for rendering and recording")
    parser.add_argument("-vi", "--visualize", type=int, help="Visualize camera render and vision: 1 or 0")
    parser.add_argument("-vg", "--visgym", type=int, help="Visualize gym background: 1 or 0")
    parser.add_argument("-g", "--gui", type=int, help="Use GUI: 1 or 0")

    # Robot
    parser.add_argument("-b", "--robot", default=["kuka", "panda"], nargs='*',
                        help="Robot to train: kuka, panda, jaco ...")
    parser.add_argument("-bi", "--robot_init", nargs="*", type=float, help="Initial robot's end-effector position")
    parser.add_argument("-ba", "--robot_action", type=str, help="Robot's action control: step - end-effector relative position, absolute - end-effector absolute position, joints - joints' coordinates")
    parser.add_argument("-mv", "--max_velocity", type=float, help="Maximum velocity of robotic arm")
    parser.add_argument("-mf", "--max_force", type=float, help="Maximum force of robotic arm")
    parser.add_argument("-ar", "--action_repeat", type=int, help="Substeps of simulation without action from env")
    #Task
    parser.add_argument("-tt", "--task_type", type=str,  help="Type of task to learn: reach, push, throw, pick_and_place")
    parser.add_argument("-to", "--task_objects", nargs="*", type=str, help="Object (for reach) or a pair of objects (for other tasks) to manipulate with")
    parser.add_argument("-u", "--used_objects", nargs="*", type=str, help="List of extra objects to randomly appear in the scene")
    #Distractors
    parser.add_argument("-di", "--distractors", type=str, help="Object (for reach) to evade")
    parser.add_argument("-dm", "--distractor_moveable", type=int, help="can distractor move (0/1)")
    parser.add_argument("-ds", "--distractor_constant_speed", type=int, help="is speed of distractor constant (0/1)")
    parser.add_argument("-dd", "--distractor_movement_dimensions", type=int, help="in how many directions can the distractor move (1/2/3)")
    parser.add_argument("-de", "--distractor_movement_endpoints", nargs="*", type=float, help="2 coordinates (starting point and ending point)")
    parser.add_argument("-no", "--observed_links_num", type=int, help="number of robot links in observation space")
    #Reward
    parser.add_argument("-re", "--reward", type=str,  help="Defines how to compute the reward")
    parser.add_argument("-dt", "--distance_type", type=str, help="Type of distance metrics: euclidean, manhattan")
    #Train
    parser.add_argument("-w", "--train_framework", type=str,  help="Name of the training framework you want to use: {tensorflow, pytorch}")
    parser.add_argument("-a", "--algo", type=str,  help="The learning algorithm to be used (ppo2 or her)")
    parser.add_argument("-s", "--steps", type=int, help="The number of steps to train")
    parser.add_argument("-ms", "--max_episode_steps", type=int,  help="The maximum number of steps per episode")
    parser.add_argument("-ma", "--algo_steps", type=int,  help="The number of steps per for algo training (PPO2,A2C)")
    # Active inference
    parser.add_argument("--aif_initial_random_steps", type=int, help="Initial random exploration steps for AIF agent")
    parser.add_argument("--aif_model_update_freq", type=int, help="World model update frequency (in steps) for AIF")
    parser.add_argument("--aif_policy_updates_per_step", type=int, help="How many policy updates per env step in AIF")
    parser.add_argument("--aif_epistemic_scale", type=float, help="Scale of epistemic term in AIF expected free energy")
    parser.add_argument("--aif_replay_size", type=int, help="Replay buffer size for AIF agent")
    parser.add_argument("--aif_gamma", type=float, help="Discount factor used by the AIF policy")
    parser.add_argument("--aif_device", type=str, help="Device for AIF training (cpu/cuda)")
    parser.add_argument("--aif_log_freq", type=int, help="Logging frequency (steps) for AIF training metrics")
    parser.add_argument("--aif_model_normalize", type=int, help="Enable PETS-style input normalization for AIF (1/0)")
    # Meta-AIF (belief-based, uncertainty-adaptive planning) hyperparameters
    parser.add_argument("--meta_aif_H_min", type=int, help="Min planning horizon for meta AIF")
    parser.add_argument("--meta_aif_H_max", type=int, help="Max planning horizon for meta AIF")
    parser.add_argument("--meta_aif_H_internal", type=int, help="Fixed planning horizon override for meta AIF")
    parser.add_argument("--meta_aif_K_min", type=int, help="Min number of candidate policies for meta AIF")
    parser.add_argument("--meta_aif_K_max", type=int, help="Max number of candidate policies for meta AIF")
    parser.add_argument("--meta_aif_K_internal", type=int, help="Number of candidate policies for meta AIF (training/internal)")
    parser.add_argument("--meta_aif_K_external", type=int, help="Number of candidate policies for meta AIF (evaluation/external)")
    parser.add_argument("--meta_aif_meta_candidates", type=int, help="How many (horizon, candidate) meta-options to score each step")
    parser.add_argument("--meta_aif_think_cost", type=float, help="Cost multiplier for thinking steps (horizon*candidates)")
    parser.add_argument("--meta_aif_efe_scale", type=float, help="Scaling factor applied to EFE predictor outputs")
    parser.add_argument("--meta_aif_efe_clip", type=float, help="Max clamp value for scaled EFE predictor outputs")
    parser.add_argument("--meta_aif_K", type=int, help="Legacy: unified candidate count for meta AIF")
    parser.add_argument("--meta_aif_lambda_think", type=float, help="Per-step cost for internal actions")
    parser.add_argument("--meta_aif_lambda_depth", type=float, help="Cost per unit planning depth")
    parser.add_argument("--meta_aif_lambda_cand", type=float, help="Cost per candidate policy")
    parser.add_argument("--meta_aif_beta_state", type=float, help="Weight for state epistemic term")
    parser.add_argument("--meta_aif_beta_param", type=float, help="Weight for parameter epistemic term")
    parser.add_argument("--meta_aif_alpha_obs", type=float, help="Weight for observation free energy")
    parser.add_argument("--meta_aif_gamma", type=float, help="Softmax temperature over policies")
    parser.add_argument("--meta_aif_param_dim", type=int, help="Latent parameter dimension for meta AIF")
    parser.add_argument("--meta_aif_hidden_size", type=int, help="Hidden layer width for meta AIF nets")
    parser.add_argument("--meta_aif_lr", type=float, help="Learning rate for meta AIF optimizer")
    parser.add_argument("--meta_aif_internal_threshold", type=float, help="Magnitude threshold to mark actions as internal (planning) vs external")
    parser.add_argument("--meta_aif_debug", action="store_true", help="Enable debug logs for meta AIF")
    parser.add_argument("--meta_aif_log_thinking", type=int, help="Enable per-step meta-planning logs (1/0)")
    # MBRL common
    parser.add_argument("-mbph", "--mbrl_planning_horizon", type=int, help="mbrl planning horizon")
    parser.add_argument("-mbnp", "--mbrl_num_particles", type=int, help="particles for model rollout")
    parser.add_argument("-mbir", "--mbrl_init_random", type=int, help="initial random steps")
    parser.add_argument("-mbtf", "--mbrl_train_freq", type=int, help="steps between model updates")
    parser.add_argument("-mbes", "--ensemble_size", type=int, help="dyn model ensemble size")
    parser.add_argument("-mbhs", "--mbrl_hid_size", type=int, help="dyn model hidden size")
    parser.add_argument("-mbnl", "--mbrl_num_layers", type=int, help="dyn model layers")
    parser.add_argument("-mblr", "--mbrl_model_lr", type=float, help="dyn model lr")
    parser.add_argument("-mbwd", "--mbrl_model_wd", type=float, help="dyn model weight decay")
    parser.add_argument("-mbbs", "--mbrl_batch_size", type=int, help="model train batch size")
    parser.add_argument("--mbrl_learned_rewards", action="store_true", help="learn reward in model")
    # CEM/MPPI (used by PETS/MBPO)
    parser.add_argument("-mbns", "--cem_num_samples", type=int, help="CEM population")
    parser.add_argument("-mbne", "--cem_num_elites", type=int, help="CEM elites")
    parser.add_argument("-mbte", "--cem_temperature", type=float, help="CEM temperature")
    parser.add_argument("-mbpi", "--mppi_num_samples", type=int, help="MPPI samples")
    parser.add_argument("-mbla", "--mppi_lambda", type=float, help="MPPI lambda")
    #Evaluation
    parser.add_argument("-ef", "--eval_freq", type=int,  help="Evaluate the agent every eval_freq steps")
    parser.add_argument("-e", "--eval_episodes", type=int,  help="Number of episodes to evaluate performance of the robot")
    #Saving and Logging
    parser.add_argument("-l", "--logdir", type=str,  help="Where to save results of training and trained models")
    parser.add_argument("-r", "--record", type=int, help="1: make a gif of model perfomance, 2: make a video of model performance, 0: don't record")
    #Mujoco
    parser.add_argument("-i", "--multiprocessing", type=int, help="True: multiprocessing on (specify also the number of vectorized environemnts), False: multiprocessing off")
    parser.add_argument("-v", "--vectorized_envs", type=int,  help="The number of vectorized environments to run at once (mujoco multiprocessing only)")
    #Paths
    parser.add_argument("-m", "--model_path", type=str, help="Path to the the trained model to test")
    parser.add_argument("-vp", "--vae_path", type=str, help="Path to a trained VAE in 2dvu reward type")
    parser.add_argument("-yp", "--yolact_path", type=str, help="Path to a trained Yolact in 3dvu reward type")
    parser.add_argument("-yc", "--yolact_config", type=str, help="Path to saved config obj or name of an existing one in the data/Config script (e.g. 'yolact_base_config') or None for autodetection")
    parser.add_argument('-ptm', "--pretrained_model", type=str, help="Path to a model that you want to continue training")
    #Language
    parser.add_argument("-nl", "--natural_language", type=str, default="",
                        help="If passed, instead of training the script will produce a natural language output "
                             "of the given type, save it to the predefined file (for communication with other scripts) "
                             "and exit the program (without the actual training taking place). Expected values are \"description\" "
                             "(generate a task description) or \"new_tasks\" (generate new tasks)")
    
    # Moving target (optional)
    parser.add_argument("--moving_target", type=int, help="Enable moving target (1) or disable (0)")
    parser.add_argument("--moving_target_urdf", type=str, help="URDF filename used for the moving target")
    parser.add_argument("--moving_target_bounds", nargs="*", type=float,
                        help="Workspace [x_low x_high y_low y_high]")
    parser.add_argument("--moving_target_reach_bounds", nargs="*", type=float,
                        help="Robot reach [x_low x_high y_low y_high]")
    parser.add_argument("--moving_target_speed_range", nargs="*", type=float,
                        help="Speed range [vmin vmax]")
    parser.add_argument("--moving_target_z", type=float, help="Fixed Z of moving target")
    parser.add_argument("--moving_target_hit_time_window", nargs="*", type=float,
                        help="Time window [tmin tmax] to guarantee a pass through reach area")

    return parser


def get_arguments(parser):
    args = parser.parse_args()
    commands = {}
    with open(args.config, "r") as f:
        arg_dict = commentjson.load(f)
    for key, value in arg_dict.items():
        if value is not None and key != "config":
            if key in ["robot_init"] or key in ["end_effector_orn"]:
                arg_dict[key] = [float(arg_dict[key][i]) for i in range(len(arg_dict[key]))]
            elif type(value) is list and len(value) <= 1 and key != "task_objects":
                arg_dict[key] = value[0]
    for key, value in vars(args).items():
        if value is not None and key != "config":
            if key not in arg_dict or arg_dict[key] is None:
                arg_dict[key] = value
            if value != parser.get_default(key):
                commands[key] = value
                if key in ["task_objects"]:
                    arg_dict[key] = task_objects_replacement(value, arg_dict[key], arg_dict["task_type"])
                    if len(value) == 1:
                        commands[key] = value[0]
                elif type(value) is list and len(value) <= 1:
                    arg_dict[key] = value[0]
                else:
                    arg_dict[key] = value
    return arg_dict, commands


def task_objects_replacement(task_objects_new, task_objects_old, task_type):
    """
    If task_objects is given as a parameter, this method converts string into a proper format depending on task_type
    (null init for task_type reach)

    [{"init":{"obj_name":"null"}, "goal":{"obj_name":"cube_holes","fixed":1,"rand_rot":0, "sampling_area":[-0.5, 0.2,
    0.3, 0.6, 0.1, 0.4]}}]
    """
    ret = copy.deepcopy(task_objects_old)
    if len(task_objects_new) > len(task_objects_old):
        msg = "More objects given than there are subtasks."
        raise Exception(msg)
    if task_type == "reach":
        dest = "goal"
    else:
        dest = "init"
    for i in range(len(task_objects_new)):
        ret[i][dest]["obj_name"] = task_objects_new[i]
    return ret


def process_natural_language_command(cmd, env,
                                     output_relative_path=os.path.join("envs", "examples", "natural_language.txt")):
    env.reset()
    nl = NaturalLanguage(env)
    if cmd in ["description", "new_tasks"]:
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), output_relative_path), "w") as file:
            file.write(nl.generate_task_description() if cmd == "description" else "\n".join(nl.generate_new_tasks()))
    else:
        msg = f"Unknown natural language command: {cmd}"
        raise Exception(msg)

def automatic_argument_assignment(arg_dict):
    
    task_type_str = arg_dict.get("task_type")
    if task_type_str and isinstance(task_type_str, str):
        arg_dict["num_networks"] = len(task_type_str)
        arg_dict["reward"] = arg_dict["task_type"]
        arg_dict["logdir"] = "./trained_models/"  + arg_dict["robot"] + "/" + arg_dict["task_type"]
        arg_dict["algo_steps"] = arg_dict["max_episode_steps"]
        print("Number of networks from task type is:", arg_dict["num_networks"])
        print("Reward type set to:", arg_dict["reward"])
        print("Log directory set to:", arg_dict["logdir"])
        print("Algorithm steps set to:", arg_dict["algo_steps"])
    else:
        arg_dict["num_networks"] = 1
        arg_dict["reward"] = "None"
    
     # Default if task_type is missing, None, not a string, or empty
    return arg_dict

def main():
    parser = get_parser()
    arg_dict, commands = get_arguments(parser)
    args = parser.parse_args()
    arg_dict["top_grasp"] = False

    # Defaults for optional PETS policy distillation
    arg_dict.setdefault("pets_policy_mimic", False)
    arg_dict.setdefault("pets_mimic_use_policy", arg_dict.get("pets_policy_mimic", False))
    arg_dict.setdefault("pets_mimic_batch_size", 512)
    arg_dict.setdefault("pets_mimic_epochs", 10)
    arg_dict.setdefault("pets_mimic_hidden_sizes", [256, 256])
    arg_dict.setdefault("pets_mimic_learning_rate", 1e-3)
    arg_dict.setdefault("pets_mimic_val_split", 0.1)

    # for key, arg in arg_dict.items():
    #     if type(arg_dict[key]) == list:
    #         if len(arg_dict[key]) > 1 and key != "robot_init":
    #             if key != "task_objects":
    #                 parameters[key] = arg
    #                 if key in commands:
    #                     commands.pop(key)

    # Check if we chose one of the existing engines
    if arg_dict["engine"] not in AVAILABLE_SIMULATION_ENGINES:
        print(f"Invalid simulation engine. Valid arguments: --engine {AVAILABLE_SIMULATION_ENGINES}.")
        return

    #Automatic argument assigment from task type
    arg_dict = automatic_argument_assignment(arg_dict)
    
    if not os.path.isabs(arg_dict["logdir"]):
        arg_dict["logdir"] = os.path.join("./", arg_dict["logdir"])
    os.makedirs(arg_dict["logdir"], exist_ok=True)
    model_logdir_ori = os.path.join(arg_dict["logdir"], "_".join(
        (arg_dict["robot_action"], arg_dict["algo"])))

    model_logdir = model_logdir_ori
    add = 1
    if not arg_dict["pretrained_model"]:
        #If training from scratch, make a new logdir for the model
        #logdir includes train.json file and monitor.csv file
        while True:
            try:
                os.makedirs(model_logdir, exist_ok=False)
                break
            except:
                model_logdir = "_".join((model_logdir_ori, str(add)))
                add += 1
    else:
        #In case of renewing training from a checkpoint, logdir with monitor.csv
        #and train.json are located in the directory where pretrained model is stored
        model_logdir = os.path.dirname(os.path.dirname(arg_dict["pretrained_model"]))
    if arg_dict["algo"] in ["aif", "meta_aif"]:
        if arg_dict["multiprocessing"]:
            print("AIF does not support multiprocessing; running single environment instead.")
            arg_dict["multiprocessing"] = None
        env = configure_env(arg_dict, model_logdir, for_train=1)
        if arg_dict["algo"] == "meta_aif":
            env = MetaAIFActionWrapper(env)
    elif arg_dict["multiprocessing"]:
        NUM_CPU = max(int(arg_dict["multiprocessing"]), 1)
        env = SubprocVecEnv([make_env(arg_dict, i, model_logdir=model_logdir) for i in range(NUM_CPU)])
        env = VecMonitor(env, model_logdir)
    else:
        env = configure_env(arg_dict, model_logdir, for_train=1)

    log_action_and_joint_info(env)
    implemented_combos = configure_implemented_combos(env, model_logdir, arg_dict)
    train(env, implemented_combos, model_logdir, arg_dict, arg_dict["pretrained_model"])



if __name__ == "__main__":
    main()
