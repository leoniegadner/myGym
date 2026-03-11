"""
Latent-state (partially observed) variant of an Amortized AIF Agent, as in
'Scaling Active Inference' TODO: [CITE!!!].

Key components:
- Encoder:       q_phi(s_t | o_t)  = N(mu_phi(o_t), diag(exp(logvar_phi(o_t))))
- Decoder:       p_lambda(o_t | s_t) = N(mu_lambda(s_t), diag(exp(logvar_lambda(s_t))))
- Dynamics:      p_theta(s_{t+1} | s_t, a_t, theta) as a Bayesian NN
- Preferences:   p(o_t | C) captures desired observations / features

Training objective per transition (o_{t-1}, a_{t-1}, o_t):

F_t ~= E_{q(theta)} E_{q(s_{t-1}|o_{t-1})} KL[ q(s_t|o_t) || p(s_t|s_{t-1}, a_{t-1}, theta) ]
       + KL(q(theta) || p(theta))
       - E_{q(s_t|o_t)} [ log p(o_t | s_t) ]

We approximate expectations via Monte Carlo (reparameterization).

Expected Free Energy (EFE) for action selection:
  -G = Extrinsic Value + State Info Gain + Parameter Info Gain

Where:
  - Extrinsic Value = E_q(o|π)[ln p(o|C)]  (pragmatic value)
  - State Info Gain = I(s; o | π)  (state information gain)
  - Parameter Info Gain = I(θ; s | π)  (parameter/model information gain)

Planning uses CEM over action sequences in latent space, minimizing G under
model uncertainty. Ensemble disagreement serves as proxy for information gain.
"""

import inspect
import json
import math
import os
import random
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from torch.distributions import Normal

try:
    from tqdm import tqdm
except Exception:
    tqdm = None
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

try:
    from myGym.active_inference.meta_planning import MetaPlanningController
except Exception:
    MetaPlanningController = None

try:
    import mbrl.planning as mbrl_planning
except Exception:
    mbrl_planning = None
try:
    import mbrl.models as mbrl_models
except Exception:
    mbrl_models = None
try:
    from mbrl.util.math import Normalizer as MBRLNormalizer
except Exception:
    MBRLNormalizer = None


# ======================================================================
#  Bayesian Linear Layer (Bayes by Backprop)
# ======================================================================

class BayesianLinear(nn.Module):
    """
    Linear layer with Normal weight posterior:
        w ~ N(mu_w, sigma_w^2), with sigma_w = softplus(rho_w)
        b ~ N(mu_b, sigma_b^2)

    Prior: N(0, 1) for all weights and biases.
    """

    def __init__(self, in_features, out_features, deterministic=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.deterministic = deterministic  # For debugging: disable sampling

        # Variational posterior parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))

        self.reset_parameters()

        # Standard Normal prior
        self.weight_prior = Normal(
            loc=torch.zeros_like(self.weight_mu),
            scale=torch.ones_like(self.weight_mu),
        )
        self.bias_prior = Normal(
            loc=torch.zeros_like(self.bias_mu),
            scale=torch.ones_like(self.bias_mu),
        )

    def reset_parameters(self):
        std = 0.1
        self.weight_mu.data.normal_(0.0, std)
        self.weight_rho.data.fill_(-3.0)  # softplus(-3) ~ 0.05
        self.bias_mu.data.normal_(0.0, std)
        self.bias_rho.data.fill_(-3.0)

    @property
    def weight_sigma(self):
        return F.softplus(self.weight_rho)

    @property
    def bias_sigma(self):
        return F.softplus(self.bias_rho)

    def sample_eps(self):
        """Sample weight/bias noise for Bayes by Backprop."""
        eps_w = torch.randn_like(self.weight_mu)
        eps_b = torch.randn_like(self.bias_mu)
        return eps_w, eps_b

    def forward(self, x, sample=True, eps=None):
        """
        If eps is provided, reuse those noise tensors (eps_w, eps_b) to keep a
        coherent weight sample across multiple forward passes.
        If deterministic=True, always use mean weights (for debugging).
        """
        # Override sample if deterministic mode is enabled
        if self.deterministic:
            sample = False
        
        if sample:
            if eps is None:
                eps_w, eps_b = self.sample_eps()
            else:
                eps_w, eps_b = eps
            weight = self.weight_mu + self.weight_sigma * eps_w
            bias = self.bias_mu + self.bias_sigma * eps_b
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def kl_loss(self):
        weight_q = Normal(self.weight_mu, self.weight_sigma)
        bias_q = Normal(self.bias_mu, self.bias_sigma)
        kl_w = torch.distributions.kl_divergence(weight_q, self.weight_prior).sum()
        kl_b = torch.distributions.kl_divergence(bias_q, self.bias_prior).sum()
        return kl_w + kl_b


# ======================================================================
#  Encoder / Decoder for latent states
# ======================================================================

class Encoder(nn.Module):
    """
    q_phi(s_t | o_t) = N(mu_phi(o_t), diag(exp(logvar_phi(o_t))))
    """

    def __init__(self, obs_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, obs):
        x = self.net(obs)
        mu = self.mu_head(x)
        logvar = self.logvar_head(x).clamp(min=-10.0, max=2.0)
        return mu, logvar


class Decoder(nn.Module):
    """
    p_lambda(o_t | s_t) = N(mu_lambda(s_t), diag(exp(logvar_lambda(s_t))))
    """

    def __init__(self, latent_dim, obs_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, obs_dim)
        self.logvar_head = nn.Linear(hidden_dim, obs_dim)

    def forward(self, s):
        x = self.net(s)
        mu = self.mu_head(x)
        logvar = self.logvar_head(x).clamp(min=-10.0, max=2.0)
        return mu, logvar


class PolicyNet(nn.Module):
    """
    Habitual policy: q_phi_a(a_t | s_t) = N(mu(s_t), diag(exp(logstd)^2))
    """

    def __init__(self, latent_dim, action_dim, hidden_dim=128, log_std_min=-5.0, log_std_max=2.0, deterministic=False):
        super().__init__()
        self.deterministic = deterministic
        self.fc1 = BayesianLinear(latent_dim, hidden_dim, deterministic=deterministic)
        self.fc2 = BayesianLinear(hidden_dim, hidden_dim, deterministic=deterministic)
        self.mean_head = BayesianLinear(hidden_dim, action_dim, deterministic=deterministic)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self._squash_eps = 1e-6
        self._kl_norm = (
            self.fc1.weight_mu.numel()
            + self.fc1.bias_mu.numel()
            + self.fc2.weight_mu.numel()
            + self.fc2.bias_mu.numel()
            + self.mean_head.weight_mu.numel()
            + self.mean_head.bias_mu.numel()
        )

    def sample_eps(self):
        """Sample and return per-layer eps to reuse across a rollout."""
        return {
            "fc1": self.fc1.sample_eps(),
            "fc2": self.fc2.sample_eps(),
            "mean_head": self.mean_head.sample_eps(),
        }

    def forward(self, s, sample=True, eps_cache=None):
        eps_cache = eps_cache or {}
        h = self.fc1(s, sample=sample, eps=eps_cache.get("fc1"))
        h = F.relu(h)
        h = self.fc2(h, sample=sample, eps=eps_cache.get("fc2"))
        h = F.relu(h)
        mean = self.mean_head(h, sample=sample, eps=eps_cache.get("mean_head"))
        log_std = torch.clamp(self.log_std, min=self.log_std_min, max=self.log_std_max)
        std = torch.exp(log_std)
        return mean, std

    def _squash(self, u, action_low, action_high):
        a_tanh = torch.tanh(u)
        scale = (action_high - action_low) / 2.0
        bias = (action_high + action_low) / 2.0
        action = a_tanh * scale + bias
        # Jacobian correction terms for tanh and scaling
        log_squash = torch.log(1 - a_tanh.pow(2) + self._squash_eps).sum(dim=-1)
        log_scale = torch.log(scale + self._squash_eps).sum(dim=-1)
        return action, log_squash, log_scale

    def _unsquash(self, action, action_low, action_high):
        scale = (action_high - action_low) / 2.0
        bias = (action_high + action_low) / 2.0
        a_tanh = (action - bias) / (scale + self._squash_eps)
        a_tanh = torch.clamp(a_tanh, -1 + self._squash_eps, 1 - self._squash_eps)
        u = torch.atanh(a_tanh)
        return u, a_tanh, scale

    def sample_action(self, s, action_low, action_high, eps_cache=None):
        """
        Sample a bounded action using a tanh-squashed Gaussian and return the
        corrected log-probability.
        """
        mean, std = self(s, eps_cache=eps_cache)
        dist = Normal(mean, std)
        eps = torch.randn_like(mean)
        u = mean + std * eps
        action, log_squash, log_scale = self._squash(u, action_low, action_high)
        log_prob_u = dist.log_prob(u).sum(dim=-1)
        log_prob = log_prob_u - log_scale - log_squash
        return action, log_prob, u

    def log_prob_action(self, actions, mean, std, action_low, action_high):
        """
        Compute log-prob of bounded actions under the tanh-squashed Gaussian.
        """
        dist = Normal(mean, std)
        u, a_tanh, scale = self._unsquash(actions, action_low, action_high)
        log_prob_u = dist.log_prob(u).sum(dim=-1)
        log_scale = torch.log(scale + self._squash_eps).sum(dim=-1)
        log_squash = torch.log(1 - a_tanh.pow(2) + self._squash_eps).sum(dim=-1)
        return log_prob_u - log_scale - log_squash

    def kl_loss(self):
        kl = self.fc1.kl_loss() + self.fc2.kl_loss() + self.mean_head.kl_loss()
        return kl / float(self._kl_norm + 1e-8)


# ======================================================================
#  Dynamics Model: Bayesian MLP for p(s_{t+1} | s_t, a_t, theta)
# ======================================================================

class BayesianDynamicsModel(nn.Module):
    """
    Bayesian NN mapping [s_t, a_t] -> (s_{t+1}_mean, s_{t+1}_logvar)

    We model:
        s_{t+1} ~ N(mu_theta(x), diag(exp(logvar_theta(x))))
    """

    def __init__(
        self,
        latent_dim,
        action_dim,
        hidden_dim=128,
        deterministic=False,
        predict_delta: bool = False,
        logvar_min: float = -10.0,
        logvar_max: float = 2.0,
        device: str = "cpu",
        normalize_inputs: bool = False,
        normalize_double_precision: bool = False,
    ):
        super().__init__()
        in_dim = latent_dim + action_dim
        out_dim = 2 * latent_dim
        self.deterministic = deterministic
        self.predict_delta = bool(predict_delta)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self._device = torch.device(device)

        self.fc1 = BayesianLinear(in_dim, hidden_dim, deterministic=deterministic)
        self.fc2 = BayesianLinear(hidden_dim, hidden_dim, deterministic=deterministic)
        self.fc_out = BayesianLinear(hidden_dim, out_dim, deterministic=deterministic)

        self.latent_dim = latent_dim
        self.input_normalizer = None
        if normalize_inputs:
            if MBRLNormalizer is None:
                raise RuntimeError("mbrl.util.math.Normalizer is required for Bayesian dynamics normalization.")
            dtype = torch.double if normalize_double_precision else torch.float
            self.input_normalizer = MBRLNormalizer(in_dim, self._device, dtype=dtype)

    def sample_eps(self):
        """Sample and return per-layer eps to reuse across a rollout."""
        return {
            "fc1": self.fc1.sample_eps(),
            "fc2": self.fc2.sample_eps(),
            "fc_out": self.fc_out.sample_eps(),
        }

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_normalizer is None:
            return x
        if x.ndim == 1:
            x = x.unsqueeze(0)
            x_norm = self.input_normalizer.normalize(x).float()
            return x_norm.squeeze(0)
        return self.input_normalizer.normalize(x).float()

    def update_normalizer(self, obs: np.ndarray, actions: np.ndarray) -> None:
        if self.input_normalizer is None:
            return
        obs_arr = np.asarray(obs)
        act_arr = np.asarray(actions)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr.reshape(1, -1)
        if act_arr.ndim == 1:
            act_arr = act_arr.reshape(1, -1)
        if obs_arr.shape[0] != act_arr.shape[0]:
            raise ValueError("Bayesian normalizer expects matching batch sizes for obs and actions.")
        model_in = np.concatenate([obs_arr, act_arr], axis=-1)
        self.input_normalizer.update_stats(model_in)

    def forward(self, s, a, sample_theta=True, eps_cache=None):
        """
        s: [B, latent_dim]
        a: [B, action_dim]
        returns: next_mu [B, latent_dim], next_logvar [B, latent_dim]
        """
        # Override sample_theta if deterministic mode is enabled
        if self.deterministic:
            sample_theta = False
        
        eps_cache = eps_cache or {}
        x = torch.cat([s, a], dim=-1)
        x = self._normalize_input(x)
        x = torch.tanh(self.fc1(x, sample=sample_theta, eps=eps_cache.get("fc1")))
        x = torch.tanh(self.fc2(x, sample=sample_theta, eps=eps_cache.get("fc2")))
        out = self.fc_out(x, sample=sample_theta, eps=eps_cache.get("fc_out"))

        mu, logvar = out[..., :self.latent_dim], out[..., self.latent_dim:]
        if self.predict_delta:
            mu = mu + s
        logvar = logvar.clamp(min=self.logvar_min, max=self.logvar_max)
        return mu, logvar

    def kl_loss(self):
        return (
            self.fc1.kl_loss() +
            self.fc2.kl_loss() +
            self.fc_out.kl_loss()
        )

    def sample_next_state(self, s, a, eps_cache=None):
        """
        Sample s_{t+1} from p(s_{t+1}|s_t,a_t,theta) with theta ~ q(theta).
        """
        mu, logvar = self.forward(s, a, sample_theta=True, eps_cache=eps_cache)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps


# ======================================================================
#  PETS-style dynamics model for fully observable MDPs
# ======================================================================

class GaussianMLP(nn.Module):
    """
    Simple Gaussian MLP that outputs mean and log-variance.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=200, num_layers=3, logvar_min=-10.0, logvar_max=2.0):
        super().__init__()
        num_layers = max(1, int(num_layers))
        layers = []
        dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(dim, output_dim)
        self.logvar_head = nn.Linear(dim, output_dim)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

    def forward(self, x):
        h = self.net(x)
        mu = self.mean_head(h)
        logvar = self.logvar_head(h).clamp(min=self.logvar_min, max=self.logvar_max)
        return mu, logvar


class PETSDynamicsModel(nn.Module):
    """
    Ensemble of Gaussian MLPs that predicts next observation (optionally as delta).
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim=200,
        num_layers=3,
        ensemble_size=5,
        predict_delta=True,
        logvar_min=-10.0,
        logvar_max=2.0,
        device="cpu",
        normalize_inputs: bool = False,
        normalize_double_precision: bool = False,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.ensemble_size = max(1, int(ensemble_size))
        self.predict_delta = bool(predict_delta)
        self._device = torch.device(device)
        input_dim = self.obs_dim + self.action_dim
        if mbrl_models is None:
            raise RuntimeError("mbrl.models is required for PETS dynamics in fully observable MDPs.")
        self.model = mbrl_models.GaussianMLP(
            in_size=input_dim,
            out_size=self.obs_dim,
            device=self._device,
            num_layers=num_layers,
            ensemble_size=self.ensemble_size,
            hid_size=hidden_dim,
            deterministic=False,
            propagation_method=None,
        )
        if hasattr(self.model, "min_logvar") and hasattr(self.model, "max_logvar"):
            with torch.no_grad():
                self.model.min_logvar.fill_(float(logvar_min))
                self.model.max_logvar.fill_(float(logvar_max))
        self.input_normalizer = None
        if normalize_inputs:
            if MBRLNormalizer is None:
                raise RuntimeError("mbrl.util.math.Normalizer is required for PETS-style normalization.")
            dtype = torch.double if normalize_double_precision else torch.float
            self.input_normalizer = MBRLNormalizer(input_dim, self._device, dtype=dtype)

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_normalizer is None:
            return x
        if x.ndim == 1:
            x = x.unsqueeze(0)
            x_norm = self.input_normalizer.normalize(x).float()
            return x_norm.squeeze(0)
        return self.input_normalizer.normalize(x).float()

    def update_normalizer(self, obs: np.ndarray, actions: np.ndarray) -> None:
        if self.input_normalizer is None:
            return
        obs_arr = np.asarray(obs)
        act_arr = np.asarray(actions)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr.reshape(1, -1)
        if act_arr.ndim == 1:
            act_arr = act_arr.reshape(1, -1)
        if obs_arr.shape[0] != act_arr.shape[0]:
            raise ValueError("PETS normalizer expects matching batch sizes for obs and actions.")
        model_in = np.concatenate([obs_arr, act_arr], axis=-1)
        self.input_normalizer.update_stats(model_in)

    def sample_eps(self):
        """Return a model index for fixed-model rollouts."""
        if self.ensemble_size <= 1:
            return 0
        return random.randrange(self.ensemble_size)

    def _select_model(self, eps_cache=None, sample_theta=True):
        if self.ensemble_size <= 1:
            return 0
        if not sample_theta:
            return None
        if eps_cache is None:
            return random.randrange(self.ensemble_size)
        return int(eps_cache)

    def forward_all_models(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = self._normalize_input(x)
        mu, logvar = self.model(x, use_propagation=False)
        if mu.dim() == 2:
            mu = mu.unsqueeze(0)
            if logvar is not None:
                logvar = logvar.unsqueeze(0)
        if self.predict_delta:
            mu = mu + s.unsqueeze(0)
        return mu, logvar

    def forward(self, s, a, sample_theta=True, eps_cache=None):
        mu_stack, logvar_stack = self.forward_all_models(s, a)
        if not sample_theta or self.ensemble_size <= 1:
            mu = mu_stack.mean(dim=0)
            logvar = logvar_stack.mean(dim=0) if logvar_stack is not None else None
            return mu, logvar
        idx = self._select_model(eps_cache=eps_cache, sample_theta=sample_theta)
        if idx is None:
            mu = mu_stack.mean(dim=0)
            logvar = logvar_stack.mean(dim=0) if logvar_stack is not None else None
            return mu, logvar
        mu = mu_stack[idx]
        logvar = logvar_stack[idx] if logvar_stack is not None else None
        return mu, logvar

    def kl_loss(self):
        return torch.zeros((), device=self._device)

    def sample_next_state(self, s, a, eps_cache=None):
        mu, logvar = self.forward(s, a, sample_theta=True, eps_cache=eps_cache)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps


# ======================================================================
#  Simple Replay Buffer
# ======================================================================

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, o, a, o_next, r, done):
        self.buffer.append((o, a, o_next, r, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        o, a, o_next, r, d = zip(*batch)
        return (
            torch.tensor(np.array(o), dtype=torch.float32),
            torch.tensor(np.array(a), dtype=torch.float32),
            torch.tensor(np.array(o_next), dtype=torch.float32),
            torch.tensor(np.array(r), dtype=torch.float32),
            torch.tensor(np.array(d), dtype=torch.float32),
        )

    def get_all(self):
        if not self.buffer:
            return None
        o, a, o_next, r, d = zip(*self.buffer)
        return (
            np.asarray(o, dtype=np.float32),
            np.asarray(a, dtype=np.float32),
            np.asarray(o_next, dtype=np.float32),
            np.asarray(r, dtype=np.float32),
            np.asarray(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ======================================================================
#  Latent Scaling Active Inference Agent
# ======================================================================

class AIFAgent(nn.Module):
    """
    Latent-state (partially observed) variant.

    - Encoder q_phi(s_t | o_t)
    - Decoder p_lambda(o_t | s_t)
    - Bayesian dynamics p_theta(s_{t+1} | s_t, a_t)
    - Preferences p_pref(o_t) used for action selection (no reward regression)
    - CEM planning in latent space with MC over theta.
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        action_low,
        action_high,
        latent_dim=None,
        device="cpu",
        hidden_dim=128,
        gamma=0.99,
        cem_num_samples=64,
        cem_num_elites=6,
        cem_num_iters=5,
        cem_horizon=20,
        mc_num_models=5,
        mc_num_trajectories=3,
        kl_theta_beta=1.0,
        policy_mode="cem",
        info_gain_weight: float = 0.0,
        info_gain_mode: str = "ensemble_var",
        preference_mean=None,
        preference_std=None,
        preference_mode: str = "gaussian",
        pref_target_weight: float = 0.0,
        pref_target_scale: float = 0.1,
        preference_linear_scale: float = 1.0,
        log_efe_terms: bool = False,
        deterministic: bool = False,
        policy_plan_horizon: Optional[int] = None,
        policy_plan_samples: Optional[int] = None,
        policy_recompute_freq: int = 1,
        policy_cem_elites: Optional[int] = None,
        policy_elite_temperature: float = 0.0,
        policy_update_steps_per_plan: int = 1,
        policy_kl_beta: float = 1e-4,
        policy_std_penalty_weight: float = 0.0,
        policy_posterior_mc_samples: int = 4,
        policy_efe_baseline_momentum: float = 0.9,
        policy_plan_action_rollouts: int = 1,
        loss_weight_kl_state: float = 1.0,
        loss_weight_nll_obs: float = 1.0,
        use_mse_loss: bool = False,
        model_train_epochs: int = 1,
        model_val_ratio: float = 0.1,
        model_bootstrap: bool = True,
        model_bootstrap_permutes: bool = True,
        model_shuffle_each_epoch: bool = True,
        model_lr: float = 3e-4,
        model_wd: float = 0.0,
        logvar_reg_weight: float = 0.01,
        fully_observable_mdp: bool = False,
        world_model_type: Optional[str] = None,
        dynamics_ensemble_size: int = 1,
        dynamics_num_layers: int = 3,
        dynamics_target_is_delta: bool = True,
        dynamics_logvar_min: float = -10.0,
        dynamics_logvar_max: float = 2.0,
        mbrl_optimizer_cfg: Optional[Dict[str, Dict[str, Any]]] = None,
        normalize_inputs: bool = False,
        normalize_double_precision: bool = False,
        meta_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.normalize_inputs = bool(normalize_inputs)

        # Support vector-valued action bounds (from gymnasium Box spaces).
        action_low = np.array(action_low, dtype=np.float32)
        action_high = np.array(action_high, dtype=np.float32)
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)

        normalized_world_model = self._normalize_world_model_type(world_model_type)
        if normalized_world_model is None:
            normalized_world_model = "bayesian" if bool(fully_observable_mdp) else "vae"
        if normalized_world_model not in ("vae", "bayesian", "pets"):
            raise ValueError(
                "world_model_type must be one of: 'vae', 'bayesian', or 'pets'. "
                f"Got {world_model_type}."
            )
        self.world_model_type = normalized_world_model
        self.fully_observable_mdp = normalized_world_model != "vae"
        bayes_normalize_inputs = self.normalize_inputs and normalized_world_model == "bayesian"
        if self.fully_observable_mdp:
            self.latent_dim = obs_dim
            self.encoder = None
            self.decoder = None
            if normalized_world_model == "pets":
                self.dynamics = PETSDynamicsModel(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_dim=hidden_dim,
                    num_layers=dynamics_num_layers,
                    ensemble_size=dynamics_ensemble_size,
                    predict_delta=dynamics_target_is_delta,
                    logvar_min=dynamics_logvar_min,
                    logvar_max=dynamics_logvar_max,
                    device=self.device,
                    normalize_inputs=self.normalize_inputs,
                    normalize_double_precision=normalize_double_precision,
                ).to(self.device)
            else:
                self.dynamics = BayesianDynamicsModel(
                    latent_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_dim=hidden_dim,
                    deterministic=False,
                    predict_delta=dynamics_target_is_delta,
                    logvar_min=dynamics_logvar_min,
                    logvar_max=dynamics_logvar_max,
                    device=self.device,
                    normalize_inputs=bayes_normalize_inputs,
                    normalize_double_precision=normalize_double_precision,
                ).to(self.device)
        else:
            if latent_dim is None:
                latent_dim = obs_dim
            self.latent_dim = latent_dim
            self.encoder = Encoder(obs_dim, latent_dim, hidden_dim).to(self.device)
            self.decoder = Decoder(latent_dim, obs_dim, hidden_dim).to(self.device)
            self.dynamics = BayesianDynamicsModel(
                latent_dim,
                action_dim,
                hidden_dim,
                deterministic=False,
                logvar_min=dynamics_logvar_min,
                logvar_max=dynamics_logvar_max,
                device=self.device,
                normalize_inputs=False,
                normalize_double_precision=normalize_double_precision,
            ).to(self.device)

        self.policy_net = PolicyNet(self.latent_dim, action_dim, hidden_dim, deterministic=deterministic).to(self.device)
        self._world_model_modules = (
            [self.dynamics] if self.fully_observable_mdp else [self.encoder, self.decoder, self.dynamics]
        )
        self.optimizer = torch.optim.Adam(
            [p for module in self._world_model_modules for p in module.parameters()],
            lr=float(model_lr),
            weight_decay=float(model_wd),
        )
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=3e-4)

        self.gamma = gamma

        # CEM hyperparameters
        self.cem_num_samples = cem_num_samples
        self.cem_num_elites = cem_num_elites
        self.cem_num_iters = cem_num_iters
        self.cem_horizon = cem_horizon

        # Monte Carlo parameters for planning
        self.mc_num_models = mc_num_models   # B: number of theta samples
        self.mc_num_trajectories = mc_num_trajectories  # J: trajectories per theta
        self.kl_theta_beta = kl_theta_beta
        self.policy_mode = policy_mode
        self.info_gain_weight = float(info_gain_weight)
        self.info_gain_mode = self._normalize_info_gain_mode(info_gain_mode)
        self.log_efe_terms = bool(log_efe_terms)
        self.deterministic = bool(deterministic)
        self._warned_info_gain_unavailable = False
        
        # Loss weights for world model training
        self.loss_weight_kl_state = float(loss_weight_kl_state)
        self.loss_weight_nll_obs = float(loss_weight_nll_obs)
        self.use_mse_loss = bool(use_mse_loss)
        self.model_train_epochs = max(1, int(model_train_epochs))
        self.model_val_ratio = float(model_val_ratio)
        self.model_bootstrap = bool(model_bootstrap)
        self.model_bootstrap_permutes = bool(model_bootstrap_permutes)
        self.model_shuffle_each_epoch = bool(model_shuffle_each_epoch)
        self.logvar_reg_weight = float(logvar_reg_weight)
        self.policy_plan_horizon = policy_plan_horizon or cem_horizon
        self.policy_plan_samples = int(policy_plan_samples) if policy_plan_samples is not None else int(cem_num_samples)
        self.policy_recompute_freq = max(1, int(policy_recompute_freq))
        self._cached_policy_action: Optional[torch.Tensor] = None
        self._cached_policy_action_seq: Optional[torch.Tensor] = None
        self._policy_replan_counter = 0
        self._cached_policy_score: Optional[float] = None
        self._last_action_logprob: Optional[torch.Tensor] = None
        self._last_policy_bc_info: Optional[Dict[str, float]] = None
        self._last_plan_sequence: Optional[torch.Tensor] = None
        self.policy_cem_elites = int(policy_cem_elites) if policy_cem_elites is not None else int(cem_num_elites)
        self.policy_elite_temperature = float(policy_elite_temperature)
        self.policy_update_steps_per_plan = max(0, int(policy_update_steps_per_plan))
        self.policy_kl_beta = float(policy_kl_beta)
        self.policy_std_penalty_weight = float(policy_std_penalty_weight)
        self.policy_posterior_mc_samples = max(1, int(policy_posterior_mc_samples))
        self.policy_plan_action_rollouts = max(1, int(policy_plan_action_rollouts))
        # Running baseline for single-sample EFE updates to avoid zero advantage.
        self._policy_efe_baseline: Optional[torch.Tensor] = None
        self._policy_efe_baseline_momentum: float = float(policy_efe_baseline_momentum)
        # Track planner score extrema for logging.
        self._last_plan_score_best: Optional[float] = None
        self._last_plan_score_worst: Optional[float] = None
        # Optional MBRL optimizer settings.
        self._mbrl_optimizer_cfg: Dict[str, Dict[str, Any]] = mbrl_optimizer_cfg or {}
        self._mbrl_optimizer = None
        self._mbrl_optimizer_name: Optional[str] = None
        self._mbrl_optimizer_horizon: Optional[int] = None
        self._mbrl_prev_solution = None

        # Preferences: diagonal Gaussian or linear distance-based over observations.
        self.preference_mode = str(preference_mode).lower()
        if self.preference_mode not in ["gaussian", "linear"]:
            raise ValueError(f"preference_mode must be 'gaussian' or 'linear', got {self.preference_mode}")
        self.preference_mean, self.preference_logvar = self._init_preferences(
            preference_mean, preference_std
        )
        # Linear preference scaling: reward = -preference_linear_scale * distance
        self.preference_linear_scale = float(preference_linear_scale)
        # Optional extra probability mass on the exact target (sharper component).
        self.pref_target_weight = float(pref_target_weight)
        self.pref_target_scale = float(pref_target_scale)

        self._meta_controller = None
        self._last_meta_info = None
        self._cached_meta_override: Optional[Dict[str, Any]] = None
        self._cached_meta_info: Optional[Dict[str, Any]] = None
        self._meta_recompute_freq = max(1, int(meta_cfg.get("recompute_freq", 1))) if meta_cfg else 1
        self._meta_step_counter = 0
        self._meta_needs_recompute = False
        self._meta_selection_count = 0
        if MetaPlanningController is not None and meta_cfg is not None:
            self._meta_controller = MetaPlanningController(self, meta_cfg.get("modes"), meta_cfg)


    def _init_preferences(self, pref_mean, pref_std):
        """
        Initialize diagonal Gaussian preferences p_pref(o) with provided mean/std.
        Accepts scalars or array-like; broadcasts to observation dimension.
        """
        mean = np.zeros(self.obs_dim, dtype=np.float32) if pref_mean is None else np.asarray(pref_mean, dtype=np.float32).flatten()
        std = np.ones(self.obs_dim, dtype=np.float32) if pref_std is None else np.asarray(pref_std, dtype=np.float32).flatten()

        if mean.size == 1:
            mean = np.repeat(mean, self.obs_dim)
        if std.size == 1:
            std = np.repeat(std, self.obs_dim)
        mean = mean[: self.obs_dim]
        std = np.clip(std[: self.obs_dim], 1e-5, None)

        mean_t = torch.as_tensor(mean, dtype=torch.float32, device=self.device)
        logvar_t = torch.as_tensor(2 * np.log(std), dtype=torch.float32, device=self.device)
        return mean_t, logvar_t

    def set_preference_mean(self, pref_vec: np.ndarray, pref_mask: Optional[np.ndarray] = None):
        """
        Update the preference mean to target the current goal observation.
        """
        if pref_vec is None:
            return
        pref_vec = np.asarray(pref_vec, dtype=np.float32).flatten()
        pref_vec = pref_vec[: self.obs_dim]
        self.preference_mean = torch.as_tensor(pref_vec, dtype=torch.float32, device=self.device)
        if pref_mask is not None:
            mask = np.asarray(pref_mask, dtype=np.float32).flatten()
            mask = mask[: self.obs_dim]
            self.preference_mask = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
        elif hasattr(self, "preference_mask"):
            del self.preference_mask

    def _compute_neg_extrinsic_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute negative extrinsic value: -E[ln p(o|C)].

        This is the negative log probability of observations under the preference
        distribution. Higher values mean observations are further from preferences.

        In the EFE formula: G = -extrinsic_value - info_gain
        This function returns -extrinsic_value (i.e., the negative pragmatic value).

        Supports two modes:
        - "gaussian": Based on Gaussian log-likelihood with learned variance
        - "linear": Based on L2 distance to goal with fixed scaling
        """
        delta = obs - self.preference_mean
        pref_mask = getattr(self, "preference_mask", None)
        if pref_mask is not None:
            delta = delta * pref_mask
            logvar_pref = self.preference_logvar * pref_mask
        else:
            logvar_pref = self.preference_logvar

        # Linear distance-based preference
        # This was just for testing, to make it equivalent to the
        # "normal" distance-based reward.
        if self.preference_mode == "linear":
            # L2 distance scaled by preference_linear_scale
            distance = torch.norm(delta, p=2, dim=-1)
            return self.preference_linear_scale * distance

        # Gaussian preference (default)
        var_pref = torch.exp(logvar_pref)
        neg_extrinsic_base = (delta.pow(2) / var_pref).sum(dim=-1)

        weight = float(self.pref_target_weight)
        if weight <= 0.0:
            return neg_extrinsic_base
        weight = min(max(weight, 0.0), 1.0 - 1e-6)
        scale = max(float(self.pref_target_scale), 1e-6)
        var_target = var_pref * (scale ** 2)
        logvar_target = self.preference_logvar + (2.0 * math.log(scale))
        if pref_mask is not None:
            logvar_target = logvar_target * pref_mask

        logp_base = -0.5 * ((delta.pow(2) / var_pref) + logvar_pref).sum(dim=-1)
        logp_target = -0.5 * ((delta.pow(2) / var_target) + logvar_target).sum(dim=-1)
        logp_mix = torch.logsumexp(
            torch.stack(
                [
                    math.log(1.0 - weight) + logp_base,
                    math.log(weight) + logp_target,
                ],
                dim=0,
            ),
            dim=0,
        )
        return -logp_mix

    @staticmethod
    def _normalize_world_model_type(world_model_type: Optional[str]) -> Optional[str]:
        if world_model_type is None:
            return None
        key = str(world_model_type).strip().lower()
        aliases = {
            "vae": "vae",
            "latent": "vae",
            "latent_vae": "vae",
            "bayesian": "bayesian",
            "bayes": "bayesian",
            "fully_observable": "bayesian",
            "fully_observable_bayesian": "bayesian",
            "obs_bayesian": "bayesian",
            "pets": "pets",
            "pets_style": "pets",
            "pets_ensemble": "pets",
            "ensemble": "pets",
        }
        return aliases.get(key, key)

    @staticmethod
    def _normalize_info_gain_mode(mode: Optional[str]) -> str:
        mode = str(mode or "ensemble_var").lower()
        if mode not in ("logvar", "ensemble_var", "both"):
            raise ValueError(f"Unknown info_gain_mode '{mode}'. Expected logvar, ensemble_var, or both.")
        return mode

    def _use_ensemble_info_gain(self) -> bool:
        """Check if ensemble disagreement should be used for info gain computation."""
        if self.info_gain_weight <= 0.0:
            return False
        if self.info_gain_mode not in ("ensemble_var", "both"):
            return False
        if isinstance(self.dynamics, PETSDynamicsModel):
            return int(getattr(self.dynamics, "ensemble_size", 1)) > 1
        if isinstance(self.dynamics, BayesianDynamicsModel):
            return max(1, int(self.mc_num_models or 1)) > 1
        return False

    @staticmethod
    def _ensemble_disagreement_from_mu_stack(mu_stack: torch.Tensor) -> torch.Tensor:
        """Compute ensemble disagreement (variance of means) as proxy for parameter info gain."""
        return mu_stack.var(dim=0, unbiased=False).mean(dim=-1)

    def _theta_mu_logvar_stack(self, s, a, num_models=None, theta_eps_list=None):
        if isinstance(self.dynamics, PETSDynamicsModel):
            mu_stack, logvar_stack = self.dynamics.forward_all_models(s, a)
            return mu_stack, logvar_stack, None
        if not isinstance(self.dynamics, BayesianDynamicsModel):
            return None, None, None
        if theta_eps_list is not None and len(theta_eps_list) > 0:
            num_models = len(theta_eps_list)
        else:
            num_models = max(1, int(num_models or self.mc_num_models or 1))
        if num_models <= 1:
            return None, None, None
        if theta_eps_list is None:
            theta_eps_list = [self.dynamics.sample_eps() for _ in range(num_models)]
        mu_list = []
        logvar_list = []
        for eps in theta_eps_list:
            mu_k, logvar_k = self.dynamics(s, a, sample_theta=True, eps_cache=eps)
            mu_list.append(mu_k)
            logvar_list.append(logvar_k)
        mu_stack = torch.stack(mu_list, dim=0)
        logvar_stack = torch.stack(logvar_list, dim=0) if logvar_list else None
        return mu_stack, logvar_stack, theta_eps_list

    # ------------------------------------------------------------------
    #  Training: free-energy-like loss for latent model
    # ------------------------------------------------------------------

    @staticmethod
    def _kl_gaussian_diagonal(mu_q, logvar_q, mu_p, logvar_p):
        """
        KL( N(mu_q, diag(exp(logvar_q))) || N(mu_p, diag(exp(logvar_p))) )
        returns: [B] (sum over dims)
        """
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        term1 = logvar_p - logvar_q
        term2 = (var_q + (mu_q - mu_p).pow(2)) / var_p
        kl_per_dim = 0.5 * (term1 + term2 - 1.0)
        return kl_per_dim.sum(dim=-1)  # [B]

    def _obs_loss_fully_observable(
        self,
        o_prev: torch.Tensor,
        a_prev: torch.Tensor,
        o_t: torch.Tensor,
        model_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute observation loss for a single ensemble member (or averaged model).
        """
        eps_cache = model_idx
        if isinstance(self.dynamics, BayesianDynamicsModel) and not isinstance(model_idx, dict):
            eps_cache = None
        mu_next, logvar_next = self.dynamics(o_prev, a_prev, sample_theta=True, eps_cache=eps_cache)
        diff = o_t - mu_next
        if self.use_mse_loss:
            return (diff.pow(2)).sum(dim=-1).mean()
        var_next = torch.exp(logvar_next)
        return 0.5 * (logvar_next + math.log(2 * math.pi) + diff.pow(2) / var_next).sum(dim=-1).mean()

    def free_energy_loss(self, batch):
        """
        Implements a sampled version of:

        F_t ~= E_{q(theta)} E_{q(s_{t-1}|o_{t-1})}
                KL[ q(s_t|o_t) || p(s_t|s_{t-1}, a_{t-1}, theta) ]
              + KL(q(theta) || p(theta))
              - E_{q(s_t|o_t)} [ log p(o_t | s_t) + log p(r_t | s_t) ]
        """
        o_prev, a_prev, o_t, r_t, done = batch
        o_prev = o_prev.to(self.device)
        a_prev = a_prev.to(self.device)
        o_t = o_t.to(self.device)
        _ = r_t.to(self.device)  # kept for compatibility; preferences replace reward usage

        batch_size = o_t.shape[0]

        if self.fully_observable_mdp:
            if isinstance(self.dynamics, PETSDynamicsModel):
                ensemble_size = getattr(self.dynamics, "ensemble_size", 1)
                obs_loss_terms = []
                if ensemble_size > 1:
                    for model_idx in range(ensemble_size):
                        mu_next, logvar_next = self.dynamics(
                            o_prev, a_prev, sample_theta=True, eps_cache=model_idx
                        )
                        diff = o_t - mu_next
                        if self.use_mse_loss:
                            # MSE: mean squared error
                            obs_loss = (diff.pow(2)).sum(dim=-1).mean()
                        else:
                            # NLL: Gaussian negative log-likelihood
                            var_next = torch.exp(logvar_next)
                            obs_loss = 0.5 * (
                                logvar_next + math.log(2 * math.pi) + diff.pow(2) / var_next
                            ).sum(dim=-1).mean()
                        obs_loss_terms.append(obs_loss)
                    obs_loss_final = torch.stack(obs_loss_terms).mean()
                else:
                    mu_next, logvar_next = self.dynamics(o_prev, a_prev, sample_theta=True, eps_cache=None)
                    diff = o_t - mu_next
                    if self.use_mse_loss:
                        # MSE: mean squared error
                        obs_loss_final = (diff.pow(2)).sum(dim=-1).mean()
                    else:
                        # NLL: Gaussian negative log-likelihood
                        var_next = torch.exp(logvar_next)
                        obs_loss_final = 0.5 * (
                            logvar_next + math.log(2 * math.pi) + diff.pow(2) / var_next
                        ).sum(dim=-1).mean()
                kl_theta = (self.dynamics.kl_loss() / batch_size)
                kl_state = torch.zeros((), device=self.device)
                logvar_reg = torch.zeros((), device=self.device)
                model = getattr(self.dynamics, "model", None)
                if model is not None and hasattr(model, "max_logvar") and hasattr(model, "min_logvar"):
                    logvar_reg = self.logvar_reg_weight * (model.max_logvar.sum() - model.min_logvar.sum())
                F = (self.kl_theta_beta * kl_theta + self.loss_weight_nll_obs * obs_loss_final + logvar_reg)
                obs_loss_key = "mse" if self.use_mse_loss else "nll_obs"
                info = {
                    "kl_state": kl_state.item() * self.loss_weight_kl_state,
                    "kl_theta": kl_theta.item() * self.kl_theta_beta,
                    obs_loss_key: obs_loss_final.item() * self.loss_weight_nll_obs,
                    "logvar_reg": logvar_reg.item(),
                    "F": F.item(),
                }
                return F, info

            mu_next, logvar_next = self.dynamics(o_prev, a_prev, sample_theta=True, eps_cache=None)
            diff = o_t - mu_next
            if self.use_mse_loss:
                obs_loss_final = (diff.pow(2)).sum(dim=-1).mean()
            else:
                var_next = torch.exp(logvar_next)
                obs_loss_final = 0.5 * (
                    logvar_next + math.log(2 * math.pi) + diff.pow(2) / var_next
                ).sum(dim=-1).mean()
            kl_theta = (self.dynamics.kl_loss() / batch_size)
            kl_state = torch.zeros((), device=self.device)
            logvar_reg = torch.zeros((), device=self.device)
            F = (self.kl_theta_beta * kl_theta + self.loss_weight_nll_obs * obs_loss_final + logvar_reg)
            obs_loss_key = "mse" if self.use_mse_loss else "nll_obs"
            info = {
                "kl_state": kl_state.item() * self.loss_weight_kl_state,
                "kl_theta": kl_theta.item() * self.kl_theta_beta,
                obs_loss_key: obs_loss_final.item() * self.loss_weight_nll_obs,
                "logvar_reg": logvar_reg.item(),
                "F": F.item(),
            }
            return F, info

        # Encode q(s_{t-1}|o_{t-1}) and q(s_t|o_t)
        mu_prev, logvar_prev = self.encoder(o_prev)
        mu_t, logvar_t = self.encoder(o_t)

        # Sample s_{t-1} from q(s_{t-1}|o_{t-1})
        eps_prev = torch.randn_like(mu_prev)
        s_prev = mu_prev + torch.exp(0.5 * logvar_prev) * eps_prev

        # ----- State KL term: E_{q(theta)} KL[q(s_t|o_t) || p(s_t|s_{t-1}, a_{t-1}, theta)] -----
        num_theta_samples = 3
        kl_state_total = 0.0
        for _ in range(num_theta_samples):
            dyn_eps = self.dynamics.sample_eps()
            mu_trans, logvar_trans = self.dynamics(s_prev, a_prev, sample_theta=True, eps_cache=dyn_eps)
            kl_s = self._kl_gaussian_diagonal(mu_q=mu_t,
                                              logvar_q=logvar_t,
                                              mu_p=mu_trans,
                                              logvar_p=logvar_trans)  # [B]
            kl_state_total += kl_s.mean()
        kl_state = kl_state_total / num_theta_samples

        # ----- Observation Loss: -E_{q(s_t|o_t)} log p(o_t | s_t) -----
        # sample s_t from q(s_t|o_t)
        eps_t = torch.randn_like(mu_t)
        s_t_sample = mu_t + torch.exp(0.5 * logvar_t) * eps_t

        mu_o, logvar_o = self.decoder(s_t_sample)
        diff_o = o_t - mu_o
        if self.use_mse_loss:
            # MSE: mean squared error
            nll_obs = (diff_o.pow(2)).sum(dim=-1).mean()
        else:
            # NLL: Gaussian negative log-likelihood
            var_o = torch.exp(logvar_o)
            nll_obs = 0.5 * (
                logvar_o + math.log(2 * math.pi) + diff_o.pow(2) / var_o
            ).sum(dim=-1).mean()  # scalar

        # ----- Parameter KL: KL(q(theta) || p(theta)) -----
        kl_theta =  (self.dynamics.kl_loss() / batch_size)

        # Total free energy-like loss with weighted terms
        F = (self.loss_weight_kl_state * kl_state 
             + self.kl_theta_beta * kl_theta
             + self.loss_weight_nll_obs * nll_obs)
        obs_loss_key = "mse" if self.use_mse_loss else "nll_obs"
        info = {
            "kl_state": kl_state.item() * self.loss_weight_kl_state,
            "kl_theta": kl_theta.item() * self.kl_theta_beta,
            obs_loss_key: nll_obs.item() * self.loss_weight_nll_obs,
            "F": F.item(),
        }
        return F, info

    def train_world_model_epochs(self, replay_buffer, batch_size=64, num_epochs=None):
        """
        Train the world model for multiple epochs, sampling fresh batches each epoch.
        Similar to PETS ModelTrainer approach.
        
        Args:
            replay_buffer: replay buffer to sample from
            batch_size: batch size for sampling
            num_epochs: number of epochs to train (uses self.model_train_epochs if None)
        
        Returns:
            List of info dicts from each epoch
        """
        if len(replay_buffer) < batch_size:
            return None

        if num_epochs is None:
            num_epochs = self.model_train_epochs

        if self.normalize_inputs and self.fully_observable_mdp and hasattr(self.dynamics, "update_normalizer"):
            self._update_input_normalizer(replay_buffer)

        data = replay_buffer.get_all()
        if not data:
            return None
        o, a, o_next, r, d = data
        num_samples = o.shape[0]
        if num_samples < batch_size:
            return None

        val_ratio = max(0.0, min(float(self.model_val_ratio), 0.9))
        indices = np.arange(num_samples)
        if self.model_shuffle_each_epoch:
            indices = np.random.permutation(indices)
        val_size = int(num_samples * val_ratio)
        if val_size > 0 and num_samples - val_size >= 1:
            val_idx = indices[:val_size]
            train_idx = indices[val_size:]
        else:
            val_idx = None
            train_idx = indices
        if train_idx.size == 0:
            return None

        def _batch_to_tensors(batch_ids):
            return (
                torch.as_tensor(o[batch_ids], dtype=torch.float32, device=self.device),
                torch.as_tensor(a[batch_ids], dtype=torch.float32, device=self.device),
                torch.as_tensor(o_next[batch_ids], dtype=torch.float32, device=self.device),
                torch.as_tensor(r[batch_ids], dtype=torch.float32, device=self.device),
                torch.as_tensor(d[batch_ids], dtype=torch.float32, device=self.device),
            )

        ensemble_size = 1
        if self.fully_observable_mdp and isinstance(self.dynamics, PETSDynamicsModel):
            ensemble_size = max(1, int(getattr(self.dynamics, "ensemble_size", 1)))
        use_bootstrap = bool(self.model_bootstrap) and ensemble_size > 1
        rng = np.random.default_rng()
        num_batches = max(1, math.ceil(len(train_idx) / batch_size))

        epoch_infos = []
        for _ in range(num_epochs):
            if self.model_shuffle_each_epoch:
                base_order = rng.permutation(len(train_idx))
            else:
                base_order = np.arange(len(train_idx))

            member_indices = None
            if use_bootstrap:
                if self.model_bootstrap_permutes:
                    member_indices = [rng.permutation(len(train_idx)) for _ in range(ensemble_size)]
                else:
                    member_indices = [
                        rng.choice(len(train_idx), size=len(train_idx), replace=True)
                        for _ in range(ensemble_size)
                    ]

            batch_infos = []
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, len(train_idx))
                if start >= end:
                    continue

                self.optimizer.zero_grad()
                if use_bootstrap and member_indices is not None:
                    obs_losses = []
                    batch_size_local = None
                    for model_idx in range(ensemble_size):
                        member_order = member_indices[model_idx]
                        batch_pos = member_order[start:end]
                        batch_ids = train_idx[batch_pos]
                        o_prev_t, a_prev_t, o_t_t, _, _ = _batch_to_tensors(batch_ids)
                        batch_size_local = int(o_prev_t.shape[0])
                        obs_losses.append(
                            self._obs_loss_fully_observable(o_prev_t, a_prev_t, o_t_t, model_idx=model_idx)
                        )
                    obs_loss_final = torch.stack(obs_losses).mean()
                    kl_theta = self.dynamics.kl_loss() / max(1, batch_size_local or batch_size)
                    logvar_reg = torch.zeros((), device=self.device)
                    model = getattr(self.dynamics, "model", None)
                    if model is not None and hasattr(model, "max_logvar") and hasattr(model, "min_logvar"):
                        logvar_reg = self.logvar_reg_weight * (model.max_logvar.sum() - model.min_logvar.sum())
                    F = (self.kl_theta_beta * kl_theta + self.loss_weight_nll_obs * obs_loss_final + logvar_reg)
                    obs_loss_key = "mse" if self.use_mse_loss else "nll_obs"
                    info = {
                        "kl_state": 0.0,
                        "kl_theta": kl_theta.item() * self.kl_theta_beta,
                        obs_loss_key: obs_loss_final.item() * self.loss_weight_nll_obs,
                        "logvar_reg": logvar_reg.item(),
                        "F": F.item(),
                    }
                else:
                    batch_ids = train_idx[base_order[start:end]]
                    batch = _batch_to_tensors(batch_ids)
                    F, info = self.free_energy_loss(batch)

                F.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for module in self._world_model_modules for p in module.parameters()],
                    max_norm=10.0,
                )
                self.optimizer.step()
                batch_infos.append(info)

            if not batch_infos:
                continue
            avg_info = {k: float(np.mean([bi[k] for bi in batch_infos if k in bi])) for k in batch_infos[0]}

            if val_idx is not None and val_idx.size > 0:
                val_batches = max(1, math.ceil(len(val_idx) / batch_size))
                val_infos = []
                with torch.no_grad():
                    for batch_idx in range(val_batches):
                        start = batch_idx * batch_size
                        end = min((batch_idx + 1) * batch_size, len(val_idx))
                        if start >= end:
                            continue
                        batch_ids = val_idx[start:end]
                        batch = _batch_to_tensors(batch_ids)
                        if use_bootstrap:
                            o_prev_t, a_prev_t, o_t_t, _, _ = batch
                            obs_losses = []
                            for model_idx in range(ensemble_size):
                                obs_losses.append(
                                    self._obs_loss_fully_observable(
                                        o_prev_t, a_prev_t, o_t_t, model_idx=model_idx
                                    )
                                )
                            obs_loss_final = torch.stack(obs_losses).mean()
                            kl_theta = self.dynamics.kl_loss() / max(1, o_prev_t.shape[0])
                            logvar_reg = torch.zeros((), device=self.device)
                            model = getattr(self.dynamics, "model", None)
                            if model is not None and hasattr(model, "max_logvar") and hasattr(model, "min_logvar"):
                                logvar_reg = self.logvar_reg_weight * (model.max_logvar.sum() - model.min_logvar.sum())
                            F = (self.kl_theta_beta * kl_theta + self.loss_weight_nll_obs * obs_loss_final + logvar_reg)
                            obs_loss_key = "mse" if self.use_mse_loss else "nll_obs"
                            val_info = {
                                "kl_state": 0.0,
                                "kl_theta": kl_theta.item() * self.kl_theta_beta,
                                obs_loss_key: obs_loss_final.item() * self.loss_weight_nll_obs,
                                "logvar_reg": logvar_reg.item(),
                                "F": F.item(),
                            }
                        else:
                            F, val_info = self.free_energy_loss(batch)
                        val_infos.append(val_info)
                if val_infos:
                    for key in val_infos[0]:
                        avg_info[f"val_{key}"] = float(np.mean([vi[key] for vi in val_infos if key in vi]))

            epoch_infos.append(avg_info)

        if not epoch_infos:
            return None
        avg_info = {}
        for key in epoch_infos[0].keys():
            avg_info[key] = float(np.mean([info[key] for info in epoch_infos if key in info]))
        return avg_info

    def update(self, replay_buffer, batch_size=64):
        """Legacy update method for single-step training. Uses epoch-based training if configured."""
        return self.train_world_model_epochs(replay_buffer, batch_size, num_epochs=self.model_train_epochs)

    def _update_input_normalizer(self, replay_buffer) -> None:
        if not hasattr(replay_buffer, "get_all"):
            return
        batch = replay_buffer.get_all()
        if not batch:
            return
        obs, actions, _, _, _ = batch
        if obs is None or actions is None:
            return
        if getattr(obs, "size", 0) == 0 or getattr(actions, "size", 0) == 0:
            return
        update_fn = getattr(self.dynamics, "update_normalizer", None)
        if not callable(update_fn):
            return
        update_fn(obs, actions)

    def _expected_free_energy_step(
        self,
        mu_o: torch.Tensor,
        logvar_o: torch.Tensor,
        logvar_next: Optional[torch.Tensor] = None,
        return_terms: bool = False,
        ensemble_disagreement: Optional[torch.Tensor] = None,
        mu_stack: Optional[torch.Tensor] = None,
        logvar_stack: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute expected free energy for action selection.

        -G = Extrinsic Value + State Info Gain + Parameter Info Gain

        Where:
        - Extrinsic Value = E_q(o|π)[ln p(o|C)]
          Expected log probability of observations under preference distribution.

        - State Info Gain = I(s; o | π) = H[q(o|π)] - E_q(s|π)[H[q(o|s,π)]]
          Approximated by average aleatoric uncertainty from model predictions.

        - Parameter Info Gain = I(θ; s | π) = H[q(s|π)] - E_q(θ)[H[q(s|π,θ)]]
          Approximated by ensemble disagreement (variance of means across models).

        Args:
            ensemble_disagreement: Precomputed variance of means across ensemble/samples.
            mu_stack: Stack of mean predictions from multiple models [K, B, dim].
            logvar_stack: Stack of logvar predictions from multiple models [K, B, dim].
        """
        # === Extrinsic Value: E_q(o|π)[ln p(o|C)] ===
        # For Gaussian p(o|C) = N(C, σ²_C) and q(o|π) = N(μ_o, σ²_o):
        # E[ln p(o|C)] = -0.5 * Σ_i [ ((μ_oi - Ci)² + σ²_oi) / σ²_Ci + ln(σ²_Ci) ]
        delta = mu_o - self.preference_mean
        pref_mask = getattr(self, "preference_mask", None)
        if pref_mask is not None:
            delta = delta * pref_mask
            logvar_pref = self.preference_logvar * pref_mask
            logvar_obs = logvar_o * pref_mask if logvar_o is not None else None
        else:
            logvar_pref = self.preference_logvar
            logvar_obs = logvar_o

        var_pref = torch.exp(logvar_pref)
        var_o = torch.exp(logvar_obs) if logvar_obs is not None else torch.zeros_like(var_pref)

        # E[(o - C)²] = (μ_o - C)² + σ²_o
        expected_sq_diff = delta.pow(2) + var_o

        # Extrinsic value = E[ln p(o|C)]
        extrinsic_value = -0.5 * (expected_sq_diff / var_pref + logvar_pref).sum(dim=-1)

        # === Information Gain Terms ===
        state_info_gain = torch.zeros_like(extrinsic_value)
        param_info_gain = torch.zeros_like(extrinsic_value)

        if self.info_gain_weight > 0.0:
            # Parameter Info Gain: I(θ; s | π) = H[q(s|π)] - E_q(θ)[H[q(s|π,θ)]]
            # Approximated by ensemble disagreement (variance of means across models)
            if ensemble_disagreement is not None:
                # PETS ensemble: disagreement already computed
                param_info_gain = self.info_gain_weight * ensemble_disagreement
            elif mu_stack is not None:
                # Bayesian model: compute from stacks
                # Var[μ] across samples = parameter uncertainty
                param_info_gain = self.info_gain_weight * mu_stack.var(dim=0, unbiased=False).mean(dim=-1)

            # State Info Gain: I(s; o | π) = H[q(o|π)] - E_q(s|π)[H[q(o|s,π)]]
            # Approximated by aleatoric uncertainty (average predictive variance)
            if logvar_stack is not None:
                # Average aleatoric uncertainty across ensemble/samples
                # E_q(θ)[σ²] = mean of individual model variances
                avg_aleatoric_var = torch.exp(logvar_stack).mean(dim=0).mean(dim=-1)
                state_info_gain = self.info_gain_weight * avg_aleatoric_var
            elif logvar_next is not None:
                # Single model: use predictive variance as state uncertainty
                state_info_gain = self.info_gain_weight * torch.exp(logvar_next).mean(dim=-1)

        # -G = Extrinsic + State Info Gain + Parameter Info Gain
        # G = -Extrinsic - State Info Gain - Parameter Info Gain
        info_gain = state_info_gain + param_info_gain
        efe = -extrinsic_value - info_gain

        if return_terms:
            # Return: efe, negative extrinsic value, total info gain
            return efe, -extrinsic_value, info_gain
        return efe

    def _policy_std_penalty(self) -> torch.Tensor:
        """
        Compute a penalty term that encourages the policy std to decrease.

        This creates gentle pressure for the policy to become more confident
        over time, enabling transition to habitual/reactive modes.
        """
        if self.policy_std_penalty_weight <= 0.0:
            return torch.tensor(0.0, device=self.device)
        # Get the log_std parameter (fixed, not state-dependent)
        log_std = self.policy_net.log_std
        std = torch.exp(log_std)
        # Penalty is mean std value - higher std = higher penalty
        penalty = self.policy_std_penalty_weight * std.mean()
        return penalty

    def update_policy_with_imagined_rollouts(
        self,
        replay_buffer,
        batch_size: int = 64,
        rollout_horizon: Optional[int] = None,
    ):
        """
        Train policy_net with imagined rollouts under the learned dynamics and preferences.
        Only policy_net parameters are updated.
        """
        if len(replay_buffer) < batch_size:
            return None

        if rollout_horizon is None:
            rollout_horizon = self.cem_horizon

        # Sample starting observations and encode to latent states (no grad to encoder).
        obs_batch, _, _, _, _ = replay_buffer.sample(batch_size)
        obs_batch = obs_batch.to(self.device)
        if self.fully_observable_mdp:
            s = obs_batch
        else:
            with torch.no_grad():
                mu, logvar = self.encoder(obs_batch)
                eps = torch.randn_like(mu)
                s = mu + torch.exp(0.5 * logvar) * eps

        efes = []
        log_probs = []
        discounts = torch.pow(self.gamma, torch.arange(rollout_horizon, device=self.device, dtype=torch.float32))
        neg_extrinsic_terms = [] if self.log_efe_terms else None
        info_gain_terms = [] if self.log_efe_terms else None
        policy_eps = self.policy_net.sample_eps()
        dynamics_eps = self.dynamics.sample_eps()
        use_ensemble_info_gain = self._use_ensemble_info_gain()
        use_bayes = isinstance(self.dynamics, BayesianDynamicsModel)
        num_models = max(1, int(self.mc_num_models or 1))
        theta_eps_list = None
        theta_model_idx = None
        if use_bayes and use_ensemble_info_gain and num_models > 1:
            theta_eps_list = [self.dynamics.sample_eps() for _ in range(num_models)]
            theta_model_idx = random.randrange(num_models)

        for t in range(rollout_horizon):
            mean, std = self.policy_net(s, eps_cache=policy_eps)
            a, log_prob, _ = self.policy_net.sample_action(
                s, self.action_low, self.action_high, eps_cache=policy_eps
            )
            log_probs.append(log_prob)
            with torch.no_grad():
                ensemble_disagr = None
                mu_stack = None
                logvar_stack = None
                if use_ensemble_info_gain:
                    mu_stack, logvar_stack, theta_eps_list = self._theta_mu_logvar_stack(
                        s, a, num_models=num_models, theta_eps_list=theta_eps_list
                    )
                    if mu_stack is not None:
                        ensemble_disagr = self._ensemble_disagreement_from_mu_stack(mu_stack)
                        if use_bayes:
                            idx = theta_model_idx if theta_model_idx is not None else 0
                            mu_next = mu_stack[int(idx)]
                            logvar_next = logvar_stack[int(idx)] if logvar_stack is not None else None
                        else:
                            model_idx = self.dynamics._select_model(eps_cache=dynamics_eps, sample_theta=True)
                            if model_idx is None:
                                mu_next = mu_stack.mean(dim=0)
                                logvar_next = logvar_stack.mean(dim=0) if logvar_stack is not None else None
                            else:
                                mu_next = mu_stack[int(model_idx)]
                                logvar_next = logvar_stack[int(model_idx)] if logvar_stack is not None else None
                    else:
                        mu_next, logvar_next = self.dynamics(s, a, sample_theta=True, eps_cache=dynamics_eps)
                else:
                    mu_next, logvar_next = self.dynamics(s, a, sample_theta=True, eps_cache=dynamics_eps)
                if logvar_next is None:
                    s = mu_next
                else:
                    std_next = torch.exp(0.5 * logvar_next)
                    s = mu_next + std_next * torch.randn_like(mu_next)
                if self.fully_observable_mdp:
                    mu_o, logvar_o = mu_next, logvar_next
                else:
                    mu_o, logvar_o = self.decoder(s)
            if self.log_efe_terms:
                efe_t, neg_extrinsic_t, info_gain_t = self._expected_free_energy_step(
                    mu_o,
                    logvar_o,
                    logvar_next,
                    return_terms=True,
                    ensemble_disagreement=ensemble_disagr,
                    mu_stack=mu_stack,
                    logvar_stack=logvar_stack,
                )
                neg_extrinsic_terms.append(neg_extrinsic_t)
                info_gain_terms.append(info_gain_t)
            else:
                efe_t = self._expected_free_energy_step(
                    mu_o, logvar_o, logvar_next,
                    ensemble_disagreement=ensemble_disagr,
                    mu_stack=mu_stack,
                    logvar_stack=logvar_stack,
                )
            efes.append(efe_t)

        efes = torch.stack(efes, dim=1)  # [B, H]
        log_probs = torch.stack(log_probs, dim=1)  # [B, H]
        discounted_efe = (efes * discounts.unsqueeze(0)).sum(dim=1)  # [B]
        discounted_neg_extrinsic = None
        discounted_info_gain = None
        if self.log_efe_terms:
            neg_extrinsic_stack = torch.stack(neg_extrinsic_terms, dim=1)  # [B, H]
            info_gain_stack = torch.stack(info_gain_terms, dim=1)  # [B, H]
            discounted_neg_extrinsic = (neg_extrinsic_stack * discounts.unsqueeze(0)).sum(dim=1)
            discounted_info_gain = (info_gain_stack * discounts.unsqueeze(0)).sum(dim=1)

        # REINFORCE-style objective to minimize expected free energy.
        adv = (discounted_efe - discounted_efe.mean()).detach()
        loss = (adv * log_probs.sum(dim=1)).mean()
        loss = loss + self.policy_kl_beta * self.policy_net.kl_loss()
        loss = loss + self._policy_std_penalty()
        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self._zero_non_policy_grads()
        self.policy_optimizer.step()

        out = {
            "policy_imag_efe": discounted_efe.mean().item(),
            "policy_imag_logprob": log_probs.sum(dim=1).mean().item(),
            "policy_std": self.policy_net.log_std.exp().mean().item(),
        }
        if self.log_efe_terms and discounted_neg_extrinsic is not None and discounted_info_gain is not None:
            out.update(
                {
                    "policy_imag_neg_extrinsic": discounted_neg_extrinsic.mean().item(),
                    "policy_imag_info_gain": discounted_info_gain.mean().item(),
                }
            )
        return out

    def update_policy_from_planned_action(
        self, obs_np: np.ndarray, planned_action: torch.Tensor, compute_weight: float = 1.0
    ):
        """
        Train policy_net to imitate the best action from planning rollouts.

        This is behavioral cloning from the planner: the policy learns to directly
        output actions that the deliberative planner would choose. Episodes with
        high extrinsic value (good performance) update the policy more strongly.

        Args:
            obs_np: Current observation [obs_dim]
            planned_action: Best first action from planning rollouts [action_dim]
            compute_weight: Scaling factor from extrinsic value (episode performance).
                High extrinsic value → stronger update; deterministic (weight=0) → skip.
        """
        if compute_weight <= 0.0:
            return None  # Skip for deterministic mode
        if obs_np is None or planned_action is None:
            return None

        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        target_action = planned_action.detach().unsqueeze(0) if planned_action.dim() == 1 else planned_action.detach()

        # Encode observation to latent state
        with torch.no_grad():
            if self.fully_observable_mdp:
                s = obs
            else:
                mu, logvar = self.encoder(obs)
                s = mu  # Use mean for stability

        # Get policy output
        policy_eps = self.policy_net.sample_eps()
        mean, std = self.policy_net(s, eps_cache=policy_eps)

        # Behavioral cloning loss: maximize log prob of planned action under policy
        log_prob = self.policy_net.log_prob_action(
            target_action, mean, std, self.action_low, self.action_high
        )

        # Scale loss by compute_weight: high extrinsic value → stronger habituation
        loss = -compute_weight * log_prob.mean()  # Negative because we maximize log prob
        loss = loss + compute_weight * self.policy_kl_beta * self.policy_net.kl_loss()
        loss = loss + compute_weight * self._policy_std_penalty()

        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self._zero_non_policy_grads()
        self.policy_optimizer.step()

        return {
            "policy_bc_logprob": log_prob.mean().item(),
            "policy_bc_compute_weight": compute_weight,
            "policy_std": self.policy_net.log_std.exp().mean().item(),
        }

    def update_policy_with_real_efe(
        self, obs_np=None, action_np=None, next_obs_np=None, compute_weight: float = 1.0
    ):
        """
        Train policy_net on real transitions by minimizing expected free energy.
        If next_obs_np is provided, compute the negative extrinsic value directly
        from the real next observation (masked preference distance) instead of
        relying on model predictions. Uses on-policy (latest) transition; off-policy
        replay is avoided.

        Args:
            compute_weight: Scaling factor for the policy update, based on extrinsic value
                (episode performance). High extrinsic value updates policy more strongly
                (habituating successful actions); deterministic mode (weight=0) skips the update.
        """
        if compute_weight <= 0.0:
            return None  # Skip update for deterministic/zero-compute plans
        if obs_np is None or action_np is None:
            return None

        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions = torch.as_tensor(action_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        policy_eps = self.policy_net.sample_eps()
        use_ensemble_info_gain = self._use_ensemble_info_gain()
        use_bayes = isinstance(self.dynamics, BayesianDynamicsModel)
        num_models = max(1, int(self.mc_num_models or 1))
        theta_eps_list = None
        theta_model_idx = None
        with torch.no_grad():
            if self.fully_observable_mdp:
                s = obs
            else:
                mu, logvar = self.encoder(obs)
                s = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

        # Prefer real next observation for EFE computation; fall back to model if absent.
        # For real observations: no prediction uncertainty, so info_gain = 0.
        efe = None
        neg_extrinsic = None
        info_gain = torch.zeros(1, device=self.device)
        if next_obs_np is not None:
            with torch.no_grad():
                obs_next = torch.as_tensor(next_obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                # Compute extrinsic value = E[ln p(o|C)] for real observation
                # Since observation is known, no variance: E[(o-C)²] = (o-C)²
                efe, neg_extrinsic, info_gain = self._expected_free_energy_step(
                    obs_next,
                    logvar_o=None,
                    logvar_next=None,
                    return_terms=True,
                    ensemble_disagreement=None,
                    mu_stack=None,
                    logvar_stack=None,
                )
        if efe is None:
            dynamics_eps = self.dynamics.sample_eps()
            with torch.no_grad():
                ensemble_disagr = None
                mu_stack = None
                logvar_stack = None
                if use_ensemble_info_gain:
                    if use_bayes and theta_eps_list is None and num_models > 1:
                        theta_eps_list = [self.dynamics.sample_eps() for _ in range(num_models)]
                        theta_model_idx = random.randrange(num_models)
                    mu_stack, logvar_stack, theta_eps_list = self._theta_mu_logvar_stack(
                        s, actions, num_models=num_models, theta_eps_list=theta_eps_list
                    )
                    if mu_stack is not None:
                        ensemble_disagr = self._ensemble_disagreement_from_mu_stack(mu_stack)
                        if use_bayes:
                            idx = theta_model_idx if theta_model_idx is not None else 0
                            mu_next = mu_stack[int(idx)]
                            logvar_next = logvar_stack[int(idx)] if logvar_stack is not None else None
                        else:
                            model_idx = self.dynamics._select_model(eps_cache=dynamics_eps, sample_theta=True)
                            if model_idx is None:
                                mu_next = mu_stack.mean(dim=0)
                                logvar_next = logvar_stack.mean(dim=0) if logvar_stack is not None else None
                            else:
                                mu_next = mu_stack[int(model_idx)]
                                logvar_next = logvar_stack[int(model_idx)] if logvar_stack is not None else None
                    else:
                        mu_next, logvar_next = self.dynamics(s, actions, sample_theta=True, eps_cache=dynamics_eps)
                else:
                    mu_next, logvar_next = self.dynamics(s, actions, sample_theta=True, eps_cache=dynamics_eps)
                if logvar_next is None:
                    s_pred = mu_next
                else:
                    std_next = torch.exp(0.5 * logvar_next)
                    s_pred = mu_next + std_next * torch.randn_like(mu_next)
                if self.fully_observable_mdp:
                    mu_o, logvar_o = mu_next, logvar_next
                else:
                    mu_o, logvar_o = self.decoder(s_pred)
                if self.log_efe_terms:
                    efe, neg_extrinsic, info_gain = self._expected_free_energy_step(
                        mu_o,
                        logvar_o,
                        logvar_next,
                        return_terms=True,
                        ensemble_disagreement=ensemble_disagr,
                        mu_stack=mu_stack,
                        logvar_stack=logvar_stack,
                    )
                else:
                    efe = self._expected_free_energy_step(
                        mu_o, logvar_o, logvar_next,
                        ensemble_disagreement=ensemble_disagr,
                        mu_stack=mu_stack,
                        logvar_stack=logvar_stack,
                    )

        mean, std = self.policy_net(s, eps_cache=policy_eps)
        log_prob = self.policy_net.log_prob_action(
            actions, mean, std, self.action_low, self.action_high
        )

        # Single-sample update: use an EMA baseline to avoid zero advantage.
        efe_detached = efe.detach()
        if self._policy_efe_baseline is None:
            self._policy_efe_baseline = efe_detached.mean()
        else:
            self._policy_efe_baseline = (
                self._policy_efe_baseline_momentum * self._policy_efe_baseline
                + (1.0 - self._policy_efe_baseline_momentum) * efe_detached.mean()
            )
        adv = (efe - self._policy_efe_baseline).detach()
        # Scale loss by compute_weight: high extrinsic value → stronger policy update
        # This habituates successful actions more than unsuccessful ones
        loss = compute_weight * (adv * log_prob).mean()
        loss = loss + compute_weight * self.policy_kl_beta * self.policy_net.kl_loss()
        loss = loss + compute_weight * self._policy_std_penalty()
        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self._zero_non_policy_grads()
        self.policy_optimizer.step()

        out = {
            "policy_real_efe": efe.mean().item(),
            "policy_real_logprob": log_prob.mean().item(),
            "policy_std": self.policy_net.log_std.exp().mean().item(),
            "policy_compute_weight": compute_weight,
        }
        if self.log_efe_terms:
            out.update(
                {
                    "policy_real_neg_extrinsic": neg_extrinsic.mean().item() if neg_extrinsic is not None else None,
                    "policy_real_info_gain": info_gain.mean().item() if info_gain is not None else 0.0,
                }
            )
        return out

    def evaluate_real_efe(self, obs_np=None, action_np=None):
        """
        Stateless EFE estimate on a single real transition (no gradients, no policy update).
        """
        if obs_np is None or action_np is None:
            return None

        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions = torch.as_tensor(action_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        use_ensemble_info_gain = self._use_ensemble_info_gain()
        use_bayes = isinstance(self.dynamics, BayesianDynamicsModel)
        num_models = max(1, int(self.mc_num_models or 1))
        theta_eps_list = None
        theta_model_idx = None

        with torch.no_grad():
            if self.fully_observable_mdp:
                s = obs
            else:
                mu, logvar = self.encoder(obs)
                s = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            dynamics_eps = self.dynamics.sample_eps()
            ensemble_disagr = None
            mu_stack = None
            logvar_stack = None
            if use_ensemble_info_gain:
                if use_bayes and theta_eps_list is None and num_models > 1:
                    theta_eps_list = [self.dynamics.sample_eps() for _ in range(num_models)]
                    theta_model_idx = random.randrange(num_models)
                mu_stack, logvar_stack, theta_eps_list = self._theta_mu_logvar_stack(
                    s, actions, num_models=num_models, theta_eps_list=theta_eps_list
                )
                if mu_stack is not None:
                    ensemble_disagr = self._ensemble_disagreement_from_mu_stack(mu_stack)
                    if use_bayes:
                        idx = theta_model_idx if theta_model_idx is not None else 0
                        mu_next = mu_stack[int(idx)]
                        logvar_next = logvar_stack[int(idx)] if logvar_stack is not None else None
                    else:
                        model_idx = self.dynamics._select_model(eps_cache=dynamics_eps, sample_theta=True)
                        if model_idx is None:
                            mu_next = mu_stack.mean(dim=0)
                            logvar_next = logvar_stack.mean(dim=0) if logvar_stack is not None else None
                        else:
                            mu_next = mu_stack[int(model_idx)]
                            logvar_next = logvar_stack[int(model_idx)] if logvar_stack is not None else None
                else:
                    mu_next, logvar_next = self.dynamics(s, actions, sample_theta=True, eps_cache=dynamics_eps)
            else:
                mu_next, logvar_next = self.dynamics(s, actions, sample_theta=True, eps_cache=dynamics_eps)
            if logvar_next is None:
                s_pred = mu_next
            else:
                std_next = torch.exp(0.5 * logvar_next)
                s_pred = mu_next + std_next * torch.randn_like(mu_next)
            if self.fully_observable_mdp:
                mu_o, logvar_o = mu_next, logvar_next
            else:
                mu_o, logvar_o = self.decoder(s_pred)
            if self.log_efe_terms:
                efe, neg_extrinsic, info_gain = self._expected_free_energy_step(
                    mu_o,
                    logvar_o,
                    logvar_next,
                    return_terms=True,
                    ensemble_disagreement=ensemble_disagr,
                    mu_stack=mu_stack,
                    logvar_stack=logvar_stack,
                )
            else:
                efe = self._expected_free_energy_step(
                    mu_o, logvar_o, logvar_next,
                    ensemble_disagreement=ensemble_disagr,
                    mu_stack=mu_stack,
                    logvar_stack=logvar_stack,
                )

        out = {
            "policy_real_efe": efe.mean().item(),
        }
        if self.log_efe_terms:
            out.update(
                {
                    "policy_real_neg_extrinsic": neg_extrinsic.mean().item(),
                    "policy_real_info_gain": info_gain.mean().item(),
                }
            )
        return out

    def _zero_non_policy_grads(self):
        for module in (self.encoder, self.decoder, self.dynamics):
            if module is None:
                continue
            for p in module.parameters():
                if p.grad is not None:
                    p.grad = None

    # ------------------------------------------------------------------
    #  CEM planning in latent space
    # ------------------------------------------------------------------

    def _infer_latent_state(self, obs_np):
        """
        Given current observation o_t (np array), return a latent sample s_t
        from q(s_t | o_t).
        """
        with torch.no_grad():
            o = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            if self.fully_observable_mdp:
                s = o
            else:
                mu, logvar = self.encoder(o)
                eps = torch.randn_like(mu)
                s = mu + torch.exp(0.5 * logvar) * eps  # [1, latent_dim]
            return s.squeeze(0)  # [latent_dim]

    def _infer_latent_mean(self, obs_np):
        """
        Deterministic latent mean used for uncertainty estimation.
        """
        with torch.no_grad():
            o = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            if self.fully_observable_mdp:
                s = o
            else:
                mu, _ = self.encoder(o)
                s = mu
            return s.squeeze(0)

    def policy_net_mean_action(self, obs_np: np.ndarray) -> torch.Tensor:
        """
        Return the deterministic policy action from the policy_net mean.
        """
        with torch.no_grad():
            o = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            if self.fully_observable_mdp:
                s = o
            else:
                mu, _ = self.encoder(o)
                s = mu
            mean, _ = self.policy_net(s, sample=False)
            action, _, _ = self.policy_net._squash(mean, self.action_low, self.action_high)
            return action.squeeze(0)

    @contextmanager
    def _temporary_planning_override(self, override: Optional[Dict[str, Any]]):
        if not override:
            yield
            return
        old_vals: Dict[str, Any] = {}

        def _set_attr(attr: str, value: Any):
            if attr not in old_vals:
                old_vals[attr] = getattr(self, attr)
            setattr(self, attr, value)

        if "aif_plan_horizon" in override and override["aif_plan_horizon"] is not None:
            horizon = max(1, int(override["aif_plan_horizon"]))
            _set_attr("cem_horizon", horizon)
            _set_attr("policy_plan_horizon", horizon)

        if "aif_plan_candidates" in override and override["aif_plan_candidates"] is not None:
            candidates = max(1, int(override["aif_plan_candidates"]))
            _set_attr("cem_num_samples", candidates)
            _set_attr("policy_plan_samples", candidates)
            _set_attr("cem_num_elites", min(int(self.cem_num_elites), candidates))
            _set_attr("policy_cem_elites", min(int(self.policy_cem_elites), candidates))

        if "aif_mc_models" in override and override["aif_mc_models"] is not None:
            _set_attr("mc_num_models", max(1, int(override["aif_mc_models"])))

        if "aif_mc_trajectories" in override and override["aif_mc_trajectories"] is not None:
            _set_attr("mc_num_trajectories", max(1, int(override["aif_mc_trajectories"])))

        if "aif_policy_plan_action_rollouts" in override and override["aif_policy_plan_action_rollouts"] is not None:
            _set_attr("policy_plan_action_rollouts", max(1, int(override["aif_policy_plan_action_rollouts"])))

        try:
            yield
        finally:
            for attr, value in old_vals.items():
                setattr(self, attr, value)

    def _print_meta_selection(self, info: Optional[Dict[str, Any]]) -> None:
        if not info:
            return
        # Meta planning params are logged by MetaPlanningController.select_mode

    def plan_action(self, obs_np, plan_override: Optional[Dict[str, Any]] = None):
        """
        Plan an action from current observation using CEM in latent space.

        obs_np: numpy array [obs_dim]
        returns: action_np [action_dim]
        """
        s0 = self._infer_latent_state(obs_np)  # [latent_dim]
        s0 = s0.to(self.device)

        mode = getattr(self, "policy_mode", "cem")
        cached_seq = self._cached_policy_action_seq
        max_cached_steps = 0
        if cached_seq is not None and cached_seq.ndim == 2:
            max_cached_steps = min(self.policy_recompute_freq, int(cached_seq.shape[0]))

        meta_override = {}
        meta_info = None
        if self._meta_controller is not None and getattr(self._meta_controller, "enabled", False):
            if self._cached_meta_override is None or self._meta_needs_recompute:
                self._cached_meta_override, self._cached_meta_info = self._meta_controller.select_mode(obs_np)
                self._meta_needs_recompute = False
                self._meta_selection_count += 1
                self._print_meta_selection(self._cached_meta_info)
            meta_override = dict(self._cached_meta_override or {})
            meta_info = dict(self._cached_meta_info or {})
            self._last_meta_info = meta_info
            self._meta_step_counter += 1
            if self._meta_step_counter >= self._meta_recompute_freq:
                self._meta_needs_recompute = True
                self._meta_step_counter = 0
        else:
            self._last_meta_info = None

        if cached_seq is not None and self._policy_replan_counter < max_cached_steps:
            a0 = cached_seq[self._policy_replan_counter]
            self._policy_replan_counter += 1
            self._cached_policy_action = a0
        else:
            self._last_plan_sequence = None
            if plan_override:
                meta_override = dict(meta_override)
                meta_override.update(plan_override)

            deterministic_meta = bool(meta_info.get("meta_deterministic")) if meta_info else False
            if deterministic_meta:
                # Clear stale BC metrics when we skip planning/habituation.
                self._last_policy_bc_info = None
                a0 = self.policy_net_mean_action(obs_np)
                self._cached_policy_score = None
                self._last_action_logprob = None
                self._last_plan_sequence = a0.unsqueeze(0)
            else:
                with self._temporary_planning_override(meta_override):
                    if mode == "policy_net":
                        a0, best_score, logprob0 = self._plan_action_with_policy_net_sampling(s0)
                        self._cached_policy_score = best_score
                        self._last_action_logprob = logprob0
                    elif mode in ("cem", "icem"):
                        with torch.no_grad():
                            a0 = self._plan_action_with_mbrl_optimizer(s0, mode)
                            self._cached_policy_score = self._last_plan_score_best
                            self._last_action_logprob = None
                    else:
                        raise ValueError(
                            f"Unknown policy_mode '{mode}'. Expected 'policy_net', 'cem', or 'icem'."
                        )

                # Train policy to imitate the planned best action (behavioral cloning from planner)
                # Weight by extrinsic value (episode performance); disabled for deterministic mode
                compute_weight = 0.0
                if meta_info and self._meta_controller is not None:
                    if not meta_info.get("meta_deterministic", False):
                        compute_weight = max(0.0, float(meta_info.get("meta_extrinsic_value", 0.0)))
                if compute_weight > 0.0:
                    self._last_policy_bc_info = self.update_policy_from_planned_action(
                        obs_np, a0, compute_weight=compute_weight
                    )
                else:
                    self._last_policy_bc_info = None

            plan_seq = self._last_plan_sequence
            if plan_seq is None:
                plan_seq = a0.unsqueeze(0)
            self._cached_policy_action_seq = plan_seq.detach()
            self._cached_policy_action = a0
            self._policy_replan_counter = 1

        return a0.detach().cpu().numpy()

    def _reshape_action_sequences(
        self, actions: torch.Tensor, horizon: int, action_dim: int
    ) -> torch.Tensor:
        if actions.ndim == 1:
            actions = actions.view(1, -1)
        if actions.ndim == 2:
            if actions.shape[1] == horizon * action_dim:
                return actions.view(actions.shape[0], horizon, action_dim)
            if actions.shape[0] == horizon and actions.shape[1] == action_dim:
                return actions.unsqueeze(0)
        if actions.ndim == 3:
            return actions
        raise ValueError(f"Unexpected action sequence shape: {tuple(actions.shape)}")

    def _dynamics_step_batched(
        self,
        s_flat: torch.Tensor,
        a_flat: torch.Tensor,
        num_sequences: int,
        model_indices: Optional[torch.Tensor],
        deterministic_plan: bool,
        dyn_eps_cache: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.fully_observable_mdp and isinstance(self.dynamics, PETSDynamicsModel):
            if deterministic_plan:
                return self.dynamics(s_flat, a_flat, sample_theta=False, eps_cache=None)
            if model_indices is None:
                return self.dynamics(s_flat, a_flat, sample_theta=True, eps_cache=None)
            mu_stack, logvar_stack = self.dynamics.forward_all_models(s_flat, a_flat)
            idx_flat = model_indices.repeat_interleave(num_sequences)
            batch_idx = torch.arange(idx_flat.shape[0], device=s_flat.device)
            mu_next = mu_stack[idx_flat, batch_idx]
            logvar_next = logvar_stack[idx_flat, batch_idx] if logvar_stack is not None else None
            return mu_next, logvar_next
        return self.dynamics(s_flat, a_flat, sample_theta=not deterministic_plan, eps_cache=dyn_eps_cache)

    def _evaluate_action_sequences(
        self,
        s0: torch.Tensor,
        action_seqs: torch.Tensor,
        model_indices: Optional[torch.Tensor] = None,
        *,
        theta_eps_list: Optional[list] = None,
        theta_eps_cache: Optional[dict] = None,
        noise_cache: Optional[Any] = None,
        model_indices_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if action_seqs.ndim == 2:
            action_seqs = action_seqs.unsqueeze(0)
        if action_seqs.ndim != 3:
            raise ValueError(f"Unexpected action sequence shape: {tuple(action_seqs.shape)}")

        action_seqs = action_seqs.to(self.device)
        s0 = s0.to(self.device)
        num_sequences = int(action_seqs.shape[0])
        horizon = int(action_seqs.shape[1])
        if model_indices_cache is None:
            model_indices_cache = model_indices
        model_indices = model_indices_cache

        theta_list_len = len(theta_eps_list) if theta_eps_list is not None else 0
        num_models = max(1, int(theta_list_len or self.mc_num_models or 1))
        num_traj = max(1, int(self.mc_num_trajectories or 1))
        num_particles = num_models * num_traj
        deterministic_rollout = bool(self.deterministic)
        use_ensemble_info_gain = self._use_ensemble_info_gain()
        use_bayes = isinstance(self.dynamics, BayesianDynamicsModel)
        use_pets = isinstance(self.dynamics, PETSDynamicsModel)
        if deterministic_rollout and not use_ensemble_info_gain:
            num_particles = 1

        s = s0.view(1, 1, -1).expand(num_particles, num_sequences, -1)
        actions = action_seqs.unsqueeze(0).expand(num_particles, -1, -1, -1)

        if model_indices is None and num_particles > 1:
            if use_pets:
                ensemble_size = max(1, int(getattr(self.dynamics, "ensemble_size", 1)))
                if ensemble_size > 1 and num_particles % ensemble_size == 0:
                    per_model = num_particles // ensemble_size
                    model_indices = torch.arange(ensemble_size, device=self.device).repeat_interleave(per_model)
                    model_indices = model_indices[torch.randperm(num_particles, device=self.device)]
                else:
                    model_indices = torch.randint(0, ensemble_size, (num_particles,), device=self.device)
            elif use_bayes and num_models > 1:
                if num_particles % num_models == 0:
                    per_model = num_particles // num_models
                    model_indices = torch.arange(num_models, device=self.device).repeat_interleave(per_model)
                    model_indices = model_indices[torch.randperm(num_particles, device=self.device)]
                else:
                    model_indices = torch.randint(0, num_models, (num_particles,), device=self.device)

        dyn_eps_cache = theta_eps_cache
        theta_eps_list_local = theta_eps_list
        if use_bayes and not deterministic_rollout:
            if num_models > 1:
                if theta_eps_list_local is None:
                    theta_eps_list_local = [self.dynamics.sample_eps() for _ in range(num_models)]
            else:
                if dyn_eps_cache is None and theta_eps_list_local is None:
                    dyn_eps_cache = self.dynamics.sample_eps()

        discounts = torch.pow(
            torch.as_tensor(self.gamma, device=self.device, dtype=torch.float32),
            torch.arange(horizon, device=self.device, dtype=torch.float32),
        )
        total_efe = torch.zeros(num_particles, num_sequences, device=self.device)

        def _get_transition_noise(timestep: int) -> Optional[torch.Tensor]:
            if noise_cache is None:
                return None
            if isinstance(noise_cache, dict):
                cached = noise_cache.get(num_sequences)
                if cached is None or cached.shape[1] != num_particles:
                    cached = torch.randn(
                        horizon,
                        num_particles,
                        num_sequences,
                        self.latent_dim,
                        device=self.device,
                        dtype=s0.dtype,
                    )
                    noise_cache[num_sequences] = cached
                return cached[timestep]
            if noise_cache.shape[0] <= timestep or noise_cache.shape[1] != num_particles:
                return None
            if noise_cache.shape[2] < num_sequences:
                return None
            return noise_cache[timestep, :, :num_sequences, :]

        with torch.no_grad():
            for t in range(horizon):
                a_t = actions[:, :, t, :]
                s_flat = s.reshape(num_particles * num_sequences, -1)
                a_flat = a_t.reshape(num_particles * num_sequences, -1)
                ensemble_disagr = None
                mu_stack = None
                logvar_stack = None
                if use_bayes and num_models > 1:
                    if theta_eps_list_local is not None:
                        mu_stack, logvar_stack, theta_eps_list_local = self._theta_mu_logvar_stack(
                            s_flat, a_flat, num_models=num_models, theta_eps_list=theta_eps_list_local
                        )
                    if mu_stack is not None and use_ensemble_info_gain:
                        ensemble_disagr = self._ensemble_disagreement_from_mu_stack(mu_stack)
                    if mu_stack is not None:
                        if model_indices is None:
                            mu_next = mu_stack.mean(dim=0)
                            logvar_next = logvar_stack.mean(dim=0) if logvar_stack is not None else None
                        else:
                            idx_flat = model_indices.repeat_interleave(num_sequences)
                            batch_idx = torch.arange(idx_flat.shape[0], device=self.device)
                            mu_next = mu_stack[idx_flat, batch_idx]
                            logvar_next = logvar_stack[idx_flat, batch_idx] if logvar_stack is not None else None
                    else:
                        mu_next, logvar_next = self._dynamics_step_batched(
                            s_flat,
                            a_flat,
                            num_sequences=num_sequences,
                            model_indices=model_indices,
                            deterministic_plan=deterministic_rollout,
                            dyn_eps_cache=dyn_eps_cache,
                        )
                elif use_ensemble_info_gain and use_pets:
                    mu_stack, logvar_stack, _ = self._theta_mu_logvar_stack(s_flat, a_flat)
                    if mu_stack is not None:
                        ensemble_disagr = self._ensemble_disagreement_from_mu_stack(mu_stack)
                        if deterministic_rollout:
                            mu_next = mu_stack.mean(dim=0)
                            logvar_next = logvar_stack.mean(dim=0) if logvar_stack is not None else None
                        else:
                            if model_indices is None:
                                mu_next = mu_stack.mean(dim=0)
                                logvar_next = logvar_stack.mean(dim=0) if logvar_stack is not None else None
                            else:
                                idx_flat = model_indices.repeat_interleave(num_sequences)
                                batch_idx = torch.arange(idx_flat.shape[0], device=self.device)
                                mu_next = mu_stack[idx_flat, batch_idx]
                                logvar_next = logvar_stack[idx_flat, batch_idx] if logvar_stack is not None else None
                    else:
                        mu_next, logvar_next = self._dynamics_step_batched(
                            s_flat,
                            a_flat,
                            num_sequences=num_sequences,
                            model_indices=model_indices,
                            deterministic_plan=deterministic_rollout,
                            dyn_eps_cache=None,
                        )
                else:
                    mu_next, logvar_next = self._dynamics_step_batched(
                        s_flat,
                        a_flat,
                        num_sequences=num_sequences,
                        model_indices=model_indices,
                        deterministic_plan=deterministic_rollout,
                        dyn_eps_cache=dyn_eps_cache,
                    )
                if deterministic_rollout or logvar_next is None:
                    s_flat = mu_next
                else:
                    std_next = torch.exp(0.5 * logvar_next)
                    noise_t = _get_transition_noise(t)
                    if noise_t is None:
                        s_flat = mu_next + std_next * torch.randn_like(mu_next)
                    else:
                        noise_flat = noise_t.reshape(num_particles * num_sequences, -1)
                        s_flat = mu_next + std_next * noise_flat

                if self.fully_observable_mdp:
                    mu_o, logvar_o = mu_next, logvar_next
                else:
                    mu_o, logvar_o = self.decoder(s_flat)

                efe_t = self._expected_free_energy_step(
                    mu_o, logvar_o, logvar_next,
                    ensemble_disagreement=ensemble_disagr,
                    mu_stack=mu_stack,
                    logvar_stack=logvar_stack,
                )
                total_efe += discounts[t] * efe_t.view(num_particles, num_sequences)
                s = s_flat.view(num_particles, num_sequences, -1)

        return total_efe.mean(dim=0).view(-1)

    def _get_action_bounds_flat(self, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        low = self.action_low.detach().cpu().numpy()
        high = self.action_high.detach().cpu().numpy()
        low = np.tile(low, (horizon, 1))
        high = np.tile(high, (horizon, 1))
        return low, high

    def _instantiate_mbrl_optimizer(self, optimizer_cls, cfg: Dict[str, Any]):
        sig = inspect.signature(optimizer_cls.__init__)
        params = sig.parameters
        kwargs = {k: v for k, v in cfg.items() if k in params}
        return optimizer_cls(**kwargs)

    def _get_mbrl_optimizer(self, optimizer_name: str, horizon: int):
        if mbrl_planning is None:
            return None
        if (
            self._mbrl_optimizer is not None
            and self._mbrl_optimizer_name == optimizer_name
            and self._mbrl_optimizer_horizon == int(horizon)
        ):
            return self._mbrl_optimizer
        optimizer_cls = getattr(
            mbrl_planning, "CEMOptimizer" if optimizer_name == "cem" else "ICEMOptimizer", None
        )
        if optimizer_cls is None:
            return None
        population_size = max(1, int(self.cem_num_samples))
        elite_ratio = min(1.0, max(1.0 / population_size, self.cem_num_elites / float(population_size)))
        lower_bound, upper_bound = self._get_action_bounds_flat(horizon)
        base_cfg = {
            "num_iterations": int(self.cem_num_iters),
            "population_size": int(population_size),
            "elite_ratio": float(elite_ratio),
            "device": self.device,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }
        if optimizer_name == "cem":
            base_cfg.update(
                {
                    "alpha": 0.1,
                    "return_mean_elites": True,
                    "clipped_normal": False,
                }
            )
        else:
            base_cfg.update(
                {
                    "alpha": 0.1,
                    "return_mean_elites": True,
                    "population_decay_factor": 1.25,
                    "colored_noise_exponent": 2.0,
                    "keep_elite_frac": 0.1,
                }
            )
        overrides = (self._mbrl_optimizer_cfg or {}).get(optimizer_name, {})
        if overrides:
            base_cfg.update({k: v for k, v in overrides.items() if v is not None})
        self._mbrl_optimizer = self._instantiate_mbrl_optimizer(optimizer_cls, base_cfg)
        self._mbrl_optimizer_name = optimizer_name
        self._mbrl_optimizer_horizon = int(horizon)
        return self._mbrl_optimizer

    def _mbrl_optimize(self, optimizer, objective_fn, lower_bound, upper_bound):
        optimize_fn = getattr(optimizer, "optimize", None)
        if optimize_fn is None:
            raise RuntimeError("mbrl optimizer does not expose optimize().")
        sig = inspect.signature(optimize_fn)
        params = sig.parameters
        kwargs = {}
        
        # The optimizer.optimize() method only accepts: obj_fun, x0, callback, **kwargs
        # Bounds are already set in __init__, do NOT pass them to optimize()
        
        prev_solution = getattr(self, "_mbrl_prev_solution", None)
        horizon = int(self.cem_horizon)
        action_dim = int(self.action_dim)
        if prev_solution is not None:
            prev_solution = np.asarray(prev_solution, dtype=np.float32)
            if prev_solution.ndim == 1:
                if prev_solution.size == horizon * action_dim:
                    prev_solution = prev_solution.reshape(horizon, action_dim)
                else:
                    prev_solution = None
            elif prev_solution.ndim == 2:
                if prev_solution.shape != (horizon, action_dim):
                    flat = prev_solution.reshape(-1)
                    if flat.size == horizon * action_dim:
                        prev_solution = flat.reshape(horizon, action_dim)
                    else:
                        prev_solution = None
            else:
                prev_solution = None

        if prev_solution is not None:
            # Convert numpy array to torch.Tensor on the correct device
            x0 = torch.as_tensor(prev_solution, dtype=torch.float32, device=self.device)
            if "x0" in params:
                kwargs["x0"] = x0
            if "init_mean" in params:
                kwargs["init_mean"] = x0
        else:
            # Match mbrl's default: midpoint of action bounds, repeated across horizon.
            mid = (self.action_low + self.action_high) / 2.0
            default_init = mid.unsqueeze(0).repeat(horizon, 1)
            if "x0" in params:
                kwargs["x0"] = default_init
            if "init_mean" in params:
                kwargs["init_mean"] = default_init
        
        return optimize_fn(objective_fn, **kwargs)

    def _plan_action_with_mbrl_optimizer(self, s0: torch.Tensor, optimizer_name: str) -> torch.Tensor:
        if mbrl_planning is None:
            raise RuntimeError(
                "mbrl not installed; set aif_policy_mode='policy_net' or install mbrl to use CEM/ICEM."
            )
        horizon = int(self.cem_horizon)
        optimizer = self._get_mbrl_optimizer(optimizer_name, horizon)
        if optimizer is None:
            raise RuntimeError(f"mbrl optimizer '{optimizer_name}' is unavailable.")
        all_scores_during_optimization = []  # Track scores from all iterations
        call_count = [0]  # Track how many times objective_fn is called
        deterministic_plan = bool(self.deterministic)
        use_ensemble_info_gain = self._use_ensemble_info_gain()
        num_models = max(1, int(self.mc_num_models or 1))
        num_traj = max(1, int(self.mc_num_trajectories or 1))
        num_particles = num_models * num_traj
        if deterministic_plan and not use_ensemble_info_gain:
            num_particles = 1
        theta_eps_list = None
        theta_eps_cache = None
        cached_model_indices = None
        if isinstance(self.dynamics, BayesianDynamicsModel) and not deterministic_plan:
            if num_models > 1:
                theta_eps_list = [self.dynamics.sample_eps() for _ in range(num_models)]
            else:
                theta_eps_cache = self.dynamics.sample_eps()
        if cached_model_indices is None and not (deterministic_plan and not use_ensemble_info_gain):
            if isinstance(self.dynamics, PETSDynamicsModel):
                ensemble_size = max(1, int(getattr(self.dynamics, "ensemble_size", 1)))
                if num_particles > 1:
                    if ensemble_size > 1 and num_particles % ensemble_size == 0:
                        per_model = num_particles // ensemble_size
                        cached_model_indices = torch.arange(ensemble_size, device=self.device).repeat_interleave(per_model)
                        cached_model_indices = cached_model_indices[
                            torch.randperm(num_particles, device=self.device)
                        ]
                    else:
                        cached_model_indices = torch.randint(
                            0, ensemble_size, (num_particles,), device=self.device
                        )
            elif isinstance(self.dynamics, BayesianDynamicsModel) and num_models > 1 and num_particles > 1:
                if num_particles % num_models == 0:
                    per_model = num_particles // num_models
                    cached_model_indices = torch.arange(num_models, device=self.device).repeat_interleave(per_model)
                    cached_model_indices = cached_model_indices[
                        torch.randperm(num_particles, device=self.device)
                    ]
                else:
                    cached_model_indices = torch.randint(0, num_models, (num_particles,), device=self.device)

        noise_cache = None
        if not deterministic_plan:
            population_size = int(getattr(optimizer, "population_size", self.cem_num_samples))
            population_size = max(1, population_size)
            noise_cache = {
                population_size: torch.randn(
                    horizon,
                    num_particles,
                    population_size,
                    self.latent_dim,
                    device=self.device,
                    dtype=s0.dtype,
                )
            }

        def objective_fn(action_sequences):
            nonlocal all_scores_during_optimization
            call_count[0] += 1
            actions = torch.as_tensor(action_sequences, dtype=torch.float32, device=self.device)
            actions = self._reshape_action_sequences(actions, horizon, self.action_dim)
            actions = torch.clamp(actions, self.action_low, self.action_high)
            efe_scores = self._evaluate_action_sequences(
                s0,
                actions,
                model_indices_cache=cached_model_indices,
                theta_eps_list=theta_eps_list,
                theta_eps_cache=theta_eps_cache,
                noise_cache=noise_cache,
            )

            # Track all evaluated EFE scores from this population (lower is better).
            if isinstance(efe_scores, torch.Tensor):
                all_scores_during_optimization.extend(efe_scores.detach().cpu().numpy().tolist())
            else:
                all_scores_during_optimization.extend(
                    efe_scores.tolist() if hasattr(efe_scores, 'tolist') else list(efe_scores)
                )

            # mbrl optimizers maximize; negate EFE to minimize it.
            opt_scores = -efe_scores
            if isinstance(action_sequences, np.ndarray):
                return opt_scores.detach().cpu().numpy()
            return opt_scores

        lower_bound, upper_bound = self._get_action_bounds_flat(horizon)
        try:
            result = self._mbrl_optimize(optimizer, objective_fn, lower_bound, upper_bound)
        except Exception as e:
            print(f"[DEBUG] Exception in _mbrl_optimize: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise

        # result is a torch.Tensor of shape [H, A] or flat [H*A]
        if isinstance(result, torch.Tensor):
            solution = result
        else:
            solution = result[0] if isinstance(result, (tuple, list)) else result
        
        if solution is None:
            raise RuntimeError("mbrl optimizer returned no solution.")
        
        # Ensure solution is a tensor on the correct device
        if not isinstance(solution, torch.Tensor):
            solution = torch.as_tensor(solution, dtype=torch.float32, device=self.device)
        else:
            solution = solution.to(self.device)
        
        sol_seq = self._reshape_action_sequences(solution, horizon, self.action_dim)[0]
        sol_seq = torch.clamp(sol_seq, self.action_low, self.action_high)
        self._last_plan_sequence = sol_seq.detach()

        replan_steps = max(1, int(getattr(self, "policy_recompute_freq", 1)))
        init_action = ((self.action_low + self.action_high) / 2.0).detach().cpu().numpy()
        prev_solution = sol_seq.detach().cpu().numpy()
        if replan_steps >= horizon:
            prev_solution = np.tile(init_action, (horizon, 1))
        else:
            prev_solution = np.roll(prev_solution, -replan_steps, axis=0)
            prev_solution[-replan_steps:] = init_action
        self._mbrl_prev_solution = prev_solution

        # Use all scores from optimization iterations for best/worst
        if all_scores_during_optimization and len(all_scores_during_optimization) > 0:
            scores_array = np.array(all_scores_during_optimization)
            self._last_plan_score_best = float(np.min(scores_array))
            self._last_plan_score_worst = float(np.max(scores_array))
            # print(f"[DEBUG] Total scores collected: {len(all_scores_during_optimization)}, calls to objective_fn: {call_count[0]}")
            # print(f"[DEBUG] Score stats: min={self._last_plan_score_best:.4f}, max={self._last_plan_score_worst:.4f}, mean={np.mean(scores_array):.4f}")
        else:
            # Fallback if no scores were tracked
            score = self._evaluate_action_sequence(s0, sol_seq).item()
            self._last_plan_score_best = float(score)
            self._last_plan_score_worst = float(score)
            print(f"[DEBUG] No scores accumulated during optimization! Using fallback evaluation.")

        a0 = sol_seq[0]
        return a0

    def _plan_action_with_policy_net(self, s0: torch.Tensor) -> torch.Tensor:
        """Sample action from the habitual policy network."""
        with torch.no_grad():
            eps_cache = self.policy_net.sample_eps()
            a, _, _ = self.policy_net.sample_action(
                s0.unsqueeze(0), self.action_low, self.action_high, eps_cache=eps_cache
            )
            a = a.squeeze(0)
            return a

    def _plan_action_with_policy_net_sampling(self, s0: torch.Tensor) -> Tuple[torch.Tensor, float, Optional[torch.Tensor]]:
        """
        MPC-style planning using the policy_net as a persistent proposal.
        Sample trajectories from the current policy distribution, score by EFE,
        update the policy_net to increase likelihood of elite trajectories via
        a weighted negative log-likelihood of the elite first actions at the
        current state (planner-as-teacher), and execute the first action of the
        best (min EFE) sampled trajectory.
        """
        horizon = max(1, int(self.policy_plan_horizon))
        num_samples = max(1, int(self.policy_plan_samples))
        elite_size = max(1, min(self.policy_cem_elites, num_samples))
        temperature = max(0.0, float(self.policy_elite_temperature))

        best_action: Optional[torch.Tensor] = None
        best_sequence: Optional[torch.Tensor] = None
        best_logprob: Optional[torch.Tensor] = None
        best_score: float = float("inf")
        last_scores_t: Optional[torch.Tensor] = None
        action_rollouts = max(1, int(self.policy_plan_action_rollouts))
        deterministic_plan = bool(self.deterministic)
        use_ensemble_info_gain = self._use_ensemble_info_gain()
        use_bayes = isinstance(self.dynamics, BayesianDynamicsModel)
        num_models = max(1, int(self.mc_num_models or 1))

        for _ in range(self.policy_update_steps_per_plan):
            trajectories = []
            scores = []

            for _ in range(num_samples):
                s = s0.unsqueeze(0)  # [1, latent_dim]
                policy_eps = self.policy_net.sample_eps()
                rollout_scores = []
                rep_states = None
                rep_actions = None

                for rollout_idx in range(action_rollouts):
                    s_roll = s0.unsqueeze(0)  # [1, latent_dim]
                    states = []
                    actions = []
                    discount = torch.tensor(1.0, device=self.device)
                    traj_efe = torch.zeros(1, device=self.device)
                    dynamics_eps = None if deterministic_plan else self.dynamics.sample_eps()
                    theta_eps_list = None
                    theta_model_idx = None
                    if use_bayes and use_ensemble_info_gain and num_models > 1:
                        theta_eps_list = [self.dynamics.sample_eps() for _ in range(num_models)]
                        theta_model_idx = random.randrange(num_models)

                    for _ in range(horizon):
                        mean, std = self.policy_net(s_roll, eps_cache=policy_eps)
                        action, _, _ = self.policy_net.sample_action(
                            s_roll, self.action_low, self.action_high, eps_cache=policy_eps
                        )
                        if rollout_idx == 0:
                            states.append(s_roll.squeeze(0).detach())
                            actions.append(action.squeeze(0).detach())

                        with torch.no_grad():
                            ensemble_disagr = None
                            mu_stack = None
                            logvar_stack = None
                            if use_ensemble_info_gain:
                                mu_stack, logvar_stack, theta_eps_list = self._theta_mu_logvar_stack(
                                    s_roll, action, num_models=num_models, theta_eps_list=theta_eps_list
                                )
                                if mu_stack is not None:
                                    ensemble_disagr = self._ensemble_disagreement_from_mu_stack(mu_stack)
                                    if deterministic_plan:
                                        mu_next = mu_stack.mean(dim=0)
                                        logvar_next = logvar_stack.mean(dim=0) if logvar_stack is not None else None
                                    elif use_bayes:
                                        idx = theta_model_idx if theta_model_idx is not None else 0
                                        mu_next = mu_stack[int(idx)]
                                        logvar_next = logvar_stack[int(idx)] if logvar_stack is not None else None
                                    else:
                                        model_idx = self.dynamics._select_model(
                                            eps_cache=dynamics_eps, sample_theta=True
                                        )
                                        if model_idx is None:
                                            mu_next = mu_stack.mean(dim=0)
                                            logvar_next = logvar_stack.mean(dim=0) if logvar_stack is not None else None
                                        else:
                                            mu_next = mu_stack[int(model_idx)]
                                            logvar_next = logvar_stack[int(model_idx)] if logvar_stack is not None else None
                                else:
                                    mu_next, logvar_next = self.dynamics(
                                        s_roll,
                                        action,
                                        sample_theta=not deterministic_plan,
                                        eps_cache=dynamics_eps,
                                    )
                            else:
                                mu_next, logvar_next = self.dynamics(
                                    s_roll,
                                    action,
                                    sample_theta=not deterministic_plan,
                                    eps_cache=dynamics_eps,
                                )
                            if deterministic_plan or logvar_next is None:
                                s_roll = mu_next
                            else:
                                std_next = torch.exp(0.5 * logvar_next)
                                s_roll = mu_next + std_next * torch.randn_like(mu_next)
                            if self.fully_observable_mdp:
                                mu_o, logvar_o = mu_next, logvar_next
                            else:
                                mu_o, logvar_o = self.decoder(s_roll)
                            efe_t = self._expected_free_energy_step(
                                mu_o, logvar_o, logvar_next,
                                ensemble_disagreement=ensemble_disagr,
                                mu_stack=mu_stack,
                                logvar_stack=logvar_stack,
                            )
                            traj_efe = traj_efe + discount * efe_t
                            discount = discount * self.gamma

                    rollout_scores.append(traj_efe.detach().squeeze())
                    if rollout_idx == 0 and states and actions:
                        rep_states = torch.stack(states, dim=0)
                        rep_actions = torch.stack(actions, dim=0)

                trajectories.append(
                    {"states": rep_states if rep_states is not None else torch.zeros(horizon, self.latent_dim, device=self.device),
                     "actions": rep_actions if rep_actions is not None else torch.zeros(horizon, self.action_dim, device=self.device),
                     "eps": policy_eps}
                )
                scores.append(torch.stack(rollout_scores).mean())

            scores_t = torch.stack(scores)  # [N]
            last_scores_t = scores_t
            elite_k = max(1, min(elite_size, scores_t.shape[0]))
            elite_vals, elite_idx = torch.topk(scores_t, k=elite_k, largest=False)

            if temperature > 0.0:
                weights = torch.softmax(-elite_vals / temperature, dim=0)
            else:
                weights = torch.ones_like(elite_vals) / float(elite_k)

            loss_terms = []
            for weight, idx in zip(weights, elite_idx):
                traj = trajectories[idx]
                mc_logprob = 0.0
                # Distill the entire trajectory (optionally discounted).
                discounts = torch.pow(self.gamma, torch.arange(traj["states"].shape[0], device=self.device, dtype=torch.float32))
                for t in range(traj["states"].shape[0]):
                    state_t = traj["states"][t].unsqueeze(0)
                    action_t = traj["actions"][t].unsqueeze(0)
                    step_logprob = 0.0
                    for _ in range(self.policy_posterior_mc_samples):
                        student_eps = self.policy_net.sample_eps()
                        mean, std = self.policy_net(state_t, eps_cache=student_eps)
                        lp = self.policy_net.log_prob_action(
                            action_t, mean, std, self.action_low, self.action_high
                        )
                        step_logprob = step_logprob + lp.squeeze()
                    step_logprob = step_logprob / float(self.policy_posterior_mc_samples)
                    mc_logprob = mc_logprob + discounts[t] * step_logprob
                loss_terms.append(weight * mc_logprob)

            loss = -(torch.stack(loss_terms).sum() / (weights.sum() + 1e-8))
            loss = loss + self.policy_kl_beta * self.policy_net.kl_loss()
            loss = loss + self._policy_std_penalty()
            self.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
            self._zero_non_policy_grads()
            self.policy_optimizer.step()

            best_idx = torch.argmin(scores_t).item()
            candidate_action = trajectories[best_idx]["actions"][0].detach()
            candidate_score = scores_t[best_idx].item()
            if candidate_score < best_score:
                best_score = candidate_score
                best_action = candidate_action
                candidate_seq = trajectories[best_idx]["actions"]
                if candidate_seq is not None:
                    best_sequence = candidate_seq.detach()
                # Monte Carlo posterior predictive log-prob for the executed action.
                with torch.no_grad():
                    logprob_accum = 0.0
                    for _ in range(self.policy_posterior_mc_samples):
                        student_eps = self.policy_net.sample_eps()
                        mean, std = self.policy_net(s0.unsqueeze(0), eps_cache=student_eps)
                        lp = self.policy_net.log_prob_action(
                            candidate_action.unsqueeze(0), mean, std, self.action_low, self.action_high
                        )
                        logprob_accum = logprob_accum + lp.squeeze()
                    best_logprob = logprob_accum / float(self.policy_posterior_mc_samples)

        if best_action is None:
            best_action = torch.zeros(self.action_dim, device=self.device)
        if best_sequence is None:
            best_sequence = best_action.unsqueeze(0).repeat(horizon, 1)

        if last_scores_t is not None and last_scores_t.numel() > 0:
            self._last_plan_score_best = float(last_scores_t.min().item())
            self._last_plan_score_worst = float(last_scores_t.max().item())
        else:
            self._last_plan_score_best = None
            self._last_plan_score_worst = None

        self._last_plan_sequence = torch.clamp(best_sequence, self.action_low, self.action_high).detach()
        best_action = torch.clamp(best_action, self.action_low, self.action_high)
        return best_action, float(best_score), best_logprob

    def _evaluate_action_sequence(self, s0, action_seq):
        """
        Approximate discounted expected free energy under model uncertainty.

        s0: [latent_dim] tensor
        action_seq: [H, action_dim]
        """
        return self._evaluate_action_sequences(s0, action_seq.unsqueeze(0)).squeeze(0)

    def act(self, obs):
        return self.plan_action(obs)

    def reset_planner_state(self) -> None:
        """
        Reset cached planner state between episodes (mirrors mbrl TrajectoryOptimizer reset).
        """
        self._cached_policy_action = None
        self._cached_policy_action_seq = None
        self._policy_replan_counter = 0
        self._cached_policy_score = None
        self._last_action_logprob = None
        self._last_plan_sequence = None
        self._mbrl_prev_solution = None
        self._last_meta_info = None
        self._cached_meta_override = None
        self._cached_meta_info = None
        self._meta_step_counter = 0
        self._meta_needs_recompute = True

        # Reset meta controller's episode-specific tracking to prevent cross-episode contamination
        if self._meta_controller is not None:
            self._meta_controller.on_episode_reset()

    def update_meta_stats(self, obs_np: np.ndarray, action_np: np.ndarray, next_obs_np: np.ndarray) -> None:
        if self._meta_controller is None or not getattr(self._meta_controller, "enabled", False):
            return
        self._meta_controller.update_from_transition(obs_np, action_np, next_obs_np)


# ======================================================================
#  SB3-style wrappers for myGym integration
# ======================================================================


class ActiveInferenceSB3:
    """
    Lightweight SB3-compatible wrapper around AIFAgent.
    Provides learn/predict/save/load so it plugs into train.py/test.py.
    """

    def __init__(self, env: gym.Env, arg_dict: Optional[Dict[str, Any]] = None, device: Optional[str] = None, **_unused_kwargs):
        if env is None:
            raise ValueError("Environment must be provided to ActiveInferenceSB3.")
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.arg_dict: Dict[str, Any] = dict(arg_dict) if arg_dict is not None else {}
        self.device = torch.device(device or self.arg_dict.get("aif_device", "cpu"))
        self.logger = None  # SB3 callbacks expect this attribute
        self._pref_debug_printed = False
        self._last_onpolicy_obs = None
        self._last_onpolicy_action = None

        # Defaults for training loop
        self.arg_dict.setdefault("aif_initial_random_steps", 1000)
        self.arg_dict.setdefault("aif_model_update_freq", 50)
        self.arg_dict.setdefault("aif_model_train_epochs", 1)
        self.arg_dict.setdefault("aif_model_updates_per_step", 1)
        self.arg_dict.setdefault("aif_policy_updates_per_step", 1)
        self.arg_dict.setdefault("aif_replay_size", 100000)
        self.arg_dict.setdefault("aif_gamma", 0.99)
        self.arg_dict.setdefault("aif_info_gain_weight", 0.0)
        self.arg_dict.setdefault("aif_info_gain_mode", "ensemble_var")
        self.arg_dict.setdefault("aif_pref_mean", None)
        self.arg_dict.setdefault("aif_pref_std", None)
        self.arg_dict.setdefault("aif_pref_target_weight", 0.0)
        self.arg_dict.setdefault("aif_pref_target_scale", 0.1)
        self.arg_dict.setdefault("aif_log_freq", 0)
        self.arg_dict.setdefault("aif_plan_horizon", 20)
        self.arg_dict.setdefault("aif_optimisation_iters", 5)
        self.arg_dict.setdefault("aif_plan_candidates", 64)
        self.arg_dict.setdefault("aif_plan_elites", 6)
        self.arg_dict.setdefault("aif_mc_models", None)
        self.arg_dict.setdefault("aif_kl_theta_beta", 1.0)
        self.arg_dict.setdefault("aif_policy_mode", "cem")  # options: "cem", "icem", "policy_net"
        self.arg_dict.setdefault("aif_policy_eval_mode", "plan")  # options: "plan", "mean" (policy_net only)
        self.arg_dict.setdefault("aif_policy_imagination_freq", 0)  # 0 disables
        self.arg_dict.setdefault("aif_policy_imagination_horizon", None)
        self.arg_dict.setdefault("aif_policy_imagination_updates", 1)
        self.arg_dict.setdefault("aif_policy_real_efe_freq", 0)  # 0 disables
        self.arg_dict.setdefault("aif_policy_real_efe_updates", 1)
        self.arg_dict.setdefault("aif_batch_size", 64)
        self.arg_dict.setdefault("aif_log_vfe_terms", False)
        self.arg_dict.setdefault("aif_log_efe_terms", False)
        self.arg_dict.setdefault("aif_model_lr", 3e-4)
        self.arg_dict.setdefault("aif_model_wd", 0.0)
        self.arg_dict.setdefault("aif_logvar_reg_weight", 0.01)
        self.arg_dict.setdefault("aif_model_normalize", False)
        self.arg_dict.setdefault("aif_model_val_ratio", 0.1)
        self.arg_dict.setdefault("aif_model_bootstrap", True)
        self.arg_dict.setdefault("aif_model_bootstrap_permutes", True)
        self.arg_dict.setdefault("aif_model_shuffle_each_epoch", True)
        # Unified planning params used by both CEM and policy_net planners
        # (horizon, candidates, elites) with optional mode-specific recompute freq.
        self.arg_dict.setdefault("aif_plan_recompute_freq", 1)
        self.arg_dict.setdefault("aif_policy_elite_temperature", 0.0)
        self.arg_dict.setdefault("aif_policy_net_update_steps", 1)
        self.arg_dict.setdefault("aif_policy_kl_beta", 1e-4)
        self.arg_dict.setdefault("aif_policy_posterior_mc_samples", 4)
        self.arg_dict.setdefault("aif_policy_efe_baseline_momentum", 0.9)
        self.arg_dict.setdefault("aif_policy_plan_action_rollouts", 1)
        self.arg_dict.setdefault("aif_meta_enabled", False)
        self.arg_dict.setdefault("aif_meta_selection", "greedy")
        self.arg_dict.setdefault("aif_meta_softmax_temp", 1.0)
        self.arg_dict.setdefault("aif_meta_allow_deterministic", False)
        self.arg_dict.setdefault("aif_meta_mode_count", 3)
        self.arg_dict.setdefault("aif_meta_modes", None)
        self.arg_dict.setdefault("aif_meta_max_modes", None)
        self.arg_dict.setdefault("aif_meta_recompute_freq", 1)
        self.arg_dict.setdefault("aif_meta_cost_log", True)
        self.arg_dict.setdefault("aif_meta_policy_ambiguity_weight", 1.0)
        self.arg_dict.setdefault("aif_meta_model_ambiguity_weight", 1.0)
        # Observation preference precisions
        self.arg_dict.setdefault("aif_meta_pref_extrinsic_precision", 1.0)
        self.arg_dict.setdefault("aif_meta_pref_policy_uncert_precision", 1.0)
        self.arg_dict.setdefault("aif_meta_pref_model_uncert_precision", 1.0)
        self.arg_dict.setdefault("aif_meta_pref_effort_precision", 1.0)
        self.arg_dict.setdefault("aif_meta_error_ema_beta", 0.9)
        self.arg_dict.setdefault("aif_meta_uncert_ema_beta", 0.9)
        self.arg_dict.setdefault("aif_meta_policy_uncert_ema_beta", 0.9)
        self.arg_dict.setdefault("aif_meta_extrinsic_value_weight", 0.3)
        self.arg_dict.setdefault("aif_meta_risk_include_variance", False)
        self.arg_dict.setdefault("aif_meta_uncertainty_probe_actions", 3)
        self.arg_dict.setdefault("aif_meta_uncertainty_action_noise", 0.1)
        self.arg_dict.setdefault("aif_meta_policy_uncertainty_samples", None)
        self.arg_dict.setdefault("aif_meta_horizon_step", 1)
        self.arg_dict.setdefault("aif_meta_candidates_step", 1)
        self.arg_dict.setdefault("aif_meta_mc_models_step", 1)
        self.arg_dict.setdefault("aif_meta_mc_trajectories_step", 1)
        self.arg_dict.setdefault("aif_meta_action_rollouts_step", 1)
        self.arg_dict.setdefault("aif_tensorboard_log", True)
        self.arg_dict.setdefault("aif_loss_weight_kl_state", 1.0)
        self.arg_dict.setdefault("aif_loss_weight_nll_obs", 1.0)
        self.arg_dict.setdefault("aif_use_mse_loss", False)
        self.arg_dict.setdefault("aif_drop_goal_from_state", False)
        if "fully_observable_mdp" not in self.arg_dict:
            if "fully_observable mdp" in self.arg_dict:
                self.arg_dict["fully_observable_mdp"] = self.arg_dict.get("fully_observable mdp")
            elif "aif_fully_observable_mdp" in self.arg_dict:
                self.arg_dict["fully_observable_mdp"] = self.arg_dict.get("aif_fully_observable_mdp")
        self.arg_dict.setdefault("fully_observable_mdp", False)

        def _pick_cfg(*keys, default=None):
            for key in keys:
                if key in self.arg_dict:
                    return self.arg_dict.get(key)
            return default

        def _prune_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
            return {k: v for k, v in cfg.items() if v is not None}

        cem_cfg = {
            "alpha": _pick_cfg("aif_cem_alpha", "cem_alpha", default=0.1),
            "clipped_normal": _pick_cfg("aif_cem_clipped_normal", "cem_clipped_normal", default=False),
            "return_mean_elites": _pick_cfg("aif_cem_return_mean_elites", "cem_return_mean_elites", default=True),
        }
        icem_cfg = {
            "num_iterations": _pick_cfg("aif_icem_num_iters", "icem_num_iters", default=None),
            "population_size": _pick_cfg("aif_icem_population_size", "icem_population_size", default=None),
            "elite_ratio": _pick_cfg("aif_icem_elite_ratio", "icem_elite_ratio", default=None),
            "population_decay_factor": _pick_cfg(
                "aif_icem_population_decay_factor", "icem_population_decay_factor", default=1.25
            ),
            "colored_noise_exponent": _pick_cfg(
                "aif_icem_colored_noise_exponent", "icem_colored_noise_exponent", default=2.0
            ),
            "keep_elite_frac": _pick_cfg("aif_icem_keep_elite_frac", "icem_keep_elite_frac", default=0.1),
            "alpha": _pick_cfg("aif_icem_alpha", "icem_alpha", default=cem_cfg["alpha"]),
            "return_mean_elites": _pick_cfg(
                "aif_icem_return_mean_elites", "icem_return_mean_elites", default=cem_cfg["return_mean_elites"]
            ),
        }
        mbrl_optimizer_cfg = {
            "cem": _prune_cfg(cem_cfg),
            "icem": _prune_cfg(icem_cfg),
        }

        self._obs_layout: Optional[Dict[str, Any]] = self._compute_obs_layout()
        self._drop_goal_from_state = bool(self.arg_dict.get("aif_drop_goal_from_state", False))
        if self._drop_goal_from_state and (self._obs_layout is None or "goal" not in self._obs_layout):
            if not getattr(ActiveInferenceSB3, "_warned_drop_goal", False):
                ActiveInferenceSB3._warned_drop_goal = True
                print("[AIF][warn] aif_drop_goal_from_state=True but goal slice not found; keeping goal in model state.")
            self._drop_goal_from_state = False
        self._state_layout: Optional[Dict[str, Any]] = self._compute_state_layout()

        raw_world_model = self.arg_dict.get("aif_world_model", {}) or {}
        world_model_cfg = raw_world_model if isinstance(raw_world_model, dict) else {}
        world_model_type = None
        if isinstance(raw_world_model, str):
            world_model_type = raw_world_model
        elif isinstance(raw_world_model, dict):
            world_model_type = raw_world_model.get("type") or raw_world_model.get("model") or raw_world_model.get("name")
        if "aif_world_model_type" in self.arg_dict:
            world_model_type = self.arg_dict.get("aif_world_model_type")
        world_model_type = AIFAgent._normalize_world_model_type(world_model_type)
        hidden_dim = world_model_cfg.get("hid_size", 128)
        ensemble_size = world_model_cfg.get("ensemble_size", 5)
        dynamics_num_layers = int(world_model_cfg.get("num_layers", 3))
        dynamics_target_is_delta = bool(world_model_cfg.get("target_is_delta", True))
        dynamics_logvar_min = float(world_model_cfg.get("logvar_min", -10.0))
        dynamics_logvar_max = float(world_model_cfg.get("logvar_max", 2.0))
        mc_traj = self.arg_dict.get("aif_mc_trajectories", 3)
        mc_models = self.arg_dict.get("aif_mc_models", None)
        if mc_models is None:
            mc_models = ensemble_size
        self.arg_dict["aif_mc_models"] = mc_models
        meta_policy_samples = self.arg_dict.get("aif_meta_policy_uncertainty_samples")
        if meta_policy_samples is None:
            meta_policy_samples = self.arg_dict.get("aif_policy_posterior_mc_samples", 4)
        meta_cfg = {
            "enabled": bool(self.arg_dict.get("aif_meta_enabled", False)),
            "selection": self.arg_dict.get("aif_meta_selection", "greedy"),
            "softmax_temp": self.arg_dict.get("aif_meta_softmax_temp", 1.0),
            "allow_deterministic": bool(self.arg_dict.get("aif_meta_allow_deterministic", False)),
            "mode_count": self.arg_dict.get("aif_meta_mode_count", 3),
            "modes": self.arg_dict.get("aif_meta_modes"),
            "max_modes": self.arg_dict.get("aif_meta_max_modes"),
            "recompute_freq": self.arg_dict.get("aif_meta_recompute_freq", 1),
            "horizon_min": self.arg_dict.get("aif_meta_horizon_min", self.arg_dict.get("aif_plan_horizon", 20)),
            "horizon_max": self.arg_dict.get("aif_meta_horizon_max", self.arg_dict.get("aif_plan_horizon", 20)),
            "candidates_min": self.arg_dict.get("aif_meta_candidates_min", self.arg_dict.get("aif_plan_candidates", 64)),
            "candidates_max": self.arg_dict.get("aif_meta_candidates_max", self.arg_dict.get("aif_plan_candidates", 64)),
            "mc_models_min": self.arg_dict.get("aif_meta_mc_models_min", mc_models),
            "mc_models_max": self.arg_dict.get("aif_meta_mc_models_max", mc_models),
            "mc_trajectories_min": self.arg_dict.get("aif_meta_mc_trajectories_min", mc_traj),
            "mc_trajectories_max": self.arg_dict.get("aif_meta_mc_trajectories_max", mc_traj),
            "action_rollouts_min": self.arg_dict.get(
                "aif_meta_action_rollouts_min", self.arg_dict.get("aif_policy_plan_action_rollouts", 1)
            ),
            "action_rollouts_max": self.arg_dict.get(
                "aif_meta_action_rollouts_max", self.arg_dict.get("aif_policy_plan_action_rollouts", 1)
            ),
            "horizon_step": self.arg_dict.get("aif_meta_horizon_step", 1),
            "candidates_step": self.arg_dict.get("aif_meta_candidates_step", 1),
            "mc_models_step": self.arg_dict.get("aif_meta_mc_models_step", 1),
            "mc_trajectories_step": self.arg_dict.get("aif_meta_mc_trajectories_step", 1),
            "action_rollouts_step": self.arg_dict.get("aif_meta_action_rollouts_step", 1),
            "cost_log": bool(self.arg_dict.get("aif_meta_cost_log", True)),
            "policy_ambiguity_weight": self.arg_dict.get("aif_meta_policy_ambiguity_weight", 1.0),
            "model_ambiguity_weight": self.arg_dict.get("aif_meta_model_ambiguity_weight", 1.0),
            # Observation preference precisions
            "pref_extrinsic_precision": self.arg_dict.get("aif_meta_pref_extrinsic_precision", 1.0),
            "pref_policy_uncert_precision": self.arg_dict.get("aif_meta_pref_policy_uncert_precision", 1.0),
            "pref_model_uncert_precision": self.arg_dict.get("aif_meta_pref_model_uncert_precision", 1.0),
            "pref_effort_precision": self.arg_dict.get("aif_meta_pref_effort_precision", 1.0),
            # Observation preference means
            "pref_extrinsic_mean": self.arg_dict.get("aif_meta_pref_extrinsic_mean", 1.0),
            "pref_policy_uncert_mean": self.arg_dict.get("aif_meta_pref_policy_uncert_mean", 0.0),
            "pref_model_uncert_mean": self.arg_dict.get("aif_meta_pref_model_uncert_mean", 0.0),
            "pref_effort_mean": self.arg_dict.get("aif_meta_pref_effort_mean", 0.0),
            "capacity_exponent": self.arg_dict.get("aif_meta_capacity_exponent", 1.0),
            "variance_exponent": self.arg_dict.get("aif_meta_variance_exponent", 2.65),
            "extrinsic_value_weight": self.arg_dict.get("aif_meta_extrinsic_value_weight", 0.3),
            "risk_include_variance": bool(self.arg_dict.get("aif_meta_risk_include_variance", False)),
            "error_ema_beta": self.arg_dict.get("aif_meta_error_ema_beta", 0.9),
            "uncert_ema_beta": self.arg_dict.get("aif_meta_uncert_ema_beta", 0.9),
            "policy_uncert_ema_beta": self.arg_dict.get("aif_meta_policy_uncert_ema_beta", 0.9),
            "uncertainty_probe_actions": self.arg_dict.get("aif_meta_uncertainty_probe_actions", 3),
            "uncertainty_action_noise": self.arg_dict.get("aif_meta_uncertainty_action_noise", 0.1),
            "policy_uncertainty_samples": meta_policy_samples,
            "warmup_transitions": self.arg_dict.get("aif_meta_warmup_transitions", 100),
            "update_model_before_meta": (
                str(self.arg_dict.get("aif_model_update_freq", "")).lower() == "meta_update"
            ),
            "debug": bool(self.arg_dict.get("aif_meta_debug", False)),
            # Initial EMA values (pessimistic to favor reactive modes at start)
            "initial_model_error": self.arg_dict.get("aif_meta_initial_model_error", 2.0),
            "initial_model_uncert": self.arg_dict.get("aif_meta_initial_model_uncert", 1.0),
            "initial_policy_uncert": self.arg_dict.get("aif_meta_initial_policy_uncert", 1.0),
            "initial_extrinsic_value": self.arg_dict.get("aif_meta_initial_extrinsic_value", 0.0),
            "dimension_velocity": float(self.arg_dict.get("dimension_velocity", 0.05)),
        }

        obs_dim = self._infer_obs_dim(self.observation_space)
        if self._drop_goal_from_state and self._obs_layout and "goal" in self._obs_layout:
            g_start, g_end = self._obs_layout["goal"]
            goal_len = max(0, int(g_end - g_start))
            if goal_len > 0 and obs_dim >= goal_len:
                obs_dim -= goal_len
        action_dim = int(np.prod(self.action_space.shape))
        if world_model_type is None:
            fully_observable_mdp = bool(self.arg_dict.get("fully_observable_mdp", False))
            world_model_type = "bayesian" if fully_observable_mdp else "vae"
        else:
            fully_observable_mdp = world_model_type != "vae"
        self.arg_dict["fully_observable_mdp"] = fully_observable_mdp
        self.arg_dict["aif_world_model_type"] = world_model_type
        latent_dim = obs_dim if fully_observable_mdp else self.arg_dict.get("aif_latent_dim")

        self.agent = AIFAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_low=self.action_space.low,
            action_high=self.action_space.high,
            latent_dim=latent_dim,
            device=self.device,
            hidden_dim=hidden_dim,
            gamma=self.arg_dict.get("aif_gamma", 0.99),
            cem_num_samples=self.arg_dict.get("aif_plan_candidates", 64),
            cem_num_elites=self.arg_dict.get("aif_plan_elites", 6),
            cem_num_iters=self.arg_dict.get("aif_optimisation_iters", 5),
            cem_horizon=self.arg_dict.get("aif_plan_horizon", 20),
            mc_num_models=mc_models,
            mc_num_trajectories=mc_traj,
            kl_theta_beta=self.arg_dict.get("aif_kl_theta_beta", 1.0),
            policy_mode=str(self.arg_dict.get("aif_policy_mode", "cem")).lower(),
            info_gain_weight=self.arg_dict.get("aif_info_gain_weight", 0.0),
            info_gain_mode=self.arg_dict.get("aif_info_gain_mode", "ensemble_var"),
            preference_mean=self.arg_dict.get("aif_pref_mean"),
            preference_std=self.arg_dict.get("aif_pref_std"),
            preference_mode=self.arg_dict.get("aif_preference_mode", "gaussian"),
            pref_target_weight=self.arg_dict.get("aif_pref_target_weight", 0.0),
            pref_target_scale=self.arg_dict.get("aif_pref_target_scale", 0.1),
            preference_linear_scale=self.arg_dict.get("aif_preference_linear_scale", 1.0),
            log_efe_terms=self.arg_dict.get("aif_log_efe_terms", False),
            deterministic=self.arg_dict.get("aif_deterministic", False),
            policy_plan_horizon=self.arg_dict.get("aif_plan_horizon", 20),
            policy_plan_samples=self.arg_dict.get("aif_plan_candidates", 64),
            policy_recompute_freq=self.arg_dict.get("aif_plan_recompute_freq", 1),
            policy_cem_elites=self.arg_dict.get("aif_plan_elites", 6),
            policy_elite_temperature=self.arg_dict.get("aif_policy_elite_temperature", 0.0),
            policy_update_steps_per_plan=self.arg_dict.get("aif_policy_net_update_steps", 0),
            policy_kl_beta=self.arg_dict.get("aif_policy_kl_beta", 1e-4),
            policy_std_penalty_weight=self.arg_dict.get("aif_policy_std_penalty_weight", 0.0),
            policy_posterior_mc_samples=self.arg_dict.get("aif_policy_posterior_mc_samples", 4),
            policy_efe_baseline_momentum=self.arg_dict.get("aif_policy_efe_baseline_momentum", 0.9),
            policy_plan_action_rollouts=self.arg_dict.get("aif_policy_plan_action_rollouts", 1),
            loss_weight_kl_state=self.arg_dict.get("aif_loss_weight_kl_state", 1.0),
            loss_weight_nll_obs=self.arg_dict.get("aif_loss_weight_nll_obs", 1.0),
            use_mse_loss=self.arg_dict.get("aif_use_mse_loss", False),
            model_train_epochs=self.arg_dict.get("aif_model_train_epochs", 1),
            model_val_ratio=self.arg_dict.get("aif_model_val_ratio", 0.1),
            model_bootstrap=self.arg_dict.get("aif_model_bootstrap", True),
            model_bootstrap_permutes=self.arg_dict.get("aif_model_bootstrap_permutes", True),
            model_shuffle_each_epoch=self.arg_dict.get("aif_model_shuffle_each_epoch", True),
            model_lr=self.arg_dict.get("aif_model_lr", 3e-4),
            model_wd=self.arg_dict.get("aif_model_wd", 0.0),
            logvar_reg_weight=self.arg_dict.get("aif_logvar_reg_weight", 0.01),
            fully_observable_mdp=fully_observable_mdp,
            world_model_type=world_model_type,
            dynamics_ensemble_size=ensemble_size,
            dynamics_num_layers=dynamics_num_layers,
            dynamics_target_is_delta=dynamics_target_is_delta,
            dynamics_logvar_min=dynamics_logvar_min,
            dynamics_logvar_max=dynamics_logvar_max,
            mbrl_optimizer_cfg=mbrl_optimizer_cfg,
            normalize_inputs=bool(self.arg_dict.get("aif_model_normalize", False)),
            meta_cfg=meta_cfg,
        )

        self.replay_buffer = ReplayBuffer(self.arg_dict["aif_replay_size"])
        self.initial_random_steps = int(self.arg_dict["aif_initial_random_steps"])
        # Handle "meta_update" mode: model updates triggered by meta controller, not by step count
        _model_update_freq = self.arg_dict.get("aif_model_update_freq", 250)
        if str(_model_update_freq).lower() == "meta_update":
            self.model_update_freq = 0  # Disable step-based updates (meta controller handles it)
        else:
            self.model_update_freq = int(_model_update_freq)
        self.model_updates_per_step = int(self.arg_dict.get("aif_model_updates_per_step", 1))
        self.policy_updates_per_step = int(self.arg_dict["aif_policy_updates_per_step"])
        self.log_freq = int(self.arg_dict.get("aif_log_freq", 0) or 0)
        self.batch_size = int(self.arg_dict.get("aif_batch_size", 64))
        # Store references on agent for meta controller's model updates
        self.agent._meta_replay_buffer = self.replay_buffer
        self.agent._meta_batch_size = self.batch_size
        self.num_timesteps = 0
        self._latest_update_info: Optional[Dict[str, float]] = None
        self._progress_bar_enabled = bool(self.arg_dict.get("aif_progress_bar", False))
        self._episode_pbar = None
        self._warned_no_tqdm = False
        self._model_update_debug_printed = False
        self._last_meta_selection_count = 0
        self._max_episode_steps = int(
            self.arg_dict.get("max_episode_steps")
            or getattr(getattr(self.env, "env", None), "_max_episode_steps", 0)
            or getattr(self.env, "_max_episode_steps", 0)
            or 0
        )
        self._policy_eval_mode = self._normalize_policy_eval_mode(
            self.arg_dict.get("aif_policy_eval_mode", "plan")
        )
        self.arg_dict["aif_policy_eval_mode"] = self._policy_eval_mode
        self._tb_writer = None
        self._tb_log_dir = None
        if bool(self.arg_dict.get("aif_tensorboard_log", False)) and SummaryWriter is not None:
            base_logdir = self.arg_dict.get("aif_tensorboard_dir") or self.arg_dict.get("logdir")
            if base_logdir:
                self._tb_log_dir = os.path.join(base_logdir, "aif_tb")
                os.makedirs(self._tb_log_dir, exist_ok=True)
                self._tb_writer = SummaryWriter(log_dir=self._tb_log_dir)

    @staticmethod
    def _infer_obs_dim(space: gym.Space) -> int:
        if isinstance(space, gym.spaces.Dict):
            if "observation" in space.spaces:
                return int(np.prod(space.spaces["observation"].shape))
            return int(sum(np.prod(s.shape) for s in space.spaces.values()))
        return int(np.prod(space.shape))

    @staticmethod
    def _normalize_policy_eval_mode(mode: Optional[str]) -> str:
        mode = str(mode or "plan").lower()
        if mode not in ("plan", "mean"):
            raise ValueError(f"Unknown aif_policy_eval_mode '{mode}'. Expected 'plan' or 'mean'.")
        return mode

    def _compute_obs_layout(self) -> Optional[Dict[str, Any]]:
        """
        Infer index ranges for actual, goal and end-effector blocks in the flattened observation.
        Returns None if layout cannot be determined reliably.
        """
        obs_cfg = self.arg_dict.get("observation", {}) or {}
        if not isinstance(obs_cfg, dict):
            return None

        length_map = {
            "obj_xyz": 3,
            "obj_6D": 7,
            "endeff_xyz": 3,
            "endeff_6D": 7,
            "touch": 1,
        }

        actual_key = obs_cfg.get("actual_state")
        goal_key = obs_cfg.get("goal_state")
        if actual_key not in length_map or goal_key not in length_map:
            return None

        layout: Dict[str, Any] = {}
        idx = 0
        layout["actual"] = (idx, idx + length_map[actual_key])
        idx += length_map[actual_key]
        layout["goal"] = (idx, idx + length_map[goal_key])
        idx += length_map[goal_key]

        additional_slices = {}
        for key in obs_cfg.get("additional_obs") or []:
            length = length_map.get(key)
            if length is None:
                # Unknown-sized component; keep known slices but stop inferring further offsets.
                break
            start, end = idx, idx + length
            if key.startswith("endeff"):
                additional_slices[key] = (start, end)
            idx = end

        layout["additional"] = additional_slices
        return layout

    def _compute_state_layout(self) -> Optional[Dict[str, Any]]:
        """
        Return layout for the model state (optionally dropping the goal slice).
        """
        layout = getattr(self, "_obs_layout", None)
        if layout is None:
            return None
        if not self._drop_goal_from_state or "goal" not in layout:
            return layout

        g_start, g_end = layout["goal"]
        goal_len = max(0, int(g_end - g_start))
        state_layout: Dict[str, Any] = {}
        if "actual" in layout:
            state_layout["actual"] = layout["actual"]
        additional = {}
        for key, (start, end) in (layout.get("additional") or {}).items():
            if start >= g_end:
                additional[key] = (start - goal_len, end - goal_len)
            else:
                additional[key] = (start, end)
        state_layout["additional"] = additional
        return state_layout

    def _flatten_obs_with_pref(self, obs: Any) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Flatten observation; if goal_state is present, also build a preference
        vector that targets the goal for actual/goal/end-effector components. Returns an
        optional mask that marks which entries should be used for distance-based
        preferences (e.g., the actual_state / end-effector slice).
        """
        def _normalize_xyz_blocks(vec: np.ndarray, block_indices) -> np.ndarray:
            normed = np.asarray(vec, dtype=np.float32).copy()
            for start, end in block_indices:
                if start >= normed.size:
                    continue
                clamped_end = min(end, normed.size)
                if clamped_end > start:
                    normed[start:clamped_end] = (normed[start:clamped_end] - 0.5) / 0.5
            return normed

        def _default_norm_blocks(size: int):
            blocks = []
            if size >= 3:
                blocks.append((0, 3))
            if size >= 6:
                blocks.append((3, 6))
            if size >= 9:
                blocks.append((6, 9))
            return blocks

        def _layout_norm_blocks(layout, size: int):
            if layout is None:
                return []
            blocks = []
            for key in ("actual", "goal"):
                if key in layout:
                    start, end = layout[key]
                    pos_end = min(start + 3, end, size)
                    if pos_end > start:
                        blocks.append((start, pos_end))
            additional = layout.get("additional") or {}
            for key in ("endeff_xyz", "endeff_6D"):
                if key in additional:
                    start, end = additional[key]
                    pos_end = min(start + 3, end, size)
                    if pos_end > start:
                        blocks.append((start, pos_end))
            return blocks

        def _extract_goal_block(full_arr: np.ndarray):
            layout = getattr(self, "_obs_layout", None)
            if layout and "goal" in layout:
                g_start, g_end = layout["goal"]
                if g_start < full_arr.size:
                    goal_block = full_arr[g_start : min(g_end, full_arr.size)]
                    if goal_block.size > 0:
                        return goal_block
            if full_arr.size >= 6:
                return full_arr[3:6]
            return None

        def _drop_goal_slice(full_arr: np.ndarray):
            layout = getattr(self, "_obs_layout", None)
            if layout and "goal" in layout:
                g_start, g_end = layout["goal"]
                if g_start < full_arr.size:
                    return np.concatenate([full_arr[:g_start], full_arr[min(g_end, full_arr.size):]])
            if full_arr.size >= 6:
                return np.concatenate([full_arr[:3], full_arr[6:]])
            return full_arr

        pref_vec = None
        pref_mask = None
        obs_arr_full = None
        obs_arr_state = None
        goal_block = None

        if isinstance(obs, dict):
            if "observation" in obs:
                obs_arr_full = np.asarray(obs["observation"], dtype=np.float32).ravel()
            else:
                arrays_full = []
                arrays_state = []
                if "actual_state" in obs:
                    actual = np.asarray(obs["actual_state"], dtype=np.float32).ravel()
                    arrays_full.append(actual)
                    arrays_state.append(actual)
                if "goal_state" in obs:
                    goal = np.asarray(obs["goal_state"], dtype=np.float32).ravel()
                    arrays_full.append(goal)
                    goal_block = goal
                    if not self._drop_goal_from_state:
                        arrays_state.append(goal)
                additional = obs.get("additional_obs")
                if isinstance(additional, dict):
                    for key in additional:
                        arr = np.asarray(additional[key], dtype=np.float32).ravel()
                        arrays_full.append(arr)
                        arrays_state.append(arr)
                else:
                    for key in sorted(k for k in obs.keys() if k not in ("actual_state", "goal_state", "additional_obs")):
                        arr = np.asarray(obs[key], dtype=np.float32).ravel()
                        arrays_full.append(arr)
                        arrays_state.append(arr)
                obs_arr_full = np.concatenate(arrays_full).astype(np.float32) if arrays_full else np.array([], dtype=np.float32)
                obs_arr_state = np.concatenate(arrays_state).astype(np.float32) if arrays_state else np.array([], dtype=np.float32)
        else:
            obs_arr_full = np.asarray(obs, dtype=np.float32).ravel()

        if obs_arr_full is None:
            obs_arr_full = np.array([], dtype=np.float32)
        if obs_arr_state is None:
            obs_arr_state = obs_arr_full.copy()
            if self._drop_goal_from_state:
                obs_arr_state = _drop_goal_slice(obs_arr_full)

        if goal_block is None:
            goal_block = _extract_goal_block(obs_arr_full)

        if self._drop_goal_from_state:
            self._last_full_obs = obs_arr_full.copy()

        state_layout = getattr(self, "_state_layout", None)
        if goal_block is not None and state_layout is not None:
            pref_vec = obs_arr_state.copy()
            goal_len = min(3, goal_block.size)

            def _fill_slice(slice_def):
                if not slice_def:
                    return
                s, e = slice_def
                length = min(goal_len, e - s, pref_vec.size - s)
                if length > 0:
                    pref_vec[s : s + length] = goal_block[:length]

            _fill_slice(state_layout.get("actual"))
            if not self._drop_goal_from_state:
                _fill_slice(state_layout.get("goal"))
            for slice_def in (state_layout.get("additional") or {}).values():
                _fill_slice(slice_def)

            if self._drop_goal_from_state:
                pref_mask = np.zeros_like(pref_vec, dtype=np.float32)

                def _mark_slice(slice_def):
                    if not slice_def:
                        return
                    s, e = slice_def
                    length = min(goal_len, e - s, pref_vec.size - s)
                    if length > 0:
                        pref_mask[s : s + length] = 1.0

                _mark_slice(state_layout.get("actual"))
                for slice_def in (state_layout.get("additional") or {}).values():
                    _mark_slice(slice_def)

        if pref_vec is None and obs_arr_full.size >= 6:
            goal_slice = goal_block if goal_block is not None else obs_arr_full[3:6]
            pref_vec = obs_arr_state.copy()
            tgt_len = min(3, goal_slice.size, pref_vec.size)
            if tgt_len > 0:
                pref_vec[0:tgt_len] = goal_slice[:tgt_len]
                if self._drop_goal_from_state:
                    if pref_vec.size >= 6:
                        pref_vec[3:3 + tgt_len] = goal_slice[:tgt_len]
                    pref_mask = np.zeros_like(pref_vec, dtype=np.float32)
                    pref_mask[0:tgt_len] = 1.0
                    if pref_vec.size >= 6:
                        pref_mask[3:3 + tgt_len] = 1.0
                else:
                    if pref_vec.size >= 6:
                        pref_vec[3:3 + tgt_len] = goal_slice[:tgt_len]
                    if pref_vec.size >= 9:
                        ee_len = min(3, pref_vec.size - 6, goal_slice.size)
                        if ee_len > 0:
                            pref_vec[6 : 6 + ee_len] = goal_slice[:ee_len]
            if not getattr(ActiveInferenceSB3, "_pref_print_once", False):
                ActiveInferenceSB3._pref_print_once = True
                mask_sum = pref_mask.sum() if pref_mask is not None else None
                print("[AIF][debug] obs[:9]:", obs_arr_state[:9], "pref_vec[:9]:", pref_vec[:9], "pref_mask.sum():", mask_sum)
        return obs_arr_state, pref_vec, pref_mask

    def _reset_env(self):
        reset_out = self.env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs = reset_out[0]
        else:
            obs = reset_out
        reset_fn = getattr(getattr(self, "agent", None), "reset_planner_state", None)
        if callable(reset_fn):
            reset_fn()
        return obs

    def _update_final_extrinsic_value(self, final_obs_np: np.ndarray) -> None:
        """Update extrinsic value EMA with per-episode AIF preference improvement before episode reset.

        This ensures successful episodes (which may end before the next meta recompute)
        contribute their improvement signal to the EMA.
        """
        meta_ctrl = getattr(self.agent, "_meta_controller", None)
        if meta_ctrl is None or not getattr(meta_ctrl, "enabled", False):
            return
        if meta_ctrl._episode_start_neg_extrinsic is None:
            return

        try:
            with torch.no_grad():
                obs_t = torch.as_tensor(
                    final_obs_np, dtype=torch.float32, device=self.agent.device
                ).unsqueeze(0)
                final_neg_ext = self.agent._compute_neg_extrinsic_value(obs_t).mean().item()

            start_val = meta_ctrl._episode_start_neg_extrinsic
            denom = max(abs(start_val), 1e-6)
            improvement = math.tanh((start_val - final_neg_ext) / denom)

            # Update EMA via shared helper
            from myGym.active_inference.meta_planning import _update_ema
            meta_ctrl.extrinsic_value_ema = _update_ema(
                meta_ctrl.extrinsic_value_ema, improvement, meta_ctrl.extrinsic_value_ema_beta
            )

            print(f"[META][extrinsic] terminal: episode_improvement={improvement:.6f} "
                  f"start_neg_ext={start_val:.4f} final_neg_ext={final_neg_ext:.4f} "
                  f"ema={meta_ctrl.extrinsic_value_ema:.6f}")

        except Exception:
            pass  # Fail silently if computation fails

    @staticmethod
    def _unpack_step(step_out):
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            return obs, reward, bool(terminated or truncated), info
        obs, reward, done, info = step_out
        return obs, reward, bool(done), info

    # ----------------------- Progress bar helpers -----------------------
    def _start_episode_pbar(self, episode_idx: int):
        if not self._progress_bar_enabled:
            return
        if tqdm is None:
            if not self._warned_no_tqdm:
                print("[AIF] tqdm not installed; progress bar disabled.")
                self._warned_no_tqdm = True
            return
        total = self._max_episode_steps if self._max_episode_steps > 0 else None
        desc = f"Episode {episode_idx}"
        self._episode_pbar = tqdm(total=total, desc=desc, leave=False)

    def _update_episode_pbar(self, step: int):
        if self._episode_pbar is not None:
            delta = step - self._episode_pbar.n
            if delta > 0:
                self._episode_pbar.update(delta)

    def _close_episode_pbar(self):
        if self._episode_pbar is not None:
            self._episode_pbar.close()
            self._episode_pbar = None

    def _log_tb_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        if self._tb_writer is None:
            return
        for key, val in metrics.items():
            if val is None:
                continue
            if isinstance(val, torch.Tensor):
                if val.numel() == 1:
                    self._tb_writer.add_scalar(key, float(val.item()), step)
                else:
                    self._tb_writer.add_histogram(key, val.detach().cpu().numpy(), step)
                continue
            if isinstance(val, (int, float, np.number)):
                self._tb_writer.add_scalar(key, float(val), step)
                continue
            if isinstance(val, (list, tuple, np.ndarray)):
                arr = np.asarray(val)
                if arr.size == 1:
                    self._tb_writer.add_scalar(key, float(arr.reshape(-1)[0]), step)
                else:
                    self._tb_writer.add_histogram(key, arr, step)
                continue
        try:
            self._tb_writer.flush()
        except Exception:
            pass

    def _close_tb_writer(self) -> None:
        if self._tb_writer is None:
            return
        try:
            self._tb_writer.flush()
            self._tb_writer.close()
        except Exception:
            pass
        self._tb_writer = None

    def _setup_callback(self, callback):
        if callback is None:
            return None
        if isinstance(callback, list):
            callback = CallbackList(callback)
        elif not isinstance(callback, BaseCallback):
            callback = CallbackList([callback])
        callback.init_callback(self)
        return callback

    def get_env(self):
        return self.env

    def learn(self, total_timesteps=None, callback=None):
        if total_timesteps is None:
            total_timesteps = int(self.arg_dict.get("steps", 0))
        else:
            self.arg_dict["steps"] = int(total_timesteps)

        callback = self._setup_callback(callback)
        if callback is not None and hasattr(callback, "on_training_start"):
            callback.on_training_start(locals(), globals())

        self._training_start_time = time.time()

        obs = self._reset_env()
        episode_step = 0
        episode_idx = 0
        self._start_episode_pbar(episode_idx)
        debug_obs_print_limit = 5

        for _ in range(total_timesteps):
            flat_obs, pref_vec, pref_mask = self._flatten_obs_with_pref(obs)
            if debug_obs_print_limit > 0:
                # Quick check: raw dict (if present) and flattened slices.
                actual_raw = goal_raw = endeff_raw = None
                if isinstance(obs, dict):
                    actual_raw = obs.get("actual_state")
                    goal_raw = obs.get("goal_state")
                    endeff_raw = obs.get("endeff_xyz")
                    if endeff_raw is None:
                        endeff_raw = (obs.get("additional_obs") or {}).get("endeff_xyz")

                layout = getattr(self, "_state_layout", None) if self._drop_goal_from_state else getattr(self, "_obs_layout", None)
                full_layout = getattr(self, "_obs_layout", None)
                actual_flat = goal_flat = endeff_flat = None
                if layout:
                    a_slice = layout.get("actual")
                    endeff_slice = (layout.get("additional") or {}).get("endeff_xyz")
                    if a_slice:
                        s, e = a_slice
                        actual_flat = flat_obs[s : min(e, flat_obs.shape[0])]
                    if not self._drop_goal_from_state:
                        g_slice = layout.get("goal")
                        if g_slice:
                            s, e = g_slice
                            goal_flat = flat_obs[s : min(e, flat_obs.shape[0])]
                    elif goal_raw is not None:
                        goal_flat = np.asarray(goal_raw, dtype=np.float32).ravel()
                    elif full_layout and "goal" in full_layout:
                        g_start, g_end = full_layout["goal"]
                        full_obs = getattr(self, "_last_full_obs", None)
                        if isinstance(full_obs, np.ndarray) and g_start < full_obs.size:
                            goal_flat = full_obs[g_start : min(g_end, full_obs.size)]
                    if endeff_slice:
                        s, e = endeff_slice
                        endeff_flat = flat_obs[s : min(e, flat_obs.shape[0])]
                else:
                    # Fallback heuristic: assume actual[0:3], goal[3:6], endeff[6:9]
                    actual_flat = flat_obs[0 : min(3, flat_obs.shape[0])]
                    if self._drop_goal_from_state:
                        if flat_obs.shape[0] > 3:
                            endeff_flat = flat_obs[3 : min(6, flat_obs.shape[0])]
                        if goal_raw is not None:
                            goal_flat = np.asarray(goal_raw, dtype=np.float32).ravel()
                    else:
                        goal_flat = flat_obs[3 : min(6, flat_obs.shape[0])]
                        if flat_obs.shape[0] > 6:
                            endeff_flat = flat_obs[6 : min(9, flat_obs.shape[0])]

                print("[AIF][debug_obs] actual_state raw:", actual_raw, "flat:", actual_flat)
                print("[AIF][debug_obs] goal_state   raw:", goal_raw, "flat:", goal_flat)
                print("[AIF][debug_obs] endeff_xyz   raw:", endeff_raw, "flat:", endeff_flat)
                debug_obs_print_limit -= 1
            if pref_vec is not None:
                self.agent.set_preference_mean(pref_vec, pref_mask)

            is_random_step = self.num_timesteps < self.initial_random_steps
            is_first_policy_step = self.num_timesteps == self.initial_random_steps
            buffer_ready = len(self.replay_buffer) >= self.batch_size
            log_info = {}
            if buffer_ready and not is_random_step:
                # model_update_freq=0 means step-based updates disabled (meta controller handles it)
                step_based_update = (
                    self.model_update_freq > 0
                    and (is_first_policy_step or self.num_timesteps % self.model_update_freq == 0)
                )
                if step_based_update:
                    last_info = None
                    model_updates = max(1, int(self.model_updates_per_step))
                    for _ in range(model_updates):
                        last_info = self.agent.update(self.replay_buffer, batch_size=self.batch_size)
                    if last_info:
                        model_loss = last_info.get("F")
                        if model_loss is None:
                            model_loss = last_info.get("mse", last_info.get("nll_obs"))
                        if model_loss is not None:
                            print(f"[AIF] dynamics_model_loss: {float(model_loss):.6f}")
                            log_info["dynamics_train_loss"] = float(model_loss)
                    if not self._model_update_debug_printed:
                        print(
                            f"[AIF][debug] world-model update at step {self.num_timesteps} (grad_steps={model_updates})"
                        )
                        self._model_update_debug_printed = True
                    if last_info and self.arg_dict.get("aif_log_vfe_terms", False):
                        log_info.update(last_info)
                        vfe_parts = []
                        for key in ("F", "kl_state", "kl_theta", "nll_obs", "mse", "logvar_reg"):
                            if key in last_info and last_info[key] is not None:
                                vfe_parts.append(f"{key}={float(last_info[key]):.4f}")
                        for key in sorted(k for k in last_info.keys() if k.startswith("val_")):
                            val = last_info.get(key)
                            if val is None:
                                continue
                            vfe_parts.append(f"{key}={float(val):.4f}")
                        if vfe_parts:
                            print(f"[AIF] step {self.num_timesteps} vfe: " + ", ".join(vfe_parts))
                    self._latest_update_info = last_info
            if is_random_step:
                action = self.action_space.sample()
            else:
                if not self._pref_debug_printed:
                    pref = getattr(self.agent, "preference_mean", None)
                    mask = getattr(self.agent, "preference_mask", None)
                    k = min(10, flat_obs.shape[0])
                    print("[AIF][debug] pref_mean[:10]:", pref[:10].detach().cpu().numpy() if pref is not None else None)
                    mask_sum = mask.sum().item() if mask is not None else None
                    print("[AIF][debug] pref_mask.sum():", mask_sum)
                    print("[AIF][debug] obs[:k] vs pref_vec[:k]:", flat_obs[:k], pref_vec[:k] if pref_vec is not None else None)
                    self._pref_debug_printed = True
                action = self.agent.act(flat_obs)

            self._last_onpolicy_obs = flat_obs.copy()
            self._last_onpolicy_action = np.asarray(action, dtype=np.float32).copy()

            step_out = self.env.step(action)
            next_obs, reward, done, info = self._unpack_step(step_out)
            flat_next_obs, _, _ = self._flatten_obs_with_pref(next_obs)

            self.replay_buffer.add(
                flat_obs,
                np.asarray(action, dtype=np.float32),
                flat_next_obs,
                float(reward),
                float(done),
            )
            self.agent.update_meta_stats(
                flat_obs,
                np.asarray(action, dtype=np.float32),
                flat_next_obs,
            )
            self.num_timesteps += 1
            buffer_ready = len(self.replay_buffer) >= self.batch_size

            meta_count = getattr(self.agent, "_meta_selection_count", 0)
            if meta_count != self._last_meta_selection_count:
                self._last_meta_selection_count = meta_count
                meta_info = getattr(self.agent, "_last_meta_info", None)
                if meta_info:
                    meta_metrics = {f"aif_meta/{k}": v for k, v in meta_info.items() if v is not None}
                    self._log_tb_metrics(meta_metrics, self.num_timesteps)

            # Skip policy learning in deterministic/habitual mode
            # (no useful planning signal when just executing policy mean)
            # NOTE: Get meta_info directly from agent, not from the logging block above
            current_meta_info = getattr(self.agent, "_last_meta_info", None)
            is_deterministic_mode = bool(current_meta_info.get("meta_deterministic", False)) if current_meta_info else False

            # Optional imagined rollouts to train policy_net
            pol_freq = int(self.arg_dict.get("aif_policy_imagination_freq", 0) or 0)
            pol_updates = int(self.arg_dict.get("aif_policy_imagination_updates", 1) or 1)
            if pol_freq > 0 and self.num_timesteps >= self.initial_random_steps and self.num_timesteps % pol_freq == 0:
                if not is_deterministic_mode:  # Skip in deterministic mode
                    horizon = self.arg_dict.get("aif_policy_imagination_horizon", None)
                    pol_info = None
                    for _ in range(pol_updates):
                        pol_info = self.agent.update_policy_with_imagined_rollouts(
                            self.replay_buffer,
                            batch_size=self.batch_size,
                            rollout_horizon=horizon or self.arg_dict.get("aif_plan_horizon", self.agent.cem_horizon),
                        )
                    if pol_info and self.arg_dict.get("aif_log_efe_terms", False):
                        log_info.update({f"{k}": v for k, v in pol_info.items()})
            
            # Optional policy_net updates using real transitions scored by EFE
            # Weight by extrinsic value (episode performance); disabled for deterministic mode
            real_efe_freq = int(self.arg_dict.get("aif_policy_real_efe_freq", 0) or 0)
            real_efe_updates = int(self.arg_dict.get("aif_policy_real_efe_updates", 1) or 1)
            real_info = None
            if real_efe_freq > 0 and self.num_timesteps >= self.initial_random_steps and self.num_timesteps % real_efe_freq == 0:
                # Weight by extrinsic value EMA from meta-planning (episode improvement)
                compute_weight = 0.0
                if current_meta_info and not is_deterministic_mode:
                    compute_weight = max(0.0, float(current_meta_info.get("meta_extrinsic_value", 0.0)))
                for _ in range(real_efe_updates):
                    real_info = self.agent.update_policy_with_real_efe(
                        obs_np=self._last_onpolicy_obs,
                        action_np=self._last_onpolicy_action,
                        next_obs_np=flat_next_obs,
                        compute_weight=compute_weight,
                    )
                if real_info and self.arg_dict.get("aif_log_efe_terms", False):
                    log_info.update(real_info)

            if self.log_freq and self.num_timesteps % self.log_freq == 0:
                if not is_random_step:
                    plan_score = getattr(self.agent, "_cached_policy_score", None)
                    if plan_score is not None:
                        log_info["policy_plan_efe"] = float(plan_score)
                    action_logprob = getattr(self.agent, "_last_action_logprob", None)
                    if action_logprob is not None:
                        log_info["policy_action_logprob"] = float(action_logprob.detach().cpu().item())
                    # Log policy behavioral cloning from planned actions
                    bc_info = getattr(self.agent, "_last_policy_bc_info", None)
                    if bc_info:
                        log_info.update({f"policy_bc/{k}": v for k, v in bc_info.items()})
                    with torch.no_grad():
                        state_t = torch.as_tensor(flat_obs, dtype=torch.float32, device=self.agent.device).unsqueeze(0)
                        if not self.agent.fully_observable_mdp:
                            mu, _ = self.agent.encoder(state_t)
                            state_t = mu
                        mean, std = self.agent.policy_net(state_t, sample=False)
                        log_info["policy_mean"] = mean.squeeze(0).detach().cpu().numpy().tolist()
                        log_info["policy_std"] = std.squeeze(0).detach().cpu().numpy().tolist()
                        entropy = (0.5 * (1.0 + math.log(2 * math.pi)) + torch.log(std)).sum(dim=-1)
                        log_info["policy_entropy"] = float(entropy.item())
                # Add meta controller EMAs to log output
                meta_ctrl = getattr(self.agent, "meta_controller", None)
                if meta_ctrl is not None and meta_ctrl.enabled:
                    log_info["model_error_ema"] = float(meta_ctrl.model_error_ema)
                    log_info["model_uncert_ema"] = float(meta_ctrl.model_uncert_ema)
                    log_info["policy_uncert_ema"] = float(meta_ctrl.policy_uncert_ema)
                    # Meta-risk is pre-computed per mode.
                if log_info:
                    tb_metrics = {f"aif/{k}": v for k, v in log_info.items()}
                    elapsed = time.time() - self._training_start_time
                    if elapsed > 0:
                        tb_metrics["time/fps"] = self.num_timesteps / elapsed
                    self._log_tb_metrics(tb_metrics, self.num_timesteps)
                if log_info:
                    print(f"[AIF] step {self.num_timesteps}: {log_info}")

            if callback is not None:
                callback.num_timesteps = self.num_timesteps
                # Prefer SB3-style on_step when available; otherwise fallback to __call__.
                keep_training = True
                if hasattr(callback, "on_step"):
                    keep_training = bool(callback.on_step())
                else:
                    try:
                        keep_training = bool(callback())
                    except TypeError:
                        keep_training = True
                if not keep_training:
                    break

            # Update extrinsic value EMA with terminal observation before episode reset
            # This ensures successful episodes contribute their improvement signal
            if done:
                self._update_final_extrinsic_value(flat_next_obs)

            obs = self._reset_env() if done else next_obs
            episode_step += 1
            self._update_episode_pbar(episode_step)

            if done:
                if isinstance(next_obs, dict):
                    ee_arr = np.asarray(next_obs.get("endeff_xyz"), dtype=np.float32).ravel() if "endeff_xyz" in next_obs else None
                else:
                    ee_arr = None
                if ee_arr is not None and ee_arr.size >= 3:
                    print(f"[AIF] episode {episode_idx} end effector final position: {ee_arr[:3]}")
                # Negative extrinsic value (distance to preferences) on real observation
                neg_extrinsic_real = None
                try:
                    pref_mean = getattr(self.agent, "preference_mean", None)
                    pref_logvar = getattr(self.agent, "preference_logvar", None)
                    if pref_mean is not None and pref_logvar is not None and flat_next_obs is not None:
                        with torch.no_grad():
                            obs_t = torch.as_tensor(
                                flat_next_obs, dtype=torch.float32, device=self.agent.device
                            ).unsqueeze(0)
                            # Compute -E[ln p(o|C)] = negative extrinsic value
                            neg_extrinsic_real = self.agent._compute_neg_extrinsic_value(obs_t).mean().item()
                except Exception:
                    neg_extrinsic_real = None
                if neg_extrinsic_real is not None:
                    print(f"[AIF] episode {episode_idx} real_obs_neg_extrinsic: {neg_extrinsic_real:.4f}")
                # Planner score extrema from last planning call
                best_score = getattr(self.agent, "_last_plan_score_best", None)
                worst_score = getattr(self.agent, "_last_plan_score_worst", None)
                if best_score is not None and worst_score is not None:
                    print(f"[AIF] episode {episode_idx} plan_efe_best: {best_score:.4f} plan_efe_worst: {worst_score:.4f}")
                self._close_episode_pbar()
                episode_idx += 1
                episode_step = 0
                obs = self._reset_env()
                self._start_episode_pbar(episode_idx)

        if callback is not None and hasattr(callback, "on_training_end"):
            callback.on_training_end()
        self._close_tb_writer()
        self._close_episode_pbar()
        return self

    def predict(self, observation, deterministic: bool = False):
        # Before any model training, use random actions (like PETS)
        if self.num_timesteps < self.initial_random_steps:
            return self.action_space.sample(), None
        obs_vec, pref_vec, pref_mask = self._flatten_obs_with_pref(observation)
        if pref_vec is not None:
            self.agent.set_preference_mean(pref_vec, pref_mask)
        eval_mode = self._policy_eval_mode
        policy_mode = str(self.arg_dict.get("aif_policy_mode", "cem")).lower()
        if eval_mode == "mean" and policy_mode == "policy_net":
            action = self.agent.policy_net_mean_action(obs_vec)
        else:
            action = self.agent.act(obs_vec)
        action = np.asarray(action, dtype=np.float32)
        if action.ndim > 1:
            action = action.squeeze()
        return action, None

    def save(self, path, steps=None, best=False, **_kwargs):
        base_dir = path
        os.makedirs(base_dir, exist_ok=True)

        if steps is not None:
            try:
                with open(os.path.join(base_dir, "trained_steps.txt"), "a") as f:
                    f.write(f"{int(steps)}\n")
            except Exception:
                pass

        if best:
            target_dir = os.path.join(base_dir, "best_model")
        elif steps is not None:
            target_dir = os.path.join(base_dir, f"steps_{int(steps)}")
        else:
            target_dir = base_dir

        os.makedirs(target_dir, exist_ok=True)
        torch.save(
            {
                "agent_state": self.agent.state_dict(),
                "optimizer_state": self.agent.optimizer.state_dict(),
                "policy_optimizer_state": self.agent.policy_optimizer.state_dict(),
                "num_timesteps": self.num_timesteps,
                "arg_dict": self.arg_dict,
            },
            os.path.join(target_dir, "aif_agent.pt"),
        )

    @staticmethod
    def _find_train_root(path: str) -> str:
        candidate = path if os.path.isdir(path) else os.path.dirname(path)
        for _ in range(3):
            if os.path.isfile(os.path.join(candidate, "train.json")):
                return candidate
            parent = os.path.dirname(candidate)
            if parent == candidate:
                break
            candidate = parent
        return path

    @staticmethod
    def _find_checkpoint_dir(path: str) -> str:
        if os.path.isfile(os.path.join(path, "aif_agent.pt")):
            return path
        best_dir = os.path.join(path, "best_model")
        if os.path.isfile(os.path.join(best_dir, "aif_agent.pt")):
            return best_dir

        trained_steps = os.path.join(path, "trained_steps.txt")
        if os.path.isfile(trained_steps):
            try:
                with open(trained_steps, "r") as f:
                    steps = [int(ln) for ln in f.read().splitlines() if ln.strip().isdigit()]
                for step in reversed(steps):
                    candidate = os.path.join(path, f"steps_{step}")
                    if os.path.isfile(os.path.join(candidate, "aif_agent.pt")):
                        return candidate
            except Exception:
                pass

        step_dirs = []
        try:
            for entry in os.listdir(path):
                if entry.startswith("steps_"):
                    step_dirs.append((int(entry.split("_")[1]), os.path.join(path, entry)))
        except Exception:
            step_dirs = []
        if step_dirs:
            _, latest_dir = sorted(step_dirs, key=lambda x: x[0])[-1]
            if os.path.isfile(os.path.join(latest_dir, "aif_agent.pt")):
                return latest_dir
        return path

    @classmethod
    def load(cls, path, env=None, device="cpu", **_unused_kwargs):
        if env is None:
            raise ValueError("Environment is required when loading ActiveInferenceSB3.")
        base_path = path if os.path.isdir(path) else os.path.dirname(path)
        train_root = cls._find_train_root(base_path)
        ckpt_dir = cls._find_checkpoint_dir(base_path)

        cfg_path = os.path.join(train_root, "train.json")
        loaded_args: Dict[str, Any] = {}
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r") as f:
                    loaded_args = json.load(f)
            except Exception:
                loaded_args = {}

        loaded_args["aif_device"] = device
        instance = cls(env, loaded_args, device=device)

        state_path = os.path.join(ckpt_dir, "aif_agent.pt")
        state = torch.load(state_path, map_location=device) if os.path.isfile(state_path) else {}
        agent_state = state.get("agent_state")
        if agent_state:
            instance.agent.load_state_dict(agent_state)
        optim_state = state.get("optimizer_state")
        if optim_state:
            instance.agent.optimizer.load_state_dict(optim_state)
        policy_optim_state = state.get("policy_optimizer_state")
        if policy_optim_state:
            instance.agent.policy_optimizer.load_state_dict(policy_optim_state)
        instance.num_timesteps = int(state.get("num_timesteps", 0))
        return instance


class MetaAIFActionWrapper(gym.ActionWrapper):
    """
    Placeholder action wrapper for Meta-AIF runs. Currently passes actions through
    unchanged but keeps the API stable for train/test scripts.
    """

    def action(self, action):
        return action


class MetaAIFSB3(ActiveInferenceSB3):
    """
    Meta-AIF placeholder that reuses the base ActiveInferenceSB3 implementation.
    """

    def __init__(self, env: gym.Env, arg_dict: Optional[Dict[str, Any]] = None, device: Optional[str] = None, **kwargs):
        print("MetaAIFSB3 currently reuses ActiveInferenceSB3 behaviour.")
        if arg_dict is None:
            arg_dict = {}
        if "aif_meta_enabled" not in arg_dict:
            arg_dict = dict(arg_dict)
            arg_dict["aif_meta_enabled"] = True
        super().__init__(env, arg_dict=arg_dict, device=device, **kwargs)
