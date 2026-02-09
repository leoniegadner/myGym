import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class MetaFeatures:
    """Sufficient statistics for meta-level EFE computation.

    Note: Habitual deviation (used in preference risk) is pre-computed per mode,
    not a feature of the current state. See _precompute_mode_arrays.
    """

    model_error: float = 0.0  # Proxy for ambiguity potential
    model_uncert: float = 0.0  # Epistemic uncertainty (model)
    policy_uncert: float = 0.0  # Epistemic uncertainty (policy)
    extrinsic_value: float = 0.0  # Policy success: improvement rate toward preferred state (positive = improving)

    @property
    def total_uncertainty(self) -> float:
        return self.model_uncert + self.policy_uncert

    @property
    def model_reliability(self) -> float:
        return math.exp(-self.model_error)

    @property
    def realizable_epistemic(self) -> float:
        """Epistemic value only realizable with reliable model."""
        return self.total_uncertainty * self.model_reliability

    def to_dict(self) -> Dict[str, float]:
        return {
            "model_error": self.model_error,
            "model_uncert": self.model_uncert,
            "policy_uncert": self.policy_uncert,
            "extrinsic_value": self.extrinsic_value,
        }


class MetaGenerativeModel:
    """
    Explicit specification of a meta-level generative model with hidden policy quality.

    Hidden state:
      - θ ∈ [0, 1] (policy quality)

    Prior belief:
      - Q(θ): current uncertainty about policy quality (Beta distribution)

    Likelihood:
      - P(o | θ, mode): how observations depend on policy quality and mode
      - Observations: (extrinsic_value, policy_uncert, model_uncert, effort)

    Observation preferences p(o|C):
      - Preferences are properly defined over the OBSERVATION space, not modes
      - extrinsic_value: prefer high (μ=1, goal achievement)
      - policy_uncert: prefer low (μ=0, confident policy)
      - model_uncert: prefer low (μ=0, reliable model)
      - effort: prefer low (μ=0, computational efficiency)

    Expected Free Energy:
      - G = Ambiguity + Risk
      - Ambiguity = E_Q(θ)[H(P(o|θ, mode))] - uncertainty about outcomes
      - Risk = KL[q(o|mode) || p(o|C)] - divergence from preferred observations
        where q(o|mode) = E_Q(θ)[P(o|θ, mode)]

    This formulation properly separates the action space (modes) from the state/
    observation space, computing risk as how far expected observations under each
    mode deviate from preferences over all observation dimensions.
    """

    def __init__(
        self,
        policy_ambiguity_weight: float,
        model_ambiguity_weight: float,
        total_capacity: np.ndarray,
        depth_capacity: np.ndarray,
        complexity: np.ndarray,
        habit_deviation: np.ndarray,
        is_deterministic: np.ndarray,
        initial_belief_alpha: float = 1.0,
        initial_belief_beta: float = 1.0,
        observation_weight: float = 1.0,
        belief_decay: float = 0.995,
        pref_extrinsic_precision: float = 1.0,
        pref_policy_uncert_precision: float = 1.0,
        pref_model_uncert_precision: float = 1.0,
        pref_effort_precision: float = 1.0,
    ):
        """
        Initialize the meta generative model with pre-computed mode arrays.

        Args:
            policy_ambiguity_weight: Weight on ambiguity term E_Q[H(P(o|θ, mode))]
            model_ambiguity_weight: Scales model uncertainty in likelihood sharpness
            total_capacity: Planning capacity per mode [n_modes]
            depth_capacity: Horizon depth per mode [n_modes]
            complexity: Computational complexity per mode [n_modes]
            habit_deviation: Deviation from habitual prior per mode [n_modes]
            is_deterministic: Boolean mask for deterministic modes [n_modes]
            initial_belief_alpha: Initial α for Beta prior over policy quality
            initial_belief_beta: Initial β for Beta prior over policy quality
            observation_weight: Base pseudo-count weight per observation
            belief_decay: Decay factor for belief (1.0 = no decay, <1.0 allows adaptation)
            pref_extrinsic_precision: Precision for extrinsic_value preference (μ=1)
            pref_policy_uncert_precision: Precision for policy_uncert preference (μ=0)
            pref_model_uncert_precision: Precision for model_uncert preference (μ=0)
            pref_effort_precision: Precision for effort preference (μ=0)
        """
        self.policy_ambiguity_weight = policy_ambiguity_weight
        self.model_ambiguity_weight = model_ambiguity_weight

        # Observation preference precisions (higher = stronger preference)
        self.pref_extrinsic_precision = pref_extrinsic_precision
        self.pref_policy_uncert_precision = pref_policy_uncert_precision
        self.pref_model_uncert_precision = pref_model_uncert_precision
        self.pref_effort_precision = pref_effort_precision

        # Pre-computed mode characteristics (vectorized)
        self._total_capacity = total_capacity
        self._depth_capacity = depth_capacity
        self._complexity = complexity
        self._habit_deviation = habit_deviation
        self._is_deterministic = is_deterministic

        # Normalize capacities to [0, 1]
        self._max_capacity = np.max(total_capacity) + 1e-6
        self._max_depth = max(np.max(depth_capacity), 1.0)
        self._max_complexity = np.max(complexity) + 1e-6

        # Numerical integration grid for policy quality θ ∈ (0, 1)
        self._theta_eps = 1e-6
        self._theta_grid = np.linspace(self._theta_eps, 1.0 - self._theta_eps, num=51, dtype=np.float64)
        self._theta_min_concentration = 2.0  # Uniform Beta when policy_uncert is maximal
        self._theta_max_concentration = 50.0  # Narrow Beta when policy_uncert is minimal
        self._policy_quality_mean = 0.5  # Default mean when no estimate is available
        self._min_sensitivity = 0.1
        self._sensitivity_gain = 1.0
        self._entropy_norm = math.log(2.0)

        # Persistent Bayesian belief over policy quality θ ~ Beta(α, β)
        # This is the posterior from previous observations, used as prior for next decision
        self._belief_alpha: float = float(initial_belief_alpha)
        self._belief_beta: float = float(initial_belief_beta)
        self._observation_weight: float = float(observation_weight)
        self._belief_decay: float = float(belief_decay)

    def normalize_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Normalize raw features to [0, 1] range for prior/likelihood shaping.

        Returns dict with:
          - normalized_error: sigmoid(raw_error) → (0, 1)
          - normalized_model_uncert: x/(1+x) → [0, 1)
          - normalized_policy_uncert: x/(1+x) → [0, 1)
          - normalized_extrinsic_value: tanh(x) → (-1, 1), positive = improving
          - model_reliability: combined reliability measure
        """
        raw_error = float(features.get("model_error", 0.0))
        raw_model_uncert = max(0.0, float(features.get("model_uncert", 0.0)))
        if self.model_ambiguity_weight == 0.0:
            raw_model_uncert = 0.0
        raw_policy_uncert = max(0.0, float(features.get("policy_uncert", 0.0)))
        raw_extrinsic_value = float(features.get("extrinsic_value", 0.0))

        # Numerically stable sigmoid for NLL error: (-∞, +∞) → (0, 1)
        if raw_error >= 0:
            z = math.exp(-raw_error)
            normalized_error = z / (1.0 + z)
        else:
            z = math.exp(raw_error)
            normalized_error = 1.0 / (1.0 + z)
        # Saturation x/(1+x) for uncertainties: [0, ∞) → [0, 1)
        normalized_model_uncert = raw_model_uncert / (1.0 + raw_model_uncert)
        normalized_policy_uncert = raw_policy_uncert / (1.0 + raw_policy_uncert)
        # Tanh for extrinsic value: (-∞, +∞) → (-1, 1), positive = improvement
        normalized_extrinsic_value = math.tanh(raw_extrinsic_value)
        # Reliability: both error and uncertainty must be low
        # Clamp to prevent underflow (exp of large negative → 0)
        uncert_factor = math.exp(-min(500.0, raw_model_uncert))
        model_reliability = normalized_error * uncert_factor

        return {
            "normalized_error": normalized_error,
            "normalized_model_uncert": normalized_model_uncert,
            "normalized_policy_uncert": normalized_policy_uncert,
            "normalized_extrinsic_value": normalized_extrinsic_value,
            "model_reliability": model_reliability,
        }

    def bayesian_update(
        self, normalized_observations: Dict[str, float], mode_idx: int = 0
    ) -> Dict[str, float]:
        """
        Bayesian update of system quality belief based on multiple observations.

        Uses the same likelihood model as EFE computation (_likelihood):
          P(o | θ, mode) is a multivariate Gaussian over observations.

        The observations were generated under a specific mode, so we condition
        on that mode when computing the likelihood for inference.

        Inference:
          posterior(θ) ∝ P(observations | θ, mode) × prior(θ)

        Uses moment-matching to fit posterior to Beta distribution.

        Args:
            normalized_observations: Dict with normalized features
            mode_idx: Index of the mode that was active when observations were generated

        Returns:
            Dict with updated belief parameters for logging
        """
        # Get actual observations
        obs_ev = float(normalized_observations.get("normalized_extrinsic_value", 0.0))
        obs_pu = float(normalized_observations.get("normalized_policy_uncert", 0.5))
        obs_mu = float(normalized_observations.get("normalized_model_uncert", 0.5))
        observations = np.array([obs_ev, obs_pu, obs_mu])  # [3]

        # Apply decay to allow adaptation to non-stationary quality
        if self._belief_decay < 1.0:
            self._belief_alpha = 1.0 + self._belief_decay * (self._belief_alpha - 1.0)
            self._belief_beta = 1.0 + self._belief_decay * (self._belief_beta - 1.0)

        # Compute prior weights over θ grid
        prior_weights = self._beta_weights(self._belief_alpha, self._belief_beta)

        # Get likelihood parameters from the generative model
        # expected: [n_theta, n_modes, 3], variances: [n_theta, n_modes, 3]
        expected, variances = self._likelihood(self._theta_grid, normalized_observations)

        # Extract parameters for the specific mode that generated the observations
        # expected_mode: [n_theta, 3], variances_mode: [n_theta, 3]
        expected_mode = expected[:, mode_idx, :]
        variances_mode = variances[:, mode_idx, :]

        # Scale variances by observation weight (lower weight = higher variance = less informative)
        variances_mode = variances_mode / (self._observation_weight + 1e-6)

        # Compute log-likelihood for each θ: log P(observations | θ, mode)
        # Sum over observation dimensions (assuming independence)
        diff = observations[None, :] - expected_mode  # [n_theta, 3]
        log_likelihood = -0.5 * np.sum(diff**2 / (variances_mode + 1e-12), axis=-1)  # [n_theta]
        log_likelihood = log_likelihood - np.max(log_likelihood)
        likelihood = np.exp(log_likelihood)

        # Posterior ∝ likelihood × prior
        unnormalized_posterior = likelihood * prior_weights
        posterior_weights = unnormalized_posterior / (np.sum(unnormalized_posterior) + 1e-12)

        # Compute posterior moments
        posterior_mean = float(np.sum(posterior_weights * self._theta_grid))
        posterior_var = float(np.sum(posterior_weights * (self._theta_grid - posterior_mean) ** 2))

        # Fit Beta to posterior moments (method of moments)
        posterior_mean = max(self._theta_eps, min(1.0 - self._theta_eps, posterior_mean))
        max_var = posterior_mean * (1.0 - posterior_mean) - 1e-6
        if posterior_var > 0 and posterior_var < max_var:
            concentration = posterior_mean * (1.0 - posterior_mean) / posterior_var - 1.0
            concentration = max(self._theta_min_concentration, min(100.0, concentration))
            self._belief_alpha = posterior_mean * concentration
            self._belief_beta = (1.0 - posterior_mean) * concentration

        return {
            "belief_alpha": self._belief_alpha,
            "belief_beta": self._belief_beta,
            "belief_mean": self._belief_alpha / (self._belief_alpha + self._belief_beta),
            "belief_concentration": self._belief_alpha + self._belief_beta,
            "posterior_mean": posterior_mean,
            "posterior_var": posterior_var,
        }

    def _theta_prior(self, normalized_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Construct Q(θ) using the persistent Bayesian belief.

        The belief (α, β) is updated via bayesian_update() and represents our
        posterior over policy quality from all previous observations. This posterior
        becomes the prior for the current decision.

        Policy uncertainty from the current state modulates how much we trust
        the accumulated belief vs. being more uncertain:
          - High policy_uncert → scale down effective counts (spread out the prior)
          - Low policy_uncert → use full belief concentration
        """
        policy_uncert = float(normalized_features["normalized_policy_uncert"])

        # Use persistent Bayesian belief as base
        alpha_belief = self._belief_alpha
        beta_belief = self._belief_beta
        belief_mean = alpha_belief / (alpha_belief + beta_belief)
        belief_concentration = alpha_belief + beta_belief

        # Policy uncertainty modulates effective concentration
        # High uncertainty → treat as if we have less data (scale down counts)
        # This allows current state information to influence the prior spread
        uncertainty_scale = 1.0 - 0.5 * policy_uncert  # Range [0.5, 1.0]
        effective_concentration = max(
            self._theta_min_concentration,
            belief_concentration * uncertainty_scale
        )

        # Reconstruct α, β with scaled concentration but preserved mean
        alpha = belief_mean * effective_concentration
        beta = (1.0 - belief_mean) * effective_concentration

        weights = self._beta_weights(alpha, beta)
        return {
            "mean": belief_mean,
            "concentration": effective_concentration,
            "alpha": alpha,
            "beta": beta,
            "belief_alpha": alpha_belief,
            "belief_beta": beta_belief,
            "weights": weights,
        }

    def _beta_weights(self, alpha: float, beta: float) -> np.ndarray:
        """Return normalized Beta(α, β) weights over the θ grid."""
        theta = self._theta_grid
        log_beta = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
        log_pdf = (alpha - 1.0) * np.log(theta) + (beta - 1.0) * np.log(1.0 - theta) - log_beta
        log_pdf = log_pdf - np.max(log_pdf)
        weights = np.exp(log_pdf)
        return weights / (np.sum(weights) + 1e-12)

    def _likelihood(
        self, theta_values: np.ndarray, normalized_features: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute likelihood P(o | θ, mode) for all θ and modes.

        Generative model: observations given hidden state θ and planning mode.

        P(o | θ, mode) is a multivariate Gaussian over:
          - extrinsic_value | θ, mode ~ N(μ_ev(θ, mode), σ²_ev(θ, mode))
          - policy_uncert | θ ~ N(1 - θ, σ²_pu)
          - model_uncert | θ ~ N(1 - θ, σ²_mu)

        Mode affects extrinsic value prediction:
          - Higher capacity modes leverage good θ more effectively
          - μ_ev(θ, mode) = (2θ - 1) + capacity × θ × (1 - model_uncert)
          - Planning amplifies θ when model is reliable

        Mode affects precision of extrinsic value:
          - Higher capacity + good θ → more predictable outcomes
          - σ²_ev(θ, mode) = base_var / (1 + capacity × θ)
        
        Key insight: Planning helps when model is GOOD and θ is BAD.
        The variance of outcomes should differ DRAMATICALLY based on mode:
        - No planning + bad θ: very unpredictable outcomes
        - Planning + good model + bad θ: can predict improvement

        Returns:
            Tuple of (expected_observations, variances) each [n_theta, n_modes, 3]
            where 3 = (extrinsic_value, policy_uncert, model_uncert)
        """
        model_uncert = float(normalized_features["normalized_model_uncert"])
        model_reliability = 1.0 - min(1.0, model_uncert * self.model_ambiguity_weight)

        capacity = np.clip(self._total_capacity / self._max_capacity, 0.0, 1.0)
        capacity = capacity[None, :]  # [1, n_modes]
        theta = theta_values[:, None]  # [n_theta, 1]
        n_theta = len(theta_values)
        n_modes = len(self._total_capacity)

        room_to_improve = 1.0 - theta

        # === EXPECTED EXTRINSIC VALUE ===
        base_ev = 2.0 * theta - 1.0
        planning_boost = capacity * room_to_improve * model_reliability
        mu_ev = np.clip(base_ev + planning_boost, -1.0, 1.0)

        # === VARIANCE OF EXTRINSIC VALUE ===
        # Requirements:
        # 1. World model BAD + policy quality BAD → HIGH variance for BOTH modes
        # 2. World model GOOD + policy variance HIGH → LOW for deliberate, HIGH for deterministic
        # 3. Policy quality GOOD + variance LOW → LOW for deterministic (so it can take over)

        # Deterministic mode: variance from policy quality
        # When theta is high → low variance (good policy = predictable)
        # When theta is low → high variance (bad policy = unpredictable)
        var_deterministic = 0.01 + room_to_improve**2.65  # [n_theta, 1]

        # Deliberate mode: planning can reduce variance when there's room to improve AND model is good
        # Key insight:
        #   - When room_to_improve is HIGH and model is GOOD → planning reduces variance significantly
        #   - When room_to_improve is LOW (theta is high) → planning benefit is minimal
        #   - When model is BAD → planning adds noise instead of reducing it
        model_unreliability = 1.0 - model_reliability

        # Planning benefit scales with both room to improve AND model reliability
        planning_variance_reduction = room_to_improve**2 * model_reliability * 0.8  # Can reduce up to 80%
        planning_noise = model_unreliability * 1.0  # But adds noise when model is bad

        var_deliberate = var_deterministic - planning_variance_reduction + planning_noise
        var_deliberate = np.maximum(var_deliberate, 0.01)  # Ensure minimum variance

        # Interpolate based on capacity
        var_ev = var_deterministic * (1.0 - capacity) + var_deliberate * capacity

        # Ensure minimum variance for numerical stability
        var_ev = np.maximum(var_ev, 0.01)

        # === POLICY/MODEL UNCERTAINTY (mode-independent) ===
        mu_pu = 1.0 - theta
        mu_mu = 1.0 - theta
        # These use fixed variance since they don't depend on mode choice
        var_pu = np.full((n_theta, n_modes), 0.25)
        var_mu = np.full((n_theta, n_modes), 0.25)

        # Stack into [n_theta, n_modes, 3]
        expected = np.stack([
            mu_ev,
            np.broadcast_to(mu_pu, (n_theta, n_modes)),
            np.broadcast_to(mu_mu, (n_theta, n_modes))
        ], axis=-1)
        variances = np.stack([var_ev, var_pu, var_mu], axis=-1)

        return expected, variances

    def _observation_preferences(self) -> Dict[str, Any]:
        """
        Define preferences p(o|C) over the observation space.

        Preferences are over observations, NOT modes:
          - extrinsic_value: prefer high (close to 1)
          - policy_uncert: prefer low (close to 0)
          - model_uncert: prefer low (close to 0)
          - effort: prefer low (close to 0)

        Returns Gaussian preference parameters for each observation dimension.
        """
        return {
            "extrinsic_value": {"mean": 1.0, "precision": self.pref_extrinsic_precision},
            "policy_uncert": {"mean": 0.0, "precision": self.pref_policy_uncert_precision},
            "model_uncert": {"mean": 0.0, "precision": self.pref_model_uncert_precision},
            "effort": {"mean": 0.0, "precision": self.pref_effort_precision},
        }

    def _expected_observations(
        self, theta_values: np.ndarray, theta_weights: np.ndarray,
        normalized_features: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute q(o|mode) = E_Q(θ)[P(o|θ, mode)] for each mode.

        Returns:
            expected_obs: [n_modes, 4] - expected observation per mode
            obs_variance: [n_modes, 4] - variance of observations per mode
        """
        expected, variances = self._likelihood(theta_values, normalized_features)
        # expected: [n_theta, n_modes, 3], variances: [n_theta, n_modes, 3]

        # Marginalize over θ: E_Q(θ)[μ(θ, mode)]
        # weights: [n_theta] -> [n_theta, 1, 1] for broadcasting
        w = theta_weights[:, None, None]

        expected_obs = np.sum(expected * w, axis=0)  # [n_modes, 3]

        # Total variance = E[Var] + Var[E] (law of total variance)
        expected_var = np.sum(variances * w, axis=0)  # E[Var]
        var_of_expected = np.sum(w * (expected - expected_obs[None, :, :]) ** 2, axis=0)  # Var[E]
        obs_variance = expected_var + var_of_expected  # [n_modes, 3]

        # Add effort as 4th observation dimension
        # P(effort | mode) = δ(effort - normalized_cost[mode])
        # So expected effort = normalized_cost, variance ≈ 0 (deterministic)
        normalized_cost = self._complexity / (self._max_complexity + 1e-6)  # [n_modes]

        expected_obs = np.concatenate([
            expected_obs,
            normalized_cost[:, None]  # [n_modes, 1]
        ], axis=-1)  # [n_modes, 4]

        obs_variance = np.concatenate([
            obs_variance,
            np.full((len(self._complexity), 1), 1e-6)  # near-zero variance for deterministic effort
        ], axis=-1)  # [n_modes, 4]

        return expected_obs, obs_variance

    def _compute_risk(
        self, theta_values: np.ndarray, theta_weights: np.ndarray,
        normalized_features: Dict[str, float]
    ) -> np.ndarray:
        """
        Compute Risk = KL[q(o|mode) || p(o|C)] for each mode.

        For Gaussian q and p, KL has closed form:
        KL = 0.5 * (log(σ_p²/σ_q²) + σ_q²/σ_p² + (μ_q - μ_p)²/σ_p² - 1)

        Returns:
            risk: [n_modes] - KL divergence for each mode
        """
        prefs = self._observation_preferences()
        expected_obs, obs_variance = self._expected_observations(
            theta_values, theta_weights, normalized_features
        )
        # expected_obs: [n_modes, 4], obs_variance: [n_modes, 4]

        # Build preference parameters as arrays
        pref_means = np.array([
            prefs["extrinsic_value"]["mean"],
            prefs["policy_uncert"]["mean"],
            prefs["model_uncert"]["mean"],
            prefs["effort"]["mean"],
        ])  # [4]

        pref_precisions = np.array([
            prefs["extrinsic_value"]["precision"],
            prefs["policy_uncert"]["precision"],
            prefs["model_uncert"]["precision"],
            prefs["effort"]["precision"],
        ])  # [4]

        pref_variances = 1.0 / (pref_precisions + 1e-6)  # [4]

        # KL[q || p] for each mode, summed over observation dimensions
        # Using simplified form for Gaussians
        mu_diff_sq = (expected_obs - pref_means[None, :]) ** 2  # [n_modes, 4]

        # KL = 0.5 * (σ_q²/σ_p² + (μ_q - μ_p)²/σ_p² - 1 + log(σ_p²/σ_q²))
        kl_per_dim = 0.5 * (
            obs_variance / pref_variances[None, :] +
            mu_diff_sq / pref_variances[None, :] - 1.0 +
            np.log(pref_variances[None, :] / (obs_variance + 1e-6))
        )  # [n_modes, 4]

        risk = np.sum(kl_per_dim, axis=-1)  # [n_modes]

        return risk

    def _expected_entropy(
        self, theta_values: np.ndarray, theta_weights: np.ndarray,
        normalized_features: Dict[str, float]
    ) -> np.ndarray:
        """
        Compute E_Q(θ)[H(P(o | θ, mode))] for all modes.
        
        IMPORTANT: Only extrinsic_value variance is mode-conditional.
        We focus the ambiguity calculation on this dimension to avoid
        diluting the signal with mode-independent observations.
        """
        _, variances = self._likelihood(theta_values, normalized_features)
        # variances: [n_theta, n_modes, 3]
        
        # Option A: Use only extrinsic value for ambiguity (conceptually cleanest)
        var_ev = variances[:, :, 0]  # [n_theta, n_modes]
        
        # Differential entropy of Gaussian: H = 0.5 * log(2πeσ²)
        # For comparison across modes, we can use 0.5 * log(σ²)
        log_var_ev = np.log(var_ev + 1e-12)
        entropy = 0.5 * (1.0 + np.log(2 * np.pi) + log_var_ev)
        
        # Normalize to reasonable range [0, 1]
        # With var ranging from 0.05 to 1.0:
        #   log(0.05) ≈ -3.0, log(1.0) = 0
        #   entropy range ≈ [0.5*(1+1.84-3), 0.5*(1+1.84+0)] = [-0.08, 1.42]
        # Shift and scale to [0, 1]
        entropy_min = 0.5 * (1.0 + np.log(2 * np.pi) + np.log(0.01))  # min possible
        entropy_max = 0.5 * (1.0 + np.log(2 * np.pi) + np.log(2.0))   # max possible
        entropy = (entropy - entropy_min) / (entropy_max - entropy_min + 1e-6)
        entropy = np.clip(entropy, 0.0, 1.0)

        # Expected entropy under Q(θ)
        return np.sum(entropy * theta_weights[:, None], axis=0)  # [n_modes]

    def expected_free_energy(self, features: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute Expected Free Energy G(mode) for all modes.

        G = Ambiguity + Risk

        where:
          - Ambiguity = E_Q(θ)[H(P(o|θ, mode))] (uncertainty about outcomes)
          - Risk = KL[q(o|mode) || p(o|C)] (divergence from preferred observations)

        Preferences p(o|C) are now properly defined over OBSERVATIONS, not modes:
          - extrinsic_value: prefer high (close to 1)
          - policy_uncert: prefer low (close to 0)
          - model_uncert: prefer low (close to 0)
          - effort: prefer low (close to 0)

        Args:
            features: Raw features dict with model_error, model_uncert, policy_uncert

        Returns:
            Tuple of (G scores [n_modes], debug_info dict)
        """
        # Normalize features
        norm_features = self.normalize_features(features)

        # Construct prior Q(θ)
        theta_prior = self._theta_prior(norm_features)

        # Ambiguity: expected entropy of likelihood
        expected_entropy = self._expected_entropy(
            self._theta_grid, theta_prior["weights"], norm_features
        )
        ambiguity = self.policy_ambiguity_weight * expected_entropy

        # Risk: KL divergence from observation preferences
        risk = self._compute_risk(
            self._theta_grid, theta_prior["weights"], norm_features
        )

        # EFE scores: G = Ambiguity + Risk
        G = ambiguity + risk

        # Debug info
        debug_info = {
            "normalized_features": norm_features,
            "theta_prior": {
                "mean": theta_prior["mean"],
                "concentration": theta_prior["concentration"],
                "alpha": theta_prior["alpha"],
                "beta": theta_prior["beta"],
            },
            "expected_entropy": expected_entropy,
            "ambiguity": ambiguity,
            "risk": risk,
        }

        return G, debug_info


class MetaPlanningController:
    def __init__(self, agent, modes: Optional[List[Dict[str, Any]]], config: Dict[str, Any]):
        self.agent = agent
        self.enabled = bool(config.get("enabled", False))
        self.selection = str(config.get("selection", "greedy")).lower()
        if self.selection not in ("greedy", "softmax"):
            raise ValueError(f"Unknown meta selection '{self.selection}'. Expected greedy or softmax.")
        self.softmax_temp = float(config.get("softmax_temp", 1.0))
        self.allow_deterministic = bool(config.get("allow_deterministic", False))

        # Meta-EFE scores: G = Ambiguity + Risk
        # Ambiguity = policy_ambiguity_weight * E_Q[H(P(o|θ, mode))]
        # Risk = KL[q(o|mode) || p(o|C)] over observation space
        self.cost_log = bool(config.get("cost_log", True))
        self.policy_ambiguity_weight = float(config.get("policy_ambiguity_weight", 1.0))
        self.model_ambiguity_weight = float(config.get("model_ambiguity_weight", 1.0))

        # Observation preference precisions (higher = stronger preference)
        self.pref_extrinsic_precision = float(config.get("pref_extrinsic_precision", 1.0))
        self.pref_policy_uncert_precision = float(config.get("pref_policy_uncert_precision", 1.0))
        self.pref_model_uncert_precision = float(config.get("pref_model_uncert_precision", 1.0))
        self.pref_effort_precision = float(config.get("pref_effort_precision", 1.0))

        # EMA betas for feature tracking
        self.error_ema_beta = float(config.get("error_ema_beta", 0.9))
        self.uncert_ema_beta = float(config.get("uncert_ema_beta", 0.9))
        self.policy_uncert_ema_beta = float(config.get("policy_uncert_ema_beta", 0.9))
        self.extrinsic_value_ema_beta = float(config.get("extrinsic_value_ema_beta", 0.9))

        self.model_uncertainty_probe_actions = max(1, int(config.get("uncertainty_probe_actions", 3)))
        self.model_uncertainty_action_noise = float(config.get("uncertainty_action_noise", 0.1))
        self.policy_uncertainty_samples = max(1, int(config.get("policy_uncertainty_samples", 1)))

        self.horizon_min = _coerce_int(config.get("horizon_min"), default=1)
        self.candidates_min = _coerce_int(config.get("candidates_min"), default=1)
        self.mc_models_min = _coerce_int(config.get("mc_models_min"), default=1)
        self.mc_traj_min = _coerce_int(config.get("mc_trajectories_min"), default=1)
        self.action_rollouts_min = _coerce_int(config.get("action_rollouts_min"), default=1)

        self.horizon_max = _coerce_int(config.get("horizon_max"), default=self.horizon_min)
        self.candidates_max = _coerce_int(config.get("candidates_max"), default=self.candidates_min)
        self.mc_models_max = _coerce_int(config.get("mc_models_max"), default=self.mc_models_min)
        self.mc_traj_max = _coerce_int(config.get("mc_trajectories_max"), default=self.mc_traj_min)
        self.action_rollouts_max = _coerce_int(config.get("action_rollouts_max"), default=self.action_rollouts_min)

        self.horizon_step = _coerce_int(config.get("horizon_step"), default=1)
        self.candidates_step = _coerce_int(config.get("candidates_step"), default=1)
        self.mc_models_step = _coerce_int(config.get("mc_models_step"), default=1)
        self.mc_traj_step = _coerce_int(config.get("mc_trajectories_step"), default=1)
        self.action_rollouts_step = _coerce_int(config.get("action_rollouts_step"), default=1)
        self.max_modes = config.get("max_modes")

        self.mode_count = max(1, int(config.get("mode_count", 3)))
        self.modes = self._init_modes(modes)

        # Pre-compute mode parameter arrays for vectorized scoring
        self._precompute_mode_arrays()

        # Bayesian belief hyperparameters
        self.initial_belief_alpha = float(config.get("initial_belief_alpha", 1.0))
        self.initial_belief_beta = float(config.get("initial_belief_beta", 1.0))
        self.observation_weight = float(config.get("observation_weight", 1.0))
        self.belief_decay = float(config.get("belief_decay", 0.995))

        # Create explicit generative model for EFE computation
        self.generative_model = MetaGenerativeModel(
            policy_ambiguity_weight=self.policy_ambiguity_weight,
            model_ambiguity_weight=self.model_ambiguity_weight,
            total_capacity=self._total_capacity,
            depth_capacity=self._depth_capacity,
            complexity=self._complexity,
            habit_deviation=self._habit_deviation,
            is_deterministic=self._is_deterministic,
            initial_belief_alpha=self.initial_belief_alpha,
            initial_belief_beta=self.initial_belief_beta,
            observation_weight=self.observation_weight,
            belief_decay=self.belief_decay,
            pref_extrinsic_precision=self.pref_extrinsic_precision,
            pref_policy_uncert_precision=self.pref_policy_uncert_precision,
            pref_model_uncert_precision=self.pref_model_uncert_precision,
            pref_effort_precision=self.pref_effort_precision,
        )

        # Initialize EMAs with pessimistic values (assume untrained model)
        # model_error=2.0 → exp(-2) ≈ 0.135 reliability → favors reactive modes
        # model_uncert=1.0 → high uncertainty → exploration needed
        # extrinsic_value=0.0 → no improvement yet (neutral starting point)
        # These will quickly adapt once real data comes in
        self.model_error_ema: float = float(config.get("initial_model_error", 2.0))
        self.model_uncert_ema: float = float(config.get("initial_model_uncert", 1.0))
        self.policy_uncert_ema: float = float(config.get("initial_policy_uncert", 1.0))
        self.extrinsic_value_ema: float = float(config.get("initial_extrinsic_value", 0.0))

        # Tracking for extrinsic value improvement (computed between meta planning calls)
        self._meta_period_initial_obs: Optional[np.ndarray] = None
        self._meta_period_steps: int = 0

        # Track last selected mode for Bayesian update (observations were generated under this mode)
        self._last_mode_idx: int = 0  # Default to deterministic mode

        # Warmup: force reactive mode until enough transitions collected
        self._transition_count = 0
        self.warmup_transitions = int(config.get("warmup_transitions", 100))

        # Model update before meta: if True, trigger model training before each meta recompute
        self.update_model_before_meta = bool(config.get("update_model_before_meta", False))

        # Debug logging for diagnostics
        self.debug = bool(config.get("debug", False))
        self._last_debug_info: Optional[Dict[str, Any]] = None

    def _precompute_mode_arrays(self) -> None:
        """Pre-compute numpy arrays for vectorized score_modes."""
        n_modes = len(self.modes)
        self._H = np.zeros(n_modes, dtype=np.float32)
        self._N = np.zeros(n_modes, dtype=np.float32)
        self._M = np.zeros(n_modes, dtype=np.float32)
        self._T = np.zeros(n_modes, dtype=np.float32)
        self._R = np.zeros(n_modes, dtype=np.float32)

        for i, mode in enumerate(self.modes):
            self._H[i] = max(0, int(mode.get("horizon", 0)))
            self._N[i] = max(0, int(mode.get("candidates", 0)))
            self._M[i] = max(0, int(mode.get("mc_models", 0)))
            self._T[i] = max(0, int(mode.get("mc_trajectories", 0)))
            self._R[i] = max(1, int(mode.get("action_rollouts", 1)))

        # Pre-compute derived quantities that don't depend on features
        self._raw_cost = self._H * self._N * self._M * self._T * self._R
        if self.cost_log:
            self._complexity = np.log1p(self._raw_cost)
        else:
            self._complexity = self._raw_cost.copy()

        self._is_deterministic = np.array(
            [self._mode_is_deterministic(mode) for mode in self.modes], dtype=bool
        )
        det_indices = np.where(self._is_deterministic)[0]
        self._deterministic_idx = int(det_indices[0]) if det_indices.size > 0 else None

        self._depth_capacity = self._H
        self._breadth_capacity = np.log1p(self._N)
        self._ensemble_capacity = np.log1p(self._M) + np.log1p(self._T)
        self._rollout_capacity = np.log1p(self._R)
        H_max = max(self._H.max(), 1.0)
        N_max = max(self._N.max(), 1.0)
        M_max = max(self._M.max(), 1.0)
        T_max = max(self._T.max(), 1.0)
        R_max = max(self._R.max(), 1.0)
        eps = 1e-6
        # Balanced planning capacity: geometric mean of normalized knobs.
        components = np.stack(
            [
                self._H / H_max + eps,
                self._N / N_max + eps,
                self._M / M_max + eps,
                self._T / T_max + eps,
                self._R / R_max + eps,
            ],
            axis=1,
        )
        self._total_capacity = np.exp(np.mean(np.log(components), axis=1))
        self._max_compute_idx = int(np.argmax(self._complexity)) if n_modes > 0 else None

        # === HABITUAL DEVIATION: heuristic distance from habitual prior ===
        # Habitual prior = minimal computation (H=1, N=1, M=1, T=1, R=1)
        # Habitual deviation = squared deviation from habitual, normalized by max values
        # Normalized deviation from habitual prior (H=1, N=1, M=1, T=1, R=1)
        self._habit_deviation = (
            ((self._H - 1.0) / H_max) ** 2 +
            ((self._N - 1.0) / N_max) ** 2 +
            ((self._M - 1.0) / M_max) ** 2 +
            ((self._T - 1.0) / T_max) ** 2 +
            ((self._R - 1.0) / R_max) ** 2
        )
        # Deterministic mode has zero habitual deviation (it IS the habitual prior)
        self._habit_deviation = np.where(self._is_deterministic, 0.0, self._habit_deviation)

    def _init_modes(self, modes: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if modes:
            normalized = [self._normalize_mode(mode, idx) for idx, mode in enumerate(modes)]
            if self.allow_deterministic and not _has_deterministic_mode(normalized, self.horizon_min, self.candidates_min):
                normalized.insert(0, self._make_deterministic_mode())
            return normalized
        modes = self._build_grid_modes()
        if self.allow_deterministic:
            modes.insert(0, self._make_deterministic_mode())
        return modes

    def _build_grid_modes(self) -> List[Dict[str, Any]]:
        """Build aligned mode grid where parameters scale together.

        Instead of all combinations (exponential), creates aligned modes where
        low values combine with low values, high with high, etc.
        """
        h_vals = _range_inclusive(self.horizon_min, self.horizon_max, self.horizon_step)
        n_vals = _range_inclusive(self.candidates_min, self.candidates_max, self.candidates_step)
        m_vals = _range_inclusive(self.mc_models_min, self.mc_models_max, self.mc_models_step)
        t_vals = _range_inclusive(self.mc_traj_min, self.mc_traj_max, self.mc_traj_step)
        r_vals = _range_inclusive(self.action_rollouts_min, self.action_rollouts_max, self.action_rollouts_step)

        # Find the maximum number of steps across all parameters
        all_vals = [h_vals, n_vals, m_vals, t_vals, r_vals]
        max_steps = max(len(v) for v in all_vals)

        # Interpolate each parameter list to have max_steps entries
        def interpolate_to_length(vals: List[int], target_len: int) -> List[int]:
            if len(vals) == target_len:
                return vals
            if len(vals) == 1:
                return vals * target_len
            # Linear interpolation of indices
            result = []
            for i in range(target_len):
                # Map i in [0, target_len-1] to index in [0, len(vals)-1]
                idx_float = i * (len(vals) - 1) / (target_len - 1)
                idx = int(round(idx_float))
                result.append(vals[idx])
            return result

        h_aligned = interpolate_to_length(h_vals, max_steps)
        n_aligned = interpolate_to_length(n_vals, max_steps)
        m_aligned = interpolate_to_length(m_vals, max_steps)
        t_aligned = interpolate_to_length(t_vals, max_steps)
        r_aligned = interpolate_to_length(r_vals, max_steps)

        # Build aligned modes (one per step level)
        modes = []
        for i in range(max_steps):
            modes.append(
                {
                    "horizon": int(h_aligned[i]),
                    "candidates": int(n_aligned[i]),
                    "mc_models": int(m_aligned[i]),
                    "mc_trajectories": int(t_aligned[i]),
                    "action_rollouts": int(r_aligned[i]),
                }
            )
        return modes

    def _make_deterministic_mode(self) -> Dict[str, Any]:
        """Create deterministic mode: execute policy net mean without planning."""
        return {
            "horizon": 0,
            "candidates": 0,
            "mc_models": 0,
            "mc_trajectories": 0,
            "action_rollouts": 0,
            "deterministic": True,
        }

    def _normalize_mode(self, mode: Dict[str, Any], idx: int) -> Dict[str, Any]:
        if not isinstance(mode, dict):
            raise ValueError("Meta planning modes must be dicts with horizon/candidates/mc fields.")

        def _pick(keys, default):
            for key in keys:
                if key in mode and mode[key] is not None:
                    return mode[key]
            return default

        horizon = _pick(["horizon", "H", "aif_plan_horizon"], self.horizon_min)
        candidates = _pick(["candidates", "N", "aif_plan_candidates"], self.candidates_min)
        mc_models = _pick(["mc_models", "M", "aif_mc_models"], self.mc_models_min)
        mc_traj = _pick(["mc_trajectories", "T", "aif_mc_trajectories"], self.mc_traj_min)
        rollouts = _pick(["action_rollouts", "R", "aif_policy_plan_action_rollouts"], self.action_rollouts_min)
        name = mode.get("name")

        return {
            "name": name,
            "horizon": _coerce_int(horizon, default=self.horizon_min, allow_negative=True),
            "candidates": _coerce_int(candidates, default=self.candidates_min, allow_negative=True),
            "mc_models": _coerce_int(mc_models, default=self.mc_models_min, allow_negative=True),
            "mc_trajectories": _coerce_int(mc_traj, default=self.mc_traj_min, allow_negative=True),
            "action_rollouts": _coerce_int(rollouts, default=self.action_rollouts_min, allow_negative=True),
            "deterministic": bool(mode.get("deterministic", False)),
        }

    def _trigger_model_update(self) -> None:
        """Trigger world model training before meta recompute (if configured)."""
        if self.agent is None:
            return
        try:
            # Get replay buffer and batch size from agent (set by wrapper)
            replay_buffer = getattr(self.agent, "_meta_replay_buffer", None)
            batch_size = getattr(self.agent, "_meta_batch_size", 64)
            if replay_buffer is not None and len(replay_buffer) >= batch_size:
                # Call agent's update method with replay buffer
                if hasattr(self.agent, "update"):
                    self.agent.update(replay_buffer, batch_size=batch_size)
                elif hasattr(self.agent, "train_world_model_epochs"):
                    self.agent.train_world_model_epochs(replay_buffer, batch_size)
        except Exception:
            pass  # Silently fail if model update not available

    def update_from_transition(self, obs_np: np.ndarray, action_np: np.ndarray, next_obs_np: np.ndarray) -> None:
        """Update model-error from a single transition (obs, action, next_obs).
        Also tracks steps and initial observation for extrinsic value improvement computation.
        """
        if not self.enabled:
            return
        self._transition_count += 1

        # Track for extrinsic value improvement (initial obs → current obs per step)
        if obs_np is not None:
            if self._meta_period_initial_obs is None:
                self._meta_period_initial_obs = obs_np.copy()
            self._meta_period_steps += 1

        if self.agent is None:
            raise RuntimeError("MetaPlanningController.update_from_transition requires an agent.")
        if obs_np is None or action_np is None or next_obs_np is None:
            return
        try:
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.agent.device).unsqueeze(0)
            act_t = torch.as_tensor(action_np, dtype=torch.float32, device=self.agent.device).unsqueeze(0)
            next_obs_t = torch.as_tensor(next_obs_np, dtype=torch.float32, device=self.agent.device).unsqueeze(0)
        except Exception:
            return

        error_val = None
        with torch.no_grad():
            if self.agent.fully_observable_mdp:
                mu_next, logvar_next = self.agent.dynamics(obs_t, act_t, sample_theta=False, eps_cache=None)
                diff = next_obs_t - mu_next
                # Use NLL for meta error (Gaussian negative log-likelihood)
                var_next = torch.exp(logvar_next)
                error_val = 0.5 * (logvar_next + diff.pow(2) / var_next).mean().item()
            else:
                mu, _ = self.agent.encoder(obs_t)
                s = mu
                mu_next, _ = self.agent.dynamics(s, act_t, sample_theta=False, eps_cache=None)
                s_pred = mu_next
                mu_o, logvar_o = self.agent.decoder(s_pred)
                diff = next_obs_t - mu_o
                # Use NLL for meta error (Gaussian negative log-likelihood)
                var_o = torch.exp(logvar_o)
                error_val = 0.5 * (logvar_o + diff.pow(2) / var_o).mean().item()
            # NOTE: Task-level risk (preference violation) is no longer used.
            # Meta-level risk uses per-mode compute cost/habit preferences from _precompute_mode_arrays.

        if error_val is not None:
            self.model_error_ema = _update_ema(self.model_error_ema, error_val, self.error_ema_beta)


    def compute_features(self, obs_np: np.ndarray, update_policy_uncert_ema: bool = True) -> Dict[str, Any]:
        """
        Compute features for mode selection with lazy evaluation.

        Only computes features whose corresponding weights are non-zero,
        saving computation when some EFE terms are disabled.
        """
        if not self.enabled:
            return {}
        if self.agent is None:
            raise RuntimeError("MetaPlanningController.compute_features requires an agent.")

        # Lazy evaluation: only compute features that affect scoring
        # Policy/model uncertainties are used when ambiguity depends on them
        needs_policy_uncert = self.policy_ambiguity_weight != 0.0
        needs_model_uncert = self.model_ambiguity_weight != 0.0
        if self.model_ambiguity_weight == 0.0:
            needs_model_uncert = False

        extrinsic_value_improvement = None
        with torch.no_grad():
            # Only infer state if we need uncertainty estimates
            if needs_policy_uncert or needs_model_uncert:
                s = self.agent._infer_latent_mean(obs_np)
            else:
                s = None

            # Policy uncertainty (needed for ambiguity via Q(θ))
            # Update EMA only when requested (e.g., non-deterministic mode).
            policy_uncert = None
            if needs_policy_uncert and s is not None:
                policy_uncert = self._estimate_policy_uncertainty(s)
                if policy_uncert is not None and update_policy_uncert_ema:
                    self.policy_uncert_ema = _update_ema(
                        self.policy_uncert_ema, policy_uncert, self.policy_uncert_ema_beta
                    )

            # Model uncertainty (needed to shape likelihood sharpness)
            if needs_model_uncert and s is not None:
                model_uncert = self._estimate_model_uncertainty(obs_np, s)
                if model_uncert is None:
                    model_uncert = self._estimate_random_model_uncertainty(s)
                if model_uncert is not None:
                    self.model_uncert_ema = _update_ema(
                        self.model_uncert_ema, model_uncert, self.uncert_ema_beta
                    )

            # Extrinsic value improvement: measures policy success as reduction in
            # divergence from preferred state, normalized by number of steps.
            # Positive values indicate improvement toward goals.
            if self._meta_period_initial_obs is not None and self._meta_period_steps > 0:
                try:
                    initial_obs_t = torch.as_tensor(
                        self._meta_period_initial_obs, dtype=torch.float32, device=self.agent.device
                    ).unsqueeze(0)
                    current_obs_t = torch.as_tensor(
                        obs_np, dtype=torch.float32, device=self.agent.device
                    ).unsqueeze(0)

                    # neg_extrinsic is higher when further from preferences (worse)
                    initial_neg_extrinsic = self.agent._compute_neg_extrinsic_value(initial_obs_t).mean().item()
                    current_neg_extrinsic = self.agent._compute_neg_extrinsic_value(current_obs_t).mean().item()

                    # Improvement = reduction in neg_extrinsic (initial - current), per step
                    # Positive means we moved closer to preferred state
                    extrinsic_value_improvement = (initial_neg_extrinsic - current_neg_extrinsic) / initial_neg_extrinsic
                    self.extrinsic_value_ema = _update_ema(
                        self.extrinsic_value_ema, extrinsic_value_improvement, self.extrinsic_value_ema_beta
                    )
                    print(f"[META][extrinsic] mid-episode: initial_neg={initial_neg_extrinsic:.4f} "
                          f"current_neg={current_neg_extrinsic:.4f} steps={self._meta_period_steps} "
                          f"improvement={extrinsic_value_improvement:.6f} ema={self.extrinsic_value_ema:.6f}")
                except Exception:
                    pass  # Keep previous EMA if computation fails

            # Reset tracking for next meta planning period
            self._meta_period_initial_obs = obs_np.copy() if obs_np is not None else None
            self._meta_period_steps = 0

        # Use EMA for scoring; keep the raw measurement separate for optional updates/logging.
        policy_uncert_value = float(self.policy_uncert_ema)

        return {
            "model_error": float(self.model_error_ema),
            "model_uncert": float(self.model_uncert_ema),
            "policy_uncert": float(policy_uncert_value),
            "policy_uncert_ema": float(self.policy_uncert_ema),
            "policy_uncert_measurement": float(policy_uncert) if policy_uncert is not None else None,
            "extrinsic_value": float(self.extrinsic_value_ema),
            "extrinsic_value_measurement": float(extrinsic_value_improvement) if extrinsic_value_improvement is not None else None,
            # Compute cost/habit preferences are
            # pre-computed per mode in _precompute_mode_arrays and used in meta-risk scoring.
        }

    def select_mode(self, obs_np: np.ndarray) -> Tuple[Dict[str, int], Dict[str, Any]]:
        if not self.enabled:
            return {}, {}

        # Trigger model update before meta recompute if configured
        if self.update_model_before_meta and self._transition_count >= self.warmup_transitions:
            self._trigger_model_update()

        # Always compute features (policy EMA updates are gated by selected mode)
        features = self.compute_features(obs_np, update_policy_uncert_ema=False)

        # Bayesian update of system quality belief based on observations
        # Use the last selected mode since observations were generated under that mode
        normalized_features = self.generative_model.normalize_features(features)
        belief_update = self.generative_model.bayesian_update(
            normalized_features, mode_idx=self._last_mode_idx
        )

        policy_uncert_ema = features.get("policy_uncert_ema")
        if policy_uncert_ema is None:
            policy_uncert_ema = features.get("policy_uncert", 0.0)

        # Force reactive mode during warmup (EMAs still accumulating)
        if self._transition_count < self.warmup_transitions:
            mode = self._make_deterministic_mode()
            override = self._mode_to_override(mode, deterministic=True)
            info = {
                "meta_mode_idx": -1,
                "meta_warmup": True,
                "meta_warmup_progress": self._transition_count / max(1, self.warmup_transitions),
                "meta_model_error_ema": features.get("model_error", 0.0),
                "meta_model_uncert_ema": features.get("model_uncert", 0.0),
                "meta_policy_uncert_ema": policy_uncert_ema,
                "meta_risk": 0.0,
                "meta_compute_cost_selected": 0.0,
                "meta_G_selected": 0.0,
                "meta_deterministic": True,
                "meta_plan_horizon": 0,
                "meta_plan_candidates": 0,
                "meta_mc_models": 0,
                "meta_mc_trajectories": 0,
                "meta_action_rollouts": 0,
            }
            return override, info

        # Normal selection after warmup (EMAs are now warmed up)
        idx, mode, scores, costs = self.select_mode_from_features(features)
        deterministic = self._mode_is_deterministic(mode)
        override = self._mode_to_override(mode, deterministic=deterministic)
        debug_info = self._last_debug_info or {}
        risk_values = debug_info.get("risk")
        meta_risk = float(risk_values[idx]) if risk_values is not None else float(self._habit_deviation[idx])

        policy_uncert_measurement = features.get("policy_uncert_measurement")
        if not deterministic and policy_uncert_measurement is not None:
            self.policy_uncert_ema = _update_ema(
                self.policy_uncert_ema, policy_uncert_measurement, self.policy_uncert_ema_beta
            )
            policy_uncert_ema = float(self.policy_uncert_ema)

        # Log features, belief state, and selected planning params
        print(f"[META] features: model_error={features.get('model_error', 0.0):.4f} "
              f"model_uncert={features.get('model_uncert', 0.0):.4f} "
              f"policy_uncert={policy_uncert_ema:.4f} "
              f"extrinsic_value={features.get('extrinsic_value', 0.0):.4f} | "
              f"belief: mean={belief_update.get('belief_mean', 0.5):.3f} "
              f"conc={belief_update.get('belief_concentration', 2.0):.1f} | "
              f"selected: H={mode.get('horizon')} N={mode.get('candidates')} "
              f"M={mode.get('mc_models')} T={mode.get('mc_trajectories')} R={mode.get('action_rollouts')}")

        info = {
            "meta_mode_idx": int(idx),
            "meta_warmup": False,
            "meta_model_error_ema": features.get("model_error", 0.0),
            "meta_model_uncert_ema": features.get("model_uncert", 0.0),
            "meta_policy_uncert_ema": policy_uncert_ema,
            "meta_extrinsic_value": features.get("extrinsic_value", 0.0),
            "meta_belief_mean": belief_update.get("belief_mean", 0.5),
            "meta_belief_concentration": belief_update.get("belief_concentration", 2.0),
            "meta_risk": meta_risk,
            "meta_compute_cost_selected": float(costs[idx]),
            "meta_G_selected": float(scores[idx]),
            "meta_deterministic": bool(deterministic),
            "meta_plan_horizon": int(mode.get("horizon", 0)),
            "meta_plan_candidates": int(mode.get("candidates", 0)),
            "meta_mc_models": int(mode.get("mc_models", 0)),
            "meta_mc_trajectories": int(mode.get("mc_trajectories", 0)),
            "meta_action_rollouts": int(mode.get("action_rollouts", 0)),
        }
        if mode.get("name") is not None:
            info["meta_mode_name"] = str(mode.get("name"))

        # Track selected mode for next Bayesian update
        self._last_mode_idx = idx

        return override, info

    def select_mode_from_features(self, features: Dict[str, Any]) -> Tuple[int, Dict[str, Any], List[float], List[float]]:
        scores, costs = self.score_modes(features)
        if not scores:
            raise RuntimeError("MetaPlanningController has no modes to select from.")
        if self.selection == "greedy":
            idx = int(np.argmin(scores))
        else:
            temp = max(1e-6, float(self.softmax_temp))
            scores_arr = np.asarray(scores, dtype=np.float64)
            shifted = scores_arr - np.min(scores_arr)
            logits = -shifted / temp
            logits = logits - np.max(logits)
            probs = np.exp(logits)
            probs = probs / (np.sum(probs) + 1e-12)
            idx = int(np.random.choice(len(scores), p=probs))
        return idx, self.modes[idx], scores, costs

    def score_modes(self, features: Dict[str, Any]) -> Tuple[List[float], List[float]]:
        """
        Meta-EFE for planning mode selection using explicit generative model.

        Delegates to MetaGenerativeModel.expected_free_energy() which computes:
          G(mode) = Ambiguity + Risk

        where:
          - Ambiguity = E_Q(θ)[H(P(o|θ, mode))]
          - Risk = KL[q(o|π) || p(o|C)] (cost/habit preferences)

        Key dynamics:
          - DETERMINISTIC mode: lower risk but higher ambiguity when policy quality is uncertain.
          - DELIBERATIVE mode: lower ambiguity (sharper likelihood) but higher risk.

        Expected behavior:
          - Early (uncertain): deliberative may win if ambiguity reduction outweighs risk.
          - Late (confident): deterministic wins as ambiguity is low and risk dominates.
        """
        # Use explicit generative model for EFE computation
        scores, debug_info = self.generative_model.expected_free_energy(features)
        self._last_debug_info = debug_info

        return scores.tolist(), self._complexity.tolist()

    @property
    def max_complexity(self) -> float:
        """Maximum complexity across all modes (for normalizing compute weight)."""
        return float(np.max(self._complexity)) if len(self._complexity) > 0 else 1.0

    def _mode_is_deterministic(self, mode: Dict[str, Any]) -> bool:
        if mode.get("deterministic", False):
            return True
        if not self.allow_deterministic:
            return False
        horizon = int(mode.get("horizon", 0))
        candidates = int(mode.get("candidates", 0))
        mc_models = int(mode.get("mc_models", 0))
        mc_traj = int(mode.get("mc_trajectories", 0))
        rollouts = int(mode.get("action_rollouts", 0))
        if horizon <= 0 or candidates <= 0 or mc_models <= 0 or mc_traj <= 0 or rollouts <= 0:
            return True
        if horizon < self.horizon_min or candidates < self.candidates_min:
            return True
        if mc_models < self.mc_models_min or mc_traj < self.mc_traj_min:
            return True
        if rollouts < self.action_rollouts_min:
            return True
        return False

    def _mode_to_override(self, mode: Dict[str, Any], deterministic: bool = False) -> Dict[str, int]:
        """
        Convert mode dict to planning hyperparameter overrides.

        Deterministic mode: single-step greedy action (reactive policy).
        Otherwise: use mode's specified planning capacity.
        """
        if deterministic:
            # Reactive/habitual action: execute policy net mean, no planning
            return {
                "aif_plan_horizon": 0,
                "aif_plan_candidates": 0,
                "aif_mc_models": 0,
                "aif_mc_trajectories": 0,
                "aif_policy_plan_action_rollouts": 0,
            }

        # Deliberative planning with specified capacity
        return {
            "aif_plan_horizon": max(1, int(mode.get("horizon", self.horizon_min))),
            "aif_plan_candidates": max(1, int(mode.get("candidates", self.candidates_min))),
            "aif_mc_models": max(1, int(mode.get("mc_models", self.mc_models_min))),
            "aif_mc_trajectories": max(1, int(mode.get("mc_trajectories", self.mc_traj_min))),
            "aif_policy_plan_action_rollouts": max(1, int(mode.get("action_rollouts", self.action_rollouts_min))),
        }

    def _estimate_policy_uncertainty(self, state_t: torch.Tensor) -> Optional[float]:
        """
        Estimate policy epistemic uncertainty using Bayesian weight sampling.

        Returns ONLY the epistemic uncertainty (variance of means across weight samples),
        NOT the aleatoric uncertainty (output std). This ensures policy_uncert decreases
        as the Bayesian weights converge during training, enabling the reactive →
        deliberative → habitual trajectory.
        """
        if state_t is None:
            return None
        policy = self.agent.policy_net
        state_t = state_t.unsqueeze(0)
        if self.policy_uncertainty_samples <= 1:
            # Fallback: sample a few times even if config says 1
            # This is needed because fixed std doesn't reflect learning progress
            n_samples = 5
        else:
            n_samples = self.policy_uncertainty_samples

        means = []
        for _ in range(n_samples):
            eps = policy.sample_eps()
            mean, _ = policy(state_t, sample=True, eps_cache=eps)
            means.append(mean.squeeze(0))

        means_t = torch.stack(means, dim=0)
        # Epistemic uncertainty: variance of means across weight samples
        # This decreases as Bayesian weights converge during training
        var_means = means_t.var(dim=0, unbiased=False)
        return float(var_means.mean().item())

    def _estimate_model_uncertainty(self, obs_np: np.ndarray, state_t: torch.Tensor) -> Optional[float]:
        """Estimate model uncertainty using batched forward pass."""
        if state_t is None:
            return None
        action_batch = self._probe_actions_batched(obs_np, state_t)
        if action_batch is None:
            return None
        return self._uncertainty_for_action_batch(state_t, action_batch)

    def _estimate_random_model_uncertainty(self, state_t: torch.Tensor) -> Optional[float]:
        """Estimate uncertainty with a random action (fallback when policy unavailable)."""
        if state_t is None:
            return None
        try:
            action_low = self.agent.action_low
            action_high = self.agent.action_high
            rand = torch.rand_like(action_low)
            action_t = action_low + rand * (action_high - action_low)
            # Use batched method with single action
            return self._uncertainty_for_action_batch(state_t, action_t.unsqueeze(0))
        except Exception:
            return None

    def _probe_actions_batched(self, obs_np: np.ndarray, state_t: torch.Tensor) -> Optional[torch.Tensor]:
        """Generate probe actions as a single batched tensor [N, action_dim]."""
        try:
            mean_action = self.agent.policy_net_mean_action(obs_np)  # [action_dim]
            n_probes = self.model_uncertainty_probe_actions
            if n_probes <= 1:
                return mean_action.unsqueeze(0)  # [1, action_dim]

            # Generate all noisy actions at once
            noise_scale = self.model_uncertainty_action_noise
            noise = torch.randn(n_probes - 1, mean_action.shape[0], device=mean_action.device) * noise_scale
            noisy_actions = mean_action.unsqueeze(0) + noise  # [n_probes-1, action_dim]
            noisy_actions = torch.clamp(noisy_actions, self.agent.action_low, self.agent.action_high)

            # Stack mean action with noisy actions
            return torch.cat([mean_action.unsqueeze(0), noisy_actions], dim=0)  # [n_probes, action_dim]
        except Exception:
            return None

    def _uncertainty_for_action_batch(self, state_t: torch.Tensor, action_batch: torch.Tensor) -> Optional[float]:
        """
        Estimate predictive uncertainty for batched state-action pairs.

        Args:
            state_t: State tensor [state_dim]
            action_batch: Batched actions [N, action_dim]

        Returns:
            Mean uncertainty across all actions, or None if estimation fails.
        """
        n_actions = action_batch.shape[0]
        # Expand state to match action batch: [N, state_dim]
        s_batch = state_t.unsqueeze(0).expand(n_actions, -1)

        with torch.no_grad():
            if hasattr(self.agent, "_theta_mu_logvar_stack"):
                mu_stack, logvar_stack, _ = self.agent._theta_mu_logvar_stack(
                    s_batch, action_batch, num_models=max(1, int(self.agent.mc_num_models or 1))
                )

                if mu_stack is not None:
                    # mu_stack: [n_models, N, state_dim]
                    # Epistemic: variance across ensemble, averaged over actions and dims
                    epistemic = mu_stack.var(dim=0, unbiased=False).mean().item()

                    # Aleatoric: mean predicted variance
                    if logvar_stack is not None:
                        aleatoric = torch.exp(logvar_stack).mean().item()
                    else:
                        aleatoric = 0.0

                    return float(epistemic + aleatoric)

        return None


def _coerce_int(value: Any, default: int, allow_negative: bool = False) -> int:
    if value is None:
        return int(default)
    try:
        val = int(value)
    except Exception:
        return int(default)
    if allow_negative:
        return val
    return max(int(default), val)


def _range_inclusive(min_val: int, max_val: int, step: int) -> List[int]:
    min_val = int(min_val)
    max_val = int(max_val)
    step = max(1, int(step))
    if max_val < min_val:
        max_val = min_val
    return list(range(min_val, max_val + 1, step))


def _has_deterministic_mode(modes: List[Dict[str, Any]], h_min: int, n_min: int) -> bool:
    for mode in modes:
        if mode.get("deterministic", False):
            return True
        horizon = int(mode.get("horizon", 0))
        candidates = int(mode.get("candidates", 0))
        if horizon <= 0 or candidates <= 0 or horizon < h_min or candidates < n_min:
            return True
    return False


def _update_ema(old_val: Optional[float], new_val: float, beta: float) -> float:
    if old_val is None:
        return float(new_val)
    return float(beta * old_val + (1.0 - beta) * float(new_val))
