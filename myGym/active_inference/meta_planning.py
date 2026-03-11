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
    extrinsic_value: float = 0.0  # Policy success: mean per-step distance reduction toward target / max possible (positive = improving)

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
    Explicit specification of a meta-level generative model with point-estimate
    policy quality.

    Policy quality (composite point estimate):
      - θ_π = 1 - normalized_policy_uncert (policy uncertainty signal)
      - θ_v = (1 + normalized_extrinsic_value) / 2 (performance signal)
      - ρ = 1 - min(1, λ · normalized_model_uncert) (model reliability gate)
      - θ = θ_π + w_v · ρ · (θ_v - θ_π)  (interpolation)
      - High policy_uncert → low θ → favor deliberative modes
      - Low policy_uncert → high θ → favor deterministic mode
      - Good performance + reliable model pulls θ up (earlier habituation)
      - Bad performance + reliable model pulls θ down (triggers deliberation)

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
      - Ambiguity = H(P(o|θ, mode)) - uncertainty about outcomes at point θ
      - Risk = KL[P(o|θ, mode) || p(o|C)] - divergence from preferred observations

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
        pref_extrinsic_precision: float = 1.0,
        pref_policy_uncert_precision: float = 1.0,
        pref_model_uncert_precision: float = 1.0,
        pref_effort_precision: float = 1.0,
        pref_extrinsic_mean: float = 1.0,
        pref_policy_uncert_mean: float = 0.0,
        pref_model_uncert_mean: float = 0.0,
        pref_effort_mean: float = 0.0,
        capacity_exponent: float = 1.0,
        variance_exponent: float = 2.65,
        extrinsic_value_weight: float = 0.3,
        risk_include_variance: bool = False,
    ):
        """
        Initialize the meta generative model with pre-computed mode arrays.

        Args:
            policy_ambiguity_weight: Weight on ambiguity term H(P(o|θ, mode))
            model_ambiguity_weight: Scales model uncertainty in likelihood sharpness
            total_capacity: Planning capacity per mode [n_modes]
            depth_capacity: Horizon depth per mode [n_modes]
            complexity: Computational complexity per mode [n_modes]
            habit_deviation: Deviation from habitual prior per mode [n_modes]
            is_deterministic: Boolean mask for deterministic modes [n_modes]
            pref_extrinsic_precision: Precision for extrinsic_value preference
            pref_policy_uncert_precision: Precision for policy_uncert preference
            pref_model_uncert_precision: Precision for model_uncert preference
            pref_effort_precision: Precision for effort preference
            pref_extrinsic_mean: Preferred mean for extrinsic_value (1.0 = prefer high)
            pref_policy_uncert_mean: Preferred mean for policy_uncert (0.0 = prefer low)
            pref_model_uncert_mean: Preferred mean for model_uncert (0.0 = prefer low)
            pref_effort_mean: Preferred mean for effort (0.0 = prefer low)
            capacity_exponent: Exponent applied to normalized capacity (<1 compresses differences, >1 amplifies)
        """
        self.policy_ambiguity_weight = policy_ambiguity_weight
        self.model_ambiguity_weight = model_ambiguity_weight

        # Observation preference precisions (higher = stronger preference)
        self.pref_extrinsic_precision = pref_extrinsic_precision
        self.pref_policy_uncert_precision = pref_policy_uncert_precision
        self.pref_model_uncert_precision = pref_model_uncert_precision
        self.pref_effort_precision = pref_effort_precision

        # Observation preference means
        self.pref_extrinsic_mean = pref_extrinsic_mean
        self.pref_policy_uncert_mean = pref_policy_uncert_mean
        self.pref_model_uncert_mean = pref_model_uncert_mean
        self.pref_effort_mean = pref_effort_mean

        # Capacity exponent (<1 compresses differences between modes, >1 amplifies)
        self.capacity_exponent = capacity_exponent
        self.variance_exponent = variance_exponent

        # Weight on extrinsic value signal in composite θ estimate
        self.extrinsic_value_weight = max(0.0, min(1.0, float(extrinsic_value_weight)))

        # If True, include τ·σ² variance term in risk (full KL).
        # Gives independent control over variance preference via precisions (τ),
        # separate from the ambiguity term which uses policy_ambiguity_weight.
        self.risk_include_variance = bool(risk_include_variance)

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
        # Extrinsic value already tanh-normalized before EMA, clamp for safety
        normalized_extrinsic_value = max(-1.0, min(1.0, raw_extrinsic_value))
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

    def _likelihood(
        self, theta: float, normalized_features: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute likelihood P(o | θ, mode) for a single scalar θ and all modes.

        Generative model: observations given policy quality θ and planning mode.

        P(o | θ, mode) is a multivariate Gaussian over:
          - extrinsic_value | θ, mode ~ N(μ_ev(θ, mode), σ²_ev(θ, mode))
          - policy_uncert | θ ~ N(1 - θ, σ²_pu)
          - model_uncert | θ ~ N(1 - θ, σ²_mu)

        Mode affects extrinsic value prediction:
          - Higher capacity modes leverage good θ more effectively
          - Planning amplifies θ when model is reliable

        Mode affects precision of extrinsic value:
          - Higher capacity + good θ → more predictable outcomes

        Key insight: Planning helps when model is GOOD and θ is BAD.

        Model reliability (ρ) is used here as a STRUCTURAL property — "does
        planning actually work with this model?" — distinct from its
        epistemological role in θ ("should I trust performance evidence?").

        Args:
            theta: Scalar policy quality in [0, 1]
            normalized_features: Dict with normalized feature values

        Returns:
            Tuple of (expected_observations, variances) each [n_modes, 3]
            where 3 = (extrinsic_value, policy_uncert, model_uncert)
        """
        # Model reliability: structural property of whether planning is effective.
        model_uncert = float(normalized_features["normalized_model_uncert"])
        model_reliability = 1.0 - min(1.0, model_uncert * self.model_ambiguity_weight)

        capacity_raw = np.clip(self._total_capacity / self._max_capacity, 0.0, 1.0)
        capacity = np.power(capacity_raw, self.capacity_exponent)
        n_modes = len(self._total_capacity)

        # Room to improve: derived ONLY from θ (which already incorporates
        # extrinsic value via the composite estimate). No extrinsic_gap.
        room_to_improve = max(0.0, 1.0 - theta)

        # === EXPECTED EXTRINSIC VALUE ===
        base_ev = 2.0 * theta - 1.0
        planning_boost = capacity * room_to_improve * model_reliability  # [n_modes]
        mu_ev = base_ev + planning_boost  # no upper clip

        # === VARIANCE OF EXTRINSIC VALUE ===
        # Deterministic mode: variance from policy quality
        # When theta is high → low variance (good policy = predictable)
        # When theta is low → high variance (bad policy = unpredictable)
        var_deterministic = 0.01 + room_to_improve**self.variance_exponent  # scalar

        # Deliberate mode: planning can reduce variance when model is good
        model_unreliability = 1.0 - model_reliability
        planning_variance_reduction = room_to_improve**2 * model_reliability * 0.8
        planning_noise = model_unreliability * 1.0

        var_deliberate = var_deterministic - planning_variance_reduction + planning_noise
        var_deliberate = max(var_deliberate, 0.01)

        # Interpolate based on capacity
        var_ev = var_deterministic * (1.0 - capacity) + var_deliberate * capacity  # [n_modes]
        var_ev = np.maximum(var_ev, 0.01)

        # === POLICY/MODEL UNCERTAINTY (mode-independent) ===
        mu_pu = 1.0 - theta  # scalar
        mu_mu = 1.0 - theta  # scalar
        var_pu = np.full(n_modes, 0.25)
        var_mu = np.full(n_modes, 0.25)

        # Stack into [n_modes, 3]
        expected = np.stack([
            mu_ev,
            np.full(n_modes, mu_pu),
            np.full(n_modes, mu_mu),
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
            "extrinsic_value": {"mean": self.pref_extrinsic_mean, "precision": self.pref_extrinsic_precision},
            "policy_uncert": {"mean": self.pref_policy_uncert_mean, "precision": self.pref_policy_uncert_precision},
            "model_uncert": {"mean": self.pref_model_uncert_mean, "precision": self.pref_model_uncert_precision},
            "effort": {"mean": self.pref_effort_mean, "precision": self.pref_effort_precision},
        }

    def _expected_observations(
        self, theta: float, normalized_features: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute P(o|θ, mode) for each mode at a single θ point.

        Returns:
            expected_obs: [n_modes, 4] - expected observation per mode
            obs_variance: [n_modes, 4] - variance of observations per mode
        """
        expected, variances = self._likelihood(theta, normalized_features)
        # expected: [n_modes, 3], variances: [n_modes, 3]

        # Add effort as 4th observation dimension
        normalized_cost = self._complexity / (self._max_complexity + 1e-6)  # [n_modes]

        expected_obs = np.concatenate([
            expected,
            normalized_cost[:, None]  # [n_modes, 1]
        ], axis=-1)  # [n_modes, 4]

        obs_variance = np.concatenate([
            variances,
            np.full((len(self._complexity), 1), 1e-6)
        ], axis=-1)  # [n_modes, 4]

        return expected_obs, obs_variance

    def _compute_risk(
        self, theta: float, normalized_features: Dict[str, float]
    ) -> np.ndarray:
        """
        Compute Risk for each mode.

        When risk_include_variance is False (default):
            Risk = Σ_d τ_d · (μ_d - μ_pref_d)²
            Uses only the mean-deviation term (limit of KL as preference
            variance → ∞). Simple and avoids overlap with ambiguity.

        When risk_include_variance is True (full KL):
            Risk = Σ_d τ_d · [(μ_d - μ_pref_d)² + σ²_d]
            Adds the τ_d · σ²_d variance term. This gives independent control
            over variance preference via τ_d, separate from the ambiguity term
            which uses policy_ambiguity_weight. For extrinsic value, deliberative
            modes have lower σ² than deterministic (when model is reliable), so
            this term provides additional preference for deliberation.

        Extrinsic value uses one-sided risk: only penalizes when predicted value
        is below the preference mean. Exceeding the preference incurs zero risk.

        Returns:
            risk: [n_modes] - weighted risk from preferences
        """
        prefs = self._observation_preferences()
        expected_obs, obs_variance = self._expected_observations(
            theta, normalized_features
        )
        # expected_obs: [n_modes, 4], obs_variance: [n_modes, 4]

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

        mu_diff = expected_obs - pref_means[None, :]  # [n_modes, 4]
        # One-sided extrinsic risk: only penalize when predicted value is BELOW
        # the preference mean (undershooting). Exceeding the preference is fine.
        mu_diff[:, 0] = np.minimum(mu_diff[:, 0], 0.0)
        mu_diff_sq = mu_diff ** 2  # [n_modes, 4]
        risk = np.sum(pref_precisions[None, :] * mu_diff_sq, axis=-1)  # [n_modes]

        if self.risk_include_variance:
            # Full KL: add τ_d · σ²_d for each observation dimension
            variance_risk = np.sum(pref_precisions[None, :] * obs_variance, axis=-1)  # [n_modes]
            risk = risk + variance_risk

        return risk

    def _compute_entropy(
        self, theta: float, normalized_features: Dict[str, float]
    ) -> np.ndarray:
        """
        Compute H(P(o | θ, mode)) for all modes at a single θ point.

        Only extrinsic_value variance is mode-conditional.
        We focus the ambiguity calculation on this dimension to avoid
        diluting the signal with mode-independent observations.
        """
        _, variances = self._likelihood(theta, normalized_features)
        # variances: [n_modes, 3]

        var_ev = variances[:, 0]  # [n_modes]

        # Differential entropy of Gaussian: H = 0.5 * log(2πeσ²)
        log_var_ev = np.log(var_ev + 1e-12)
        entropy = 0.5 * (1.0 + np.log(2 * np.pi) + log_var_ev)

        # Normalize to [0, 1]
        entropy_min = 0.5 * (1.0 + np.log(2 * np.pi) + np.log(0.01))
        entropy_max = 0.5 * (1.0 + np.log(2 * np.pi) + np.log(2.0))
        entropy = (entropy - entropy_min) / (entropy_max - entropy_min + 1e-6)
        entropy = np.clip(entropy, 0.0, 1.0)

        return entropy  # [n_modes]

    def expected_free_energy(self, features: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute Expected Free Energy G(mode) for all modes.

        G = Ambiguity + Risk

        where:
          - Ambiguity = H(P(o|θ, mode)) at point θ = 1 - normalized_policy_uncert
          - Risk = KL[P(o|θ, mode) || p(o|C)] (divergence from preferred observations)

        Args:
            features: Raw features dict with model_error, model_uncert, policy_uncert

        Returns:
            Tuple of (G scores [n_modes], debug_info dict)
        """
        # Normalize features
        norm_features = self.normalize_features(features)

        # Policy-uncertainty-based estimate (existing signal)
        theta_pi = 1.0 - float(norm_features["normalized_policy_uncert"])
        theta_pi = max(1e-6, min(1.0 - 1e-6, theta_pi))

        # Extrinsic-value-based estimate: map (-1, 1) → (0, 1)
        v_ext = float(norm_features["normalized_extrinsic_value"])
        theta_v = (1.0 + v_ext) / 2.0
        theta_v = max(1e-6, min(1.0 - 1e-6, theta_v))

        # Model reliability: gate on extrinsic value signal
        # When model is unreliable, we can't trust extrinsic value as evidence
        # of policy quality (poor performance might be the model's fault)
        model_uncert = float(norm_features["normalized_model_uncert"])
        rho = 1.0 - min(1.0, model_uncert * self.model_ambiguity_weight)

        # Composite estimate: interpolate from θ_π toward θ_v,
        # weighted by extrinsic value weight and model reliability
        w_effective = self.extrinsic_value_weight * rho
        theta = theta_pi + w_effective * (theta_v - theta_pi)
        theta = max(1e-6, min(1.0 - 1e-6, theta))

        # Ambiguity: entropy of likelihood at point θ
        entropy = self._compute_entropy(theta, norm_features)
        ambiguity = self.policy_ambiguity_weight * entropy

        # Risk: KL divergence from observation preferences at point θ
        risk = self._compute_risk(theta, norm_features)

        # EFE scores: G = Ambiguity + Risk
        G = ambiguity + risk

        # Debug info
        debug_info = {
            "normalized_features": norm_features,
            "theta": theta,
            "theta_pi": theta_pi,
            "theta_v": theta_v,
            "model_reliability": rho,
            "w_effective": w_effective,
            "entropy": entropy,
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

        # Observation preference means
        self.pref_extrinsic_mean = float(config.get("pref_extrinsic_mean", 1.0))
        self.pref_policy_uncert_mean = float(config.get("pref_policy_uncert_mean", 0.0))
        self.pref_model_uncert_mean = float(config.get("pref_model_uncert_mean", 0.0))
        self.pref_effort_mean = float(config.get("pref_effort_mean", 0.0))

        # Capacity exponent
        self.capacity_exponent = float(config.get("capacity_exponent", 1.0))
        self.variance_exponent = float(config.get("variance_exponent", 2.65))

        # Weight on extrinsic value signal in composite θ estimate
        self.extrinsic_value_weight = float(config.get("extrinsic_value_weight", 0.3))

        # Include τ·σ² variance term in risk (full KL)
        self.risk_include_variance = bool(config.get("risk_include_variance", False))

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

        # Create explicit generative model for EFE computation
        self.generative_model = MetaGenerativeModel(
            policy_ambiguity_weight=self.policy_ambiguity_weight,
            model_ambiguity_weight=self.model_ambiguity_weight,
            total_capacity=self._total_capacity,
            depth_capacity=self._depth_capacity,
            complexity=self._complexity,
            habit_deviation=self._habit_deviation,
            is_deterministic=self._is_deterministic,
            pref_extrinsic_precision=self.pref_extrinsic_precision,
            pref_policy_uncert_precision=self.pref_policy_uncert_precision,
            pref_model_uncert_precision=self.pref_model_uncert_precision,
            pref_effort_precision=self.pref_effort_precision,
            pref_extrinsic_mean=self.pref_extrinsic_mean,
            pref_policy_uncert_mean=self.pref_policy_uncert_mean,
            pref_model_uncert_mean=self.pref_model_uncert_mean,
            pref_effort_mean=self.pref_effort_mean,
            capacity_exponent=self.capacity_exponent,
            variance_exponent=self.variance_exponent,
            extrinsic_value_weight=self.extrinsic_value_weight,
            risk_include_variance=self.risk_include_variance,
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

        # Per-episode extrinsic value tracking using AIF preference distance -E[ln p(o|C)].
        # Records neg_extrinsic at episode start and current step to measure improvement.
        self._episode_start_neg_extrinsic: Optional[float] = None
        self._episode_current_neg_extrinsic: Optional[float] = None
        self._extrinsic_value_fresh: bool = False  # True when new data recorded this episode

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
        Also tracks per-episode AIF extrinsic value improvement.
        """
        if not self.enabled:
            return
        self._transition_count += 1

        # Track per-episode extrinsic value using AIF preference distance: -E[ln p(o|C)].
        # Lower neg_extrinsic = closer to preferences = better.
        if next_obs_np is not None and hasattr(self.agent, '_compute_neg_extrinsic_value'):
            try:
                with torch.no_grad():
                    next_t = torch.as_tensor(next_obs_np, dtype=torch.float32, device=self.agent.device).unsqueeze(0)
                    neg_ext = self.agent._compute_neg_extrinsic_value(next_t).mean().item()

                    if self._episode_start_neg_extrinsic is None and obs_np is not None:
                        start_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.agent.device).unsqueeze(0)
                        self._episode_start_neg_extrinsic = self.agent._compute_neg_extrinsic_value(start_t).mean().item()

                    self._episode_current_neg_extrinsic = neg_ext
                    self._extrinsic_value_fresh = True
            except Exception:
                pass

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

    def on_episode_reset(self) -> None:
        """Reset per-episode extrinsic value tracking."""
        self._episode_start_neg_extrinsic = None
        self._episode_current_neg_extrinsic = None
        self._extrinsic_value_fresh = False

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

            # Per-episode extrinsic value improvement: fractional reduction in
            # -E[ln p(o|C)] from episode start to current step.
            # Positive values indicate observations are moving closer to preferences.
            if (self._extrinsic_value_fresh
                    and self._episode_start_neg_extrinsic is not None
                    and self._episode_current_neg_extrinsic is not None):
                start_val = self._episode_start_neg_extrinsic
                current_val = self._episode_current_neg_extrinsic
                denom = max(abs(start_val), 1e-6)
                extrinsic_value_improvement = math.tanh((start_val - current_val) / denom)
                self.extrinsic_value_ema = _update_ema(
                    self.extrinsic_value_ema, extrinsic_value_improvement, self.extrinsic_value_ema_beta
                )
                print(f"[META][extrinsic] episode_improvement={extrinsic_value_improvement:.6f} "
                      f"start_neg_ext={start_val:.4f} "
                      f"current_neg_ext={current_val:.4f} "
                      f"ema={self.extrinsic_value_ema:.6f}")
                # Consume the measurement so we don't update EMA again until a new transition arrives.
                self._extrinsic_value_fresh = False

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
        if policy_uncert_measurement is not None:
            self.policy_uncert_ema = _update_ema(
                self.policy_uncert_ema, policy_uncert_measurement, self.policy_uncert_ema_beta
            )
            policy_uncert_ema = float(self.policy_uncert_ema)

        # Log features and selected planning params
        theta = debug_info.get("theta", 0.5)
        print(f"[META] features: model_error={features.get('model_error', 0.0):.4f} "
              f"model_uncert={features.get('model_uncert', 0.0):.4f} "
              f"policy_uncert={policy_uncert_ema:.4f} "
              f"extrinsic_value={features.get('extrinsic_value', 0.0):.4f} | "
              f"theta={theta:.3f} (pi={debug_info.get('theta_pi', 0):.3f} "
              f"v={debug_info.get('theta_v', 0):.3f} "
              f"rho={debug_info.get('model_reliability', 0):.3f} "
              f"w_eff={debug_info.get('w_effective', 0):.3f}) | "
              f"selected: H={mode.get('horizon')} N={mode.get('candidates')} "
              f"M={mode.get('mc_models')} T={mode.get('mc_trajectories')} "
              f"R={mode.get('action_rollouts')}")

        info = {
            "meta_mode_idx": int(idx),
            "meta_warmup": False,
            "meta_model_error_ema": features.get("model_error", 0.0),
            "meta_model_uncert_ema": features.get("model_uncert", 0.0),
            "meta_policy_uncert_ema": policy_uncert_ema,
            "meta_extrinsic_value": features.get("extrinsic_value", 0.0),
            "meta_theta": theta,
            "meta_theta_pi": debug_info.get("theta_pi", 0.0),
            "meta_theta_v": debug_info.get("theta_v", 0.0),
            "meta_model_reliability": debug_info.get("model_reliability", 0.0),
            "meta_w_effective": debug_info.get("w_effective", 0.0),
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
          - Ambiguity = H(P(o|θ, mode)) at point θ = 1 - normalized_policy_uncert
          - Risk = KL[P(o|θ, mode) || p(o|C)] (cost/habit preferences)

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
