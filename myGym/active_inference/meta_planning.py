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
        }


class MetaGenerativeModel:
    """
    Explicit specification of a meta-level generative model with hidden policy quality.

    Hidden state:
      - θ ∈ [0, 1] (policy quality)

    Prior belief:
      - Q(θ): current uncertainty about policy quality

    Likelihood:
      - P(o | θ, mode): how outcomes depend on policy quality and mode

    Pragmatic value:
      - Expected log preference over outcomes (success + cost/habit)

    Epistemic value:
      - Expected information gain I_Q(θ; o | mode)

    Scores:
      - G(mode) = - (pragmatic_value + epistemic_value)

    We treat compute-cost/habitual deviation as an internal outcome factor with a
    delta predictive distribution, so its log preference reduces to log p(o|C),
    preserving the cost/habit preference term at the meta level.
    """

    def __init__(
        self,
        model_ambiguity_weight: float,
        cost_weight: float,
        habit_weight: float,
        total_capacity: np.ndarray,
        depth_capacity: np.ndarray,
        complexity: np.ndarray,
        habit_deviation: np.ndarray,
        is_deterministic: np.ndarray,
        w_success: float = 1.0,
        p_pref_success: float = 0.9,
        w_epistemic: float = 1.0,
        gate_reliability_threshold: float = 0.3,
        gate_k: float = 10.0,
        gate_theta_threshold: Optional[float] = None,
        gate_k2: float = 10.0,
        planning_load_ambiguity_scale: float = 0.0,
    ):
        """
        Initialize the meta generative model with pre-computed mode arrays.

        Args:
            model_ambiguity_weight: Scales model uncertainty in likelihood sharpness
            cost_weight: Weight for computational cost in preferences
            habit_weight: Weight for habitual deviation in preferences
            total_capacity: Planning capacity per mode [n_modes]
            depth_capacity: Horizon depth per mode [n_modes]
            complexity: Computational complexity per mode [n_modes]
            habit_deviation: Deviation from habitual prior per mode [n_modes]
            is_deterministic: Boolean mask for deterministic modes [n_modes]
            w_success: Weight on success preference term in pragmatic value
            p_pref_success: Preferred success probability (Bernoulli preference)
            w_epistemic: Weight on information gain term
            gate_reliability_threshold: Model reliability threshold for epistemic gate
            gate_k: Sigmoid slope for model reliability gate
            gate_theta_threshold: Optional θ-mean threshold for epistemic gate
            gate_k2: Sigmoid slope for θ-mean gate
            planning_load_ambiguity_scale: Penalty scale for planning-induced ambiguity
        """
        self.model_ambiguity_weight = model_ambiguity_weight
        self.cost_weight = cost_weight
        self.habit_weight = habit_weight
        self.w_success = float(w_success)
        self.p_pref_success = float(p_pref_success)
        self.w_epistemic = float(w_epistemic)
        self.gate_reliability_threshold = float(gate_reliability_threshold)
        self.gate_k = float(gate_k)
        self.gate_theta_threshold = None if gate_theta_threshold is None else float(gate_theta_threshold)
        self.gate_k2 = float(gate_k2)
        self.planning_load_ambiguity_scale = float(planning_load_ambiguity_scale)

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

    def normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize raw features to [0, 1] range for prior/likelihood shaping.

        Returns dict with:
          - normalized_error: sigmoid(raw_error) → (0, 1)
          - normalized_model_uncert: x/(1+x) → [0, 1)
          - normalized_policy_uncert: x/(1+x) → [0, 1)
          - model_reliability: combined reliability measure
        """
        raw_error = float(features.get("model_error", 0.0))
        raw_model_uncert = max(0.0, float(features.get("model_uncert", 0.0)))
        if self.model_ambiguity_weight == 0.0:
            raw_model_uncert = 0.0
        raw_policy_uncert = max(0.0, float(features.get("policy_uncert", 0.0)))

        # Sigmoid for NLL error: (-∞, +∞) → (0, 1)
        normalized_error = 1.0 / (1.0 + math.exp(raw_error))
        # Saturation x/(1+x) for uncertainties: [0, ∞) → [0, 1)
        normalized_model_uncert = raw_model_uncert / (1.0 + raw_model_uncert)
        normalized_policy_uncert = raw_policy_uncert / (1.0 + raw_policy_uncert)
        # Reliability: both error and uncertainty must be low
        uncert_factor = math.exp(-raw_model_uncert)
        model_reliability = normalized_error * uncert_factor

        return {
            "normalized_error": normalized_error,
            "normalized_model_uncert": normalized_model_uncert,
            "normalized_policy_uncert": normalized_policy_uncert,
            "model_reliability": model_reliability,
        }

    def _theta_prior(self, features: Dict[str, float], normalized_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Construct Q(θ) as a Beta distribution over policy quality.

        Uses policy_uncert to set the concentration (broad when uncertain) and
        sets the mean to increase as policy uncertainty decreases.
        """
        policy_uncert = float(normalized_features["normalized_policy_uncert"])
        if features.get("policy_quality_mean") is not None:
            quality_mean = float(features["policy_quality_mean"])
        else:
            # High uncertainty -> neutral mean; low uncertainty -> higher mean.
            quality_mean = 0.5 + 0.5 * (1.0 - policy_uncert)
        quality_mean = min(max(quality_mean, self._theta_eps), 1.0 - self._theta_eps)
        concentration = self._theta_min_concentration + (1.0 - policy_uncert) * (
            self._theta_max_concentration - self._theta_min_concentration
        )
        alpha = quality_mean * concentration
        beta = (1.0 - quality_mean) * concentration
        weights = self._beta_weights(alpha, beta)
        return {
            "mean": quality_mean,
            "concentration": concentration,
            "alpha": alpha,
            "beta": beta,
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

    def _likelihood_success_prob(
        self, theta_values: np.ndarray, normalized_features: Dict[str, float]
    ) -> np.ndarray:
        """
        Compute P(o=1 | θ, mode) for all θ and modes.

        Outcome o is modeled as a Bernoulli success variable. Modes with higher
        planning capacity and reliable models sharpen the likelihood (lower entropy),
        making outcomes more diagnostic of θ.
        """
        reliability = float(normalized_features["model_reliability"])
        model_uncert = float(normalized_features["normalized_model_uncert"])
        model_uncert = min(1.0, model_uncert * self.model_ambiguity_weight)

        capacity = np.clip(self._total_capacity / self._max_capacity, 0.0, 1.0)
        capacity = capacity[None, :]  # [1, n_modes]
        theta = theta_values[:, None]  # [n_theta, 1]

        quality_gain = capacity * reliability * (1.0 - theta)
        effective_quality = theta + quality_gain

        sensitivity = (1.0 + capacity * reliability * self._sensitivity_gain) * (1.0 - model_uncert)
        sensitivity = np.maximum(self._min_sensitivity, sensitivity)

        p_success = 0.5 + sensitivity * (effective_quality - 0.5)
        return np.clip(p_success, self._theta_eps, 1.0 - self._theta_eps)

    def _expected_entropy(self, p_success: np.ndarray, theta_weights: np.ndarray) -> np.ndarray:
        """Compute E_Q(θ)[H(P(o|θ, mode))] for all modes."""
        p = np.clip(p_success, self._theta_eps, 1.0 - self._theta_eps)
        entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
        entropy = entropy / self._entropy_norm
        return np.sum(entropy * theta_weights[:, None], axis=0)

    def _predictive_success(self, p_success: np.ndarray, theta_weights: np.ndarray) -> np.ndarray:
        """Compute q(o=1|π) by marginalizing θ for each mode."""
        return np.sum(p_success * theta_weights[:, None], axis=0)

    def log_prior(self) -> np.ndarray:
        """
        Compute log p(o|C) for the internal cost/habit outcome factor.

        This encodes preference for low-cost, habitual behavior:
          log p(o|C) ∝ -cost_weight * complexity - habit_weight * deviation

        With q(o|π) modeled as a delta at the mode-specific cost, the KL term
        reduces to -log p(o|C), preserving the meta-level preference penalty.

        Returns:
            Array of log preference probabilities per mode [n_modes]
        """
        normalized_complexity = self._complexity / self._max_complexity  # [0, 1]
        return -(self.cost_weight * normalized_complexity + self.habit_weight * self._habit_deviation)

    def expected_free_energy(self, features: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute Expected Free Energy G(mode) for all modes.

        G = - (pragmatic_value + epistemic_value)

        Pragmatic value = success preference + cost/habit log preference
        Epistemic value = I_Q(θ; o | mode)

        Args:
            features: Raw features dict with model_error, model_uncert, policy_uncert

        Returns:
            Tuple of (G scores [n_modes], debug_info dict)
        """
        # Normalize features
        norm_features = self.normalize_features(features)
        reliability = float(norm_features["model_reliability"])

        # Construct prior Q(θ)
        theta_prior = self._theta_prior(features, norm_features)

        # Likelihood P(o|θ, mode) and entropies
        p_success = self._likelihood_success_prob(self._theta_grid, norm_features)
        expected_entropy = self._expected_entropy(p_success, theta_prior["weights"])
        planning_load = np.clip(self._total_capacity / self._max_capacity, 0.0, 1.0)
        if self.planning_load_ambiguity_scale != 0.0:
            planning_induced_ambiguity = (1.0 - reliability) * planning_load * self.planning_load_ambiguity_scale
        else:
            planning_induced_ambiguity = np.zeros_like(expected_entropy)
        effective_entropy = expected_entropy + planning_induced_ambiguity
        q_success = self._predictive_success(p_success, theta_prior["weights"])
        q_success = np.clip(q_success, self._theta_eps, 1.0 - self._theta_eps)
        predictive_entropy = -(q_success * np.log(q_success) + (1.0 - q_success) * np.log(1.0 - q_success))
        predictive_entropy = predictive_entropy / self._entropy_norm
        info_gain = predictive_entropy - effective_entropy

        # Pragmatic value from success + cost/habit preferences
        log_prior_costhabit = self.log_prior()
        p_pref_success = np.clip(float(self.p_pref_success), self._theta_eps, 1.0 - self._theta_eps)
        log_pref_success = math.log(p_pref_success)
        log_pref_fail = math.log(1.0 - p_pref_success)
        success_pref = q_success * log_pref_success + (1.0 - q_success) * log_pref_fail
        pragmatic_value = self.w_success * success_pref + log_prior_costhabit

        # Epistemic value: information gain gated by model quality
        gate_model = 1.0 / (1.0 + math.exp(-self.gate_k * (reliability - self.gate_reliability_threshold)))
        if self.gate_theta_threshold is None:
            gate_theta = 1.0
        else:
            gate_theta = 1.0 / (
                1.0 + math.exp(-self.gate_k2 * (float(theta_prior["mean"]) - self.gate_theta_threshold))
            )
        gate = gate_model * gate_theta
        epistemic_value = self.w_epistemic * gate * info_gain

        # EFE scores: G = -value
        total_value = pragmatic_value + epistemic_value
        G = -total_value

        # Preference penalty (for diagnostics)
        risk = -log_prior_costhabit

        # Debug info
        debug_info = {
            "normalized_features": norm_features,
            "theta_prior": {
                "mean": theta_prior["mean"],
                "concentration": theta_prior["concentration"],
                "alpha": theta_prior["alpha"],
                "beta": theta_prior["beta"],
            },
            "predictive_success": q_success,
            "q_success": q_success,
            "expected_entropy": expected_entropy,
            "expected_entropy_total": effective_entropy,
            "predictive_entropy": predictive_entropy,
            "info_gain": info_gain,
            "planning_load": planning_load,
            "planning_induced_ambiguity": planning_induced_ambiguity,
            "gate_model": gate_model,
            "gate_theta": gate_theta,
            "gate": gate,
            "success_pref": success_pref,
            "log_prior": log_prior_costhabit,
            "log_prior_costhabit": log_prior_costhabit,
            "pragmatic_value": pragmatic_value,
            "epistemic_value": epistemic_value,
            "total_value": total_value,
            "scores": G,
            "ambiguity": expected_entropy,
            "risk": risk,
            "meta_risk": risk,  # Backwards compatibility
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

        # Meta-EFE scores: G = - (pragmatic_value + epistemic_value)
        # Pragmatic value: success preference + cost/habit log preference.
        # Epistemic value: information gain I_Q(θ; o | mode).
        # Model uncertainty scales likelihood sharpness via model_ambiguity_weight.
        self.cost_log = bool(config.get("cost_log", True))
        self.cost_weight = float(config.get("cost_weight", 1.0))
        self.habit_weight = float(config.get("habit_weight", 1.0))
        self.meta_w_success = float(config.get("meta_w_success", 1.0))
        self.meta_p_pref_success = float(config.get("meta_p_pref_success", 0.9))
        self.meta_w_epistemic = float(config.get("meta_w_epistemic", 1.0))
        self.meta_gate_reliability_threshold = float(config.get("meta_gate_reliability_threshold", 0.3))
        self.meta_gate_k = float(config.get("meta_gate_k", 10.0))
        raw_gate_theta_threshold = config.get("meta_gate_theta_threshold")
        self.meta_gate_theta_threshold = (
            None if raw_gate_theta_threshold is None else float(raw_gate_theta_threshold)
        )
        self.meta_gate_k2 = float(config.get("meta_gate_k2", 10.0))
        self.planning_load_ambiguity_scale = float(config.get("planning_load_ambiguity_scale", 0.1))
        # model_ambiguity_weight scales likelihood sharpness
        self.model_ambiguity_weight = float(config.get("model_ambiguity_weight", 1.0))

        # EMA betas for feature tracking
        self.error_ema_beta = float(config.get("error_ema_beta", 0.9))
        self.uncert_ema_beta = float(config.get("uncert_ema_beta", 0.9))
        self.policy_uncert_ema_beta = float(config.get("policy_uncert_ema_beta", 0.9))

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
            model_ambiguity_weight=self.model_ambiguity_weight,
            cost_weight=self.cost_weight,
            habit_weight=self.habit_weight,
            total_capacity=self._total_capacity,
            depth_capacity=self._depth_capacity,
            complexity=self._complexity,
            habit_deviation=self._habit_deviation,
            is_deterministic=self._is_deterministic,
            w_success=self.meta_w_success,
            p_pref_success=self.meta_p_pref_success,
            w_epistemic=self.meta_w_epistemic,
            gate_reliability_threshold=self.meta_gate_reliability_threshold,
            gate_k=self.meta_gate_k,
            gate_theta_threshold=self.meta_gate_theta_threshold,
            gate_k2=self.meta_gate_k2,
            planning_load_ambiguity_scale=self.planning_load_ambiguity_scale,
        )

        # Initialize EMAs with pessimistic values (assume untrained model)
        # model_error=2.0 → exp(-2) ≈ 0.135 reliability → favors reactive modes
        # model_uncert=1.0 → high uncertainty → exploration needed
        # These will quickly adapt once real data comes in
        self.model_error_ema: float = float(config.get("initial_model_error", 2.0))
        self.model_uncert_ema: float = float(config.get("initial_model_uncert", 1.0))
        self.policy_uncert_ema: float = float(config.get("initial_policy_uncert", 1.0))

        # Warmup: force reactive mode until enough transitions collected
        self._transition_count = 0
        self.warmup_transitions = int(config.get("warmup_transitions", config.get("aif_meta_warmup_transitions", 100)))

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
        h_vals = _range_inclusive(self.horizon_min, self.horizon_max, self.horizon_step)
        n_vals = _range_inclusive(self.candidates_min, self.candidates_max, self.candidates_step)
        m_vals = _range_inclusive(self.mc_models_min, self.mc_models_max, self.mc_models_step)
        t_vals = _range_inclusive(self.mc_traj_min, self.mc_traj_max, self.mc_traj_step)
        r_vals = _range_inclusive(self.action_rollouts_min, self.action_rollouts_max, self.action_rollouts_step)
        total = len(h_vals) * len(n_vals) * len(m_vals) * len(t_vals) * len(r_vals)
        if self.max_modes is not None and total > int(self.max_modes):
            raise ValueError(
                f"Meta planning grid has {total} modes; exceeds max_modes={self.max_modes}. "
                "Increase step sizes or max_modes."
            )
        modes = []
        for h in h_vals:
            for n in n_vals:
                for m in m_vals:
                    for t in t_vals:
                        for r in r_vals:
                            modes.append(
                                {
                                    "horizon": int(h),
                                    "candidates": int(n),
                                    "mc_models": int(m),
                                    "mc_trajectories": int(t),
                                    "action_rollouts": int(r),
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
        """
        if not self.enabled:
            return
        self._transition_count += 1
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


    def compute_features(self, obs_np: np.ndarray, update_policy_uncert_ema: bool = True) -> Dict[str, float]:
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
        # Policy/model uncertainties are used when pragmatic/epistemic values are active
        needs_policy_uncert = (self.meta_w_success != 0.0) or (self.meta_w_epistemic != 0.0)
        needs_model_uncert = (self.meta_w_success != 0.0) or (self.meta_w_epistemic != 0.0)
        if self.model_ambiguity_weight == 0.0:
            needs_model_uncert = False

        with torch.no_grad():
            # Only infer state if we need uncertainty estimates
            if needs_policy_uncert or needs_model_uncert:
                s = self.agent._infer_latent_mean(obs_np)
            else:
                s = None

            # Policy uncertainty (needed for epistemic value)
            # Update EMA only when requested (e.g., non-deterministic mode).
            policy_uncert = None
            if needs_policy_uncert and s is not None:
                policy_uncert = self._estimate_policy_uncertainty(s)
                if policy_uncert is not None and update_policy_uncert_ema:
                    self.policy_uncert_ema = _update_ema(
                        self.policy_uncert_ema, policy_uncert, self.policy_uncert_ema_beta
                    )

            # Model uncertainty (needed for epistemic value from ensemble)
            if needs_model_uncert and s is not None:
                model_uncert = self._estimate_model_uncertainty(obs_np, s)
                if model_uncert is None:
                    model_uncert = self._estimate_random_model_uncertainty(s)
                if model_uncert is not None:
                    self.model_uncert_ema = _update_ema(
                        self.model_uncert_ema, model_uncert, self.uncert_ema_beta
                    )

        # Use EMA for scoring; keep the raw measurement separate for optional updates/logging.
        policy_uncert_value = float(self.policy_uncert_ema)

        return {
            "model_error": float(self.model_error_ema),
            "model_uncert": float(self.model_uncert_ema),
            "policy_uncert": float(policy_uncert_value),
            "policy_uncert_ema": float(self.policy_uncert_ema),
            "policy_uncert_measurement": float(policy_uncert) if policy_uncert is not None else None,
            # NOTE: Task-level "risk" removed. Compute cost/habit preferences are
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
        if not deterministic and policy_uncert_measurement is not None:
            self.policy_uncert_ema = _update_ema(
                self.policy_uncert_ema, policy_uncert_measurement, self.policy_uncert_ema_beta
            )
            policy_uncert_ema = float(self.policy_uncert_ema)

        if self.debug:
            debug_features = dict(features)
            if policy_uncert_ema is not None:
                debug_features["policy_uncert"] = policy_uncert_ema
            print(f"[META DEBUG] select_mode: idx={idx} H={mode.get('horizon')} N={mode.get('candidates')} "
                  f"M={mode.get('mc_models')} T={mode.get('mc_trajectories')} R={mode.get('action_rollouts')} "
                  f"det={deterministic} G={scores[idx]:.4f} features={debug_features}")

        info = {
            "meta_mode_idx": int(idx),
            "meta_warmup": False,
            "meta_model_error_ema": features.get("model_error", 0.0),
            "meta_model_uncert_ema": features.get("model_uncert", 0.0),
            "meta_policy_uncert_ema": policy_uncert_ema,
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

    def select_mode_from_features(self, features: Dict[str, float]) -> Tuple[int, Dict[str, Any], List[float], List[float]]:
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

    def score_modes(self, features: Dict[str, float]) -> Tuple[List[float], List[float]]:
        """
        Meta-EFE for planning mode selection using explicit generative model.

        Delegates to MetaGenerativeModel.expected_free_energy() which computes:
          value(mode) = pragmatic_value + epistemic_value
          G(mode) = -value(mode)

        where:
          - Pragmatic value = success preference + cost/habit log preference
          - Epistemic value = I_Q(θ; o | mode)

        Key dynamics:
          - DETERMINISTIC mode: stronger cost/habit preference, but typically lower info gain.
          - DELIBERATIVE mode: higher info gain when model reliability is good,
            but higher compute/habit preference cost.

        Expected behavior:
          - Early (uncertain): deliberative may win if info gain is high and the
            model is reliable enough to justify computation.
          - Late (confident): deterministic wins as success is predictable and
            cost/habit preferences dominate.
        """
        # Use explicit generative model for EFE computation
        scores, debug_info = self.generative_model.expected_free_energy(features)
        self._last_debug_info = debug_info

        min_idx = int(np.argmin(scores))
        det_idx = self._deterministic_idx
        max_idx = self._max_compute_idx
        det_score = float(scores[det_idx]) if det_idx is not None else None
        max_score = float(scores[max_idx]) if max_idx is not None else None
        det_str = f"{det_score:.4f}" if det_score is not None else "NA"
        max_str = f"{max_score:.4f}" if max_score is not None else "NA"
        best_score = float(scores[min_idx]) if scores.size > 0 else None
        if det_score is not None and best_score is not None:
            det_delta = det_score - best_score
            det_delta_str = f"{det_delta:.4f}"
        else:
            det_delta_str = "NA"
        print(
            f"[META SCORE] G_det(idx={det_idx})={det_str} G_max(idx={max_idx})={max_str} "
            f"G_det_minus_best={det_delta_str}"
        )

        if self.debug:
            max_idx = int(np.argmax(scores))
            nf = debug_info["normalized_features"]
            theta_prior = debug_info["theta_prior"]
            print(f"[META SCORE] reliability={nf['model_reliability']:.4f}")
            print(f"[META SCORE] normalized: policy_uncert={nf['normalized_policy_uncert']:.4f} "
                  f"model_uncert={nf['normalized_model_uncert']:.4f} error={nf['normalized_error']:.4f}")
            print(f"[META SCORE] theta_prior: mean={theta_prior['mean']:.4f} "
                  f"concentration={theta_prior['concentration']:.4f}")
            print(f"[META SCORE] pragmatic[best]={debug_info['pragmatic_value'][min_idx]:.4f} "
                  f"epistemic[best]={debug_info['epistemic_value'][min_idx]:.4f} "
                  f"gate={debug_info['gate']:.4f}")
            print(f"[META SCORE] success_pref[best]={debug_info['success_pref'][min_idx]:.4f} "
                  f"log_prior_costhabit[best]={debug_info['log_prior_costhabit'][min_idx]:.4f} "
                  f"info_gain[best]={debug_info['info_gain'][min_idx]:.4f}")
            print(f"[META SCORE] best_mode={min_idx} (G={scores[min_idx]:.4f}) "
                  f"worst_mode={max_idx} (G={scores[max_idx]:.4f})")

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
