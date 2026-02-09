"""
Test script for Meta-AIF mode selection under different scenarios.

Tests how the meta-level generative model selects planning modes based on:
- Model quality (error/uncertainty)
- Policy quality (uncertainty and extrinsic value)

Loads parameters from train_nico_meta_aif.json config file.
"""

import json
import os
import re
import numpy as np
from meta_planning import MetaGenerativeModel


def load_config(config_path=None):
    """Load config from JSON file, stripping comments."""
    if config_path is None:
        # Default path relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "..", "configs", "train_nico_meta_aif.json")

    with open(config_path, "r") as f:
        content = f.read()

    # Remove // comments (but not inside strings)
    content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
    return json.loads(content)


def range_inclusive(min_val, max_val, step):
    """Generate inclusive range like the meta_planning module."""
    min_val = int(min_val)
    max_val = int(max_val)
    step = max(1, int(step))
    if max_val < min_val:
        max_val = min_val
    return list(range(min_val, max_val + 1, step))


def create_test_generative_model(config=None):
    """Create a MetaGenerativeModel using parameters from config file."""
    if config is None:
        config = load_config()

    # Extract meta parameters from config
    horizon_min = config.get("aif_meta_horizon_min", 3)
    horizon_max = config.get("aif_meta_horizon_max", 15)
    horizon_step = config.get("aif_meta_horizon_step", 3)

    candidates_min = config.get("aif_meta_candidates_min", 10)
    candidates_max = config.get("aif_meta_candidates_max", 100)
    candidates_step = config.get("aif_meta_candidates_step", 5)

    mc_models_min = config.get("aif_meta_mc_models_min", 2)
    mc_models_max = config.get("aif_meta_mc_models_max", 5)
    mc_models_step = config.get("aif_meta_mc_models_step", 1)

    mc_traj_min = config.get("aif_meta_mc_trajectories_min", 2)
    mc_traj_max = config.get("aif_meta_mc_trajectories_max", 10)
    mc_traj_step = config.get("aif_meta_mc_trajectories_step", 2)

    action_rollouts_min = config.get("aif_meta_action_rollouts_min", 1)
    action_rollouts_max = config.get("aif_meta_action_rollouts_max", 10)
    action_rollouts_step = config.get("aif_meta_action_rollouts_step", 2)

    allow_deterministic = config.get("aif_meta_allow_deterministic", True)
    cost_log = config.get("aif_meta_cost_log", True)

    # EFE weights
    policy_ambiguity_weight = config.get("aif_meta_policy_ambiguity_weight", 1.0)
    model_ambiguity_weight = config.get("aif_meta_model_ambiguity_weight", 1.0)

    # Observation preference precisions
    pref_extrinsic_precision = config.get("aif_meta_pref_extrinsic_precision", 1.0)
    pref_policy_uncert_precision = config.get("aif_meta_pref_policy_uncert_precision", 1.0)
    pref_model_uncert_precision = config.get("aif_meta_pref_model_uncert_precision", 1.0)
    pref_effort_precision = config.get("aif_meta_pref_effort_precision", 1.0)

    # Bayesian belief parameters
    initial_belief_alpha = config.get("aif_meta_initial_belief_alpha", 1.0)
    initial_belief_beta = config.get("aif_meta_initial_belief_beta", 1.0)
    observation_weight = config.get("aif_meta_observation_weight", 1.0)
    belief_decay = config.get("aif_meta_belief_decay", 0.995)

    # Build mode grid with aligned parameters (low with low, high with high)
    h_vals = range_inclusive(horizon_min, horizon_max, horizon_step)
    n_vals = range_inclusive(candidates_min, candidates_max, candidates_step)
    m_vals = range_inclusive(mc_models_min, mc_models_max, mc_models_step)
    t_vals = range_inclusive(mc_traj_min, mc_traj_max, mc_traj_step)
    r_vals = range_inclusive(action_rollouts_min, action_rollouts_max, action_rollouts_step)

    # Find the maximum number of steps across all parameters
    all_vals = [h_vals, n_vals, m_vals, t_vals, r_vals]
    max_steps = max(len(v) for v in all_vals)

    # Interpolate each parameter list to have max_steps entries
    def interpolate_to_length(vals, target_len):
        if len(vals) == target_len:
            return vals
        if len(vals) == 1:
            return vals * target_len
        # Linear interpolation of indices
        result = []
        for i in range(target_len):
            idx_float = i * (len(vals) - 1) / (target_len - 1)
            idx = int(round(idx_float))
            result.append(vals[idx])
        return result

    h_aligned = interpolate_to_length(h_vals, max_steps)
    n_aligned = interpolate_to_length(n_vals, max_steps)
    m_aligned = interpolate_to_length(m_vals, max_steps)
    t_aligned = interpolate_to_length(t_vals, max_steps)
    r_aligned = interpolate_to_length(r_vals, max_steps)

    # Generate aligned mode combinations (one per step level)
    modes = []
    if allow_deterministic:
        modes.append({"horizon": 0, "candidates": 0, "mc_models": 0,
                      "mc_trajectories": 0, "action_rollouts": 0, "deterministic": True})

    for i in range(max_steps):
        modes.append({
            "horizon": h_aligned[i],
            "candidates": n_aligned[i],
            "mc_models": m_aligned[i],
            "mc_trajectories": t_aligned[i],
            "action_rollouts": r_aligned[i],
            "deterministic": False,
        })

    n_modes = len(modes)
    print(f"Generated {n_modes} modes from config grid")
    print(f"  Horizons: {h_vals}")
    print(f"  Candidates: {n_vals}")
    print(f"  MC Models: {m_vals}")
    print(f"  MC Trajectories: {t_vals}")
    print(f"  Action Rollouts: {r_vals}")

    # Extract arrays from modes
    horizons = np.array([m["horizon"] for m in modes], dtype=np.float32)
    candidates = np.array([m["candidates"] for m in modes], dtype=np.float32)
    mc_models = np.array([m["mc_models"] for m in modes], dtype=np.float32)
    mc_trajectories = np.array([m["mc_trajectories"] for m in modes], dtype=np.float32)
    action_rollouts = np.array([m["action_rollouts"] for m in modes], dtype=np.float32)
    is_deterministic = np.array([m.get("deterministic", False) for m in modes], dtype=bool)

    # Compute derived arrays
    raw_cost = horizons * candidates * mc_models * mc_trajectories * action_rollouts
    if cost_log:
        complexity = np.log1p(raw_cost)
    else:
        complexity = raw_cost.copy()

    # Total capacity (geometric mean of normalized components)
    H_max = max(horizons.max(), 1.0)
    N_max = max(candidates.max(), 1.0)
    M_max = max(mc_models.max(), 1.0)
    T_max = max(mc_trajectories.max(), 1.0)
    R_max = max(action_rollouts.max(), 1.0)
    eps = 1e-6

    components = np.stack([
        horizons / H_max + eps,
        candidates / N_max + eps,
        mc_models / M_max + eps,
        mc_trajectories / T_max + eps,
        action_rollouts / R_max + eps,
    ], axis=1)
    total_capacity = np.exp(np.mean(np.log(components), axis=1))

    # Habitual deviation (distance from minimal computation)
    habit_deviation = (
        ((horizons - 1.0) / H_max) ** 2 +
        ((candidates - 1.0) / N_max) ** 2 +
        ((mc_models - 1.0) / M_max) ** 2 +
        ((mc_trajectories - 1.0) / T_max) ** 2 +
        ((action_rollouts - 1.0) / R_max) ** 2
    )
    habit_deviation = np.where(is_deterministic, 0.0, habit_deviation)

    depth_capacity = horizons

    model = MetaGenerativeModel(
        policy_ambiguity_weight=policy_ambiguity_weight,
        model_ambiguity_weight=model_ambiguity_weight,
        total_capacity=total_capacity,
        depth_capacity=depth_capacity,
        complexity=complexity,
        habit_deviation=habit_deviation,
        is_deterministic=is_deterministic,
        initial_belief_alpha=initial_belief_alpha,
        initial_belief_beta=initial_belief_beta,
        observation_weight=observation_weight,
        belief_decay=belief_decay,
        pref_extrinsic_precision=pref_extrinsic_precision,
        pref_policy_uncert_precision=pref_policy_uncert_precision,
        pref_model_uncert_precision=pref_model_uncert_precision,
        pref_effort_precision=pref_effort_precision,
    )

    # Create mode names
    mode_names = []
    for m in modes:
        if m.get("deterministic", False):
            mode_names.append("deterministic")
        else:
            mode_names.append(f"H{m['horizon']}_N{m['candidates']}_M{m['mc_models']}_T{m['mc_trajectories']}_R{m['action_rollouts']}")

    # Print config summary
    print(f"\nEFE Weights from config:")
    print(f"  policy_ambiguity_weight={policy_ambiguity_weight}")
    print(f"  model_ambiguity_weight={model_ambiguity_weight}")
    print(f"\nObservation preference precisions:")
    print(f"  pref_extrinsic_precision={pref_extrinsic_precision}")
    print(f"  pref_policy_uncert_precision={pref_policy_uncert_precision}")
    print(f"  pref_model_uncert_precision={pref_model_uncert_precision}")
    print(f"  pref_effort_precision={pref_effort_precision}")
    print(f"\nBayesian belief params:")
    print(f"  initial_alpha={initial_belief_alpha}, initial_beta={initial_belief_beta}")
    print(f"  observation_weight={observation_weight}, belief_decay={belief_decay}")

    return model, mode_names, modes


def test_scenario(model, mode_names, scenario_name, features, n_updates=5):
    """
    Test a scenario by running Bayesian updates and computing EFE.

    Args:
        model: MetaGenerativeModel instance
        mode_names: List of mode names
        scenario_name: Description of the scenario
        features: Dict with model_error, model_uncert, policy_uncert, extrinsic_value
        n_updates: Number of Bayesian updates to run before final selection
    """
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*70}")
    print(f"Features: model_error={features['model_error']:.2f}, "
          f"model_uncert={features['model_uncert']:.2f}, "
          f"policy_uncert={features['policy_uncert']:.2f}, "
          f"extrinsic_value={features['extrinsic_value']:.2f}")

    # Reset belief to uniform prior
    model._belief_alpha = 1.0
    model._belief_beta = 1.0

    # Run several Bayesian updates to let belief converge
    print(f"\nRunning {n_updates} Bayesian updates...")
    for i in range(n_updates):
        normalized = model.normalize_features(features)
        belief_info = model.bayesian_update(normalized)
        if i == 0 or i == n_updates - 1:
            print(f"  Update {i+1}: belief_mean={belief_info['belief_mean']:.3f}, "
                  f"belief_concentration={belief_info['belief_concentration']:.1f}")

    # Compute EFE scores
    scores, debug_info = model.expected_free_energy(features)

    print(f"\nEFE Scores (lower is better):")
    print(f"  {'Mode':<15} {'EFE':>10} {'Ambiguity':>12} {'Risk':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*10}")

    ambiguity = debug_info.get("ambiguity", np.zeros(len(scores)))
    risk = debug_info.get("risk", np.zeros(len(scores)))

    for i, (name, score) in enumerate(zip(mode_names, scores)):
        print(f"  {name:<15} {score:>10.4f} {ambiguity[i]:>12.4f} {risk[i]:>10.4f}")

    best_idx = int(np.argmin(scores))
    print(f"\n>>> SELECTED MODE: {mode_names[best_idx]} (index {best_idx})")

    # Show theta prior info
    theta_prior = debug_info.get("theta_prior", {})
    print(f"\nPolicy quality belief (theta prior):")
    print(f"  mean={theta_prior.get('mean', 0.5):.3f}, "
          f"concentration={theta_prior.get('concentration', 2.0):.1f}, "
          f"alpha={theta_prior.get('alpha', 1.0):.2f}, "
          f"beta={theta_prior.get('beta', 1.0):.2f}")

    return best_idx, mode_names[best_idx]


def main():
    print("Meta-AIF Mode Selection Test")
    print("="*70)

    model, mode_names, _ = create_test_generative_model()

    # Scenario 1: Bad model, bad policy
    # High model error, high uncertainty, negative extrinsic value
    _, m0 = test_scenario(
        model, mode_names,
        "BAD MODEL + BAD POLICY",
        {
            "model_error": 1.0,      # High error (bad predictions)
            "model_uncert": 1.0,     # High uncertainty
            "policy_uncert": 0.9,    # High policy uncertainty
            "extrinsic_value": 0.0, # Negative improvement (getting worse)
        }
    )

    # Scenario 2: Good model, bad policy
    # Low model error, low uncertainty, but negative extrinsic value
    _, m1 = test_scenario(
        model, mode_names,
        "GOOD MODEL + BAD POLICY",
        {
            "model_error": -4.8,      # Low error (good predictions)
            "model_uncert": 0.0001,     # Low uncertainty
            "policy_uncert": 0.4,    # High policy uncertainty
            "extrinsic_value": 0.2, # Slightly negative (not improving)
        }
    )

    # Scenario 3: Good model, good policy
    # Low model error, low uncertainty, positive extrinsic value
    _, m2 = test_scenario(
        model, mode_names,
        "GOOD MODEL + GOOD POLICY",
        {
            "model_error": -4.9,      # Low error
            "model_uncert": 0.0001,     # Low uncertainty
            "policy_uncert": 0.577,    # Low policy uncertainty
            "extrinsic_value": 0.795,  # Strong positive improvement
        }
    )

    # Scenario 4: Bad model, good policy (edge case)
    # High model error but policy seems to be working
    _, m3 = test_scenario(
        model, mode_names,
        "BAD MODEL + GOOD POLICY (edge case)",
        {
            "model_error": 3.0,      # High error
            "model_uncert": 1.5,     # High uncertainty
            "policy_uncert": 0.2,    # Low policy uncertainty
            "extrinsic_value": 0.5,  # Positive improvement
        }
    )

    # Scenario 5: Neutral/uncertain state
    # Medium values everywhere
    _, m4 = test_scenario(
        model, mode_names,
        "NEUTRAL/UNCERTAIN STATE",
        {
            "model_error": -1.0,
            "model_uncert": 0.2,
            "policy_uncert": 0.5,
            "extrinsic_value": 0.0,  # No improvement
        }
    )

    print("\n" + "="*70)
    print("SUMMARY OF BEHAVIOR:")
    print("="*70)
    print(f"""
    - BAD MODEL + BAD POLICY: Should prefer deterministic/low modes
       Chose: {m0}

    - GOOD MODEL + BAD POLICY: Should prefer high/max planning modes
      Chose: {m1}

    - GOOD MODEL + GOOD POLICY: Should prefer deterministic/low modes
        Chose: {m2}

    - BAD MODEL + GOOD POLICY: Should prefer deterministic modes
         Chose: {m3}

    - NEUTRAL STATE: Moderate planning (balance exploration vs cost)
        Chose: {m4}
    """)


if __name__ == "__main__":
    main()
