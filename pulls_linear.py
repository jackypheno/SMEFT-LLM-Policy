import json
import math
import numpy as np
import flavio
from wilson import Wilson
from iminuit import Minuit
from flavio.statistics.likelihood import FastLikelihood

# ============================================================
# 1. CONFIGURATION & EXPERIMENTAL CONSTRAINTS
# ============================================================
"""
constraints_ewp = [
    'GammaZ', 'sigma_had', 'R_e', 'R_mu', 'R_tau',
    'AFB(Z->ee)', 'AFB(Z->mumu)', 'AFB(Z->tautau)',
    'A(Z->ee)', 'A(Z->mumu)', 'A(Z->tautau)',
    'R_b', 'R_c', 'AFB(Z->bb)', 'AFB(Z->cc)', 'A(Z->bb)', 'A(Z->cc)',
    'm_W', 'GammaW',
    'BR(W->enu)', 'BR(W->munu)', 'BR(W->taunu)',
    'R(W->cX)', 'Rmue(W->lnu)', 'Rtaue(W->lnu)', 'Rtaumu(W->lnu)',
    'A(Z->ss)'
]
"""

# : {'A(Z->bb)', 'R_b', 'A(Z->cc)', 'R_c', 'AFB(Z->cc)'}

constraints_ewp = [
    'GammaZ', 'sigma_had', 'R_e', 'R_mu', 'R_tau',
    'AFB(Z->ee)', 'AFB(Z->mumu)', 'AFB(Z->tautau)',
    'A(Z->ee)', 'A(Z->mumu)', 'A(Z->tautau)',
    'AFB(Z->bb)',
    'm_W', 'GammaW',
    'BR(W->enu)', 'BR(W->munu)', 'BR(W->taunu)',
    'R(W->cX)', 'Rmue(W->lnu)', 'Rtaue(W->lnu)', 'Rtaumu(W->lnu)',
    'A(Z->ss)'
]

OBS_NAMES = constraints_ewp
N_OBS = len(constraints_ewp)


SIGMA_TARGET = 1.0
EXPONENT_CLAMP = 50.0

# Path to pre-computed SMEFT linear expansion coefficients
COEFFICIENTS_PATH = (
    "/home/kumarj/links/projects/def-london/kumarj/gitrepos/"
    "smeft-anomaly/smeft_new/ewp_linear_coefficients.json"
)

# ============================================================
# 2. FLAVIO MEASUREMENT OVERRIDES
# ============================================================

# --- Override m_W Measurement ---
mW_new_central = 80.434  # GeV
mW_new_error = 0.009    # GeV

for name in list(flavio.Measurement.instances.keys()):
    if name == 'CDF_measurement':
        continue
    m = flavio.Measurement.instances[name]
    if 'm_W' in m.all_parameters:
        print(f"Removing old m_W measurement: {name}")
        flavio.Measurement.del_instance(name)

my_mW = flavio.Measurement('CDF_measurement')
my_mW.add_constraint(
    ['m_W'],
    flavio.statistics.probability.NormalDistribution(mW_new_central, mW_new_error)
)

# --- Override AFB(Z->bb) Measurement ---
obs_name_afb = 'AFB(Z->bb)'
sm_central = flavio.sm_prediction(obs_name_afb)
sm_error = flavio.sm_uncertainty(obs_name_afb)

afb_exp_error = 0.0016
tot_error = np.sqrt(sm_error**2 + afb_exp_error**2)
afb_new_central = sm_central + 5 * tot_error

for name in list(flavio.Measurement.instances.keys()):
    m = flavio.Measurement.instances[name]
    if obs_name_afb in m.all_parameters:
        print(f"Removing old AFB(Z->bb) measurement: {name}")
        flavio.Measurement.del_instance(name)

my_afb = flavio.Measurement('AFB_5sigma_measurement')
my_afb.add_constraint(
    [obs_name_afb],
    flavio.statistics.probability.NormalDistribution(afb_new_central, afb_exp_error)
)

print(f"AFB(Z->bb) SM Prediction: {sm_central:.5f} +/- {sm_error:.5f}")
print(f"AFB(Z->bb) New Exp Value: {afb_new_central:.5f} +/- {afb_exp_error:.5f}")


# ============================================================
# 3. LOAD LINEAR SMEFT COEFFICIENTS
# ============================================================

with open(COEFFICIENTS_PATH, "r", encoding="utf-8") as f:
    ewp_data = json.load(f)

ewp_coefficients = ewp_data["observables"]

# ============================================================
# 4. KINEMATIC & PREDICTION HELPERS
# ============================================================

def predict_ewp(obs: str, wc_dict: dict) -> float:
    """Computes linearized SMEFT prediction for a single observable."""
    if obs not in ewp_coefficients:
        raise KeyError(
            f"Observable '{obs}' not found in loaded JSON data."
        )

    data = ewp_coefficients[obs]
    
    # Fall back to 0.0 if sm_prediction isn't present
    sm = data.get("sm_prediction", 0.0)
    
    # Safely get coefficients (defaults to empty dict if key is missing)
    coeffs = data.get("absolute_coefficients", {})
    if not coeffs and "coefficients" in data:
        # Compatibility check in case the JSON key is named "coefficients"
        coeffs = data["coefficients"]

    delta = sum(coeffs[wc] * val for wc, val in wc_dict.items() if wc in coeffs)
    return sm *(1+ delta)

def predict_observables(wc_dict: dict) -> dict:
    """Computes predictions for all observables in constraints_ewp."""
    return {obs: predict_ewp(obs, wc_dict) for obs in constraints_ewp}


# ============================================================
# EXPERIMENTAL CONSTRAINTS & CHI2 HELPERS
# ============================================================

def get_experimental_constraint(obs_name: str) -> tuple:
    """Extracts central value and standard deviation for a given observable from flavio."""
    for name, measurement in flavio.Measurement.instances.items():
        if obs_name not in measurement.all_parameters:
            continue

        for distribution, parameters in measurement._constraints:
            if obs_name in parameters:
                if hasattr(distribution, "covariance"):
                    idx = parameters.index(obs_name)
                    central = distribution.central_value[idx]
                    covariance = np.array(distribution.covariance)
                    sigma = np.sqrt(covariance[idx, idx])
                    return float(central), float(sigma)
                else:
                    central = distribution.central_value
                    sigma = distribution.standard_deviation
                    if isinstance(central, (list, tuple)):
                        central = central[0]
                    if isinstance(sigma, (list, tuple)):
                        sigma = sigma[0]
                    return float(central), float(sigma)

    raise ValueError(f"No experimental constraint found for {obs_name}")


def compute_chi2(predictions: dict) -> float:
    """Computes total chi2 for a set of predictions against experimental data."""
    chi2 = 0.0
    for obs, prediction in predictions.items():
        data = ewp_coefficients.get(obs, {})

        # Extract experimental value safely or fall back to flavio instance
        if "experimental" in data:
            exp = data["experimental"]
        elif "exp" in data:
            exp = data["exp"]
        elif "central" in data:
            exp = data["central"]
        else:
            exp, _ = get_experimental_constraint(obs)

        # Extract uncertainty safely or fall back to flavio instance
        if "sigma" in data:
            sigma = data["sigma"]
        elif "exp_unc" in data:
            sigma = data["exp_unc"]
        elif "error" in data:
            sigma = data["error"]
        else:
            _, sigma = get_experimental_constraint(obs)

        chi2 += ((prediction - exp) / sigma) ** 2

    return chi2


def _compute_chi2_SM() -> float:
    """Compute baseline SM chi2 using zero Wilson coefficients."""
    predictions = predict_observables({})
    return float(compute_chi2(predictions))


# Run after helper functions are defined
CHI2_SM: float = _compute_chi2_SM()
print(f"[chi2_reward] chi2_SM = {CHI2_SM:.6f}")

def compute_baseline_pulls(obs_names):
    """
    Compute SM baseline pulls using the linearized prediction data and
    flavio experimental constraints.
    """
    pulls = []

    for obs_name in obs_names:
        if obs_name not in ewp_coefficients:
            raise ValueError(f"Unknown observable: {obs_name}")

        data = ewp_coefficients[obs_name]

        # SM prediction from JSON or linear prediction at zero WC
        prediction = data.get("sm_prediction", predict_ewp(obs_name, {}))

        # Retrieve experimental central value and uncertainty from flavio
        # (This correctly accounts for your custom mW and AFB measurements)
        central, sigma = get_experimental_constraint(obs_name)

        # Compute SM pull
        pull = (prediction - central) / sigma
        pulls.append(float(pull))

        print(
            f"{obs_name:20s} "
            f"SM={prediction:.6g} "
            f"exp={central:.6g} "
            f"sigma={sigma:.6g} "
            f"pull={pull:.3f}"
        )

    return pulls

BASELINE_PULLS = compute_baseline_pulls(constraints_ewp)

# ============================================================
# 5. MINIMIZATION & PULL EVALUATION
# ============================================================

def n_dof(n_ops: int) -> int:
    return max(N_OBS - n_ops, 1)


def _build_chi2_fn(wc_names: list):
    n = len(wc_names)

    def chi2(*values):
        try:
            wc_dict = {wc_names[i]: float(values[i]) for i in range(n)}

            if any(abs(v) > 1e7 for v in wc_dict.values()):
                return 1e20

            predictions = predict_observables(wc_dict)
            result = compute_chi2(predictions)

            if not np.isfinite(result):
                return 1e20

            return float(result)

        except Exception as e:
            print(f"[chi2] Exception: {e}")
            return 1e20

    fn_src = (
        f"def chi2_named({', '.join(wc_names)}):\n"
        f"    return chi2({', '.join(wc_names)})\n"
    )

    globs = {"chi2": chi2}
    exec(fn_src, globs)
    return globs["chi2_named"]

"""
def _minimise_chi2(wc_names: list) -> tuple:
    chi2_fn = _build_chi2_fn(wc_names)
    start = {name: 0.0 for name in wc_names}

    m = Minuit(chi2_fn, **start)
    for name in wc_names:
        m.errors[name] = 1e-3

    m.migrad()
    m.hesse()

    chi2_min = float(m.fval)
    best_wc = {k: float(v) for k, v in zip(wc_names, m.values)}
    return chi2_min, best_wc
"""
def _minimise_chi2(wc_names: list) -> tuple:

    chi2_fn = _build_chi2_fn(wc_names)

    # Initial values
    start = {
        name: 0.0
        for name in wc_names
    }

    m = Minuit(chi2_fn, **start)

    # --------------------------------------------------------
    # Set parameter errors and bounds
    # --------------------------------------------------------

    for name in wc_names:

        # Initial step size
        m.errors[name] = 1e-3

        # Restrict WC to [-1, 1]
        m.limits[name] = (-1.0, 1.0)

    # --------------------------------------------------------
    # Minimize
    # --------------------------------------------------------

    m.migrad()

    # Estimate uncertainties
    m.hesse()

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    chi2_min = float(m.fval)

    best_wc = {
        k: float(v)
        for k, v in zip(wc_names, m.values)
    }

    return chi2_min, best_wc

def flavio_pull_fn(fired_ops: list) -> tuple:
    """
    Given a list of active operators (fired_ops), minimizes chi2
    and returns predictions, pulls, and the best-fit Wilson coefficients.
    """
    if len(fired_ops) == 0:
        best_wc = {}
        chi2_min = CHI2_SM
    else:
        chi2_min, best_wc = _minimise_chi2(fired_ops)

    prediction_dict = predict_observables(best_wc)

    predictions = [
        float(prediction_dict[obs_name])
        for obs_name in constraints_ewp
    ]

    pulls = []
    for obs_name, prediction in zip(constraints_ewp, predictions):
        central, sigma = get_experimental_constraint(obs_name)
        pull = (prediction - central) / sigma
        pulls.append(float(pull))

    if not np.all(np.isfinite(pulls)):
        raise ValueError(f"Non-finite pulls encountered: {pulls}")

    if np.max(np.abs(pulls)) > 10000:
        raise ValueError(f"Unphysical pull detected: {pulls}")

    return (
        OBS_NAMES,
        pulls,
        float(chi2_min),
        predictions,
        best_wc,
    )



import numpy as np
import flavio
from wilson import Wilson

# ==============================================================================
# 1. SAMPLING & FLAVIO EVALUATION FUNCTIONS
# ==============================================================================

def get_available_wcs():
    """Extract all Wilson Coefficient names present in the linear dictionary."""
    wc_set = set()
    for obs, data in ewp_coefficients.items():
        # Check all standard coefficient keys in JSON structure
        coeffs = (
            data.get("absolute_coefficients") or 
            data.get("coefficients") or 
            data.get("relative_coefficients") or 
            {}
        )
        wc_set.update(coeffs.keys())

    wcs = sorted(list(wc_set))
    if not wcs:
        print("\n[WARNING] No WC names found! Inspecting first JSON entry:")
        if ewp_coefficients:
            sample_obs = list(ewp_coefficients.keys())[0]
            print(f"Keys inside ewp_coefficients['{sample_obs}']:", list(ewp_coefficients[sample_obs].keys()))
    return wcs


def generate_random_wcs(wc_names, n_ops=5, scale_min=-1.0, scale_max=1.0):
    """
    Generates a dictionary with Order(1) Wilson coefficients for linear expansions.
    """
    if not wc_names:
        print("[WARNING] wc_names is empty. Returning empty WC dict.")
        return {}

    selected = np.random.choice(wc_names, size=min(n_ops, len(wc_names)), replace=False)
    values = np.random.uniform(scale_min, scale_max, size=len(selected))
    return {str(wc): float(val) for wc, val in zip(selected, values)}


def evaluate_flavio_full(wc_dict, scale=91.1876, eft="SMEFT", basis="Warsaw"):
    """
    Passes Order(1) WCs straight to Wilson scaled by 1e-6 (GeV^-2).
    Iteratively filters out keys not recognized natively by Wilson in the Warsaw basis.
    """
    if not wc_dict:
        # Fallback SM predictions if no non-zero WCs provided
        return {obs: predict_ewp(obs, {}) for obs in ewp_coefficients.keys()}

    # Direct 1e-6 scaling for flavio/wilson normalization
    scaled_wc = {k: v * 1e-6 for k, v in wc_dict.items()}

    # Initialize Wilson object, filtering unsupported basis keys on failure
    try:
        w = Wilson(scaled_wc, scale=scale, eft=eft, basis=basis)
    except Exception as e:
        print(f"\n[Wilson Warning]: {e}")
        print("[Wilson Warning]: Filtering keys not recognized by Wilson Warsaw basis...")
        valid_wcs = {}
        for k, v in scaled_wc.items():
            try:
                Wilson({k: v}, scale=scale, eft=eft, basis=basis)
                valid_wcs[k] = v
            except Exception:
                print(f"  -> Skipping key '{k}'")
        
        w = Wilson(valid_wcs, scale=scale, eft=eft, basis=basis) if valid_wcs else None

    # Compute predictions
    flavio_preds = {}
    for obs in ewp_coefficients.keys():
        try:
            if w is not None:
                flavio_preds[obs] = float(flavio.np_prediction(obs, w))
            else:
                flavio_preds[obs] = float("nan")
        except Exception:
            flavio_preds[obs] = float("nan")

    return flavio_preds


# ==============================================================================
# 2. COMPARISON RUNNER
# ==============================================================================

# ==============================================================================
# PULLS w.r.t. SM
# ==============================================================================

def compute_sm_uncertainties(observables, N=50):
    """
    Theory-only uncertainty on each SM prediction, via flavio's built-in
    MC error propagation over input parameter uncertainties.

    Cache this once -- it doesn't depend on the Wilson coefficients.
    """
    sm_unc = {}
    for obs in observables:
        try:
            sm_unc[obs] = flavio.sm_uncertainty(obs, N=N)
        except Exception as e:
            print(f"[WARN] sm_uncertainty failed for {obs}: {e}")
            sm_unc[obs] = float("nan")
    return sm_unc


def compute_exp_uncertainties(observables):
    """
    Experimental uncertainty from flavio's measurement database, combined
    across all measurements of each observable (correlations included
    where flavio has them). Returns (central, uncertainty) per obs.
    Falls back to (nan, nan) if no measurement exists for that observable.
    """
    exp = {}
    for obs in observables:
        try:
            # pulls this observable's combined measurement (central value + error)
            # across all flavio Measurement objects that constrain it
            m = flavio.combine_measurements(obs)
            exp[obs] = (m.central_value, m.error_left, m.error_right)
        except Exception:
            exp[obs] = (float("nan"), float("nan"), float("nan"))
    return exp


def compare_predictions_with_pulls(sample_wcs, N_sm_unc=50):
    lin_preds = {}
    for obs in ewp_coefficients.keys():
        lin_preds[obs] = predict_ewp(obs, sample_wcs)

    fla_preds = evaluate_flavio_full(sample_wcs, scale=91.1876)

    sm_preds = {obs: flavio.sm_prediction(obs) for obs in ewp_coefficients.keys()}
    sm_unc = compute_sm_uncertainties(ewp_coefficients.keys(), N=N_sm_unc)
    exp_data = compute_exp_uncertainties(ewp_coefficients.keys())

    print("\n" + "=" * 150)
    print("SAMPLED WILSON COEFFICIENTS (Order 1):")
    for k, v in sample_wcs.items():
        print(f"  {k:<20s} = {v:+.6e}")
    print("=" * 150)

    header = (
        f"{'Observable':<20} | {'SM Pred':<12} | {'Flavio Pred':<12} | {'Abs Shift':<11} | {'Rel Shift (%)':<13} | "
        f"{'SM unc (th)':<12} | {'Pull (th)':<10} | {'Exp central':<12} | {'Exp unc':<10} | {'Pull (exp)'}"
    )
    print(header)
    print("-" * 150)

    abs_shifts = []
    rel_shifts = []

    for obs in ewp_coefficients.keys():
        p_sm = sm_preds[obs]
        p_fla = fla_preds.get(obs, float("nan"))
        sigma_th = sm_unc.get(obs, float("nan"))

        # --- NP shift relative to SM ---
        abs_shift = abs(p_fla - p_sm) if not np.isnan(p_fla) else float("nan")
        rel_shift = (abs_shift / abs(p_sm)) * 100 if p_sm != 0 and not np.isnan(abs_shift) else float("nan")
        if not np.isnan(abs_shift):
            abs_shifts.append(abs_shift)
            rel_shifts.append(rel_shift)

        pull_th = (p_fla - p_sm) / sigma_th if sigma_th and not np.isnan(sigma_th) and sigma_th != 0 else float("nan")

        central, err_lo, err_hi = exp_data.get(obs, (float("nan"), float("nan"), float("nan")))
        if not np.isnan(central):
            sigma_exp = err_hi if p_fla >= central else err_lo
            pull_exp = (p_fla - central) / sigma_exp if sigma_exp else float("nan")
        else:
            sigma_exp, pull_exp = float("nan"), float("nan")

        print(
            f"{obs:<20} | {p_sm:<12.6g} | {p_fla:<12.6g} | {abs_shift:<11.4e} | {rel_shift:<13.4f} | "
            f"{sigma_th:<12.4g} | {pull_th:<10.3f} | {central:<12.6g} | {sigma_exp:<10.4g} | {pull_exp:<8.3f}"
        )

    print("-" * 150)
    if abs_shifts:
        print(f"NP SHIFT SUMMARY OVER {len(abs_shifts)} OBSERVABLES (Flavio vs SM):")
        print(f"  Max Absolute Shift : {max(abs_shifts):.6e}")
        print(f"  Mean Absolute Shift: {np.mean(abs_shifts):.6e}")
        finite_rel = [r for r in rel_shifts if not np.isnan(r)]
        if finite_rel:
            print(f"  Max Relative Shift : {max(finite_rel):.4f}%")
            print(f"  Mean Relative Shift: {np.mean(finite_rel):.4f}%")
    print("=" * 150 + "\n")

    return {
        "sm_pred": sm_preds,
        "flavio_pred": fla_preds,
        "sm_unc": sm_unc,
        "exp": exp_data,
        "abs_shift": dict(zip(ewp_coefficients.keys(), abs_shifts)) if abs_shifts else {},
    }
# ==============================================================================
# 3. EXECUTION
# ==============================================================================

if __name__ == "__main__":
    all_wcs = get_available_wcs()
    print(f"Total active Wilson Coefficients found in JSON: {len(all_wcs)}")

    if all_wcs:
        # Sample 5 random operators at O(1) scale
        sample_wcs = generate_random_wcs(all_wcs, n_ops=5, scale_min=--0.1, scale_max=0.1)
    else:
        # Fallback manual test if JSON extraction still finds 0 keys
        print("[FALLBACK] Using manual test WCs...")
        sample_wcs = {"phiq3_33": 0.5, "ll_1122": -0.5}

    compare_predictions_with_pulls(sample_wcs)    
