#!/usr/bin/env python3
"""QuantumRadar-Lab - educational quantum illumination / quantum radar lab.

by Darion Kellar

Pure Python standard library. Zero dependencies. Zero network.
Education only - and the central lesson is stated up front:

  * This lab does FORMULA ARITHMETIC on a labelled textbook model
    of quantum illumination (Lloyd 2008; Tan et al. 2008). It
    computes error exponents and error probabilities for a binary
    target-detection question - "target present or absent?" - under
    the standard Gaussian toy model stated in full below.
  * No radar was built. No transmitter exists, no photon was sent,
    no target was detected. There is no range figure, no resolution
    figure, and no stealth claim of any kind anywhere in this lab,
    because the model produces none.
  * The famous "6 dB advantage" of quantum illumination is, in this
    lab, exactly what it is in the literature: a ratio of ERROR
    EXPONENTS inside a stated regime (dim signal, loud background,
    weak reflectivity), computed here - not quoted - and shown to
    collapse to 1 outside that regime.
  * The lossy, noisy return channel BREAKS the entanglement between
    the returned signal and the retained idler. What survives is a
    classically-correlated-but-quantum-born correlation, and the
    advantage lives in measuring the return and the idler jointly.
    The lab says this plainly wherever the advantage is printed.

Military, targeting, surveillance, stealth-detection, weapons,
air-defense, and product claims are permanent HOLD gates - with
Gemini Argon 4 Extended Thinking as DECIDER (names-only,
tighten-only), Muse as advisor (never overrides HOLD), and
Google Flow plan-only seats.

THE LABELLED MODEL (every formula in this lab, stated once here):

  Target detection is binary hypothesis testing. The target, if
  present, has reflectivity kappa (0 <= kappa <= 1, LABELLED).
  Each of M independent modes carries N_S signal photons on
  average (signal brightness) into a background of N_B noise
  photons per mode (background brightness).

  Classical benchmark - coherent-state transmitter, homodyne
  receiver (the Lloyd / Tan et al. benchmark). Its per-mode
  Chernoff error exponent is

      xi_CI = kappa * N_S * ( sqrt(N_B + 1) - sqrt(N_B) )^2

  and the error probability at M modes is the Gaussian Q-function
  of the accumulated discriminant,

      P_CI(M) = Q( sqrt(2 * M * xi_CI) ),
      Q(x) = 0.5 * erfc(x / sqrt(2)).

  Quantum illumination - entangled signal/idler source, optimal
  joint receiver. Its per-mode Chernoff error exponent is

      xi_QI = A(N_S, N_B) * xi_CI,

  where the advantage factor A interpolates the Tan et al.
  asymptotic endpoints, and is LABELLED as this lab's teaching
  interpolation between two literature results:

      A = 1 + 3 * [ N_B / (N_B + 1) ] * [ 1 / (1 + N_S) ]

      A -> 4  (the 6 dB result)  when N_S << 1 and N_B >> 1
      A -> 1  (no advantage)     when N_B -> 0 or N_S >> 1

  and the error probability is the Chernoff-form expression

      P_QI(M) = 0.5 * exp(-M * xi_QI).

  Both error formulas are asymptotic exponent statements. At small
  M the (0.5 * exp) Chernoff form is loose and can sit above the
  exact Gaussian number; the advantage is an exponent, and it is
  visible as M grows. The lab prints both forms side by side so
  the crossover is seen, not hidden.
"""
import hashlib
import json
import math
import sys

# ---------------------------------------------------------------- constants
VERSION = "1.0.0"

# Demo operating point: the textbook QI regime (dim signal, loud
# background, weak reflectivity). Labelled teaching values.
DEMO_KAPPA = 0.10
DEMO_N_S = 0.01
DEMO_N_B = 20.0
TARGET_P_ERROR = 1e-6

# Labelled practicality line for the regime classifier: if even the
# classical benchmark needs more modes than this to reach the
# target error, the operating point is labelled "hopeless" - a
# statement about this model's arithmetic, not about any device.
HOPELESS_MODE_LIMIT = 10 ** 8

REGIME_LABELS = ("qi-advantaged", "classical-equal", "hopeless")

# ---------------------------------------------------------------- HOLD gates
HOLD_GATES = {
    "military_targeting_use": "HOLD - no targeting, fire-control, or weapons-aiming use of any kind",
    "surveillance_use": "HOLD - no surveillance of people, vehicles, property, or airspace",
    "stealth_detection_claim": "HOLD - no claim that this model detects stealth aircraft or defeats low-observable design; the model produces no such result",
    "weapons_guidance_use": "HOLD - no guidance, navigation-for-weapons, or terminal-homing use",
    "air_defense_application": "HOLD - no air-defense, missile-warning, or interception application",
    "drone_detection_claim": "HOLD - no drone / UAS detection claim; nothing here detects anything",
    "real_radar_construction": "HOLD - no building a real radar, transmitter, or receiver from this lab",
    "live_target_detection": "HOLD - no live detection of real objects; this lab runs no hardware and senses nothing",
    "range_or_performance_claim": "HOLD - no range, resolution, sensitivity, or performance claim for any real system; the model outputs exponents, not ranges",
    "hardware_control": "HOLD - no control of real transmitters, receivers, antennas, or sources",
    "product_or_deployment_claim": "HOLD - no product, deployment, procurement, or capability claim; formula arithmetic is not a system",
    "advantage_marketing_claim": "HOLD - the 6 dB figure is a labelled model exponent ratio inside a stated regime; it must never be presented as a product advantage or field capability",
    "medical_safety_imaging_claim": "HOLD - no medical, safety-of-life, or imaging claim",
    "classified_or_export_design": "HOLD - no classified, export-controlled, or restricted design work",
}


def gate(action):
    """Return 'HOLD' if the action is gated, else 'PROCEED' (sandbox sim)."""
    return "HOLD" if action in HOLD_GATES else "PROCEED"


# ================================================================ 1. The labelled model
# Binary hypothesis testing under the Gaussian toy model stated in
# the module docstring. Every function here is a direct transcription
# of a labelled formula - the code invents no physics.

def _check_params(kappa, n_s, n_b):
    if not 0.0 <= kappa <= 1.0:
        raise ValueError("kappa (reflectivity) must lie in [0, 1]")
    if n_s < 0.0 or n_b < 0.0:
        raise ValueError("brightnesses N_S and N_B must be >= 0")


def q_function(x):
    """Gaussian tail probability Q(x) = P(Z > x) for standard normal Z."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def q_inverse(p):
    """Smallest x with Q(x) <= p, by bisection (Q is decreasing)."""
    if not 0.0 < p < 0.5:
        raise ValueError("p must lie in (0, 0.5)")
    lo, hi = 0.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if q_function(mid) > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def classical_exponent(kappa, n_s, n_b):
    """Per-mode Chernoff exponent, coherent-state + homodyne benchmark.

    xi_CI = kappa * N_S * ( sqrt(N_B + 1) - sqrt(N_B) )^2   (LABELLED)
    High-noise limit: xi_CI -> kappa * N_S / (4 * N_B).
    """
    _check_params(kappa, n_s, n_b)
    u = math.sqrt(n_b + 1.0) - math.sqrt(n_b)
    return kappa * n_s * u * u


def advantage_factor(n_s, n_b):
    """QI exponent advantage A(N_S, N_B) - LABELLED interpolation.

    A = 1 + 3 * [N_B / (N_B + 1)] * [1 / (1 + N_S)], built so the
    endpoints are the Tan et al. asymptotic results: A -> 4 (6 dB)
    for N_S << 1 << N_B, and A -> 1 when N_B -> 0 or N_S >> 1.
    The interpolation between the endpoints is this lab's teaching
    model; the endpoints are the literature.
    """
    if n_s < 0.0 or n_b < 0.0:
        raise ValueError("brightnesses N_S and N_B must be >= 0")
    return 1.0 + 3.0 * (n_b / (n_b + 1.0)) * (1.0 / (1.0 + n_s))


def qi_exponent(kappa, n_s, n_b):
    """Per-mode Chernoff exponent for quantum illumination (LABELLED).

    xi_QI = A(N_S, N_B) * xi_CI - the entangled-source exponent of
    the Tan et al. model, with A as defined above.
    """
    return advantage_factor(n_s, n_b) * classical_exponent(kappa, n_s, n_b)


def pc_exponent(kappa, n_s, n_b):
    """Phase-conjugation receiver exponent (LABELLED note).

    The Guha-Erkmen phase-conjugation receiver is documented to
    capture half of the QI exponent gain (3 dB of the 6 dB); the
    optimal joint receiver reaches the full Chernoff exponent.
    Modelled here as the midpoint in exponent gain.
    """
    xi_ci = classical_exponent(kappa, n_s, n_b)
    return xi_ci + 0.5 * (qi_exponent(kappa, n_s, n_b) - xi_ci)


def error_prob_classical(modes, kappa, n_s, n_b):
    """P_err for the classical benchmark: Q( sqrt(2 * M * xi_CI) )."""
    if modes < 0:
        raise ValueError("modes must be >= 0")
    return q_function(math.sqrt(2.0 * modes * classical_exponent(kappa, n_s, n_b)))


def error_prob_qi(modes, kappa, n_s, n_b):
    """P_err for QI, Chernoff form: 0.5 * exp(-M * xi_QI) (LABELLED)."""
    if modes < 0:
        raise ValueError("modes must be >= 0")
    return 0.5 * math.exp(-modes * qi_exponent(kappa, n_s, n_b))


def modes_needed(target, kind, kappa, n_s, n_b):
    """Smallest integer mode count M with P_err(M) <= target.

    Classical: invert the Q-function. QI: invert the exponential.
    Returns None when the exponent is zero (target unreachable).
    """
    if not 0.0 < target < 0.5:
        raise ValueError("target must lie in (0, 0.5)")
    if kind == "classical":
        xi = classical_exponent(kappa, n_s, n_b)
        if xi <= 0.0:
            return None
        x = q_inverse(target)
        return max(1, math.ceil(x * x / (2.0 * xi)))
    if kind == "qi":
        xi = qi_exponent(kappa, n_s, n_b)
        if xi <= 0.0:
            return None
        return max(1, math.ceil(math.log(0.5 / target) / xi))
    raise ValueError("kind must be 'classical' or 'qi'")


def roc_table(kappa=DEMO_KAPPA, n_s=DEMO_N_S, n_b=DEMO_N_B,
              mode_counts=(10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6)):
    """P_error vs M for classical vs QI at the given operating point."""
    rows = []
    for m in mode_counts:
        rows.append({
            "modes": m,
            "p_error_classical": error_prob_classical(m, kappa, n_s, n_b),
            "p_error_qi": error_prob_qi(m, kappa, n_s, n_b),
        })
    return rows


def exponent_report(kappa=DEMO_KAPPA, n_s=DEMO_N_S, n_b=DEMO_N_B):
    """The full exponent book for one operating point."""
    xi_ci = classical_exponent(kappa, n_s, n_b)
    xi_qi = qi_exponent(kappa, n_s, n_b)
    return {
        "kappa": kappa,
        "n_s": n_s,
        "n_b": n_b,
        "xi_classical_per_mode": xi_ci,
        "xi_pc_receiver_per_mode": pc_exponent(kappa, n_s, n_b),
        "xi_qi_per_mode": xi_qi,
        "advantage_factor": (xi_qi / xi_ci) if xi_ci > 0 else None,
        "advantage_db": (10.0 * math.log10(xi_qi / xi_ci)) if xi_ci > 0 else None,
        "modes_classical_for_1e-6": modes_needed(TARGET_P_ERROR, "classical",
                                                 kappa, n_s, n_b),
        "modes_qi_for_1e-6": modes_needed(TARGET_P_ERROR, "qi",
                                          kappa, n_s, n_b),
    }


def regime_boundary_tables():
    """The advantage's regime boundary, computed - not quoted.

    A(N_S, N_B) along two ladders: background brightness at fixed
    dim signal, and signal brightness at fixed loud background.
    """
    return {
        "advantage_vs_background_at_ns_0.01": [
            {"n_b": nb, "advantage": advantage_factor(0.01, nb)}
            for nb in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
        ],
        "advantage_vs_signal_at_nb_20": [
            {"n_s": ns, "advantage": advantage_factor(ns, 20.0)}
            for ns in (0.001, 0.01, 0.1, 1.0, 10.0)
        ],
    }


# ================================================================ 2. Scenario book
# Four operating points that between them map the model's regimes.
# Every number in every report is computed from section 1.

SCENARIOS = {
    "textbook_regime": {
        "kappa": DEMO_KAPPA, "n_s": DEMO_N_S, "n_b": DEMO_N_B,
        "note": "dim signal, loud background, weak reflectivity - the "
                "regime the 6 dB result is about",
    },
    "quiet_background": {
        "kappa": 0.10, "n_s": 0.01, "n_b": 0.01,
        "note": "almost no background noise - the classical benchmark "
                "is already strong and the QI advantage vanishes",
    },
    "bright_signal": {
        "kappa": 0.10, "n_s": 5.0, "n_b": 20.0,
        "note": "many photons per mode - entanglement is no longer "
                "the scarce resource and the advantage fades",
    },
    "faint_everything": {
        "kappa": 0.001, "n_s": 0.001, "n_b": 100.0,
        "note": "near-zero reflectivity into a very loud background - "
                "both exponents are so small the mode counts explode",
    },
}


def scenario_report(name):
    spec = SCENARIOS[name]
    report = exponent_report(spec["kappa"], spec["n_s"], spec["n_b"])
    report["scenario"] = name
    report["note"] = spec["note"]
    report["regime_label"] = model_label(spec["n_s"], spec["n_b"], spec["kappa"])
    return report


def scenario_book():
    return {name: scenario_report(name) for name in SCENARIOS}


# ================================================================ 3. kNN regime classifier
# Features: signal brightness N_S, background brightness N_B,
# reflectivity kappa. The training grid is labelled by the lab's own
# regime model (model_label below), which reads the computed
# exponents - the classifier interpolates the model, it does not
# replace it.

def model_label(n_s, n_b, kappa):
    """The regime model distilled to a three-way teaching label.

    hopeless        - even the classical benchmark needs more than
                      HOPELESS_MODE_LIMIT modes for P_err = 1e-6
    qi-advantaged   - exponent advantage A >= 2 at a reachable point
    classical-equal - reachable, but A < 2: no meaningful advantage
    """
    m_ci = modes_needed(TARGET_P_ERROR, "classical", kappa, n_s, n_b)
    if m_ci is None or m_ci > HOPELESS_MODE_LIMIT:
        return "hopeless"
    if advantage_factor(n_s, n_b) >= 2.0:
        return "qi-advantaged"
    return "classical-equal"


def _training_grid():
    rows = []
    for n_s in (0.001, 0.01, 0.1, 1.0, 10.0):
        for n_b in (0.01, 0.1, 1.0, 10.0, 100.0):
            for kappa in (0.001, 0.01, 0.1, 0.5):
                rows.append(((n_s, n_b, kappa),
                             model_label(n_s, n_b, kappa)))
    return rows


TRAINING_GRID = _training_grid()


def _scaled(features):
    # All three features span decades, so all three are log-scaled;
    # a linear kappa axis would let the brightness axes outvote the
    # reflectivity corner where the hopeless region lives.
    n_s, n_b, kappa = features
    return (math.log10(n_s) / 3.0, math.log10(n_b) / 2.0,
            math.log10(max(kappa, 1e-4)) / 3.0)


def classify_regime(n_s, n_b, kappa, k=5):
    """kNN over the model-labelled grid, deterministic tie-breaking."""
    target = _scaled((n_s, n_b, kappa))
    ranked = []
    for features, label in TRAINING_GRID:
        point = _scaled(features)
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(target, point)))
        ranked.append((dist, features, label))
    ranked.sort(key=lambda row: (row[0], row[1]))
    votes = {}
    for _, _, label in ranked[:k]:
        votes[label] = votes.get(label, 0) + 1
    priority = {"qi-advantaged": 0, "classical-equal": 1, "hopeless": 2}
    return sorted(votes.items(), key=lambda kv: (-kv[1], priority[kv[0]]))[0][0]


def classifier_report():
    return {name: classify_regime(spec["n_s"], spec["n_b"], spec["kappa"])
            for name, spec in SCENARIOS.items()}


# ================================================================ 4. Channel honesty
# The part of quantum illumination that marketing leaves out, kept
# as structured data so the demo, the README, and the checks all
# read the same words.

def channel_honesty():
    """What the lossy noisy channel does to the entanglement (LABELLED)."""
    return {
        "return_state_entangled_with_idler": False,
        "explanation": (
            "The signal reflects off the target region, keeps only a "
            "fraction kappa of its energy, and mixes with background "
            "noise N_B >> 1. That lossy, noisy channel is "
            "entanglement-breaking in the QI regime: the returned "
            "mode and the retained idler are NOT entangled. They "
            "retain quantum-born correlations (a separable but "
            "non-classically-correlated joint state), and the error-"
            "exponent advantage survives in those correlations - "
            "which is why the receiver must measure return and idler "
            "JOINTLY. Any account that says 'the radar detects via "
            "surviving entanglement' is wrong, and this lab does not "
            "say it."
        ),
        "receivers": [
            "homodyne on the return alone = the classical benchmark "
            "(no idler used)",
            "phase-conjugation / OPA receiver = captures half the "
            "exponent gain (3 dB of 6 dB, LABELLED)",
            "optimal joint measurement = reaches the full Chernoff "
            "exponent; no practical receiver is known (LABELLED)",
        ],
        "microwave_vs_optical": (
            "LABELLED practicality note: at microwave frequencies a "
            "room-temperature background carries hundreds to "
            "thousands of thermal photons per mode, so N_B >> 1 is "
            "natural - but cryogenics-grade sources and receivers "
            "are the price. At optical frequencies room-temperature "
            "N_B is near zero, so the N_B >> 1 regime the advantage "
            "needs does not occur naturally. The regime is a place, "
            "not a promise."
        ),
        "what_was_built": (
            "Nothing physical. This lab evaluated labelled formulas "
            "for M, kappa, N_S, N_B. No source, no detector, no "
            "target, no detection."
        ),
    }


# ================================================================ 5. Advisory seats
def _trace(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def argon4_decide(topic="quantumradar-demo"):
    """Gemini Argon 4 DECIDER seat: MAP->MODEL->STRESS->DECIDE.

    Names-only keys, live calls 0, tighten-only: can add HOLDs, never lift one.
    """
    stages = ["MAP", "MODEL", "STRESS", "DECIDE"]
    payload = {"seat": "gemini-argon-4", "topic": topic, "stages": stages}
    return {
        "seat": "Gemini Argon 4 Extended Thinking",
        "role": "DECIDER",
        "pipeline": stages,
        "tiers": [2048, 8192, 32768],
        "key": "names-only (GEMINI_API_KEY name checked, value never read)",
        "live_calls": 0,
        "decision": "PROCEED - sandbox formula model only, all military / surveillance / stealth / product gates stay HOLD",
        "can_override_hold": False,
        "trace_sha256_16": _trace(payload),
    }


def claude_advisor(topic="quantumradar-demo"):
    """Muse advisor seat: advises, can never override a HOLD."""
    return {
        "seat": "Muse",
        "role": "ADVISOR",
        "key": "names-only (MODEL_API_KEY name checked, value never read)",
        "live_calls": 0,
        "advice": "Say it in the first line of every report: this is formula arithmetic on a labelled model - no radar was built and nothing was detected - then show the exponents, the regime boundary where the advantage dies, and the sentence about the broken entanglement.",
        "can_override_hold": False,
    }


def google_flow_plan():
    """Google Flow seat: plan-only, nothing rendered, nothing live."""
    return {
        "seat": "Google Flow",
        "role": "PLAN-ONLY",
        "shots": ["twin photon streams leaving one source, signal and idler holding hands",
                  "the signal stream thinning to a faint thread labelled kappa",
                  "noise static swallowing the return thread while the idler waits",
                  "two error curves racing down a log-scale slope, one four times steeper",
                  "the advantage dial sweeping from 4 back to 1 as the background quiets"],
        "rendered": False,
        "live": 0,
    }


def brains_report():
    return {"decider": argon4_decide(), "advisor": claude_advisor(),
            "flow": google_flow_plan()}


# ================================================================ checks
def run_checks():
    c = []

    def chk(name, ok):
        c.append((name, bool(ok)))

    # 1. Gaussian tools
    chk("q_function_zero", q_function(0.0) == 0.5)
    chk("q_function_one", abs(q_function(1.0) - 0.15865525393145707) < 1e-12)
    chk("q_function_monotone_tail",
        q_function(5.0) < q_function(3.0) < q_function(1.0))
    chk("q_inverse_roundtrip", abs(q_function(q_inverse(1e-6)) - 1e-6) < 1e-12)
    chk("q_inverse_order", q_inverse(1e-3) < q_inverse(1e-6))

    # 2. exponents and the advantage's regime boundary
    xi_demo = classical_exponent(DEMO_KAPPA, DEMO_N_S, DEMO_N_B)
    chk("classical_exponent_demo_value", abs(xi_demo - 1.21969e-05) < 1e-9)
    chk("exponent_scales_linear_kappa",
        abs(classical_exponent(0.2, 0.01, 20.0) - 2.0 * xi_demo) < 1e-15)
    chk("exponent_scales_linear_ns",
        abs(classical_exponent(0.1, 0.02, 20.0) - 2.0 * xi_demo) < 1e-15)
    chk("classical_exponent_high_noise_limit",
        abs(classical_exponent(0.1, 0.01, 1e6) / (0.1 * 0.01 / (4.0 * 1e6))
            - 1.0) < 0.01)
    chk("classical_exponent_zero_kappa",
        classical_exponent(0.0, 0.01, 20.0) == 0.0)
    ratio_deep = advantage_factor(0.001, 100.0)
    chk("advantage_deep_regime_is_4x", 3.9 < ratio_deep < 4.0)
    ratio_demo = advantage_factor(DEMO_N_S, DEMO_N_B)
    chk("advantage_demo_regime", 3.5 < ratio_demo < 4.0)
    chk("advantage_low_noise_vanishes",
        abs(advantage_factor(0.01, 0.001) - 1.0) < 0.01)
    chk("advantage_bright_signal_fades",
        advantage_factor(10.0, 20.0) < 1.3 < ratio_demo)
    ladder = [advantage_factor(0.01, nb)
              for nb in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)]
    chk("advantage_monotone_in_background",
        all(a < b for a, b in zip(ladder, ladder[1:])))
    chk("qi_exponent_ratio_matches_advantage",
        abs(qi_exponent(DEMO_KAPPA, DEMO_N_S, DEMO_N_B) / xi_demo
            - ratio_demo) < 1e-9)
    chk("pc_receiver_between",
        xi_demo < pc_exponent(DEMO_KAPPA, DEMO_N_S, DEMO_N_B)
        < qi_exponent(DEMO_KAPPA, DEMO_N_S, DEMO_N_B))

    # 3. error probabilities + mode counts
    chk("perr_classical_demo_1e5",
        abs(error_prob_classical(10 ** 5, DEMO_KAPPA, DEMO_N_S, DEMO_N_B)
            - 0.05916) < 0.001)
    chk("perr_qi_demo_1e5",
        abs(error_prob_qi(10 ** 5, DEMO_KAPPA, DEMO_N_S, DEMO_N_B)
            - 0.004686) < 0.0005)
    chk("perr_decreases_with_modes",
        error_prob_classical(10 ** 6, DEMO_KAPPA, DEMO_N_S, DEMO_N_B)
        < error_prob_classical(10 ** 5, DEMO_KAPPA, DEMO_N_S, DEMO_N_B)
        and error_prob_qi(10 ** 6, DEMO_KAPPA, DEMO_N_S, DEMO_N_B)
        < error_prob_qi(10 ** 5, DEMO_KAPPA, DEMO_N_S, DEMO_N_B))
    chk("qi_wins_at_large_m",
        error_prob_qi(10 ** 6, DEMO_KAPPA, DEMO_N_S, DEMO_N_B)
        < error_prob_classical(10 ** 6, DEMO_KAPPA, DEMO_N_S, DEMO_N_B))
    m_ci = modes_needed(TARGET_P_ERROR, "classical",
                        DEMO_KAPPA, DEMO_N_S, DEMO_N_B)
    m_qi = modes_needed(TARGET_P_ERROR, "qi",
                        DEMO_KAPPA, DEMO_N_S, DEMO_N_B)
    chk("modes_needed_values", m_ci == 926259 and m_qi == 280992)
    chk("modes_qi_fewer_in_regime", 2.5 < (m_ci / m_qi) < 4.5)
    chk("modes_needed_hits_target",
        error_prob_classical(m_ci, DEMO_KAPPA, DEMO_N_S, DEMO_N_B)
        <= TARGET_P_ERROR
        and error_prob_qi(m_qi, DEMO_KAPPA, DEMO_N_S, DEMO_N_B)
        <= TARGET_P_ERROR)
    chk("modes_needed_minimal",
        error_prob_classical(m_ci - 1, DEMO_KAPPA, DEMO_N_S, DEMO_N_B)
        > TARGET_P_ERROR
        and error_prob_qi(m_qi - 1, DEMO_KAPPA, DEMO_N_S, DEMO_N_B)
        > TARGET_P_ERROR)
    chk("modes_needed_zero_kappa_unreachable",
        modes_needed(TARGET_P_ERROR, "classical", 0.0, 0.01, 20.0) is None
        and modes_needed(TARGET_P_ERROR, "qi", 0.0, 0.01, 20.0) is None)
    rows = roc_table()
    chk("roc_table_shape", len(rows) == 4
        and [r["modes"] for r in rows] == [10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6])
    chk("roc_qi_wins_in_tail_rows",
        all(r["p_error_qi"] < r["p_error_classical"] for r in rows[2:]))

    # 4. scenarios + classifier - labels come from the model itself
    book = scenario_book()
    chk("scenario_book_complete", set(book) == set(SCENARIOS))
    chk("scenario_textbook_label", book["textbook_regime"]["regime_label"]
        == "qi-advantaged")
    chk("scenario_quiet_label", book["quiet_background"]["regime_label"]
        == "classical-equal")
    chk("scenario_bright_label", book["bright_signal"]["regime_label"]
        == "classical-equal")
    chk("scenario_faint_label", book["faint_everything"]["regime_label"]
        == "hopeless")
    rep = classifier_report()
    chk("classifier_textbook", rep["textbook_regime"] == "qi-advantaged")
    chk("classifier_quiet", rep["quiet_background"] == "classical-equal")
    chk("classifier_bright", rep["bright_signal"] == "classical-equal")
    chk("classifier_faint", rep["faint_everything"] == "hopeless")
    chk("classifier_matches_model",
        all(classify_regime(*features) == model_label(*features)
            for features in [(0.01, 20.0, 0.1), (0.01, 0.01, 0.1),
                             (5.0, 20.0, 0.1), (0.001, 100.0, 0.001),
                             (0.1, 10.0, 0.5), (1.0, 1.0, 0.1)]))
    chk("training_grid_labelled_by_model",
        all(label == model_label(*features)
            for features, label in TRAINING_GRID) and len(TRAINING_GRID) == 100)

    # 5. channel honesty - the load-bearing sentences, as data
    honesty = channel_honesty()
    chk("return_not_entangled",
        honesty["return_state_entangled_with_idler"] is False)
    chk("honesty_names_joint_measurement",
        "JOINTLY" in honesty["explanation"])
    chk("honesty_no_detection_claim",
        "No source, no detector, no target, no detection"
        in honesty["what_was_built"])

    # gates + seats
    chk("hold_gates_14", len(HOLD_GATES) == 14
        and all(v.startswith("HOLD") for v in HOLD_GATES.values()))
    chk("gate_military_hold", gate("military_targeting_use") == "HOLD")
    chk("gate_surveillance_hold", gate("surveillance_use") == "HOLD")
    chk("gate_stealth_hold", gate("stealth_detection_claim") == "HOLD")
    chk("gate_weapons_hold", gate("weapons_guidance_use") == "HOLD")
    chk("gate_construction_hold", gate("real_radar_construction") == "HOLD")
    chk("gate_sandbox_proceed", gate("run_simulation") == "PROCEED")
    brains = brains_report()
    chk("argon4_no_override", brains["decider"]["can_override_hold"] is False
        and brains["decider"]["seat"]
        == "Gemini Argon 4 Extended Thinking")
    chk("claude_no_override", brains["advisor"]["can_override_hold"] is False)
    chk("flow_plan_only", brains["flow"]["rendered"] is False
        and brains["flow"]["live"] == 0)
    return c


# ================================================================ CLI / demo
def demo():
    print("=" * 68)
    print("QuantumRadar-Lab - quantum illumination / quantum radar lab")
    print("by Darion Kellar | MIT | pure stdlib | zero network | EDUCATION ONLY")
    print("=" * 68)
    print("\n[1] THE HONEST CORE - what this lab is and is not")
    print("    This lab does FORMULA ARITHMETIC on a labelled textbook")
    print("    model (Lloyd 2008; Tan et al. 2008). No radar was built,")
    print("    no photon was sent, no target was detected. The model")
    print("    outputs error exponents and error probabilities - never")
    print("    a range, a resolution, or a detection.")
    print("\n[2] The labelled model - target present or absent?")
    print("    kappa = reflectivity (labelled), N_S = signal photons per")
    print("    mode, N_B = background photons per mode, M = modes.")
    print("    xi_CI = kappa * N_S * ( sqrt(N_B+1) - sqrt(N_B) )^2")
    print("    P_CI(M) = Q( sqrt(2 * M * xi_CI) )   [Gaussian Q-function]")
    print("    xi_QI = A * xi_CI,  A = 1 + 3 * [N_B/(N_B+1)] / (1+N_S)")
    print("    P_QI(M) = 0.5 * exp(-M * xi_QI)      [Chernoff form]")
    print("    A -> 4 (6 dB) only when N_S << 1 and N_B >> 1; A -> 1")
    print("    when the background quiets or the signal brightens. The")
    print("    interpolation is labelled; the endpoints are the papers'.")
    rep = exponent_report()
    print(f"\n[3] Demo operating point - kappa {DEMO_KAPPA}, "
          f"N_S {DEMO_N_S}, N_B {DEMO_N_B:g} (the textbook regime)")
    print(f"    xi classical per mode : {rep['xi_classical_per_mode']:.6e}")
    print(f"    xi PC receiver / mode : {rep['xi_pc_receiver_per_mode']:.6e}"
          "  (labelled: half the gain)")
    print(f"    xi QI per mode        : {rep['xi_qi_per_mode']:.6e}")
    print(f"    advantage factor A    : {rep['advantage_factor']:.4f} "
          f"({rep['advantage_db']:.2f} dB in exponent)")
    tables = regime_boundary_tables()
    print("    regime boundary - A vs background (N_S = 0.01): "
          + ", ".join(f"{r['n_b']:g}:{r['advantage']:.2f}"
                      for r in tables["advantage_vs_background_at_ns_0.01"]))
    print("    regime boundary - A vs signal (N_B = 20):     "
          + ", ".join(f"{r['n_s']:g}:{r['advantage']:.2f}"
                      for r in tables["advantage_vs_signal_at_nb_20"]))
    print("\n[4] ROC-style table - P_error vs modes M (demo point)")
    print("    M            P classical     P quantum-illum.")
    for row in roc_table():
        print(f"    {row['modes']:<12d} {row['p_error_classical']:.4e}    "
              f"{row['p_error_qi']:.4e}")
    print("    -> at small M the Chernoff form is loose and sits above")
    print("       the exact Gaussian number; the advantage is an")
    print("       exponent, and it takes over as M grows.")
    print(f"\n[5] Modes needed for P_error = {TARGET_P_ERROR:g}")
    print(f"    classical : {rep['modes_classical_for_1e-6']:,} modes")
    print(f"    QI        : {rep['modes_qi_for_1e-6']:,} modes "
          f"({rep['modes_classical_for_1e-6'] / rep['modes_qi_for_1e-6']:.2f}x fewer)")
    print("\n[6] Scenario book + regime classifier (TinyML kNN, labels "
          "from the model)")
    book = scenario_book()
    labels = classifier_report()
    for name, report in book.items():
        print(f"    {name:18s} A {report['advantage_factor']:.3f} | "
              f"model {report['regime_label']:16s} | kNN {labels[name]}")
    print("\n[7] Entanglement honesty - read before quoting the 6 dB")
    honesty = channel_honesty()
    print(f"    return entangled with idler: "
          f"{honesty['return_state_entangled_with_idler']}")
    print("    The lossy noisy channel BREAKS the entanglement. The")
    print("    advantage survives in quantum-born correlations and")
    print("    needs a JOINT measurement of return + idler.")
    for line in honesty["receivers"]:
        print(f"    - {line}")
    print(f"\n[8] HOLD gates: {len(HOLD_GATES)}")
    for key in HOLD_GATES:
        print(f"    HOLD  {key}")
    print("\n[9] Advisory seats")
    brains = brains_report()
    print(f"    decider: {brains['decider']['decision']} "
          f"trace={brains['decider']['trace_sha256_16']}")
    print(f"    advisor can_override_hold = {brains['advisor']['can_override_hold']}")
    print(f"    flow rendered = {brains['flow']['rendered']} "
          f"live = {brains['flow']['live']}")
    checks = run_checks()
    passed = sum(1 for _, ok in checks if ok)
    print(f"\nCHECKS {passed}/{len(checks)}")
    for name, ok in checks:
        if not ok:
            print(f"    FAIL {name}")
    verdict = "PROCEED" if passed == len(checks) else "HOLD"
    print(f"VERDICT {verdict} (sandbox simulation only) | HOLD gates {len(HOLD_GATES)}")
    return 0 if passed == len(checks) else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "demo":
        return demo()
    cmd = argv[0]
    if cmd == "checks":
        checks = run_checks()
        for name, ok in checks:
            print(f"{'PASS' if ok else 'FAIL'} {name}")
        print(f"CHECKS {sum(1 for _, ok in checks if ok)}/{len(checks)}")
        return 0 if all(ok for _, ok in checks) else 1
    if cmd == "roc":
        print(json.dumps({"operating_point": {
            "kappa": DEMO_KAPPA, "n_s": DEMO_N_S, "n_b": DEMO_N_B},
            "rows": roc_table()}, indent=2, sort_keys=True))
        return 0
    if cmd == "modes":
        print(json.dumps(exponent_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "regimes":
        print(json.dumps({"scenarios": scenario_book(),
                          "boundaries": regime_boundary_tables()},
                         indent=2, sort_keys=True))
        return 0
    if cmd == "honesty":
        print(json.dumps(channel_honesty(), indent=2, sort_keys=True))
        return 0
    if cmd == "classify":
        print(json.dumps(classifier_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "brains":
        print(json.dumps(brains_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "holds":
        for key, value in HOLD_GATES.items():
            print(f"{key}: {value}")
        return 0
    if cmd == "hold-demo":
        action = argv[1] if len(argv) > 1 else "military_targeting_use"
        print(f"{action}: {HOLD_GATES.get(action, 'PROCEED - not gated')}")
        return 2 if gate(action) == "HOLD" else 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
