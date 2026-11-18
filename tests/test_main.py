"""Tests for QuantumRadar-Lab: model, exponents, modes, gates, seats."""
import pytest

import main as m


# ---- Gaussian tools
def test_q_function_zero():
    assert m.q_function(0.0) == 0.5


def test_q_function_one():
    assert abs(m.q_function(1.0) - 0.15865525393145707) < 1e-12


def test_q_function_monotone():
    assert m.q_function(5.0) < m.q_function(3.0) < m.q_function(1.0)


def test_q_inverse_roundtrip():
    assert abs(m.q_function(m.q_inverse(1e-6)) - 1e-6) < 1e-12


def test_q_inverse_rejects_bad_p():
    with pytest.raises(ValueError):
        m.q_inverse(0.75)


# ---- exponents
def test_classical_exponent_demo_value():
    assert abs(m.classical_exponent(0.1, 0.01, 20.0) - 1.21969e-05) < 1e-9


def test_exponent_linear_in_kappa_and_ns():
    base = m.classical_exponent(0.1, 0.01, 20.0)
    assert abs(m.classical_exponent(0.2, 0.01, 20.0) - 2 * base) < 1e-15
    assert abs(m.classical_exponent(0.1, 0.02, 20.0) - 2 * base) < 1e-15


def test_exponent_high_noise_limit():
    xi = m.classical_exponent(0.1, 0.01, 1e6)
    assert abs(xi / (0.1 * 0.01 / (4.0 * 1e6)) - 1.0) < 0.01


def test_exponent_zero_kappa():
    assert m.classical_exponent(0.0, 0.01, 20.0) == 0.0


def test_exponent_param_validation():
    with pytest.raises(ValueError):
        m.classical_exponent(1.5, 0.01, 20.0)
    with pytest.raises(ValueError):
        m.classical_exponent(0.1, -0.01, 20.0)


# ---- the advantage and its regime boundary
def test_advantage_deep_regime_is_4x():
    assert 3.9 < m.advantage_factor(0.001, 100.0) < 4.0


def test_advantage_demo_regime():
    assert 3.5 < m.advantage_factor(0.01, 20.0) < 4.0


def test_advantage_vanishes_in_low_noise():
    assert abs(m.advantage_factor(0.01, 0.001) - 1.0) < 0.01


def test_advantage_fades_for_bright_signal():
    assert m.advantage_factor(10.0, 20.0) < 1.3


def test_advantage_monotone_in_background():
    ladder = [m.advantage_factor(0.01, nb)
              for nb in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)]
    assert all(a < b for a, b in zip(ladder, ladder[1:]))


def test_qi_exponent_ratio():
    ratio = m.qi_exponent(0.1, 0.01, 20.0) / m.classical_exponent(0.1, 0.01, 20.0)
    assert abs(ratio - m.advantage_factor(0.01, 20.0)) < 1e-9


def test_pc_receiver_sits_between():
    ci = m.classical_exponent(0.1, 0.01, 20.0)
    qi = m.qi_exponent(0.1, 0.01, 20.0)
    assert ci < m.pc_exponent(0.1, 0.01, 20.0) < qi


# ---- error probabilities + mode counts
def test_perr_classical_demo():
    assert abs(m.error_prob_classical(10 ** 5, 0.1, 0.01, 20.0) - 0.05916) < 0.001


def test_perr_qi_demo():
    assert abs(m.error_prob_qi(10 ** 5, 0.1, 0.01, 20.0) - 0.004686) < 0.0005


def test_perr_decreases_with_modes():
    assert (m.error_prob_classical(10 ** 6, 0.1, 0.01, 20.0)
            < m.error_prob_classical(10 ** 5, 0.1, 0.01, 20.0))
    assert (m.error_prob_qi(10 ** 6, 0.1, 0.01, 20.0)
            < m.error_prob_qi(10 ** 5, 0.1, 0.01, 20.0))


def test_modes_needed_demo_values():
    assert m.modes_needed(1e-6, "classical", 0.1, 0.01, 20.0) == 926259
    assert m.modes_needed(1e-6, "qi", 0.1, 0.01, 20.0) == 280992


def test_modes_qi_fewer_in_regime():
    ratio = (m.modes_needed(1e-6, "classical", 0.1, 0.01, 20.0)
             / m.modes_needed(1e-6, "qi", 0.1, 0.01, 20.0))
    assert 2.5 < ratio < 4.5


def test_modes_needed_is_minimal():
    m_ci = m.modes_needed(1e-6, "classical", 0.1, 0.01, 20.0)
    m_qi = m.modes_needed(1e-6, "qi", 0.1, 0.01, 20.0)
    assert m.error_prob_classical(m_ci, 0.1, 0.01, 20.0) <= 1e-6
    assert m.error_prob_classical(m_ci - 1, 0.1, 0.01, 20.0) > 1e-6
    assert m.error_prob_qi(m_qi, 0.1, 0.01, 20.0) <= 1e-6
    assert m.error_prob_qi(m_qi - 1, 0.1, 0.01, 20.0) > 1e-6


def test_modes_needed_unreachable_at_zero_kappa():
    assert m.modes_needed(1e-6, "classical", 0.0, 0.01, 20.0) is None
    assert m.modes_needed(1e-6, "qi", 0.0, 0.01, 20.0) is None


def test_roc_table_shape_and_tail():
    rows = m.roc_table()
    assert [r["modes"] for r in rows] == [10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]
    assert all(r["p_error_qi"] < r["p_error_classical"] for r in rows[2:])


# ---- scenarios + classifier
def test_scenario_labels_from_model():
    book = m.scenario_book()
    assert book["textbook_regime"]["regime_label"] == "qi-advantaged"
    assert book["quiet_background"]["regime_label"] == "classical-equal"
    assert book["bright_signal"]["regime_label"] == "classical-equal"
    assert book["faint_everything"]["regime_label"] == "hopeless"


def test_classifier_report_matches_scenarios():
    rep = m.classifier_report()
    assert rep["textbook_regime"] == "qi-advantaged"
    assert rep["quiet_background"] == "classical-equal"
    assert rep["bright_signal"] == "classical-equal"
    assert rep["faint_everything"] == "hopeless"


def test_classifier_matches_model_on_probes():
    probes = [(0.01, 20.0, 0.1), (0.01, 0.01, 0.1), (5.0, 20.0, 0.1),
              (0.001, 100.0, 0.001), (0.1, 10.0, 0.5), (1.0, 1.0, 0.1)]
    for features in probes:
        assert m.classify_regime(*features) == m.model_label(*features)


def test_training_grid_labelled_by_model():
    assert len(m.TRAINING_GRID) == 100
    assert all(label == m.model_label(*f) for f, label in m.TRAINING_GRID)


# ---- channel honesty
def test_return_is_not_entangled():
    assert m.channel_honesty()["return_state_entangled_with_idler"] is False


def test_honesty_states_no_detection():
    assert "No source, no detector, no target, no detection" \
        in m.channel_honesty()["what_was_built"]


# ---- HOLD gates
def test_hold_gates_14():
    assert len(m.HOLD_GATES) == 14
    assert all(v.startswith("HOLD") for v in m.HOLD_GATES.values())


def test_military_and_surveillance_gates():
    assert m.gate("military_targeting_use") == "HOLD"
    assert m.gate("surveillance_use") == "HOLD"
    assert m.gate("weapons_guidance_use") == "HOLD"
    assert m.gate("air_defense_application") == "HOLD"


def test_claim_gates():
    assert m.gate("stealth_detection_claim") == "HOLD"
    assert m.gate("drone_detection_claim") == "HOLD"
    assert m.gate("range_or_performance_claim") == "HOLD"
    assert m.gate("advantage_marketing_claim") == "HOLD"


def test_construction_and_live_gates():
    assert m.gate("real_radar_construction") == "HOLD"
    assert m.gate("live_target_detection") == "HOLD"
    assert m.gate("hardware_control") == "HOLD"


def test_sandbox_proceeds():
    assert m.gate("run_simulation") == "PROCEED"


def test_hold_demo_exit_code():
    assert m.main(["hold-demo", "stealth_detection_claim"]) == 2
    assert m.main(["hold-demo", "run_simulation"]) == 0


# ---- council seats
def test_argon4_decider():
    seat = m.brains_report()["decider"]
    assert seat["seat"] == "Gemini Argon 4 Extended Thinking"
    assert seat["role"] == "DECIDER"
    assert seat["can_override_hold"] is False
    assert seat["live_calls"] == 0
    assert len(seat["trace_sha256_16"]) == 16


def test_claude_advisor_cannot_override():
    assert m.brains_report()["advisor"]["can_override_hold"] is False


def test_flow_plan_only():
    flow = m.brains_report()["flow"]
    assert flow["rendered"] is False and flow["live"] == 0


# ---- whole-suite invariants
def test_checks_all_pass():
    checks = m.run_checks()
    assert all(ok for _, ok in checks), [n for n, ok in checks if not ok]


def test_demo_exit_zero(capsys):
    assert m.main([]) == 0
    out = capsys.readouterr().out
    assert "CHECKS 52/52" in out
    assert "VERDICT PROCEED" in out
