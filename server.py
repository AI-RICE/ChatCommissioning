"""
MCP server for IPMSM drive commissioning.

Exposes the drive simulator and FCS-MPC controller as MCP tools and resources,
allowing an AI agent to commission the drive by reading state, configuring
controllers, running scenarios, and analysing results.

Tools  — actions that change state or run the simulation.
Resources — read-only reference data the agent uses for reasoning.

Run with:
    python server.py          (stdio transport, for Claude Desktop / claude CLI)
    python server.py --sse    (SSE transport, for web clients)
"""

import io
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import asdict
from typing import Any

from mcp.server.fastmcp import FastMCP, Image

from simulator.ipmsm import IPMSMParameters, IPMSMSimulator, mtpa, field_weakening, torque, flux_linkages
from controller.fcs_mpc import (
    FCSMPCController, FCSMPCParameters, FCSMPCWeights,
    count_transitions, _SWITCHING_STATES,
)

# ---------------------------------------------------------------------------
# Shared drive state (single-session; one simulator + one controller)
# ---------------------------------------------------------------------------

_motor_params = IPMSMParameters()
_sim = IPMSMSimulator(_motor_params)
_ctrl_weights = FCSMPCWeights()
_ctrl_params = FCSMPCParameters(horizon=1, weights=_ctrl_weights)
_ctrl = FCSMPCController(_motor_params, _ctrl_params)

_i_d_ref: float = 0.0
_i_q_ref: float = 0.0
_T_load: float = 0.0
_history: list[dict] = []

# Outer speed-loop PI gains (used by run_load_profile when speed control is active).
# Stored here so set_controller_weights can update them and get_controller_config
# can report them; run_load_profile reads from these unless overridden.
_speed_kp: float = 5.0    # [A/(rad/s)]
_speed_ki: float = 100.0  # [A/rad]


def _state_dict(state, action=None) -> dict:
    """Convert DriveState (+ optional ControlAction) to a JSON-safe dict."""
    d = {
        "time_s":      round(state.time, 6),
        "i_d_A":       round(state.i_d, 4),
        "i_q_A":       round(state.i_q, 4),
        "i_mag_A":     round(state.i_mag, 4),
        "omega_r_rads": round(state.omega_r, 4),
        "speed_rpm":   round(state.speed_rpm, 2),
        "theta_r_rad": round(state.theta_r, 4),
        "T_e_Nm":      round(state.T_e, 4),
        "T_load_Nm":   round(state.T_load, 4),
        "v_d_V":       round(state.v_d, 3),
        "v_q_V":       round(state.v_q, 3),
    }
    if action is not None:
        d["switching_state"] = action.switching_state.tolist()
        d["cost"] = round(action.cost, 6)
    return d


def _compute_metrics(history: list[dict], ref_key: str, ref_val: float) -> dict:
    """
    Compute commissioning metrics from a history list.
    ref_key: e.g. 'i_q_A' or 'speed_rpm'
    ref_val: the target setpoint value
    """
    if len(history) < 10:
        return {}

    vals = np.array([h[ref_key] for h in history])
    times = np.array([h["time_s"] for h in history])
    dt = times[1] - times[0]

    # Rise time: 10% → 90% of final value
    v0, vf = vals[0], vals[-1]
    span = vf - v0
    if abs(span) < 1e-9:
        return {"context": "Signal did not change — check reference and controller."}

    t10 = next((times[i] for i, v in enumerate(vals) if abs(v - v0) >= 0.1 * abs(span)), None)
    t90 = next((times[i] for i, v in enumerate(vals) if abs(v - v0) >= 0.9 * abs(span)), None)
    rise_time = round(t90 - t10, 5) if (t10 and t90) else None

    # Overshoot (relative to reference)
    if abs(ref_val) > 1e-9:
        overshoot_pct = round(100.0 * (vals.max() - ref_val) / abs(ref_val), 2) if span > 0 \
                   else round(100.0 * (ref_val - vals.min()) / abs(ref_val), 2)
    else:
        overshoot_pct = 0.0

    # Settling time (within 2% of final value)
    band = 0.02 * abs(ref_val) if abs(ref_val) > 1e-9 else 0.02
    settled_idx = len(vals) - 1
    for i in range(len(vals) - 1, -1, -1):
        if abs(vals[i] - ref_val) > band:
            settled_idx = i + 1
            break
    settling_time = round(times[min(settled_idx, len(times)-1)], 5)

    # Ripple (std of last 20% of signal)
    tail = vals[int(0.8 * len(vals)):]
    ripple = round(float(np.std(tail)), 5)

    return {
        "rise_time_s": rise_time,
        "overshoot_pct": overshoot_pct,
        "settling_time_s": settling_time,
        "steady_state_ripple": ripple,
        "final_value": round(float(vals[-1]), 4),
        "reference": ref_val,
    }


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "IPMSM Drive Commissioning",
    instructions=(
        "You are commissioning an IPMSM servo drive. "
        "Use the tools to configure motor parameters, set controller weights, "
        "run scenarios, and analyse results. "
        "Always read drive://capabilities first to know what strategies and "
        "parameters are available. Prefer set_mtpa_reference over "
        "set_current_reference unless a specific i_d is needed. "
        "When suggesting controller changes, explain the expected trade-off "
        "before applying it and confirm with the engineer."
    ),
)


# ===========================================================================
# RESOURCES — read-only reference data
# ===========================================================================

@mcp.resource("drive://capabilities")
def capabilities() -> str:
    """
    Drive capability manifest. Read this first to understand what the drive
    supports before configuring or tuning anything.
    """
    p = _motor_params
    i_d_mtpa, i_q_mtpa = mtpa(p, p.i_max)
    return json.dumps({
        "drive_type": "IPMSM 2-level VSI",
        "control_strategies": [
            {
                "id": "fcs_mpc",
                "name": "Finite Control Set MPC",
                "status": "active",
                "description": (
                    "Selects the inverter switching state that minimises a "
                    "cost function over a prediction horizon. No modulator needed. "
                    "Variable switching frequency. Cost weights are the primary "
                    "commissioning parameters."
                ),
                "commissioning_parameters": {
                    "weight_id_error":     "d-axis current tracking weight (≥0)",
                    "weight_iq_error":     "q-axis current tracking weight (≥0)",
                    "weight_switching":    "switching transition penalty — raise to reduce switching frequency at cost of higher ripple",
                    "weight_torque_ripple":"torque ripple penalty — raise for smoother torque at low speed",
                    "weight_common_mode":  "common-mode voltage penalty — raise to reduce EMI / bearing currents",
                    "horizon":             "prediction steps 1–3 — higher horizon reduces steady-state ripple but increases computation",
                },
            },
        ],
        "reference_modes": [
            {"id": "direct_dq",  "description": "Set i_d_ref and i_q_ref directly"},
            {"id": "mtpa",       "description": "Auto-compute MTPA point for a given current magnitude (recommended for below-base-speed operation)"},
            {"id": "field_weakening", "description": "Auto-compute field-weakening point for given speed and current magnitude"},
        ],
        "motor": p.describe(),
        "operating_limits": {
            "i_max_A":       p.i_max,
            "v_dc_V":        p.v_dc,
            "base_speed_rpm": round(p.base_speed * 60 / (2 * np.pi), 1),
            "rated_torque_Nm": round(p.rated_torque, 2),
            "saliency_ratio": round(p.saliency_ratio, 2),
            "mtpa_at_i_max":  {"i_d_A": round(i_d_mtpa, 3), "i_q_A": round(i_q_mtpa, 3)},
        },
    }, indent=2)


@mcp.resource("drive://motor_parameters")
def motor_parameters_resource() -> str:
    """Current motor parameters with units and engineering description."""
    p = _motor_params
    return json.dumps({
        "R_s":   {"value": p.R_s,   "unit": "Ω",     "description": "Stator resistance — affects current loop bandwidth and loss"},
        "L_d":   {"value": p.L_d,   "unit": "H",     "description": "d-axis inductance — controls d-axis current dynamics"},
        "L_q":   {"value": p.L_q,   "unit": "H",     "description": "q-axis inductance — L_q > L_d gives reluctance torque (saliency)"},
        "psi_f": {"value": p.psi_f, "unit": "Wb",    "description": "PM flux linkage — determines back-EMF constant and base torque"},
        "p":     {"value": p.p,     "unit": "—",     "description": "Pole pairs — relates mechanical to electrical speed (ωe = p·ωr)"},
        "J":     {"value": p.J,     "unit": "kg·m²", "description": "Total rotor + load inertia — determines speed loop bandwidth"},
        "B":     {"value": p.B,     "unit": "N·m·s", "description": "Viscous friction — affects steady-state torque and damping"},
        "i_max": {"value": p.i_max, "unit": "A",     "description": "Peak current limit — thermal and demagnetisation boundary"},
        "v_dc":  {"value": p.v_dc,  "unit": "V",     "description": "DC bus voltage — determines maximum achievable voltage vector"},
        "dt":    {"value": p.dt,    "unit": "s",     "description": "Simulation and control step size"},
    }, indent=2)


@mcp.resource("drive://controller_parameters")
def controller_parameters_resource() -> str:
    """Current FCS-MPC configuration with current reference and cost weights."""
    return json.dumps({
        "current_reference": {
            "i_d_ref_A": _i_d_ref,
            "i_q_ref_A": _i_q_ref,
            "i_mag_A":   round(np.sqrt(_i_d_ref**2 + _i_q_ref**2), 4),
        },
        "fcs_mpc": _ctrl_params.describe(),
        "load_torque_Nm": _T_load,
    }, indent=2)


@mcp.resource("drive://drive_state")
def drive_state_resource() -> str:
    """Current instantaneous drive state."""
    return json.dumps(_state_dict(_sim.state), indent=2)


# ===========================================================================
# TOOLS — actions
# ===========================================================================

# --- Motor configuration ---------------------------------------------------

@mcp.tool()
def get_motor_parameters() -> dict:
    """
    Return current motor nameplate and electrical parameters.
    Use this before tuning to confirm which motor is loaded.
    """
    p = _motor_params
    return {
        "parameters": p.describe(),
        "context": (
            f"Saliency ratio L_q/L_d = {p.saliency_ratio:.2f} — "
            f"{'significant reluctance torque available, MTPA will use negative i_d' if p.saliency_ratio > 1.2 else 'low saliency, MTPA ≈ i_d=0'}. "
            f"Base speed {round(p.base_speed * 60 / (2*np.pi), 0)} RPM, "
            f"rated torque {round(p.rated_torque, 2)} N·m."
        ),
    }


@mcp.tool()
def set_motor_parameters(
    R_s: float | None = None,
    L_d: float | None = None,
    L_q: float | None = None,
    psi_f: float | None = None,
    p: int | None = None,
    J: float | None = None,
    B: float | None = None,
    i_max: float | None = None,
    v_dc: float | None = None,
    k_sat_d: float | None = None,
    k_sat_q: float | None = None,
) -> dict:
    """
    Update motor parameters. Only supplied fields are changed.
    Resets the simulator and controller after any change.

    Args:
        R_s:     Stator resistance [Ω]
        L_d:     d-axis inductance [H] — must satisfy L_d < L_q for IPMSM
        L_q:     q-axis inductance [H]
        psi_f:   PM flux linkage [Wb]
        p:       Pole pairs (integer)
        J:       Rotor inertia [kg·m²]
        B:       Viscous friction [N·m·s/rad]
        i_max:   Peak current limit [A]
        v_dc:    DC bus voltage [V]
        k_sat_d: d-axis saturation coefficient (0 = linear)
        k_sat_q: q-axis saturation coefficient (0 = linear; cross-coupled with i_d)
    """
    global _motor_params, _sim, _ctrl, _history

    prev = _motor_params
    updated = IPMSMParameters(
        R_s=R_s         if R_s     is not None else prev.R_s,
        L_d=L_d         if L_d     is not None else prev.L_d,
        L_q=L_q         if L_q     is not None else prev.L_q,
        psi_f=psi_f     if psi_f   is not None else prev.psi_f,
        p=p             if p       is not None else prev.p,
        J=J             if J       is not None else prev.J,
        B=B             if B       is not None else prev.B,
        i_max=i_max     if i_max   is not None else prev.i_max,
        v_dc=v_dc       if v_dc    is not None else prev.v_dc,
        k_sat_d=k_sat_d if k_sat_d is not None else prev.k_sat_d,
        k_sat_q=k_sat_q if k_sat_q is not None else prev.k_sat_q,
        dt=prev.dt,
    )
    if updated.L_d >= updated.L_q:
        return {"status": "error", "context": "L_d must be < L_q for an IPMSM. Check your values."}

    _motor_params = updated
    _sim = IPMSMSimulator(_motor_params)
    _ctrl = FCSMPCController(_motor_params, _ctrl_params)
    _history = []

    return {
        "status": "ok",
        "parameters": _motor_params.describe(),
        "context": (
            f"Motor parameters updated. Simulator and controller reset. "
            f"New saliency ratio: {_motor_params.saliency_ratio:.2f}, "
            f"base speed: {round(_motor_params.base_speed * 60/(2*np.pi), 0)} RPM."
        ),
    }


# --- Reference setting ------------------------------------------------------

@mcp.tool()
def set_current_reference(i_d_ref: float, i_q_ref: float) -> dict:
    """
    Set dq current references directly.

    Args:
        i_d_ref: d-axis current reference [A] — negative for IPMSM MTPA / field weakening
        i_q_ref: q-axis current reference [A] — positive = motoring torque

    Prefer set_mtpa_reference unless you need a specific i_d (e.g. field weakening test).
    """
    global _i_d_ref, _i_q_ref
    i_mag = np.sqrt(i_d_ref**2 + i_q_ref**2)
    if i_mag > _motor_params.i_max:
        return {
            "status": "error",
            "context": f"|i| = {i_mag:.2f} A exceeds i_max = {_motor_params.i_max} A. Reduce reference.",
        }
    _i_d_ref, _i_q_ref = i_d_ref, i_q_ref
    _ctrl.set_reference(i_d_ref, i_q_ref)
    T_ref = torque(_motor_params, i_d_ref, i_q_ref)
    return {
        "status": "ok",
        "i_d_ref_A": i_d_ref,
        "i_q_ref_A": i_q_ref,
        "i_mag_A": round(i_mag, 3),
        "expected_torque_Nm": round(T_ref, 3),
        "context": f"Reference set. Expected torque {T_ref:.2f} N·m.",
    }


@mcp.tool()
def set_mtpa_reference(i_mag: float) -> dict:
    """
    Set current reference to the MTPA (Maximum Torque Per Ampere) point for
    a given current magnitude. Automatically computes the optimal (i_d, i_q)
    that maximises torque for this motor's saliency ratio.

    Args:
        i_mag: Current magnitude [A], must be ≤ i_max.

    Use this as the default reference mode below base speed.
    """
    global _i_d_ref, _i_q_ref
    if i_mag > _motor_params.i_max:
        return {
            "status": "error",
            "context": f"i_mag = {i_mag:.2f} A exceeds i_max = {_motor_params.i_max} A.",
        }
    i_d, i_q = mtpa(_motor_params, i_mag)
    i_d_0tor = 0.0
    T_mtpa = torque(_motor_params, i_d, i_q)
    T_id0  = torque(_motor_params, i_d_0tor, i_mag)
    gain_pct = 100.0 * (T_mtpa / T_id0 - 1.0) if abs(T_id0) > 1e-9 else 0.0

    _i_d_ref, _i_q_ref = i_d, i_q
    _ctrl.set_reference(i_d, i_q)
    return {
        "status": "ok",
        "i_d_ref_A": round(i_d, 4),
        "i_q_ref_A": round(i_q, 4),
        "expected_torque_Nm": round(T_mtpa, 3),
        "torque_gain_vs_id0_pct": round(gain_pct, 1),
        "context": (
            f"MTPA: i_d={i_d:.2f} A, i_q={i_q:.2f} A → {T_mtpa:.2f} N·m. "
            f"This is {gain_pct:.1f}% more torque than naive i_d=0 at the same current."
        ),
    }


@mcp.tool()
def set_field_weakening_reference(omega_r_rads: float, i_mag: float) -> dict:
    """
    Compute and set the field-weakening operating point for operation above base speed.
    Finds the (i_d, i_q) that satisfies both the voltage limit ellipse and current
    limit circle at the given speed.

    Args:
        omega_r_rads: Mechanical rotor speed [rad/s]
        i_mag:        Current magnitude [A], must be ≤ i_max
    """
    global _i_d_ref, _i_q_ref
    p = _motor_params
    if i_mag > p.i_max:
        return {"status": "error", "context": f"i_mag exceeds i_max = {p.i_max} A."}

    i_d, i_q = field_weakening(p, omega_r_rads, i_mag)
    speed_rpm = omega_r_rads * 60 / (2 * np.pi)
    T_ref = torque(p, i_d, i_q)
    in_fw = omega_r_rads > p.base_speed

    _i_d_ref, _i_q_ref = i_d, i_q
    _ctrl.set_reference(i_d, i_q)
    return {
        "status": "ok",
        "i_d_ref_A": round(i_d, 4),
        "i_q_ref_A": round(i_q, 4),
        "expected_torque_Nm": round(T_ref, 3),
        "field_weakening_active": in_fw,
        "context": (
            f"{'Field weakening active' if in_fw else 'Below base speed — MTPA point used'}. "
            f"At {speed_rpm:.0f} RPM: i_d={i_d:.2f} A, i_q={i_q:.2f} A, T_e={T_ref:.2f} N·m."
        ),
    }


# --- Controller configuration -----------------------------------------------

@mcp.tool()
def get_controller_config() -> dict:
    """Return current FCS-MPC cost weights and horizon, outer speed-loop PI gains,
    and the active current reference."""
    return {
        "config": _ctrl_params.describe(),
        "speed_pi": {
            "kp_A_per_rad_s": _speed_kp,
            "ki_A_per_rad":   _speed_ki,
        },
        "reference": {"i_d_ref_A": _i_d_ref, "i_q_ref_A": _i_q_ref},
        "context": (
            "Cost weights determine the inner FCS-MPC trade-offs. "
            "weight_switching > 0 reduces switching frequency at cost of higher ripple. "
            "weight_torque_ripple > 0 smooths torque, useful at low speed. "
            "horizon > 1 reduces steady-state ripple but evaluates more candidates. "
            "Outer speed PI gains apply when run_load_profile is invoked with a "
            "speed reference; tune via this tool, not via run_load_profile args."
        ),
    }


@mcp.tool()
def set_controller_weights(
    id_error: float | None = None,
    iq_error: float | None = None,
    switching: float | None = None,
    torque_ripple: float | None = None,
    common_mode: float | None = None,
    horizon: int | None = None,
    speed_kp: float | None = None,
    speed_ki: float | None = None,
) -> dict:
    """
    Update FCS-MPC cost function weights, prediction horizon, and/or outer
    speed-loop PI gains. Only supplied fields are changed; all weights must
    be ≥ 0.

    Args:
        id_error:      d-axis current tracking weight (FCS-MPC)
        iq_error:      q-axis current tracking weight (FCS-MPC)
        switching:     switching transition penalty — raise to reduce switching frequency
        torque_ripple: torque ripple penalty — raise for smoother torque at low speed
        common_mode:   common-mode voltage penalty — raise to reduce EMI
        horizon:       FCS-MPC prediction horizon (1, 2, or 3)
        speed_kp:      outer speed-loop PI proportional gain [A/(rad/s)]
        speed_ki:      outer speed-loop PI integral gain     [A/rad]

    Trade-offs:
        switching ↑     → fewer transitions, lower losses, but higher current ripple
        torque_ripple ↑ → smoother torque, may conflict with fast current tracking
        horizon ↑       → less ripple, more candidates evaluated (8^horizon per step)
        speed_kp ↑      → faster speed tracking, more overshoot
        speed_ki ↑      → faster zero steady-state error, more overshoot
    """
    global _ctrl_params, _ctrl, _speed_kp, _speed_ki
    prev = _ctrl_params
    w = prev.weights

    new_weights = FCSMPCWeights(
        id_error=id_error           if id_error      is not None else w.id_error,
        iq_error=iq_error           if iq_error      is not None else w.iq_error,
        switching=switching         if switching     is not None else w.switching,
        torque_ripple=torque_ripple if torque_ripple is not None else w.torque_ripple,
        common_mode=common_mode     if common_mode   is not None else w.common_mode,
    )
    for name, val in asdict(new_weights).items():
        if val < 0:
            return {"status": "error", "context": f"Weight '{name}' must be ≥ 0, got {val}."}

    new_horizon = horizon if horizon is not None else prev.horizon
    try:
        _ctrl_params = FCSMPCParameters(horizon=new_horizon, weights=new_weights)
    except ValueError as e:
        return {"status": "error", "context": str(e)}

    _ctrl = FCSMPCController(_motor_params, _ctrl_params)
    _ctrl.set_reference(_i_d_ref, _i_q_ref)

    if speed_kp is not None:
        if speed_kp < 0:
            return {"status": "error", "context": "speed_kp must be ≥ 0."}
        _speed_kp = speed_kp
    if speed_ki is not None:
        if speed_ki < 0:
            return {"status": "error", "context": "speed_ki must be ≥ 0."}
        _speed_ki = speed_ki

    return {
        "status": "ok",
        "config": _ctrl_params.describe(),
        "speed_pi": {"kp_A_per_rad_s": _speed_kp, "ki_A_per_rad": _speed_ki},
        "context": (
            f"Controller updated. horizon={new_horizon}, "
            f"λ_sw={new_weights.switching}, λ_tr={new_weights.torque_ripple}, "
            f"λ_cm={new_weights.common_mode}, "
            f"speed_kp={_speed_kp}, speed_ki={_speed_ki}. "
            f"Run a scenario to observe the effect."
        ),
    }


# --- Simulation scenarios ---------------------------------------------------

@mcp.tool()
def reset_drive() -> dict:
    """Reset the simulator to standstill (zero current, zero speed)."""
    global _history
    _sim.reset()
    _ctrl.reset()
    _history = []
    return {"status": "ok", "context": "Drive reset to standstill."}


@mcp.tool()
def set_load_torque(T_load_Nm: float) -> dict:
    """
    Set the constant load torque applied to the shaft.

    Args:
        T_load_Nm: Load torque [N·m]. Positive = resistive load.
    """
    global _T_load
    _T_load = T_load_Nm
    rated = _motor_params.rated_torque
    return {
        "status": "ok",
        "T_load_Nm": T_load_Nm,
        "as_fraction_of_rated": round(T_load_Nm / rated, 3) if rated > 0 else None,
        "context": f"Load torque set to {T_load_Nm} N·m ({100*T_load_Nm/rated:.0f}% of rated).",
    }


@mcp.tool()
def run_scenario(
    duration_s: float,
    record_every: int = 10,
    reset_first: bool = False,
) -> dict:
    """
    Run the simulator for a given duration with the current reference and load torque.
    Records history and returns performance metrics.

    Args:
        duration_s:   Simulation duration [s]
        record_every: Record one sample every N steps (reduces output size).
                      Default 10 → 1 kHz effective sample rate at dt=100µs.
        reset_first:  If True, reset the simulator and controller to standstill
                      before running. Use for clean verification reads after a
                      tuning sweep that may have warmed the simulator state.

    Returns drive state history and commissioning metrics (rise time, overshoot,
    settling time, ripple, average switching frequency).
    """
    global _history

    if reset_first:
        _sim.reset()
        _ctrl.reset()
        _history = []

    n_steps = int(duration_s / _motor_params.dt)
    if n_steps > 500_000:
        return {"status": "error", "context": "Duration too long. Max ~50 s at dt=100µs."}

    history_raw: list[dict] = []
    sw_transitions: list[int] = []
    prev_sw = _ctrl._prev_sw.copy()

    for k in range(n_steps):
        action = _ctrl.step(_sim.state)
        state = _sim.step(action.v_d, action.v_q, T_load=_T_load)
        transitions = count_transitions(action.switching_state, prev_sw)
        sw_transitions.append(transitions)
        prev_sw = action.switching_state.copy()

        if k % record_every == 0:
            d = _state_dict(state, action)
            d["sw_transitions"] = transitions
            history_raw.append(d)

    _history = history_raw

    # Metrics — track q-axis current (primary torque-producing axis)
    metrics_iq = _compute_metrics(history_raw, "i_q_A", _i_q_ref)
    metrics_id = _compute_metrics(history_raw, "i_d_A", _i_d_ref)

    avg_transitions = float(np.mean(sw_transitions))
    # Switching frequency: avg transitions/step / (3 phases) * (1/dt) Hz
    f_sw_hz = avg_transitions / (3 * _motor_params.dt)

    return {
        "status": "ok",
        "duration_s": duration_s,
        "n_samples": len(history_raw),
        "metrics_iq": metrics_iq,
        "metrics_id": metrics_id,
        "switching": {
            "avg_transitions_per_step": round(avg_transitions, 3),
            "effective_switching_freq_hz": round(f_sw_hz, 1),
        },
        "final_state": history_raw[-1] if history_raw else None,
        "context": (
            f"Scenario complete. i_q rise time: {metrics_iq.get('rise_time_s')} s, "
            f"overshoot: {metrics_iq.get('overshoot_pct')} %, "
            f"ripple: {metrics_iq.get('steady_state_ripple')} A. "
            f"Avg switching freq: {round(f_sw_hz, 0)} Hz."
        ),
    }


@mcp.tool()
def get_history(max_samples: int = 200) -> dict:
    """
    Return the recorded signal history from the last scenario.
    Useful for plotting or further analysis.

    Args:
        max_samples: Limit number of returned samples (downsampled uniformly).
    """
    if not _history:
        return {"status": "error", "context": "No history recorded. Run a scenario first."}

    step = max(1, len(_history) // max_samples)
    samples = _history[::step]
    return {
        "status": "ok",
        "n_samples": len(samples),
        "signals": list(samples[0].keys()),
        "data": samples,
        "context": f"Returning {len(samples)} samples (every {step}th of {len(_history)} recorded).",
    }


@mcp.tool()
def compute_operating_point(i_mag: float, omega_r_rads: float = 0.0) -> dict:
    """
    Compute the optimal (i_d, i_q) operating point for given current magnitude
    and speed, choosing between MTPA (below base speed) and field weakening
    (above base speed). Does NOT apply the reference — use set_mtpa_reference
    or set_field_weakening_reference to apply.

    Args:
        i_mag:        Current magnitude [A]
        omega_r_rads: Mechanical speed [rad/s] (default 0 → MTPA)
    """
    p = _motor_params
    i_d, i_q = field_weakening(p, omega_r_rads, i_mag) if omega_r_rads > 0 \
                else mtpa(p, i_mag)
    T = torque(p, i_d, i_q)
    in_fw = omega_r_rads > p.base_speed
    return {
        "i_d_A": round(i_d, 4),
        "i_q_A": round(i_q, 4),
        "torque_Nm": round(T, 3),
        "field_weakening_active": in_fw,
        "context": (
            f"{'Field weakening' if in_fw else 'MTPA'} operating point: "
            f"i_d={i_d:.2f} A, i_q={i_q:.2f} A → {T:.2f} N·m at "
            f"{omega_r_rads*60/(2*np.pi):.0f} RPM."
        ),
    }


# ---------------------------------------------------------------------------
# Load profile simulation
# ---------------------------------------------------------------------------

@mcp.tool()
def run_load_profile(
    profile: list[list[float]],
    speed_reference_rpm: float | None = None,
    speed_kp: float | None = None,
    speed_ki: float | None = None,
    record_every: int = 5,
) -> dict:
    """
    Run a simulation driven by a piecewise-linear load torque profile and
    return performance statistics and a plot.

    The profile is specified as a list of [time_s, torque_Nm] waypoints.
    Torque is linearly interpolated between waypoints. The simulation runs
    from t=0 to the last waypoint time.

    Two control modes:
      - Torque control (default): current reference stays as configured
        (use set_mtpa_reference first). Speed varies freely under the load.
      - Speed control: set speed_reference_rpm to activate an outer PI speed
        loop (MTPA inner loop). Speed is regulated against varying load.

    Args:
        profile:              [[t0, T0], [t1, T1], ...] — at least 2 points,
                              times monotonically increasing, t0 should be 0.
                              Example: [[0,0],[0.1,0],[0.1,3],[0.4,3],[0.4,0],[0.6,0]]
                              for a 3 N·m load step from t=0.1 s to t=0.4 s.
        speed_reference_rpm:  Target speed for outer PI loop [RPM]. If None,
                              uses current FCS-MPC reference (torque control).
        speed_kp:             Override speed PI proportional gain [A/(rad/s)].
                              If None, uses the value set via
                              set_controller_weights (default 5.0).
        speed_ki:             Override speed PI integral gain [A/rad].
                              If None, uses the value set via
                              set_controller_weights (default 100.0).
        record_every:         Record every N-th step to limit output size.

    Statistics returned:
      speed, torque, current RMS (thermal load), power, efficiency proxy,
      switching frequency.
    """
    p = _motor_params

    # Resolve speed-loop gains: explicit args override module-level state.
    eff_speed_kp = speed_kp if speed_kp is not None else _speed_kp
    eff_speed_ki = speed_ki if speed_ki is not None else _speed_ki

    # --- Validate profile ---
    if len(profile) < 2:
        return {"status": "error", "context": "Profile needs at least 2 waypoints."}
    try:
        times_p = [float(pt[0]) for pt in profile]
        torqs_p = [float(pt[1]) for pt in profile]
    except Exception:
        return {"status": "error", "context": "Profile must be [[time_s, torque_Nm], ...]."}
    if any(times_p[i] > times_p[i+1] for i in range(len(times_p)-1)):
        return {"status": "error", "context": "Profile times must be monotonically non-decreasing."}

    duration_s = times_p[-1]
    if duration_s <= 0:
        return {"status": "error", "context": "Last waypoint time must be > 0."}

    n_steps = int(duration_s / p.dt)
    if n_steps > 500_000:
        return {"status": "error", "context": "Profile too long. Reduce duration or increase dt."}

    # --- Set up simulation ---
    sim = IPMSMSimulator(p)
    ctrl_params = FCSMPCParameters(horizon=_ctrl_params.horizon, weights=_ctrl_params.weights)
    ctrl = FCSMPCController(p, ctrl_params)

    speed_mode = speed_reference_rpm is not None
    omega_ref = speed_reference_rpm * 2 * np.pi / 60.0 if speed_mode else None
    int_spd = 0.0
    id_ref_cur, iq_ref_cur = _i_d_ref, _i_q_ref
    ctrl.set_reference(id_ref_cur, iq_ref_cur)

    history: list[dict] = []
    sw_all: list[int] = []
    prev_sw = np.zeros(3)

    for k in range(n_steps):
        t_now = k * p.dt
        T_load_now = float(np.interp(t_now, times_p, torqs_p))

        # Outer speed loop (if active)
        if speed_mode:
            err_spd = omega_ref - sim.state.omega_r
            int_spd += err_spd * p.dt
            iq_cmd = float(np.clip(eff_speed_kp * err_spd + eff_speed_ki * int_spd,
                                   -p.i_max, p.i_max))
            i_mag_cmd = abs(iq_cmd)
            if i_mag_cmd > 0.1:
                id_cmd, _ = mtpa(p, i_mag_cmd)
                id_cmd = id_cmd if iq_cmd > 0 else 0.0
            else:
                id_cmd = 0.0
            ctrl.set_reference(id_cmd, iq_cmd)

        action = ctrl.step(sim.state)
        state  = sim.step(action.v_d, action.v_q, T_load=T_load_now)
        trans  = count_transitions(action.switching_state, prev_sw)
        sw_all.append(trans)
        prev_sw = action.switching_state.copy()

        if k % record_every == 0:
            history.append({
                "t_ms":      round(t_now * 1e3, 3),
                "T_load_Nm": round(T_load_now, 4),
                "T_e_Nm":    round(state.T_e, 4),
                "speed_rpm": round(state.speed_rpm, 2),
                "omega_r":   round(state.omega_r, 4),
                "i_d_A":     round(state.i_d, 4),
                "i_q_A":     round(state.i_q, 4),
                "i_mag_A":   round(state.i_mag, 4),
                "sw":        trans,
            })

    # --- Statistics ---
    spd   = np.array([h["speed_rpm"] for h in history])
    Te    = np.array([h["T_e_Nm"]    for h in history])
    imag  = np.array([h["i_mag_A"]   for h in history])
    omega = np.array([h["omega_r"]   for h in history])

    i_rms = float(np.sqrt(np.mean(imag ** 2)))
    copper_loss_W = 1.5 * p.R_s * i_rms ** 2
    mech_power_W  = float(np.mean(Te * omega))
    eff_proxy     = mech_power_W / (mech_power_W + copper_loss_W) if (mech_power_W + copper_loss_W) > 0 else 0.0
    avg_trans     = float(np.mean(sw_all))
    f_sw_hz       = avg_trans / (3 * p.dt)

    speed_stats: dict = {
        "mean_rpm":  round(float(spd.mean()), 1),
        "min_rpm":   round(float(spd.min()),  1),
        "max_rpm":   round(float(spd.max()),  1),
        "std_rpm":   round(float(spd.std()),  2),
    }
    if speed_mode:
        ref_rpm = speed_reference_rpm
        speed_stats["reference_rpm"]    = ref_rpm
        speed_stats["max_deviation_rpm"] = round(float(np.max(np.abs(spd - ref_rpm))), 2)
        speed_stats["regulation_pct"]   = round(float(np.max(np.abs(spd - ref_rpm))) / max(abs(ref_rpm), 1) * 100, 2)

    stats = {
        "speed":    speed_stats,
        "torque":   {"mean_Nm": round(float(Te.mean()), 3),
                     "peak_Nm": round(float(Te.max()),  3),
                     "rms_Nm":  round(float(np.sqrt(np.mean(Te**2))), 3)},
        "current":  {"rms_A":   round(i_rms, 3),
                     "peak_A":  round(float(imag.max()), 3)},
        "power":    {"mech_mean_W":    round(mech_power_W, 1),
                     "copper_loss_W":  round(copper_loss_W, 1),
                     "efficiency_proxy_pct": round(eff_proxy * 100, 1)},
        "switching": {"avg_freq_hz": round(f_sw_hz, 0)},
    }

    # --- Plot ---
    fig = plt.figure(figsize=(11, 8))
    mode_label = (f"Speed ctrl  ω_ref={speed_reference_rpm:.0f} RPM" if speed_mode
                  else "Torque ctrl  (current ref fixed)")
    fig.suptitle(f"Load Profile Simulation — {mode_label}", fontsize=11, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)
    t_ms = [h["t_ms"] for h in history]

    # Speed
    ax = fig.add_subplot(gs[0, :])   # full width
    ax.plot(t_ms, spd, color="#1f77b4", label="ω_r")
    if speed_mode:
        ax.axhline(speed_reference_rpm, color="red", ls="--", lw=0.9,
                   label=f"ref {speed_reference_rpm:.0f} RPM")
    ax.set_ylabel("Speed [RPM]"); ax.set_title("Speed"); ax.legend(fontsize=8); ax.grid(True)

    # Load + electromagnetic torque
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t_ms, [h["T_e_Nm"]    for h in history], color="#2ca02c", label="T_e")
    ax.plot(t_ms, [h["T_load_Nm"] for h in history], color="gray",    ls="--", lw=1.0, label="T_load")
    ax.set_ylabel("Torque [N·m]"); ax.set_title("Torque"); ax.legend(fontsize=8); ax.grid(True)

    # dq currents
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t_ms, [h["i_d_A"] for h in history], label="i_d")
    ax.plot(t_ms, [h["i_q_A"] for h in history], label="i_q")
    ax.set_ylabel("Current [A]"); ax.set_title("dq currents"); ax.legend(fontsize=8); ax.grid(True)

    # Current magnitude (thermal)
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t_ms, [h["i_mag_A"] for h in history], color="#d62728")
    ax.axhline(i_rms, color="gray", ls="--", lw=0.9, label=f"RMS {i_rms:.2f} A")
    ax.axhline(p.i_max, color="black", ls=":", lw=0.8, label=f"i_max {p.i_max} A")
    ax.set_ylabel("|i| [A]"); ax.set_xlabel("Time [ms]")
    ax.set_title("Current magnitude (thermal load)"); ax.legend(fontsize=8); ax.grid(True)

    # Stats summary text
    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    lines = [
        f"Speed:  {speed_stats['mean_rpm']:.0f} ± {speed_stats['std_rpm']:.1f} RPM",
    ]
    if speed_mode:
        lines.append(f"Max deviation:  {speed_stats.get('max_deviation_rpm','—')} RPM  "
                     f"({speed_stats.get('regulation_pct','—')} %)")
    lines += [
        f"",
        f"T_e:  mean {stats['torque']['mean_Nm']} N·m,  peak {stats['torque']['peak_Nm']} N·m",
        f"i RMS:  {stats['current']['rms_A']} A,  peak {stats['current']['peak_A']} A",
        f"",
        f"Mech. power:  {stats['power']['mech_mean_W']:.0f} W",
        f"Copper loss:  {stats['power']['copper_loss_W']:.1f} W",
        f"η proxy:  {stats['power']['efficiency_proxy_pct']:.1f} %",
        f"",
        f"Avg switch freq:  {stats['switching']['avg_freq_hz']:.0f} Hz",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=8, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    img = _fig_to_image(fig, dpi=120)

    summary = (
        f"Load profile complete ({duration_s*1e3:.0f} ms, {len(history)} samples). "
        + (f"Speed: {speed_stats['mean_rpm']:.0f} RPM mean, "
           f"max deviation {speed_stats.get('max_deviation_rpm','—')} RPM "
           f"({speed_stats.get('regulation_pct','—')} %). " if speed_mode else
           f"Speed: {speed_stats['mean_rpm']:.0f} RPM mean, ±{speed_stats['std_rpm']:.1f} RPM. ")
        + f"i RMS {stats['current']['rms_A']} A. "
        + f"η proxy {stats['power']['efficiency_proxy_pct']:.1f} %. "
        + f"Switching {stats['switching']['avg_freq_hz']:.0f} Hz."
    )

    return {"status": "ok", "statistics": stats, "context": summary, "plot": img}


# ---------------------------------------------------------------------------
# Identification tools — automatic parameter sweeps
# ---------------------------------------------------------------------------

_flux_map: dict | None = None   # set by sweep_flux_linkage_map, read by plot_flux_linkage_map


class _PICtrl:
    """Minimal PI current controller for identification sweeps (not FCS-MPC)."""
    def __init__(self, p: IPMSMParameters, bw_hz: float = 300.0):
        self.p = p
        w = 2 * np.pi * bw_hz
        self.kp_d, self.ki_d = w * p.L_d, w * p.R_s
        self.kp_q, self.ki_q = w * p.L_q, w * p.R_s
        self._ed = self._eq = 0.0

    def reset(self):
        self._ed = self._eq = 0.0

    def step(self, i_d_ref, i_q_ref, state, dt):
        ed, eq = i_d_ref - state.i_d, i_q_ref - state.i_q
        self._ed += ed * dt;  self._eq += eq * dt
        we = self.p.p * state.omega_r
        v_d = self.kp_d * ed + self.ki_d * self._ed - we * self.p.L_q * state.i_q
        v_q = self.kp_q * eq + self.ki_q * self._eq + we * (self.p.L_d * state.i_d + self.p.psi_f)
        return v_d, v_q


@mcp.tool()
def identify_resistance(v_test_V: float = 5.0, duration_s: float = 0.1) -> dict:
    """
    Estimate stator resistance R_s via a DC standstill test.

    Applies v_d = v_test at standstill (v_q = 0, rotor locked). At steady
    state, the back-EMF and speed-dependent terms vanish, so:
        R_s_est = v_d / i_d_ss

    Safe to run: i_q stays near zero, producing no torque and no motion.
    The motor stays at standstill throughout.

    Args:
        v_test_V:   DC test voltage applied on d-axis [V] (keep < R_s * i_max)
        duration_s: Duration — must be long enough for i_d to reach steady state
                    (rule of thumb: ≥ 5 * L_d / R_s)
    """
    p = _motor_params
    tau_d = p.L_d / p.R_s
    if duration_s < 3 * tau_d:
        return {
            "status": "warning",
            "context": (f"Duration {duration_s*1e3:.1f} ms may be too short. "
                        f"Time constant L_d/R_s = {tau_d*1e3:.1f} ms. "
                        f"Recommend ≥ {3*tau_d*1e3:.0f} ms."),
        }

    sim = IPMSMSimulator(p)
    n_steps = int(duration_s / p.dt)
    for _ in range(n_steps):
        sim.step(v_test_V, 0.0, T_load=0.0, lock_speed=True)

    state = sim.state
    i_d_ss = state.i_d
    if abs(i_d_ss) < 1e-3:
        return {"status": "error", "context": "Steady-state current too small. Increase v_test_V."}

    R_s_est = v_test_V / i_d_ss
    error_pct = 100.0 * (R_s_est - p.R_s) / p.R_s

    return {
        "status": "ok",
        "v_test_V": v_test_V,
        "i_d_steady_state_A": round(i_d_ss, 4),
        "R_s_estimated_ohm": round(R_s_est, 4),
        "R_s_nominal_ohm": p.R_s,
        "error_pct": round(error_pct, 2),
        "context": (
            f"Standstill test complete. "
            f"i_d settled to {i_d_ss:.3f} A → R_s = {R_s_est:.4f} Ω "
            f"(nominal {p.R_s} Ω, error {error_pct:+.1f}%). "
            f"{'Good match.' if abs(error_pct) < 5 else 'Discrepancy — check wiring or model.'}"
        ),
    }


@mcp.tool()
def sweep_flux_linkage_map(
    n_id: int = 5,
    n_iq: int = 5,
    omega_test_rpm: float = 500.0,
    settle_ms: float = 40.0,
) -> dict:
    """
    Identify the flux linkage map ψ_d(i_d, i_q) and ψ_q(i_d, i_q) by sweeping
    a grid of current operating points at a fixed test speed.

    Method (standard rotating flux observer):
      At each (i_d, i_q) setpoint and fixed speed ω_test, the steady-state
      voltages satisfy:
          v_d_ss = R_s·i_d - ω_e·ψ_q   →   ψ_q = -(v_d_ss - R_s·i_d) / ω_e
          v_q_ss = R_s·i_q + ω_e·ψ_d   →   ψ_d =  (v_q_ss - R_s·i_q) / ω_e

      Apparent inductances are extracted as:
          L_d_app = (ψ_d - ψ_f) / i_d   [where i_d ≠ 0]
          L_q_app = ψ_q / i_q             [where i_q ≠ 0]

    Non-linear variation reveals magnetic saturation — the key nonlinearity
    in IPMSM that causes the linear MTPA formula to be suboptimal.

    Args:
        n_id:            Number of i_d grid points (swept 0 → -i_max)
        n_iq:            Number of i_q grid points (swept 0 → i_max)
        omega_test_rpm:  Test speed [RPM] — must be nonzero for flux observer
        settle_ms:       Settling time per operating point [ms]
    """
    global _flux_map
    p = _motor_params
    omega_r_test = omega_test_rpm * 2 * np.pi / 60.0
    omega_e_test = p.p * omega_r_test

    if abs(omega_e_test) < 1.0:
        return {"status": "error", "context": "omega_test_rpm too low for flux observer. Use ≥ 100 RPM."}
    if n_id < 2 or n_iq < 2:
        return {"status": "error", "context": "n_id and n_iq must be ≥ 2."}

    id_grid = np.linspace(0.0, -p.i_max * 0.9, n_id)
    iq_grid = np.linspace(0.5,  p.i_max * 0.9, n_iq)
    n_settle = int(settle_ms * 1e-3 / p.dt)

    psi_d_map = np.zeros((n_id, n_iq))
    psi_q_map = np.zeros((n_id, n_iq))
    Ld_app_map = np.full((n_id, n_iq), np.nan)
    Lq_app_map = np.full((n_id, n_iq), np.nan)

    sim = IPMSMSimulator(p)
    ctrl = _PICtrl(p, bw_hz=400.0)

    for ii, id_cmd in enumerate(id_grid):
        for jj, iq_cmd in enumerate(iq_grid):
            sim.reset(omega_r=omega_r_test)
            ctrl.reset()
            for _ in range(n_settle):
                v_d, v_q = ctrl.step(id_cmd, iq_cmd, sim.state, p.dt)
                sim.step(v_d, v_q, T_load=0.0, lock_speed=True)

            # Steady-state flux observer
            st = sim.state
            psi_d = (st.v_q - p.R_s * st.i_q) / omega_e_test
            psi_q = -(st.v_d - p.R_s * st.i_d) / omega_e_test
            psi_d_map[ii, jj] = psi_d
            psi_q_map[ii, jj] = psi_q
            if abs(st.i_d) > 0.2:
                Ld_app_map[ii, jj] = (psi_d - p.psi_f) / st.i_d
            if abs(st.i_q) > 0.2:
                Lq_app_map[ii, jj] = psi_q / st.i_q

    # Summary statistics
    valid_Ld = Ld_app_map[~np.isnan(Ld_app_map)]
    valid_Lq = Lq_app_map[~np.isnan(Lq_app_map)]

    _flux_map = {
        "id_grid_A": id_grid.tolist(),
        "iq_grid_A": iq_grid.tolist(),
        "psi_d": psi_d_map.tolist(),
        "psi_q": psi_q_map.tolist(),
        "Ld_app_mH": (Ld_app_map * 1e3).tolist(),
        "Lq_app_mH": (Lq_app_map * 1e3).tolist(),
        "omega_test_rpm": omega_test_rpm,
        "n_points": n_id * n_iq,
    }

    return {
        "status": "ok",
        "grid": f"{n_id} × {n_iq} = {n_id*n_iq} points",
        "omega_test_rpm": omega_test_rpm,
        "L_d_nominal_mH": p.L_d * 1e3,
        "L_d_app_range_mH": [round(float(valid_Ld.min())*1e3, 2), round(float(valid_Ld.max())*1e3, 2)] if len(valid_Ld) else None,
        "L_q_nominal_mH": p.L_q * 1e3,
        "L_q_app_range_mH": [round(float(valid_Lq.min())*1e3, 2), round(float(valid_Lq.max())*1e3, 2)] if len(valid_Lq) else None,
        "saturation_model": {"k_sat_d": p.k_sat_d, "k_sat_q": p.k_sat_q},
        "context": (
            f"Flux map identified at {n_id}×{n_iq} points ({omega_test_rpm:.0f} RPM test speed). "
            f"L_d range: {round(float(valid_Ld.min())*1e3,2) if len(valid_Ld) else '—'}–"
            f"{round(float(valid_Ld.max())*1e3,2) if len(valid_Ld) else '—'} mH "
            f"(nominal {p.L_d*1e3:.1f} mH). "
            f"L_q range: {round(float(valid_Lq.min())*1e3,2) if len(valid_Lq) else '—'}–"
            f"{round(float(valid_Lq.max())*1e3,2) if len(valid_Lq) else '—'} mH "
            f"(nominal {p.L_q*1e3:.1f} mH). "
            f"Call plot_flux_linkage_map() to visualise."
        ),
    }


@mcp.tool()
def plot_flux_linkage_map() -> Image:
    """
    Plot the identified flux linkage map as 2D colourmaps.
    Shows ψ_d, ψ_q, apparent L_d and L_q as functions of (i_d, i_q).
    Non-uniform inductance values reveal magnetic saturation.

    Must call sweep_flux_linkage_map() first.
    """
    if _flux_map is None:
        raise ValueError("No flux map available. Run sweep_flux_linkage_map() first.")

    id_g = np.array(_flux_map["id_grid_A"])
    iq_g = np.array(_flux_map["iq_grid_A"])
    IQ, ID = np.meshgrid(iq_g, id_g)   # shapes (n_id, n_iq)

    psi_d = np.array(_flux_map["psi_d"])
    psi_q = np.array(_flux_map["psi_q"])
    Ld_mH = np.array(_flux_map["Ld_app_mH"])
    Lq_mH = np.array(_flux_map["Lq_app_mH"])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        f"Flux Linkage Map — identified at {_flux_map['omega_test_rpm']:.0f} RPM  "
        f"({_flux_map['n_points']} points)",
        fontsize=11, fontweight="bold",
    )

    datasets = [
        (psi_d, "ψ_d [Wb]",   "Blues",  axes[0, 0]),
        (psi_q, "ψ_q [Wb]",   "Oranges",axes[0, 1]),
        (Ld_mH, "L_d_app [mH]","RdYlGn", axes[1, 0]),
        (Lq_mH, "L_q_app [mH]","RdYlGn", axes[1, 1]),
    ]

    for data, label, cmap, ax in datasets:
        masked = np.ma.masked_invalid(data)
        im = ax.pcolormesh(IQ, ID, masked, cmap=cmap, shading="auto")
        plt.colorbar(im, ax=ax, label=label)
        ax.set_xlabel("i_q [A]")
        ax.set_ylabel("i_d [A]")
        ax.set_title(label)

        # Annotate each cell with its value
        for ii in range(len(id_g)):
            for jj in range(len(iq_g)):
                val = data[ii, jj]
                if not np.isnan(val):
                    ax.text(iq_g[jj], id_g[ii], f"{val:.2f}",
                            ha="center", va="center", fontsize=6, color="black")

    plt.tight_layout()
    return _fig_to_image(fig, dpi=120)


# ---------------------------------------------------------------------------
# Plot tools — return inline images to the agent / chat session
# ---------------------------------------------------------------------------

def _fig_to_image(fig: plt.Figure, dpi: int = 130) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image(data=buf.read(), format="png")


@mcp.tool()
def plot_results(title: str = "") -> Image:
    """
    Generate and return a commissioning results plot for the last scenario.
    Shows d/q currents, electromagnetic torque, speed, and switching activity.
    Call this after run_scenario() to visualise the response in the chat.

    Args:
        title: Optional title suffix (e.g. 'λ_sw=0.5, horizon=2')
    """
    if not _history:
        raise ValueError("No history to plot. Run a scenario first.")

    t   = [h["time_s"] * 1e3 for h in _history]   # ms
    i_d = [h["i_d_A"]        for h in _history]
    i_q = [h["i_q_A"]        for h in _history]
    Te  = [h["T_e_Nm"]       for h in _history]
    sw  = [h.get("sw_transitions", 0) for h in _history]

    fig = plt.figure(figsize=(10, 7))
    fig.suptitle(f"FCS-MPC Commissioning Results{' — ' + title if title else ''}",
                 fontsize=11, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # i_d
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, i_d, color="#1f77b4")
    ax.axhline(_i_d_ref, color="red", ls="--", lw=0.9, label=f"ref {_i_d_ref:.2f} A")
    ax.set_ylabel("i_d [A]"); ax.set_title("d-axis current"); ax.legend(fontsize=7); ax.grid(True)

    # i_q
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, i_q, color="#ff7f0e")
    ax.axhline(_i_q_ref, color="red", ls="--", lw=0.9, label=f"ref {_i_q_ref:.2f} A")
    ax.set_ylabel("i_q [A]"); ax.set_title("q-axis current"); ax.legend(fontsize=7); ax.grid(True)

    # Torque
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t, Te, color="#2ca02c")
    if _T_load > 0:
        ax.axhline(_T_load, color="gray", ls=":", lw=0.9, label=f"T_load {_T_load:.1f} N·m")
        ax.legend(fontsize=7)
    ax.set_ylabel("T_e [N·m]"); ax.set_xlabel("Time [ms]")
    ax.set_title("Electromagnetic torque"); ax.grid(True)

    # Current plane
    ax = fig.add_subplot(gs[0:2, 1])
    theta = np.linspace(0, np.pi, 300)
    i_max_v = _motor_params.i_max
    ax.plot(-i_max_v * np.cos(theta), i_max_v * np.sin(theta),
            "k--", lw=0.8, label="i_max")
    i_mags = np.linspace(0.5, i_max_v, 40)
    mtpa_pts = [mtpa(_motor_params, im) for im in i_mags]
    ax.plot([p[0] for p in mtpa_pts], [p[1] for p in mtpa_pts],
            "g:", lw=1.2, label="MTPA locus")
    ax.plot(i_d, i_q, color="#1f77b4", alpha=0.7, label="trajectory")
    ax.plot(_i_d_ref, _i_q_ref, "r*", ms=10, label="reference")
    ax.set_xlabel("i_d [A]"); ax.set_ylabel("i_q [A]")
    ax.set_title("Current plane"); ax.legend(fontsize=7)
    ax.set_aspect("equal"); ax.grid(True)

    # Switching activity
    ax = fig.add_subplot(gs[2, 1])
    ax.bar(t, sw, width=(t[1]-t[0]) if len(t) > 1 else 0.1,
           color="#9467bd", alpha=0.7)
    avg_t = np.mean(sw)
    ax.axhline(avg_t, color="red", ls="--", lw=0.9, label=f"avg {avg_t:.2f}")
    ax.set_ylabel("Transitions"); ax.set_xlabel("Time [ms]")
    ax.set_title("Switching activity"); ax.legend(fontsize=7); ax.grid(True)

    return _fig_to_image(fig)


@mcp.tool()
def plot_operating_region() -> Image:
    """
    Plot the motor's operating region diagram:
    current limit circle, MTPA locus, voltage limit ellipses at several speeds,
    and the field-weakening boundary. Useful for understanding the motor's
    capabilities before commissioning.
    """
    p = _motor_params
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("IPMSM Operating Region", fontsize=11, fontweight="bold")

    # Current limit circle
    theta = np.linspace(0, np.pi, 300)
    ax.plot(-p.i_max * np.cos(theta), p.i_max * np.sin(theta),
            "k-", lw=1.5, label=f"Current limit  |i| = {p.i_max} A")

    # MTPA locus
    i_mags = np.linspace(0.3, p.i_max, 60)
    mtpa_pts = [mtpa(p, im) for im in i_mags]
    ax.plot([pt[0] for pt in mtpa_pts], [pt[1] for pt in mtpa_pts],
            "g-", lw=1.5, label="MTPA locus")

    # Voltage limit ellipses at several speeds
    speeds_pu = [0.5, 1.0, 1.5, 2.0, 3.0]   # multiples of base speed
    omega_base = p.base_speed
    colors = plt.cm.autumn(np.linspace(0.1, 0.9, len(speeds_pu)))
    i_d_range = np.linspace(-p.i_max, 0.5, 400)
    for pu, col in zip(speeds_pu, colors):
        omega_e = pu * omega_base * p.p
        if omega_e < 1e-3:
            continue
        a = p.v_max / (omega_e * p.L_d)
        b = p.v_max / (omega_e * p.L_q)
        c_d = -p.psi_f / p.L_d
        arg = 1 - ((i_d_range - c_d) / a) ** 2
        mask = arg >= 0
        i_q_ellipse = b * np.sqrt(np.where(mask, arg, 0))
        ax.plot(i_d_range[mask], i_q_ellipse[mask], color=col, lw=1.0,
                label=f"V-limit  {pu:.1f}× ω_base")

    # Field-weakening trajectory at full current
    fw_speeds = np.linspace(0.8 * omega_base, 3.5 * omega_base, 80)
    fw_pts = [field_weakening(p, w, p.i_max) for w in fw_speeds]
    ax.plot([pt[0] for pt in fw_pts], [pt[1] for pt in fw_pts],
            "b--", lw=1.5, label="FW trajectory (i_max)")

    ax.set_xlabel("i_d [A]"); ax.set_ylabel("i_q [A]")
    ax.set_xlim(-p.i_max * 1.1, p.i_max * 0.3)
    ax.set_ylim(0, p.i_max * 1.1)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True)
    ax.set_aspect("equal")

    info = (f"R_s={p.R_s} Ω  L_d={p.L_d*1e3:.1f} mH  L_q={p.L_q*1e3:.1f} mH  "
            f"ψ_f={p.psi_f} Wb  ξ={p.saliency_ratio:.2f}  "
            f"ω_base={p.base_speed*60/(2*np.pi):.0f} RPM")
    fig.text(0.5, 0.01, info, ha="center", fontsize=7, color="gray")

    return _fig_to_image(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    transport = "sse" if "--sse" in sys.argv else "stdio"
    mcp.run(transport=transport)
