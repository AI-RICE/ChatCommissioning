"""
FCS-MPC (Finite Control Set Model Predictive Control) for IPMSM.

The inverter has 8 switching states (2-level, 3-phase), producing 7 unique
voltage vectors in the αβ plane (two zero vectors V0/V7 are both null).
At each control step the controller:
  1. Enumerates all reachable voltage vectors up to `horizon` steps ahead.
  2. Predicts the motor currents using the discretised dq model.
  3. Selects the switching sequence that minimises the cost function.

Cost function terms (all optional via weights):
  - Current tracking:  λ_id*(i_d_pred - i_d_ref)² + λ_iq*(i_q_pred - i_q_ref)²
  - Switching penalty: λ_sw * (number of phase transitions per step)
  - Torque ripple:     λ_tr * (T_e_pred - T_e_ref)²
  - Common-mode:       λ_cm * v_cm²   (v_cm = Vdc/3 for (1,0,0) etc.)

These weights are the primary commissioning parameters exposed to the AI agent.
Increasing λ_sw reduces switching frequency at the cost of larger current ripple —
exactly the trade-off a human engineer would negotiate.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, NamedTuple

from simulator.ipmsm import IPMSMParameters, torque


# ---------------------------------------------------------------------------
# Inverter voltage vectors
# ---------------------------------------------------------------------------

# 8 switching states of a 2-level 3-phase VSI: Sa, Sb, Sc ∈ {0, 1}
_SWITCHING_STATES = np.array([
    [0, 0, 0],   # V0 — null
    [1, 0, 0],   # V1
    [1, 1, 0],   # V2
    [0, 1, 0],   # V3
    [0, 1, 1],   # V4
    [0, 0, 1],   # V5
    [1, 0, 1],   # V6
    [1, 1, 1],   # V7 — null
], dtype=float)

# Clarke transform: S_abc → v_αβ  (v_αβ = Vdc * C · S_abc)
_C = (2.0 / 3.0) * np.array([
    [1.0, -0.5,       -0.5],
    [0.0,  np.sqrt(3)/2, -np.sqrt(3)/2],
])

N_VECTORS = len(_SWITCHING_STATES)   # 8


def switching_vectors_dq(v_dc: float, theta_e: float) -> np.ndarray:
    """
    Compute all 8 inverter voltage vectors in the dq frame.

    Args:
        v_dc:    DC bus voltage [V]
        theta_e: electrical rotor angle [rad]

    Returns:
        Array of shape (8, 2) — each row is [v_d, v_q] for one switching state.
    """
    v_ab = (v_dc * (_C @ _SWITCHING_STATES.T)).T   # (8, 2)  αβ voltages

    cos_e = np.cos(theta_e)
    sin_e = np.sin(theta_e)
    # Park transform (αβ → dq)
    v_d =  v_ab[:, 0] * cos_e + v_ab[:, 1] * sin_e
    v_q = -v_ab[:, 0] * sin_e + v_ab[:, 1] * cos_e
    return np.column_stack([v_d, v_q])   # (8, 2)


def common_mode_voltage(switching_state: np.ndarray, v_dc: float) -> float:
    """Common-mode voltage for a switching state [V]."""
    return v_dc * (switching_state.sum() / 3.0 - 0.5)


def count_transitions(state_new: np.ndarray, state_old: np.ndarray) -> int:
    """Number of phase legs that change state (0→1 or 1→0)."""
    return int(np.sum(state_new != state_old))


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class FCSMPCWeights:
    """
    Cost function weights — primary commissioning knobs.

    Typical trade-offs (useful context for the AI agent):
      - Raising λ_sw  → fewer switchings, lower losses, larger current ripple
      - Raising λ_tr  → smoother torque (better for low-speed precision), may conflict
                        with current tracking at high speeds
      - Raising λ_cm  → reduced EMI / bearing currents, slightly higher ripple
      - Raising λ_id / λ_iq independently → bias tracking priority between axes
    """
    id_error: float = 1.0      # d-axis current tracking
    iq_error: float = 1.0      # q-axis current tracking
    switching: float = 0.0     # switching transition penalty  [per transition]
    torque_ripple: float = 0.0 # torque ripple penalty
    common_mode: float = 0.0   # common-mode voltage penalty


@dataclass
class FCSMPCParameters:
    """
    Controller configuration.

    horizon:   Prediction steps. 1 = standard FCS-MPC (7 candidates).
               2 → 49, 3 → 343 candidates — rapidly increases computation
               but can reduce steady-state ripple and improve constraint handling.
               Horizon is an important commissioning parameter to demonstrate.
    """
    horizon: int = 1
    weights: FCSMPCWeights = field(default_factory=FCSMPCWeights)

    def __post_init__(self):
        if not 1 <= self.horizon <= 3:
            raise ValueError("horizon must be 1, 2, or 3")

    def describe(self) -> dict:
        w = self.weights
        return {
            "horizon": self.horizon,
            "weight_id_error": w.id_error,
            "weight_iq_error": w.iq_error,
            "weight_switching": w.switching,
            "weight_torque_ripple": w.torque_ripple,
            "weight_common_mode": w.common_mode,
        }


# ---------------------------------------------------------------------------
# Prediction model (Euler forward — fast, sufficient for FCS-MPC timestep)
# ---------------------------------------------------------------------------

def predict_currents(
    i_d: float, i_q: float, omega_r: float,
    v_d: float, v_q: float,
    motor: IPMSMParameters, dt: float,
) -> tuple:
    """One-step Euler prediction of (i_d, i_q)."""
    omega_e = motor.p * omega_r
    i_d_next = i_d + (dt / motor.L_d) * (v_d - motor.R_s * i_d + omega_e * motor.L_q * i_q)
    i_q_next = i_q + (dt / motor.L_q) * (v_q - motor.R_s * i_q - omega_e * (motor.L_d * i_d + motor.psi_f))
    return i_d_next, i_q_next


# ---------------------------------------------------------------------------
# Controller result
# ---------------------------------------------------------------------------

class ControlAction(NamedTuple):
    switching_state: np.ndarray   # shape (3,) — [Sa, Sb, Sc]
    v_d: float
    v_q: float
    cost: float
    i_d_pred: float               # predicted currents at end of horizon
    i_q_pred: float
    T_e_pred: float


# ---------------------------------------------------------------------------
# FCS-MPC controller
# ---------------------------------------------------------------------------

class FCSMPCController:
    """
    FCS-MPC current controller for IPMSM.

    The controller operates in the dq frame and selects the inverter switching
    state that minimises the cost function over the prediction horizon.

    Usage:
        ctrl = FCSMPCController(motor_params, FCSMPCParameters())
        ctrl.set_reference(i_d_ref=-3.1, i_q_ref=7.4)
        action = ctrl.step(sim.state)
        next_state = sim.step(action.v_d, action.v_q)
    """

    def __init__(self, motor: IPMSMParameters, ctrl_params: Optional[FCSMPCParameters] = None):
        self.motor = motor
        self.cp = ctrl_params or FCSMPCParameters()
        self._i_d_ref = 0.0
        self._i_q_ref = 0.0
        self._T_e_ref: Optional[float] = None   # if set, torque_ripple weight is active
        self._prev_sw = np.zeros(3, dtype=float)  # last applied switching state

    # ------------------------------------------------------------------
    # Reference setpoints
    # ------------------------------------------------------------------

    def set_reference(
        self,
        i_d_ref: float,
        i_q_ref: float,
        T_e_ref: Optional[float] = None,
    ) -> None:
        """Set current (and optionally torque) reference."""
        self._i_d_ref = i_d_ref
        self._i_q_ref = i_q_ref
        self._T_e_ref = T_e_ref

    # ------------------------------------------------------------------
    # Cost evaluation
    # ------------------------------------------------------------------

    def _cost(
        self,
        i_d_pred: float, i_q_pred: float,
        sw_new: np.ndarray, sw_old: np.ndarray,
        v_d: float, v_q: float,
    ) -> float:
        w = self.cp.weights
        T_pred = torque(self.motor, i_d_pred, i_q_pred)
        T_ref  = self._T_e_ref if self._T_e_ref is not None else T_pred

        J = (w.id_error     * (i_d_pred - self._i_d_ref) ** 2
           + w.iq_error     * (i_q_pred - self._i_q_ref) ** 2
           + w.switching    * count_transitions(sw_new, sw_old)
           + w.torque_ripple * (T_pred - T_ref) ** 2
           + w.common_mode  * common_mode_voltage(sw_new, self.motor.v_dc) ** 2)
        return J

    # ------------------------------------------------------------------
    # One-step control
    # ------------------------------------------------------------------

    def step(self, state) -> ControlAction:
        """
        Evaluate all candidates and return the optimal control action.

        Args:
            state: DriveState from the simulator (or any object with
                   .i_d, .i_q, .omega_r, .theta_r attributes)
        """
        theta_e = self.motor.p * state.theta_r
        vdq_candidates = switching_vectors_dq(self.motor.v_dc, theta_e)  # (8, 2)

        best_cost = np.inf
        best_idx = 0
        best_i_d = state.i_d
        best_i_q = state.i_q

        dt = self.motor.dt

        for idx in range(N_VECTORS):
            sw = _SWITCHING_STATES[idx]
            v_d, v_q = vdq_candidates[idx]

            # Predict over horizon
            i_d_p, i_q_p = state.i_d, state.i_q
            for _ in range(self.cp.horizon):
                i_d_p, i_q_p = predict_currents(
                    i_d_p, i_q_p, state.omega_r, v_d, v_q, self.motor, dt
                )

            cost = self._cost(i_d_p, i_q_p, sw, self._prev_sw, v_d, v_q)

            if cost < best_cost:
                best_cost = cost
                best_idx = idx
                best_i_d = i_d_p
                best_i_q = i_q_p

        best_sw = _SWITCHING_STATES[best_idx]
        best_vdq = vdq_candidates[best_idx]
        self._prev_sw = best_sw.copy()

        return ControlAction(
            switching_state=best_sw,
            v_d=float(best_vdq[0]),
            v_q=float(best_vdq[1]),
            cost=best_cost,
            i_d_pred=best_i_d,
            i_q_pred=best_i_q,
            T_e_pred=torque(self.motor, best_i_d, best_i_q),
        )

    def reset(self, switching_state: Optional[np.ndarray] = None) -> None:
        """Reset controller state (integrators, previous switching state)."""
        self._prev_sw = switching_state if switching_state is not None else np.zeros(3, dtype=float)
