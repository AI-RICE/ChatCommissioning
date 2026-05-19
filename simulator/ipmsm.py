"""
IPMSM (Interior Permanent Magnet Synchronous Motor) simulator.

Model is formulated in the synchronous dq reference frame, which is standard
for drive control. The d-axis is aligned with the PM flux vector.

State:  x = [i_d, i_q, omega_r, theta_r]
Input:  u = [v_d, v_q]          (dq voltage commands)
Disturbance: T_load              (load torque)

Key IPMSM characteristic vs surface PMSM: L_d != L_q (saliency).
This produces a reluctance torque component and enables MTPA operation
with negative i_d, which is an important commissioning parameter.
"""

import numpy as np
from dataclasses import dataclass
from typing import NamedTuple, Optional


@dataclass
class IPMSMParameters:
    """
    Physical parameters of an IPMSM drive system.

    Default values represent a small-to-medium servo motor (approx. 1 kW class)
    suitable for demonstrating commissioning scenarios including MTPA and field
    weakening.
    """
    # --- Electrical ---
    R_s: float = 0.5        # Stator resistance                    [Ω]
    L_d: float = 5.0e-3     # d-axis inductance                    [H]
    L_q: float = 12.0e-3    # q-axis inductance  (L_q > L_d)      [H]
    psi_f: float = 0.1      # PM flux linkage                      [Wb]
    p: int = 3              # Pole pairs

    # --- Mechanical ---
    J: float = 1.0e-3       # Rotor + load inertia                 [kg·m²]
    B: float = 1.0e-3       # Viscous friction coefficient         [N·m·s/rad]

    # --- Drive limits ---
    i_max: float = 10.0     # Peak current (protection limit)      [A]
    v_dc: float = 300.0     # DC bus voltage                       [V]

    # --- Magnetic saturation (optional, 0 = linear model) ---
    # q-axis saturates significantly in IPMSM due to rotor bridge flux paths.
    # k_sat_q = 0.3 produces ~23% L_q reduction at rated current — realistic.
    k_sat_d: float = 0.1    # d-axis saturation coefficient
    k_sat_q: float = 0.3    # q-axis saturation coefficient (cross-coupled)

    # --- Simulation ---
    dt: float = 1.0e-4      # Integration step                     [s]

    # --- Derived (read-only properties) ---

    @property
    def saliency_ratio(self) -> float:
        """L_q / L_d — characterises reluctance torque capability (>1 for IPMSM)."""
        return self.L_q / self.L_d

    @property
    def v_max(self) -> float:
        """Peak phase voltage (voltage limit circle radius) [V]."""
        return self.v_dc / np.sqrt(3)

    @property
    def rated_torque(self) -> float:
        """Approximate rated torque at MTPA with i_max [N·m]."""
        i_d, i_q = mtpa(self, self.i_max)
        return (3 / 2) * self.p * (self.psi_f * i_q + (self.L_d - self.L_q) * i_d * i_q)

    @property
    def base_speed(self) -> float:
        """
        Base (corner) speed: highest speed at which rated voltage can maintain
        rated current at MTPA, before field weakening is required [rad/s mechanical].
        """
        i_d, i_q = mtpa(self, self.i_max)
        # Voltage equations at steady state: v_d = R*i_d - we*Lq*iq, v_q = R*iq + we*(Ld*id + psi_f)
        # |v| = v_max; solve for omega_e
        # Simplified: ignore resistive drop for base speed estimate
        psi_d = self.L_d * i_d + self.psi_f
        psi_q = self.L_q * i_q
        omega_e = self.v_max / np.sqrt(psi_d ** 2 + psi_q ** 2)
        return omega_e / self.p   # mechanical rad/s

    def L_d_eff(self, i_d: float, i_q: float) -> float:
        """Current-dependent d-axis inductance (frozen-inductance saturation model)."""
        if self.k_sat_d == 0.0:
            return self.L_d
        return self.L_d / (1.0 + self.k_sat_d * i_d ** 2 / self.i_max ** 2)

    def L_q_eff(self, i_d: float, i_q: float) -> float:
        """Current-dependent q-axis inductance with cross-saturation."""
        if self.k_sat_q == 0.0:
            return self.L_q
        return self.L_q / (1.0 + self.k_sat_q * (i_d ** 2 + i_q ** 2) / self.i_max ** 2)

    def describe(self) -> dict:
        """Return a human-readable parameter summary for MCP resource exposure."""
        return {
            "R_s_ohm": self.R_s,
            "L_d_mH": self.L_d * 1e3,
            "L_q_mH": self.L_q * 1e3,
            "psi_f_Wb": self.psi_f,
            "pole_pairs": self.p,
            "J_kgm2": self.J,
            "B_Nms": self.B,
            "i_max_A": self.i_max,
            "v_dc_V": self.v_dc,
            "saliency_ratio": round(self.saliency_ratio, 2),
            "rated_torque_Nm": round(self.rated_torque, 2),
            "base_speed_rpm": round(self.base_speed * 60 / (2 * np.pi), 1),
            "k_sat_d": self.k_sat_d,
            "k_sat_q": self.k_sat_q,
        }


class DriveState(NamedTuple):
    """Snapshot of the drive state at one instant."""
    time: float         # [s]
    i_d: float          # d-axis current       [A]
    i_q: float          # q-axis current       [A]
    omega_r: float      # mechanical speed     [rad/s]
    theta_r: float      # rotor angle          [rad]
    T_e: float          # electromagnetic torque [N·m]
    T_load: float       # load torque          [N·m]
    v_d: float          # applied d voltage    [V]
    v_q: float          # applied q voltage    [V]

    @property
    def speed_rpm(self) -> float:
        return self.omega_r * 60 / (2 * np.pi)

    @property
    def i_mag(self) -> float:
        return np.sqrt(self.i_d ** 2 + self.i_q ** 2)

    @property
    def omega_e(self) -> float:
        """Electrical angular speed [rad/s]."""
        # pole pairs not available here; caller must multiply by p
        raise AttributeError("Use simulator.p.p * state.omega_r for omega_e")


# ---------------------------------------------------------------------------
# Standalone helper functions (used by simulator and by MPC / MCP server)
# ---------------------------------------------------------------------------

def flux_linkages(params: IPMSMParameters, i_d: float, i_q: float) -> tuple[float, float]:
    """
    Flux linkages (psi_d, psi_q) at given currents, including saturation.
    These are the quantities measured by a flux observer on a real drive.
    """
    return (params.L_d_eff(i_d, i_q) * i_d + params.psi_f,
            params.L_q_eff(i_d, i_q) * i_q)


def torque(params: IPMSMParameters, i_d: float, i_q: float) -> float:
    """Electromagnetic torque [N·m], accounting for saturation."""
    Ld = params.L_d_eff(i_d, i_q)
    Lq = params.L_q_eff(i_d, i_q)
    return (3 / 2) * params.p * (params.psi_f * i_q + (Ld - Lq) * i_d * i_q)


def mtpa(params: IPMSMParameters, i_ref: float) -> tuple[float, float]:
    """
    MTPA (Maximum Torque Per Ampere) operating point.

    Finds (i_d, i_q) on the current limit circle |i| = i_ref that maximises
    electromagnetic torque. For IPMSM, i_d < 0 (demagnetising).

    Analytical solution of the MTPA condition (dT_e/d(i_d) = 0 with constraint):

        i_d = (psi_f - sqrt(psi_f^2 + 8*(L_q - L_d)^2 * i_ref^2))
              / (4 * (L_q - L_d))

    Returns (i_d, i_q). i_ref is the current magnitude.
    """
    delta_L = params.L_q - params.L_d   # > 0 for IPMSM
    if abs(delta_L) < 1e-10:
        # Surface PMSM: no saliency, MTPA is trivially i_d = 0
        return 0.0, i_ref

    discriminant = params.psi_f ** 2 + 8 * delta_L ** 2 * i_ref ** 2
    i_d = (params.psi_f - np.sqrt(discriminant)) / (4 * delta_L)
    i_q = np.sqrt(max(i_ref ** 2 - i_d ** 2, 0.0))
    return i_d, i_q


def field_weakening(
    params: IPMSMParameters, omega_r: float, i_ref: float
) -> tuple[float, float]:
    """
    Field weakening operating point above base speed.

    Finds (i_d, i_q) satisfying simultaneously:
      - voltage limit:  (omega_e * L_d * i_d + omega_e * psi_f)^2
                      + (omega_e * L_q * i_q)^2 = v_max^2    (resistive drop neglected)
      - current limit:  i_d^2 + i_q^2 = i_ref^2

    Returns (i_d, i_q). If speed is below base speed, delegates to mtpa().
    """
    omega_e = params.p * omega_r
    if omega_e < 1e-3:
        return mtpa(params, i_ref)

    # Voltage ellipse: ((i_d + psi_f/L_d) / (v_max/(omega_e*L_d)))^2
    #                + (i_q            / (v_max/(omega_e*L_q)))^2 = 1
    a = params.v_max / (omega_e * params.L_d)   # semi-axis along d
    b = params.v_max / (omega_e * params.L_q)   # semi-axis along q
    center_d = -params.psi_f / params.L_d       # ellipse center on d-axis

    # Check if MTPA point is inside the voltage ellipse
    i_d_mtpa, i_q_mtpa = mtpa(params, i_ref)
    on_ellipse = ((i_d_mtpa - center_d) / a) ** 2 + (i_q_mtpa / b) ** 2
    if on_ellipse <= 1.0:
        return i_d_mtpa, i_q_mtpa  # voltage not saturated yet

    # Intersect current circle and voltage ellipse numerically
    from scipy.optimize import brentq

    def residual(i_d_val: float) -> float:
        # From voltage ellipse: i_q_v
        arg = 1.0 - ((i_d_val - center_d) / a) ** 2
        if arg < 0:
            return -i_ref   # outside ellipse on d-side
        i_q_v = b * np.sqrt(arg)
        # From current circle: i_q_c
        arg2 = i_ref ** 2 - i_d_val ** 2
        if arg2 < 0:
            return i_q_v   # outside current circle
        i_q_c = np.sqrt(arg2)
        return i_q_v - i_q_c

    # Search bounds: d-axis from -i_ref to 0
    try:
        i_d_sol = brentq(residual, -i_ref, 0.0, xtol=1e-6)
        i_q_sol = np.sqrt(max(i_ref ** 2 - i_d_sol ** 2, 0.0))
    except ValueError:
        # No intersection: deep field weakening, operate at voltage limit only
        i_d_sol = max(center_d - a, -i_ref)
        i_q_sol = np.sqrt(max(i_ref ** 2 - i_d_sol ** 2, 0.0))

    return i_d_sol, i_q_sol


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class IPMSMSimulator:
    """
    IPMSM simulator with RK4 integration.

    Typical usage:
        sim = IPMSMSimulator(IPMSMParameters())
        state = sim.step(v_d=0.0, v_q=50.0, T_load=2.0)
        print(state.speed_rpm)
    """

    def __init__(self, params: Optional[IPMSMParameters] = None):
        self.p = params or IPMSMParameters()
        # State vector: [i_d, i_q, omega_r, theta_r]
        self._x = np.zeros(4)
        self.time = 0.0
        self._last_u = np.zeros(2)
        self._last_T_load = 0.0

    # ------------------------------------------------------------------
    # Core integration
    # ------------------------------------------------------------------

    def _derivatives(self, x: np.ndarray, u: np.ndarray, T_load: float) -> np.ndarray:
        i_d, i_q, omega_r, _ = x
        v_d, v_q = u
        p = self.p
        omega_e = p.p * omega_r
        Ld = p.L_d_eff(i_d, i_q)
        Lq = p.L_q_eff(i_d, i_q)

        di_d = (v_d - p.R_s * i_d + omega_e * Lq * i_q) / Ld
        di_q = (v_q - p.R_s * i_q - omega_e * (Ld * i_d + p.psi_f)) / Lq

        T_e = torque(p, i_d, i_q)
        domega_r = (T_e - T_load - p.B * omega_r) / p.J
        dtheta_r = omega_r

        return np.array([di_d, di_q, domega_r, dtheta_r])

    def _clamp_voltage(self, u: np.ndarray) -> np.ndarray:
        """Clamp voltage vector to the voltage limit circle."""
        mag = np.linalg.norm(u)
        if mag > self.p.v_max:
            u = u * self.p.v_max / mag
        return u

    def step(self, v_d: float, v_q: float, T_load: float = 0.0,
             lock_speed: bool = False) -> DriveState:
        """
        Advance the simulation by one time step (dt).

        Args:
            v_d:    d-axis voltage command [V]
            v_q:    q-axis voltage command [V]
            T_load: load torque            [N·m]

        Returns:
            DriveState snapshot after the step.
        """
        u = self._clamp_voltage(np.array([v_d, v_q]))
        self._last_u = u
        self._last_T_load = T_load
        x = self._x
        dt = self.p.dt

        k1 = self._derivatives(x, u, T_load)
        k2 = self._derivatives(x + 0.5 * dt * k1, u, T_load)
        k3 = self._derivatives(x + 0.5 * dt * k2, u, T_load)
        k4 = self._derivatives(x + dt * k3, u, T_load)

        self._x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if lock_speed:
            self._x[2] = x[2]   # hold mechanical speed fixed (test-bench mode)
        self._x[3] %= 2 * np.pi   # wrap rotor angle
        self.time += dt

        return self._make_state(u, T_load)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def reset(self, i_d: float = 0.0, i_q: float = 0.0,
              omega_r: float = 0.0, theta_r: float = 0.0) -> None:
        """Reset state to given initial conditions."""
        self._x = np.array([i_d, i_q, omega_r, theta_r])
        self.time = 0.0

    @property
    def state(self) -> DriveState:
        """Current state without advancing time."""
        return self._make_state(self._last_u, self._last_T_load)

    def _make_state(self, u: np.ndarray, T_load: float) -> DriveState:
        i_d, i_q, omega_r, theta_r = self._x
        return DriveState(
            time=self.time,
            i_d=i_d,
            i_q=i_q,
            omega_r=omega_r,
            theta_r=theta_r,
            T_e=torque(self.p, i_d, i_q),
            T_load=T_load,
            v_d=u[0],
            v_q=u[1],
        )

    def run(
        self,
        n_steps: int,
        voltage_fn,        # callable(t, state) -> (v_d, v_q)
        load_fn=None,      # callable(t, state) -> T_load
    ) -> list[DriveState]:
        """
        Run the simulator for n_steps, driven by callable voltage and load functions.
        Useful for scripted scenarios and validation.
        """
        if load_fn is None:
            load_fn = lambda t, s: 0.0

        history = []
        for _ in range(n_steps):
            s = self.state
            v_d, v_q = voltage_fn(self.time, s)
            T_load = load_fn(self.time, s)
            history.append(self.step(v_d, v_q, T_load))
        return history
