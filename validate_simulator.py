"""
Validation: FCS-MPC vs PI current controller on IPMSM.

Demonstrates commissioning-relevant comparisons:
  1. MTPA vs i_d=0 current reference strategies (PI controller baseline)
  2. FCS-MPC with λ_sw=0 (aggressive) vs λ_sw>0 (switching-penalised)
  3. Load step disturbance rejection (speed + torque)
"""

import numpy as np
import matplotlib.pyplot as plt

from simulator import IPMSMParameters, IPMSMSimulator
from simulator.ipmsm import mtpa, torque
from controller import FCSMPCWeights, FCSMPCParameters, FCSMPCController


# ---------------------------------------------------------------------------
# Simple PI current controller (baseline for comparison)
# ---------------------------------------------------------------------------

class PICurrentController:
    def __init__(self, params: IPMSMParameters, bw_hz: float = 200.0):
        self.p = params
        omega_bw = 2 * np.pi * bw_hz
        self.kp_d = omega_bw * params.L_d
        self.ki_d = omega_bw * params.R_s
        self.kp_q = omega_bw * params.L_q
        self.ki_q = omega_bw * params.R_s
        self._int_d = 0.0
        self._int_q = 0.0

    def step(self, i_d_ref, i_q_ref, state, dt):
        err_d = i_d_ref - state.i_d
        err_q = i_q_ref - state.i_q
        self._int_d += err_d * dt
        self._int_q += err_q * dt
        omega_e = self.p.p * state.omega_r
        v_d = (self.kp_d * err_d + self.ki_d * self._int_d
               - omega_e * self.p.L_q * state.i_q)
        v_q = (self.kp_q * err_q + self.ki_q * self._int_q
               + omega_e * (self.p.L_d * state.i_d + self.p.psi_f))
        return v_d, v_q


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------

def run_pi(params, i_d_ref, i_q_ref, t_end, T_load=0.0):
    sim = IPMSMSimulator(params)
    ctrl = PICurrentController(params)
    history = []
    t = 0.0
    while t < t_end:
        v_d, v_q = ctrl.step(i_d_ref, i_q_ref, sim.state, params.dt)
        history.append(sim.step(v_d, v_q, T_load=T_load))
        t += params.dt
    return history


def run_fcs_mpc(params, i_d_ref, i_q_ref, t_end, T_load=0.0,
                weights=None, horizon=1):
    sim = IPMSMSimulator(params)
    cp = FCSMPCParameters(horizon=horizon, weights=weights or FCSMPCWeights())
    ctrl = FCSMPCController(params, cp)
    ctrl.set_reference(i_d_ref, i_q_ref)
    history = []
    actions = []
    t = 0.0
    while t < t_end:
        action = ctrl.step(sim.state)
        history.append(sim.step(action.v_d, action.v_q, T_load=T_load))
        actions.append(action)
        t += params.dt
    return history, actions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    params = IPMSMParameters()
    print("=== Motor parameters ===")
    for k, v in params.describe().items():
        print(f"  {k}: {v}")

    i_ref = 8.0
    i_d_mtpa, i_q_mtpa = mtpa(params, i_ref)
    t_end = 0.05   # 50 ms is enough to see current step transient at dt=100µs

    print(f"\nMTPA reference: i_d={i_d_mtpa:.2f} A, i_q={i_q_mtpa:.2f} A")

    # --- Scenario 1: PI MTPA vs i_d=0 ---
    print("\n=== Scenario 1: PI — MTPA vs i_d=0 ===")
    hist_pi_mtpa  = run_pi(params, i_d_mtpa, i_q_mtpa, t_end)
    hist_pi_id0   = run_pi(params, 0.0,      i_ref,    t_end)

    # --- Scenario 2: FCS-MPC — switching penalty trade-off ---
    print("=== Scenario 2: FCS-MPC — λ_sw=0 vs λ_sw=0.5 ===")
    hist_mpc_fast, act_fast = run_fcs_mpc(
        params, i_d_mtpa, i_q_mtpa, t_end,
        weights=FCSMPCWeights(switching=0.0),
    )
    hist_mpc_slow, act_slow = run_fcs_mpc(
        params, i_d_mtpa, i_q_mtpa, t_end,
        weights=FCSMPCWeights(switching=0.5),
    )

    # Count average switching transitions per step
    def avg_sw(actions):
        from controller.fcs_mpc import count_transitions, _SWITCHING_STATES
        sw_seq = [a.switching_state for a in actions]
        transitions = [count_transitions(sw_seq[k+1], sw_seq[k])
                       for k in range(len(sw_seq)-1)]
        return np.mean(transitions)

    print(f"  λ_sw=0.0: avg transitions/step = {avg_sw(act_fast):.2f}")
    print(f"  λ_sw=0.5: avg transitions/step = {avg_sw(act_slow):.2f}")

    # --- Plotting ---
    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    fig.suptitle("IPMSM FCS-MPC Validation", fontsize=13)

    # Column 1: PI MTPA vs id=0
    for hist, label, ls in [
        (hist_pi_mtpa, "PI  MTPA", "-"),
        (hist_pi_id0,  "PI  id=0", "--"),
    ]:
        t   = [s.time for s in hist]
        axes[0, 0].plot(t, [s.i_d for s in hist], ls=ls, label=label)
        axes[1, 0].plot(t, [s.i_q for s in hist], ls=ls, label=label)
        axes[2, 0].plot(t, [s.T_e for s in hist], ls=ls, label=label)

    axes[0, 0].axhline(i_d_mtpa, color='r', lw=0.8, ls=':')
    axes[1, 0].axhline(i_q_mtpa, color='r', lw=0.8, ls=':', label="MTPA ref")
    axes[1, 0].axhline(i_ref,    color='g', lw=0.8, ls=':', label="id=0 ref")

    axes[0, 0].set_title("PI: d-axis current")
    axes[1, 0].set_title("PI: q-axis current")
    axes[2, 0].set_title("PI: torque")
    axes[0, 0].set_ylabel("i_d [A]");  axes[1, 0].set_ylabel("i_q [A]")
    axes[2, 0].set_ylabel("T_e [N·m]"); axes[2, 0].set_xlabel("Time [s]")
    for ax in axes[:, 0]:
        ax.legend(fontsize=8); ax.grid(True)

    # Column 2: FCS-MPC switching penalty comparison
    for hist, actions, label, ls in [
        (hist_mpc_fast, act_fast, "FCS-MPC λ_sw=0.0", "-"),
        (hist_mpc_slow, act_slow, "FCS-MPC λ_sw=0.5", "--"),
    ]:
        t   = [s.time for s in hist]
        axes[0, 1].plot(t, [s.i_d for s in hist], ls=ls, label=label)
        axes[1, 1].plot(t, [s.i_q for s in hist], ls=ls, label=label)

        # Switching activity: number of transitions per step
        sw_seq = [a.switching_state for a in actions]
        from controller.fcs_mpc import count_transitions
        trans = [count_transitions(sw_seq[k+1], sw_seq[k]) for k in range(len(sw_seq)-1)]
        axes[2, 1].plot(t[1:], trans, ls=ls, alpha=0.7, label=label)

    axes[0, 1].axhline(i_d_mtpa, color='r', lw=0.8, ls=':', label="ref")
    axes[1, 1].axhline(i_q_mtpa, color='r', lw=0.8, ls=':', label="ref")

    axes[0, 1].set_title("FCS-MPC: d-axis current")
    axes[1, 1].set_title("FCS-MPC: q-axis current")
    axes[2, 1].set_title("FCS-MPC: switching transitions per step")
    axes[0, 1].set_ylabel("i_d [A]"); axes[1, 1].set_ylabel("i_q [A]")
    axes[2, 1].set_ylabel("transitions"); axes[2, 1].set_xlabel("Time [s]")
    for ax in axes[:, 1]:
        ax.legend(fontsize=8); ax.grid(True)

    plt.tight_layout()
    plt.savefig("validation.png", dpi=150)
    print("\nPlot saved to validation.png")
    plt.show()
