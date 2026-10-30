"""Visualise DP observer performance in a simulated sea state.

Usage examples
--------------
    python observer_viz.py                      # asks which observer to use
    python observer_viz.py -o kalman
    python observer_viz.py -o nonlinear_passive --duration 1800 --seed 3
    python observer_viz.py -o both --scenario step --save result.png
    python observer_viz.py -o kalman --module my_observer   # your file name

The observer module is imported by name (default: "observer"; change with
--module).  The script provides a stand-in for ``simulation.utils`` if that
package is not importable, so your file runs unchanged.

Simulation
----------
* 3-DOF vessel (surge, sway, yaw) with M, D from Appendix B, true bias force
  (NED) = constant + slow sinusoid, first-order thruster lag.
* Wave-frequency (WF) motion: sum of sinusoids with a Pierson-Moskowitz shape
  (Tp = 8 s), scaled so std = Hs/4 in N/E and 0.087 rad in yaw. It is added
  to the measurement only (no effect on the LF dynamics).
* Measurement: y = eta_LF + eta_WF + noise (0.2 m, 0.2 deg).
* Control loop: PD + bias feedforward driven by the OBSERVER estimates, so a
  poor observer shows up as poor station keeping.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import types

import numpy as np

# --------------------------------------------------------------------------
# Stand-in for simulation.utils (only used if the real package is missing)
# --------------------------------------------------------------------------
try:
    import simulation.utils  # noqa: F401
except Exception:  # pragma: no cover

    def _wrap(a):
        return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi

    def _Rz(psi):
        c, s = np.cos(psi), np.sin(psi)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    _utils = types.ModuleType("simulation.utils")
    _utils.Rz = _Rz
    _utils.wrap_angle_pi = lambda a: float(_wrap(a)) if np.ndim(a) == 0 else _wrap(a)
    _pkg = types.ModuleType("simulation")
    _pkg.utils = _utils
    sys.modules["simulation"] = _pkg
    sys.modules["simulation.utils"] = _utils

import matplotlib.pyplot as plt  # noqa: E402

# --------------------------------------------------------------------------
# Plant parameters
# --------------------------------------------------------------------------
M = np.array([[6.007e5, 0.0, 0.0], [0.0, 7.067e5, -4.733e5], [0.0, -5.712e5, 5.456e7]])
D = np.diag([1.118e3, 2.229e4, 1.950e6])
M_INV = np.linalg.inv(M)
IDX = [0, 1, 5]


def Rz(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
def wave_motion(t, Tp, Hs, sigma_yaw, rng, n_comp=60):
    """WF motion (N, E, psi) as sums of sinusoids with a PM-shaped spectrum."""
    wp = 2 * np.pi / Tp
    w = np.linspace(0.4 * wp, 3.0 * wp, n_comp)
    dw = w[1] - w[0]
    S = w**-5 * np.exp(-1.25 * (wp / w) ** 4)  # PM shape (unscaled)
    sig = np.array([Hs / 4, Hs / 4, sigma_yaw])
    out = np.zeros((len(t), 3))
    for k in range(3):
        a = np.sqrt(2 * S * dw)
        a *= sig[k] / np.sqrt(0.5 * np.sum(a**2))  # scale to target std
        phase = rng.uniform(0, 2 * np.pi, n_comp)
        out[:, k] = (a[None, :] * np.cos(np.outer(t, w) + phase)).sum(axis=1)
    return out


def true_bias(t):
    """Slowly varying environmental force in NED [N, N, Nm]."""
    mean = np.array([20e3, -15e3, 30e3])
    amp = np.array([10e3, 8e3, 15e3])
    return mean + amp * np.sin(2 * np.pi * t / 600.0 + np.array([0.0, 1.0, 2.0]))


# --------------------------------------------------------------------------
# Controller (uses observer estimates)
# --------------------------------------------------------------------------
class PDController:
    def __init__(self, wn=(0.06, 0.06, 0.08), zeta=1.0):
        wn = np.asarray(wn)
        Mii = np.diag(M)
        self.Kp = np.diag(Mii * wn**2)
        self.Kd = np.diag(np.maximum(2 * zeta * wn * Mii - np.diag(D), 0.0))
        self.sat = np.array([3e5, 3e5, 3e6])

    def __call__(self, eta_hat, nu_hat, b_hat, eta_ref):
        e = eta_hat[IDX] - eta_ref
        e[2] = wrap(e[2])
        psi = eta_hat[5]
        tau = -self.Kp @ (Rz(psi).T @ e) - self.Kd @ nu_hat[IDX] - Rz(psi).T @ b_hat[IDX]
        return np.clip(tau, -self.sat, self.sat)


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def simulate(
    observer,
    duration=1800.0,
    dt=0.1,
    Tp=8.0,
    Hs=1.5,
    seed=0,
    scenario="hold",
    sigma_pos=0.2,
    sigma_psi_deg=0.2,
    T_thr=3.0,
):
    rng = np.random.default_rng(seed)
    n = int(duration / dt) + 1
    t = np.arange(n) * dt

    eta_wf = wave_motion(t, Tp, Hs, 0.087, rng)
    noise = rng.standard_normal((n, 3)) * np.array([sigma_pos, sigma_pos, np.deg2rad(sigma_psi_deg)])

    # reference
    ref = np.zeros((n, 3))
    if scenario == "step":
        ref[t >= 300.0, 0] = 10.0  # 10 m north
        ref[t >= 600.0, 1] = 10.0  # 10 m east
        ref[t >= 900.0, 2] = np.deg2rad(20.0)  # 20 deg heading
    elif scenario != "hold":
        raise ValueError("scenario must be 'hold' or 'step'")

    # true LF state (3-DOF) and initial offset so convergence is visible
    eta = np.array([2.0, -1.0, np.deg2rad(3.0)])
    nu = np.zeros(3)
    tau_act = np.zeros(3)
    tau_cmd_prev = np.zeros(6)

    ctrl = PDController()
    eta0_6 = np.zeros(6)
    eta0_6[IDX] = eta
    observer.reset(eta0_6)

    log = {k: np.zeros((n, 3)) for k in ["eta", "nu", "b", "eta_hat", "nu_hat", "b_hat", "y"]}
    log["tau"] = np.zeros((n, 3))

    for k in range(n):
        b = true_bias(t[k])
        y6 = np.zeros(6)
        y6[IDX] = eta + eta_wf[k] + noise[k]
        y6[5] = wrap(y6[5])

        est = observer.step(t[k], dt, y6, tau_cmd_prev)
        if isinstance(est, tuple):
            eta_h, nu_h, b_h = (np.asarray(v, float) for v in est)
        else:
            eta_h, nu_h, b_h = est.eta, est.nu, est.bias

        tau_cmd = ctrl(eta_h, nu_h, b_h, ref[k])
        tau_cmd_prev = np.zeros(6)
        tau_cmd_prev[IDX] = tau_cmd

        log["eta"][k], log["nu"][k], log["b"][k] = eta, nu, b
        log["eta_hat"][k], log["nu_hat"][k], log["b_hat"][k] = eta_h[IDX], nu_h[IDX], b_h[IDX]
        log["y"][k] = y6[IDX]
        log["tau"][k] = tau_act

        # plant update (forward Euler)
        tau_act = tau_act + dt / T_thr * (tau_cmd - tau_act)
        nu_dot = M_INV @ (-D @ nu + Rz(eta[2]).T @ b + tau_act)
        eta = eta + dt * (Rz(eta[2]) @ nu)
        nu = nu + dt * nu_dot
        eta[2] = wrap(eta[2])

    log["t"] = t
    log["eta_wf"] = eta_wf
    log["ref"] = ref
    return log


def metrics(log, t_skip=200.0):
    m = log["t"] >= t_skip
    err_eta = log["eta_hat"] - log["eta"]
    err_eta[:, 2] = wrap(err_eta[:, 2])
    rmse = lambda e: np.sqrt(np.mean(e[m] ** 2, axis=0))
    return {
        "pos/heading [m,m,rad]": rmse(err_eta),
        "velocity [m/s,m/s,rad/s]": rmse(log["nu_hat"] - log["nu"]),
        "bias [N,N,Nm]": rmse(log["b_hat"] - log["b"]),
        "estimate vs ref [m,m,rad]": rmse(
            np.column_stack([log["eta_hat"][:, :2] - log["ref"][:, :2], wrap(log["eta_hat"][:, 2] - log["ref"][:, 2])])
        ),
    }


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_results(logs, names, save=None, zoom=(600.0, 700.0)):
    colors = ["tab:red", "tab:green"]
    t = logs[0]["t"]
    L = logs[0]
    dofs = ["North [m]", "East [m]", "Heading [deg]"]
    vel = ["u [m/s]", "v [m/s]", "r [deg/s]"]
    bia = ["b_N [kN]", "b_E [kN]", "b_psi [kNm]"]
    sc = [1, 1, 180 / np.pi]
    scv = [1, 1, 180 / np.pi]

    fig, ax = plt.subplots(3, 4, figsize=(19, 10), constrained_layout=True)
    fig.suptitle("Observer comparison: " + ", ".join(names) + "   (Tp = 8 s, Hs = 1.5 m)", fontsize=13)

    for i in range(3):
        # Column 0: pose
        a = ax[i, 0]
        a.plot(t, L["y"][:, i] * sc[i], color="0.75", lw=0.6, label="measurement")
        a.plot(t, L["eta"][:, i] * sc[i], "k", lw=1.2, label="true LF")
        if np.any(L["ref"][:, i]):
            a.plot(t, L["ref"][:, i] * sc[i], "k--", lw=0.8, label="reference")
        for lg, nm, c in zip(logs, names, colors):
            a.plot(t, lg["eta_hat"][:, i] * sc[i], color=c, lw=1.0, label=f"{nm} est.")
        a.set_ylabel(dofs[i])
        a.grid(alpha=0.3)
        if i == 0:
            a.set_title("Pose")
            a.legend(fontsize=7, loc="best")

        # Column 1: zoom on pose (wave filtering)
        a = ax[i, 1]
        m = (t >= zoom[0]) & (t <= zoom[1])
        a.plot(t[m], L["y"][m, i] * sc[i], color="0.7", lw=0.8, label="measurement")
        a.plot(t[m], L["eta"][m, i] * sc[i], "k", lw=1.5, label="true LF")
        for lg, nm, c in zip(logs, names, colors):
            a.plot(t[m], lg["eta_hat"][m, i] * sc[i], color=c, lw=1.2)
        a.grid(alpha=0.3)
        if i == 0:
            a.set_title(f"Zoom {zoom[0]:.0f}-{zoom[1]:.0f} s (wave filtering)")

        # Column 2: velocity
        a = ax[i, 2]
        a.plot(t, L["nu"][:, i] * scv[i], "k", lw=1.2, label="true")
        for lg, nm, c in zip(logs, names, colors):
            a.plot(t, lg["nu_hat"][:, i] * scv[i], color=c, lw=0.8, label=nm)
        a.set_ylabel(vel[i])
        a.grid(alpha=0.3)
        if i == 0:
            a.set_title("Body velocity")

        # Column 3: bias
        a = ax[i, 3]
        a.plot(t, L["b"][:, i] / 1e3, "k", lw=1.2, label="true")
        for lg, nm, c in zip(logs, names, colors):
            a.plot(t, lg["b_hat"][:, i] / 1e3, color=c, lw=1.0, label=nm)
        a.set_ylabel(bia[i])
        a.grid(alpha=0.3)
        if i == 0:
            a.set_title("Bias (NED)")

    for a in ax[2, :]:
        a.set_xlabel("t [s]")

    if save:
        fig.savefig(save, dpi=150)
        print(f"Saved figure to {save}")
    return fig


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def choose_observer_interactively():
    opts = ["kalman", "nonlinear_passive", "both"]
    print("Choose observer:")
    for i, o in enumerate(opts, 1):
        print(f"  {i}) {o}")
    while True:
        s = input("> ").strip()
        if s in opts:
            return s
        if s.isdigit() and 1 <= int(s) <= len(opts):
            return opts[int(s) - 1]
        print("Invalid choice.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--observer", choices=["kalman", "nonlinear_passive", "both"])
    p.add_argument("--module", default="observer", help="module containing select_observer (default: observer)")
    p.add_argument("--duration", type=float, default=1800.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--Tp", type=float, default=8.0)
    p.add_argument("--Hs", type=float, default=1.5)
    p.add_argument("--scenario", choices=["hold", "step"], default="hold")
    p.add_argument("--save", default=None, help="save figure to this path")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    kind = args.observer or choose_observer_interactively()
    kinds = ["kalman", "nonlinear_passive"] if kind == "both" else [kind]

    mod = importlib.import_module(args.module)
    logs = []
    for k in kinds:
        obs = mod.select_observer(k)
        lg = simulate(
            obs, duration=args.duration, dt=args.dt, Tp=args.Tp, Hs=args.Hs, seed=args.seed, scenario=args.scenario
        )
        logs.append(lg)
        print(f"\n=== {k}: RMSE after 200 s ===")
        for name, v in metrics(lg).items():
            print(f"  {name:28s}", np.array2string(v, precision=4))

    plot_results(logs, kinds, save=args.save)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
