"""
Tier 2 controller test harness (controller-only, nonlinear plant).

Purpose
-------
Drive the real nonlinear Gunnerus plant with ONLY your DPController, bypassing
the reference model, thrust allocation, current, and wind blocks (none of
which need to be finished for this test). Used to generate the evidence for
the report's "integral growth / infeasible force demands" section:

  1. Steady-state offset under a constant ocean current (motivates LQI).
  2. Thruster saturation under a random wind disturbance (motivates
     anti-windup on the integral state).

This is NOT your final closed loop. In particular the allocator here is a
throwaway ideal pseudo-inverse allocator, built only so tau_d can reach the
plant -- it is not your teammate's real part_1/thrust_allocation.py.

Drop this file anywhere on your PYTHONPATH that can `import part_1` and
`import simulation` (e.g. the repo root) and run it directly.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from part_1.config import default_thrusters_gunnerus3
from part_1.controller import DPController
from simulation.plant import GunnerusPlant3DOF
from simulation.utils import wrap_angle_pi


# --------------------------------------------------------------------------
# Throwaway ideal allocator (pseudo-inverse), for controller testing only.
# --------------------------------------------------------------------------
def build_ideal_allocator(thrusters):
    """Return an allocate(tau_d3) -> (thrust_cmd, azimuth_cmd) function.

    Fixed-direction thrusters (rot_speed == 0, e.g. the tunnel) get one
    unknown (signed magnitude along their fixed alpha0). Free azimuths
    (rot_speed > 0) get two unknowns (Fx, Fy), later converted to
    magnitude + angle. Uses the Moore-Penrose pseudo-inverse of the
    resulting configuration matrix -- exact when the system is
    controllable, not force/rate-limit aware beyond a final clip to u_max.
    """
    specs = []  # (thruster, is_fixed)
    cols = []  # columns of the 3xN configuration matrix
    for th in thrusters:
        if th.rot_speed == 0.0:
            # fixed direction: one unknown u, force = u*[cos a0, sin a0]
            c, s = np.cos(th.alpha0), np.sin(th.alpha0)
            col = np.array([c, s, th.x * s - th.y * c])
            cols.append(col)
            specs.append((th, True))
        else:
            # free azimuth: two unknowns Fx, Fy
            cols.append(np.array([1.0, 0.0, -th.y]))
            cols.append(np.array([0.0, 1.0, th.x]))
            specs.append((th, False))

    B = np.column_stack(cols)  # (3, N_unknowns)
    B_pinv = np.linalg.pinv(B)

    def allocate(tau_d3: np.ndarray):
        virtual = B_pinv @ tau_d3
        thrust_cmd, azimuth_cmd = [], []
        i = 0
        for th, is_fixed in specs:
            if is_fixed:
                u = virtual[i]
                i += 1
                alpha = th.alpha0
            else:
                Fx, Fy = virtual[i], virtual[i + 1]
                i += 2
                u = float(np.hypot(Fx, Fy))
                alpha = float(np.arctan2(Fy, Fx))
            u_sat = float(np.clip(u, -th.u_max, th.u_max))  # honest saturation
            thrust_cmd.append(u_sat)
            azimuth_cmd.append(alpha)
        return np.array(thrust_cmd), np.array(azimuth_cmd)

    return allocate


# --------------------------------------------------------------------------
# Disturbances
# --------------------------------------------------------------------------
def steady_current_ned(speed: float, beta_deg: float) -> np.ndarray:
    """Constant NED current vector, 'towards' convention (0 deg = North)."""
    beta = np.deg2rad(beta_deg)
    return np.array([speed * np.cos(beta), speed * np.sin(beta), 0, 0, 0, 0])


class GaussMarkovWind:
    """Band-limited random BODY-frame force disturbance (Ornstein-Uhlenbeck).

    Stands in for wind gust content -- NOT your Wind.step() model. Time
    constant `tau` sets how slowly it wanders; `sigma` sets its steady-state
    standard deviation [N] on Fx, Fy and [Nm] (scaled) on Mz. Must be called
    strictly in increasing time order (true here since GunnerusPlant3DOF.run
    calls loads sequentially).
    """

    def __init__(self, sigma_force=8_000.0, sigma_moment=40_000.0, tau=60.0, seed=0):
        self.sigma_force = sigma_force
        self.sigma_moment = sigma_moment
        self.tau = tau
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(3)  # [Fx, Fy, Mz]
        self.last_t = None

    def __call__(self, t, eta, nu):
        dt = 0.0 if self.last_t is None else max(t - self.last_t, 0.0)
        self.last_t = t
        if dt > 0.0:
            sig = np.array([self.sigma_force, self.sigma_force, self.sigma_moment])
            decay = np.exp(-dt / self.tau)
            noise_std = sig * np.sqrt(1 - decay**2)  # stationary-variance OU update
            self.state = self.state * decay + self.rng.normal(0.0, 1.0, 3) * noise_std
        return np.array([self.state[0], self.state[1], 0, 0, 0, self.state[2]])


# --------------------------------------------------------------------------
# Main test loop
# --------------------------------------------------------------------------
def run_tier2(
    T=600.0,
    dt=0.05,
    eta_cmd=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    current_speed=0.5,
    current_beta_deg=90.0,  # beam-on: 0=North, 90=East, worst case for sway/yaw
    wind_sigma_force=8_000.0,  # N; set 0.0 to disable wind noise
    wind_sigma_moment=40_000.0,
    wind_tau=60.0,
    seed=0,
    controller_kwargs=None,
):
    controller_kwargs = controller_kwargs or {}
    controller = DPController(**controller_kwargs)
    if hasattr(controller, "reset"):
        controller.reset()

    thrusters = default_thrusters_gunnerus3()
    plant = GunnerusPlant3DOF(
        thrusters, dt=dt, method="Euler", thruster_dynamics=False
    )  # Part 1: ideal actuators
    allocate = build_ideal_allocator(thrusters)

    eta_cmd = np.asarray(eta_cmd, dtype=float)
    nu_ref = np.zeros(6)  # stationkeeping: zero reference velocity

    current_ned = steady_current_ned(current_speed, current_beta_deg)
    wind = GaussMarkovWind(wind_sigma_force, wind_sigma_moment, wind_tau, seed=seed)

    plant.reset()
    n = int(round(T / dt))
    t_hist = np.arange(n) * dt
    eta_hist = np.zeros((n, 6))
    nu_hist = np.zeros((n, 6))
    tau_d_hist = np.zeros((n, 6))  # desired wrench (controller output)
    tau_applied_hist = np.zeros((n, 6))  # tau_thrusters actually delivered
    thrust_hist = np.zeros((n, len(thrusters)))

    eta = np.zeros(6)
    nu = np.zeros(6)
    for k, t in enumerate(t_hist):
        tau_d = np.asarray(
            controller.compute(t, dt, eta, nu, eta_cmd, nu_ref), dtype=float
        )
        tau_d3 = np.array([tau_d[0], tau_d[1], tau_d[5]])

        thrust_cmd, azimuth_cmd = allocate(tau_d3)

        step = plant.step(
            t,
            thrust_cmd,
            azimuth_cmd,
            current=current_ned,
            wind=wind,
        )
        eta, nu = step.eta, step.nu

        # feed the actually-applied wrench back for anti-windup, if defined
        if hasattr(controller, "apply_external_aw"):
            controller.apply_external_aw(step.tau_total, eta[5], dt)

        eta_hist[k] = eta
        nu_hist[k] = nu
        tau_d_hist[k] = tau_d
        tau_applied_hist[k] = step.tau_thrusters
        thrust_hist[k] = step.thrust

    return {
        "t": t_hist,
        "eta": eta_hist,
        "nu": nu_hist,
        "tau_d": tau_d_hist,
        "tau_applied": tau_applied_hist,
        "thrust": thrust_hist,
        "thrusters": thrusters,
        "eta_cmd": eta_cmd,
    }


def plot_results(res, title=""):
    t = res["t"]
    eta = res["eta"]
    eta_cmd = res["eta_cmd"]
    thrusters = res["thrusters"]

    fig, axs = plt.subplots(4, 1, figsize=(9, 11), sharex=True)

    axs[0].plot(t, eta[:, 0] - eta_cmd[0], label="N error [m]")
    axs[0].plot(t, eta[:, 1] - eta_cmd[1], label="E error [m]")
    axs[0].axhline(0, color="k", lw=0.5)
    axs[0].set_ylabel("Position error [m]")
    axs[0].legend()
    axs[0].set_title(f"Tier 2 test: {title}")

    psi_err = np.array([wrap_angle_pi(p - eta_cmd[5]) for p in eta[:, 5]])
    axs[1].plot(t, np.rad2deg(psi_err), color="tab:red", label="psi error [deg]")
    axs[1].axhline(0, color="k", lw=0.5)
    axs[1].set_ylabel("Heading error [deg]")
    axs[1].legend()

    for i in range(3):
        axs[2].plot(t, res["thrust"][:, i] / 1000.0, label=thrusters[i].name)
        axs[2].axhline(thrusters[i].u_max / 1000.0, color="gray", lw=0.5, ls="--")
        axs[2].axhline(-thrusters[i].u_max / 1000.0, color="gray", lw=0.5, ls="--")
    axs[2].set_ylabel("Thrust command [kN]")
    axs[2].legend()

    resid = res["tau_d"] - res["tau_applied"]
    axs[3].plot(t, resid[:, 0] / 1000.0, label="Fx residual [kN]")
    axs[3].plot(t, resid[:, 1] / 1000.0, label="Fy residual [kN]")
    axs[3].plot(t, resid[:, 5] / 1000.0, label="Mz residual [kNm]")
    axs[3].set_ylabel("tau_d - tau_applied")
    axs[3].set_xlabel("time [s]")
    axs[3].legend()

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # --- Test A: steady current only, no wind -> motivates the integrator
    res_current = run_tier2(
        T=600.0,
        current_speed=0.5,
        current_beta_deg=90.0,
        wind_sigma_force=0.0,
        wind_sigma_moment=0.0,
    )
    fig_a = plot_results(res_current, "steady current only (beam-on, 0.5 m/s)")
    print(
        "Steady-state N,E,psi error (last 50 s mean):",
        res_current["eta"][-1000:, [0, 1, 5]].mean(axis=0),
    )

    # --- Test B: steady current + random wind -> motivates anti-windup
    res_both = run_tier2(
        T=600.0,
        current_speed=0.5,
        current_beta_deg=90.0,
        wind_sigma_force=8_000.0,
        wind_sigma_moment=40_000.0,
        wind_tau=60.0,
        seed=1,
    )
    fig_b = plot_results(res_both, "steady current + Gauss-Markov wind gusts")

    plt.show()
