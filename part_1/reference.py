"""
Reference model

Second-order low-pass filter per controlled coordinate (N, E, psi):

    eta_d'' + 2*zeta*wn*eta_d' + wn^2*eta_d = wn^2*eta_cmd

The simulator calls, once per step:

    ref.step(t, dt, eta_cmd) -> (eta_ref, nu_ref, acc_ref)

All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
yaw]. The 3-DOF model uses indices [0, 1, 5]; the rest are left zero.
"""
from typing import Tuple
import numpy as np

from part_1.config import RefAxisConfig
from simulation.utils import wrap_angle_pi


class ReferenceModel:
    def __init__(
        self,
        dt: float,
        cfg_xy: RefAxisConfig | None = None,
        cfg_psi: RefAxisConfig | None = None,
    ):
        self.dt = float(dt)
        self.cfg_xy = cfg_xy if cfg_xy is not None else RefAxisConfig()
        self.cfg_psi = cfg_psi if cfg_psi is not None else RefAxisConfig()
        self.eta_ref = np.zeros(6)
        self.nu_ref = np.zeros(6)
        self.acc_ref = np.zeros(6)

    def reset(self, eta0: np.ndarray) -> None:
        """Initialize the reference at the vessel's current (6,) state."""
        self.eta_ref = np.asarray(eta0, dtype=float).reshape(6).copy()
        self.eta_ref[5] = wrap_angle_pi(self.eta_ref[5])
        self.nu_ref = np.zeros(6)
        self.acc_ref = np.zeros(6)

    @staticmethod
    def _axis_step(
        err: float, vel: float, dt: float, cfg: RefAxisConfig
    ) -> Tuple[float, float]:
        """One Euler step of the filter for a single axis: (new_vel, acc)."""
        acc = cfg.wn**2 * err - 2.0 * cfg.zeta * cfg.wn * vel
        vel_new = vel + acc * dt
        if cfg.rate_limit is not None:
            vel_sat = float(np.clip(vel_new, -cfg.rate_limit, cfg.rate_limit))
            if vel_sat != vel_new:
                acc = (vel_sat - vel) / dt
                vel_new = vel_sat
        return vel_new, acc

    def step(
        self, t: float, dt: float, eta_cmd: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cmd = np.asarray(eta_cmd, dtype=float).reshape(6)

        for i in (0, 1):
            err = cmd[i] - self.eta_ref[i]
            vel, acc = self._axis_step(err, self.nu_ref[i], dt, self.cfg_xy)
            self.eta_ref[i] += vel * dt
            self.nu_ref[i] = vel
            self.acc_ref[i] = acc

        # heading: wrapped error so the short way across +-pi is taken
        err = wrap_angle_pi(cmd[5] - self.eta_ref[5])
        vel, acc = self._axis_step(err, self.nu_ref[5], dt, self.cfg_psi)
        self.eta_ref[5] = wrap_angle_pi(self.eta_ref[5] + vel * dt)
        self.nu_ref[5] = vel
        self.acc_ref[5] = acc

        return self.eta_ref, self.nu_ref, self.acc_ref
