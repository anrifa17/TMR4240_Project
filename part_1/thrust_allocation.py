"""
Thrust Allocation - unweighted pseudo-inverse.

Solves B_e z = tau_c for the decision vector
z = [u_T, Fx_1, Fy_1, Fx_2, Fy_2]^T (eq. 40) using the Moore-Penrose
pseudo-inverse and recovers the per-thruster commands

Design choices:
  * Saturation: uniform scaling of tau_c preserves the wrench direction.
  * Angle ambiguity: for each azimuth we pick between (alpha, +u) and
    (alpha + pi, -u) the representation nearest alpha_now.
  * Small-thrust hold: below u_min the azimuth angle is held at alpha_now
    so the commanded angle does not chatter when the demand is ~0.

Signed thrust is returned; the tunnel is always driven at alpha = pi/2.
"""
from typing import List, Optional, Tuple
import numpy as np

from models.thruster_dynamics import ThrusterConfig
from simulation.utils import wrap_angle_pi as wrap


class ThrustAllocator:
    """Pseudo-inverse allocator for the 1-tunnel + 2-azimuth Gunnerus layout."""

    def __init__(
        self,
        thrusters: List[ThrusterConfig],
        u_min: float = 100.0,
    ):
        self.thrusters = thrusters
        self.n = len(thrusters)
        self.u_min = float(u_min)  # angle-hold threshold [N]

        self.tunnel_idx: List[int] = []
        self.az_idx: List[int] = []
        for i, th in enumerate(thrusters):
            if th.kind.lower() == "tunnel":
                self.tunnel_idx.append(i)
            else:
                self.az_idx.append(i)

        if len(self.tunnel_idx) != 1 or len(self.az_idx) != 2:
            raise ValueError(
                "ThrustAllocator expects 1 tunnel and 2 azimuth thrusters "
                f"(got {len(self.tunnel_idx)} tunnel, {len(self.az_idx)} azimuth)."
            )

        i_T = self.tunnel_idx[0]
        j1, j2 = self.az_idx
        x_T = thrusters[i_T].x
        x1, y1 = thrusters[j1].x, thrusters[j1].y
        x2, y2 = thrusters[j2].x, thrusters[j2].y

        # Extended configuration matrix (eq. 41): B_e z = tau_c.
        # Tunnel at alpha_T = pi/2 contributes (0, 1, x_T) to (Fx, Fy, Mz);
        # each azimuth j contributes (1, 0, -y_j) via Fx_j and (0, 1, x_j) via Fy_j.
        self.B_e = np.array([
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [x_T, -y1,  x1,  -y2,  x2],
        ])
        # Precompute the pseudo-inverse once (B_e is constant): z = B_e^+ tau_c.
        self.B_e_pinv = np.linalg.pinv(self.B_e)

        self.alpha_T = float(thrusters[i_T].alpha0)  # pi/2
        self.u_max_T = float(thrusters[i_T].u_max)
        self.u_max_az = np.array(
            [thrusters[j1].u_max, thrusters[j2].u_max], dtype=float
        )

    def allocate(
        self,
        t: float,
        dt: float,
        tau_d: np.ndarray,
        u_now: Optional[np.ndarray] = None,
        alpha_now: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # 6-DOF -> 3-DOF wrench: surge, sway, yaw are indices 0, 1, 5.
        tau_c = np.asarray(tau_d, dtype=float).reshape(6)[[0, 1, 5]]

        # eq. 41: unweighted pseudo-inverse (minimises ||z||_2 s.t. B_e z = tau_c)
        z = self.B_e_pinv @ tau_c

        # Uniform scaling to respect u_max (Part 1: thruster dynamics disabled,
        # so saturation is the allocator's responsibility, cf. project Sec. 3.6).
        z = self._scale_to_limits(z)

        u_cmd = np.zeros(self.n)
        alpha_cmd = np.zeros(self.n)

        # Tunnel: signed thrust, fixed angle
        i_T = self.tunnel_idx[0]
        u_cmd[i_T] = z[0]
        alpha_cmd[i_T] = self.alpha_T

        # Current azimuth angles for the closest-representation choice
        if alpha_now is None:
            a_now = np.array(
                [self.thrusters[j].alpha0 for j in self.az_idx], dtype=float
            )
        else:
            a_now = np.asarray(alpha_now, dtype=float)[self.az_idx]

        # eq. 42: recover (u_j, alpha_j) from (Fx_j, Fy_j) for each azimuth
        for k, j in enumerate(self.az_idx):
            Fx = z[1 + 2 * k]
            Fy = z[2 + 2 * k]
            mag = float(np.hypot(Fx, Fy))

            if mag < self.u_min:
                # Demand ~0: hold the current angle, command zero thrust
                u_cmd[j] = 0.0
                alpha_cmd[j] = float(a_now[k])
                continue

            alpha_pos = float(np.arctan2(Fy, Fx))
            alpha_neg = wrap(alpha_pos + np.pi)
            d_pos = abs(wrap(alpha_pos - float(a_now[k])))
            d_neg = abs(wrap(alpha_neg - float(a_now[k])))
            if d_neg < d_pos:
                alpha_cmd[j] = alpha_neg
                u_cmd[j] = -mag
            else:
                alpha_cmd[j] = alpha_pos
                u_cmd[j] = mag

        return u_cmd, alpha_cmd

    def _scale_to_limits(self, z: np.ndarray) -> np.ndarray:
        """Scale z uniformly so every actuator stays within its static limit."""
        s = 1.0
        if abs(z[0]) > 1e-12:
            s = min(s, self.u_max_T / abs(z[0]))
        for k in range(2):
            mag = float(np.hypot(z[1 + 2 * k], z[2 + 2 * k]))
            if mag > 1e-12:
                s = min(s, self.u_max_az[k] / mag)
        return z * min(1.0, s)
