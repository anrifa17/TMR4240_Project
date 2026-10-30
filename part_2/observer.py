"""Student observer templates and their common simulator interface.

The simulator calls, once per step (pose measurement only, no velocity):

    observer.step(t, dt, eta_measured, tau_est) -> ObserverEstimate(eta, nu, bias)

    eta_measured : (6,) measured NED pose [N, E, z, phi, theta, psi]
    tau_est      : (6,) desired controller wrench (before thruster dynamics)
                   of the PREVIOUS step (zero at the first step)
    eta, nu      : (6,) low-frequency NED pose and BODY velocity estimates
    bias         : (6,) slowly varying bias estimate (NED force) — may be zeros

A plain ``(eta, nu, bias)`` tuple is accepted as well.  Before every run the
engine calls ``reset(eta0)`` with the TRUE initial pose, so an observer can
start from the vessel's actual position.  Select an implementation with
``Part2SimConfig(observer_type=...)`` (Simulations 4-7 use the
``SELECTED_OBSERVER`` constant of ``run_case_part_2.py``); extra constructor
arguments go through ``Part2SimConfig(observer_kwargs=...)``.  Both classes
intentionally start as pass-through placeholders; students must implement and
tune them, keeping the tuned parameters as constructor defaults because the
checks call ``select_observer(kind)`` without arguments.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ObserverEstimate:
    eta: np.ndarray
    nu: np.ndarray
    bias: np.ndarray


class Observer:
    """Interface shared by all Part 2 observers."""

    name = "observer"

    def reset(self, eta0: np.ndarray | None = None):
        pass

    def step(
        self, t: float, dt: float, eta_measured: np.ndarray, tau_est: np.ndarray
    ) -> ObserverEstimate:
        raise NotImplementedError

    @staticmethod
    def _placeholder(eta_measured: np.ndarray) -> ObserverEstimate:
        return ObserverEstimate(
            eta=np.asarray(eta_measured, dtype=float).reshape(6).copy(),
            nu=np.zeros(6),
            bias=np.zeros(6),
        )


def right_pi(angle: float) -> float:
    """Makes degree in right intervall (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _Rz(psi: float) -> np.ndarray:
    """3x3 rotaion matrix"""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


class NonlinearPassiveObserver(Observer):
    """Nonlinear Passive Observer for R/V Gunnerus (3-DOF DP).

    Estimates low-frequency (LF) position/heading eta_hat, velocity nu_hat,
    disturbance bias b_hat, and wave-frequency (WF) motion xi_hat.
    The only measurement input is GPS + compass: y = [N, E, psi].
    """
    name = "nonlinear_passive"

    def __init__(self, *args, **kwargs):
        #  M og D from Appendix B 
        self.M = np.array([
            [6.007e5,  0.0,      0.0     ],
            [0.0,      7.067e5, -4.733e5 ],
            [0.0,     -5.712e5,  5.456e7 ],
        ])
        self.D = np.array([
            [1.118e3,  0.0,     0.0     ],
            [0.0,      2.229e4, 0.0     ],
            [0.0,      0.0,     1.950e6 ],
        ])
        self.M_inv = np.linalg.inv(self.M)

        #Wave frequency model
        #ω₀ = 2π/Tp
        omega0 = 2.0 * np.pi / 8.0   # rad/s 
        lam = 0.1                  #lamda

        # 2×2 blokk for ett DOF:  [0, 1; -ω₀², -2λω₀]
        def blk(w0):
            return np.array([[0.0,    1.0      ],
                             [-w0**2, -2*lam*w0]])

        # Block diagonal matrix 6×6: surge (N), sway (E), yaw (ψ)
    
        Z = np.zeros((2, 2))
        self.Aw = np.block([
            [blk(omega0), Z,           Z          ],
            [Z,           blk(omega0), Z          ],
            [Z,           Z,           blk(omega0)],
        ])  # (6, 6)

        
        #xi_hat = [x1_N, x2_N, x1_E, x2_E, x1_ψ, x2_ψ]
        self.Cw = np.zeros((3, 6))
        self.Cw[0, 0] = 1.0   # N-component
        self.Cw[1, 2] = 1.0   # E-Component
        self.Cw[2, 4] = 1.0   # ψ-component

        Tb = 1000.0              
        self.Tb_inv = (1.0 / Tb) * np.eye(3)

       
        # K3 = M · K3_bar 
        #trying this values
        K3_bar = np.diag([1e-4, 1e-4, 5e-6])   # positive diagonal (passivity condition)
        self.K3 = self.M @ K3_bar               # passivity condition satisfied: K3 = M · K3_bar

        self.K2 = np.diag([0.5,  0.5,  0.1 ])  # position/heading correction gain

        self.K4 = np.diag([0.01, 0.01, 0.001]) # bias correction gain

        # K1: wavestate correction
        self.K1 = np.zeros((6, 3))
        self.K1[1, 0] = 1.0    # xi2_surge
        self.K1[3, 1] = 1.0    # xi2_sway
        self.K1[5, 2] = 0.5    # xi2_yaw

        #states
        self.xi_hat  = np.zeros(6)   
        self.eta_hat = np.zeros(6)   
        self.nu_hat  = np.zeros(6)   
        self.b_hat   = np.zeros(3)   

    def reset(self, eta0=None):
        """Called by the simulator with the true initial pose before each run."""
        self.xi_hat  = np.zeros(6)
        self.eta_hat = np.zeros(6)
        self.nu_hat  = np.zeros(6)
        self.b_hat   = np.zeros(3)
        if eta0 is not None:
            self.eta_hat = np.asarray(eta0, dtype=float).reshape(6).copy()

    def step(self, t, dt, eta_measured, tau_est):
        IDX = [0, 1, 5]
        eta3 = self.eta_hat[IDX].copy()          # [N̂, Ê, ψ̂]
        nu3  = self.nu_hat[IDX].copy()            # [û, v̂, r̂]
        y    = np.asarray(eta_measured, dtype=float).reshape(6)[IDX]
        tau3 = np.asarray(tau_est,      dtype=float).reshape(6)[IDX]

       
        y_hat   = eta3 + self.Cw @ self.xi_hat
        y_tilde = y - y_hat
        y_tilde[2] = right_pi(y_tilde[2])

        #Rotaion matrix
        psi = eta3[2]
        J   = _Rz(psi)     # skipsramme → NED
        Jt  = J.T          # NED → skipsramme

        xi_dot  = self.Aw @ self.xi_hat + self.K1 @ y_tilde

        eta_dot = J @ nu3  + self.K2 @ y_tilde

        nu_dot  = self.M_inv @ (
            -self.D @ nu3
            + Jt @ self.b_hat
            + tau3
            + self.K3 @ y_tilde
        )

        b_dot   = -self.Tb_inv @ self.b_hat + self.K4 @ y_tilde

        #Forward Euler
        self.xi_hat        += dt * xi_dot
        self.eta_hat[IDX]   = eta3 + dt * eta_dot
        self.nu_hat[IDX]    = nu3  + dt * nu_dot
        self.b_hat         += dt * b_dot

        #bias
        bias6 = np.zeros(6)
        bias6[0] = self.b_hat[0]   # bN
        bias6[1] = self.b_hat[1]   # bE
        bias6[5] = self.b_hat[2]   # bψ

        return ObserverEstimate(
            eta  = self.eta_hat.copy(),
            nu   = self.nu_hat.copy(),
            bias = bias6,
        )


class KalmanFilterObserver(Observer):
    """Template for a discrete KF/EKF observer.

    Students define the process/measurement models and tune Q, R and P0.
    The input ``tau_est`` is the desired controller wrench before thruster
    dynamics.
    """

    name = "kalman"

    def __init__(self, *args, **kwargs):
        pass

    def step(self, t, dt, eta_measured, tau_est):
        # TODO: implement the Kalman filter or extended Kalman filter.
        return self._placeholder(eta_measured)


def select_observer(kind: str, **kwargs) -> Observer:
    """Construct one observer without changing simulator code."""
    choices = {
        "nonlinear_passive": NonlinearPassiveObserver,
        "kalman": KalmanFilterObserver,
    }
    try:
        return choices[kind](**kwargs)
    except KeyError as exc:
        raise ValueError(f"observer_type must be one of {tuple(choices)}") from exc

