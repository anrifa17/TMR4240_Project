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

from simulation.utils import Rz, wrap_angle_pi


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

    def step(self, t: float, dt: float, eta_measured: np.ndarray, tau_est: np.ndarray) -> ObserverEstimate:
        raise NotImplementedError

    @staticmethod
    def _placeholder(eta_measured: np.ndarray) -> ObserverEstimate:
        return ObserverEstimate(
            eta=np.asarray(eta_measured, dtype=float).reshape(6).copy(),
            nu=np.zeros(6),
            bias=np.zeros(6),
        )


class KalmanFilterObserver(Observer):
    """Template for a discrete KF/EKF observer.

    Students define the process/measurement models and tune Q, R and P0.
    The input ``tau_est`` is the desired controller wrench before thruster
    dynamics.
    """

    name = "kalman"

    def __init__(self, *args, **kwargs):
        # Vessel model
        self.M = np.array([[6.007e5, 0.0, 0.0], [0.0, 7.067e5, -4.733e5], [0.0, -5.712e5, 5.456e7]])
        self.D = np.array([[1117.6, 0.0, 0.0], [0.0, 2.229e4, 0.0], [0.0, 0.0, 1.95e6]])
        self.M_inv = np.linalg.inv(self.M3)

        # Wave model
        omega = 0.628 * np.ones(3)  # dominant wave frequency [rad/s]
        zeta = 0.1 * np.ones(3)  # relative damping
        sigma_wf = np.array([0.375, 0.375, 0.087])  # std of WF motion [m, m, rad]
        Kw = 2.0 * np.sqrt(zeta * omega) * sigma_wf  # K_w,i = 2 sqrt(zeta_i omega_i) sigma_wf,i

        # WF model
        Z3, I3 = np.zeros((3, 3)), np.eye(3)
        # np.block glues the four 3x3 blocks into one 6x6 matrix
        self.Aw = np.block([[Z3, I3], [-np.diag(omega**2), -2.0 * np.diag(zeta * omega)]])
        Cw = np.hstack([Z3, I3])  # 3x6, picks the "output" half of xi_w

        # Bias model
        Tb = 1000.0
        self.Tb_inv = np.eye(3) / Tb

        # Measurement matrix
        self.H = np.hstack([Cw, I3, np.zeros((3, 6))])

        # Noise Input matrix E
        self.E = np.zeros((15, 6))
        self.E[3:6, 0:3] = np.diag(Kw)  # wave noise enters the xi_w "velocity" half
        self.E[9:12, 3:6] = np.eye(3)  # bias noise (E_b = I)

        # Covariances
        sigma_b = np.array([40e3, 40e3, 80e3])  # bias std [N, N, Nm]
        # Continuous-time intensities Qc; converted to per-sample Q = Qc/dt inside step()
        self.Qc = np.diag(np.concatenate([np.ones(3), 2.0 * sigma_b**2 / Tb]))
        self.R = np.diag([0.2**2, 0.2**2, np.deg2rad(0.20) ** 2])

        sigma_nu = np.array([0.3, 0.3, 0.3])  # initial velocity std
        self.P0 = np.diag(
            np.concatenate(
                [
                    sigma_wf**2 / omega**2,  # integrated wave states
                    sigma_wf**2,  # wave output states
                    sigma_wf**2,  # eta
                    sigma_b**2,  # bias
                    sigma_nu**2,  # nu
                ]
            )
        )

        self.reset()

    def step(self, t, dt, eta_measured, tau_est):
        idx = [0, 1, 5]

        return self._placeholder(eta_measured)


def select_observer(kind: str, **kwargs) -> Observer:
    """Construct one observer without changing simulator code."""
    choices = {
        # "nonlinear_passive": NonlinearPassiveObserver,
        "kalman": KalmanFilterObserver,
    }
    try:
        return choices[kind](**kwargs)
    except KeyError as exc:
        raise ValueError(f"observer_type must be one of {tuple(choices)}") from exc
