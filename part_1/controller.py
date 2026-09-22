"""
Controller template

Students should implement a controller that maps the vessel state and the
full reference to a body-frame wrench. The simulator calls, once per step:

    controller.compute(t, dt, eta, nu, eta_ref, nu_ref, acc_ref) -> tau_d

All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
yaw]. The 3-DOF model uses indices [0, 1, 5]; the remaining components are
zero on input and ignored on output.

Inputs (full loop state and full reference):
    t       : current simulation time [s]
    dt      : time step [s]
    eta     : (6,) vessel NED state [N, E, z, phi, theta, psi]
              (use N = eta[0], E = eta[1], psi = eta[5])
    nu      : (6,) vessel BODY velocities [u, v, w, p, q, r]
              (use u = nu[0], v = nu[1], r = nu[5])
    eta_ref : (6,) NED reference state
              (use N_d = eta_ref[0], E_d = eta_ref[1], psi_d = eta_ref[5])
    nu_ref  : (6,) NED-frame reference velocities
              (use Ndot_d = nu_ref[0], Edot_d = nu_ref[1], psidot_d = nu_ref[5])
    acc_ref : (6,) NED-frame reference accelerations, same layout as nu_ref
              (use for model-based / inertia feedforward)

Output:
    tau_d   : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz] (N, Nm)
              (fill in Fx = tau_d[0], Fy = tau_d[1], Mz = tau_d[5];
               leave the other components zero)

Optional hooks the simulator will use IF you define them (safe to omit):
    reset()                                  — called before each run
    apply_external_aw(tau_applied, psi, dt)  — anti-windup with the (6,)
                                               wrench actually applied after
                                               allocation and the actuator
                                               model (ideal in Part 1)
    last_pid_body  : {"P","I","D"} -> (6,) BODY components   (logged)
    int_ned (2,), int_psi (float)            — integrator states (logged)

Constructor contract — the automated checks (``python check.py``, ``pytest``,
``notebooks/part_1_demo.ipynb``) construct your controller as
``DPController()`` with NO arguments, so your final tuned gains must be the
constructor defaults. Tuning only inside ``run_case_part1.py`` will pass your
own runs but fail the checks.
"""

import numpy as np
from scipy.linalg import solve_continuous_are
from simulation.utils import Rz, wrap_angle_pi


class DPController:
    """
    Template for student DP controller.

    Students may implement any type of controller (PID, LQR, backstepping,
    ...). Only compute() is required; everything else is optional.
    """

    def __init__(self, *args, **kwargs):
        # Physical 3-DOF matrices
        self.M3 = np.array(
            [[6.007e5, 0.0, 0.0], [0.0, 7.067e5, -4.733e5], [0.0, -5.712e5, 5.456e7]]
        )
        self.D3 = np.array(
            [[1117.6, 0.0, 0.0], [0.0, 2.229e4, 0.0], [0.0, 0.0, 1.95e6]]
        )
        self.M3_inv = np.linalg.inv(self.M3)

        # State-space matrices
        O3 = np.zeros((3, 3))
        I3 = np.eye(3)
        self.A_c = np.block([[O3, I3], [O3, -self.M3_inv @ self.D3]])
        self.B_c = np.block([[O3], [self.M3_inv]])

        # Tuning weights
        ## Q matrix penalizes position (0:3) and velocity (3:6) errors
        self.Q = np.diag([0.25, 0.25, 1.5791367, 4, 4, 100])

        # R penalizes actuator usage
        self.R = np.diag([3.90625000e-11, 7.97193878e-11, 4.93151117e-13])

        # Gain matrix K
        P = solve_continuous_are(a=self.A_c, b=self.B_c, q=self.Q, r=self.R)
        self.K = np.linalg.inv(self.R) @ self.B_c.T @ P

    def reset(self) -> None:
        """Optional: reset internal states (integrators, filters) before a run."""
        pass

    def compute(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
        eta_ref: np.ndarray,
        nu_ref: np.ndarray | None = None,
        acc_ref: np.ndarray | None = None,
    ) -> np.ndarray:
        # Return the (6,) desired BODY wrench — fill in tau_d[0] = Fx,
        # tau_d[1] = Fy, tau_d[5] = Mz and leave the rest zero.

        J = Rz(eta[5])  # rotation matrix

        # Get inputs
        eta3_n = np.array([eta[0], eta[1], eta[5]])  # NED, current state
        nu3_b = np.array([nu[0], nu[1], nu[5]])  # BODY, current velocity,

        eta3_ref_n = np.array(
            [eta_ref[0], eta_ref[1], eta_ref[5]]
        )  # NED, reference position

        if not nu_ref is None:
            nu3_ref_n = np.array(
                [nu_ref[0], nu_ref[1], nu_ref[5]]
            )  # NED, reference velocity

        nu3_ref_b = J.T @ nu3_ref_n  # BODY, reference velocity

        e_N = eta[0] - eta_ref[0]  # NED, error in N
        e_E = eta[1] - eta_ref[1]  # NED, error in E
        e_psi = wrap_angle_pi(eta[5] - eta_ref[5])  # NED, heading error psi

        e_eta3_n = np.array([e_N, e_E, e_psi])  # NED, error matrix position
        e_eta3_b = J.T @ e_eta3_n  #  BODY, error matrix position

        e_nu3_b = nu3_b - nu3_ref_b  # BODY, error matrix velocity

        # State vector
        x_c = np.hstack((e_eta3_b.T, e_nu3_b))  # BODY, state vector

        # State matrices
        O3 = np.zeros((3, 3))
        I3 = np.eye(3)

        A_c = np.block([[O3, I3], [O3, -self.M3_inv @ self.D3]])
        B_c = np.block([[O3], [self.M3_inv]])

        # Q must be Positive Semi-Definite (Q >= 0)
        if not self.is_positive_semidefinite(self.Q):
            raise ValueError(
                "Matrix Q must be positive semi-definite (Q >= 0). "
                "Ensure Q is symmetric and all eigenvalues are non-negative."
            )

        # R must be Positive Definite (R > 0)
        if not self.is_positive_definite(self.R):
            raise ValueError(
                "Matrix R must be strictly positive definite (R > 0). "
                "Ensure R is symmetric and all eigenvalues are strictly positive."
            )

        # (A, B) must be Controllable
        if not self.is_controllable(A_c, B_c):
            raise ValueError(
                "The system pair (A, B) is not controllable. "
                "The rank of the controllability matrix is less than the state dimension."
            )

        # Solve LQR
        P = solve_continuous_are(a=A_c, b=B_c, q=self.Q, r=self.R)
        K = np.linalg.inv(self.R) @ B_c.T @ P
        u = -K @ x_c
        return [u[0], u[1], 0, 0, 0, u[2]]

    # HELPERS START
    def is_positive_definite(self, A: np.ndarray, tol: float = 1e-12) -> bool:
        """Checks if A is square, symmetric, and positive definite (all eigenvalues > 0)."""
        # A must be square
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            print("A is not square")
            return False

        # A must be symmetric
        if not np.allclose(A, A.T, atol=tol):
            print("A is not symmetric")
            return False

        # All eigenvalues must be strictly > 0 (with tolerance for floating-point noise)
        eigvals = np.linalg.eigvalsh(A)
        if not np.all(eigvals > 0):
            print(
                f"Not all eigenvalues are > 0 (Min eigenvalue: {np.min(eigvals):.3e})"
            )
            return False

        return True

    def is_positive_semidefinite(self, A: np.ndarray, tol: float = 1e-12) -> bool:
        """Checks if A is square, symmetric, and positive semi-definite (all eigenvalues >= 0)."""
        # A must be square
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            print("A is not square")
            return False

        # A must be symmetric
        if not np.allclose(A, A.T, atol=tol):
            print("A is not symmetric")
            return False

        # All eigenvalues must be >= 0 (allowing for minor negative numerical noise down to -tol)
        eigvals = np.linalg.eigvalsh(A)
        if not np.all(eigvals >= -tol):
            print(
                f"Not all eigenvalues are >= 0 (Min eigenvalue: {np.min(eigvals):.3e})"
            )
            return False

        return True

    def is_controllable(self, A: np.ndarray, B: np.ndarray) -> bool:
        # A must be square
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            print("A is not square")
            return False

        # A is nxn, B is nxm
        n = A.shape[0]
        m = B.shape[1]

        if B.shape[0] != n:
            print(f"B does not have {n} rows")
            return False

        blocks = [np.linalg.matrix_power(A, i) @ B for i in range(n)]
        M_c = np.hstack(blocks)  # Controllability matrix
        rank = np.linalg.matrix_rank(M_c)

        return rank == n
    # HELPERS END

def tests():
    controller = DPController()

    print(
        f"Q is positive semi-definite: {controller.is_positive_semidefinite(controller.Q)}"
    )
    print(
        f"R is positive semi-definite: {controller.is_positive_definite(controller.R)}"
    )
    print(
        f"(A,B) is controllable: {controller.is_controllable(controller.A_c,controller.B_c)}"
    )


def main():
    tests()


if __name__ == "__main__":
    main()