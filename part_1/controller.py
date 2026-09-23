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
from scipy.integrate import solve_ivp, trapezoid, cumulative_trapezoid

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
        default_Q = np.diag([0.04, 0.04, 131.31, 4, 4, 100])

        # R penalizes actuator usage
        default_R = np.diag([3.90625000e-11, 7.97193878e-11, 4.93151117e-13])
        
        self.Q = kwargs.get('Q', default_Q)
        self.R = kwargs.get('R', default_R)

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

        # --- DEFINITIONS FROM DOCSTRING ---
        # eta NED
        N, E, psi = eta[0], eta[1], eta[5]

        # nu BODY
        u, v, r = nu[0], nu[1], nu[5]

        # eta_ref NED
        N_d, E_d, psi_d = eta_ref[0], eta_ref[1], eta_ref[5]

        # nu_ref NED
        Ndot_d, Edot_d, psidot_d = nu_ref[0], nu_ref[1], nu_ref[5]

        J = Rz(psi)  # rotation matrix

        # eta
        # --- Nothing to handle here

        # nu
        # nu3_b = np.array([nu[0], nu[1], nu[5]])  # BODY, current velocity,
        nu3_b = np.array([u, v, r])  # BODY, current velocity,
        nu3_ref_b = np.zeros_like(nu3_b)  # BODY, reference velocity

        if not nu_ref is None:  # if nu_ref is defined
            nu3_ref_n = np.array([Ndot_d, Edot_d, psidot_d])  # NED, reference velocity
            nu3_ref_b = J.T @ nu3_ref_n  # BODY, update reference velocity

        # state errors
        e_N = N - N_d  # NED, error in N
        e_E = E - E_d  # NED, error in E
        e_psi = wrap_angle_pi(psi - psi_d)  # NED, heading error psi

        e_eta3_n = np.array([e_N, e_E, e_psi])  # NED, error matrix position
        e_eta3_b = J.T @ e_eta3_n  #  BODY, error matrix position
        e_nu3_b = nu3_b - nu3_ref_b  # BODY, error matrix velocity

        # State vector
        x_c = np.hstack((e_eta3_b.T, e_nu3_b.T))  # BODY, state vector

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
        if not self.is_controllable(self.A_c, self.B_c):
            raise ValueError(
                "The system pair (A, B) is not controllable. "
                "The rank of the controllability matrix is less than the state dimension."
            )

        u = -self.K @ x_c
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

    # INFORMATION RETRIEVAL START
    def LQRConditions(self) -> str:
        """Returns LQR conditions status"""
        s = ""
        s += f"    Q is {'NOT ' if not self.is_positive_semidefinite(self.Q) else ''}positive semi-definite\n"
        s += f"    R is {'NOT ' if not self.is_positive_definite(self.R) else ''}positive definite\n"
        s += f"(A,B) is {'NOT ' if not self.is_controllable(self.A_c, self.B_c) else ''}controllable"
        return s

    def eigenvalues(self):
        return np.linalg.eigvals(self.A_c - self.B_c @ self.K)

    # INFORMATION RETRIEVAL END


def print_force_vector_kn(vec, precision=2):
    """
    GEMINI made this
    Formats and prints a 6-DOF force/moment vector converted to kN and kN·m.
    """
    labels = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    units = ["kN ", "kN ", "kN ", "kN·m", "kN·m", "kN·m"]

    # Convert from N (and N·m) to kN (and kN·m)
    vec_kn = np.asarray(vec, dtype=float) / 1000.0

    print("─── Force Vector Summary ───")
    for label, val, unit in zip(labels, vec_kn, units):
        val_str = f"{val:>{precision+8}.{precision}f}"
        print(f"  {label}: {val_str} {unit}")
    print("───────────────────────────")


def simulate_error_dynamics(controller, x0: np.ndarray, t_end=200, num_points=1000):
    """
    GEMINI MADE THIS
    Simulation of error dynamics to tune controller
    """

    # 1. Closed-loop system matrix
    Acl = controller.A_c - controller.B_c @ controller.K

    # 2. Simulate continuous ODE
    sol = solve_ivp(lambda t, x: Acl @ x, [0, t_end], x0, dense_output=True)

    # 3. Generate dense arrays for smooth plotting and integration
    t = np.linspace(0, t_end, num_points)
    x = sol.sol(t)  # Shape: (6, num_points)

    # 4. Calculate Actuator Usage (u = -Kx)
    # LQR control law applies a negative feedback gain to the state errors
    u = -controller.K @ x  # Shape: (3, num_points) -> (tau_x, tau_y, tau_psi)

    # 5. Calculate Performance Metrics (IAE and ISE)
    # Integrating over time using the trapezoidal rule
    iae = trapezoid(np.abs(x), t, axis=1)  # Integral Absolute Error
    ise = trapezoid(x**2, t, axis=1)  # Integral Square Error
    iae_ts = cumulative_trapezoid(np.abs(x), t, axis=1, initial=0)
    ise_ts = cumulative_trapezoid(x**2, t, axis=1, initial=0)

    return {
        "ivp_sol": sol,
        "time": t,
        "state_error": x,
        "actuator_usage": u,
        "metrics": {
            "IAE": iae,
            "ISE": ise,
            "Cumulative IAE": iae_ts,
            "Cumulative ISE": ise_ts,
        },
    }


def tests():
    controller = DPController()
    print(
        f"Controller initialized successfully. Gain matrix K shape: {controller.K.shape}"
    )

    # Test computation step
    # eta = np.array([-10.0, 5.0, 0.0, 0.0, 0.0, 5.0])
    # nu = np.array([1.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    # eta_ref = np.zeros(6)
    # tau = controller.compute(0.0, 0.01, eta, nu, eta_ref)

    # sol = simulate_error_dynamics(controller)

    # print(controller.LQRConditions())
    # print(f"Eigenvalues:\n{controller.eigenvalues()}")
    # print(f"Time constants:\n{1/abs(np.real(controller.eigenvalues()))}")
    # print_force_vector_kn(tau)

    controller = DPController()

    x0_surge = 0  # m
    x0_sway = -20  # m
    x0_yaw = 0 * np.pi / 180  # rad
    x0_vel_surge = 0  # m/s
    x0_vel_sway = 0  # m/s
    x0_vel_yaw = 0 * np.pi / 180  # rad/s

    x0 = np.array([x0_surge, x0_sway, x0_yaw, x0_vel_surge, x0_vel_sway, x0_vel_yaw])

    # --- Usage ---
    # print(f"IAE for Position X: {results['metrics']['IAE'][0]:.2f}")
    # print(f"ISE for Position Y: {results['metrics']['ISE'][1]:.2f}")
    # 1. Run simulation
    results = simulate_error_dynamics(controller, x0, t_end=40)


def main():
    tests()


if __name__ == "__main__":
    main()
