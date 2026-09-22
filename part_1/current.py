"""
Current template

Students should provide the ambient current as a generalized NED velocity
vector. The simulator calls, once per step:

    current.step(t, dt, eta, nu) -> nu_c_ned

Inputs (full 6-DOF state — use what your model needs):
    t    : current simulation time [s]        (time-varying currents)
    dt   : time step [s]                      (slowly-varying components)
    eta  : (6,) vessel state [N, E, z, phi, theta, psi] in NED
           (heading is eta[5]; position for spatially varying fields)
    nu   : (6,) vessel BODY velocities [u, v, w, p, q, r]
           (only indices [0, 1, 5] are nonzero in the 3-DOF model)

Output:
    nu_c_ned : (6,) generalized NED current velocity [m/s]
               [V_N, V_E, V_D, 0, 0, 0]
               Only the horizontal components are used by the 3-DOF model.
               Direction convention is 'towards' (the direction the current
               flows to): a current with V_N > 0, V_E = 0 pushes the vessel
               North.
"""
import numpy as np


class Current:
    """Uniform, horizontally-constant ocean current in the NED frame.

    The current is not applied as a force.  The Gunnerus model takes the NED
    current velocity, rotates it into BODY with ``J(psi).T`` and forms the
    relative velocity ``nu_r = nu - nu_c_body``, which then drives the damping
    and Coriolis terms (project text, Sections 3.4 and 3.5).  This class
    therefore only has to produce the velocity vector itself.

    Constructor contract — the automated checks (``python check.py``,
    ``pytest``, ``notebooks/part_1_demo.ipynb``) construct your model with
    this signature, so keep it working:

        Current(speed, beta, semantics=..., beta_end=..., duration=...)

    Parameters
    ----------
    speed : current speed [m/s].
    beta : direction [rad] in NED (0 = North, pi/2 = East).
    semantics : ``"towards"`` (default) — ``beta`` is the direction the
        current flows to — or ``"from"`` — the direction it comes from.
    beta_end, duration : if given, the direction varies linearly from
        ``beta`` to ``beta_end`` over ``duration`` seconds (Simulation 2),
        then stays at ``beta_end``.  Constant direction if ``beta_end`` is
        ``None``.
    """

    def __init__(self, speed: float = 0.0, beta: float = 0.0, *,
                 semantics: str = "towards",
                 beta_end: float | None = None, duration: float = 0.0):
        if semantics not in ("towards", "from"):
            raise ValueError(
                f"semantics must be 'towards' or 'from', got {semantics!r}")
        self.speed = float(speed)
        self.beta = float(beta)
        self.semantics = semantics
        self.beta_end = None if beta_end is None else float(beta_end)
        self.duration = float(duration)

        # A 'from' direction is the reverse of the direction the water flows
        # to, so it is converted to 'towards' by adding pi: "from east"
        # (beta = pi/2) becomes a westward flow (V_E < 0).
        self._offset = np.pi if semantics == "from" else 0.0

    def _beta_at(self, t: float) -> float:
        """Input direction at time ``t``, in the convention of ``semantics``."""
        if self.beta_end is None:
            return self.beta
        if self.duration <= 0.0:
            return self.beta_end
        # Linear ramp over `duration`, then held at beta_end (Simulation 2).
        s = min(max(t / self.duration, 0.0), 1.0)
        return self.beta + s * (self.beta_end - self.beta)

    def step(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
    ) -> np.ndarray:
        beta_towards = self._beta_at(t) + self._offset

        nu_c_ned = np.zeros(6)
        nu_c_ned[0] = self.speed * np.cos(beta_towards)   # V_N
        nu_c_ned[1] = self.speed * np.sin(beta_towards)   # V_E
        return nu_c_ned
