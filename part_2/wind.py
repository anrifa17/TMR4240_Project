"""
Wind template (Project Part 2)

Part 2 extends the Part 1 wind model.  In addition to the mean and slowly
varying wind speed, the wind field must now contain

* a fluctuating **gust** component of the wind speed, generated from a wind
  spectrum of your choice (Harris, Davenport, NORSOK, ...), and
* a **slowly varying wind direction**, without a gust component, bounded to
  at most +/- 5 degrees around the mean direction.

You built such a wind generator in Assignment 2 -- reuse it.  The coefficient
lookup (relative wind -> loads) is unchanged from Part 1; only the ambient
wind field (speed and direction as functions of time) becomes richer.

Students should compute generalized BODY-frame wind loads:
    tau_w6 = [Fx, Fy, Fz, Mx, My, Mz]

The simulator uses the 3-DOF subset [Fx, Fy, Mz] = tau_w6 indices [0, 1, 5]
and calls, once per step:

    wind.step(t, dt, eta, nu) -> (tau_w6, info)

Inputs (full 6-DOF state — use what your model needs):
    t    : current simulation time [s]        (gust spectra, time variation)
    dt   : time step [s]                      (slowly-varying components)
    eta  : (6,) vessel state [N, E, z, phi, theta, psi] in NED
           (heading is eta[5])
    nu   : (6,) vessel BODY velocities [u, v, w, p, q, r]
           (RELATIVE wind: compute the loads from V_rw = V_wind - V_vessel,
            using the horizontal components nu[0], nu[1])

Outputs:
    tau_w6 : (6,) BODY loads
    info   : optional dict for logging, e.g.
             {"U": ambient speed (total), "beta_ned": direction (towards, rad),
              "alpha_body": relative wind angle in BODY (rad),
              "U_mean": mean + slowly varying speed, "U_gust": gust speed}
             The Part 2 checks read "U_mean", "U_gust" and "beta_ned" to
             verify your wind field (gust present, direction within limit),
             so fill them in.
             Return {} (or None) if you do not need it.
             NOTE: "beta_ned" is always the direction the wind blows
             TOWARDS, even when the constructor semantics is "from" —
             convert before logging, do not log the raw constructor value.

Wind coefficient data
---------------------
The vessel wind coefficients C(alpha) = [Cx, Cy, Cz, Cphi, Ctheta, Cpsi] are
provided in `data/wind_coeff.csv` (repository root), tabulated against the relative
wind angle alpha in degrees (0..360). Load them with:

    alpha_deg, C6 = load_wind_coefficients()

The wind loads are then computed as F_wind = U_rw^2 * C(alpha_rw), where U_rw
and alpha_rw are the relative wind speed and angle in the BODY frame.
"""
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

_WIND_COEFF_FILE = Path(__file__).resolve().parent.parent / "data" / "wind_coeff.csv"


def load_wind_coefficients() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the vessel wind coefficient table.

    Returns
    -------
    alpha_deg : (M,) ndarray
        Relative wind angle grid [deg], from 0 to 360.
    C6 : (M, 6) ndarray
        Coefficients [Cx, Cy, Cz, Cphi, Ctheta, Cpsi] at each angle.
    """
    table = np.loadtxt(_WIND_COEFF_FILE, delimiter=",", skiprows=1)
    return table[:, 0], table[:, 1:]


def norsok_spectrum(f: np.ndarray, U10: float, z: float = 10.0) -> np.ndarray:
    """NORSOK wind spectrum S(f) [m^2/s^2/Hz], f in Hz (Sorensen 2018, eq. 6.31)."""
    n = 0.468
    x = 172.0 * f * (z / 10.0) ** (2.0 / 3.0) * (U10 / 10.0) ** (-0.75)
    return (320.0 * (U10 / 10.0) ** 2 * (z / 10.0) ** 0.45
            / (1.0 + x ** n) ** (5.0 / (3.0 * n)))


class Wind:
    """Template for the Part 2 wind model.

    Constructor contract -- the automated checks (``python check.py``,
    ``pytest``) and the Part 2 runner construct your model with this
    signature, so keep it working.  It is a superset of the Part 1 contract:
    your Part 1 model can be pasted in and extended.

        Wind(mean_speed, beta, semantics=..., sigma_slow=..., tau_slow=...,
             gust=..., gust_params=..., sigma_dir=..., tau_dir=...,
             dir_limit=..., seed=...)

    Parameters
    ----------
    mean_speed : mean wind speed [m/s].
    beta : mean direction [rad] in NED (0 = North, pi/2 = East).
    semantics : ``"from"`` (default, the usual meteorological convention --
        "wind from south" blows northward) or ``"towards"``.
    sigma_slow : standard deviation of the slowly varying wind speed
        component [m/s] (Part 1; 0 disables it).
    tau_slow : time constant of the slow speed variation [s].
    gust : switch the fluctuating (gust) speed component on.  The gust must
        be zero-mean and follow a wind spectrum; the spectrum and its
        parameters are your choice (see Assignment 2).  ``False`` gives the
        Part 1 wind field, which the checks use to isolate the gust.
    gust_params : dict with the parameters of *your* gust model (spectrum
        type, elevation, turbulence intensity, number of harmonics, ...).
        ``None`` means your own defaults.
    sigma_dir : standard deviation of the slowly varying wind direction
        [rad] (0 disables it).  The direction has NO gust component.
    tau_dir : time constant of the slow direction variation [s].
    dir_limit : hard bound on the direction excursion |beta(t) - beta| [rad];
        the project requires at most 5 degrees.
    seed : random seed for all stochastic components, so runs are
        reproducible.
    """

    def __init__(self, mean_speed: float = 0.0, beta: float = 0.0, *,
                 semantics: str = "from", sigma_slow: float = 0.0,
                 tau_slow: float = 120.0, gust: bool = False,
                 gust_params: Dict | None = None, sigma_dir: float = 0.0,
                 tau_dir: float = 300.0, dir_limit: float = np.deg2rad(5.0),
                 seed: int | None = None):
        if semantics not in ("towards", "from"):
            raise ValueError(
                f"semantics must be 'towards' or 'from', got {semantics!r}")
        self.mean_speed = float(mean_speed)
        self.beta = float(beta)
        self.semantics = semantics
        self.sigma_slow = float(sigma_slow)
        self.tau_slow = float(tau_slow)
        self.gust = bool(gust)
        self.gust_params = dict(gust_params or {})
        self.sigma_dir = float(sigma_dir)
        self.tau_dir = float(tau_dir)
        self.dir_limit = float(dir_limit)
        self.seed = seed

        # Mean direction converted to 'towards' (as in Part 1).
        offset = np.pi if semantics == "from" else 0.0
        self._beta_towards = float(np.arctan2(np.sin(self.beta + offset),
                                              np.cos(self.beta + offset)))
        self._alpha_deg, self._C6 = load_wind_coefficients()

        # One random stream per stochastic component, so switching the gust
        # on or off does not change the slow-speed or direction realisation.
        self._rng = np.random.default_rng(seed)
        self._rng_dir = np.random.default_rng(None if seed is None else seed + 1)
        rng_gust = np.random.default_rng(None if seed is None else seed + 2)

        self._u_slow = 0.0      # slowly varying speed [m/s]
        self._dbeta = 0.0       # slowly varying direction offset [rad]

        if self.gust:
            self._init_gust(rng_gust)

    def _init_gust(self, rng: np.random.Generator) -> None:
        """Discretise the NORSOK wind spectrum into N harmonic components."""
        p = {"z": 10.0, "N": 100, "f_min": 0.005, "f_max": 0.5}
        p.update(self.gust_params)
        self.gust_params = p
        z, N = float(p["z"]), int(p["N"])

        df = (p["f_max"] - p["f_min"]) / N
        f = p["f_min"] + df * (np.arange(N) + 0.5)          # bin centres [Hz]
        S = (norsok_spectrum(f, self.mean_speed, z) if self.mean_speed > 0.0
             else np.zeros(N))                              # no mean wind, no gust      

        self._gust_amp = np.sqrt(2.0 * S * df)              # a_i
        self._gust_freq = f + rng.uniform(-0.5 * df, 0.5 * df, N)
        self._gust_phase = rng.uniform(0.0, 2.0 * np.pi, N)


    def step(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        # --- Slowly varying speed: 1st-order Gauss-Markov (Part 1) ---
        if self.sigma_slow > 0.0 and self.tau_slow > 0.0:
            a = np.exp(-dt / self.tau_slow)
            self._u_slow = (a * self._u_slow + self.sigma_slow
                            * np.sqrt(1.0 - a ** 2) * self._rng.standard_normal())
        U_mean = max(self.mean_speed + self._u_slow, 0.0)

        # --- Gust: sum of harmonics from the wind spectrum ---
        U_gust = 0.0
        if self.gust:
            U_gust = float(np.sum(self._gust_amp * np.cos(
                2.0 * np.pi * self._gust_freq * t + self._gust_phase)))
        U = U_mean + U_gust

        # --- Slowly varying direction: Gauss-Markov, bounded to dir_limit ---
        if self.sigma_dir > 0.0 and self.tau_dir > 0.0:
            a = np.exp(-dt / self.tau_dir)
            self._dbeta = (a * self._dbeta + self.sigma_dir
                           * np.sqrt(1.0 - a ** 2) * self._rng_dir.standard_normal())
            self._dbeta = float(np.clip(self._dbeta, -self.dir_limit, self.dir_limit))
        beta_t = float(np.arctan2(np.sin(self._beta_towards + self._dbeta),
                                  np.cos(self._beta_towards + self._dbeta)))

        # --- Relative wind and loads (Part 1) ---
        U_load = max(U, 0.0)
        psi = float(eta[5])
        V_n = U_load * np.cos(beta_t)
        V_e = U_load * np.sin(beta_t)
        c, s = np.cos(psi), np.sin(psi)
        V_rw_x = c * V_n + s * V_e - float(nu[0])
        V_rw_y = -s * V_n + c * V_e - float(nu[1])

        U_rw = float(np.hypot(V_rw_x, V_rw_y))
        alpha_rw = float(np.arctan2(V_rw_y, V_rw_x))

        alpha_deg = np.rad2deg(alpha_rw) % 360.0
        C6 = np.array([np.interp(alpha_deg, self._alpha_deg, self._C6[:, j])
                       for j in range(6)])
        tau_w6 = U_rw ** 2 * C6

        info = {"U": U, "beta_ned": beta_t, "alpha_body": alpha_rw,
                "U_mean": U_mean, "U_gust": U_gust}
        return tau_w6, info
