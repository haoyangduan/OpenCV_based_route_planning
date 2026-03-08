# pid_controller.py
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float
    k_psi: float = 0.0   # heading feedforward (optional)


@dataclass
class PIDState:
    u: float             # control output
    p: float             # proportional term
    i: float             # integral term
    d: float             # derivative term
    e_y: float           # lateral error
    e_psi: float         # heading error


class PIDController:
    """
    PID Controller for path tracking.

    Input:
        - e_y   : lateral error (signed)
        - e_psi : heading error (rad, optional)

    Output:
        - u     : control command

    Notes:
        - This controller is STATEFUL (keeps integral & last error)
        - Reset when path is reset / replanned
    """

    def __init__(
        self,
        gains: PIDGains,
        *,
        dt_min: float = 1e-3,
        i_limit: Optional[float] = None,
        u_limit: Optional[float] = None,
        d_filter_alpha: float = 0.3,
    ):
        self.gains = gains
        self.dt_min = dt_min
        self.i_limit = i_limit
        self.u_limit = u_limit
        self.d_filter_alpha = d_filter_alpha

        self._last_time: Optional[float] = None
        self._last_e: Optional[float] = None
        self._i_term: float = 0.0
        self._d_filt: float = 0.0

    # --------------------------------------------------------
    def reset(self):
        self._last_time = None
        self._last_e = None
        self._i_term = 0.0
        self._d_filt = 0.0

    # --------------------------------------------------------
    def update(
        self,
        e_y: float,
        *,
        e_psi: float = 0.0,
        now: Optional[float] = None,
    ) -> PIDState:
        """
        Compute PID output.

        Args:
            e_y   : lateral error
            e_psi : heading error (rad)
            now   : timestamp (seconds). If None, time.time() is used.
        """
        if now is None:
            now = time.time()

        if self._last_time is None:
            dt = self.dt_min
        else:
            dt = max(now - self._last_time, self.dt_min)

        # ---------------------------
        # P term
        p = self.gains.kp * e_y

        # ---------------------------
        # I term
        self._i_term += e_y * dt
        if self.i_limit is not None:
            self._i_term = max(-self.i_limit, min(self.i_limit, self._i_term))
        i = self.gains.ki * self._i_term

        # ---------------------------
        # D term (on measurement)
        if self._last_e is None:
            d_raw = 0.0
        else:
            d_raw = (e_y - self._last_e) / dt

        # low-pass filter derivative
        self._d_filt = (
            self.d_filter_alpha * d_raw
            + (1.0 - self.d_filter_alpha) * self._d_filt
        )
        d = self.gains.kd * self._d_filt

        # ---------------------------
        # Heading feedforward
        h = self.gains.k_psi * e_psi

        # ---------------------------
        # Total output
        u = p + i + d + h

        if self.u_limit is not None:
            u = max(-self.u_limit, min(self.u_limit, u))

        # ---------------------------
        # Update state
        self._last_time = now
        self._last_e = e_y

        return PIDState(
            u=u,
            p=p,
            i=i,
            d=d,
            e_y=e_y,
            e_psi=e_psi,
        )
