# tracker.py
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrackState:
    s: float                # arc-length position on reference path
    e_y: float              # signed lateral error (pixels)
    e_psi: float            # heading error (radians), optional if you have heading
    proj_xy: Tuple[float, float]      # projection point on path
    lookahead_xy: Tuple[float, float] # lookahead point on path


class ProjectionTracker:
    """
    Projection-based tracker.

    - Works in the same coordinate space as ReferencePath.curve_points:
        default: ROI pixel space

    - Produces:
        s (progress), e_y (cross-track), e_psi (heading error if heading is available)
        plus a lookahead point for Pure Pursuit / Stanley style control.

    Notes:
    - You can later convert all px to mm, tracker math stays identical.
    """

    def __init__(
        self,
        ref_path,
        *,
        lookahead_dist: float = 30.0,   # pixels, tune later
        s_ema_alpha: float = 0.35,      # smooth the estimated s to reduce jitter
        max_s_jump: float = 80.0,       # clamp sudden jumps in s (pixels of arc-length)
    ):
        self.ref = ref_path
        self.lookahead_dist = float(lookahead_dist)
        self.s_ema_alpha = float(s_ema_alpha)
        self.max_s_jump = float(max_s_jump)

        self._s_filt: Optional[float] = None

        # cached curve arrays
        self._P = self.ref.curve_points          # (N,2)
        self._S = self.ref.s_values              # (N,)
        if len(self._P) < 2:
            raise ValueError("ReferencePath curve_points must have at least 2 points")

        # precompute segments
        self._A = self._P[:-1]                   # (N-1,2)
        self._B = self._P[1:]                    # (N-1,2)
        self._AB = self._B - self._A             # (N-1,2)
        self._AB2 = np.sum(self._AB * self._AB, axis=1) + 1e-9  # avoid div0

    def reset(self):
        self._s_filt = None

    def _project_point_to_polyline(self, x: float, y: float):
        """
        Project point Q onto polyline segments A->B.
        Returns:
            s_proj, proj_xy, seg_t, seg_idx
        """
        Q = np.array([x, y], dtype=float)

        # vector from A to Q
        AQ = Q - self._A                         # (N-1,2)
        # t = dot(AQ,AB)/|AB|^2, clamped [0,1]
        t = np.sum(AQ * self._AB, axis=1) / self._AB2
        t = np.clip(t, 0.0, 1.0)

        # projection points
        Pproj = self._A + self._AB * t[:, None]  # (N-1,2)
        d2 = np.sum((Pproj - Q) ** 2, axis=1)    # (N-1,)

        i = int(np.argmin(d2))
        proj = Pproj[i]
        ti = float(t[i])

        # s along the curve: interpolate between vertex s-values
        s0 = float(self._S[i])
        s1 = float(self._S[i + 1])
        s_proj = s0 + ti * (s1 - s0)

        return s_proj, (float(proj[0]), float(proj[1])), ti, i

    def _interp_on_curve(self, s_query: float) -> Tuple[float, float]:
        """Fast linear interpolation on stored (S,P)."""
        s_query = float(np.clip(s_query, 0.0, float(self._S[-1])))

        j = int(np.searchsorted(self._S, s_query) - 1)
        j = max(0, min(j, len(self._S) - 2))

        s0, s1 = float(self._S[j]), float(self._S[j + 1])
        t = (s_query - s0) / (s1 - s0 + 1e-9)

        p0 = self._P[j]
        p1 = self._P[j + 1]
        p = (1 - t) * p0 + t * p1
        return float(p[0]), float(p[1])

    def _tangent_on_curve(self, s_query: float, ds: float = 5.0) -> Tuple[float, float]:
        """Finite-difference tangent."""
        s_query = float(np.clip(s_query, 0.0, float(self._S[-1])))
        s0 = np.clip(s_query - ds, 0.0, float(self._S[-1]))
        s1 = np.clip(s_query + ds, 0.0, float(self._S[-1]))
        p0 = np.array(self._interp_on_curve(s0))
        p1 = np.array(self._interp_on_curve(s1))
        v = p1 - p0
        n = np.linalg.norm(v)
        if n < 1e-9:
            return 1.0, 0.0
        v /= n
        return float(v[0]), float(v[1])

    def update(
        self,
        tip_xy: Tuple[float, float],
        *,
        tip_heading: Optional[float] = None,  # radians, optional
    ) -> TrackState:
        """
        Args:
            tip_xy: (x,y) current tip position in same coord space as ReferencePath (ROI px)
            tip_heading: optional heading angle (radians). If None, e_psi = 0.

        Returns:
            TrackState with s, e_y, e_psi, projection point, lookahead point.
        """
        x, y = float(tip_xy[0]), float(tip_xy[1])

        # 1) project to path -> raw s
        s_raw, proj_xy, _, _ = self._project_point_to_polyline(x, y)

        # 2) smooth / clamp s to avoid jumping due to measurement noise
        if self._s_filt is None:
            s_f = s_raw
        else:
            # clamp jump
            s_raw = np.clip(s_raw, self._s_filt - self.max_s_jump, self._s_filt + self.max_s_jump)
            s_f = self.s_ema_alpha * s_raw + (1.0 - self.s_ema_alpha) * self._s_filt
        self._s_filt = float(s_f)

        # 3) compute signed lateral error e_y
        # e_y = dot( (tip - proj), normal )
        tx, ty = self._tangent_on_curve(self._s_filt)
        nx, ny = -ty, tx  # left normal
        v = np.array([x - proj_xy[0], y - proj_xy[1]], dtype=float)
        e_y = float(v[0] * nx + v[1] * ny)

        # 4) heading error e_psi (optional)
        if tip_heading is None:
            e_psi = 0.0
        else:
            path_heading = float(np.arctan2(ty, tx))
            # wrap to [-pi,pi]
            d = float(tip_heading - path_heading)
            e_psi = float((d + np.pi) % (2 * np.pi) - np.pi)

        # 5) lookahead point
        s_la = float(np.clip(self._s_filt + self.lookahead_dist, 0.0, float(self._S[-1])))
        lookahead_xy = self._interp_on_curve(s_la)

        return TrackState(
            s=self._s_filt,
            e_y=e_y,
            e_psi=e_psi,
            proj_xy=proj_xy,
            lookahead_xy=lookahead_xy,
        )
