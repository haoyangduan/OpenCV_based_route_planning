# reference_path.py
import numpy as np
from typing import List, Tuple
from Map_config import MapConfig


class ReferencePath:
    """
    ReferencePath
    -------------
    Pure geometric layer.

    Input:
        path_cells : List[(cx, cy)]  -- discrete safe path from Path Planner
        cfg        : MapConfig

    Output (via methods / properties):
        - continuous curve points (ROI pixel space)
        - arc-length parameterization
        - position(s), tangent(s), normal(s), curvature(s)

    Design principles:
        - NO planning
        - NO collision checking
        - NO dependency on Planner's Bezier
        - ONLY geometry for Tracker / Controller
    """

    # =========================================================
    # Construction
    # =========================================================
    def __init__(
        self,
        path_cells: List[Tuple[int, int]],
        cfg: MapConfig,
        *,
        corner_angle_deg: float = 10.0,
        samples_per_segment: int = 20,
    ):
        if path_cells is None or len(path_cells) < 2:
            raise ValueError("ReferencePath requires at least 2 cells")

        self.cfg = cfg
        self.path_cells = path_cells
        self.corner_angle_deg = corner_angle_deg
        self.samples_per_segment = samples_per_segment

        # build pipeline
        self._pts_px = self._cells_to_points(path_cells)
        self._ctrl_pts = self._extract_corners(self._pts_px)
        self._curve_pts = self._build_catmull_rom_curve(self._ctrl_pts)
        self._build_arc_length()


    # Step 1: cell -> continuous point (cell center, ROI px)
    def _cells_to_points(self, cells):
        pts = []
        for cx, cy in cells:
            px, py = self.cfg.cell_to_roi_px(cx, cy)
            pts.append((float(px), float(py)))
        return pts

    # Step 2: corner extraction (remove micro zig-zag)
    def _extract_corners(self, pts):
        if len(pts) < 3:
            return pts

        ang_th = np.deg2rad(self.corner_angle_deg)
        pts = [np.array(p, float) for p in pts]

        out = [tuple(pts[0])]
        for i in range(1, len(pts) - 1):
            a, b, c = pts[i - 1], pts[i], pts[i + 1]
            v1 = b - a
            v2 = c - b

            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue

            u1 = v1 / n1
            u2 = v2 / n2
            ang = np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))
            if ang >= ang_th:
                out.append(tuple(b))

        out.append(tuple(pts[-1]))

        # deduplicate
        dedup = [out[0]]
        for p in out[1:]:
            if np.linalg.norm(np.array(p) - np.array(dedup[-1])) > 1e-3:
                dedup.append(p)
        return dedup

    # Step 3: Catmull-Rom spline (C1 continuous)
    def _catmull_rom_to_bezier(self, Pm1, P0, P1, P2, tension=0.5):
        Pm1 = np.array(Pm1)
        P0  = np.array(P0)
        P1  = np.array(P1)
        P2  = np.array(P2)

        B0 = P0
        B3 = P1
        B1 = P0 + (P1 - Pm1) * (tension / 3.0)
        B2 = P1 - (P2 - P0)  * (tension / 3.0)
        return B0, B1, B2, B3

    def _sample_bezier(self, B0, B1, B2, B3, n):
        pts = []
        for i in range(n):
            t = i / (n - 1)
            u = 1.0 - t
            P = (
                (u**3) * B0
                + 3 * (u**2) * t * B1
                + 3 * u * (t**2) * B2
                + (t**3) * B3
            )
            pts.append((float(P[0]), float(P[1])))
        return pts

    def _build_catmull_rom_curve(self, ctrl_pts):
        if len(ctrl_pts) <= 2:
            return ctrl_pts

        pts = [ctrl_pts[0]] + ctrl_pts + [ctrl_pts[-1]]
        curve = []

        for i in range(1, len(pts) - 2):
            Pm1, P0, P1, P2 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
            B0, B1, B2, B3 = self._catmull_rom_to_bezier(Pm1, P0, P1, P2)
            seg = self._sample_bezier(B0, B1, B2, B3, self.samples_per_segment)
            if curve:
                seg = seg[1:]  # avoid duplicate joint
            curve.extend(seg)

        return curve
    
    # Step 4: arc-length parameterization
    def _build_arc_length(self):
        pts = np.array(self._curve_pts, dtype=float)

        if len(pts) < 2:
            self._s = np.array([0.0])
            self.length = 0.0
            return

        ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self._s = np.concatenate([[0.0], np.cumsum(ds)])
        self.length = float(self._s[-1])


    #Public read-only properties for Tracker
    @property
    def curve_points(self) -> np.ndarray:
        """(N,2) array of continuous path points (ROI pixel space)."""
        return np.array(self._curve_pts, dtype=float)

    @property
    def s_values(self) -> np.ndarray:
        """(N,) arc-length values corresponding to curve_points."""
        return np.array(self._s, dtype=float)

    # Query interface 
    def position(self, s_query: float) -> Tuple[float, float]:
        s_query = float(np.clip(s_query, 0.0, self.length))
        idx = int(np.searchsorted(self._s, s_query) - 1)
        idx = max(0, min(idx, len(self._curve_pts) - 2))

        s0, s1 = self._s[idx], self._s[idx + 1]
        t = (s_query - s0) / (s1 - s0 + 1e-9)

        p0 = np.array(self._curve_pts[idx])
        p1 = np.array(self._curve_pts[idx + 1])
        p = (1 - t) * p0 + t * p1
        return float(p[0]), float(p[1])

    def tangent(self, s_query: float) -> Tuple[float, float]:
        ds = 3.0
        s0 = np.clip(s_query - ds, 0.0, self.length)
        s1 = np.clip(s_query + ds, 0.0, self.length)
        p0 = np.array(self.position(s0))
        p1 = np.array(self.position(s1))
        v = p1 - p0
        n = np.linalg.norm(v)
        if n < 1e-9:
            return 1.0, 0.0
        v /= n
        return float(v[0]), float(v[1])

    def normal(self, s_query: float) -> Tuple[float, float]:
        tx, ty = self.tangent(s_query)
        return -ty, tx

    def curvature(self, s_query: float) -> float:
        ds = 3.0
        p_prev = np.array(self.position(s_query - ds))
        p      = np.array(self.position(s_query))
        p_next = np.array(self.position(s_query + ds))

        v1 = p - p_prev
        v2 = p_next - p
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return 0.0

        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = np.dot(v1, v2)
        angle = np.arctan2(cross, dot)
        return float(angle / (n1 + n2 + 1e-6))
