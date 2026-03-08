# path_planner.py
import cv2
import json
import heapq
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from src.Map_config import MapConfig


# Yellow tip (HSV) parameters
LOWER_YELLOW = np.array([20, 80, 80])
UPPER_YELLOW = np.array([35, 255, 255])
MIN_AREA = 80
SMOOTH_ALPHA = 0.05


# Planning parameters (center bias)
WALL_WEIGHT = 6.0   # bigger -> more center bias, less wall hugging


# Optional: Bezier smoothing for VISUALIZATION ONLY
# (Planner output remains path_cells)
SHOW_BEZIER_PREVIEW = True
CORNER_ANGLE_DEG = 10.0
BEZIER_SAMPLES_PER_SEG = 18
COLLISION_CHECK = True



# Helpers: clearance / masks / tip detection
def compute_clearance(grid: np.ndarray) -> np.ndarray:
    """
    grid: 1 free, 0 obstacle
    returns: distance to nearest obstacle (in pixel units of distanceTransform grid)
    """
    obstacle = (grid == 0).astype(np.uint8)
    free = (obstacle == 0).astype(np.uint8) * 255
    return cv2.distanceTransform(free, cv2.DIST_L2, 5)


def render_grid_mask_obstacles(grid: np.ndarray, roi_h: int, roi_w: int, cell_size: int) -> np.ndarray:
    """Obstacle mask in ROI pixel space (for visualization and optional collision check)."""
    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    gh, gw = grid.shape
    for gy in range(gh):
        for gx in range(gw):
            if grid[gy, gx] == 0:
                y1 = gy * cell_size
                y2 = min(y1 + cell_size, roi_h)
                x1 = gx * cell_size
                x2 = min(x1 + cell_size, roi_w)
                mask[y1:y2, x1:x2] = 255
    return mask


def detect_yellow_tip(roi_frame: np.ndarray, prev_tip: Optional[np.ndarray] = None) -> Tuple[Optional[Tuple[int, int]], np.ndarray]:
    """
    Return (tip_xy, mask). tip_xy in ROI pixel coordinates.
    """
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_AREA:
        return None, mask

    pts = cnt.reshape(-1, 2)
    center = np.mean(pts, axis=0)

    if prev_tip is None:
        dists = np.linalg.norm(pts - center, axis=1)
    else:
        dists = np.linalg.norm(pts - prev_tip, axis=1)

    tip = pts[np.argmax(dists)]
    return tuple(map(int, tip)), mask


# Core: A* with clearance-weighted cost (center bias)
def astar_clearance_weighted(
    grid: np.ndarray,
    clearance: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    wall_weight: float,
) -> Optional[List[Tuple[int, int]]]:
    """
    grid[y,x] == 1 free, 0 obstacle
    returns: list of (cx,cy) from start to goal, or None
    """
    h, w = grid.shape
    sx, sy = start
    gx, gy = goal

    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return None
    if grid[sy, sx] == 0 or grid[gy, gx] == 0:
        return None

    # 8-connected
    nbrs = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]

    def heur(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def step_cost(nx, ny, dx, dy):
        base = np.hypot(dx, dy)  # 1 or sqrt(2)
        if wall_weight <= 0:
            return base
        d = float(clearance[ny, nx])
        # d small near obstacle -> penalty big; d large in center -> penalty small
        wall_pen = wall_weight / (d + 1e-3)
        return base + wall_pen

    openq = []
    heapq.heappush(openq, (heur(start, goal), 0.0, start))
    came = {start: None}
    gscore = {start: 0.0}

    while openq:
        _, g, cur = heapq.heappop(openq)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = came[cur]
            return path[::-1]

        if g > gscore.get(cur, 1e18):
            continue

        cx, cy = cur
        for dx, dy in nbrs:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if grid[ny, nx] == 0:
                continue

            ng = g + step_cost(nx, ny, dx, dy)
            nxt = (nx, ny)
            if ng < gscore.get(nxt, 1e18):
                gscore[nxt] = ng
                came[nxt] = cur
                f = ng + heur(nxt, goal)
                heapq.heappush(openq, (f, ng, nxt))

    return None


# Optional: Bezier smoothing for preview (NOT output)
def polyline_collides(points: List[Tuple[int, int]], obstacle_mask: np.ndarray) -> bool:
    h, w = obstacle_mask.shape
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        steps = max(abs(x2 - x1), abs(y2 - y1), 1)
        for s in range(steps + 1):
            t = s / steps
            x = int(round(x1 + (x2 - x1) * t))
            y = int(round(y1 + (y2 - y1) * t))
            if 0 <= x < w and 0 <= y < h:
                if obstacle_mask[y, x] > 0:
                    return True
    return False


def extract_corners(points: List[Tuple[int, int]], angle_deg: float = 12.0) -> List[Tuple[int, int]]:
    if points is None or len(points) < 3:
        return points

    ang_th = np.deg2rad(angle_deg)
    pts = [np.array(p, dtype=float) for p in points]
    out = [tuple(map(int, pts[0]))]

    for i in range(1, len(pts) - 1):
        a = pts[i - 1]
        b = pts[i]
        c = pts[i + 1]

        v1 = b - a
        v2 = c - b
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue

        u1 = v1 / n1
        u2 = v2 / n2
        dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
        ang = np.arccos(dot)
        if ang >= ang_th:
            out.append(tuple(map(int, b)))

    out.append(tuple(map(int, pts[-1])))

    dedup = [out[0]]
    for p in out[1:]:
        if p != dedup[-1]:
            dedup.append(p)
    return dedup


def bezier_sample(P0, P1, P2, P3, n: int = 18) -> List[Tuple[int, int]]:
    P0 = np.array(P0, float)
    P1 = np.array(P1, float)
    P2 = np.array(P2, float)
    P3 = np.array(P3, float)

    out = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.0
        u = 1.0 - t
        B = (u**3) * P0 + 3 * (u**2) * t * P1 + 3 * u * (t**2) * P2 + (t**3) * P3
        out.append((int(round(B[0])), int(round(B[1]))))
    return out


def catmullrom_to_bezier(Pm1, P0, P1, P2, tension: float = 0.5):
    Pm1 = np.array(Pm1, float)
    P0  = np.array(P0, float)
    P1  = np.array(P1, float)
    P2  = np.array(P2, float)

    B0 = P0
    B3 = P1
    B1 = P0 + (P1 - Pm1) * (tension / 3.0)
    B2 = P1 - (P2 - P0)  * (tension / 3.0)
    return tuple(B0), tuple(B1), tuple(B2), tuple(B3)


def bezier_preview_polyline(
    raw_pts: List[Tuple[int, int]],
    obstacle_mask: np.ndarray,
    angle_deg: float = 12.0,
    samples_per_seg: int = 18,
) -> List[Tuple[int, int]]:
    if raw_pts is None or len(raw_pts) < 2:
        return raw_pts

    ctrl = extract_corners(raw_pts, angle_deg=angle_deg)
    if ctrl is None or len(ctrl) < 2:
        return raw_pts
    if len(ctrl) == 2:
        return ctrl

    pts = [ctrl[0]] + ctrl + [ctrl[-1]]

    smooth = []
    for i in range(1, len(pts) - 2):
        Pm1 = pts[i - 1]
        P0  = pts[i]
        P1  = pts[i + 1]
        P2  = pts[i + 2]

        B0, B1, B2, B3 = catmullrom_to_bezier(Pm1, P0, P1, P2, tension=0.5)
        seg = bezier_sample(B0, B1, B2, B3, n=max(4, int(samples_per_seg)))

        if smooth:
            seg = seg[1:]
        smooth.extend(seg)

    if COLLISION_CHECK and polyline_collides(smooth, obstacle_mask):
        return ctrl

    return smooth


# PathPlanner class (formal output: cell list)
class PathPlanner:
    def __init__(self, cfg: MapConfig, wall_weight: float = WALL_WEIGHT):
        self.cfg = cfg
        self.grid = cfg.grid
        self.clearance = compute_clearance(self.grid)
        self.wall_weight = float(wall_weight)

    def plan(self, start_cell: Tuple[int, int], goal_cell: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        return astar_clearance_weighted(
            self.grid, self.clearance, start_cell, goal_cell, self.wall_weight
        )


# Demo main: Live camera + ROI + tip start + mouse goal + plan once
goal_cell: Optional[Tuple[int, int]] = None
planned_cells: Optional[List[Tuple[int, int]]] = None
status_msg = "Click ROI to set GOAL. Yellow tip is START."
planned_once = False


def on_mouse(event, mx, my, flags, cfg: MapConfig):
    global goal_cell, planned_cells, planned_once, status_msg
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    cx, cy = cfg.roi_px_to_cell(mx, my)
    if not cfg.cell_in_bounds(cx, cy):
        status_msg = "Goal: out of bounds"
        return
    if not cfg.cell_is_free(cx, cy):
        status_msg = "Goal: blocked cell"
        return

    goal_cell = (cx, cy)
    planned_cells = None
    planned_once = False
    status_msg = f"Goal set: {goal_cell} (planning when START is detected)"


def save_cells(path_cells: List[Tuple[int, int]], start_cell: Tuple[int, int], goal_cell: Tuple[int, int]):
    out = {
        "start_cell": [int(start_cell[0]), int(start_cell[1])],
        "goal_cell": [int(goal_cell[0]), int(goal_cell[1])],
        "path_cells": [[int(cx), int(cy)] for (cx, cy) in path_cells],
    }
    with open("planned_cells.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n=== PathPlanner Output ===")
    print(f"Start: {start_cell}")
    print(f"Goal : {goal_cell}")
    print(f"Cells: {len(path_cells)}")
    print("Saved to planned_cells.json")


def main():
    global goal_cell, planned_cells, status_msg, planned_once

    cfg = MapConfig()
    planner = PathPlanner(cfg, wall_weight=WALL_WEIGHT)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    x0, y0, w, h = cfg.roi
    obstacle_mask = render_grid_mask_obstacles(cfg.grid, h, w, cfg.cell_size)

    cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROI", on_mouse, cfg)

    prev_tip = None
    filtered_tip = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[y0:y0+h, x0:x0+w].copy()

        # start: yellow tip
        tip, _ = detect_yellow_tip(roi, prev_tip)
        start_cell = None

        if tip is not None:
            filtered_tip = np.array(tip) if filtered_tip is None else (
                SMOOTH_ALPHA * np.array(tip) + (1.0 - SMOOTH_ALPHA) * filtered_tip
            )
            prev_tip = filtered_tip.copy()
            tx, ty = map(int, filtered_tip)
            start_cell = cfg.roi_px_to_cell(tx, ty)

        # plan once
        if (not planned_once) and (goal_cell is not None) and (start_cell is not None):
            if cfg.cell_is_free(*start_cell):
                status_msg = "Planning (once)..."
                path = planner.plan(start_cell, goal_cell)
                if path and len(path) >= 2:
                    planned_cells = path
                    planned_once = True
                    status_msg = "Planned ONCE. Press R to reset or click to change goal."
                    save_cells(planned_cells, start_cell, goal_cell)
                else:
                    status_msg = "No path. Try another goal."
            else:
                status_msg = "Start cell blocked (tip on obstacle?)"

        # visualization
        vis = roi.copy()
        vis[obstacle_mask > 0] = (180, 180, 180)

        # draw start
        if start_cell is not None and tip is not None:
            tx, ty = map(int, filtered_tip)
            cv2.circle(vis, (tx, ty), 4, (0, 0, 255), -1)
            cv2.putText(vis, "START", (tx + 6, ty - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # draw goal
        if goal_cell is not None:
            gx, gy = cfg.cell_to_roi_px(*goal_cell)
            cv2.circle(vis, (gx, gy), 6, (0, 255, 0), -1)
            cv2.putText(vis, "GOAL", (gx + 6, gy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # draw path (cell polyline, and optional bezier preview)
        if planned_cells and len(planned_cells) >= 2:
            raw = [cfg.cell_to_roi_px(cx, cy) for (cx, cy) in planned_cells]
            if SHOW_BEZIER_PREVIEW:
                smooth = bezier_preview_polyline(
                    raw, obstacle_mask,
                    angle_deg=CORNER_ANGLE_DEG,
                    samples_per_seg=BEZIER_SAMPLES_PER_SEG
                )
                cv2.polylines(vis, [np.array(smooth, dtype=np.int32)], False, (0, 255, 255), 2)
            else:
                cv2.polylines(vis, [np.array(raw, dtype=np.int32)], False, (0, 255, 255), 2)

        # status
        cv2.putText(vis, status_msg, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("ROI", vis)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break
        if k in (ord('r'), ord('R')):
            goal_cell = None
            planned_cells = None
            planned_once = False
            status_msg = "Reset. Click ROI to set GOAL. Yellow tip is START."

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
