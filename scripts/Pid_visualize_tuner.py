import cv2
import json
import numpy as np

from Map_config import MapConfig
from Reference_path import ReferencePath
from Tracker import ProjectionTracker
from Pid_controller import PIDController, PIDGains
from Draw_tracker_pid_viz import draw_tracker_pid


# ============================================================
# Yellow tip detection (same as planner)
# ============================================================
LOWER_YELLOW = np.array([20, 80, 80])
UPPER_YELLOW = np.array([35, 255, 255])
MIN_AREA = 80
SMOOTH_ALPHA = 0.05


def detect_yellow_tip(roi_frame, prev_tip=None):
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_AREA:
        return None

    pts = cnt.reshape(-1, 2)
    center = np.mean(pts, axis=0)

    if prev_tip is None:
        dists = np.linalg.norm(pts - center, axis=1)
    else:
        dists = np.linalg.norm(pts - prev_tip, axis=1)

    tip = pts[np.argmax(dists)]
    return tuple(map(int, tip))


# ============================================================
# Grid obstacle mask (grid==0 → gray overlay)
# ============================================================
def build_obstacle_mask(grid, roi_h, roi_w, cell_size):
    """
    grid: (gh, gw), 0 = obstacle
    return: uint8 mask, 255 = obstacle pixel
    """
    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    gh, gw = grid.shape

    for gy in range(gh):
        for gx in range(gw):
            if grid[gy, gx] == 0:   # obstacle
                y1 = gy * cell_size
                y2 = min(y1 + cell_size, roi_h)
                x1 = gx * cell_size
                x2 = min(x1 + cell_size, roi_w)
                mask[y1:y2, x1:x2] = 255

    return mask


# ============================================================
# Main loop
# ============================================================
def main():
    # --------------------------------------------------------
    # Load map & reference path
    # --------------------------------------------------------
    cfg = MapConfig()

    with open("planned_cells.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    path_cells = [tuple(p) for p in data["path_cells"]]
    ref = ReferencePath(path_cells, cfg)

    # --------------------------------------------------------
    # Build obstacle mask (ONCE)
    # --------------------------------------------------------
    x0, y0, w, h = cfg.roi

    obstacle_mask = build_obstacle_mask(
        cfg.grid,
        roi_h=h,
        roi_w=w,
        cell_size=cfg.cell_size
    )

    # --------------------------------------------------------
    # Tracker & PID
    # --------------------------------------------------------
    tracker = ProjectionTracker(
        ref,
        lookahead_dist=30.0,
        s_ema_alpha=0.35
    )

    pid = PIDController(
        PIDGains(
            kp=0.04,
            ki=0.0,
            kd=0.004,
            k_psi=0.8
        ),
        i_limit=200.0,
        u_limit=1.0
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)

    prev_tip = None
    filtered_tip = None

    print("=== Tracker + PID visualization ===")
    print("ESC: quit")

    # --------------------------------------------------------
    # Live loop
    # --------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[y0:y0+h, x0:x0+w].copy()
        vis = roi.copy()

        # ------------------------------------
        # Obstacle overlay (same as path_planner)
        # ------------------------------------
        vis[obstacle_mask > 0] = (180, 180, 180)

        # ------------------------------------
        # Detect & smooth tip
        # ------------------------------------
        tip = detect_yellow_tip(roi, prev_tip)

        tip_xy = None
        if tip is not None:
            filtered_tip = np.array(tip, float) if filtered_tip is None else (
                SMOOTH_ALPHA * np.array(tip)
                + (1.0 - SMOOTH_ALPHA) * filtered_tip
            )
            prev_tip = filtered_tip.copy()
            tx, ty = map(int, filtered_tip)
            tip_xy = (tx, ty)

        # ------------------------------------
        # Tracker + PID (only if tip exists)
        # ------------------------------------
        if tip_xy is not None:
            track_state = tracker.update(tip_xy)
            pid_state = pid.update(
                e_y=track_state.e_y,
                e_psi=track_state.e_psi
            )

            vis = draw_tracker_pid(
                vis,
                tip_xy,
                ref,
                track_state,
                pid_state
            )
        else:
            cv2.putText(
                vis,
                "No TIP detected",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        cv2.imshow("ROI", vis)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:   # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
