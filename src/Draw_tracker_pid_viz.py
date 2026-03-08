import cv2
import numpy as np


def draw_tracker_pid(
    vis,                       # ROI image (BGR)
    tip_xy,                    # (tx, ty)
    ref_path,                  # ReferencePath
    track_state,               # TrackState
    pid_state,                 # PIDState
):
    """
    Draw tracker + PID visualization on ROI image.
    """

    h, w = vis.shape[:2]

    # 1. draw reference path
    curve = ref_path.curve_points.astype(int)
    if len(curve) >= 2:
        cv2.polylines(vis, [curve], False, (192, 0, 192), 2)


    # 2. draw tip
    tx, ty = int(tip_xy[0]), int(tip_xy[1])
    cv2.circle(vis, (tx, ty), 4, (0, 0, 255), -1)
    cv2.putText(vis, "TIP", (tx + 6, ty - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


    # 3. draw projection point
    px, py = map(int, track_state.proj_xy)
    cv2.circle(vis, (px, py), 4, (255, 0, 0), -1)
    cv2.putText(vis, "PROJ", (px + 6, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 4. draw lookahead point
    lx, ly = map(int, track_state.lookahead_xy)
    cv2.circle(vis, (lx, ly), 4, (0, 255, 0), -1)
    cv2.putText(vis, "LA", (lx + 6, ly - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 5. draw lateral error vector (e_y)
    # from projection to tip
    cv2.arrowedLine(
        vis,
        (px, py),
        (tx, ty),
        (255, 0, 255),
        2,
        tipLength=0.2
    )

    # 6. text panel (PID info)
    y0 = 20
    dy = 18

    def put(line, text, color=(255, 255, 255)):
        cv2.putText(
            vis,
            text,
            (10, y0 + line * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

    put(0, f"s      : {track_state.s:7.1f}")
    put(1, f"e_y    : {track_state.e_y:7.2f}")
    put(2, f"e_psi  : {track_state.e_psi:7.3f}")
    put(3, f"u      : {pid_state.u:7.3f}", (0, 255, 255))
    put(4, f"P term : {pid_state.p:7.3f}")
    put(5, f"I term : {pid_state.i:7.3f}")
    put(6, f"D term : {pid_state.d:7.3f}")

    return vis
