import cv2
import json
import time
import numpy as np
import serial

from Map_config import MapConfig
from Reference_path import ReferencePath
from Tracker import ProjectionTracker
from Pid_controller import PIDController, PIDGains


# ============================================================
# serial setting
# ============================================================
ADVANCE_PORT = "COM6"     # 推进 MCU
STEER_PORT   = "COM3"     # 偏转 MCU
BAUDRATE = 115200


# ============================================================
# control const
# ============================================================
U_PERIOD = 0.05            # 20 Hz
V_ACTUAL_CONST = 3.0       # mm/s（真实推进速度）
MAX_STEER_DEG = 2.0        # deg / step（每周期最大角度增量）


# ============================================================
# advance speed calibrate
# v_actual = 6.24 * velocity + 0.24
# ============================================================
def advance_cmd_from_mmps(v_mmps):
    return max(0.0, (v_mmps - 0.24) / 6.24)


VELOCITY_CMD = advance_cmd_from_mmps(V_ACTUAL_CONST)


# ============================================================
# 黄尖端检测
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
# 主程序
# ============================================================
def main():
    # --------------------------------------------------------
    # 串口初始化
    # --------------------------------------------------------
    ser_adv = serial.Serial(ADVANCE_PORT, BAUDRATE, timeout=0.1)
    ser_steer = serial.Serial(STEER_PORT, BAUDRATE, timeout=0.1)
    time.sleep(2.0)

    print("[INFO] Advance MCU connected")
    print("[INFO] Steer MCU connected")

    # --------------------------------------------------------
    # 地图 & 参考路径
    # --------------------------------------------------------
    cfg = MapConfig()

    with open("planned_cells.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    path_cells = [tuple(p) for p in data["path_cells"]]
    ref = ReferencePath(path_cells, cfg)

    # --------------------------------------------------------
    # Tracker & PID
    # --------------------------------------------------------
    tracker = ProjectionTracker(
        ref,
        lookahead_dist=60.0,
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
    # 摄像头
    # --------------------------------------------------------
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    x0, y0, w, h = cfg.roi

    prev_tip = None
    filtered_tip = None
    last_cmd_time = 0.0

    print("=== LIVE LOOP (Δθ steering control) ===")
    print("ESC: quit")

    # --------------------------------------------------------
    # 实时循环
    # --------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[y0:y0 + h, x0:x0 + w]
        now = time.time()

        tip = detect_yellow_tip(roi, prev_tip)

        # -----------------------------
        # Tip 丢失 → 停推进 & 停转
        # -----------------------------
        if tip is None:
            if now - last_cmd_time >= U_PERIOD:
                ser_adv.write(b"ADVANCE 0\n")
                ser_steer.write(b"0\n")   # Δθ = 0
                last_cmd_time = now
            continue

        # -----------------------------
        # 平滑 tip
        # -----------------------------
        filtered_tip = np.array(tip, float) if filtered_tip is None else (
            SMOOTH_ALPHA * np.array(tip)
            + (1.0 - SMOOTH_ALPHA) * filtered_tip
        )
        prev_tip = filtered_tip.copy()
        tip_xy = tuple(map(int, filtered_tip))

        # -----------------------------
        # Tracker + PID
        # -----------------------------
        track_state = tracker.update(tip_xy)

        pid_state = pid.update(
            e_y=track_state.e_y,
            e_psi=track_state.e_psi
        )

        # -----------------------------
        # 下位机命令（限频）
        # -----------------------------
        if now - last_cmd_time >= U_PERIOD:
            u = max(-1.0, min(1.0, pid_state.u))

            # 角度增量（deg / step）
            delta_deg = u * MAX_STEER_DEG

            # 推进（恒速）
            ser_adv.write(f"{VELOCITY_CMD:.3f}\n".encode())
            # 偏转（Δθ 语义）
            ser_steer.write(f"{delta_deg:.3f}\n".encode())

            last_cmd_time = now

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # --------------------------------------------------------
    # 退出清理
    # --------------------------------------------------------
    ser_adv.write(b"ADVANCE 0\n")
    ser_steer.write(b"0\n")
    ser_adv.close()
    ser_steer.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

