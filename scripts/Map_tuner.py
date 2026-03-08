import cv2
import numpy as np
import json


def nothing(x):
    pass



# State definition

STATE_PREVIEW = 0
STATE_BINARY  = 1
STATE_GRID    = 2
STATE_LOCKED  = 3

state = STATE_PREVIEW


# Camera

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")



# Windows (create ONLY Live initially)

cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live", 640, 480)
cv2.moveWindow("Live", 50, 50)


# Trackbars (Binary / Grid)

def create_binary_window():
    cv2.namedWindow("Binary", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Binary", 480, 480)
    cv2.moveWindow("Binary", 720, 50)

    dummy = np.zeros((10, 10), dtype=np.uint8)
    cv2.imshow("Binary", dummy)
    cv2.waitKey(1)

    cv2.createTrackbar("Gray Thresh", "Binary", 10, 255, nothing)
    cv2.createTrackbar("Invert", "Binary", 1, 1, nothing)
    cv2.createTrackbar("Morph Close", "Binary", 1, 5, nothing)


def create_grid_window():
    cv2.namedWindow("Grid", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Grid", 480, 480)
    cv2.moveWindow("Grid", 720, 560)

    dummy = np.zeros((10, 10), dtype=np.uint8)
    cv2.imshow("Grid", dummy)
    cv2.waitKey(1)

    cv2.createTrackbar("Cell Size", "Grid", 10, 60, nothing)
    cv2.createTrackbar("Occ Thresh %", "Grid", 50, 100, nothing)
    cv2.createTrackbar("Erode Radius", "Grid", 0, 15, nothing)



# State data

roi = None
sampled = False

sample_frame = None
roi_frame = None

binary_frozen = None
grid_frozen = None
grid_params = None


print("Controls:")
print("  c : select ROI")
print("  s : sample ONE frame")
print("  b : lock binary")
print("  g : lock grid")
print("  ESC : quit")



# Binary → Grid

def binary_to_grid(binary, cell_size, occ_thresh):
    h, w = binary.shape
    gh = h // cell_size
    gw = w // cell_size

    grid = np.zeros((gh, gw), dtype=np.uint8)

    for gy in range(gh):
        for gx in range(gw):
            cell = binary[
                gy * cell_size:(gy + 1) * cell_size,
                gx * cell_size:(gx + 1) * cell_size
            ]
            ratio = cv2.countNonZero(cell) / (cell_size * cell_size)
            grid[gy, gx] = 1 if ratio > occ_thresh else 0

    return grid


def render_grid_on_roi(grid, roi_h, roi_w, cell_size):
    canvas = np.zeros((roi_h, roi_w), dtype=np.uint8)
    gh, gw = grid.shape

    for gy in range(gh):
        for gx in range(gw):
            if grid[gy, gx]:
                y1 = gy * cell_size
                y2 = min(y1 + cell_size, roi_h)
                x1 = gx * cell_size
                x2 = min(x1 + cell_size, roi_w)
                canvas[y1:y2, x1:x2] = 255

    return canvas



# Main loop

while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF

    # -------------------- ROI selection ---------------------
    if key == ord('c') and not sampled:
        roi = cv2.selectROI("Live", frame, fromCenter=False, showCrosshair=True)

    if roi is None:
        cv2.imshow("Live", frame)
        if key == 27:
            break
        continue

    x, y, w, h = roi

    # -------------------- Sample frame ----------------------
    if key == ord('s') and not sampled:
        sample_frame = frame.copy()
        roi_frame = sample_frame[y:y+h, x:x+w]
        sampled = True
        state = STATE_BINARY

        create_binary_window()
        print("Frame sampled → Binary tuning")


    # PREVIEW

    if state == STATE_PREVIEW:
        preview = frame.copy()
        cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            preview, "Press 's' to sample",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 0, 255), 2
        )
        cv2.imshow("Live", preview)


    # BINARY TUNING

    elif state == STATE_BINARY:
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        thresh = cv2.getTrackbarPos("Gray Thresh", "Binary")
        invert = cv2.getTrackbarPos("Invert", "Binary")
        k = cv2.getTrackbarPos("Morph Close", "Binary")

        mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, binary = cv2.threshold(gray, thresh, 255, mode)

        if k > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1)
            )
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        cv2.imshow("Live", roi_frame)
        cv2.imshow("Binary", binary)

        if key == ord('b'):
            binary_frozen = binary.copy()
            state = STATE_GRID

            cv2.destroyWindow("Live")
            create_grid_window()

            with open("binary_params.json", "w") as f:
                json.dump({
                    "ROI": list(roi),
                    "GRAY_THRESH": int(thresh),
                    "INVERT": int(invert),
                    "MORPH_CLOSE_K": int(k)
                }, f, indent=2)

            print("Binary locked → Grid tuning")

    
    # GRID TUNING

    elif state == STATE_GRID:
        cell_size = max(2, cv2.getTrackbarPos("Cell Size", "Grid"))
        occ_thresh = cv2.getTrackbarPos("Occ Thresh %", "Grid") / 100.0
        erode_r = cv2.getTrackbarPos("Erode Radius", "Grid")

        binary_safe = binary_frozen.copy()
        if erode_r > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2*erode_r+1, 2*erode_r+1)
            )
            binary_safe = cv2.erode(binary_safe, kernel)

        grid = binary_to_grid(binary_safe, cell_size, occ_thresh)
        roi_h, roi_w = binary_safe.shape
        grid_vis = render_grid_on_roi(grid, roi_h, roi_w, cell_size)

        cv2.imshow("Binary", binary_safe)
        cv2.imshow("Grid", grid_vis)

        if key == ord('g'):
            grid_frozen = grid.copy()
            state = STATE_LOCKED

            cv2.destroyWindow("Binary")

            grid_params = {
                "CELL_SIZE": int(cell_size),
                "OCC_THRESH": float(occ_thresh),
                "ERODE_RADIUS": int(erode_r)
            }

            with open("grid_params.json", "w") as f:
                json.dump(grid_params, f, indent=2)

            np.save("grid.npy", grid_frozen)

            print("Grid locked")


    # LOCKED

    elif state == STATE_LOCKED:
        roi_h, roi_w = binary_frozen.shape
        grid_vis = render_grid_on_roi(
            grid_frozen,
            roi_h,
            roi_w,
            grid_params["CELL_SIZE"]
        )

        cv2.putText(
            grid_vis,
            "GRID LOCKED",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, 255, 2
        )
        cv2.imshow("Grid", grid_vis)

    if key == 27:
        break



# Cleanup

cap.release()
cv2.destroyAllWindows()
