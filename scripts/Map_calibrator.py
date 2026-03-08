import cv2
import numpy as np
import json

# Chessboard specification
# 12 x 9 squares  ->  11 x 8 inner corners
CHESSBOARD_COLS = 11
CHESSBOARD_ROWS = 8
SQUARE_SIZE_MM = 6.0


# Load ROI from Map_Tuner
with open("binary_params.json", "r") as f:
    params = json.load(f)

ROI = tuple(params["ROI"])  # (x, y, w, h)


# Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)


# UI state
status_msg = "Press [S] to lock camera"
status_color = (0, 255, 0)


# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    x, y, w, h = ROI
    roi = frame[y:y+h, x:x+w]

    vis = roi.copy()
    cv2.putText(
        vis,
        status_msg,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
        cv2.LINE_AA
    )
    cv2.imshow("ROI", vis)

    key = cv2.waitKey(1) & 0xFF

    # Press S -> lock & calibrate
    if key == ord('s'):
        frozen_roi = roi.copy()
        gray = cv2.cvtColor(frozen_roi, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            (CHESSBOARD_COLS, CHESSBOARD_ROWS),
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not found:
            status_msg = "Chessboard not detected"
            status_color = (0, 0, 255)
            continue

        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )
        )

        # Object points (mm)
        objp = np.zeros((CHESSBOARD_ROWS * CHESSBOARD_COLS, 3), np.float32)
        objp[:, :2] = np.mgrid[
            0:CHESSBOARD_COLS,
            0:CHESSBOARD_ROWS
        ].T.reshape(-1, 2)
        objp *= SQUARE_SIZE_MM

        h_img, w_img = gray.shape


        # Camera calibration
        _, K, D, rvecs, tvecs = cv2.calibrateCamera(
            [objp],
            [corners],
            (w_img, h_img),
            None,
            None
        )

        # Reprojection error
        proj, _ = cv2.projectPoints(objp, rvecs[0], tvecs[0], K, D)
        reproj_err = cv2.norm(corners, proj, cv2.NORM_L2) / len(proj)

        # Homography (pixel -> mm)
        pixel_pts = corners.reshape(-1, 2)
        world_pts = objp[:, :2]

        H, _ = cv2.findHomography(pixel_pts, world_pts)

        # Save results
        calib = {
            "camera_matrix": K.tolist(),
            "dist_coeffs": D.flatten().tolist(),
            "reprojection_error": float(reproj_err),
            "roi": list(ROI),
            "image_size": [w_img, h_img],
            "chessboard": {
                "squares_cols": 12,
                "squares_rows": 9,
                "inner_cols": CHESSBOARD_COLS,
                "inner_rows": CHESSBOARD_ROWS,
                "square_size_mm": SQUARE_SIZE_MM
            }
        }

        with open("calibration.json", "w") as f:
            json.dump(calib, f, indent=2)

        np.save("homography.npy", H)

        # Success UI: freeze 2 seconds then exit
        status_msg = "Calibration OK"
        status_color = (0, 255, 0)

        vis = frozen_roi.copy()
        cv2.drawChessboardCorners(
            vis,
            (CHESSBOARD_COLS, CHESSBOARD_ROWS),
            corners,
            True
        )
        cv2.putText(
            vis,
            status_msg,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
            cv2.LINE_AA
        )
        cv2.imshow("ROI", vis)

        cv2.waitKey(2000)  # freeze 2 seconds
        break

    if key == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
