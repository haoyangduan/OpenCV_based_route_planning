import json
import numpy as np
from pathlib import Path


class MapConfig:
    """
    Runtime map configuration loader.

    AUTHORITATIVE GRID SEMANTIC:
        grid[y, x] == 1  -> FREE
        grid[y, x] == 0  -> OBSTACLE

    COORDINATE SYSTEMS:
        - Frame pixel (px, py)
        - ROI pixel   (rx, ry)
        - Grid cell   (cx, cy)
        - World mm    (X, Y)
    """

    def __init__(
        self,
        binary_param_path="binary_params.json",
        grid_param_path="grid_params.json",
        grid_path="grid.npy",
        homography_path="homography.npy"
    ):
        self._load_binary_params(binary_param_path)
        self._load_grid_params(grid_param_path)
        self._load_grid(grid_path)
        self._load_homography(homography_path)


    # Loaders
    def _load_binary_params(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")

        with open(path, "r") as f:
            data = json.load(f)

        self.roi = tuple(data["ROI"])   # (x, y, w, h)

    def _load_grid_params(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")

        with open(path, "r") as f:
            data = json.load(f)

        self.cell_size = int(data["CELL_SIZE"])  # pixel
        self.occ_thresh = float(data["OCC_THRESH"])
        self.erode_radius = int(data["ERODE_RADIUS"])

    def _load_grid(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")

        self.grid = np.load(path)
        self.grid_h, self.grid_w = self.grid.shape

        if not np.any(self.grid == 1):
            raise ValueError("Grid contains no FREE cells")
        if not np.any(self.grid == 0):
            raise ValueError("Grid contains no OBSTACLE cells")

    def _load_homography(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path} (run calibrator first)"
            )

        self.H = np.load(path)

        # Precompute inverse for convenience
        self.Hinv = np.linalg.inv(self.H)


    # Pixel / ROI / Grid transforms (UNCHANGED)
    def frame_px_to_roi_px(self, px, py):
        x, y, _, _ = self.roi
        return px - x, py - y

    def roi_px_to_cell(self, rx, ry):
        cx = int(rx // self.cell_size)
        cy = int(ry // self.cell_size)
        return cx, cy

    def frame_px_to_cell(self, px, py):
        rx, ry = self.frame_px_to_roi_px(px, py)
        return self.roi_px_to_cell(rx, ry)

    def cell_to_roi_px(self, cx, cy, center=True):
        if center:
            px = cx * self.cell_size + self.cell_size / 2
            py = cy * self.cell_size + self.cell_size / 2
        else:
            px = cx * self.cell_size
            py = cy * self.cell_size
        return int(px), int(py)

    # Grid semantics (AUTHORITATIVE)

    def cell_in_bounds(self, cx, cy):
        return 0 <= cx < self.grid_w and 0 <= cy < self.grid_h

    def cell_is_free(self, cx, cy):
        if not self.cell_in_bounds(cx, cy):
            return False
        return self.grid[cy, cx] == 1

    def cell_is_obstacle(self, cx, cy):
        if not self.cell_in_bounds(cx, cy):
            return True
        return self.grid[cy, cx] == 0


    # ===================  NEW: MM SPACE  =======================

    def pixel_to_mm(self, px, py):
        """
        Frame pixel -> world mm
        """
        p = np.array([px, py, 1.0], dtype=float)
        w = self.H @ p
        w /= w[2]
        return float(w[0]), float(w[1])

    def roi_px_to_mm(self, rx, ry):
        """
        ROI pixel -> world mm
        """
        x0, y0, _, _ = self.roi
        return self.pixel_to_mm(rx + x0, ry + y0)

    def cell_center_mm(self, cx, cy):
        """
        Grid cell center -> world mm
        """
        rx, ry = self.cell_to_roi_px(cx, cy, center=True)
        return self.roi_px_to_mm(rx, ry)


    # Derived physical properties
    
    @property
    def cell_size_mm(self):
        """
        Physical size (mm) of ONE grid cell (average of x/y).
        """
        rx0, ry0 = self.cell_to_roi_px(0, 0, center=True)
        rx1, ry1 = self.cell_to_roi_px(1, 0, center=True)

        x0, y0 = self.roi_px_to_mm(rx0, ry0)
        x1, y1 = self.roi_px_to_mm(rx1, ry1)

        return float(np.hypot(x1 - x0, y1 - y0))

    @property
    def grid_extent_mm(self):
        """
        Physical width & height of the ROI in mm.
        """
        rx0, ry0 = 0, 0
        rx1 = self.grid_w * self.cell_size
        ry1 = self.grid_h * self.cell_size

        x0, y0 = self.roi_px_to_mm(rx0, ry0)
        x1, y1 = self.roi_px_to_mm(rx1, ry1)

        return abs(x1 - x0), abs(y1 - y0)
