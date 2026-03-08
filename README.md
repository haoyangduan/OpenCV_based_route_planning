# OpenCV-Based Route Planning and PIV Closed-Loop Control

This repository documents my technical contributions to a continuum
robot project for vascular intervention.

The open-source content here focuses on the **software pipeline I
developed**, including:

-   OpenCV-based environment perception
-   Grid-based path planning
-   Continuous reference path generation
-   Projection-based path tracking
-   PID closed-loop control

The repository showcases the algorithmic pipeline used for
**vision-guided navigation and actuation control** in a laboratory
robotic system.

------------------------------------------------------------------------

# System Pipeline

Camera Input\
↓\
Map Tuning (ROI, Binary Segmentation, Grid Map)\
↓\
Camera Calibration\
↓\
Path Planning (A\*)\
↓\
Reference Path Generation (Spline)\
↓\
Projection-Based Tracking\
↓\
PID Controller\
↓\
Hardware Actuation

------------------------------------------------------------------------

# Repository Structure

    opencv-route-planning/
    │
    ├── README.md
    ├── LICENSE
    ├── requirements.txt
    │
    ├── src/
    │   ├── mapping/
    │   │   └── map_config.py
    │   │
    │   ├── planning/
    │   │   ├── path_planner.py
    │   │   └── reference_path.py
    │   │
    │   ├── tracking/
    │   │   └── tracker.py
    │   │
    │   ├── control/
    │   │   └── pid_controller.py
    │   │
    │   └── visualization/
    │       └── draw_tracker_pid_viz.py
    │
    ├── scripts/
    │   ├── map_tuner.py
    │   ├── map_calibrator.py
    │   ├── path_planner.py
    │   ├── pid_visualize_tuner.py
    │   └── live_loop.py
    │
    ├── configs/
    │   ├── binary_params.json
    │   ├── grid_params.json
    │   └── calibration.json
    │
    └── data/
        ├── grid.npy
        ├── homography.npy
        └── planned_cells.json

------------------------------------------------------------------------

# Installation

Clone the repository:

``` bash
git clone https://github.com/haoyangduan/OpenCV_based_route_planning.git
cd opencv-route-planning
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# Usage

## 1. Map Tuning

Select ROI and tune binary/grid parameters.

``` bash
python scripts/map_tuner.py
```

This step generates:

    configs/binary_params.json
    configs/grid_params.json
    data/grid.npy

------------------------------------------------------------------------

## 2. Camera Calibration

Perform camera calibration using a chessboard pattern.

``` bash
python scripts/map_calibrator.py
```

This produces:

    configs/calibration.json
    data/homography.npy

------------------------------------------------------------------------

## 3. Path Planning

Run the path planner to generate a safe route on the grid map.

``` bash
python scripts/path_planner.py
```

This produces:

    data/planned_cells.json

------------------------------------------------------------------------

## 4. PID Controller Tuning

Visualize tracking errors and tune PID gains.

``` bash
python scripts/pid_visualize_tuner.py
```

------------------------------------------------------------------------

## 5. Closed-Loop Control

Run the real-time control loop.

``` bash
python scripts/live_loop.py
```

This script performs:

-   vision-based tip detection
-   path tracking
-   PID control
-   MCU communication

------------------------------------------------------------------------

# Key Components

## Path Planning

Grid-based A\* planning with obstacle clearance weighting to avoid wall
hugging.

Features include:

-   obstacle-aware cost function
-   distance-transform clearance field
-   optional Bezier smoothing for visualization

------------------------------------------------------------------------

## Reference Path

Discrete grid paths are converted into continuous curves using:

-   corner extraction
-   Catmull--Rom spline interpolation
-   arc-length parameterization

------------------------------------------------------------------------

## Tracking

A projection-based tracker computes:

-   arc-length progress `s`
-   lateral error `e_y`
-   heading error `e_psi`
-   lookahead point

------------------------------------------------------------------------

## Control

The controller follows a PID structure:

    u = kp * e_y + ki * ∫e_y dt + kd * de_y/dt + kψ * e_psi

Features include:

-   derivative filtering
-   integral windup protection
-   output saturation

------------------------------------------------------------------------

# Notes

This repository only contains the **software components developed and
integrated by the author**.

It does **not include confidential hardware designs, datasets, or
internal research code**.

------------------------------------------------------------------------

# License

This project is licensed under the MIT License.
