# Perception Coding Challenge

## Overview
This project extracts and visualizes the ego-vehicle trajectory from stereo camera data. The main output includes a 2D bird’s-eye view (BEV) of the vehicle’s path and an animation showing the trajectory with key markers.

## Method
1. **Bounding Box Loading:** Reads CSV files containing detected traffic light bounding boxes.
2. **3D Point Extraction:** Loads corresponding `.npz` files containing 3D points and calculates the median location within each bounding box.
3. **Filtering & Smoothing:** Removes outlier points and smooths the trajectory using a Gaussian filter.
4. **Coordinate Transformation:** Converts the trajectory to a BEV frame with the traffic light as the origin.
5. **Visualization:** Generates both a static plot (`trajectory.png`) and an animated video (`trajectory.mp4`) with start, end, ego path, and traffic light markers.

## Assumptions
- Bounding boxes correspond accurately to the traffic light in each frame.
- 3D points are valid and not corrupted; outliers are removed.
- The traffic light is fixed and serves as the origin in BEV coordinates.

## Results
- The final BEV trajectory clearly shows the ego-vehicle path relative to the traffic light.
- Start and end positions are marked, and the animation visualizes movement along the path.
- Trajectory data can be used for further analysis or comparison with ground truth paths.

## Files
- `simulate_trajectory_from_pixels.py` – main script for processing and visualization.
- `trajectory.png` – static BEV plot of the trajectory.
- `trajectory.mp4` – animated trajectory visualization.
- `.gitignore` – ignores large files like the `xyz` folder.
- `README.md` – this file.

## Notes
- The `xyz` folder contains 3D point cloud `.npz` files. This folder isn't pushed to the repository yet due to its large size, but will be added soon.
