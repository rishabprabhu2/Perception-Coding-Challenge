import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from scipy.ndimage import gaussian_filter1d

# ---- Constants ----
PATCH_DIM = 11
HALF_PATCH_DIM = PATCH_DIM // 2
BBOX_CSVS = ["bboxes_light.csv", "bbox_light.csv"]
XYZ_PATH = "xyz"

def load_bboxes():
    """Load bounding box CSV and print columns."""
    bbox_file = None
    for fname in BBOX_CSVS:
        if os.path.exists(fname):
            bbox_file = fname
            break
    if bbox_file is None:
        print(f"Could not locate any bounding box CSV in {BBOX_CSVS}")
        exit()
    bboxes = pd.read_csv(bbox_file)
    print("CSV fields detected:", bboxes.columns.tolist())
    return bboxes

def process_frame(row, bboxes):
    """Extract median 3D location for a single frame using its bounding box."""
    # Determine frame index
    if 'frame_id' in bboxes.columns:
        frame_idx = int(row["frame_id"])
    elif 'frame' in bboxes.columns:
        frame_idx = int(row["frame"])
    else:
        frame_idx = row.name + 1
    # Bounding box columns
    bbox_keys = ['x_min', 'y_min', 'x_max', 'y_max']
    alt_bbox_keys = ['x1', 'y1', 'x2', 'y2']
    if all(col in bboxes.columns for col in bbox_keys):
        x1, y1, x2, y2 = row[bbox_keys]
    elif all(col in bboxes.columns for col in alt_bbox_keys):
        x1, y1, x2, y2 = row[alt_bbox_keys]
    else:
        return None
    if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
        return None
    u = int((x1 + x2) / 2)
    v = int((y1 + y2) / 2)
    # Find xyz file
    xyz_cands = [
        os.path.join(XYZ_PATH, f"frame_{frame_idx:04d}.npz"),
        os.path.join(XYZ_PATH, f"depth{frame_idx:06d}.npz"),
        os.path.join(XYZ_PATH, f"frame_{frame_idx:03d}.npz"),
    ]
    xyz_file = next((f for f in xyz_cands if os.path.exists(f)), None)
    if xyz_file is None:
        return None
    try:
        data = np.load(xyz_file)
        if "points" in data.keys():
            xyz = data["points"]
        elif "xyz" in data.keys():
            xyz = data["xyz"]
        else:
            keys = list(data.keys())
            if len(keys) > 0:
                xyz = data[keys[0]]
            else:
                return None
    except Exception as e:
        print(f"Could not load {xyz_file}: {e}")
        return None
    print(f"Frame {frame_idx}: XYZ shape = {xyz.shape}")
    # Handle shape
    if len(xyz.shape) == 3 and xyz.shape[2] == 3:
        h, w = xyz.shape[:2]
    elif len(xyz.shape) == 3 and xyz.shape[2] == 4:
        xyz = xyz[:, :, :3]
        h, w = xyz.shape[:2]
        print("  Using first 3 channels from 4-channel array")
    elif len(xyz.shape) == 2 and xyz.shape[1] == 3:
        if len(xyz) == 0:
            return None
        mask = ~np.isnan(xyz).any(axis=1)
        mask &= ~np.isinf(xyz).any(axis=1)
        mask &= np.linalg.norm(xyz, axis=1) > 0.1
        mask &= np.linalg.norm(xyz, axis=1) < 100
        valid = xyz[mask]
        if len(valid) < 3:
            return None
        X, Y, Z = np.median(valid, axis=0)
        return [frame_idx, X, Y, Z]
    elif len(xyz.shape) == 2:
        total = xyz.shape[0] * xyz.shape[1]
        if total % 3 == 0:
            n_pts = total // 3
            options = []
            for hh in range(100, 2000):
                if n_pts % hh == 0:
                    ww = n_pts // hh
                    if 100 <= ww <= 2000:
                        options.append((hh, ww))
            if options:
                hh, ww = min(options, key=lambda x: abs(x[0] - x[1]))
                try:
                    xyz = xyz.reshape(hh, ww, 3)
                    h, w = hh, ww
                    print(f"  Reshaped to ({hh}, {ww}, 3)")
                except:
                    return None
            else:
                return None
        else:
            return None
    else:
        print(f"  Unexpected XYZ shape: {xyz.shape}")
        return None
    # Patch extraction
    vmin, vmax = max(0, v-HALF_PATCH_DIM), min(h, v+HALF_PATCH_DIM+1)
    umin, umax = max(0, u-HALF_PATCH_DIM), min(w, u+HALF_PATCH_DIM+1)
    if vmax <= vmin or umax <= umin:
        return None
    try:
        patch = xyz[vmin:vmax, umin:umax, :].reshape(-1, 3)
    except Exception as e:
        print(f"  Patch extraction error: {e}")
        return None
    mask = ~np.isnan(patch).any(axis=1)
    mask &= ~np.isinf(patch).any(axis=1)
    mask &= np.linalg.norm(patch, axis=1) > 0.1
    mask &= np.linalg.norm(patch, axis=1) < 100
    valid = patch[mask]
    if len(valid) < 3:
        return None
    X, Y, Z = np.median(valid, axis=0)
    return [frame_idx, X, Y, Z]

def filter_points(data_arr):
    """Remove outlier points based on percentiles in each dimension."""
    def pct_mask(arr, lo=10, hi=90):
        a = np.percentile(arr, lo)
        b = np.percentile(arr, hi)
        return (arr >= a) & (arr <= b)
    zmsk = pct_mask(data_arr[:, 3])
    xmsk = pct_mask(data_arr[:, 1])
    ymsk = pct_mask(data_arr[:, 2])
    mask = zmsk & xmsk & ymsk
    return data_arr[mask]

def smooth_and_scale(filtered_arr):
    """Smooth trajectory, then convert to BEV and rescale."""
    forward_vals = filtered_arr[:,1]
    lateral_vals = filtered_arr[:,2]
    up_vals = filtered_arr[:,3]
    if len(forward_vals) > 3:
        sigma = max(0.8, len(forward_vals) / 20)
        forward_vals = gaussian_filter1d(forward_vals, sigma=sigma)
        lateral_vals = gaussian_filter1d(lateral_vals, sigma=sigma)
        up_vals = gaussian_filter1d(up_vals, sigma=sigma)
    print(f"After smoothing: {len(forward_vals)} points")
    print(f"Forward range: {forward_vals.min():.2f} to {forward_vals.max():.2f}")
    print(f"Lateral range: {lateral_vals.min():.2f} to {lateral_vals.max():.2f}")
    print(f"Upward range: {up_vals.min():.2f} to {up_vals.max():.2f}")
    # Ego world: flip sign for forward
    ego_x = -forward_vals
    ego_y = lateral_vals
    print("World coords post-smoothing:")
    print(f"Ego X: {ego_x.min():.2f} to {ego_x.max():.2f}")
    print(f"Ego Y: {ego_y.min():.2f} to {ego_y.max():.2f}")
    # BEV conversion
    lat_positions = -ego_y[::-1]  # BEV X (sideways)
    long_positions = ego_x[::-1]  # BEV Y (forward)
    # Center to (0,0) at final point
    lat_positions -= lat_positions[-1]
    long_positions -= long_positions[-1]
    # Rescale longitudinal
    initial_long = long_positions[0]
    desired_start = 14.0
    if abs(initial_long) > 0.1:
        yfac = desired_start / initial_long
        long_positions *= yfac
    # Optionally scale lateral
    if len(lat_positions) > 1:
        lat_rng = np.ptp(lat_positions)
        if lat_rng > 0.1:
            latfac = 2.0 / lat_rng
            lat_positions *= latfac
    print("BEV values before any final scaling:")
    print(f"Lateral: {lat_positions.min():.2f} to {lat_positions.max():.2f}")
    print(f"Longitudinal: {long_positions.min():.2f} to {long_positions.max():.2f}")
    # Final scale if needed
    max_rng = max(np.ptp(lat_positions), np.ptp(long_positions))
    if max_rng > 50:
        sc = 20 / max_rng
        lat_positions *= sc
        long_positions *= sc
        print(f"Applied final scale: {sc:.3f}")
    print("Final BEV trajectory:")
    print(f"Lateral: {lat_positions.min():.2f} to {lat_positions.max():.2f}")
    print(f"Longitudinal: {long_positions.min():.2f} to {long_positions.max():.2f}")
    return lat_positions, long_positions

def plot_static(lat_positions, long_positions):
    fig, ax = plt.subplots(figsize=(10, 8))
    # Ego trajectory line
    ax.plot(lat_positions, long_positions, color='teal', linewidth=3,
            linestyle='-.', alpha=0.9, label='Ego Trajectory')
    # Start and End points with distinct markers
    ax.scatter(lat_positions[0], long_positions[0], c='blue', s=180,
               marker='D', edgecolors='black', linewidths=2, label='Start')
    ax.scatter(lat_positions[-1], long_positions[-1], c='green', s=180,
               marker='s', edgecolors='black', linewidths=2, label='End')
    # Traffic light origin
    ax.scatter(0, 0, c='red', s=220, marker='^', edgecolors='black',
               linewidths=2, label='Traffic Light (Origin)', zorder=5)

    ax.set_xlabel("Lateral X (m)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Longitudinal Y (m)", fontsize=14, fontweight='bold')
    ax.set_title("Ego Trajectory (BEV) - Static", fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle=':')
    ax.axis('equal')
    ax.legend(fontsize=12, loc='best')
    plt.tight_layout()
    plt.savefig("trajectory.png", dpi=300, bbox_inches='tight')
    print("Static BEV trajectory saved as trajectory.png")
    plt.show()


def plot_animation(lat_positions, long_positions):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlabel("Lateral X (m)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Longitudinal Y (m)", fontsize=14, fontweight='bold')
    ax.set_title("Ego Trajectory (BEV) - Animation", fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle=':')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-2, 15)
    ax.axis('equal')

    # Traffic light origin
    ax.scatter(0, 0, c='red', s=220, marker='^', edgecolors='black',
               linewidths=2, label='Traffic Light (Origin)', zorder=5)
    # Ego trajectory line (animated)
    line, = ax.plot([], [], '-.', color='teal', linewidth=3, alpha=0.9, label='Ego Trajectory')
    # Trail and markers
    trail = ax.scatter([], [], c='blue', s=40, alpha=0.6, zorder=3)
    ax.scatter(lat_positions[0], long_positions[0], c='blue', s=180, marker='D',
               edgecolors='black', linewidths=2, label='Start', zorder=5)
    end_marker = ax.scatter([], [], c='green', s=180, marker='s',
                            edgecolors='black', linewidths=2, label='End', zorder=5)
    ax.legend(fontsize=12, loc='best')

    def animate(frame):
        n = min(frame + 1, len(lat_positions))
        line.set_data(lat_positions[:n], long_positions[:n])
        if n > 1:
            trail.set_offsets(np.column_stack((lat_positions[:n], long_positions[:n])))
        else:
            trail.set_offsets(np.empty((0, 2)))
        if n == len(lat_positions):
            end_marker.set_offsets([[lat_positions[-1], long_positions[-1]]])
        else:
            end_marker.set_offsets(np.empty((0, 2)))
        return line, trail, end_marker

    anim = animation.FuncAnimation(fig, animate, frames=len(lat_positions)+10,
                                   interval=100, blit=False, repeat=True)
    from matplotlib.animation import FFMpegWriter
    writer = FFMpegWriter(fps=10, bitrate=1800)
    print("Writing animated trajectory to MP4...")
    anim.save("trajectory.mp4", writer=writer)
    print("Animation saved as trajectory.mp4")
    plt.tight_layout()
    plt.show()



# ---- Main Execution ----
if __name__ == "__main__":
    bboxes = load_bboxes()
    data_rows = []
    for idx, row in bboxes.iterrows():
        pt = process_frame(row, bboxes)
        if pt is not None:
            data_rows.append(pt)
    if len(data_rows) == 0:
        print("No valid 3D points extracted.")
        exit()
    data_arr = np.array(data_rows)
    filtered_arr = filter_points(data_arr)
    lat_positions, long_positions = smooth_and_scale(filtered_arr)
    plot_static(lat_positions, long_positions)
    plot_animation(lat_positions, long_positions)
