import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def visualize_directions(directions):
    """
    Scatter the unit vectors in 3D to visualize sampling.
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(directions[:, 0], directions[:, 1], directions[:, 2], s=5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def visualize_hoa_basis(lookup, order=1, scale_amplitude=True):
    """
    Visualize the spherical harmonic basis functions on a unit sphere.
    
    Args:
        lookup: dict returned by generate_hoa_lookup()
        order: int, Ambisonic order to visualize (1 for FOA, etc.)
        scale_amplitude: bool, whether to scale points by SH amplitude
    """
    dirs = lookup["directions"]
    Y = lookup["Y"]
    n_channels = (order + 1) ** 2

    n_cols = min(4, n_channels)
    n_rows = int(np.ceil(n_channels / n_cols))
    fig = plt.figure(figsize=(3*n_cols, 3*n_rows))

    for i in range(n_channels):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
        vals = Y[:, i]

        # Safe scaling: avoid divide-by-zero for constant channels (like ACN_0)
        if scale_amplitude:
            if vals.max() == vals.min():
                scale = np.ones_like(vals)  # constant channel
            else:
                scale = 0.5 + 0.5 * (vals - vals.min()) / (vals.max() - vals.min())
            pts = dirs * scale[:, None]
        else:
            pts = dirs  # unit sphere

        # Plot points colored by SH value
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=vals, cmap='coolwarm', s=10)
        ax.set_title(f"ACN {i}", fontsize=10)
        ax.set_axis_off()
        ax.view_init(elev=0, azim=0)

    plt.tight_layout()
    plt.show()



def visualize_room_interactive(metadata, save_path=None, arrow_scale=0.1):
    """
    Visualize a 3D room interactively using metadata.
    - Shows microphone, stationary sources with orientation, and moving trajectories.
    - Different colors for directivity patterns.
    - Arrows indicate source orientation (opaque if facing mic, translucent otherwise).
    - Moving trajectories with per-trajectory orientation arrows.
    
    Args:
        metadata: dict containing room, mic, and sources info
        save_path: str or Path to save figure (optional)
        arrow_scale: float, scaling factor for orientation arrows
    """
    room_dim = np.array(metadata["room"]["dimensions"])
    mic_pos = np.array(metadata["mic"]["mic_position"])
    
    # --- Extract stationary sources ---
    static_positions = np.array(metadata["sources"]["stationary"]["stationary_source_positions"])
    directivities = metadata["sources"]["stationary"]["stationary_source_directivity"]
    
    # --- Extract moving sources ---
    moving_trajs = []
    moving_directivities = []
    if "moving" in metadata["sources"]:
        moving_trajs = [np.array(t["trajectory_points"]) for t in metadata["sources"]["moving"]["trajectories"]]
        moving_directivities = [t["moving_source_directivity"] for t in metadata["sources"]["moving"]["trajectories"]]

    fig = go.Figure()

    # --- Microphone ---
    fig.add_trace(go.Scatter3d(
        x=[mic_pos[0]], y=[mic_pos[1]], z=[mic_pos[2]],
        mode='markers',
        marker=dict(size=6, color='red'),
        name='Microphone'
    ))

    # --- Stationary sources ---
    if static_positions.size > 0:
        fig.add_trace(go.Scatter3d(
            x=static_positions[:,0], y=static_positions[:,1], z=static_positions[:,2],
            mode='markers',
            marker=dict(size=4, color='gray', opacity=0.6),
            name='Stationary Sources (omni pattern)'
        ))

    # --- Color map for directivity patterns ---
    pattern_colors = {
        "cardioid": "blue",
        "subcardioid": "green",
        "hypercardioid": "purple"
    }

    # Dummy traces for legend
    for pattern, color in pattern_colors.items():
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=4, color=color),
            name=f"{pattern} pattern (opaque arrows face mic, translucent otherwise)"
        ))

    # --- Orientation arrows for stationary sources ---
    for src, d in zip(static_positions, directivities):
        pattern = d.get("pattern", "omni")
        color = pattern_colors.get(pattern, "black")
        az, el = d.get("orientation_az_deg"), d.get("orientation_el_deg")
        face_mic = d.get("face_mic", False)

        if az is not None and el is not None:
            az_rad, el_rad = np.radians(az), np.radians(el)
            dx = arrow_scale * np.cos(el_rad) * np.cos(az_rad)
            dy = arrow_scale * np.cos(el_rad) * np.sin(az_rad)
            dz = arrow_scale * np.sin(el_rad)
            opacity = 1.0 if face_mic else 0.3

            fig.add_trace(go.Cone(
                x=[src[0]], y=[src[1]], z=[src[2]],
                u=[dx], v=[dy], w=[dz],
                colorscale=[[0, color], [1, color]],
                sizemode="absolute",
                showscale=False,
                anchor="tail",
                hoverinfo="skip",
                opacity=opacity
            ))

    # --- Moving source trajectories and orientation ---
    colors = ['blue','green','orange','purple','cyan','magenta','yellow','brown','pink','lime']
    for i, traj in enumerate(moving_trajs):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter3d(
            x=traj[:,0], y=traj[:,1], z=traj[:,2],
            mode='lines+markers',
            line=dict(color=color, width=4),
            marker=dict(size=3),
            name=f'Moving Trajectory {i}'
        ))

        # Visualize orientation if available
        if i < len(moving_directivities) and moving_directivities[i]:
            dir_meta = moving_directivities[i][0]  # assume one orientation per trajectory
            pattern = dir_meta.get("pattern", "None")
            color_dir = pattern_colors.get(pattern, "black")
            az, el = dir_meta.get("orientation_az_deg"), dir_meta.get("orientation_el_deg")
            face_mic = dir_meta.get("face_mic", False)

            if pattern != "None" and az is not None and el is not None:
                az_rad, el_rad = np.radians(az), np.radians(el)
                dx = arrow_scale * np.cos(el_rad) * np.cos(az_rad)
                dy = arrow_scale * np.cos(el_rad) * np.sin(az_rad)
                dz = arrow_scale * np.sin(el_rad)
                opacity = 1.0 if face_mic else 0.3

                # Arrow at first trajectory point
                src = traj[0]
                fig.add_trace(go.Cone(
                    x=[src[0]], y=[src[1]], z=[src[2]],
                    u=[dx], v=[dy], w=[dz],
                    colorscale=[[0, color_dir], [1, color_dir]],
                    sizemode="absolute",
                    showscale=False,
                    anchor="tail",
                    hoverinfo="skip",
                    opacity=opacity
                ))

    # --- Layout ---
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, room_dim[0]], title='X (m)'),
            yaxis=dict(range=[0, room_dim[1]], title='Y (m)'),
            zaxis=dict(range=[0, room_dim[2]], title='Z (m)'),
        ),
        title="Interactive Room Visualization with Source Orientation and Trajectories",
        legend=dict(itemsizing='constant')
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved interactive 3D room visualization to {save_path}")

    fig.show()
