import numpy as np
import pyroomacoustics as pra

# -----------------------------
#  --- Source Directivity ---
# -----------------------------

def create_source_directivity_and_orientation(src_pos, mic_pos=None, pattern=None):
    """
    Create a random or mic-facing directivity pattern and orientation metadata.

    Args:
        src_pos: np.ndarray (3,) - source position
        mic_pos: np.ndarray (3,) - mic position
        pattern: one of pyroomacoustics.DirectivityPattern

    Returns:
        directivity: pyroomacoustics directivity object
        metadata: dict with az/el in degrees and mic-to-src angles
    """

    # --- Orientation of source ---
    # randomly decide whether to face mic
    face_mic = np.random.rand() < 0.5

    if face_mic and mic_pos is not None:
        vec = mic_pos - src_pos
        az_deg = np.degrees(np.arctan2(vec[1], vec[0]))
        el_deg = np.degrees(np.arctan2(vec[2], np.linalg.norm(vec[:2])))
    else:
        az_deg = np.random.uniform(0, 360)
        el_deg = np.random.uniform(-30, 30)

    # --- Mic-to-src angles (always computed if mic_pos provided) ---
    if mic_pos is not None:
        vec_mic = src_pos - mic_pos
        mic_to_src_az_deg = np.degrees(np.arctan2(vec_mic[1], vec_mic[0]))
        mic_to_src_el_deg = np.degrees(np.arctan2(vec_mic[2], np.linalg.norm(vec_mic[:2])))
    else:
        mic_to_src_az_deg = None
        mic_to_src_el_deg = None

    # --- Create directivity object ---
    if pattern == 'cardioid':
        directivity = pra.directivities.Cardioid(pra.DirectionVector(azimuth=az_deg, colatitude=90 - el_deg, degrees=True), gain=1.0)
    elif pattern == 'subcardioid':
        directivity = pra.directivities.SubCardioid(pra.DirectionVector(azimuth=az_deg, colatitude=90 - el_deg, degrees=True), gain=1.0)
    elif pattern == 'hypercardioid':
        directivity = pra.directivities.HyperCardioid(pra.DirectionVector(azimuth=az_deg, colatitude=90 - el_deg, degrees=True), gain=1.0)
    else:
        raise ValueError("pattern must be one of 'cardioid', 'subcardioid', or 'hypercardioid'")
    
    metadata = {
        "pattern": pattern,
        "orientation_az_deg": az_deg,
        "orientation_el_deg": el_deg,
        "face_mic": face_mic,
        "mic_to_src_az_deg": mic_to_src_az_deg,
        "mic_to_src_el_deg": mic_to_src_el_deg
    }

    return directivity, metadata


def create_source_directivity_and_orientation_for_moving_sources(traj_points, mic_pos=None, pattern="cardioid"):
    """
    Create a directivity pattern for a moving source oriented along its trajectory direction.

    Args:
        traj_points: np.ndarray (N, 3) - full trajectory points
        mic_pos: np.ndarray (3,) - microphone array mean position
        pattern: str - one of ['cardioid', 'subcardioid', 'hypercardioid']

    Returns:
        directivity: pyroomacoustics directivity object (fixed orientation)
        metadata: dict with pattern, orientation (az/el), and per-point mic-to-src angles
    """

    traj_points = np.asarray(traj_points)

    # --- Compute direction of motion (start → end vector) ---
    traj_vec = traj_points[-1] - traj_points[0]
    traj_vec = traj_vec / (np.linalg.norm(traj_vec) + 1e-8)

    # --- Orientation (source faces along movement) ---
    az_deg = np.degrees(np.arctan2(traj_vec[1], traj_vec[0]))
    el_deg = np.degrees(np.arctan2(traj_vec[2], np.linalg.norm(traj_vec[:2])))

    # --- Mic-to-source angles for each trajectory point ---
    mic_to_src_az_deg = []
    mic_to_src_el_deg = []
    if mic_pos is not None:
        for p in traj_points:
            vec_mic = p - mic_pos
            mic_to_src_az_deg.append(np.degrees(np.arctan2(vec_mic[1], vec_mic[0])))
            mic_to_src_el_deg.append(np.degrees(np.arctan2(vec_mic[2], np.linalg.norm(vec_mic[:2]))))
    else:
        mic_to_src_az_deg = [None] * len(traj_points)
        mic_to_src_el_deg = [None] * len(traj_points)

    # --- Create directivity object ---
    if pattern == 'cardioid':
        directivity = pra.directivities.Cardioid(
            pra.DirectionVector(azimuth=az_deg, colatitude=90 - el_deg, degrees=True), gain=1.0)
    elif pattern == 'subcardioid':
        directivity = pra.directivities.SubCardioid(
            pra.DirectionVector(azimuth=az_deg, colatitude=90 - el_deg, degrees=True), gain=1.0)
    elif pattern == 'hypercardioid':
        directivity = pra.directivities.HyperCardioid(
            pra.DirectionVector(azimuth=az_deg, colatitude=90 - el_deg, degrees=True), gain=1.0)
    else:
        raise ValueError("pattern must be one of 'cardioid', 'subcardioid', or 'hypercardioid'")

    metadata = {
        "pattern": pattern,
        "orientation_az_deg": float(az_deg),
        "orientation_el_deg": float(el_deg),
        "face_mic": False,  # since it's following movement, not mic-facing
        "mic_to_src_az_deg": mic_to_src_az_deg,
        "mic_to_src_el_deg": mic_to_src_el_deg
    }

    return directivity, metadata



# -----------------------------
#    --- Moving Source ---
# -----------------------------


def _generate_fallback_trajectory(room_dim, mic_pos, margin, desired_length, num_points=26, min_distance_to_mic=0.3):
    """
    Deterministic fallback trajectory with mic clearance.
    - Horizontal line along X if possible, else along Y.
    - Ensures minimum distance to mic if possible.
    """
    print("Using fallback trajectory generation.")

    # --- Try X-axis line first ---
    if (room_dim[0] - 2*margin) >= desired_length:
        y_fixed = margin + 0.5
        start_xy = np.array([margin, y_fixed])
        end_xy = np.array([margin + desired_length, y_fixed])

        # Check min distance to mic
        dist_to_mic = _line_mic_distance(start_xy, end_xy, mic_pos[:2])
        if dist_to_mic < min_distance_to_mic:
            # shift along Y away from mic
            if mic_pos[1] + min_distance_to_mic + 0.1 < room_dim[1]-margin:
                y_fixed = mic_pos[1] + min_distance_to_mic + 0.1
            else:
                y_fixed = max(margin, mic_pos[1] - min_distance_to_mic - 0.1)
            start_xy[1] = y_fixed
            end_xy[1] = y_fixed

    else:
        # --- Use Y-axis line ---
        x_fixed = margin + 0.5
        start_xy = np.array([x_fixed, margin])
        end_xy = np.array([x_fixed, min(margin + desired_length, room_dim[1]-margin)])

        # Check min distance to mic
        dist_to_mic = _line_mic_distance(start_xy, end_xy, mic_pos[:2])
        if dist_to_mic < min_distance_to_mic:
            # shift along X away from mic
            if mic_pos[0] + min_distance_to_mic + 0.1 < room_dim[0]-margin:
                x_fixed = mic_pos[0] + min_distance_to_mic + 0.1
            else:
                x_fixed = max(margin, mic_pos[0] - min_distance_to_mic - 0.1)
            start_xy[0] = x_fixed
            end_xy[0] = x_fixed

    # Compute points along line
    xy_points = np.linspace(start_xy, end_xy, num_points)

    return xy_points, np.linalg.norm(end_xy - start_xy), np.linalg.norm(end_xy - start_xy)/(num_points-1)
    

def add_jitter_to_trajectory(traj_points, xy_jitter=0.02, z_jitter=0.01):
    """
    Add small random jitter to a 3D trajectory.

    Args:
        traj_points: (num_points, 3) np.ndarray of trajectory points
        xy_jitter: max random displacement in XY plane (meters)
        z_jitter: max random displacement in Z (meters)
    
    Returns:
        jittered_traj: (num_points, 3) np.ndarray
    """
    jitter = np.zeros_like(traj_points)
    # XY jitter
    jitter[:, 0:2] = np.random.uniform(-xy_jitter, xy_jitter, size=(traj_points.shape[0], 2))
    # Z jitter
    jitter[:, 2] = np.random.uniform(-z_jitter, z_jitter, size=traj_points.shape[0])
    
    return traj_points + jitter

def _sample_valid_start_point(room_dim, margin):
    """Sample a valid XY start point inside the room margins."""
    x = np.random.uniform(margin, room_dim[0] - margin)
    y = np.random.uniform(margin, room_dim[1] - margin)
    return np.array([x, y])


def _generate_trajectory(room_dim, mic_pos, margin, desired_length, num_points=26, max_attempts=100, min_distance_to_mic=0.3):
    """
    Generate a single moving source trajectory in the XY plane with fixed num_points.
    - Ensures end point is inside room margins
    - Ensures min distance from mic
    """
    for _ in range(max_attempts):
        start_xy = _sample_valid_start_point(room_dim, margin)
        angle = np.random.uniform(0, 2*np.pi)
        vec = np.array([np.cos(angle), np.sin(angle)])
        end_xy = start_xy + vec * desired_length

        # Check if end point is inside room margins
        if not (margin <= end_xy[0] <= room_dim[0] - margin and margin <= end_xy[1] <= room_dim[1] - margin):
            continue

        # Check distance to mic
        dist_to_mic = _line_mic_distance(start_xy, end_xy, mic_pos[:2])
        if dist_to_mic < min_distance_to_mic:
            continue

        # Compute points along line
        xy_points = np.linspace(start_xy, end_xy, num_points)

        return xy_points, np.linalg.norm(end_xy - start_xy), np.linalg.norm(end_xy - start_xy)/(num_points-1)

    # If we reach here, use a fallback strategy    
    return _generate_fallback_trajectory(room_dim, mic_pos, margin, desired_length, num_points, min_distance_to_mic=min_distance_to_mic)


def _line_mic_distance(start_xy, end_xy, mic_xy):
    """Compute min distance from mic to line segment in XY plane."""
    v = end_xy - start_xy
    w = mic_xy - start_xy
    c1 = np.dot(w, v)
    if c1 <= 0:
        return np.linalg.norm(mic_xy - start_xy)
    c2 = np.dot(v, v)
    if c2 <= c1:
        return np.linalg.norm(mic_xy - end_xy)
    b = c1 / c2
    pb = start_xy + b * v
    return np.linalg.norm(mic_xy - pb)


def sample_moving_sources_in_room(room, num_moving_sources=6, margin=0.5, num_points=26, min_distance_to_mic=0.3, apply_directivity=False): # 26 comes from max 5m trajectory with 0.2m step.
    """
    Generate moving sources trajectories in the room.
    - Automatically selects trajectory length based on room size
    - First trajectory near floor for footsteps
    Args:
        room: pyroomacoustics.ShoeBox object
        num_moving_sources: int
        margin: float, distance from walls
        num_points: int, number of points per trajectory
        min_distance_to_mic: float, minimum distance from mic to trajectory line
        apply_directivity: bool, whether to apply directivity to sources
    """

    room_dim = np.array(room.shoebox_dim)
    mic_pos = np.mean(room.mic_array.R, axis=1)

    # Determine desired trajectory length based on room XY size
    effective_xy = room_dim[:2] - 2*margin
    max_diag = np.linalg.norm(effective_xy)
    if max_diag < 3.5:
        traj_len = 2.0
    elif max_diag < 5.0:
        traj_len = 3.0
    elif max_diag < 7.0:
        traj_len = 4.0
    else:
        traj_len = 5.0

    trajectories = []

    for i in range(num_moving_sources):
        xy_points, total_dist, step_size = _generate_trajectory(room_dim, mic_pos, margin, traj_len, num_points, min_distance_to_mic=min_distance_to_mic)

        # Assign height
        if i < 2:  # two trajectories near floor for footsteps
            z = np.random.uniform(0.05, 0.1)  # footsteps
            traj_type = "footsteps"
        else:
            z = np.random.uniform(0.4*room_dim[2], 0.9*room_dim[2])
            traj_type = "moving_source"

        # Build full 3D points
        traj_points = np.hstack([xy_points, np.full((num_points, 1), z)])
        
        # --- Add small jitter ---
        traj_points = add_jitter_to_trajectory(traj_points, xy_jitter=0.02, z_jitter=0.01)

        # Compute distances to microphone
        distances_to_mic = [float(np.linalg.norm(p - mic_pos)) for p in traj_points]
        avg_distance_to_mic = float(np.mean(distances_to_mic))

        # Apply directivity if needed
        src_directivities = []
        if apply_directivity:
            if i < 4: 
                pattern = "omni"
                directivity = None
                directivity_metadata = {"pattern": pattern, "orientation_az_deg": None, "orientation_el_deg": None, "face_mic": None, "mic_to_src_az_deg": None, "mic_to_src_el_deg": None}
            else:
                pattern = "cardioid"
                directivity, directivity_metadata = create_source_directivity_and_orientation_for_moving_sources(traj_points, mic_pos=mic_pos, pattern=pattern)
        else:
            pattern = "None"
            directivity = None
            directivity_metadata = {"pattern": pattern, "orientation_az_deg": None, "orientation_el_deg": None, "face_mic": None, "mic_to_src_az_deg": None, "mic_to_src_el_deg": None}
        
        src_directivities.append(directivity_metadata)

        # Add sources to room
        for p in traj_points:
            room.add_source(p.tolist(), directivity=directivity)

        # Store metadata
        trajectories.append({
            "id": i,
            "type": traj_type,
            "start": traj_points[0].tolist(),
            "end": traj_points[-1].tolist(),
            "num_points": num_points,
            "trajectory_points": traj_points.tolist(),
            "trajectory_length": float(total_dist),
            "point_spacing": float(step_size),
            "point_distances_to_mic": distances_to_mic,
            "avg_distance_to_mic": avg_distance_to_mic,
            "moving_source_directivity": src_directivities
        })

    metadata = {
        "num_moving_sources": num_moving_sources,
        "trajectories": trajectories,
    }

    return room, metadata


# -----------------------------
#    --- Static Source ---
# -----------------------------

def sample_stationary_sources_in_room(room, num_stationary_sources=64, margin=0.5, min_distance_to_mic=0.3, apply_directivity=False):
    """
    Add stationary sources uniformly inside a room while excluding a cube around the mic.
    
    Args:
        room: pyroomacoustics.ShoeBox object
        num_stationary_sources: int
        margin: float, distance from walls
        min_distance_to_mic: float, exclusion distance around mic
        apply_directivity: bool, whether to apply directivity to sources
    Returns:
        room: room object with sources added
        metadata: dict containing source positions
    """
    room_dim = np.array(room.shoebox_dim)
    mic_pos = np.mean(room.mic_array.R, axis=1)

    src_positions = []
    src_directivities = []
    src_distances = []

    for idx in range(num_stationary_sources):
        pos = np.zeros(3)
        for i in range(3):  # x, y, z
            lower1 = margin
            upper1 = mic_pos[i] - min_distance_to_mic
            lower2 = mic_pos[i] + min_distance_to_mic
            upper2 = room_dim[i] - margin

            # Determine which side to sample from
            if upper1 > lower1 and upper2 > lower2:
                # Compute available lengths on both sides
                len1 = upper1 - lower1
                len2 = upper2 - lower2
                
                # Probability proportional to distance from mic
                p_side2 = len2 / (len1 + len2)  # probability to pick side 2 (farther side)
                if np.random.rand() < p_side2:
                    pos[i] = np.random.uniform(lower2, upper2)
                else:
                    pos[i] = np.random.uniform(lower1, upper1)
            elif upper1 > lower1:
                pos[i] = np.random.uniform(lower1, upper1)
            elif upper2 > lower2:
                pos[i] = np.random.uniform(lower2, upper2)
            else:
                # room too small, place at margin
                pos[i] = margin

        src_positions.append(pos)
        src_distances.append(np.linalg.norm(pos - mic_pos))

        if apply_directivity:
            # --- Decide pattern based on index ---
            if idx < num_stationary_sources // 2:
                pattern = "omni"
                directivity = None
                directivity_metadata = {"pattern": pattern, "orientation_az_deg": None, "orientation_el_deg": None, "face_mic": None, "mic_to_src_az_deg": None, "mic_to_src_el_deg": None}
            elif idx < 3 * num_stationary_sources // 4:
                pattern = "cardioid"
                directivity, directivity_metadata = create_source_directivity_and_orientation(src_pos=pos, mic_pos=mic_pos, pattern=pattern)
            else:
                pattern = np.random.choice(["subcardioid", "hypercardioid"])
                directivity, directivity_metadata = create_source_directivity_and_orientation(src_pos=pos, mic_pos=mic_pos, pattern=pattern)

        else:
            pattern = "None"
            directivity = None
            directivity_metadata = {"pattern": pattern, "orientation_az_deg": None, "orientation_el_deg": None, "face_mic": None, "mic_to_src_az_deg": None, "mic_to_src_el_deg": None}
            
        src_directivities.append(directivity_metadata)
        room.add_source(pos.tolist(), directivity=directivity)

    src_positions = np.array(src_positions)

    metadata = {
        "num_stationary_sources": num_stationary_sources,
        "stationary_source_positions": src_positions.tolist(),
        "stationary_source_distances_to_mic": src_distances,
        "stationary_source_directivity": src_directivities
    }

    return room, metadata



def sample_sources_in_room(room, num_stationary_sources=64, num_moving_sources=3, margin=0.5, min_distance_to_mic=0.3, apply_directivity=False):

    sources_metadata = {}

    if num_stationary_sources > 0:
        room, stationary_metadata = sample_stationary_sources_in_room(room, num_stationary_sources, margin, min_distance_to_mic=min_distance_to_mic, apply_directivity=apply_directivity)
        sources_metadata["stationary"] = stationary_metadata
    if num_moving_sources > 0:
        room, moving_metadata = sample_moving_sources_in_room(room, num_moving_sources, margin, min_distance_to_mic=min_distance_to_mic, apply_directivity=apply_directivity)
        sources_metadata["moving"] = moving_metadata


    return room, sources_metadata
    


