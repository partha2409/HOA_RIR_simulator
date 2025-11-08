# mic_sim.py

import numpy as np
import pyroomacoustics as pra
from hoa_directivity import HOADirectivity

def place_hoa_receiver(room, hoa_lookup, order=3, margin=0.5):
    """
    Place an HOA receiver randomly inside a room using measured directivity.

    Args:
        room (pyroomacoustics.ShoeBox): The room object
        hoa_lookup (dict): Dictionary with 'directions' (Nx3) and 'Y' (NxC) for HOA channels
        order (int): Ambisonic order (1=FOA, 3=3rd-order, etc.)
        margin (float): Minimum distance from walls (meters)

    Returns:
        receiver_pos: np.ndarray of shape (3,)
        metadata: dict with mic type, position, HOA order, and number of channels
    """

    C = ( order + 1 ) ** 2  # number of HOA channels

    # --- Random position inside room ---
    room_dim = np.asarray(room.shoebox_dim)

    x = np.random.uniform(margin, room_dim[0] - margin)
    y = np.random.uniform(margin, room_dim[1] - margin)
    z = np.random.uniform(1.0, room_dim[2] - margin)
    receiver_pos = np.array([x, y, z])
    receiver_pos = np.round(receiver_pos, 1)

    # create virtual mic array: same position for all channels
    R = np.tile(receiver_pos.reshape(3, 1), (1, C))

    # --- Create the HOADirectivity object ---
    directivities = [HOADirectivity( directions=hoa_lookup['directions'], weights=hoa_lookup['Y'], channel_idx=c) for c in range(C)] # <--- each channel gets its own pattern

    # --- Create measured receiver in Pyroomacoustics ---
    mic_array = pra.MicrophoneArray(R=R, directivity=directivities, fs=room.fs)
    room.add_microphone_array(mic_array)

    # --- Metadata ---
    metadata = {
        "mic_type": f"{order}-order HOA",
        "mic_position": receiver_pos.tolist(),
        "hoa_order": order,
        "num_channels": C
    }

    return room, metadata
