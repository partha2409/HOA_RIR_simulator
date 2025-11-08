import numpy as np
import pyroomacoustics as pra


def create_random_shoebox_room(fs, max_order, min_dim, max_dim, materials_dict, room_id=None):
    """
    Generate a random 3D shoebox room with per-wall random materials and RT60 estimation.
    """
    # Random integer room dimensions
    dims = np.round(min_dim + (max_dim - min_dim) * np.random.rand(3))
    
    # --- Randomly select materials for each surface ---
    east_mat = np.random.choice(materials_dict["wall"])
    west_mat = np.random.choice(materials_dict["wall"])
    north_mat = np.random.choice(materials_dict["wall"])
    south_mat = np.random.choice(materials_dict["wall"])
    floor_mat = np.random.choice(materials_dict["floor"])
    ceiling_mat = np.random.choice(materials_dict["ceiling"])

    chosen_materials = { "east": east_mat, "west": west_mat, "north": north_mat, "south": south_mat, "floor": floor_mat, "ceiling": ceiling_mat}

    
    # Create Pyroomacoustics Material objects for each wall
    materials_for_room = {
        "east": pra.Material(east_mat), "west": pra.Material(west_mat), "north": pra.Material(north_mat), "south": pra.Material(south_mat),
        "floor": pra.Material(floor_mat), "ceiling": pra.Material(ceiling_mat)
    }
    
    # --- Create the shoebox room ---
    room = pra.ShoeBox(dims, fs=fs, materials=materials_for_room, max_order=max_order, use_rand_ism=True, ray_tracing=False, air_absorption=True)
    
    # --- Room metadata ---
    room_metadata = {
        "room_id": room_id,
        "dimensions": dims.tolist(),
        "materials": chosen_materials,
        "fs": fs,
        "max_order": max_order,
        "air_absorption": True,
    }
    
    return room, room_metadata