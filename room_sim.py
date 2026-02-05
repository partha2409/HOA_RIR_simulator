import numpy as np
import pyroomacoustics as pra
import math


def getRT60formula(room_xyz, absCoeffs, c=343.0):
    Lx, Ly, Lz = room_xyz
    V = Lx * Ly * Lz
    Sx = Ly * Lz
    Sy = Lx * Lz
    Sz = Lx * Ly
    A = (
        Sx * (absCoeffs[0] + absCoeffs[1]) +
        Sy * (absCoeffs[2] + absCoeffs[3]) +
        Sz * (absCoeffs[4] + absCoeffs[5])
    )
    const = math.log(10**6) / c
    rt60 = 4 * const * V / A
    return rt60


def getISMorderFromTime(room_xyz, maxTime, c=343.0):
    d_max = maxTime * c
    return np.ceil(d_max / room_xyz).astype(int)


def create_random_shoebox_room(fs, min_dim, max_dim, materials_dict, room_id=None, early_fraction=0.7, max_order=30, min_order=10, c=343.0):
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

    # --- Extract absorption coefficients ---
    absCoeffs = np.array([
    np.mean(materials_for_room["east"].energy_absorption["coeffs"]),
    np.mean(materials_for_room["west"].energy_absorption["coeffs"]),
    np.mean(materials_for_room["north"].energy_absorption["coeffs"]),
    np.mean(materials_for_room["south"].energy_absorption["coeffs"]),
    np.mean(materials_for_room["ceiling"].energy_absorption["coeffs"]),
    np.mean(materials_for_room["floor"].energy_absorption["coeffs"])
    ])

    # --- Compute RT60 analytically ---
    rt60 = getRT60formula(dims, absCoeffs)

    # --- Convert RT60 to ISM order ---
    orders_xyz = getISMorderFromTime(dims, early_fraction * rt60)
    ism_order = int(np.max(orders_xyz))  

    # --- Apply optional ism_order cap ---
    ism_order_capped = min(ism_order, max_order)
    ism_order_capped = max(ism_order_capped, min_order)

    # --- Create the shoebox room ---
    room = pra.ShoeBox(dims, fs=fs, materials=materials_for_room, max_order=ism_order_capped, use_rand_ism=True, ray_tracing=False, air_absorption=True)
    
    # --- Room metadata ---
    room_metadata = {
        "room_id": room_id,
        "dimensions": dims.tolist(),
        "materials": chosen_materials,
        "absCoeffs": absCoeffs.tolist(),
        "fs": fs,
        "air_absorption": True,
        "estimated_rt60": float(rt60),
        "ism_order": ism_order,
        "ism_order_capped": ism_order_capped,
        "early_fraction": early_fraction,
    }
    
    return room, room_metadata