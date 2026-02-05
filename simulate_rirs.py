import os

# set for speedup in multiprocessing, I don't fully understand the internals myself :)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pyroomacoustics as pra
import json
from mic_sim import place_hoa_receiver
from src_sim import sample_sources_in_room
from room_sim import create_random_shoebox_room  
from ambisonics_encoding_matrix import generate_hoa_lookup
from materials_database import materials_dict
from tqdm import tqdm
from visualizations import visualize_room_interactive

import time
import concurrent.futures 


def simulate_single_room(min_dim, max_dim, hoa_lookup, hoa_order=3, n_stationary_src_per_room=64, n_moving_src_per_room=3, fs=16000,
                         early_fraction=0.7, max_order=30, min_order=10, rir_dir="rirs", room_idx=0, visualize=False, apply_directivity=False):
    """
    Simulate a single room, place HOA mic, sample sources, compute RIRs, and save RIR + metadata.
    Args:
        min_dim (list/tuple): [length, width, height] minimum
        max_dim (list/tuple): [length, width, height] maximum
        materials_dict (dict): {material_name: absorption_coeff}
        hoa_lookup (np.ndarray): HOA encoding lookup table
        hoa_order (int): Ambisonic order (1 = FOA, 3 = 3rd order, etc.)
        n_stationary_src_per_room (int): Number of stationary sources per room
        n_moving_src_per_room (int): Number of moving sources per room
        fs (int): Sampling rate
        early_fraction (float): Fraction of RT60 to consider as early reflections
        max_order (int): Max reflection order
        min_order (int): Minimum reflection order for ISM to ensure some reverberation even in very dry rooms
        rir_dir (str): Directory to save RIRs
        room_idx (int): Room index for saving
        visualize (bool): Whether to visualize the room interactively
        apply_directivity (bool): Whether to apply directivity to sources
    """
    #  Ensure each process has a unique seed
    np.random.seed(int(time.time() * 1000) % 2**32 + room_idx)

    # start timing
    start_time = time.time()

    metadata = {}
    # --- Create random room ---
    room, room_metadata = create_random_shoebox_room(fs, np.array(min_dim), np.array(max_dim), materials_dict, room_idx, early_fraction=early_fraction, max_order=max_order, min_order=min_order)
    metadata["room"] = room_metadata

    # --- Place HOA mic ---
    room, mic_metadata = place_hoa_receiver(room=room, hoa_lookup=hoa_lookup, order=hoa_order)
    metadata["mic"] = mic_metadata
    
    # --- place sources ---
    room, src_metadata = sample_sources_in_room(room=room, num_stationary_sources=n_stationary_src_per_room, num_moving_sources=n_moving_src_per_room, apply_directivity=apply_directivity)
    metadata["sources"] = src_metadata

    # --- Compute RIRs for all sources and mic---
    room.compute_rir()

    # --- Pack RIRs into (n_src, n_mics, max_len) array ---
    M = int(room.mic_array.M) if hasattr(room, "mic_array") else 1
    n_src = len(room.sources)
    
    max_len = 0
    for m in range(M):
        for s in range(n_src):
            max_len = max(max_len, len(room.rir[m][s]))
            
    rirs_np = np.zeros((n_src, M, max_len), dtype=float)
    
    for m in range(M):
        for s in range(n_src):
            ir = room.rir[m][s]
            rirs_np[s, m, :len(ir)] = ir

    # --- Save RIRs and metadata ---
    room_save_path = os.path.join(rir_dir, f"room_{room_idx}")
    os.makedirs(room_save_path, exist_ok=True)

    np.save(os.path.join(room_save_path, "rirs.npy"), rirs_np)

    with open(os.path.join(room_save_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    
    # --- Visualize and save plot ---
    if visualize:
        visualize_room_interactive(metadata, save_path=os.path.join(room_save_path, "room_interactive.html"))
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    return elapsed_time
    

def simulate_rooms(n_rooms, min_dim, max_dim, sphere_points_path, hoa_order=3, n_stationary_src_per_room=64, n_moving_src_per_room=3, fs=16000, 
                   early_fraction=0.7, max_order=30, min_order=10, rir_dir="rirs", apply_directivity=False, visualize=False):

    """
    Simulate multiple rooms, place HOA mic, sample sources, compute RIRs, and save RIR + metadata.

    Args:
        n_rooms (int): Number of rooms to simulate
        min_dim (list/tuple): [length, width, height] minimum
        max_dim (list/tuple): [length, width, height] maximum
        materials_dict (dict): {material_name: absorption_coeff}
        sphere_points_path (str): Path to .dat file with Nx3 Cartesian unit vectors
        hoa_order (int): Ambisonic order (1 = FOA, 3 = 3rd order, etc.)
        n_stationary_src_per_room (int): Number of stationary sources per room
        n_moving_src_per_room (int): Number of moving sources per room
        fs (int): Sampling rate
        early_fraction (float): Fraction of RT60 to consider as early reflections
        max_order (int): Maximum reflection order for ISM in case RT60 estimation gives very high order
        min_order (int): Minimum reflection order for ISM to ensure some reverberation even in very dry rooms
        rir_dir (str): Directory to save RIRs
        apply_directivity (bool): Whether to apply directivity to sources
        visualize (bool): Whether to visualize the room and sources
    """
    # Create output directory
    rir_dir = f"{rir_dir}_hoa_order_{hoa_order}" 
    os.makedirs(rir_dir, exist_ok=True)

    # --- Step 1: Generate HOA lookup ---
    print(f"Generating HOA lookup for order {hoa_order} ...")
    hoa_lookup = generate_hoa_lookup(order=hoa_order, sphere_points_path=sphere_points_path)
    print(f"Completed HOA lookup generation")
    
    # --- Step 2: Simulate rooms in parallel ---
    start_time_overall = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(simulate_single_room, min_dim, max_dim, hoa_lookup, hoa_order, n_stationary_src_per_room, n_moving_src_per_room, fs,
                                   early_fraction, max_order, min_order, rir_dir,  room_idx, apply_directivity=apply_directivity, visualize=visualize) for room_idx in range(n_rooms)]

        for room_idx, f in enumerate(concurrent.futures.as_completed(results)):
            elapsed_time = f.result()
            print(f"Finished room {room_idx+1}/{n_rooms} in {elapsed_time:.2f} seconds")
    
    end_time_overall = time.time()
    total_elapsed_time = end_time_overall - start_time_overall
    print(f"Simulated {n_rooms} rooms in {total_elapsed_time:.2f} seconds")
    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simulate multiple rooms with HOA RIRs."
    )

    parser.add_argument("--n_rooms", type=int, default=24)
    parser.add_argument("--min_dim", type=float, nargs=3, default=[4, 3, 3])
    parser.add_argument("--max_dim", type=float, nargs=3, default=[10, 8, 5])
    parser.add_argument("--sphere_points_path", type=str, default="N050_M1296_Octa.dat")
    parser.add_argument("--hoa_order", type=int, default=1)
    parser.add_argument("--n_stationary_src_per_room", type=int, default=64)
    parser.add_argument("--n_moving_src_per_room", type=int, default=6)
    parser.add_argument("--fs", type=int, default=24000)
    parser.add_argument("--early_fraction", type=float, default=0.7, help="Fraction of RT60 to consider as early reflections")
    parser.add_argument("--max_order", type=int, default=30, help="Maximum reflection order for ISM in case RT60 estimation gives very high order")
    parser.add_argument("--min_order", type=float, default=10, help="Minimum reflection order for ISM to ensure some reverberation even in very dry rooms")
    parser.add_argument("--rir_dir", type=str, default="/media/partha/LaCie/simulated_rirs_test")
    parser.add_argument("--apply_directivity", action="store_true", help="Whether to apply directivity to sources")
    parser.add_argument("--visualize", action="store_true", help="Whether to visualize the room and sources")

    args = parser.parse_args()

    # --- Run simulation ---
    simulate_rooms(
        n_rooms=args.n_rooms,
        min_dim=args.min_dim,
        max_dim=args.max_dim,
        sphere_points_path=args.sphere_points_path,
        hoa_order=args.hoa_order,
        n_stationary_src_per_room=args.n_stationary_src_per_room,
        n_moving_src_per_room=args.n_moving_src_per_room,
        fs=args.fs,
        early_fraction=args.early_fraction,
        max_order=args.max_order,
        min_order=args.min_order,
        rir_dir=args.rir_dir,
        apply_directivity=args.apply_directivity,
        visualize=args.visualize
    )
