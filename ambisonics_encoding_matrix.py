import numpy as np
import spaudiopy as spa  
from visualizations import visualize_directions, visualize_hoa_basis


def convert_n3d_to_sn3d(Y, order):
    """Convert N3D-normalized SH matrix to SN3D."""
    Y_sn3d = Y.copy()
    idx = 0
    for n in range(order + 1):
        scale = 1 / np.sqrt(2 * n + 1)
        for m in range(-n, n + 1):
            Y_sn3d[:, idx] *= scale
            idx += 1
    return Y_sn3d


def generate_hoa_lookup(order=3, sphere_points_path=None, visualize=False):
    """
    Generate Ambisonic spherical harmonic lookup table (AMBIX / ACN format).

    Args:
        order (int): Ambisonic order (1 = FOA, 3 = 3rd order, etc.)
        sphere_points_path (str): Path to .dat file with Nx3 Cartesian unit vectors.
        visualize (bool): Whether to scatter-plot the sampling directions.

    Returns:
        dict with:
            - directions: (N, 3) array of unit vectors
            - Y: (N, (order+1)^2) real spherical harmonic values (AMBIX / ACN ordering)
            - channels: list of channel labels (e.g. "ACN_0", "ACN_1", …)
    """
    # Load directions
    dirs = np.loadtxt(sphere_points_path, skiprows=2)

    # Assert all vectors are unit length
    norms = np.linalg.norm(dirs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6), "Some direction vectors are not unit length!"
    # dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True) 

    if visualize:
        visualize_directions(dirs)

    # Convert to spherical angles expected by spaudiopy:
    az = np.arctan2(dirs[:,1], dirs[:,0])
    zen = np.arccos(np.clip(dirs[:,2], -1.0, 1.0))

    # --- Compute N3D spherical harmonics (real) ---
    Y_n3d = spa.sph.sh_matrix(order, az, zen, sh_type='real')  # spa library uses N3D by default

    # --- Convert to SN3D (AMBIX normalization) ---
    Y_sn3d = convert_n3d_to_sn3d(Y_n3d, order)
    
    # --- Channel labels ---
    channels = [f"ACN_{i}" for i in range((order + 1) ** 2)]

    lookup = {"directions": dirs.tolist(), "Y": Y_sn3d.tolist(), "channels": channels}

    if visualize:
        visualize_hoa_basis({"directions": np.array(lookup["directions"]),"Y": np.array(lookup["Y"])}, order=order)

    return lookup


if __name__ == "__main__":
    sampled_points_path = "N050_M1296_Octa.dat"  # Path to your sphere points file
    lookup = generate_hoa_lookup(order=1, sphere_points_path=sampled_points_path, visualize=True)