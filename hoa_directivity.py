import numpy as np
from pyroomacoustics.directivities import Directivity

class HOADirectivity(Directivity):
    def __init__(self, directions, weights, channel_idx=None):
        """
        directions: (N, 3) array of unit vectors
        weights: (N, C) array, each column = HOA channel directivity
        channel_idx: int, which HOA channel this instance represents
        """
        self.directions = np.asarray(directions)
        self.weights = np.asarray(weights)
        self.channel_idx = channel_idx if channel_idx is not None else 0

    def get_response(self, azimuth, colatitude=None, magnitude=False, frequency=None, degrees=True):
        """
        Returns response for given angles (azimuth, colatitude) as required by PyroomAcoustics.
        """
        if colatitude is None:
            colatitude = np.zeros_like(azimuth)

        if degrees:
            azimuth = np.deg2rad(azimuth)
            colatitude = np.deg2rad(colatitude)

        # Convert angles to unit direction vectors
        vecs = np.stack([
            np.cos(azimuth) * np.sin(colatitude),
            np.sin(azimuth) * np.sin(colatitude),
            np.cos(colatitude)
        ], axis=-1)

        # Compute dot product with all lookup directions
        dots = vecs @ self.directions.T
        idx = np.argmax(dots, axis=-1)

        # Get corresponding weight for this HOA channel
        resp = self.weights[idx, self.channel_idx]

        if magnitude:
            resp = np.abs(resp)

        return np.atleast_1d(resp)

    @property
    def is_impulse_response(self):
        """False because this is a coefficient-based directivity, not an IR."""
        return False

    @property
    def filter_len_ir(self):
        """Length of impulse responses. Return 1 because no IR."""
        return 1
