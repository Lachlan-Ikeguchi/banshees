"""
mic_selection.py

Given an estimated bearing, picks which single physical microphone is
positioned closest to that direction -- that mic's raw audio is what
gets handed to the classifier, since it's the cleanest single-channel
view of whatever's happening in that direction.
"""

import numpy as np


def nearest_mic_index(bearing_deg, mic_positions):
    """
    Each mic's own "preferred direction" is simply the angle from the
    array's centre out to that mic's position (MIC2, sitting at
    (+h, +h), naturally faces 45 degrees; MIC0 at (-h, -h) naturally
    faces 225 degrees; and so on). The mic whose preferred direction
    is angularly closest to the estimated bearing is the one
    physically nearest to facing the source.

    This effectively splits the circle into 4 wedges, one centred on
    each mic -- a simple, defensible choice, though a delay-and-sum
    beamform toward the exact bearing would give better signal-to-noise
    than any single mic alone (a documented upgrade path, not built
    here since a single fixed mic per direction was the explicit
    design choice for this version).
    """
    mic_angles_deg = np.degrees(np.arctan2(mic_positions[:, 1], mic_positions[:, 0]))
    angular_diff = np.abs(((bearing_deg - mic_angles_deg) + 180) % 360 - 180)
    return int(np.argmin(angular_diff))
