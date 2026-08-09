"""
pi_optimized.py

EVERY Pi-2-specific performance optimization lives here, separated
from the code that decides WHAT the system does (that's main.py) so
neither file has to think about the other's concerns. If you're
trying to understand the pipeline, you shouldn't need to read this
file. If you're trying to make it faster, this is the only file that
should need touching.

Nothing in here changes correctness -- FastBearingEstimator was
verified to match dsp.py's reference implementation to within 1e-14
degrees (floating-point noise) before being used anywhere. The other
functions here are system-level (scheduling, CPU affinity, garbage
collection) and fail SAFELY: if a Pi-only feature isn't available
(e.g. testing on a Mac), they print a note and continue rather than
crashing.
"""

import os
import gc
import numpy as np

from dsp import find_shift


class FastBearingEstimator:
    """
    Produces IDENTICAL results to dsp.estimate_bearing_least_squares()
    -- this is a speed optimization, not a different algorithm.

    THE INSIGHT: that function rebuilds a 6x2 matrix from the mic
    geometry and re-solves a linear system from scratch on every
    single call, even though the mic geometry never changes between
    calls. Only the 6 measured delays change each time. So the matrix
    (and its pseudo-inverse, the expensive part to compute) is built
    ONCE here, and each live estimate becomes a single small matrix
    multiply instead of a fresh least-squares solve.
    """

    def __init__(self, mic_positions, speed_of_sound=343.0):
        n_mics = len(mic_positions)
        self.pairs = [(i, j) for i in range(n_mics) for j in range(i + 1, n_mics)]

        A_rows = [-(mic_positions[i] - mic_positions[j]) / speed_of_sound
                  for i, j in self.pairs]
        self.A = np.array(A_rows)
        self.A_pinv = np.linalg.pinv(self.A)   # the expensive part, done once

    def estimate(self, channels, fs, interp=16):
        y = np.array([
            find_shift(channels[i], channels[j], interp=interp) / fs
            for i, j in self.pairs
        ])
        cos_est, sin_est = self.A_pinv @ y
        return np.degrees(np.arctan2(sin_est, cos_est))


def apply_realtime_priority(priority=10):
    """
    Asks the OS to schedule this process with less pre-emption than
    normal, so the timing-critical audio capture loop is disturbed as
    little as possible by other processes. Best-effort: needs root on
    the Pi, and simply doesn't exist as a concept on some platforms --
    either way, the script keeps running with default scheduling.
    """
    if not hasattr(os, "sched_setscheduler"):
        print("Real-time scheduling not available on this platform -- continuing anyway.")
        return
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
        print(f"Real-time scheduling enabled (priority {priority}).")
    except PermissionError:
        print("Could not set real-time priority (try running with sudo). Continuing anyway.")


def pin_to_core(core_id):
    """
    Dedicates one of the Pi 2's 4 CPU cores to this process, so the
    audio-capture timing loop isn't sharing a core with, and getting
    interrupted by, whatever else the OS is doing.
    """
    if not hasattr(os, "sched_setaffinity"):
        print("CPU pinning not available on this platform -- continuing anyway.")
        return
    try:
        os.sched_setaffinity(0, {core_id})
        print(f"Pinned to CPU core {core_id}.")
    except OSError:
        print(f"Could not pin to core {core_id} (out of range?) -- continuing anyway.")


class PauseGCDuringCapture:
    """
    Python's automatic garbage collector can pause execution at
    unpredictable moments -- exactly the kind of jitter a real-time
    audio capture loop can't afford. This disables automatic
    collection for the duration of one capture, then re-enables it
    and runs one manual collection afterward, so garbage still gets
    cleaned up -- just at a moment that doesn't matter.

    Usage:
        with PauseGCDuringCapture():
            channels = capture.capture()
    """

    def __enter__(self):
        gc.disable()
        return self

    def __exit__(self, *args):
        gc.enable()
        gc.collect()
