"""
m0_mock_mic.py

SIMULATED M0 -- now extended to a 4-mic square array.

The original version of this file only handled two mics in a line,
using a shortcut formula (sin of the angle). That shortcut only works
because a 2-mic line is a special, simple case. Four mics arranged in
a square is not a line, so we need the general version of the same
idea -- and it turns out to be barely more code, not less intuitive.

EVERYTHING BELOW THE "ORIGINAL 2-MIC VERSION" SECTION IS UNCHANGED.
M1 (m1_delay_test.py) still imports from there and still works exactly
as before. The new square-array functions are additions, not
replacements.
"""

import numpy as np

FS = 16000               # samples/second we're PRETENDING our fake ADC runs at
MIC_SPACING = 0.15       # metres -- the side length of the square board
SPEED_OF_SOUND = 343.0   # metres/second


# =====================================================================
# SHARED HELPERS (used by both the old 2-mic version and the new array)
# =====================================================================

def make_clap(duration=0.05, fs=FS):
    """
    A short burst of noise with a sharp attack and quick decay --
    stands in for a real clap or tap. We only need realistic TIMING
    structure (a sudden onset), since timing is all the algorithm
    cares about.
    """
    n = int(duration * fs)
    envelope = np.exp(-np.linspace(0, 12, n))
    return np.random.randn(n) * envelope


def _place(event, total_samples, start_index):
    """Drops a short event into an otherwise-silent buffer at start_index."""
    out = np.zeros(total_samples)
    end_index = min(start_index + len(event), total_samples)
    if end_index > start_index:
        out[start_index:end_index] = event[: end_index - start_index]
    return out


# =====================================================================
# ORIGINAL 2-MIC VERSION -- unchanged, still used by M1
# =====================================================================

def max_possible_delay_samples(spacing=MIC_SPACING, fs=FS):
    """Physics ceiling for two mics `spacing` metres apart."""
    seconds = spacing / SPEED_OF_SOUND
    return seconds * fs


def simulate_mic_pair(true_delay_samples, duration=0.3, fs=FS,
                       noise_level=0.05, seed=None):
    """Two fake recordings of the same clap, a known number of samples apart."""
    if seed is not None:
        np.random.seed(seed)

    total_samples = int(duration * fs)
    clap = make_clap(fs=fs)
    start = total_samples // 3

    ch0 = _place(clap, total_samples, start)
    ch1 = _place(clap, total_samples, start + int(true_delay_samples))

    ch0 = ch0 + noise_level * np.random.randn(total_samples)
    ch1 = ch1 + noise_level * np.random.randn(total_samples)

    return ch0, ch1


def record_burst(fake_delay=None):
    """2-mic drop-in stand-in for a real hardware read."""
    if fake_delay is None:
        fake_delay = np.random.randint(
            -int(max_possible_delay_samples()), int(max_possible_delay_samples()) + 1
        )
    return simulate_mic_pair(fake_delay, fs=FS)


# =====================================================================
# NEW: 4-mic square array
# =====================================================================
#
# Layout, viewed from above. Forward (0 degrees) points along +x.
# Angles increase counter-clockwise, same as ordinary maths convention.
#
#      MIC1 (-h, +h) ●───────────────● MIC2 (+h, +h)
#                     │               │
#                     │   ⊕ centre    │      +x = 0°  (forward)
#                     │  (0, 0)       │      +y = 90°
#                     │               │
#      MIC0 (-h, -h) ●───────────────● MIC3 (+h, -h)
#
#           h = MIC_SPACING / 2 = 0.075 m
#
# THE IDEA FOR ONE MIC:
#   Pick a direction the source is in -- a unit arrow u = (cos θ, sin θ)
#   pointing from the array's centre toward the source.
#
#   For any microphone, "how far along that arrow does this mic sit?"
#   is exactly what a dot product measures. A mic sitting further
#   along the arrow is physically closer to the source, so it hears
#   the sound SOONER. Since our delay number means "how much LATER
#   than the centre", a mic that's closer must come out NEGATIVE --
#   hence the minus sign below.
#
#   This is the exact same "closer = earlier" reasoning M2 used for
#   two mics in a line -- just written so it works for a mic
#   ANYWHERE on the board, not only ones sitting exactly left/right.

# Geometry now lives in config.py (the single source of truth for the
# whole project); re-exported here so every existing import keeps working.
from config import MIC_POSITIONS, MIC_SPACING_M as MIC_SPACING, SPEED_OF_SOUND as _CFG_C
MIC_NAMES = ["MIC0", "MIC1", "MIC2", "MIC3"]


def mic_delay_seconds(mic_pos, source_bearing_deg):
    """
    How much LATER (or earlier, if negative) ONE mic hears the source
    compared to the array's centre point. Shown separately from the
    array version below purely so the idea is easy to follow one mic
    at a time before we do all four at once.
    """
    theta_rad = np.radians(source_bearing_deg)
    u = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    return -np.dot(mic_pos, u) / SPEED_OF_SOUND


def mic_array_delays_seconds(source_bearing_deg, mic_positions=MIC_POSITIONS):
    """
    Same formula as mic_delay_seconds, applied to all four mics at
    once. `mic_positions` is a 4x2 table of (x, y) positions; the dot
    product with the direction vector `u` gives all four answers in
    a single line.
    """
    theta_rad = np.radians(source_bearing_deg)
    u = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    return -(mic_positions @ u) / SPEED_OF_SOUND


def max_single_mic_delay_samples(mic_positions=MIC_POSITIONS, fs=FS):
    """
    Largest possible delay any ONE mic can have relative to the
    array's centre (as opposed to the pairwise max between two
    opposite mics, which is roughly double this).
    """
    max_radius = max(np.linalg.norm(p) for p in mic_positions)
    return (max_radius / SPEED_OF_SOUND) * fs


def apply_fractional_delay(x, delay_samples):
    """
    Delays a whole signal by an EXACT amount, including fractions of
    a sample -- something simply moving array indices around can't do
    (you can only round to the nearest whole sample that way).

    This is the same trick used for real-hardware skew correction: a
    time delay is a straight-line phase ramp in the frequency domain,
    so we FFT, multiply by that ramp, and inverse-FFT back. The ramp
    works for ANY delay_samples value, fractional or not.
    """
    n = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n)
    phase_ramp = np.exp(-2j * np.pi * freqs * delay_samples)
    return np.fft.irfft(X * phase_ramp, n=n)


def simulate_mic_array(source_bearing_deg, mic_positions=MIC_POSITIONS,
                        duration=0.3, fs=FS, noise_level=0.05, seed=None):
    """
    Generates one fake recording per microphone, all of the SAME
    clap, each delayed by EXACTLY the amount that mic's real position
    implies for a source at source_bearing_deg -- fractional samples
    included, not rounded off.

    Returns: a list of 4 numpy arrays, in the same order as
    mic_positions (MIC0, MIC1, MIC2, MIC3 by default).
    """
    if seed is not None:
        np.random.seed(seed)

    delays_samples = mic_array_delays_seconds(source_bearing_deg, mic_positions) * fs

    total_samples = int(duration * fs)
    clap = make_clap(fs=fs)
    start = total_samples // 3
    base = _place(clap, total_samples, start)   # un-delayed reference placement

    channels = []
    for delay in delays_samples:
        ch = apply_fractional_delay(base, delay)
        ch = ch + noise_level * np.random.randn(total_samples)
        channels.append(ch)

    return channels


def record_burst_array(source_bearing_deg=None):
    """
    4-mic drop-in stand-in for a real hardware read. On the Pi, a
    function with this name will read all four ADC channels
    back-to-back. Here, it fakes a source at a given (or random)
    bearing and returns four channels.
    """
    if source_bearing_deg is None:
        source_bearing_deg = np.random.uniform(0, 360)
    return simulate_mic_array(source_bearing_deg)


if __name__ == "__main__":
    # -- original 2-mic sanity check, unchanged --
    max_d = max_possible_delay_samples()
    print(f"[2-mic check] max possible delay at {FS} Hz: +/-{max_d:.1f} samples")
    ch0, ch1 = simulate_mic_pair(true_delay_samples=5, seed=0)
    print(f"[2-mic check] ch0 peak {np.argmax(np.abs(ch0))}, "
          f"ch1 peak {np.argmax(np.abs(ch1))} (difference should be close to 5)\n")

    # -- new 4-mic square array sanity check --
    print("=" * 62)
    print("4-mic square array check")
    print("=" * 62)
    print("Mic positions (metres, array frame):")
    for name, pos in zip(MIC_NAMES, MIC_POSITIONS):
        print(f"  {name}: ({pos[0]:+.3f}, {pos[1]:+.3f})")

    for bearing in [0, 45, 90, 180, 270]:
        delays_samp = mic_array_delays_seconds(bearing) * FS
        earliest = MIC_NAMES[int(np.argmin(delays_samp))]
        latest = MIC_NAMES[int(np.argmax(delays_samp))]
        print(f"\nSource at {bearing:>3} degrees  (expect {earliest} earliest, {latest} latest):")
        for name, d in zip(MIC_NAMES, delays_samp):
            tag = "  <- earliest" if name == earliest else ("  <- latest" if name == latest else "")
            print(f"  {name}: {d:+6.2f} samples{tag}")

    print(f"\nMax possible single-mic delay (relative to array centre): "
          f"+/-{max_single_mic_delay_samples():.2f} samples")
