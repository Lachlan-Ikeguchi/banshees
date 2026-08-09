"""
dsp.py

The one real signal-processing function in this whole project. It does
not know or care whether its input came from a real microphone or a
fake one -- that separation is exactly what lets us test it here, on
a Mac, with zero hardware, and then reuse it unchanged everywhere else.

This is the same "which mic heard it first" sliding-match idea from
earlier: compare timing only (not loudness), find where the two
recordings line up best, that offset is the answer.
"""

import numpy as np


def find_shift(sig, ref, return_curve=False, interp=1):
    """
    Slide `sig` past `ref` and find the offset (in samples) where they
    line up best.

      positive  -> sig arrived LATER than ref
      negative  -> sig arrived EARLIER than ref

    interp: how finely to sub-divide each whole sample when looking
    for the peak. interp=1 gives whole-sample answers only (fine for
    M1's original validation). Anything using the answer for a
    precise bearing estimate needs interp > 1 -- otherwise the
    estimate can only take a handful of discrete values, and a
    closed loop built on it will overshoot and oscillate between
    them instead of settling. See the M2 notes for what that looked
    like in practice.
    """
    sig = sig - np.mean(sig)   # remove each channel's constant background level
    ref = ref - np.mean(ref)

    n = len(sig) + len(ref)
    SIG = np.fft.rfft(sig, n=n)
    REF = np.fft.rfft(ref, n=n)

    R = SIG * np.conj(REF)
    R /= (np.abs(R) + 1e-9)     # match on TIMING, ignore loudness

    n_up = n * interp
    cc = np.fft.irfft(R, n=n_up)   # zero-padding in frequency = interpolating in time

    max_idx = int(np.argmax(cc))
    if max_idx > n_up // 2:
        max_idx -= n_up             # convert to a signed "earlier / later" number
    shift = max_idx / interp        # back to units of whole samples, now fractional

    if return_curve:
        return shift, cc
    return shift


def estimate_bearing_least_squares(channels, mic_positions, fs, speed_of_sound=343.0, interp=16):
    """
    Combines EVERY microphone pair's timing measurement into a single
    best-fit bearing, instead of trusting one pair or averaging just
    two of them.

    Each pair (i, j) gives one equation:

        tau_ij = -(dx_ij * cos(theta) + dy_ij * sin(theta)) / c

    where tau_ij is "how much later mic i is than mic j" (which is
    exactly what find_shift measures) and (dx_ij, dy_ij) is that
    pair's known physical separation. With 4 mics there are 6 such
    pairs -- 6 equations, only 2 unknowns (cos theta, sin theta).
    That's more equations than unknowns, so instead of solving
    exactly we find the (cos theta, sin theta) that best explains ALL
    SIX measurements at once, in the least-squared-error sense. This
    is the same idea as fitting a line through noisy scattered points
    -- more measurements, combined intelligently, beats trusting any
    single one.
    """
    n_mics = len(mic_positions)
    A_rows = []
    y_rows = []

    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            shift_samples = find_shift(channels[i], channels[j], interp=interp)
            tau_seconds = shift_samples / fs

            d = mic_positions[i] - mic_positions[j]
            A_rows.append(-d / speed_of_sound)
            y_rows.append(tau_seconds)

    A = np.array(A_rows)   # shape (6, 2)
    y = np.array(y_rows)   # shape (6,)

    solution, residuals, rank, _ = np.linalg.lstsq(A, y, rcond=None)
    cos_est, sin_est = solution
    theta_deg = np.degrees(np.arctan2(sin_est, cos_est))
    return theta_deg
