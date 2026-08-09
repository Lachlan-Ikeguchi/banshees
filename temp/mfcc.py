"""
mfcc.py

A from-scratch MFCC feature extractor using only numpy and scipy --
no librosa. This file is used UNCHANGED on both the Mac (training)
and the Pi (inference), which matters: if training and inference
computed features even slightly differently, the model would be
learning one thing and predicting on another. Using the exact same
code for both removes that risk entirely, rather than trying to
carefully match two different implementations' parameters.

It's also lighter to deploy: librosa pulls in numba/llvmlite, which
are genuinely painful to install on a 32-bit ARM board like the Pi 2
(often no prebuilt wheel, and building from source can take hours or
fail outright). numpy and scipy both have solid ARM support via
piwheels.org (which Raspberry Pi OS's pip is usually already
configured to use).

THE PIPELINE, IN PLAIN TERMS:
  1. Chop the clip into short overlapping frames (sound changes too
     fast to analyse as one lump, but each small frame is roughly
     "steady" for the length of the frame).
  2. FFT each frame -> how much energy is at each frequency, right now.
  3. Squash those frequencies onto a "mel" scale, which matches how
     human hearing perceives pitch (more resolution at low
     frequencies, less at high) -- the same idea used for the
     log-mel spectrograms in the original build guide.
  4. Take the log (loudness is perceived logarithmically too).
  5. Compress that down further with a DCT, which concentrates the
     useful information into a handful of numbers instead of dozens.
"""

import numpy as np
from scipy.fftpack import dct

FS = 16000
N_FFT = 512          # samples per frame (32 ms at 16 kHz)
HOP_LENGTH = 256      # samples between frame starts (16 ms -- 50% overlap)
N_MELS = 26
N_MFCC = 13


def _hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700.0)


def _mel_to_hz(m):
    return 700 * (10 ** (m / 2595.0) - 1)


def _mel_filterbank(n_filters=N_MELS, n_fft=N_FFT, fs=FS, fmin=0, fmax=None):
    """
    Builds the triangular mel filters once. Each filter is a triangle
    that's zero everywhere except a narrow frequency band, peaking at
    1 in the middle -- overlapping filters that get WIDER at higher
    frequencies, matching the mel scale.
    """
    fmax = fmax or fs / 2
    mel_min, mel_max = _hz_to_mel(fmin), _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / fs).astype(int)

    filters = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(1, n_filters + 1):
        left, center, right = bin_points[i - 1], bin_points[i], bin_points[i + 1]
        for j in range(left, center):
            filters[i - 1, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            filters[i - 1, j] = (right - j) / max(right - center, 1)
    return filters


_FILTERBANK = _mel_filterbank()   # built once at import time, reused for every clip


def extract_mfcc_frames(audio, fs=FS, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mfcc=N_MFCC):
    """
    Returns MFCCs for every frame: shape (n_mfcc, n_frames), matching
    the convention the rest of this project's code expects.
    """
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    window = np.hamming(n_fft)

    mfccs = np.zeros((n_frames, n_mfcc))
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + n_fft] * window

        spectrum_power = np.abs(np.fft.rfft(frame, n=n_fft)) ** 2
        mel_energy = _FILTERBANK @ spectrum_power
        log_mel = np.log(mel_energy + 1e-10)

        mfccs[i] = dct(log_mel, type=2, norm="ortho")[:n_mfcc]

    return mfccs.T   # (n_mfcc, n_frames)


def extract_features(audio, fs=FS):
    """
    The actual per-clip feature vector used everywhere in this
    project: mean and spread of each MFCC coefficient across the
    whole clip -> 2 * N_MFCC numbers (26, with the current settings).
    """
    mfccs = extract_mfcc_frames(audio, fs=fs)
    return np.concatenate([mfccs.mean(axis=1), mfccs.std(axis=1)])
