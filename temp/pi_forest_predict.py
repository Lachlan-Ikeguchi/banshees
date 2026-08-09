"""
pi_forest_predict.py

Classifier inference with ZERO scikit-learn dependency -- only numpy
and scipy, both cleanly installable on the Pi via piwheels. Loads the
plain tree arrays exported by m4_train_classifier.py and re-walks
them; verified earlier to match sklearn's own predictions to
floating-point exactness (0 label mismatches, 0.00e+00 probability
difference on held-out data).

TWO BEHAVIOURS WORTH KNOWING ABOUT:

  1. LAZY LOADING: the model file is opened on the FIRST classify()
     call, not at import time -- so if forest_export.pkl is missing,
     you get one clear, actionable error message instead of a
     confusing crash the moment anything imports this module.

  2. AUTOMATIC RESAMPLING: capture on the Pi free-runs at whatever
     rate the hardware achieves (maybe ~8kHz), but the classifier was
     trained on audio at config.CANONICAL_FS (16kHz). classify()
     resamples incoming audio to the canonical rate before extracting
     features, so training and inference always see the same kind of
     input regardless of what rate the capture actually ran at.
"""

import pickle
import numpy as np
from scipy.signal import resample_poly
from fractions import Fraction

from mfcc import extract_features
import config as cfg

_forest = None   # loaded lazily on first use


def _load():
    global _forest
    if _forest is None:
        try:
            with open(cfg.MODEL_PATH, "rb") as f:
                _forest = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Classifier model '{cfg.MODEL_PATH}' not found.\n"
                f"On the Mac: run m4_record_audio.py then m4_train_classifier.py,\n"
                f"then copy the produced forest_export.pkl next to main.py on the Pi."
            )
    return _forest


def _predict_one_tree(tree, x):
    node = 0
    while tree["feature"][node] != -2:      # -2 marks a leaf in sklearn's layout
        if x[tree["feature"][node]] <= tree["threshold"][node]:
            node = tree["left"][node]
        else:
            node = tree["right"][node]
    return tree["value"][node]


def _resample_to_canonical(audio, fs):
    """Bring audio at any measured rate to the canonical training rate."""
    if abs(fs - cfg.CANONICAL_FS) < 1.0:
        return audio
    ratio = Fraction(cfg.CANONICAL_FS / fs).limit_denominator(1000)
    return resample_poly(audio, ratio.numerator, ratio.denominator)


def classify(audio, fs):
    """
    audio: 1-D float array (any sample rate); fs: that audio's actual
    sample rate in Hz. Returns (label, {class_name: probability}).
    """
    forest = _load()
    classes = forest["classes"]

    audio = _resample_to_canonical(np.asarray(audio, dtype=np.float64), fs)
    x = extract_features(audio, fs=cfg.CANONICAL_FS)

    votes = np.zeros(len(classes))
    for tree in forest["trees"]:
        leaf_counts = _predict_one_tree(tree, x)
        total = leaf_counts.sum()
        if total > 0:
            votes += leaf_counts / total
    probs = votes / len(forest["trees"])

    prob_dict = {str(label): float(p) for label, p in zip(classes, probs)}
    best = str(classes[int(np.argmax(probs))])
    return best, prob_dict
