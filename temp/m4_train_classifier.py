"""
m4_train_classifier.py

Trains a classifier to tell human sounds from machinery and
background, using whatever clips are sitting in data/human,
data/machinery, data/background (produced by m4_record_audio.py).

Feature extraction is handled by mfcc.py (a from-scratch numpy/scipy
implementation, no librosa) -- see that file for why. The Pi-side
inference script uses the SAME mfcc.py, unchanged, so training and
inference are guaranteed to agree on what a feature vector means.

WHY RANDOM FOREST FIRST:
  It trains in seconds, needs very little data, and gives a
  trustworthy baseline before reaching for anything fancier.

SETUP:
  pip3 install scikit-learn scipy numpy

USAGE:
  python3 m4_train_classifier.py
"""

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from scipy.io import wavfile

from mfcc import extract_features

DATA_DIR = "data"
FS = 16000
CLASSES = ["human", "machinery", "background"]


def load_dataset(data_dir=DATA_DIR):
    X, y_labels = [], []
    for label in CLASSES:
        folder = os.path.join(data_dir, label)
        if not os.path.isdir(folder):
            print(f"WARNING: no folder for '{label}' -- skipping. "
                  f"Did you run m4_record_audio.py?")
            continue
        files = [f for f in os.listdir(folder) if f.endswith(".wav")]
        for fname in files:
            sr, audio = wavfile.read(os.path.join(folder, fname))
            audio = audio.astype(np.float32)
            if audio.max() > 1.5:          # int16 wav -> normalise to -1..1
                audio = audio / 32768.0
            X.append(extract_features(audio, fs=FS))
            y_labels.append(label)
        print(f"  {label}: {len(files)} clips")

    return np.array(X), np.array(y_labels)


def train_and_evaluate(X, y, save_model=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n--- Results on held-out test clips ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels_present = sorted(set(y) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels_present)
    print("Confusion matrix (rows = true class, columns = predicted class):")
    print("             " + "  ".join(f"{l:>10}" for l in labels_present))
    for label, row in zip(labels_present, cm):
        print(f"{label:>12} " + "  ".join(f"{v:>10d}" for v in row))

    # The number that actually matters for this project: how often does
    # the model call MACHINERY a HUMAN sound? That false alarm is exactly
    # what would make the servo turn toward the noise it should ignore.
    if "machinery" in labels_present and "human" in labels_present:
        m_idx = labels_present.index("machinery")
        h_idx = labels_present.index("human")
        machinery_total = cm[m_idx].sum()
        machinery_called_human = cm[m_idx][h_idx]
        if machinery_total > 0:
            rate = 100 * machinery_called_human / machinery_total
            print(f"\nMachinery misclassified as human: {machinery_called_human}/{machinery_total} "
                  f"({rate:.0f}%)  <-- this is the number that actually matters for the demo")

    if save_model:
        import joblib
        joblib.dump(clf, "classifier.joblib")
        print("\nSaved trained model to classifier.joblib")

    return clf


def export_for_pi(clf, path="forest_export.pkl"):
    """
    Extracts JUST the raw decision-tree arithmetic from the trained
    forest -- no scikit-learn objects, no scipy needed to READ this
    file back. Only plain numpy arrays of numbers.

    WHY THIS MATTERS FOR THE PI:
      A trained RandomForestClassifier is, underneath the scikit-learn
      wrapper, just a set of trees, and every tree is just a sequence
      of "is feature[i] <= threshold? go left or right" comparisons,
      ending at a leaf that holds a class vote. That's it -- no
      matrix math, no BLAS, nothing scikit-learn's C extensions are
      actually needed for AT INFERENCE TIME. They're needed for
      TRAINING (building the trees from data), which only ever
      happens on the Mac.

      So instead of installing scikit-learn on the Pi (real risk of
      slow/failed ARM builds), we pull the raw tree data out here,
      save it as a plain dictionary of numpy arrays, and write a
      ~15-line pure-numpy tree-walker to run on the Pi instead. It
      makes EXACTLY the same predictions as clf.predict() would --
      this isn't an approximation, it's the identical decision logic,
      just with the scikit-learn wrapper peeled off.
    """
    import pickle

    trees = []
    for est in clf.estimators_:
        t = est.tree_
        trees.append({
            "feature": t.feature.copy(),      # which feature this node compares (-2 = leaf)
            "threshold": t.threshold.copy(),  # the comparison threshold
            "left": t.children_left.copy(),   # node index to go to if <= threshold
            "right": t.children_right.copy(), # node index to go to if > threshold
            "value": t.value[:, 0, :].copy(),  # class vote counts at every node
        })

    with open(path, "wb") as f:
        pickle.dump({"classes": list(clf.classes_), "trees": trees}, f)

    n_nodes = sum(len(t["feature"]) for t in trees)
    print(f"Exported {len(trees)} trees, {n_nodes} total nodes, to {path}")
    print(f"File size: {os.path.getsize(path) / 1024:.1f} KB")


def main():
    print("Loading dataset...")
    X, y = load_dataset()

    if len(X) == 0:
        print("No data found. Run m4_record_audio.py first.")
        return

    print(f"\nTotal clips: {len(X)}")
    clf = train_and_evaluate(X, y, save_model=True)
    export_for_pi(clf)


if __name__ == "__main__":
    main()
