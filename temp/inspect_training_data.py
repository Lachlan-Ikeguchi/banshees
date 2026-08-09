"""
inspect_training_data.py

DIAGNOSTIC TOOL. Plots a sample of your ACTUAL recorded clips
(data/human, data/machinery, data/background) so you can see, for
real, two things:

  1. Does the peak-energy windowing (mock_hardware.py's
     _load_real_clip) correctly find the actual event in each clip,
     or is it grabbing the wrong moment?
  2. Do your classes actually LOOK distinctive from each other --
     specifically, is "machinery" a brief burst (looks similar to a
     tap) or sustained noise across the whole clip (looks different)?
     See the project notes for why this specific distinction matters:
     tested directly, a tap vs. a brief machinery burst are barely
     separable in feature space, while a tap vs. sustained machinery
     noise is ~6x more separable.

Writes training_data_inspection.png -- no live display needed, same
reasoning as channel_monitor.py (works whether or not a monitor is
attached).

USAGE:
  python3 inspect_training_data.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile

import config as cfg

OUTPUT_PATH = "training_data_inspection.png"
N_EXAMPLES_PER_CLASS = 4
CLASSES = ["human", "machinery", "background"]


def find_active_window(audio, n):
    """Same logic as mock_hardware.py's _load_real_clip -- duplicated
    here (not imported) so this tool works standalone even before
    mock_hardware.py exists or without needing the servo/hardware
    machinery it pulls in."""
    if len(audio) <= n:
        return 0, len(audio)
    window = max(256, n // 8)
    energy = np.convolve(audio ** 2, np.ones(window) / window, mode="same")
    peak_idx = int(np.argmax(energy))
    start = max(0, min(peak_idx - n // 2, len(audio) - n))
    return start, start + n


def load_clip(path):
    fs, audio = wavfile.read(path)
    audio = audio.astype(np.float32)
    if audio.max() > 1.5:
        audio = audio / 32768.0
    return audio, fs


def main():
    fig, axes = plt.subplots(len(CLASSES), N_EXAMPLES_PER_CLASS,
                             figsize=(4 * N_EXAMPLES_PER_CLASS, 3 * len(CLASSES)),
                             squeeze=False)

    for row, cls in enumerate(CLASSES):
        folder = os.path.join("data", cls)
        if not os.path.isdir(folder):
            for col in range(N_EXAMPLES_PER_CLASS):
                axes[row][col].text(0.5, 0.5, f"no data/{cls}/", ha="center", va="center")
                axes[row][col].axis("off")
            continue

        files = sorted(f for f in os.listdir(folder) if f.endswith(".wav"))
        print(f"{cls}: {len(files)} clips found")

        for col in range(N_EXAMPLES_PER_CLASS):
            ax = axes[row][col]
            if col >= len(files):
                ax.axis("off")
                continue

            audio, fs = load_clip(os.path.join(folder, files[col]))
            start, end = find_active_window(audio, cfg.N_SAMPLES_PER_CLIP)

            ax.plot(audio, linewidth=0.5, color="#888888")
            ax.axvspan(start, end, color="orange", alpha=0.3,
                      label="window mock_hardware.py would extract")
            ax.set_title(f"{cls} #{col}\n{files[col]}", fontsize=9)
            ax.set_xticks([])
            if col == 0:
                ax.set_ylabel(cls, fontsize=11, fontweight="bold")

            windowed_rms = np.sqrt(np.mean(audio[start:end] ** 2))
            full_rms = np.sqrt(np.mean(audio ** 2))
            sustain_ratio = windowed_rms / (full_rms + 1e-9)
            # Bursts are EXPECTED and correct for "human" -- a tap should
            # look brief. The problem case is specifically "machinery"
            # clips that ALSO look brief (ratio near a tap's ~1.9, not
            # near sustained noise's ~1.0) -- that's what erases the
            # duration cue the classifier needs to tell them apart.
            flag = ""
            if cls == "machinery" and sustain_ratio > 1.4:
                flag = "  <-- BRIEF BURST, not sustained (looks like a tap to the classifier)"
            print(f"  {files[col]}: full-clip RMS={full_rms:.3f}, "
                  f"extracted-window RMS={windowed_rms:.3f}, "
                  f"sustain_ratio={sustain_ratio:.2f}{flag}")

    axes[0][0].legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=110)
    print(f"\nSaved: {OUTPUT_PATH}")
    print("\nsustain_ratio = extracted-window RMS / full-clip RMS.")
    print("  Background clips should sit near 1.0 (uniform throughout).")
    print("  A brief event (a tap, or a machinery clip that's only a short")
    print("  burst) pulls this notably above 1.0 -- expected and fine for")
    print("  'human', but a red flag for 'machinery' (flagged above), since")
    print("  it means the recording doesn't give the classifier a sustained-")
    print("  noise signature to tell it apart from a tap. If flagged,")
    print("  re-record with the device running for the FULL clip duration.")
    print("\nAlso check visually:")
    print("  - Orange band should sit ON TOP of the visible spike in each waveform.")
    print("    If it's sitting on flat/quiet sections instead, the windowing")
    print("    picked the wrong moment for that specific clip.")


if __name__ == "__main__":
    main()