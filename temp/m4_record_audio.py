"""
m4_record_audio.py

RUN THIS ON YOUR MAC DIRECTLY -- it needs a real microphone, which
this sandboxed environment doesn't have. This is the one file in the
whole project I genuinely cannot test myself.

Records short labeled clips for three classes:

  human       -- taps, claps, your voice saying something, a whistle
  machinery   -- a hairdryer, vacuum, electric toothbrush, blender...
                 anything loud, mechanical, and droning
  background  -- just the room, doing nothing

WHY ONLY THREE CLASSES (not six, like the original build guide):
  The actual thesis of this project is "ignore the loud machinery,
  respond to the quiet human sound underneath it." Three classes is
  the minimum that proves that story, and it means a few minutes of
  recording instead of an hour. You can always split "human" into
  tap/voice/whistle later if you want finer categories.

SETUP:
  pip3 install sounddevice scipy numpy

USAGE:
  python3 m4_record_audio.py
  (follow the prompts -- it asks you to make each kind of sound
  repeatedly, with a short countdown before each recording)

  Run it again any time to add MORE clips to an existing class --
  it won't overwrite what's already there.
"""

import os
import time
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from config import CANONICAL_FS as FS   # single source of truth -- matches training and Pi-side resampling
CLIP_DURATION = 1.0    # seconds per clip
DATA_DIR = "data"
CLIPS_PER_CLASS = 15

CLASSES = {
    "human": "Tap on a desk, clap, say a short word, or whistle. Vary it between takes.",
    # "machinery": "IMPORTANT: turn the device ON NOW, before the first clip, and leave "
                #  "it running continuously for ALL clips in this section -- do not turn "
                #  "it on/off between clips. A brief on/off burst is nearly "
                #  "indistinguishable from a tap to the classifier; sustained noise for "
                #  "the WHOLE clip is what actually teaches it apart. Hairdryer, vacuum, "
                #  "electric toothbrush, blender -- whatever you use, switch it on and "
                #  "just hold position near the mic while clips are recorded.",
    #  "background": "Stay quiet. Just the normal room sound, nothing deliberate.",
}


def record_clip(duration=CLIP_DURATION, fs=FS):
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def next_filename(folder):
    existing = [f for f in os.listdir(folder) if f.endswith(".wav")]
    return os.path.join(folder, f"clip_{len(existing):03d}.wav")


def main():
    for label, instruction in CLASSES.items():
        folder = os.path.join(DATA_DIR, label)
        os.makedirs(folder, exist_ok=True)
        already = len([f for f in os.listdir(folder) if f.endswith(".wav")])

        print(f"\n{'=' * 60}")
        print(f"CLASS: {label}   ({already} clips already saved)")
        print(f"  {instruction}")
        print(f"Recording {CLIPS_PER_CLASS} more clips, {CLIP_DURATION}s each.")
        if label == "machinery":
            input("Turn the device ON now, get it running steadily, THEN press Enter "
                  "(leave it running for every clip below -- don't turn it off/on "
                  "between clips)...")
        else:
            input("Press Enter when ready to start (Ctrl+C to skip this class)...")

        for i in range(CLIPS_PER_CLASS):
            print(f"  [{i + 1}/{CLIPS_PER_CLASS}] recording in...", end=" ", flush=True)
            for n in (3, 2, 1):
                print(n, end=" ", flush=True)
                time.sleep(0.4)
            print("GO")
            audio = record_clip()
            path = next_filename(folder)
            wavfile.write(path, FS, audio)
            time.sleep(0.3)   # brief gap before the next clip

        print(f"Done with '{label}'.")

    print("\nAll done. Check the 'data/' folder for your recordings.")
    print("Next: python3 m4_train_classifier.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped early -- whatever was already saved is still there.")
