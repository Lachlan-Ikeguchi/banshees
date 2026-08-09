"""
main.py

=======================================================================
  THE SCRIPT YOU RUN. Everything is configured in config.py -- this
  file should not need editing.

    On your Mac (test):    config.SIMULATION_MODE = True   -> python3 main.py
    On the Pi (deploy):    config.SIMULATION_MODE = False  -> python3 main.py
=======================================================================

THE PIPELINE, EVERY ITERATION:
  1. Capture a clip from all 4 mics (real ADC, or simulated -- decided
     once at startup by hardware.py; this file never needs to know which)
  2. Skip if it's basically silence
  3. Estimate the sound's bearing from all 6 mic-pair timing differences
  4. Pick the single mic physically closest to that bearing
  5. Classify that mic's audio: human / machinery / background
  6. Only if confidently human: turn the servo toward the bearing

Direction (step 3) and identity (step 5) are two separate calculations
on the same audio. The classifier doesn't help find the direction --
it decides whether the direction is worth ACTING on. A loud drill gets
located and then deliberately ignored; a quiet tap gets located and
turned toward. That behaviour IS the project.

The loop runs continuously by default (config.MAX_ITERATIONS = None);
that's the intended mode -- the system's whole job is to keep
listening. Set a number in config only for scripted tests.
"""

import time
import numpy as np

import config as cfg
from hardware import get_hardware
from pi_optimized import FastBearingEstimator, apply_realtime_priority, pin_to_core
from mic_selection import nearest_mic_index
from pi_forest_predict import classify


def clip_energy(channels):
    """Rough loudness of this capture -- used to skip silent frames."""
    return float(sum(np.std(ch) for ch in channels))


def main():
    print("=== AfterShock ===")

    # System-level niceties; each degrades gracefully (prints a note and
    # continues) on platforms where it isn't available, e.g. a Mac.
    if not cfg.SIMULATION_MODE:
        apply_realtime_priority()
        pin_to_core(3)

    capture, servo = get_hardware()
    bearing_estimator = FastBearingEstimator(cfg.MIC_POSITIONS, cfg.SPEED_OF_SOUND)

    print("Listening. Ctrl+C to stop.\n")
    iteration = 0
    try:
        while cfg.MAX_ITERATIONS is None or iteration < cfg.MAX_ITERATIONS:
            iteration += 1

            # 1. capture (returns the audio AND the rate it was really taken at)
            channels, measured_fs = capture.capture()

            # 2. silence gate
            energy = clip_energy(channels)
            if energy < cfg.MIN_ENERGY_TO_ACT:
                continue

            # 3. where is it? (always computed, whatever the sound is)
            theta_est = bearing_estimator.estimate(channels, fs=measured_fs)

            # 4. which mic faces that way most directly?
            mic_idx = nearest_mic_index(theta_est, cfg.MIC_POSITIONS)

            # 5. what is it?
            label, probs = classify(channels[mic_idx], fs=measured_fs)

            print(f"[{iteration:4d}] fs={measured_fs:7.0f}Hz  energy={energy:5.2f}  "
                  f"bearing={theta_est:+7.1f}  mic={mic_idx}  "
                  f"label={label:<10}  p(human)={probs['human']:.2f}")

            # 6. act only on confident humans
            if label == "human" and probs["human"] >= cfg.CONFIDENCE_THRESHOLD:
                servo.turn_toward(theta_est)
                time.sleep(cfg.SERVO_SETTLE_SECONDS)   # let servo noise die down
                                                        # before the next capture

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        servo.stop()
        capture.close()


if __name__ == "__main__":
    main()
