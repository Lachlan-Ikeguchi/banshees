"""
channel_monitor.py

DIAGNOSTIC TOOL -- not part of the main pipeline, doesn't touch
main.py's real-time loop at all. Run this separately, before running
main.py for real, to SEE all 4 mic channels' waveforms and levels --
this is the tool the "trimpot leveling" step in HANDOFF.md refers to.

Works in BOTH simulation and real-hardware mode automatically (same
config.SIMULATION_MODE switch, same hardware.py factory main.py
uses) -- so you can preview it on the Mac before ever touching the Pi.

WHY IT WRITES A FILE INSTEAD OF POPPING UP A WINDOW:
  The Pi will likely be run headless, over SSH, with no monitor
  attached -- a script that needs a live display window would simply
  fail there. Instead this repeatedly captures and OVERWRITES
  channel_levels.png on disk; open that file (SCP it over, view it
  over VNC, whatever's easiest) and refresh to see the current state.
  Works identically whether or not a display exists, which matters
  because we don't know which situation your team will be in.

USAGE:
  python3 channel_monitor.py             loops until Ctrl+C, rewriting
                                          channel_levels.png each capture
  python3 channel_monitor.py --once      single capture, single plot, exit
"""

import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display required -- safe on a headless Pi or a Mac
import matplotlib.pyplot as plt

from hardware import get_hardware

OUTPUT_PATH = "channel_levels.png"
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]


def plot_channels(channels, measured_fs, iteration):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))

    # --- top: raw waveforms, all 4 stacked with a vertical offset so
    # they don't sit on top of one another and hide differences ---
    ax = axes[0]
    offset_step = 2.5
    for i, ch in enumerate(channels):
        ax.plot(ch + i * offset_step, color=COLORS[i], linewidth=0.8, label=f"MIC{i}")
    ax.set_yticks([i * offset_step for i in range(4)])
    ax.set_yticklabels([f"MIC{i}" for i in range(4)])
    ax.set_xlabel("sample")
    ax.set_title(f"Raw channel waveforms  (capture #{iteration}, measured fs={measured_fs:.0f} Hz)")
    ax.legend(loc="upper right", fontsize=8)

    # --- bottom: the actual number the trimpot-leveling step cares
    # about -- are all 4 bars roughly the same height after a clap
    # from directly above the board's centre? ---
    ax2 = axes[1]
    peaks = [float(np.max(np.abs(ch))) for ch in channels]
    rms = [float(np.sqrt(np.mean(ch ** 2))) for ch in channels]
    x = np.arange(4)
    width = 0.35
    ax2.bar(x - width / 2, peaks, width, label="peak", color="#ff7f0e")
    ax2.bar(x + width / 2, rms, width, label="RMS", color="#1f77b4")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"MIC{i}" for i in range(4)])
    ax2.set_ylabel("amplitude (normalised, roughly -1..1)")
    ax2.set_title("Per-channel level -- these four bars should be SIMILAR "
                  "after a clap from directly above the board's centre")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=110)
    plt.close(fig)

    print(f"[{iteration}] fs={measured_fs:.0f}Hz   " +
          "  ".join(f"MIC{i} peak={peaks[i]:.2f} rms={rms[i]:.3f}" for i in range(4)))

    return peaks, rms


def main():
    once = "--once" in sys.argv
    print(f"channel_monitor.py -- writing {OUTPUT_PATH} after every capture.")
    print("Open that file and refresh it to see the current channel levels.\n")

    capture, servo = get_hardware()
    iteration = 0
    try:
        while True:
            iteration += 1
            channels, measured_fs = capture.capture()
            plot_channels(channels, measured_fs, iteration)
            if once:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        servo.stop()
        capture.close()
        print(f"\nDone. Final plot: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
