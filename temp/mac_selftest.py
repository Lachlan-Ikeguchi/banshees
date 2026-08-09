"""
mac_selftest.py

===========================================================================
  RUN THIS ON YOUR MAC BEFORE SENDING ANYTHING TO THE TEAM.
  If it prints ALL CHECKS PASSED, the software is ready to flash.
===========================================================================

What it verifies, in order:
  1. All modules import (with a clear note about the two Pi-only ones)
  2. Bearing math recovers known angles across the full circle
  3. The classifier model file loads and predicts (or tells you to train one)
  4. The ACTUAL main.py runs end-to-end in simulation, and:
       - with a HUMAN source, the servo converges onto the bearing
       - with a MACHINERY source, the servo does not move
That last check is the whole project's behaviour, tested for real.

Usage:  python3 mac_selftest.py
"""

import importlib
import sys
import numpy as np

PASS, FAIL = "  [PASS]", "  [FAIL]"
failures = []


def check(name, ok, detail=""):
    print(f"{PASS if ok else FAIL} {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


print("=" * 70)
print("1. IMPORTS")
print("=" * 70)
for mod in ["config", "dsp", "mfcc", "mic_selection", "pi_optimized",
            "mock_hardware", "hardware", "m0_mock_mic", "pi_forest_predict"]:
    try:
        importlib.import_module(mod)
        check(f"import {mod}", True)
    except Exception as e:
        check(f"import {mod}", False, str(e))

print("  [NOTE] hardware_capture.py and servo_control.py are Pi-only "
      "(need spidev/gpiozero) -- not importable on a Mac, by design.")

import config as cfg
if not cfg.SIMULATION_MODE:
    print("\n  [WARN] config.SIMULATION_MODE is False -- flip it to True to "
          "test on this Mac.\n")

print()
print("=" * 70)
print("2. BEARING MATH (known angle in -> same angle out, full circle)")
print("=" * 70)
from pi_optimized import FastBearingEstimator
from m0_mock_mic import simulate_mic_array

est = FastBearingEstimator(cfg.MIC_POSITIONS, cfg.SPEED_OF_SOUND)
worst = 0.0
for true_bearing in range(0, 360, 30):
    ch = simulate_mic_array(true_bearing, noise_level=0.05)
    got = est.estimate(ch, fs=cfg.CANONICAL_FS)
    err = abs(((got - true_bearing) + 180) % 360 - 180)
    worst = max(worst, err)
check("bearing recovery, 12 angles", worst < 2.0, f"worst error {worst:.2f} deg")

print()
print("=" * 70)
print("3. CLASSIFIER MODEL")
print("=" * 70)
have_model = True
try:
    from pi_forest_predict import classify
    label, probs = classify(np.random.randn(cfg.N_SAMPLES_PER_CLIP) * 0.01,
                            fs=cfg.CANONICAL_FS)
    check("model loads and predicts", True, f"quiet noise -> '{label}'")
except FileNotFoundError as e:
    have_model = False
    check("model loads and predicts", False, "no forest_export.pkl")
    print("       -> run m4_record_audio.py then m4_train_classifier.py first.")

print()
print("=" * 70)
print("4. FULL main.py IN SIMULATION (the real file, the real loop)")
print("=" * 70)
if have_model:
    import mock_hardware, hardware, main

    def fresh_run(source_class, iters=5):
        cfg.MAX_ITERATIONS = iters
        cfg.SIM_SOURCE_CLASS = source_class
        cfg.SIM_SOURCE_BEARING_DEG = 55
        cfg.SERVO_SETTLE_SECONDS = 0.0
        importlib.reload(mock_hardware)
        importlib.reload(hardware)
        importlib.reload(main)
        # capture the mock servo's final heading by re-creating hardware inside main;
        # easiest reliable probe: monkeypatch get_hardware to keep a reference.
        holder = {}
        orig = hardware.get_hardware
        def wrapped():
            cap, srv = orig()
            holder["servo"] = srv
            return cap, srv
        hardware.get_hardware = wrapped
        main.get_hardware = wrapped
        main.main()
        return holder["servo"].heading_deg

    print("--- human source at +55 (servo should end near 145 = 90 start + 55) ---")
    final_h = fresh_run("human")
    check("servo converged toward human", abs(final_h - 145.0) < 10.0,
          f"final heading {final_h:.1f}, expected ~145")

    print("--- machinery source at +55 (servo should stay at 90) ---")
    final_m = fresh_run("machinery")
    check("servo ignored machinery", abs(final_m - cfg.SERVO_START_DEG) < 1.0,
          f"final heading {final_m:.1f}, expected {cfg.SERVO_START_DEG}")
else:
    print("  [SKIP] needs a trained model -- see step 3.")

print()
print("=" * 70)
print("5. CHANNEL VISUALISATION TOOL")
print("=" * 70)
try:
    import subprocess
    result = subprocess.run(
        [sys.executable, "channel_monitor.py", "--once"],
        capture_output=True, text=True, timeout=30,
    )
    import os
    ok = result.returncode == 0 and os.path.exists("channel_levels.png")
    check("channel_monitor.py runs and produces channel_levels.png", ok,
          result.stderr.strip().splitlines()[-1] if (not ok and result.stderr) else "")
except Exception as e:
    check("channel_monitor.py runs and produces channel_levels.png", False, str(e))

print()
print("=" * 70)
if failures:
    print(f"RESULT: {len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED -- ready to hand off.")
