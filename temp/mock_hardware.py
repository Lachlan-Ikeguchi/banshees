"""
mock_hardware.py

Simulation stand-ins used when config.SIMULATION_MODE is True.
They present EXACTLY the same interface as the real hardware classes,
so main.py contains no if-statements about which world it's in --
the factory in hardware.py picks once, at startup.

The mock capture and mock servo share state (the servo's current
heading), which is what makes the simulation a genuine CLOSED loop:
when the mock servo turns, the mock mics really do hear the source
from a new relative angle on the next capture -- the same validated
feedback structure as m2_closed_loop.py, now running inside the
actual production main.py.
"""

import os
import numpy as np
from scipy.io import wavfile

import config as cfg


def _wrap(a):
    return (a + 180) % 360 - 180


class MockServo:
    """Same interface as servo_control.ServoHead, no GPIO required."""

    def __init__(self):
        self.heading_deg = cfg.SERVO_START_DEG

    def turn_toward(self, relative_bearing_deg):
        self.heading_deg += cfg.SERVO_GAIN * cfg.SERVO_DIRECTION * relative_bearing_deg
        self.heading_deg = max(cfg.SERVO_MIN_DEG, min(cfg.SERVO_MAX_DEG, self.heading_deg))
        print(f"    [MockServo] turned to {self.heading_deg:.1f} deg")

    def stop(self):
        print("    [MockServo] released")


class MockCapture:
    """Same interface as hardware_capture.HardwareCapture."""

    def __init__(self, servo):
        self._servo = servo   # shared state -- this is what closes the simulated loop

    def _load_real_clip(self, kind, n):
        """
        If real recorded clips exist for this class (data/<kind>/*.wav,
        produced by m4_record_audio.py), use a random one -- this is
        what makes the simulation test the ACTUAL trained model against
        ACTUAL audio it was trained on, rather than a guess at what
        "human" or "machinery" sounds like. Returns None if no real data
        is available yet, so the caller can fall back to a placeholder.
        """
        folder = os.path.join("data", kind)
        if not os.path.isdir(folder):
            return None
        files = [f for f in os.listdir(folder) if f.endswith(".wav")]
        if not files:
            return None

        fname = np.random.choice(files)
        fs, audio = wavfile.read(os.path.join(folder, fname))
        audio = audio.astype(np.float32)
        if audio.max() > 1.5:          # int16 wav -> normalise to -1..1
            audio = audio / 32768.0

        if len(audio) <= n:
            return np.pad(audio, (0, n - len(audio)))

        # BUG FIX: originally took audio[:n], assuming the sound starts
        # at sample 0. It doesn't -- m4_record_audio.py prints a
        # countdown then starts recording immediately, and real human
        # reaction time (measured: ~200-500ms) means the actual clap/tap
        # lands well after sample 0. Taking the first n samples of a
        # 4096-sample window at 16kHz (~256ms) can miss the event
        # entirely and extract pure lead-in silence instead -- confirmed
        # directly: a simulated 350ms reaction delay put the tap at
        # sample 5600, outside a 4096-sample first-slice window.
        #
        # Fix: find where the clip's short-time energy actually peaks,
        # and centre the extracted window there instead of at sample 0.
        window = max(256, n // 8)
        energy = np.convolve(audio ** 2, np.ones(window) / window, mode="same")
        peak_idx = int(np.argmax(energy))

        start = peak_idx - n // 2
        start = max(0, min(start, len(audio) - n))
        return audio[start:start + n]

    def _make_source_sound(self, n):
        """
        The audio "heard" by the simulated array. Prefers a REAL
        recorded clip (see _load_real_clip); only falls back to a crude
        synthetic placeholder if no real data has been recorded yet --
        which happens before step 1 of the pipeline, or if you're
        testing the wiring in isolation. Once real data exists, this
        function stops being used for that class at all.
        """
        kind = cfg.SIM_SOURCE_CLASS

        real_clip = self._load_real_clip(kind, n)
        if real_clip is not None:
            return real_clip

        print(f"    [MockCapture] no real data/{kind}/ clips found -- "
              f"using a crude synthetic placeholder. Run m4_record_audio.py "
              f"for a meaningful test.")
        if kind == "human":
            env = np.exp(-np.linspace(0, 15, n))
            return np.random.randn(n) * env * 0.5
        if kind == "machinery":
            t = np.arange(n) / cfg.CANONICAL_FS
            return 0.3 * np.sin(2 * np.pi * 120 * t) + 0.05 * np.random.randn(n)
        return 0.005 * np.random.randn(n)

    def capture(self):
        # Where does the fixed simulated source appear, relative to the
        # direction the (mock) servo currently points the array?
        # SERVO_START_DEG is defined as "array facing forward", so the
        # array's world facing = servo heading - start offset.
        from m0_mock_mic import mic_array_delays_seconds, apply_fractional_delay
        array_facing = self._servo.heading_deg - cfg.SERVO_START_DEG
        theta = _wrap(cfg.SIM_SOURCE_BEARING_DEG - array_facing)

        n = cfg.N_SAMPLES_PER_CLIP
        base = self._make_source_sound(n)

        # BUG FIX: SIM_NOISE_LEVEL used to be added as a fixed absolute
        # amplitude (0.05), which was comparable to or LARGER than real
        # recorded clip levels (measured ~0.01-0.06 RMS) -- confirmed
        # directly: this swamped real signal and left behind sustained
        # broadband noise, which is exactly the machinery class's own
        # signature, biasing everything toward "machinery" regardless
        # of what the real clip actually was. Noise now scales with the
        # loaded clip's OWN level instead of a fixed absolute floor, so
        # it behaves sensibly whether a real clip happens to be loud or
        # quiet, rather than silently overwhelming quiet ones.
        base_rms = float(np.sqrt(np.mean(base ** 2))) + 1e-6
        noise_amplitude = cfg.SIM_NOISE_LEVEL * base_rms

        delays = mic_array_delays_seconds(theta, cfg.MIC_POSITIONS) * cfg.CANONICAL_FS
        channels = [
            (apply_fractional_delay(base, d)
             + noise_amplitude * np.random.randn(n)).astype(np.float32)
            for d in delays
        ]
        return channels, float(cfg.CANONICAL_FS)

    def close(self):
        pass
