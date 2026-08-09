"""
hardware_capture.py

RUNS ON THE PI ONLY (needs spidev + a real MCP3008). Cannot be tested
without hardware -- but you never run this file directly: main.py in
SIMULATION_MODE exercises the identical downstream pipeline with
mock_hardware.py standing in for this file, so by the time this runs
on the Pi, everything EXCEPT the physical SPI read has been verified.

DESIGN: FREE-RUNNING CAPTURE (this is different from earlier drafts)

  Earlier versions paced the read loop to hit a target sample rate.
  Field testing showed why that's fragile: ask for a rate the Pi
  can't deliver (like the 200kHz attempt) and timing silently
  corrupts. So this version doesn't pace at all:

    1. Read the ADC in a tight loop, as fast as it goes.
    2. Time how long the whole clip actually took.
    3. Report the MEASURED rate, and hand it downstream --
       the bearing math uses the measured rate, and the classifier
       resamples to the canonical training rate before classifying.

  There is no target to miss. Whatever rate your specific Pi + SPI
  setup achieves IS the rate, correctly accounted for everywhere.

WHAT ELSE THIS FILE HANDLES:
  - Channel skew: the MCP3008 reads its 4 channels one after another,
    not simultaneously. Each channel k is sampled ~k/4 of a sample
    period later than channel 0. Corrected with the same FFT
    phase-ramp used (and validated) in m0_mock_mic's fractional-delay
    code. Assumes the 4 reads within a round take equal time --
    reasonable, since they are identical operations.
  - Normalisation: raw ADC counts (0..1023, centred ~512) are scaled
    to roughly -1..1 floats, matching both the simulator's output and
    the classifier's training data convention.
"""

import time
import numpy as np
import spidev

from m0_mock_mic import apply_fractional_delay
import config as cfg


class HardwareCapture:
    def __init__(self):
        self._spi = spidev.SpiDev()
        self._spi.open(cfg.SPI_BUS, cfg.SPI_DEVICE)
        self._spi.max_speed_hz = cfg.SPI_SPEED_HZ

        # Ordered so buffer index 0 always means MIC0, matching
        # config.MIC_POSITIONS' ordering.
        self._adc_channels = [cfg.MIC_ADC_CHANNELS[f"MIC{i}"] for i in range(4)]

        self.n_samples = cfg.N_SAMPLES_PER_CLIP

        # Buffers allocated once, reused every capture -- avoids
        # per-clip allocation churn in a loop that runs for hours.
        self._raw = np.zeros((4, self.n_samples), dtype=np.float32)

    def _read_adc_channel(self, channel):
        cmd = [1, (8 + channel) << 4, 0]
        reply = self._spi.xfer2(cmd)
        return ((reply[1] & 3) << 8) | reply[2]

    def capture(self):
        """
        Returns (channels, measured_fs):
          channels    -- list of 4 float32 arrays, normalised ~-1..1,
                         deskewed, index order MIC0..MIC3
          measured_fs -- the per-channel sample rate ACTUALLY achieved
                         on this capture, in Hz
        """
        t_start = time.perf_counter()
        for i in range(self.n_samples):
            for k, ch in enumerate(self._adc_channels):
                self._raw[k, i] = self._read_adc_channel(ch)
        t_elapsed = time.perf_counter() - t_start

        measured_fs = self.n_samples / t_elapsed

        channels = []
        for k in range(4):
            x = (self._raw[k] - 512.0) / 512.0          # counts -> ~-1..1
            # channel k was sampled k/4 of a sample period late; undo it
            x = apply_fractional_delay(x, -k / 4.0)
            channels.append(x.astype(np.float32))

        return channels, measured_fs

    def close(self):
        self._spi.close()
