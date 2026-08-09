import struct
import sys
import threading
import time
import wave
import spidev
from gpiozero import DigitalOutputDevice
import numpy as np

from task_bearing_est import loop_bearing_est

STEP_DELAY = 1

SAMPLE_RATE = 18000
DURATION = 20.0

M1A_PIN = 27
M1B_PIN = 22
step_pin = DigitalOutputDevice(M1A_PIN)
direction_pin = DigitalOutputDevice(M1B_PIN)

# Motor movement thread management
_movement_thread: threading.Thread | None = None
_stop_event = threading.Event()
_movement_lock = threading.Lock()

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 3_600_000  # 3.6 MHz
spi.mode = 0b00  # SPI Mode 0


def read_adc(channel):
    if channel < 0 or channel > 7:
        raise ValueError("Channel must be between 0 and 8")

    command = [1, (8 + channel) << 4, 0]

    response = spi.xfer2(command)

    # Convert the two returned bytes into a 10-bit ADC value
    value = ((response[1] & 0x03) << 8) | response[2]

    return value


def record_wav(channel: int, duration: float, sample_rate: int, output_path: str) -> float:
    """Sample one channel at a target rate and write it out as a WAV file.

    Returns the actual achieved sample rate, which may fall short of
    `sample_rate` if the SPI/Python overhead per read is the bottleneck.
    """
    period = 1.0 / sample_rate
    samples = []

    start = time.perf_counter()
    next_sample_time = start
    end_time = start + duration

    try:
        while time.perf_counter() < end_time:
            samples.append(read_adc(channel))
            next_sample_time += period
            sleep_for = next_sample_time - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\nRecording stopped early", file=sys.stderr)

    elapsed = time.perf_counter() - start
    actual_rate = len(samples) / elapsed if elapsed > 0 else 0.0

    # MCP3008 readings are unsigned 10-bit (0-1023). Center on 0 and scale
    # up to 16-bit signed PCM range for the WAV file.
    frames = bytearray()
    for value in samples:
        centered = (value - 512) * 32
        centered = max(-32768, min(32767, centered))
        frames += struct.pack("<h", centered)

    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(frames))

    return actual_rate


def _move_clockwise_blocking(duration: float) -> None:
    """Internal blocking function to move motor clockwise for duration seconds."""
    direction_pin.on()
    end_time = time.time() + duration
    while time.time() < end_time and not _stop_event.is_set():
        step_pin.on()
        time.sleep(STEP_DELAY)
        step_pin.off()
        time.sleep(STEP_DELAY)


def _move_counter_clockwise_blocking(duration: float) -> None:
    """Internal blocking function to move motor counter-clockwise for duration seconds."""
    direction_pin.off()
    end_time = time.time() + duration
    while time.time() < end_time and not _stop_event.is_set():
        step_pin.on()
        time.sleep(STEP_DELAY)
        step_pin.off()
        time.sleep(STEP_DELAY)


def move_clockwise(duration: float) -> threading.Thread:
    """Start moving motor clockwise for duration seconds. Returns immediately.

    Returns the Thread object for optional management (join, etc.).
    If a movement is already in progress, it will be stopped first.
    """
    global _movement_thread

    with _movement_lock:
        # Stop any current movement
        if _movement_thread is not None:
            _stop_event.set()
            _movement_thread.join(timeout=0.05)  # Short timeout to signal stop
            _movement_thread = None

        # Clear the stop event for the new movement
        _stop_event.clear()

        # Start new movement thread
        _movement_thread = threading.Thread(
            target=_move_clockwise_blocking,
            args=(duration,),
            daemon=True
        )
        _movement_thread.start()
        return _movement_thread


def move_counter_clockwise(duration: float) -> threading.Thread:
    """Start moving motor counter-clockwise for duration seconds. Returns immediately.

    Returns the Thread object for optional management (join, etc.).
    If a movement is already in progress, it will be stopped first.
    """
    global _movement_thread

    with _movement_lock:
        # Stop any current movement
        if _movement_thread is not None:
            _stop_event.set()
            _movement_thread.join(timeout=0.05)  # Short timeout to signal stop
            _movement_thread = None

        # Clear the stop event for the new movement
        _stop_event.clear()

        # Start new movement thread
        _movement_thread = threading.Thread(
            target=_move_counter_clockwise_blocking,
            args=(duration,),
            daemon=True
        )
        _movement_thread.start()
        return _movement_thread


def stop_motor() -> None:
    """Stop any currently running motor movement."""
    global _movement_thread

    with _movement_lock:
        if _movement_thread is not None:
            _stop_event.set()
            # Wait briefly for the thread to finish
            _movement_thread.join(timeout=0.05)
            _movement_thread = None
        _stop_event.clear()


def main() -> None:
    try:
        # Round-robin recording parameters
        duration = 0.1  # 100ms
        sample_rate = SAMPLE_RATE  # 18000 Hz (max documented)
        num_channels = 4

        # Initialize sample storage
        channel_samples = {i: [] for i in range(num_channels)}

        # Calculate timing
        sample_period = 1.0 / sample_rate
        total_samples = int(duration * sample_rate)  # 1800 samples
        samples_per_channel = total_samples // num_channels  # 450 per channel

        # Round-robin recording loop
        start_time = time.perf_counter()
        end_time = start_time + duration
        current_channel = 0
        next_sample_time = start_time
        sample_count = 0

        while time.perf_counter() < end_time and sample_count < total_samples:
            # Record sample from current channel
            channel_samples[current_channel].append(read_adc(current_channel))

            # Move to next channel
            current_channel = (current_channel + 1) % num_channels
            sample_count += 1

            # Timing
            next_sample_time += sample_period
            sleep_for = next_sample_time - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)

        # Signal processing placeholder
        # TODO: Put bearing estimation logic here using the 4 data streams
        channel_0_data = channel_samples[0]
        channel_1_data = channel_samples[1]
        channel_2_data = channel_samples[2]
        channel_3_data = channel_samples[3]
        bearing_est = loop_bearing_est(np.array(channel_samples), duration*sample_rate+1, sample_rate)

        # Point motor towards sound (rhythmic for now)
        while True:
            move_clockwise(0.5)
            move_counter_clockwise(0.5)

    finally:
        spi.close()


if __name__ == "__main__":
    main()
