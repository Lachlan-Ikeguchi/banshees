import argparse
import struct
import sys
import time
import wave

import spidev

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1000000  # 1 MHz
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read the ADC multiplexer")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Emit flushed CSV rows (timestamp,ch0,...,chN) to stdout "
        "instead of pretty-printing. Use this when piping into the local "
        "live_plot.py viewer.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of multiplexed channels to read each cycle, "
        "starting at channel 0 (default: 1)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Seconds to sleep between reads (default: 1.0)",
    )
    parser.add_argument(
        "--record",
        metavar="PATH",
        help="Record a single channel to a WAV file at PATH instead of "
        "streaming/printing, then exit. Play back with `aplay PATH` on "
        "the Pi or any media player on a laptop.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel to sample when using --record (default: 0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Seconds to record when using --record (default: 5.0)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=8000,
        help="Target sample rate in Hz when using --record (default: 8000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.record:
        try:
            actual_rate = record_wav(
                args.channel, args.duration, args.sample_rate, args.record
            )
        finally:
            spi.close()
        print(
            f"Wrote {args.record} "
            f"(target {args.sample_rate} Hz, actual {actual_rate:.0f} Hz)",
            file=sys.stderr,
        )
        return

    channels = list(range(args.channels))

    try:
        while True:
            values = [read_adc(channel) for channel in channels]

            if args.stream:
                row = ",".join([f"{time.time():.6f}", *map(str, values)])
                print(row, flush=True)
            else:
                readings = " ".join(
                    f"Channel {ch} = {value}" for ch, value in zip(channels, values)
                )
                print(readings)

            time.sleep(args.rate)
    except KeyboardInterrupt:
        print("\nProgram stopped", file=sys.stderr)
    finally:
        spi.close()


if __name__ == "__main__":
    main()
