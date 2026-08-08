import argparse
import sys
import time

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read the ADC multiplexer")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Emit flushed CSV rows (timestamp,ch0,...,chN) to stdout "
        "instead of pretty-printing. Use this when piping over SSH to a "
        "remote plotting client.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
