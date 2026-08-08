import spidev
import time

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

def main() -> None:
    try:
        while True:
            channel = 0
            adc_value = read_adc(channel)

            print(
                f"Channel {channel} "
                f"ADC = {adc_value} "
            )
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nProgram stopped")
    finally:
            spi.close()

if __name__ == "__main__":
    main()
