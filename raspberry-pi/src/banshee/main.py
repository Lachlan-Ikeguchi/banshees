import spidev
import time

spi = spidev.SpiDev() 
spi.open(0, 0)
spi.max_speed_hz = 2000  # 1 MHz
spi.mode = 0b00  # SPI Mode 0

def read_adc(channel):
    if channel < 0 or channel > 7: 
        raise ValueError("Channel must be between 0 and 8")

    command = [(8 + channel) << 4, 0b000000000, 0b0000000000]

    response = spi.xfer(command)

    return response

def main() -> None:
    try:
        while True:
            channel = 0b0000
            adc_value = read_adc(channel)

            print(
                f"Channel {channel} "
                f"ADC = {adc_value} "
            )
    except KeyboardInterrupt:
        print("\nProgram stopped")
    finally:
            spi.close()

if __name__ == "__main__":
    main()
