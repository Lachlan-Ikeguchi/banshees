import spidev

# open SPI bus 0
spi = spidev.SpiDev() 
spi.open(0, 0)

# max speed 200kHz
spi.max_speed_hz = 200000

def read_adc(channel):
    if channel < 0 or channel > 3: 
        raise ValueError("Channel must be between 0 and 3")

    command = [1, (8 + channel) << 4, 0]

def main():
    print("Hello from pi!")


if __name__ == "__main__":
    main()
