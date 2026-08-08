import spidev


def main() -> None:
    # Open SPI bus 0, device 0 (CE0)
    # Default Raspberry Pi SPI0 pins:
    #   MOSI: GPIO 10 (Pin 19)
    #   MISO: GPIO 9  (Pin 21)
    #   CLK:  GPIO 11 (Pin 23)
    #   CE0:  GPIO 8  (Pin 24)
    spi = spidev.SpiDev()
    try:
        spi.open(0, 0)
        spi.max_speed_hz = 1000000  # 1 MHz
        spi.mode = 0b00  # SPI Mode 0
        
        # Test: send 3 bytes and read response
        tx_data = [0x01, 0x02, 0x03]
        rx_data = spi.xfer2(tx_data)
        
        print(f"SPI Test - Sent: {tx_data}, Received: {list(rx_data)}")
        
    except OSError as e:
        print(f"SPI Error: {e}")
        print("Ensure SPI is enabled: sudo raspi-config -> Interface Options -> SPI -> Enable")
    finally:
        spi.close()


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
