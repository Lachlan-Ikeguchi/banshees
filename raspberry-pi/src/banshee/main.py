import spidev
import time

spi = spidev.SpiDev() 

def read_adc(channel):
    if channel < 0 or channel > 7: 
        raise ValueError("Channel must be between 0 and 8")

    command = [1, (8 + channel) << 4, 0]

    response = spi.xfer2(command)

    # Convert the two returned bytes into a 10-bit ADC value
    value = ((response[1] & 0x03) << 8) | response[2]

    return value

def main() -> None:
    # Open SPI bus 0, device 0 (CE0)
    # Default Raspberry Pi SPI0 pins:
    #   MOSI: GPIO 10 (Pin 19)
    #   MISO: GPIO 9  (Pin 21)
    #   CLK:  GPIO 11 (Pin 23)
    #   CE0:  GPIO 8  (Pin 24)
    
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
    

    print("ADC TEST")

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
