"""
hardware.py

The ONE place where the real-vs-simulated decision is made. main.py
asks this factory for a capture device and a servo; everything after
that is identical in both worlds.

This is why the team only ever edits config.py: flipping
SIMULATION_MODE here swaps the entire hardware layer with no other
code changes anywhere.
"""

import config as cfg


def get_hardware():
    """Returns (capture, servo), either real or simulated per config."""
    if cfg.SIMULATION_MODE:
        from mock_hardware import MockCapture, MockServo
        servo = MockServo()
        capture = MockCapture(servo)
        print("Hardware: SIMULATION (mock mics + mock servo)")
    else:
        from hardware_capture import HardwareCapture
        from servo_control import ServoHead
        servo = ServoHead()
        capture = HardwareCapture()
        print("Hardware: REAL (SPI ADC + GPIO servo)")
    return capture, servo
