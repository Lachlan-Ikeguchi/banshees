"""
servo_control.py

RUNS ON THE PI ONLY (needs gpiozero + a real GPIO servo). The heading
update MATH is not new -- it's the proportional control validated in
m2_closed_loop.py -- this file just drives a physical motor with it,
plus two hardware realities the simulation never had:

  1. SERVO_DIRECTION (config): whether increasing servo angle turns
     the board left or right depends on how the servo is physically
     mounted. If the head turns AWAY from sounds, flip the sign in
     config -- that's the fix, not a code change.

  2. Clamping: a hobby servo covers ~180 degrees, not 360. Bearings
     behind the array may be physically unreachable; the head goes as
     far as it can. This is the "sector-scanning instrument" framing
     from the original build guide, not a bug.
"""

from gpiozero import Servo

import config as cfg


class ServoHead:
    def __init__(self):
        self._servo = Servo(cfg.SERVO_GPIO_PIN)
        self.heading_deg = cfg.SERVO_START_DEG
        self._apply()

    def _apply(self):
        self.heading_deg = max(cfg.SERVO_MIN_DEG, min(cfg.SERVO_MAX_DEG, self.heading_deg))
        self._servo.value = (self.heading_deg - 90) / 90.0

    def turn_toward(self, relative_bearing_deg):
        """relative_bearing_deg: theta_est from the bearing estimator --
        where the source appears relative to the array's CURRENT facing."""
        self.heading_deg += cfg.SERVO_GAIN * cfg.SERVO_DIRECTION * relative_bearing_deg
        self._apply()

    def stop(self):
        self._servo.value = None   # release; don't hold torque forever
