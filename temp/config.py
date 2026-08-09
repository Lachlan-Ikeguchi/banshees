"""
config.py

=======================================================================
  THE ONLY FILE YOU SHOULD NEED TO EDIT.
=======================================================================

Every setting for the whole system lives here -- both the Mac-side
scripts (recording, training, self-test) and the Pi-side scripts
(main.py) read this same file.

THE ONE SWITCH THAT MATTERS MOST:

  SIMULATION_MODE = True    -> runs anywhere (Mac, no hardware needed),
                                using simulated mics and a pretend servo.
                                Use this to test before deploying.
  SIMULATION_MODE = False   -> runs on the Pi with real SPI/GPIO hardware.

Everything else is annotated one of three ways:

  # CHECK:    must be verified against your actual wiring/hardware.
              A wrong value here produces WRONG BEHAVIOUR, often
              silently (no error message).
  # TUNE:     a sensible default; adjust once running if needed.
  # DECISION: a judgment call made during development, explained so
              you know why -- change only if you understand the note.
"""

import numpy as np

# =====================================================================
# MODE
# =====================================================================

SIMULATION_MODE = True   # CHECK: set to False when flashing onto the Pi.
                          #        True = mock mics + mock servo, runs anywhere.

# =====================================================================
# MICROPHONE ARRAY GEOMETRY
# =====================================================================

MIC_SPACING_M = 0.15   # CHECK: side length of the square mic board, in metres,
                        #        measured centre-to-centre between adjacent mics.
                        #        The bearing math depends directly on this number.

# Mic positions derived from the spacing. Viewed from above, forward = 0 deg:
#
#      MIC1 (-h,+h) o----------------o MIC2 (+h,+h)
#                   |                |
#                   |   + centre     |        +x = 0 deg (forward)
#                   |                |        +y = 90 deg (left)
#      MIC0 (-h,-h) o----------------o MIC3 (+h,-h)
#
# DO NOT reorder these -- MIC_ADC_CHANNELS below and the wiring doc
# both assume exactly this labelling.
_h = MIC_SPACING_M / 2
MIC_POSITIONS = np.array([
    [-_h, -_h],   # MIC0  back-left
    [-_h,  _h],   # MIC1  front-left
    [ _h,  _h],   # MIC2  front-right
    [ _h, -_h],   # MIC3  back-right
])
SPEED_OF_SOUND = 343.0

# =====================================================================
# ADC / SPI  (only used when SIMULATION_MODE = False)
# =====================================================================

SPI_BUS = 0          # CHECK: usually 0 on a Pi 2
SPI_DEVICE = 0        # CHECK: 0 = CE0 pin, 1 = CE1 pin
SPI_SPEED_HZ = 1_350_000   # DECISION: the MCP3008 datasheet's max SPI clock at
                            #           VDD=2.7V is 1.35MHz; at 3.3V the true limit is
                            #           slightly above that but not specified, so this
                            #           stays at the guaranteed-safe figure. Raising it
                            #           gains sample rate at the cost of leaving the
                            #           datasheet's guaranteed envelope.

# WHICH MCP3008 CHANNEL EACH MIC IS WIRED TO.
# CHECK: this is the single most important thing to verify in this file.
# If MIC0's wire actually goes to a different CHx pin than listed here,
# the bearing math will point confidently in the WRONG direction with
# no error message of any kind.
MIC_ADC_CHANNELS = {"MIC0": 0, "MIC1": 1, "MIC2": 2, "MIC3": 3}

# =====================================================================
# SAMPLE RATE AND CLIP LENGTH
# =====================================================================

# DECISION: there is no "set the sample rate" knob anymore, on purpose.
#
#   You observed that 200,000 Hz failed and 16,000 Hz "worked". The
#   200kHz failure is expected: 200ksps is the MCP3008's absolute
#   hardware ceiling under perfect conditions, and Python adds large
#   per-read overhead on top -- the Pi 2 cannot reach it. But a paced
#   loop targeting ANY fixed rate has the same fragility: if the Pi
#   can't keep the pace, timing silently corrupts and bearings go bad.
#
#   So capture now FREE-RUNS: it reads the ADC as fast as it can,
#   MEASURES the rate it actually achieved, and passes that measured
#   rate into all downstream math. There is no target to miss.
#   Expect roughly 5,000-15,000 Hz per channel on a Pi 2 in Python --
#   the startup printout will tell you the real number.

N_SAMPLES_PER_CLIP = 4096   # DECISION: clip length is set in SAMPLES, not seconds,
                             #           because wall-clock duration depends on the
                             #           measured rate. 4096 samples ~= 0.5s at 8kHz,
                             #           ~0.3s at 13kHz -- either is enough to catch a
                             #           tap/word and short enough for a responsive loop.

CANONICAL_FS = 16000   # DECISION: the fixed "reference" rate that the classifier is
                        #           trained at and that captured audio is RESAMPLED to
                        #           before classification. Do not change without
                        #           retraining the classifier.

# =====================================================================
# CLASSIFIER
# =====================================================================

MODEL_PATH = "forest_export.pkl"   # CHECK: must sit next to main.py on the Pi.
                                     #        Produced on the Mac by m4_train_classifier.py.

CONFIDENCE_THRESHOLD = 0.5   # TUNE: minimum p(human) before the servo may move.
                              #       Raise if it chases false positives; lower if
                              #       it ignores real taps.

MIN_ENERGY_TO_ACT = 0.08     # TUNE: skip near-silent clips entirely (saves compute,
                              #       prevents twitching at room noise). Audio is
                              #       normalised to roughly -1..1, so this is in those
                              #       units. Watch the printed energy values live and
                              #       set this between "quiet room" and "clap".

# =====================================================================
# SERVO  (only used when SIMULATION_MODE = False)
# =====================================================================

SERVO_GPIO_PIN = 18       # CHECK: BCM numbering; physical pin 12 on the header
SERVO_MIN_DEG = 0.0        # CHECK: your servo's physical travel limits
SERVO_MAX_DEG = 180.0
SERVO_START_DEG = 90.0     # CHECK: the servo angle at which the mic board faces "forward"
SERVO_DIRECTION = +1       # CHECK: if the head turns AWAY from sounds instead of toward
                            #        them, set this to -1. Whether increasing servo angle
                            #        turns the board left or right depends entirely on
                            #        how the servo is physically mounted -- there is no
                            #        way to know without trying it once.
SERVO_GAIN = 1.0           # TUNE: fraction of the estimated correction applied per
                            #       update. 1.0 = jump straight to the estimate
                            #       (validated in simulation); reduce if it overshoots.
SERVO_SETTLE_SECONDS = 0.3  # DECISION: pause after each servo move before capturing
                             #           again -- the servo's own movement noise and
                             #           whine would otherwise contaminate the very
                             #           next audio clip.

# =====================================================================
# MAIN LOOP
# =====================================================================

MAX_ITERATIONS = None   # DECISION: None = run forever (the normal mode -- the whole
                         #           point is continuous listening). Set to a number
                         #           (e.g. 20) only for scripted tests.

# =====================================================================
# SIMULATION SETTINGS  (only used when SIMULATION_MODE = True)
# =====================================================================

SIM_SOURCE_BEARING_DEG = 55    # TUNE: where the pretend sound source sits in the
                                #       simulated world
SIM_NOISE_LEVEL = 0.05          # TUNE: fraction of the loaded clip's OWN RMS level
                                 #       (not an absolute amplitude -- fixed as such,
                                 #       since real recordings vary in loudness and a
                                 #       fixed absolute value can swamp quiet clips
                                 #       entirely, confirmed directly during testing)
SIM_SOURCE_CLASS = "human"      # TUNE: what KIND of sound the simulated source makes --
                                #       "human" (transient tap-like: servo should turn),
                                #       "machinery" (steady tone: servo should NOT turn),
                                #       or "background" (near-silence: skipped entirely).
                                #       Run main.py once with each to verify both the
                                #       act and the ignore branches before deploying.
