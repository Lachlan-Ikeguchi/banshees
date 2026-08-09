# AfterShock — Team Handoff

**The one rule: the only file you edit is `config.py`.** Everything else is
tested and works together. If something needs changing elsewhere, that's a
bug — tell us, don't patch around it.

## Order of operations

1. **Classifier (Mac, real audio):** `m4_record_audio.py` → `m4_train_classifier.py` → `forest_export.pkl`. Check the "machinery misclassified as human" number before moving on.
2. **Mac test, with the real model:** `mac_selftest.py` must print green before you touch hardware.
3. **Wire the hardware** per section 2 below.
4. **Copy files to the Pi** (list below) + your `forest_export.pkl`. Install deps.
5. **Edit `config.py`:** `SIMULATION_MODE = False` + every `# CHECK` item (some, like `SERVO_DIRECTION`, you won't know until step 7 — that's expected, come back to them).
6. **Calibrate:** `channel_monitor.py` — both level balance AND channel identity (see section 3 below, these are two different checks).
7. **First live run:** `python3 main.py`, tune `MIN_ENERGY_TO_ACT` and `SERVO_DIRECTION` from what you observe.

---

## 1. What to copy onto the Pi 2

```
config.py            <- EDIT THIS ONE (set SIMULATION_MODE = False + verify CHECK items)
main.py               <- run this:  python3 main.py
channel_monitor.py    <- diagnostic tool: python3 channel_monitor.py --once
hardware.py
hardware_capture.py
servo_control.py
mock_hardware.py
m0_mock_mic.py
dsp.py
mfcc.py
mic_selection.py
pi_optimized.py
pi_forest_predict.py
forest_export.pkl    <- produced on the Mac (section 4); MUST sit next to main.py
```

Install on the Pi (piwheels provides prebuilt ARM packages, so this should
be quick and compile nothing):

```bash
sudo raspi-config     # Interface Options -> SPI -> Enable -> reboot
pip3 install numpy scipy spidev gpiozero
```

Then: edit `config.py` (see section 3), and `python3 main.py`.

---

## 2. Wiring / physical configuration

### MCP3008 ADC -> Pi 2 (40-pin header, PHYSICAL pin numbers)

| MCP3008 pin | Goes to | Pi physical pin |
|---|---|---|
| VDD (16) | 3.3V | pin 1 |
| VREF (15) | 3.3V | pin 17 |
| AGND (14) | GND | pin 6 |
| DGND (9) | GND | pin 9 |
| CLK (13) | GPIO11 / SCLK | pin 23 |
| DOUT (12) | GPIO9 / MISO | pin 21 |
| DIN (11) | GPIO10 / MOSI | pin 19 |
| CS (10) | GPIO8 / CE0 | pin 24 |

**Why 3.3V, not 5V:** the ADC's DOUT talks straight to the Pi, whose GPIO is
3.3V-only. Running the chip at 3.3V keeps its output safe with zero extra
parts. Consequence: turn each mic's gain trimpot DOWN until loud claps peak
below ~3V, or the tops of loud sounds clip.

### Microphones (XC-4438) -> MCP3008

| Mic | Connection |
|---|---|
| Every mic `+` | Pi 5V (pin 2 or 4) |
| Every mic `G` | Pi GND |
| Every mic `AO` | its MCP3008 CHx (below) |
| Every mic `DO` | **leave unconnected** |

Default channel map (change `MIC_ADC_CHANNELS` in config if wired differently):
MIC0→CH0, MIC1→CH1, MIC2→CH2, MIC3→CH3.

### Mic board orientation — THIS DIAGRAM IS LOAD-BEARING

Viewed from ABOVE, "forward" = the direction the board faces at servo start:

```
                     LEFT (+90 deg)
     MIC1 o----------------o MIC2
          |                |
          |    (centre)    |------->  FORWARD (0 deg)
          |                |
     MIC0 o----------------o MIC3

     FORWARD (0 deg) = from the centre out through the MIC2/MIC3 edge.
     LEFT  (+90 deg) = from the centre out through the MIC1/MIC2 edge.
     Positive bearings are to the LEFT, negative to the RIGHT.
```

Adjacent mics exactly **15.0 cm** apart centre-to-centre (or measure yours
and set `MIC_SPACING_M`). If bearings come out mirrored or rotated, the
physical board orientation doesn't match this diagram — rotate the board,
or remap `MIC_ADC_CHANNELS`, don't touch the math.

**Trimpot leveling and channel identity (do both, use the tool):**

```bash
python3 channel_monitor.py --once
```

Open `channel_levels.png`. Top panel: all 4 raw waveforms, stacked so you
can compare them at a glance. Bottom panel: a peak/RMS bar chart per mic.
Two SEPARATE checks, not one:

- **Level balance:** clap once from directly above the CENTRE of the board.
  All four bars should be roughly the same height — this is what trimpots
  fix. Adjust and re-run until they match.
- **Channel identity:** clap right next to ONE physical mic at a time. The
  SAME-NUMBERED `MICn` bar should spike for whichever mic you're next to.
  Check all 4. If clapping next to the mic you believe is wired to `MIC1`
  actually spikes the `MIC2` bar, `MIC_ADC_CHANNELS` in config is wrong —
  fix the mapping there, don't touch the wiring. **This check matters
  separately from level balance:** a swapped mapping can still look
  "balanced" from a centred clap while silently pointing every bearing
  estimate in the wrong direction.

Run with no `--once` flag to keep re-capturing every 0.5s while you work,
watching the file update live.

### Servo

| | |
|---|---|
| Signal | GPIO18 (physical pin 12) |
| Power | **separate 5V supply — NOT the Pi's 5V rail** (servo current spikes brown-out the Pi) |
| Grounds | servo supply GND joined to Pi GND (any Pi GND pin) |

Mount so servo angle 90 = board facing forward. If the head turns AWAY from
sounds on first test: set `SERVO_DIRECTION = -1` in config. That's the fix.

---

## 3. Config checklist before first run on the Pi

1. `SIMULATION_MODE = False`
2. `MIC_ADC_CHANNELS` — matches your actual wiring (**the highest-stakes
   setting in the file**: wrong mapping = confidently wrong bearings, no error)
3. `MIC_SPACING_M` — measured, in metres
4. `SERVO_DIRECTION` — flip to -1 if it turns the wrong way
5. `MIN_ENERGY_TO_ACT` — watch the printed `energy=` values in a quiet room
   vs. during a clap; set the threshold between them

Startup prints the **measured** sample rate (`fs=...` each line). There is no
sample-rate setting to get wrong anymore — the code free-runs the ADC and
uses whatever rate it actually achieves. Expect roughly 5–15 kHz per channel;
anything in that band is fine.

---

## 4. Classifier: yes, it must be retrained (Mac side)

The old model is invalid — feature extraction changed (librosa → mfcc.py)
and the audio normalisation convention changed. On the Mac:

```bash
pip3 install sounddevice scipy numpy scikit-learn
python3 m4_record_audio.py       # ~4 min of guided recording, 3 classes
python3 m4_train_classifier.py   # prints confusion matrix, writes forest_export.pkl
scp forest_export.pkl pi@<pi-address>:~/aftershock/
```

The number to look at in the training output is **"Machinery misclassified
as human"** — that's the false-alarm rate that would make the servo chase a
drill. If it's high, record more machinery clips at varied distances/volumes
and retrain. For best results, record the machinery clips in the same room
the demo happens in.

Only the `.pkl` goes to the Pi. Retraining never requires touching Pi code.

---

## 5. Test before you ship (Mac side)

```bash
python3 mac_selftest.py
```

Green `ALL CHECKS PASSED` means: imports OK, bearing math recovers known
angles to <2 deg around the full circle, model loads, `channel_monitor.py`
produces a plot, and the **actual main.py** run in simulation (a) converges
the servo onto a simulated human source and (b) refuses to move for a
simulated machine at the same bearing. You can also run `python3 main.py`
directly with `SIMULATION_MODE = True` to watch the loop live;
`SIM_SOURCE_CLASS` in config picks what the pretend source sounds like, and
`python3 channel_monitor.py --once` previews the level-visualisation tool
against simulated mics before ever touching real hardware.

---

## 6. Decisions already made (so you don't re-litigate them)

- **Free-running capture, measured rate.** A 200 kHz target failed in
  testing because it exceeds what the chip+Python can do; ANY fixed target
  is fragile the same way. The code now reads as fast as possible, measures
  the true rate, and feeds the measured number into the math. Nothing to
  configure, nothing to get wrong.
- **Clip length = 4096 samples** (~0.3–0.5 s depending on achieved rate):
  long enough for a tap or word, short enough for a responsive loop.
- **Continuous while-loop: yes** (`MAX_ITERATIONS = None`). The system's job
  is to keep listening. Set a number only for scripted tests.
- **Audio is resampled to 16 kHz before classification**, automatically —
  so the classifier works identically regardless of the Pi's achieved
  capture rate.
- **Servo settle pause (0.3 s) after each move** so the servo's own noise
  doesn't contaminate the next capture.
- **~180-degree servo = sector-scanning instrument.** Sources behind the
  array may be located but not fully turned to. Known, accepted, not a bug.
