# AI SIMON — stable touch / immediate score revision

## Rules
- Round 1: 3 selections, 20 total points (7 + 7 + 6)
- Round 2: 5 selections, 30 total points (6 each)
- Round 3: 8 selections, 50 total points (7 + 7 + 6 + 6 + 6 + 6 + 6 + 6)
- Correct input adds points immediately. Complete all rounds for 100/100.
- Remember the previous round sequence, which is extended by 2, then by 3 colors.

## Run at home (PowerShell inside Project01)

```powershell
.\.venv\Scripts\python.exe main.py --mode laptop --verbose
```

Remove hands and press C to capture the **stationary** four-color board, then SPACE to start. During YOUR TURN, aim index fingertip at the center of a color for 0.35 seconds, then withdraw into the black gap for 0.22 seconds, or take the whole hand away for 0.50 seconds. Temporary hand tracking gaps below 0.20 seconds no longer reset a press.

## Isolate scoring vs camera recognition

```powershell
.\.venv\Scripts\python.exe main.py --mode laptop --debug-input
```

After calibration and SPACE, during YOUR TURN press keyboard 1=RED, 2=BLUE, 3=GREEN, 4=YELLOW to test the scoring logic separately. This debug input is **not part of the live gesture demo**. Ordinary execution does not enable these keys.

## Jetson
- Default camera remains `jetson_usb`, with existing GStreamer code untouched.
- Run `python main.py --mode jetson_usb` using a tested Jetson Python environment and existing MediaPipe task model.
- Do not copy a Windows `.venv` into Jetson.
- Use the same stable camera and paper-board angle for calibration and gameplay.
- `models/hand_landmarker.task` should be provided separately.
