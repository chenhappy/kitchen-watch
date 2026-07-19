# Kitchen Watch

A phone-based pickleball non-volley-zone ("kitchen") monitor. Place the phone on the
extension of the net line, calibrate the two kitchens with 6 taps, and the app beeps
whenever a foot lands in a kitchen — a different sound for each side of the net.

- Runs entirely in the browser (Safari on iPhone, Chrome on Android). No install, no server.
- Pose detection: TensorFlow.js MoveNet (MultiPose Lightning), on-device.
- Sounds are synthesized with the Web Audio API — side A is a low double-beep,
  side B a single high beep. Swap or mute from the bottom bar.
- Calibration is saved on the phone, so repeat visits to the same court are one tap.

## Use

1. Open the site over **https** (camera access requires it).
2. Tap **Start camera**, allow camera access, wait for the model to load.
3. Tap the 6 points in the order shown (net near/far ends, then the outer edge of each
   kitchen line's near and far corners). Drag dots to fine-tune. Tap **Done**.
4. Prop the phone on a tripod on the net extension line, landscape, Low Power Mode off.

## Files

- `index.html` — page, styles, script tags
- `app.js` — all logic (camera, calibration, detection, sounds)
