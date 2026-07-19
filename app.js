"use strict";

/* ============================================================
   Kitchen Watch — pickleball non-volley-zone monitor
   Camera on the net extension line, looking along the net.
   Left/right kitchens are calibrated with 6 tapped points,
   feet are tracked with MoveNet, and each side gets its own beep.
   ============================================================ */

// ---------- DOM ----------
const video   = document.getElementById("cam");
const canvas  = document.getElementById("overlay");
const ctx     = canvas.getContext("2d");
const startPanel = document.getElementById("startPanel");
const startBtn   = document.getElementById("startBtn");
const loadmsg    = document.getElementById("loadmsg");
const calibbar   = document.getElementById("calibbar");
const calibmsg   = document.getElementById("calibmsg");
const undoBtn    = document.getElementById("undoBtn");
const clearBtn   = document.getElementById("clearBtn");
const doneBtn    = document.getElementById("doneBtn");
const topbar     = document.getElementById("topbar");
const botbar     = document.getElementById("botbar");
const statusEl   = document.getElementById("status");
const recalBtn   = document.getElementById("recalBtn");
const swapBtn    = document.getElementById("swapBtn");
const muteBtn    = document.getElementById("muteBtn");
const testBtn    = document.getElementById("testBtn");
const banner     = document.getElementById("banner");

// ---------- State ----------
const S = {
  screen: "start",            // start | calibrate | watch
  points: [],                 // up to 6 calibration points in VIDEO pixel coords
  dragIndex: -1,
  soundsSwapped: false,
  muted: false,
  lastBeep: { left: 0, right: 0 },
  flashUntil: { left: 0, right: 0 },
  streak: { left: 0, right: 0 },
  feet: [],                   // ankle points from the last pose pass
  fps: 0,
};

const POINT_LABELS = [
  "NET line — NEAR end (bottom of the net line, closest to you)",
  "NET line — FAR end (where the net meets the far sideline)",
  "LEFT kitchen line — NEAR corner (tap the OUTER edge of the line)",
  "LEFT kitchen line — FAR corner (outer edge)",
  "RIGHT kitchen line — NEAR corner (outer edge)",
  "RIGHT kitchen line — FAR corner (outer edge)",
];

const COLORS = { left: "#ff8a3d", right: "#4dc9ff", accent: "#f5b52e" };
const BEEP_COOLDOWN_MS = 2200;   // min time between beeps per side
const STREAK_FRAMES    = 2;      // consecutive frames required (debounce)
const MIN_KP_SCORE     = 0.35;   // ankle keypoint confidence threshold
const STORAGE_KEY      = "kw_calibration_v1";

let detector = null;
let wakeLock = null;
let audioCtx = null;

// ---------- Kitchen polygons from the 6 points ----------
// p0 net-near, p1 net-far, p2 leftLine-near, p3 leftLine-far,
// p4 rightLine-near, p5 rightLine-far
function leftPoly()  { const p = S.points; return [p[0], p[1], p[3], p[2]]; }
function rightPoly() { const p = S.points; return [p[0], p[1], p[5], p[4]]; }
function calibrated() { return S.points.length === 6; }

// ---------- Geometry ----------
function pointInPoly(pt, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x, yi = poly[i].y, xj = poly[j].x, yj = poly[j].y;
    if (((yi > pt.y) !== (yj > pt.y)) &&
        (pt.x < (xj - xi) * (pt.y - yi) / (yj - yi) + xi)) inside = !inside;
  }
  return inside;
}

// Map a touch/click (client coords) to video pixel coords,
// accounting for object-fit: contain letterboxing.
function clientToVideo(cx, cy) {
  const r = video.getBoundingClientRect();
  const vr = canvas.width / canvas.height;
  const er = r.width / r.height;
  let w, h, ox, oy;
  if (er > vr) { h = r.height; w = h * vr; ox = r.left + (r.width - w) / 2; oy = r.top; }
  else         { w = r.width;  h = w / vr; ox = r.left; oy = r.top + (r.height - h) / 2; }
  return { x: (cx - ox) / w * canvas.width, y: (cy - oy) / h * canvas.height };
}

// ---------- Audio (synthesized beeps, no sound files needed) ----------
function ensureAudio() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") audioCtx.resume();
}

function tone(freq, when, dur, vol) {
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = "square";
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0, when);
  gain.gain.linearRampToValueAtTime(vol, when + 0.01);
  gain.gain.setValueAtTime(vol, when + dur - 0.03);
  gain.gain.linearRampToValueAtTime(0, when + dur);
  osc.connect(gain).connect(audioCtx.destination);
  osc.start(when);
  osc.stop(when + dur + 0.02);
}

// Sound A: two short LOW beeps.  Sound B: one long HIGH beep.
function playSound(which) {
  if (S.muted) return;
  ensureAudio();
  const t = audioCtx.currentTime;
  if (which === "A") { tone(330, t, 0.15, 0.6); tone(330, t + 0.22, 0.15, 0.6); }
  else               { tone(950, t, 0.45, 0.5); }
}

function soundForSide(side) {
  const aSide = S.soundsSwapped ? "right" : "left";
  return side === aSide ? "A" : "B";
}

// ---------- Camera / model ----------
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((res) => { video.onloadedmetadata = res; });
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

async function loadModel() {
  await tf.setBackend("webgl");
  await tf.ready();
  detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet,
    {
      modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING,
      enableTracking: true,
      trackerType: poseDetection.TrackerType.BoundingBox,
    }
  );
}

// ---------- Wake lock (keep screen on) ----------
async function requestWakeLock() {
  try {
    if ("wakeLock" in navigator) {
      wakeLock = await navigator.wakeLock.request("screen");
    }
  } catch (_) { /* not critical */ }
}
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && S.screen === "watch") requestWakeLock();
});

// ---------- Motion bump warning ----------
let motionBaseline = null;
function setupMotionWarning() {
  const handler = (e) => {
    const a = e.accelerationIncludingGravity;
    if (!a) return;
    const mag = Math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    if (motionBaseline === null) { motionBaseline = mag; return; }
    motionBaseline = motionBaseline * 0.98 + mag * 0.02;
    if (Math.abs(mag - motionBaseline) > 4 && S.screen === "watch") {
      showBanner("Phone may have moved — tap Recalibrate to check the court lines.");
    }
  };
  if (typeof DeviceMotionEvent !== "undefined" &&
      typeof DeviceMotionEvent.requestPermission === "function") {
    // iOS: must be requested from a user gesture; failure is fine.
    DeviceMotionEvent.requestPermission()
      .then((st) => { if (st === "granted") window.addEventListener("devicemotion", handler); })
      .catch(() => {});
  } else if (typeof DeviceMotionEvent !== "undefined") {
    window.addEventListener("devicemotion", handler);
  }
}

let bannerTimer = null;
function showBanner(text) {
  banner.textContent = text;
  banner.classList.remove("hidden");
  clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => banner.classList.add("hidden"), 6000);
}
banner.addEventListener("click", () => banner.classList.add("hidden"));

// ---------- Calibration storage ----------
function saveCalibration() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(S.points)); } catch (_) {}
}
function loadCalibration() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const pts = JSON.parse(raw);
    return Array.isArray(pts) && pts.length === 6 ? pts : null;
  } catch (_) { return null; }
}

// ---------- Screens ----------
function enterCalibration() {
  S.screen = "calibrate";
  startPanel.classList.add("hidden");
  topbar.classList.add("hidden");
  botbar.classList.add("hidden");
  calibbar.classList.remove("hidden");
  updateCalibMsg();
}

function enterWatch() {
  S.screen = "watch";
  saveCalibration();
  calibbar.classList.add("hidden");
  topbar.classList.remove("hidden");
  botbar.classList.remove("hidden");
  requestWakeLock();
}

function updateCalibMsg() {
  const n = S.points.length;
  if (n < 6) {
    calibmsg.innerHTML =
      `<span class="step">Point ${n + 1} of 6:</span> tap the ${POINT_LABELS[n]}.` +
      ` &nbsp;Drag any dot to fine-tune.`;
  } else {
    calibmsg.innerHTML =
      `All 6 points set. Drag dots to fine-tune so the shaded areas hug the painted lines` +
      ` (tap the <b>outer</b> edge of lines so the lines count as inside). Then press Done.`;
  }
  undoBtn.disabled = n === 0;
  doneBtn.disabled = n < 6;
}

// ---------- Touch handling for calibration ----------
function nearestPointIndex(v) {
  let best = -1, bestD = 1e12;
  const grabRadius = canvas.width * 0.035; // generous finger-sized grab area
  S.points.forEach((p, i) => {
    const d = Math.hypot(p.x - v.x, p.y - v.y);
    if (d < bestD) { bestD = d; best = i; }
  });
  return bestD <= grabRadius ? best : -1;
}

canvas.addEventListener("pointerdown", (e) => {
  if (S.screen !== "calibrate") return;
  const v = clientToVideo(e.clientX, e.clientY);
  const idx = nearestPointIndex(v);
  if (idx >= 0) {
    S.dragIndex = idx;
  } else if (S.points.length < 6) {
    S.points.push(v);
    S.dragIndex = S.points.length - 1;
    updateCalibMsg();
  }
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  if (S.screen !== "calibrate" || S.dragIndex < 0) return;
  const v = clientToVideo(e.clientX, e.clientY);
  v.x = Math.max(0, Math.min(canvas.width, v.x));
  v.y = Math.max(0, Math.min(canvas.height, v.y));
  S.points[S.dragIndex] = v;
});
canvas.addEventListener("pointerup", () => { S.dragIndex = -1; });
canvas.addEventListener("pointercancel", () => { S.dragIndex = -1; });

// ---------- Buttons ----------
startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  ensureAudio(); // unlock audio inside the user gesture
  try {
    loadmsg.innerHTML = `<span class="spin"></span>Starting camera…`;
    await startCamera();
    loadmsg.innerHTML = `<span class="spin"></span>Loading the AI model (first time takes ~10–20 s)…`;
    await loadModel();
    loadmsg.textContent = "";
    setupMotionWarning();

    const saved = loadCalibration();
    if (saved && confirm("Use the saved court calibration from last time?")) {
      S.points = saved;
      enterCalibration();     // show it for a quick visual check
      updateCalibMsg();
    } else {
      S.points = [];
      enterCalibration();
    }
    requestAnimationFrame(loop);
  } catch (err) {
    startBtn.disabled = false;
    loadmsg.textContent =
      "Could not start: " + (err && err.message ? err.message : err) +
      " — check that camera access is allowed and you are on an https:// address.";
  }
});

undoBtn.addEventListener("click", () => { S.points.pop(); updateCalibMsg(); });
clearBtn.addEventListener("click", () => { S.points = []; updateCalibMsg(); });
doneBtn.addEventListener("click", () => { if (calibrated()) enterWatch(); });
recalBtn.addEventListener("click", () => enterCalibration());
swapBtn.addEventListener("click", () => {
  S.soundsSwapped = !S.soundsSwapped;
  showBanner("Sounds swapped: " +
    (S.soundsSwapped ? "LEFT = high beep, RIGHT = low double-beep"
                     : "LEFT = low double-beep, RIGHT = high beep"));
});
muteBtn.addEventListener("click", () => {
  S.muted = !S.muted;
  muteBtn.textContent = S.muted ? "Unmute" : "Mute";
});
testBtn.addEventListener("click", () => {
  playSound(soundForSide("left"));
  setTimeout(() => playSound(soundForSide("right")), 900);
});

// ---------- Detection + violation logic ----------
async function detectFeet() {
  const poses = await detector.estimatePoses(video, { maxPoses: 6, flipHorizontal: false });
  const feet = [];
  for (const pose of poses) {
    for (const kp of pose.keypoints) {
      if ((kp.name === "left_ankle" || kp.name === "right_ankle") && kp.score >= MIN_KP_SCORE) {
        feet.push({ x: kp.x, y: kp.y });
      }
    }
  }
  return feet;
}

function checkViolations(now) {
  const polys = { left: leftPoly(), right: rightPoly() };
  for (const side of ["left", "right"]) {
    const hit = S.feet.some((f) => pointInPoly(f, polys[side]));
    S.streak[side] = hit ? S.streak[side] + 1 : 0;
    if (S.streak[side] >= STREAK_FRAMES && now - S.lastBeep[side] > BEEP_COOLDOWN_MS) {
      S.lastBeep[side] = now;
      S.flashUntil[side] = now + 600;
      playSound(soundForSide(side));
    }
  }
}

// ---------- Drawing ----------
function drawPoly(poly, color, fillAlpha, lineW) {
  if (poly.some((p) => !p)) return;
  ctx.beginPath();
  poly.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
  ctx.closePath();
  ctx.fillStyle = hexA(color, fillAlpha);
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineW;
  ctx.stroke();
}

function hexA(hex, a) {
  const r = parseInt(hex.slice(1, 3), 16),
        g = parseInt(hex.slice(3, 5), 16),
        b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${a})`;
}

function draw(now) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const sc = canvas.width / 1280; // scale UI elements to resolution

  // Kitchen polygons
  if (S.points.length >= 4) {
    const p = S.points;
    if (p[2] && p[3]) {
      const flash = now < S.flashUntil.left;
      drawPoly([p[0], p[1], p[3], p[2]], flash ? "#ff2d2d" : COLORS.left, flash ? 0.4 : 0.14, 3 * sc);
    }
    if (p[4] && p[5]) {
      const flash = now < S.flashUntil.right;
      drawPoly([p[0], p[1], p[5], p[4]], flash ? "#ff2d2d" : COLORS.right, flash ? 0.4 : 0.14, 3 * sc);
    }
  }

  // Calibration points
  if (S.screen === "calibrate") {
    S.points.forEach((p, i) => {
      const c = i < 2 ? COLORS.accent : i < 4 ? COLORS.left : COLORS.right;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 14 * sc, 0, Math.PI * 2);
      ctx.fillStyle = hexA(c, 0.35);
      ctx.fill();
      ctx.lineWidth = 3 * sc;
      ctx.strokeStyle = c;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3 * sc, 0, Math.PI * 2);
      ctx.fillStyle = c;
      ctx.fill();
      ctx.fillStyle = "#ffffff";
      ctx.font = `bold ${20 * sc}px sans-serif`;
      ctx.fillText(String(i + 1), p.x + 18 * sc, p.y - 10 * sc);
    });
  }

  // Feet
  if (S.screen === "watch") {
    for (const f of S.feet) {
      ctx.beginPath();
      ctx.arc(f.x, f.y, 9 * sc, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(74,222,128,0.85)";
      ctx.fill();
      ctx.lineWidth = 2.5 * sc;
      ctx.strokeStyle = "#0d0f12";
      ctx.stroke();
    }
  }
}

// ---------- Main loop ----------
let lastFpsT = performance.now(), frames = 0, detBusy = false;

function loop() {
  const now = performance.now();

  if (S.screen === "watch" && detector && !detBusy) {
    detBusy = true;
    detectFeet()
      .then((feet) => { S.feet = feet; checkViolations(performance.now()); })
      .catch(() => {})
      .finally(() => { detBusy = false; });
  }

  draw(now);

  frames++;
  if (now - lastFpsT >= 1000) {
    S.fps = frames;
    frames = 0;
    lastFpsT = now;
    if (S.screen === "watch") {
      statusEl.innerHTML = `<span class="dot">●</span> Watching · ${S.fps} fps`;
    }
  }
  requestAnimationFrame(loop);
}
