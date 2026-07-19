"use strict";

/* ============================================================
   Kitchen Watch v2 — pickleball non-volley-zone monitor
   Detection = fusion of two signals:
     1) Shoe locations from pose tracking (ankles + shin-extended
        sole points), so paddles/balls/shadows can never beep.
     2) Ground-contact confirmation:
        - ON the line: the kitchen line is sampled into small
          patches; a patch changing appearance (shadow-rejected)
          right where a shoe is = a real line touch.
        - INSIDE the kitchen: the shoe must be planted (near-zero
          velocity for a few frames), so airborne feet passing
          over the zone stay silent.
   Every beep drops an evidence marker at the exact contact spot.
   ============================================================ */

// ---------- Tuning constants (edit these after court testing) ----------
const BEEP_COOLDOWN_MS = 2200;  // min time between beeps per side
const MIN_KP_SCORE     = 0.35;  // pose keypoint confidence threshold
const SHIN_EXTEND      = 0.35;  // knee→ankle extension to reach the sole

const N_PATCHES        = 32;    // line patches per kitchen line
const PROC_W           = 320;   // width of the small processing frame
const LEARN_FRAMES     = 45;    // frames used to learn line appearance
const EMA_ALPHA        = 0.02;  // how fast the baseline adapts to light
const PATCH_STREAK     = 2;     // consecutive frames a patch must differ
const CHROMA_T         = 0.10;  // chromaticity change = real occlusion
const DARK_RATIO       = 0.40;  // darker than this ratio = occlusion too
const BRIGHT_RATIO     = 1.70;  // brighter than this = occlusion
const MASS_FRAC        = 0.60;  // >60% patches occluded = light change,
const MASS_FRAMES      = 30;    //   sustained → auto-relearn, no beeps

const PROX_FRAC        = 0.30;  // shoe-to-patch distance to confirm touch,
                                //   as fraction of local kitchen depth
const PLANT_MOVE_FRAC  = 0.05;  // "planted" = per-frame movement below this
                                //   fraction of local kitchen depth
const PLANT_STREAK     = 3;     // pose frames a planted foot must persist

const STORAGE_KEY = "kw_calibration_v1";

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
const debugBtn   = document.getElementById("debugBtn");
const relearnBtn = document.getElementById("relearnBtn");
const banner     = document.getElementById("banner");

// ---------- State ----------
const S = {
  screen: "start",              // start | calibrate | watch
  points: [],                   // 6 calibration points, video pixel coords
  dragIndex: -1,
  soundsSwapped: false,
  muted: false,
  debug: false,
  lastBeep: { left: 0, right: 0 },
  flashUntil: { left: 0, right: 0 },
  feet: [],                     // {x, y, speed} — ankles + sole estimates
  prevFeet: [],
  plantStreak: { left: 0, right: 0 },
  events: [],                   // evidence markers {x, y, until, label}
  learning: 0,                  // frames of baseline learning remaining
  massCount: { left: 0, right: 0 },
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

let detector = null;
let wakeLock = null;
let audioCtx = null;

// Small processing frame for line-patch pixel sampling
const proc = document.createElement("canvas");
const pctx = proc.getContext("2d", { willReadFrequently: true });
let procScale = 1, procImg = null;

// patches.left / patches.right: arrays of
// { cx, cy (video px), t (0..1 along line), rad (video px),
//   base [r,g,b] | null, cur [r,g,b], state: 0 clear|1 shadow|2 occluded,
//   streak, acc [r,g,b,count] }
const patches = { left: [], right: [] };

// ---------- Geometry helpers ----------
const lerp = (a, b, t) => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

function leftPoly()  { const p = S.points; return [p[0], p[1], p[3], p[2]]; }
function rightPoly() { const p = S.points; return [p[0], p[1], p[5], p[4]]; }
function polyFor(side) { return side === "left" ? leftPoly() : rightPoly(); }
function calibrated() { return S.points.length === 6; }

function pointInPoly(pt, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x, yi = poly[i].y, xj = poly[j].x, yj = poly[j].y;
    if (((yi > pt.y) !== (yj > pt.y)) &&
        (pt.x < (xj - xi) * (pt.y - yi) / (yj - yi) + xi)) inside = !inside;
  }
  return inside;
}

// Project a point onto the net line (p0→p1) to get a 0..1 depth parameter.
function projT(pt) {
  const p = S.points, dx = p[1].x - p[0].x, dy = p[1].y - p[0].y;
  const L2 = dx * dx + dy * dy;
  if (!L2) return 0;
  return Math.max(0, Math.min(1, ((pt.x - p[0].x) * dx + (pt.y - p[0].y) * dy) / L2));
}

// Local kitchen depth (net → kitchen line) in pixels at parameter t.
// This is the perspective-aware yardstick everything scales against.
function localDepth(side, t) {
  const p = S.points;
  const net  = lerp(p[0], p[1], t);
  const line = side === "left" ? lerp(p[2], p[3], t) : lerp(p[4], p[5], t);
  return dist(net, line);
}

function clientToVideo(cx, cy) {
  const r = video.getBoundingClientRect();
  const vr = canvas.width / canvas.height;
  const er = r.width / r.height;
  let w, h, ox, oy;
  if (er > vr) { h = r.height; w = h * vr; ox = r.left + (r.width - w) / 2; oy = r.top; }
  else         { w = r.width;  h = w / vr; ox = r.left; oy = r.top + (r.height - h) / 2; }
  return { x: (cx - ox) / w * canvas.width, y: (cy - oy) / h * canvas.height };
}

// ---------- Audio ----------
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
  // Wait for dimensions — but don't hang if metadata already fired (race)
  // or never fires (safety timeout).
  if (video.readyState < 1 || !video.videoWidth) {
    await new Promise((res) => {
      const t = setTimeout(res, 4000);
      video.onloadedmetadata = () => { clearTimeout(t); res(); };
    });
  }
  try { await video.play(); } catch (_) { /* autoplay quirks; stream still runs */ }
  if (!video.videoWidth) throw new Error("camera stream has no video size");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  procScale = PROC_W / canvas.width;
  proc.width = PROC_W;
  proc.height = Math.round(canvas.height * procScale);
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

// ---------- Wake lock ----------
async function requestWakeLock() {
  try {
    if ("wakeLock" in navigator) wakeLock = await navigator.wakeLock.request("screen");
  } catch (_) {}
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
    DeviceMotionEvent.requestPermission()
      .then((st) => { if (st === "granted") window.addEventListener("devicemotion", handler); })
      .catch(() => {});
  } else if (typeof DeviceMotionEvent !== "undefined") {
    window.addEventListener("devicemotion", handler);
  }
}

let bannerTimer = null;
function showBanner(text, ms = 6000) {
  banner.textContent = text;
  banner.classList.remove("hidden");
  clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => banner.classList.add("hidden"), ms);
}
banner.addEventListener("click", () => banner.classList.add("hidden"));

// Surface any crash on screen so silent freezes become readable bug reports.
window.addEventListener("error", (e) => {
  showBanner("Error: " + (e.message || "unknown") +
    (e.lineno ? " (line " + e.lineno + ")" : ""), 15000);
});
window.addEventListener("unhandledrejection", (e) => {
  const r = e.reason;
  showBanner("Error: " + (r && r.message ? r.message : String(r)), 15000);
});

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

// ---------- Line patches ----------
function buildPatches() {
  const p = S.points;
  for (const side of ["left", "right"]) {
    const a = side === "left" ? p[2] : p[4];
    const b = side === "left" ? p[3] : p[5];
    const arr = [];
    for (let i = 0; i < N_PATCHES; i++) {
      const t = (i + 0.5) / N_PATCHES;
      const c = lerp(a, b, t);
      const rad = Math.max(3, Math.min(18, localDepth(side, t) * 0.06));
      arr.push({ cx: c.x, cy: c.y, t, rad, base: null, cur: [0, 0, 0],
                 state: 0, streak: 0, acc: [0, 0, 0, 0] });
    }
    patches[side] = arr;
  }
}

function startLearning() {
  for (const side of ["left", "right"]) {
    for (const pa of patches[side]) { pa.base = null; pa.acc = [0, 0, 0, 0]; pa.state = 0; pa.streak = 0; }
  }
  S.learning = LEARN_FRAMES;
  showBanner("Learning line appearance — keep the kitchen lines clear for 2 seconds.", 3000);
}

// Mean RGB of a patch from the small processing frame.
function samplePatch(pa) {
  const img = procImg;
  if (!img) return null;
  const sx = Math.round(pa.cx * procScale);
  const sy = Math.round(pa.cy * procScale);
  const rs = Math.max(1, Math.round(pa.rad * procScale));
  let r = 0, g = 0, b = 0, n = 0;
  for (let y = sy - rs; y <= sy + rs; y++) {
    if (y < 0 || y >= proc.height) continue;
    for (let x = sx - rs; x <= sx + rs; x++) {
      if (x < 0 || x >= proc.width) continue;
      const k = (y * proc.width + x) * 4;
      r += img.data[k]; g += img.data[k + 1]; b += img.data[k + 2]; n++;
    }
  }
  return n ? [r / n, g / n, b / n] : null;
}

// Classify a patch: 0 clear, 1 shadow (darker, same color), 2 occluded.
function classifyPatch(cur, base) {
  const sB = base[0] + base[1] + base[2] + 1e-6;
  const sC = cur[0] + cur[1] + cur[2] + 1e-6;
  const chroma =
    Math.abs(cur[0] / sC - base[0] / sB) +
    Math.abs(cur[1] / sC - base[1] / sB) +
    Math.abs(cur[2] / sC - base[2] / sB);
  const ratio = sC / sB;
  if (chroma > CHROMA_T) return 2;                          // color replaced → object
  if (ratio < DARK_RATIO || ratio > BRIGHT_RATIO) return 2; // extreme light block
  if (ratio < 0.9) return 1;                                // dimmer, same color → shadow
  return 0;
}

function footNear(pt, maxD) {
  for (const f of S.feet) if (dist(f, pt) <= maxD) return true;
  return false;
}

// Runs every display frame: sample all patches, learn or classify.
function processPatches(now) {
  pctx.drawImage(video, 0, 0, proc.width, proc.height);
  procImg = pctx.getImageData(0, 0, proc.width, proc.height);

  const learning = S.learning > 0;
  for (const side of ["left", "right"]) {
    let occluded = 0;
    for (const pa of patches[side]) {
      const cur = samplePatch(pa);
      if (!cur) continue;
      pa.cur = cur;
      if (learning) {
        pa.acc[0] += cur[0]; pa.acc[1] += cur[1]; pa.acc[2] += cur[2]; pa.acc[3]++;
        continue;
      }
      if (!pa.base) continue;
      const st = classifyPatch(cur, pa.base);
      pa.streak = st === 2 ? pa.streak + 1 : 0;
      pa.state = st;
      if (st === 2) occluded++;
      // Slowly adapt the baseline to changing light — but only when the
      // patch is clear and no shoe is nearby (never learn a shoe as court).
      if (st === 0 && !footNear(pa, localDepth(side, pa.t) * PROX_FRAC)) {
        pa.base[0] += (cur[0] - pa.base[0]) * EMA_ALPHA;
        pa.base[1] += (cur[1] - pa.base[1]) * EMA_ALPHA;
        pa.base[2] += (cur[2] - pa.base[2]) * EMA_ALPHA;
      }
    }
    // Most of the line "occluded" at once = lighting/camera change, not shoes.
    if (!learning) {
      const frac = occluded / N_PATCHES;
      S.massCount[side] = frac > MASS_FRAC ? S.massCount[side] + 1 : 0;
      if (S.massCount[side] > MASS_FRAMES) {
        S.massCount = { left: 0, right: 0 };
        startLearning();
        showBanner("Lighting changed a lot — relearning the lines.", 3000);
      }
    }
  }

  if (learning) {
    S.learning--;
    if (S.learning === 0) {
      for (const side of ["left", "right"]) {
        for (const pa of patches[side]) {
          if (pa.acc[3] > 0) pa.base = [pa.acc[0] / pa.acc[3], pa.acc[1] / pa.acc[3], pa.acc[2] / pa.acc[3]];
        }
      }
      showBanner("Lines learned — watching.", 2000);
    }
  }
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
  buildPatches();
  startLearning();
  S.plantStreak = { left: 0, right: 0 };
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
      `All 6 points set. Drag dots so the kitchen-line dots sit ON the painted line` +
      ` (outer edge) — the line monitor samples exactly there. Then press Done.`;
  }
  undoBtn.disabled = n === 0;
  doneBtn.disabled = n < 6;
}

// ---------- Touch handling for calibration ----------
function nearestPointIndex(v) {
  let best = -1, bestD = 1e12;
  const grabRadius = canvas.width * 0.035;
  S.points.forEach((p, i) => {
    const d = dist(p, v);
    if (d < bestD) { bestD = d; best = i; }
  });
  return bestD <= grabRadius ? best : -1;
}
canvas.addEventListener("pointerdown", (e) => {
  if (S.screen !== "calibrate") return;
  const v = clientToVideo(e.clientX, e.clientY);
  const idx = nearestPointIndex(v);
  if (idx >= 0) S.dragIndex = idx;
  else if (S.points.length < 6) {
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
  ensureAudio();
  try {
    loadmsg.innerHTML = `<span class="spin"></span>Starting camera…`;
    await startCamera();
    loadmsg.innerHTML = `<span class="spin"></span>Loading the AI model (first time takes ~10–20 s)…`;
    await loadModel();
    loadmsg.textContent = "";
    setupMotionWarning();
    const saved = loadCalibration();
    if (saved) {
      S.points = saved;
      enterCalibration();
      showBanner("Loaded saved calibration — adjust the dots if needed, or tap Start over.", 5000);
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
debugBtn.addEventListener("click", () => {
  S.debug = !S.debug;
  debugBtn.textContent = S.debug ? "Debug ✓" : "Debug";
});
relearnBtn.addEventListener("click", () => startLearning());

// ---------- Pose detection ----------
async function detectFeet() {
  const poses = await detector.estimatePoses(video, { maxPoses: 6, flipHorizontal: false });
  const feet = [];
  for (const pose of poses) {
    const kps = {};
    for (const kp of pose.keypoints) kps[kp.name] = kp;
    for (const side of ["left", "right"]) {
      const ankle = kps[side + "_ankle"];
      if (!ankle || ankle.score < MIN_KP_SCORE) continue;
      feet.push({ x: ankle.x, y: ankle.y });
      const knee = kps[side + "_knee"];
      if (knee && knee.score >= MIN_KP_SCORE) {
        feet.push({
          x: ankle.x + (ankle.x - knee.x) * SHIN_EXTEND,
          y: ankle.y + (ankle.y - knee.y) * SHIN_EXTEND,
        });
      }
    }
  }
  // Per-foot speed: distance to the nearest foot in the previous pose frame.
  for (const f of feet) {
    let d = Infinity;
    for (const pf of S.prevFeet) d = Math.min(d, dist(f, pf));
    f.speed = S.prevFeet.length ? d : Infinity;
  }
  S.prevFeet = feet;
  return feet;
}

// ---------- Violation logic (fusion) ----------
function fireViolation(side, spot, label, now) {
  if (now - S.lastBeep[side] < BEEP_COOLDOWN_MS) return;
  S.lastBeep[side] = now;
  S.flashUntil[side] = now + 600;
  S.events.push({ x: spot.x, y: spot.y, until: now + 3000, label });
  if (S.events.length > 6) S.events.shift();
  playSound(soundForSide(side));
}

function checkViolations(now) {
  if (S.learning > 0) return;

  for (const side of ["left", "right"]) {
    // Skip while the scene is globally disturbed (lighting/camera event).
    if (S.massCount[side] > 3) continue;

    // --- Trigger 1: line touch ---
    // A patch persistently changed AND a shoe is at that spot.
    for (const pa of patches[side]) {
      if (pa.state === 2 && pa.streak >= PATCH_STREAK && pa.base) {
        const prox = localDepth(side, pa.t) * PROX_FRAC;
        if (footNear(pa, prox)) {
          fireViolation(side, pa, "line touch", now);
          break;
        }
      }
    }

    // --- Trigger 2: planted inside the kitchen ---
    // A shoe inside the polygon moving slower than the planted threshold.
    const poly = polyFor(side);
    let plantedFoot = null;
    for (const f of S.feet) {
      if (!pointInPoly(f, poly)) continue;
      const limit = localDepth(side, projT(f)) * PLANT_MOVE_FRAC;
      if (f.speed <= limit) { plantedFoot = f; break; }
    }
    S.plantStreak[side] = plantedFoot ? S.plantStreak[side] + 1 : 0;
    if (plantedFoot && S.plantStreak[side] >= PLANT_STREAK) {
      fireViolation(side, plantedFoot, "in kitchen", now);
    }
  }
}

// ---------- Drawing ----------
function hexA(hex, a) {
  const r = parseInt(hex.slice(1, 3), 16),
        g = parseInt(hex.slice(3, 5), 16),
        b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${a})`;
}
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

function draw(now) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const sc = canvas.width / 1280;
  const p = S.points;

  if (p.length >= 4 && p[2] && p[3]) {
    const flash = now < S.flashUntil.left;
    drawPoly([p[0], p[1], p[3], p[2]], flash ? "#ff2d2d" : COLORS.left, flash ? 0.4 : 0.12, 3 * sc);
  }
  if (p.length >= 6 && p[4] && p[5]) {
    const flash = now < S.flashUntil.right;
    drawPoly([p[0], p[1], p[5], p[4]], flash ? "#ff2d2d" : COLORS.right, flash ? 0.4 : 0.12, 3 * sc);
  }

  if (S.screen === "calibrate") {
    S.points.forEach((pt, i) => {
      const c = i < 2 ? COLORS.accent : i < 4 ? COLORS.left : COLORS.right;
      ctx.beginPath(); ctx.arc(pt.x, pt.y, 14 * sc, 0, Math.PI * 2);
      ctx.fillStyle = hexA(c, 0.35); ctx.fill();
      ctx.lineWidth = 3 * sc; ctx.strokeStyle = c; ctx.stroke();
      ctx.beginPath(); ctx.arc(pt.x, pt.y, 3 * sc, 0, Math.PI * 2);
      ctx.fillStyle = c; ctx.fill();
      ctx.fillStyle = "#ffffff";
      ctx.font = `bold ${20 * sc}px sans-serif`;
      ctx.fillText(String(i + 1), pt.x + 18 * sc, pt.y - 10 * sc);
    });
  }

  if (S.screen === "watch") {
    // Line patches: dots on the kitchen lines. Debug shows their state.
    for (const side of ["left", "right"]) {
      for (const pa of patches[side]) {
        let color, r;
        if (S.learning > 0) { color = "rgba(245,181,46,0.8)"; r = 3 * sc; }
        else if (S.debug) {
          color = pa.state === 2 ? "rgba(255,45,45,0.95)"
               : pa.state === 1 ? "rgba(245,181,46,0.9)"
               : "rgba(74,222,128,0.8)";
          r = pa.state === 2 ? 6 * sc : 4 * sc;
        } else if (pa.state === 2 && pa.streak >= PATCH_STREAK) {
          color = "rgba(255,45,45,0.95)"; r = 6 * sc;
        } else continue;
        ctx.beginPath(); ctx.arc(pa.cx, pa.cy, r, 0, Math.PI * 2);
        ctx.fillStyle = color; ctx.fill();
      }
    }

    // Feet
    for (const f of S.feet) {
      ctx.beginPath(); ctx.arc(f.x, f.y, 8 * sc, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(74,222,128,0.85)"; ctx.fill();
      ctx.lineWidth = 2.5 * sc; ctx.strokeStyle = "#0d0f12"; ctx.stroke();
      if (S.debug && isFinite(f.speed)) {
        ctx.fillStyle = "#ffffff";
        ctx.font = `${13 * sc}px sans-serif`;
        ctx.fillText(f.speed.toFixed(0), f.x + 10 * sc, f.y - 8 * sc);
      }
    }

    // Evidence markers at contact spots
    S.events = S.events.filter((ev) => ev.until > now);
    for (const ev of S.events) {
      ctx.beginPath(); ctx.arc(ev.x, ev.y, 20 * sc, 0, Math.PI * 2);
      ctx.lineWidth = 4 * sc; ctx.strokeStyle = "#ff2d2d"; ctx.stroke();
      ctx.fillStyle = "#ff2d2d";
      ctx.font = `bold ${17 * sc}px sans-serif`;
      ctx.fillText(ev.label, ev.x + 24 * sc, ev.y + 5 * sc);
    }
  }
}

// ---------- Main loop ----------
let lastFpsT = performance.now(), frames = 0, detBusy = false;

function loop() {
  const now = performance.now();

  if (S.screen === "watch") {
    processPatches(now);
    if (detector && !detBusy) {
      detBusy = true;
      detectFeet()
        .then((feet) => { S.feet = feet; })
        .catch(() => {})
        .finally(() => { detBusy = false; });
    }
    checkViolations(now);
  }

  draw(now);

  frames++;
  if (now - lastFpsT >= 1000) {
    S.fps = frames; frames = 0; lastFpsT = now;
    if (S.screen === "watch") {
      statusEl.innerHTML = S.learning > 0
        ? `<span style="color:#f5b52e">●</span> Learning lines…`
        : `<span class="dot">●</span> Watching · ${S.fps} fps`;
    }
  }
  requestAnimationFrame(loop);
}
