const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const RECORD_EPS = 1e-9;
const TEX = { "\\times": "×", "\\cdot": "·", "\\sim": "~", "\\approx": "≈", "\\le": "≤", "\\ge": "≥", "\\leq": "≤", "\\geq": "≥",
  "\\ne": "≠", "\\to": "→", "\\pm": "±", "\\infty": "∞", "\\langle": "⟨", "\\rangle": "⟩", "\\pi": "π", "\\theta": "θ", "\\epsilon": "ε",
  "\\varepsilon": "ε", "\\delta": "δ", "\\lambda": "λ", "\\mathrm": "", "\\text": "", "\\,": " ", "\\;": " ", "--": "–" };

// Model text uses light LaTeX ($10^{-7}$, \times); render it as readable HTML.
function prose(s) {
  let h = esc(s).replace(/\$/g, "");
  h = h.replace(/\\[a-zA-Z]+|\\[,;]|--/g, (m) => (m in TEX ? TEX[m] : m.slice(1)));
  h = h.replace(/\^\{([^}]*)\}/g, "<sup>$1</sup>").replace(/\^(-?\w)/g, "<sup>$1</sup>").replace(/_\{([^}]*)\}/g, "<sub>$1</sub>");
  return h.replace(/[{}]/g, "");
}

const state = {
  index: [], run: null, events: [], programs: new Map(), pos: 0, playing: false, speed: 4, timer: null,
  points: [], bestLine: [], best: null, recordPos: -1, pollTimer: null,
};

function modelName(m) {
  if (!m || m === "seed") return "Seed";
  if (m.startsWith("muse-wonder")) return "Muse Wonder";
  if (m.startsWith("muse-spark")) return "Muse Spark";
  return m;
}

// Positive = behind the published record, negative = ahead of it.
function behind(run, score) {
  if (run.best_known == null) return null;
  return run.maximize ? run.best_known - score : score - run.best_known;
}

function fmtBehind(run, score) {
  const d = behind(run, score);
  if (d == null) return "";
  const tol = RECORD_EPS * Math.max(1, Math.abs(run.best_known));
  if (d < -tol) return `<span class="goldtxt">beats the record by ${(-d).toExponential(2)}</span>`;
  if (Math.abs(d) <= tol) return `<span class="muted">ties the record</span>`;
  return `<span class="muted">${d.toExponential(2)} behind the record</span>`;
}

const isRecordRun = (r) => r.record_program != null;

function friendlyTitle(r) {
  const sc = /spherical_code_(\d+)d_(\d+)/.exec(r.problem || "");
  if (sc) return `${sc[2]} points on a sphere in ${sc[1]} dimensions`;
  const cp = /circle_packing_(\d+)/.exec(r.problem || "");
  if (cp) return `${cp[1]} circles in a square (warm-up test)`;
  return r.title;
}
const dataUrl = (run, rel) => `data/${encodeURIComponent(run)}/${rel}`;

async function loadIndex() {
  const res = await fetch("data/index.json", { cache: "no-store" });
  state.index = (await res.json()).runs;
  renderStats();
  renderRecords();
  const sel = $("runSelect");
  const ordered = [...state.index].sort((a, b) => (isRecordRun(b) - isRecordRun(a)) || (b.started - a.started));
  sel.innerHTML = ordered.map((r) => `<option value="${esc(r.name)}">${isRecordRun(r) ? "★ New record: " : ""}${esc(friendlyTitle(r))}</option>`).join("");
  const wanted = new URLSearchParams(location.search).get("run");
  const first = ordered.find((r) => r.name === wanted) || ordered[0];
  if (first) { sel.value = first.name; await selectRun(first.name); }
}

function renderStats() {
  const runs = state.index;
  const records = runs.filter(isRecordRun).length;
  const programs = runs.reduce((s, r) => s + r.programs, 0);
  $("stats").innerHTML = [
    [records, "new world records"],
    [programs.toLocaleString(), "programs written by the AI"],
    ["50 digits", "precision of every check"],
  ].map(([v, l]) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`).join("");
}

function renderRecords() {
  const recs = state.index.filter(isRecordRun).sort((a, b) => (a.gap_to_record ?? 0) - (b.gap_to_record ?? 0));
  $("records").innerHTML = recs.map((r) => `
    <div class="record" data-run="${esc(r.name)}">
      <div class="tag">NEW WORLD RECORD · VERIFIED</div>
      <h3>${esc(friendlyTitle(r))}</h3>
      <div class="row"><span>Published record</span><span>${r.best_known.toFixed(12)}</span></div>
      <div class="row"><span>Found by Muse</span><span class="new">${r.best_score.toFixed(12)}</span></div>
      <div class="row"><span>Improvement</span><span class="new">${Math.abs(r.gap_to_record).toExponential(2)}</span></div>
      <div class="row"><span>Attempts</span><span>${r.programs} programs</span></div>
      <div class="cta">Replay the discovery →</div>
    </div>`).join("") || `<div class="muted">No records yet.</div>`;
  document.querySelectorAll(".record").forEach((el) => el.addEventListener("click", async () => {
    $("runSelect").value = el.dataset.run;
    await selectRun(el.dataset.run);
    $("lab").scrollIntoView({ behavior: "smooth" });
    play();
  }));
}

function parseEvents(text) {
  return text.split("\n").filter(Boolean).map((l) => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
}

async function fetchRunData(name) {
  const [evText, progs] = await Promise.all([
    fetch(dataUrl(name, "events.jsonl"), { cache: "no-store" }).then((r) => r.text()),
    fetch(dataUrl(name, "programs.json"), { cache: "no-store" }).then((r) => r.json()),
  ]);
  return { events: parseEvents(evText), programs: new Map(progs.map((p) => [p.id, p])) };
}

async function selectRun(name) {
  pause();
  clearInterval(state.pollTimer);
  state.run = state.index.find((r) => r.name === name);
  const { events, programs } = await fetchRunData(name);
  state.events = events;
  state.programs = programs;
  state.recordPos = events.findIndex((e) => e.type === "new_best" && e.id === state.run.record_program);
  $("recordBtn").hidden = state.recordPos < 0;
  $("liveBadge").hidden = state.run.finished;
  const r = state.run;
  $("runTitle").textContent = `${friendlyTitle(r)} · best answer published before: ${r.best_known ?? "unknown"} (lower is better)`;
  seek(0);
  if (!r.finished) state.pollTimer = setInterval(poll, 4000);
}

async function poll() {
  const { events, programs } = await fetchRunData(state.run.name);
  state.programs = programs;
  if (events.length > state.events.length) {
    state.events = events;
    $("scrubber").max = events.length;
    if (state.playing && !state.timer) schedule();
  }
  if (events.some((e) => e.type === "finished" || e.type === "budget_stop")) {
    $("liveBadge").hidden = true;
    clearInterval(state.pollTimer);
  }
}

function resetView() {
  $("feed").innerHTML = "";
  state.points = [];
  state.bestLine = [];
  state.best = null;
  $("bestImg").removeAttribute("src");
  $("bestNums").innerHTML = "";
  $("bestLabel").textContent = "";
}

function seek(target) {
  resetView();
  state.pos = 0;
  while (state.pos < Math.min(target, state.events.length)) apply(state.events[state.pos++], false);
  $("scrubber").max = state.events.length;
  $("scrubber").value = state.pos;
  drawChart();
}

function step() {
  if (state.pos >= state.events.length) {
    state.timer = null;
    if (state.run.finished) pause();
    return;
  }
  apply(state.events[state.pos++], true);
  $("scrubber").value = state.pos;
  drawChart();
  schedule();
}

function schedule() {
  state.timer = setTimeout(step, 1100 / state.speed);
}

function play() {
  if (state.pos >= state.events.length && state.run.finished) seek(0);
  state.playing = true;
  $("playBtn").textContent = "Pause";
  clearTimeout(state.timer);
  schedule();
}

function pause() {
  state.playing = false;
  clearTimeout(state.timer);
  state.timer = null;
  $("playBtn").textContent = "Play";
}

function addCard(html, cls, onClick) {
  const el = document.createElement("div");
  el.className = `card ${cls}`;
  el.innerHTML = html;
  if (onClick) el.addEventListener("click", onClick);
  $("feed").prepend(el);
  return el;
}

function apply(ev, animate) {
  const run = state.run;
  if (ev.type === "candidate") {
    const prog = state.programs.get(ev.id) || {};
    const parent = ev.parent != null ? state.programs.get(ev.parent) : null;
    if (ev.valid) state.points.push({ x: state.points.length, y: behind(run, ev.score) });
    else state.points.push({ x: state.points.length, y: null });
    const dir = prog.direction ? `<span class="chip dir" title="${esc(prog.direction)}">following Muse Wonder's direction</span>` : "";
    const saw = ev.picture && ev.picture.toLowerCase() !== "none" && parent?.image
      ? `<div class="saw"><img loading="lazy" src="${dataUrl(run.name, parent.image)}" alt="Picture the worker studied"><div><b>What it saw:</b> ${prose(ev.picture)}</div></div>` : "";
    const result = ev.valid
      ? `score ${Number(ev.score).toFixed(12)} · ${fmtBehind(run, ev.score)}`
      : `<span class="bad">didn't pass the checker</span>`;
    addCard(`
      <div class="meta"><span class="chip ${ev.parent == null ? "dir" : "spark"}">${modelName(prog.model)}</span>program #${ev.id}${ev.parent != null ? ` · built on #${ev.parent}` : ""} ${dir}</div>
      <div class="idea">${prose(prog.idea || ev.idea)}</div>${saw}
      <div class="result">${result}</div>`, ev.valid ? "" : "invalid", () => openProgram(ev.id));
  } else if (ev.type === "new_best") {
    state.best = ev;
    state.bestLine.push({ x: state.points.length - 1, y: behind(run, ev.score) });
    const prog = state.programs.get(ev.id);
    const top = $("feed").firstElementChild;
    if (top) top.classList.add("best");
    if (prog?.image) $("bestImg").src = dataUrl(run.name, prog.image);
    $("bestLabel").textContent = `· program #${ev.id}`;
    $("bestNums").innerHTML = `<span>Score</span><span>${Number(ev.score).toFixed(13)}</span>`
      + `<span>Published</span><span>${run.best_known}</span><span>Status</span><span>${fmtBehind(run, ev.score)}</span>`;
    if (ev.id === run.record_program) {
      const el = addCard(`<div class="meta"><span class="chip wonder">verified</span>program #${ev.id}</div>
        <h3>NEW WORLD RECORD</h3>
        <div>Beats the published value ${run.best_known} by ${(-behind(run, ev.score)).toExponential(2)}. The lab keeps pushing after this.</div>
        <div class="idea muted">${esc(prog?.idea || ev.idea)}</div>`, "recordcard", () => openProgram(ev.id));
      if (animate) { el.classList.add("flash"); $("bestImg").classList.add("flash"); setTimeout(() => $("bestImg").classList.remove("flash"), 1700); }
    }
  } else if (ev.type === "strategy") {
    addCard(`<div class="meta"><span class="chip wonder">Muse Wonder</span>principal investigator · new research directions: ${ev.directions}</div>
      <div class="diag">${prose(ev.diagnosis)}</div>`, "wonder", null);
  } else if (ev.type === "finished" || ev.type === "budget_stop") {
    addCard(`Research session complete · ${ev.programs ?? state.run.programs} programs written`, "note", null);
  }
}

function drawChart() {
  const cv = $("chart"), ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height, padL = 64, padR = 12, padT = 12, padB = 26;
  ctx.clearRect(0, 0, W, H);
  const pts = state.points.filter((p) => p.y != null);
  const n = Math.max(state.events.filter((e) => e.type === "candidate").length, 10);
  const bestYs = state.bestLine.map((p) => p.y);
  const lo = Math.min(0, ...bestYs);
  const sorted = pts.map((p) => p.y).sort((a, b) => a - b);
  const median = sorted.length ? sorted[Math.floor(sorted.length * 0.5)] : 1;
  const hi = Math.max(median, Math.abs(lo) * 2.5, 1e-12);
  const yMin = lo - (hi - lo) * 0.08, yMax = hi;
  const X = (x) => padL + (x / Math.max(1, n - 1)) * (W - padL - padR);
  const Y = (y) => padT + (1 - (Math.min(y, yMax) - yMin) / (yMax - yMin)) * (H - padT - padB);

  ctx.font = "11px Consolas, monospace";
  ctx.fillStyle = "#8b95a7";
  ctx.strokeStyle = "#232a37";
  for (let i = 0; i <= 4; i++) {
    const v = yMin + (yMax - yMin) * (i / 4);
    const y = Y(v);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillText(v === 0 ? "0" : v.toExponential(1), 4, y + 4);
  }
  ctx.fillText("behind record ↑   ahead ↓", padL, H - 8);

  ctx.setLineDash([6, 5]);
  ctx.strokeStyle = "#f5c542";
  ctx.beginPath(); ctx.moveTo(padL, Y(0)); ctx.lineTo(W - padR, Y(0)); ctx.stroke();
  ctx.setLineDash([]);

  for (const p of pts) {
    const clipped = p.y > yMax;
    ctx.fillStyle = clipped ? "transparent" : (p.y < 0 ? "#f5c542" : "#5b6b8a");
    ctx.strokeStyle = "#5b6b8a";
    ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 3, 0, Math.PI * 2);
    clipped ? ctx.stroke() : ctx.fill();
  }

  if (state.bestLine.length) {
    ctx.strokeStyle = "#3ddc97"; ctx.lineWidth = 2;
    ctx.beginPath();
    state.bestLine.forEach((p, i) => {
      const x = X(Math.max(0, p.x)), y = Y(p.y);
      if (i === 0) ctx.moveTo(x, y); else { ctx.lineTo(x, Y(state.bestLine[i - 1].y)); ctx.lineTo(x, y); }
    });
    ctx.lineTo(X(Math.max(0, state.points.length - 1)), Y(state.bestLine.at(-1).y));
    ctx.stroke(); ctx.lineWidth = 1;
  }
}

function openProgram(id) {
  const p = state.programs.get(id);
  if (!p) return;
  const run = state.run;
  const metrics = Object.entries(p.metrics || {}).map(([k, v]) => `${esc(k)}: ${esc(typeof v === "number" ? +v.toPrecision(12) : v)}`).join(" · ");
  $("modalBody").innerHTML = `
    <div class="meta"><span class="chip spark">${modelName(p.model)}</span> program #${p.id}${p.parent_id != null ? ` · built on #${p.parent_id}` : ""} · generation ${p.generation}</div>
    <h3>${prose(p.idea)}</h3>
    ${p.direction ? `<p><b>Direction from Muse Wonder:</b> ${prose(p.direction)}</p>` : ""}
    <p class="result">${p.valid ? `score ${p.score} · ${fmtBehind(run, p.score)}` : "didn't pass the checker"}</p>
    ${p.image ? `<img src="${dataUrl(run.name, p.image)}" alt="Solution picture">` : ""}
    <details><summary>Show the program the AI wrote</summary>
      ${metrics ? `<p class="muted">${metrics}</p>` : ""}
      <pre><code>${esc(p.code)}</code></pre>
    </details>`;
  $("modal").showModal();
}

$("watchBtn").addEventListener("click", async () => {
  const best = state.index.filter(isRecordRun).sort((a, b) => (a.gap_to_record ?? 0) - (b.gap_to_record ?? 0))[0];
  if (!best) return;
  $("runSelect").value = best.name;
  await selectRun(best.name);
  $("lab").scrollIntoView({ behavior: "smooth" });
  play();
});
$("playBtn").addEventListener("click", () => (state.playing ? pause() : play()));
$("runSelect").addEventListener("change", (e) => selectRun(e.target.value));
$("scrubber").addEventListener("input", (e) => { pause(); seek(+e.target.value); });
$("recordBtn").addEventListener("click", () => { pause(); seek(state.recordPos + 1); });
$("speeds").addEventListener("click", (e) => {
  const s = e.target.dataset.speed;
  if (!s) return;
  state.speed = +s;
  document.querySelectorAll("#speeds button").forEach((b) => b.classList.toggle("on", b === e.target));
});

loadIndex();
