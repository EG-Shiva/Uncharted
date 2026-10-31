const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const RECORD_EPS = 1e-9;
const NORMAL_MS = 1600;
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
  index: [], run: null, events: [], programs: new Map(), pos: 0, playing: false, delay: 450, timer: null,
  points: [], bestLine: [], recordPos: -1, pollTimer: null, director: false, attemptsTotal: 0, attempts: 0,
};

const isRecordRun = (r) => r.record_program != null;
const dataUrl = (run, rel) => `data/${encodeURIComponent(run)}/${rel}`;

function friendlyTitle(r) {
  const sc = /spherical_code_(\d+)d_(\d+)/.exec(r.problem || "");
  if (sc) return `${sc[2]} points on a sphere in ${sc[1]} dimensions`;
  const cp = /circle_packing_(\d+)/.exec(r.problem || "");
  if (cp) return `${cp[1]} circles in a square (warm-up test)`;
  return r.title;
}

function modelName(m) {
  if (!m || m === "seed") return "Starting point";
  if (m.startsWith("muse-wonder")) return "Muse Wonder";
  if (m.startsWith("muse-spark")) return "Muse Spark";
  return m;
}

// Positive = behind the published record, negative = ahead of it.
function behind(run, score) {
  if (run.best_known == null) return null;
  return run.maximize ? run.best_known - score : score - run.best_known;
}

function plain(x) {
  const a = Math.abs(x);
  if (a === 0) return "0";
  if (a >= 0.01) return a.toFixed(4);
  return a.toFixed(Math.min(14, Math.ceil(-Math.log10(a)) + 1));
}

function verdict(run, score) {
  const d = behind(run, score);
  if (d == null) return { kind: "none", html: "" };
  const tol = RECORD_EPS * Math.max(1, Math.abs(run.best_known));
  if (d < -tol) return { kind: "beat", html: `<span class="goldtxt">beats the published record by ${plain(d)}</span>` };
  if (Math.abs(d) <= tol) return { kind: "tie", html: `<span class="muted">equal to the published record</span>` };
  return { kind: "behind", html: `<span class="muted">${plain(d)} short of the record</span>` };
}

async function loadIndex() {
  const res = await fetch("data/index.json", { cache: "no-store" });
  state.index = (await res.json()).runs;
  renderStats();
  renderRecords();
  const sel = $("runSelect");
  const ordered = [...state.index].sort((a, b) => (isRecordRun(b) - isRecordRun(a)) || ((a.gap_to_record ?? 0) - (b.gap_to_record ?? 0)));
  sel.innerHTML = ordered.map((r) => `<option value="${esc(r.name)}">${isRecordRun(r) ? "★ Record: " : ""}${esc(friendlyTitle(r))}</option>`).join("");
  const wanted = new URLSearchParams(location.search).get("run");
  const first = ordered.find((r) => r.name === wanted) || ordered[0];
  if (first) { sel.value = first.name; await selectRun(first.name); }
}

function renderStats() {
  const runs = state.index;
  const records = runs.filter(isRecordRun).length;
  const programs = runs.reduce((s, r) => s + r.programs, 0);
  $("recCount").textContent = records;
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
      <div class="row"><span>Best published before</span><span>${r.best_known.toFixed(12)}</span></div>
      <div class="row"><span>Found by the AI</span><span class="new">${r.best_score.toFixed(12)}</span></div>
      <div class="row"><span>Better by</span><span class="new">${plain(r.gap_to_record)}</span></div>
      <div class="cta">▶ Watch how it was found</div>
    </div>`).join("") || `<div class="muted">No records yet.</div>`;
  document.querySelectorAll(".record").forEach((el) => el.addEventListener("click", () => watch(el.dataset.run)));
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

function narrate(kind, label, text, sub = "") {
  $("narrator").className = `narrator k-${kind}`;
  $("narrKind").textContent = label;
  $("narrText").innerHTML = text;
  $("narrSub").innerHTML = sub;
}

async function selectRun(name) {
  pause();
  clearInterval(state.pollTimer);
  state.run = state.index.find((r) => r.name === name);
  $("runSelect").value = name;
  $("feed").innerHTML = `<div class="empty">Loading this research session…</div>`;
  narrate("spark", "Loading", "Loading the research session…");
  const { events, programs } = await fetchRunData(name);
  state.events = events;
  state.programs = programs;
  state.attemptsTotal = events.filter((e) => e.type === "candidate").length;
  state.recordPos = events.findIndex((e) => e.type === "new_best" && e.id === state.run.record_program);
  $("recordBtn").hidden = state.recordPos < 0;
  $("liveBadge").hidden = state.run.finished;
  seek(0);
  narrate("spark", "Ready", `${esc(friendlyTitle(state.run))}`,
    `The best answer ever published is <b>${state.run.best_known}</b> (lower is better). Press Play to watch the AI try to beat it.`);
  if (!state.run.finished) state.pollTimer = setInterval(poll, 4000);
}

async function poll() {
  const { events, programs } = await fetchRunData(state.run.name);
  state.programs = programs;
  if (events.length > state.events.length) {
    state.events = events;
    state.attemptsTotal = events.filter((e) => e.type === "candidate").length;
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
  state.attempts = 0;
  $("bestImg").removeAttribute("src");
}

function seek(target) {
  resetView();
  state.pos = 0;
  while (state.pos < Math.min(target, state.events.length)) apply(state.events[state.pos++], false);
  if (!$("feed").children.length) $("feed").innerHTML = `<div class="empty">Attempts will appear here as the replay plays.</div>`;
  updateProgress();
  drawChart();
}

function updateProgress() {
  const total = Math.max(1, state.attemptsTotal);
  $("progFill").style.width = `${(100 * state.attempts) / total}%`;
  $("progText").textContent = `Attempt ${state.attempts} of ${state.attemptsTotal}`;
  const best = state.bestLine.at(-1);
  if (!best) $("statusText").textContent = "";
  else if (best.y < -RECORD_EPS * Math.max(1, Math.abs(state.run.best_known))) $("statusText").innerHTML = `<b>Record broken · better by ${plain(best.y)}</b>`;
  else if (Math.abs(best.y) <= RECORD_EPS) $("statusText").textContent = "Best so far: equal to the published record";
  else $("statusText").textContent = `Best so far: ${plain(best.y)} short of the record`;
  $("scrubber").max = state.events.length;
  $("scrubber").value = state.pos;
}

function step() {
  state.timer = null;
  if (state.pos >= state.events.length) {
    if (state.run.finished) {
      pause();
      narrate("best", "Finished", "That's the whole research session.",
        state.run.record_program != null ? "The record found here was verified independently to 50 digits." : "");
    }
    return;
  }
  if (state.director && state.recordPos >= 0 && state.pos >= state.recordPos - 4 && state.delay < NORMAL_MS) state.delay = NORMAL_MS;
  const ev = state.events[state.pos++];
  apply(ev, true);
  updateProgress();
  drawChart();
  if (state.director && ev.type === "new_best" && ev.id === state.run.record_program) {
    pause();
    celebrate(ev);
    return;
  }
  schedule();
}

function schedule() {
  clearTimeout(state.timer);
  state.timer = setTimeout(step, state.delay);
}

function play() {
  if (state.pos >= state.events.length && state.run.finished) seek(0);
  state.playing = true;
  $("playBtn").textContent = "❚❚ Pause";
  schedule();
}

function pause() {
  state.playing = false;
  clearTimeout(state.timer);
  state.timer = null;
  $("playBtn").textContent = "▶ Play";
}

async function watch(name) {
  await selectRun(name);
  $("lab").scrollIntoView({ behavior: "smooth" });
  state.director = true;
  state.delay = 650;
  play();
}

function addCard(html, cls, onClick) {
  const empty = $("feed").querySelector(".empty");
  if (empty) empty.remove();
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
    state.attempts++;
    const prog = state.programs.get(ev.id) || {};
    const parent = ev.parent != null ? state.programs.get(ev.parent) : null;
    state.points.push({ x: state.points.length, y: ev.valid ? behind(run, ev.score) : null });
    const isSeed = ev.parent == null;
    const v = ev.valid ? verdict(run, ev.score) : { kind: "bad" };
    const thumb = parent && (parent.thumb || parent.image);
    const saw = ev.picture && ev.picture.toLowerCase() !== "none" && thumb
      ? `<div class="saw"><img loading="lazy" src="${dataUrl(run.name, thumb)}" alt="Picture the AI studied"><div><b>What the AI noticed in the picture:</b> ${prose(ev.picture)}</div></div>` : "";
    const dir = prog.direction ? `<span class="chip dir">following Muse Wonder's plan</span>` : "";
    const result = ev.valid ? `Result: ${v.html}` : `<span class="bad">The checker rejected this attempt.</span>`;
    addCard(`
      <div class="meta"><span class="chip ${isSeed ? "dir" : "spark"}">${modelName(prog.model)}</span>Attempt #${ev.id} ${dir}</div>
      <div class="idea">${isSeed ? "Starting point: the best arrangement published so far." : prose(prog.idea || ev.idea)}</div>${saw}
      <div class="result">${result}</div>`, ev.valid ? "" : "invalid", () => openProgram(ev.id));
    if (animate) {
      if (isSeed) narrate("spark", "Starting point", "The AI starts from the best answer mathematicians have published.", `Score to beat: <b>${run.best_known}</b>`);
      else if (!ev.valid) narrate("bad", `Attempt ${state.attempts} · Muse Spark`, "Tried a new idea, but the checker rejected the answer.", prose(prog.idea || ev.idea));
      else narrate("spark", `Attempt ${state.attempts} · Muse Spark`, `Tried a new idea. Result: ${v.html}.`, prose(prog.idea || ev.idea));
    }
  } else if (ev.type === "new_best") {
    state.bestLine.push({ x: state.points.length - 1, y: behind(run, ev.score) });
    const prog = state.programs.get(ev.id);
    const top = $("feed").firstElementChild;
    if (top) top.classList.add("best");
    if (prog?.image) $("bestImg").src = dataUrl(run.name, prog.image);
    if (ev.id === run.record_program) {
      const el = addCard(`<div class="meta"><span class="chip wonder">verified</span>Attempt #${ev.id}</div>
        <h3>NEW WORLD RECORD</h3>
        <div>Better than the best published answer (${run.best_known}) by ${plain(behind(run, ev.score))}. The team keeps improving after this.</div>`,
        "recordcard", () => openProgram(ev.id));
      if (animate) {
        el.classList.add("flash");
        narrate("record", "New world record", `Attempt #${ev.id} beat the best answer ever published.`,
          `${run.best_known} → <b class="goldtxt">${Number(ev.score).toFixed(12)}</b> (lower is better)`);
      }
    } else if (animate && ev.id !== 0) {
      narrate("best", "New best for the team", `Attempt #${ev.id} is the team's best answer so far.`, verdict(run, ev.score).html);
    }
  } else if (ev.type === "strategy") {
    addCard(`<div class="meta"><span class="chip wonder">Muse Wonder</span>the lead AI reviewed all attempts and set a new plan</div>
      <div class="diag">${prose(ev.diagnosis)}</div>`, "wonder", null);
    if (animate) narrate("wonder", "Muse Wonder · lead AI", "Reviewed every attempt so far and gave the team a new plan.",
      prose((ev.diagnosis || "").split(/(?<=\.)\s/)[0]));
  } else if (ev.type === "finished" || ev.type === "budget_stop") {
    addCard(`Research session complete · ${state.attemptsTotal} attempts`, "note", null);
  }
}

function celebrate(ev) {
  const run = state.run;
  const prog = state.programs.get(ev.id) || {};
  $("celebrateBody").innerHTML = `
    <div class="trophy">NEW WORLD RECORD</div>
    <h2>${esc(friendlyTitle(run))}</h2>
    <p>After ${state.attempts} attempts, the AI found an arrangement better than the best one ever published.</p>
    <div class="compare">
      <div><span>Best published answer</span><b>${run.best_known.toFixed(12)}</b></div>
      <div class="ours"><span>Found by the AI (lower is better)</span><b>${Number(ev.score).toFixed(12)}</b></div>
    </div>
    <p class="muted">Idea that worked: ${prose(prog.idea || ev.idea)}</p>
    <p>The team kept improving afterwards. Its final answer, <b class="goldtxt">${Number(run.best_score).toFixed(12)}</b>,
      was checked independently with 50-digit arithmetic. Coordinates and verification files are in the public repository.</p>`;
  $("celebrate").showModal();
}

function drawChart() {
  const cv = $("chart"), ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height, padL = 78, padR = 12, padT = 14, padB = 26;
  ctx.clearRect(0, 0, W, H);
  const pts = state.points.filter((p) => p.y != null);
  const n = Math.max(state.attemptsTotal, 10);
  const lo = Math.min(0, ...state.bestLine.map((p) => p.y));
  const sorted = pts.map((p) => p.y).sort((a, b) => a - b);
  const median = sorted.length ? sorted[Math.floor(sorted.length * 0.5)] : 1;
  const hi = Math.max(median, Math.abs(lo) * 2.5, 1e-7);
  const yMin = lo - (hi - lo) * 0.1, yMax = hi;
  const X = (x) => padL + (x / Math.max(1, n - 1)) * (W - padL - padR);
  const Y = (y) => padT + (1 - (Math.min(y, yMax) - yMin) / (yMax - yMin)) * (H - padT - padB);

  ctx.font = "12px Segoe UI, sans-serif";
  ctx.fillStyle = "#8b95a7";
  ctx.fillText("worse", 6, padT + 10);
  ctx.fillText("better", 6, H - padB);
  ctx.fillText("attempts →", W - 80, H - 6);
  ctx.setLineDash([6, 5]);
  ctx.strokeStyle = "#f5c542";
  ctx.beginPath(); ctx.moveTo(padL, Y(0)); ctx.lineTo(W - padR, Y(0)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#f5c542";
  ctx.fillText("published record", 6, Y(0) + 4);

  for (const p of pts) {
    const clipped = p.y > yMax;
    ctx.fillStyle = p.y < 0 ? "#f5c542" : "#5b6b8a";
    ctx.strokeStyle = "#5b6b8a";
    ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 3.2, 0, Math.PI * 2);
    clipped ? ctx.stroke() : ctx.fill();
  }
  if (state.bestLine.length) {
    ctx.strokeStyle = "#3ddc97"; ctx.lineWidth = 2.5;
    ctx.beginPath();
    state.bestLine.forEach((p, i) => {
      const x = X(Math.max(0, p.x)), y = Y(p.y);
      if (i === 0) ctx.moveTo(x, y); else { ctx.lineTo(x, Y(state.bestLine[i - 1].y)); ctx.lineTo(x, y); }
    });
    ctx.lineTo(X(Math.max(0, state.points.length - 1)), Y(state.bestLine.at(-1).y));
    ctx.stroke(); ctx.lineWidth = 1;
  }
}

async function openProgram(id) {
  const p = state.programs.get(id);
  if (!p) return;
  const run = state.run;
  const v = p.valid ? verdict(run, p.score) : null;
  $("modalBody").innerHTML = `
    <div class="meta"><span class="chip spark">${modelName(p.model)}</span> Attempt #${p.id}${p.parent_id != null ? ` · improved on attempt #${p.parent_id}` : ""}</div>
    <h3>${p.parent_id == null ? "Starting point: the best published arrangement" : prose(p.idea)}</h3>
    ${p.direction ? `<p><b>Plan from Muse Wonder:</b> ${prose(p.direction)}</p>` : ""}
    <p>${p.valid ? `Score <b>${p.score}</b> · ${v.html}` : "The checker rejected this attempt."}</p>
    ${p.image ? `<img src="${dataUrl(run.name, p.image)}" alt="Picture of this answer">` : ""}
    <details id="codeDetails"><summary>Show the program the AI wrote</summary><pre><code id="codeBox">Loading…</code></pre></details>`;
  $("codeDetails").addEventListener("toggle", async () => {
    const code = p.code ?? await fetch(dataUrl(run.name, `code/${p.id}.py`)).then((r) => (r.ok ? r.text() : "Code unavailable."));
    p.code = code;
    $("codeBox").textContent = code;
  }, { once: true });
  $("modal").showModal();
}

$("watchBtn").addEventListener("click", () => {
  const best = state.index.filter(isRecordRun).sort((a, b) => (a.gap_to_record ?? 0) - (b.gap_to_record ?? 0))[0];
  if (best) watch(best.name);
});
$("playBtn").addEventListener("click", () => {
  if (state.playing) return pause();
  state.director = false;
  state.delay = +$("speedSelect").value;
  play();
});
$("speedSelect").addEventListener("change", (e) => { state.delay = +e.target.value; state.director = false; });
$("runSelect").addEventListener("change", (e) => selectRun(e.target.value));
$("scrubber").addEventListener("input", (e) => { pause(); seek(+e.target.value); });
$("recordBtn").addEventListener("click", () => {
  pause();
  seek(state.recordPos + 1);
  celebrate(state.events[state.recordPos]);
});
$("keepWatching").addEventListener("click", () => {
  $("celebrate").close();
  state.director = false;
  state.delay = +$("speedSelect").value;
  play();
});
$("closeCelebrate").addEventListener("click", () => $("celebrate").close());

loadIndex();
