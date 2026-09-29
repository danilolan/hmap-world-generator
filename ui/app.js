// World generator tuning tool (WORLDGEN.md section 4). The page builds itself
// from /api/stages: one card per stage, one control per parameter.

const $ = (id) => document.getElementById(id);
const state = { stages: [], defaults: {}, params: {}, seed: 1, res: 512, selected: null, view: {}, mode: "2d",
  detail: false, cx: 0.5, cy: 0.5, detailKm: 2, scale: 1, heightFollow: true };
const SCALES = [1, 1.25, 1.5, 2, 2.5, 3, 4];
let requestId = 0, timer = null, three = null, lastHeight = null;

function save() {
  try { localStorage.setItem("worldgen", JSON.stringify({ params: state.params, defaults: state.defaults, seed: state.seed,
    res: state.res, scale: state.scale, heightFollow: state.heightFollow, selected: state.selected, view: state.view }));
  } catch (e) {}
}
function load() {
  try { return JSON.parse(localStorage.getItem("worldgen") || "{}"); } catch (e) { return {}; }
}

async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body) } : undefined);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

function fmt(p, v) {
  if (p.kind === "int") return String(v);
  const s = p.step || 0.01;
  const d = s >= 1 ? 0 : Math.min(4, Math.ceil(-Math.log10(s)));
  return Number(v).toFixed(d);
}

function buildStages() {
  const root = $("stages");
  root.innerHTML = "";
  state.stages.forEach((st, i) => {
    const card = document.createElement("div");
    card.className = "stage" + (st.id === state.selected ? " selected" : "");
    card.innerHTML = `<div class="stageHead"><span class="num">${i + 1}</span><strong>${st.title}</strong></div>
      <div class="stageBody"><div class="stageDesc">${st.description || ""}</div></div>`;
    card.querySelector(".stageHead").onclick = () => select(st.id);
    const body = card.querySelector(".stageBody");
    st.params.filter((p) => !p.advanced).forEach((p) => body.appendChild(control(st, p)));
    const adv = st.params.filter((p) => p.advanced);
    if (adv.length) {
      const det = document.createElement("details");
      det.innerHTML = `<summary>Advanced (${adv.length})</summary>`;
      adv.forEach((p) => det.appendChild(control(st, p)));
      body.appendChild(det);
    }
    if (st.id === "export") body.appendChild(exportBox());
    root.appendChild(card);
  });
}

function control(st, p) {
  const wrap = document.createElement("div");
  wrap.className = "param";
  const val = state.params[st.id][p.key];
  let input;
  if (p.kind === "bool") {
    wrap.innerHTML = `<label class="row"><span>${p.label}</span><input type="checkbox"></label>`;
    input = wrap.querySelector("input");
    input.checked = !!val;
    input.onchange = () => set(st, p, input.checked);
  } else if (p.kind === "choice") {
    wrap.innerHTML = `<div class="row"><span>${p.label}</span><select></select></div>`;
    input = wrap.querySelector("select");
    p.choices.forEach((c) => input.add(new Option(c, c, false, c === val)));
    input.onchange = () => set(st, p, input.value);
  } else {
    wrap.innerHTML = `<div class="row"><span>${p.label}</span><span class="val"></span></div>
      <input type="range" min="${p.min}" max="${p.max}" step="${p.step}">`;
    input = wrap.querySelector("input");
    const out = wrap.querySelector(".val");
    input.value = val;
    out.textContent = fmt(p, val);
    input.oninput = () => { out.textContent = fmt(p, input.value); set(st, p, Number(input.value)); };
  }
  if (p.help) wrap.insertAdjacentHTML("beforeend", `<div class="help">${p.help}</div>`);
  return wrap;
}

function set(st, p, v) {
  state.params[st.id][p.key] = v;
  if (state.selected !== st.id) select(st.id, false);
  schedule();
}

function select(id, rerender = true) {
  state.selected = id;
  document.querySelectorAll(".stage").forEach((el, i) =>
    el.classList.toggle("selected", state.stages[i].id === id));
  buildTabs();
  if (rerender) schedule(0);
}

function current() { return state.stages.find((s) => s.id === state.selected); }

function buildTabs() {
  const st = current(), tabs = $("viewTabs");
  tabs.innerHTML = "";
  if (!st) return;
  if (!state.view[st.id] || !st.views.find((v) => v.id === state.view[st.id])) state.view[st.id] = st.views[0]?.id;
  st.views.forEach((v) => {
    const b = document.createElement("button");
    b.textContent = v.label;
    b.className = v.id === state.view[st.id] ? "active" : "";
    b.onclick = () => { state.view[st.id] = v.id; buildTabs(); schedule(0); };
    tabs.appendChild(b);
  });
  $("mode3d").disabled = !st.has_height;
  if (!st.has_height && state.mode === "3d") setMode("2d");
  $("detailBtn").disabled = !st.has_detail;
  $("detailSize").disabled = !st.has_detail;
  if (!st.has_detail) state.detail = false;
  $("detailBtn").classList.toggle("active", state.detail);
  $("viewTabs").style.visibility = state.detail ? "hidden" : "visible";
}

// seed, preview resolution and world scale: what every request shares
function world() {
  return { seed: state.seed, res: state.res, scale: state.scale, height_follow: state.heightFollow };
}

function schedule(delay = 120) {
  clearTimeout(timer);
  timer = setTimeout(renderNow, delay);
  save();
}

async function renderNow() {
  const st = current();
  if (!st) return;
  const id = ++requestId;
  $("busy").classList.remove("hidden");
  try {
    const out = state.detail
      ? await api("/api/detail", { ...world(), params: state.params, stage: st.id,
          cx: state.cx, cy: state.cy, size_km: state.detailKm, height: state.mode === "3d" })
      : await api("/api/render", { ...world(), params: state.params, stage: st.id,
          view: state.view[st.id], height: state.mode === "3d" });
    if (id !== requestId) return;
    baseImage = out.image;
    $("map").src = out.image;
    $("stats").textContent = Object.entries(out.stats || {}).map(([k, v]) => `${k}: ${v}`).join("   ·   ");
    $("timing").textContent = `${out.ms} ms  (` + out.timings.map((t) => `${t.stage} ${t.cached ? "cached" : t.ms + " ms"}`).join(", ") + ")";
    if (out.height) lastHeight = out.height;
    showLegend(out.legend);
    applyHighlight();
  } catch (e) {
    if (id === requestId) $("stats").textContent = "Error: " + e.message;
  } finally {
    if (id === requestId) $("busy").classList.add("hidden");
  }
}

// ---------------------------------------------------------------- legend
// What the colours mean: a list of swatches (categorical maps, with notes for the biomes)
// or a colour bar (continuous maps). Collapsed state is remembered. Where the server sends
// the legend item of each pixel, clicking an item highlights it on the map (the rest is
// dimmed) until clicked again; the highlight follows the item's name across re-renders, so
// it stays while sliders move.
let legendNow = null, highlighted = null, baseImage = null, highlightId = 0;

function showLegend(lg) {
  legendNow = lg;
  const el = $("legend");
  if (!lg) { el.classList.add("hidden"); return; }
  const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  let body = "";
  if (lg.kind === "items") {
    const pick = !!lg.labels;
    body = (pick ? `<div class="lgNote">Click an item to highlight it on the map</div>` : "") +
      lg.items.map((it, i) => `<div class="lgItem${pick ? " pick" : ""}${it.label === highlighted ? " sel" : ""}" data-i="${i}">
      <span class="sw" style="background:${rgb(it.color)}"></span>
      <div><div>${esc(it.label)}</div>${it.note ? `<div class="lgNote">${esc(it.note)}</div>` : ""}</div></div>`).join("");
  } else {
    const fmtN = (v) => Math.abs(v) >= 100 ? v.toFixed(0) : Number(v.toFixed(2)).toString();
    body = `<div class="lgBar" style="background:linear-gradient(to right, ${lg.colors.map(rgb).join(",")})"></div>
      <div class="lgScale"><span>${fmtN(lg.lo)}</span><span>${esc(lg.unit || "")}</span><span>${fmtN(lg.hi)}</span></div>`;
  }
  const stored = () => { try { return localStorage.getItem("worldgenLegendCollapsed") === "1"; } catch (e) { return false; } };
  const collapsed = stored();
  el.innerHTML = `<div class="lgHead"><strong>${esc(lg.title || "Legend")}</strong><button id="lgToggle">${collapsed ? "show" : "hide"}</button></div>
    <div class="lgBody${collapsed ? " hidden" : ""}">${body}</div>`;
  el.classList.remove("hidden");
  $("lgToggle").onclick = () => {
    const now = stored() ? "0" : "1";
    try { localStorage.setItem("worldgenLegendCollapsed", now); } catch (e) {}
    showLegend(lg);
  };
  el.querySelectorAll(".lgItem.pick").forEach((row) => {
    row.onclick = () => {
      const label = lg.items[Number(row.dataset.i)].label;
      highlighted = highlighted === label ? null : label;
      showLegend(lg);
      applyHighlight();
    };
  });
}

function loadImage(src) {
  return new Promise((ok, fail) => { const im = new Image(); im.onload = () => ok(im); im.onerror = fail; im.src = src; });
}

// The map with only the highlighted legend item in full colour (or the plain map).
async function applyHighlight() {
  const id = ++highlightId;
  const lg = legendNow;
  const index = lg && lg.labels && highlighted != null ? lg.items.findIndex((it) => it.label === highlighted) : -1;
  if (index < 0) {
    $("map").src = baseImage;
    update3d(baseImage);
    return;
  }
  const [img, lab] = await Promise.all([loadImage(baseImage), loadImage(lg.labels)]);
  if (id !== highlightId) return;
  const w = img.naturalWidth, h = img.naturalHeight;
  const canvas = document.createElement("canvas");
  canvas.width = w; canvas.height = h;
  const g = canvas.getContext("2d");
  g.imageSmoothingEnabled = false;
  g.drawImage(lab, 0, 0, w, h);
  const labels = g.getImageData(0, 0, w, h).data;
  g.drawImage(img, 0, 0);
  const px = g.getImageData(0, 0, w, h);
  const d = px.data, want = index + 1;
  for (let i = 0; i < d.length; i += 4) {
    if (labels[i] === want) continue;
    const grey = (d[i] + d[i + 1] + d[i + 2]) / 3;
    d[i] = d[i + 1] = d[i + 2] = 20 + grey * 0.25;
  }
  g.putImageData(px, 0, 0);
  const url = canvas.toDataURL();
  if (id !== highlightId) return;
  $("map").src = url;
  update3d(url);
}

// ---------------------------------------------------------------- export
function exportBox() {
  const box = document.createElement("div");
  box.className = "exportBox";
  box.innerHTML = `<div class="param"><div class="row"><span>Map id</span><input id="mapId" type="text"></div>
      <div class="help">Folder name under Assets/StreamingAssets/World/ (and the full-resolution files beside the repository)</div></div>
    <div class="param"><div class="row"><span>Grid</span><select id="exportRes">
      <option value="2048">2048² (31 m): the final map</option><option value="1024">1024² (62 m): a quick test</option></select></div>
      <div class="help">The grid the stages run on before the 2 m detail; 2048² takes ~20-30 min in all</div></div>
    <button id="exportBtn">Export the map</button>
    <div id="exportStatus" class="help"></div>`;
  box.querySelector("#mapId").value = `world-${state.seed}`;
  box.querySelector("#exportBtn").onclick = () => startExport(false);
  return box;
}

async function startExport(overwrite) {
  const body = { ...world(), res: Number($("exportRes").value), params: state.params, map_id: $("mapId").value.trim(), overwrite };
  let out;
  try { out = await api("/api/export", body); } catch (e) {
    showExport({ state: "failed", map_id: body.map_id, error: "could not start the export: " + e.message });
    return;
  }
  if (out.exists) {
    if (confirm(`${body.map_id} exists. Replace it?
${out.package_dir}
${out.full_dir}`)) return startExport(true);
    return;
  }
  if (out.error) { showExport({ state: "failed", map_id: body.map_id, error: out.error }); return; }
  exportDismissed = false;
  showExport({ state: "running", step: "starting", map_id: body.map_id, done: 0, total: 400, started: Date.now() / 1000,
    now: Date.now() / 1000 });
  pollExport();
}

// Export progress: a panel over the map, whatever stage is selected, polled every 2 s while
// an export runs: files written of the total, how many are left, time spent and left; the
// error (and the end of the log) if it fails; a warning if the tool cannot be reached.
let exportDismissed = false, exportLastOk = null;

async function pollExport() {
  let st;
  try {
    st = await api("/api/export");
    exportLastOk = st;
  } catch (e) {
    // the tool itself is down or restarting: say so, keep trying while an export was running
    if (exportLastOk && exportLastOk.state === "running") {
      showExport({ ...exportLastOk, unreachable: e.message });
      setTimeout(pollExport, 4000);
    }
    return;
  }
  if (st.state === "none") return;
  const btn = $("exportBtn");
  if (btn) btn.disabled = st.state === "running";
  showExport(st);
  if (st.state === "running") setTimeout(pollExport, 2000);
}

function showExport(st) {
  const el = $("exportPanel");
  const small = $("exportStatus");
  if (!el) return;
  const esc = (s) => String(s ?? "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const mins = (s) => s == null ? "?" : s < 90 ? `${Math.round(s)} s` : `${Math.round(s / 60)} min`;
  const elapsed = st.started ? (st.now || Date.now() / 1000) - st.started : null;
  let html = "";
  if (st.state === "running") {
    const total = st.total || 400, done = st.done || 0, tiles = st.step === "tiles";
    const pct = tiles ? Math.round(done / total * 100) : 0;
    const steps = { starting: "starting", pipeline: "running the stages (erosion takes minutes at 2048²)",
      "spawn and water tables": "spawn point and water tables", tiles: "writing the 2 m files" };
    html = `<div class="epHead"><strong>Exporting ${esc(st.map_id)}</strong><span>${pct}%</span></div>
      <div class="epBar${tiles ? "" : " busy"}"><div style="width:${tiles ? pct : 100}%"></div></div>
      <div class="epLine">${esc(steps[st.step] || st.step)}</div>
      <div class="epLine"><b>${done}</b> of ${total} files written · <b>${total - done}</b> left</div>
      <div class="epLine muted">elapsed ${mins(elapsed)}${tiles && st.eta_s != null ? ` · about ${mins(st.eta_s)} left` : ""}</div>
      ${st.unreachable ? `<div class="epWarn">The tool is not answering (${esc(st.unreachable)}); the export runs on by itself — retrying…</div>` : ""}`;
    if (small) small.textContent = `${st.map_id}: ${done}/${total} files`;
  } else if (st.state === "done") {
    html = `<div class="epHead"><strong>✓ ${esc(st.map_id)} exported</strong><button id="epClose">close</button></div>
      <div class="epBar"><div style="width:100%"></div></div>
      <div class="epLine">${st.total || 400} files in ${esc(st.minutes)} min · spawn corner ${esc(st.spawn)}</div>
      <div class="epLine muted">${esc(st.package_dir)}</div>`;
    if (small) small.textContent = `${st.map_id} exported in ${st.minutes} min`;
  } else {
    html = `<div class="epHead"><strong>✗ Export of ${esc(st.map_id || "the map")} failed</strong><button id="epClose">close</button></div>
      <div class="epError">${esc(st.error || "unknown error")}</div>
      ${st.step ? `<div class="epLine">during: ${esc(st.step)}${st.total ? ` (${st.done || 0} of ${st.total} files written)` : ""}</div>` : ""}
      ${st.full_dir ? `<div class="epLine muted">log: ${esc(st.full_dir)}\\export.log</div>` : ""}
      ${st.trace ? `<details><summary>details</summary><pre>${esc(st.trace)}</pre></details>` : ""}`;
    if (small) small.textContent = `${st.map_id || "export"} failed: ${st.error}`;
  }
  if (st.state !== "running" && exportDismissed) return;
  el.className = "ep-" + st.state;
  el.innerHTML = html;
  const close = $("epClose");
  if (close) close.onclick = () => { exportDismissed = true; el.classList.add("hidden"); };
}

// ---------------------------------------------------------------- 3D view
function setMode(m) {
  state.mode = m;
  $("mode2d").classList.toggle("active", m === "2d");
  $("mode3d").classList.toggle("active", m === "3d");
  $("map").classList.toggle("hidden", m === "3d");
  $("three").classList.toggle("hidden", m !== "3d");
  $("exagLabel").classList.toggle("hidden", m !== "3d");
  schedule(0);
}

async function ensure3d() {
  if (three) return three;
  const THREE = await import("three");
  const { OrbitControls } = await import("three/addons/controls/OrbitControls.js");
  const el = $("three");
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  el.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0e1013);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  camera.position.set(0, 0.9, 1.1);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.AmbientLight(0xffffff, 0.9));
  const resize = () => {
    const w = el.clientWidth, h = el.clientHeight;
    renderer.setSize(w, h); camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(el);
  resize();
  (function loop() { controls.update(); renderer.render(scene, camera); requestAnimationFrame(loop); })();
  three = { THREE, scene, mesh: null };
  return three;
}

async function update3d(imageUrl) {
  if (state.mode !== "3d" || !lastHeight) return;
  const t = await ensure3d(), { THREE } = t;
  const n = lastHeight.n;
  const bytes = Uint8Array.from(atob(lastHeight.data), (c) => c.charCodeAt(0));
  const h = new Float32Array(bytes.buffer);
  // the detail window is small: true scale (slider 3 = 1x) reads right there
  const exag = Number($("exag").value) / (state.detail ? 3 : 1);
  const world = n * lastHeight.cell_m;
  const geo = new THREE.PlaneGeometry(1, 1, n - 1, n - 1);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) pos.setY(i, Math.max(h[i], 0) / world * exag);
  geo.computeVertexNormals();
  const tex = new THREE.TextureLoader().load(imageUrl);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.MeshBasicMaterial({ map: tex });
  if (t.mesh) { t.scene.remove(t.mesh); t.mesh.geometry.dispose(); t.mesh.material.dispose(); }
  t.mesh = new THREE.Mesh(geo, mat);
  t.scene.add(t.mesh);
}

// ---------------------------------------------------------------- gallery, presets
async function gallery() {
  const st = current();
  $("gallery").classList.remove("hidden");
  const grid = $("galleryGrid");
  grid.innerHTML = "<div class='muted'>Rendering…</div>";
  const seeds = Array.from({ length: 12 }, () => Math.floor(Math.random() * 1e6));
  try {
    const thumbs = await api("/api/gallery", { ...world(), seeds, res: 160, params: state.params, stage: st.id, view: state.view[st.id] });
    grid.innerHTML = "";
    thumbs.forEach((t) => {
      const d = document.createElement("div");
      d.className = "thumb";
      d.innerHTML = `<img src="${t.image}"><div>seed ${t.seed}</div>`;
      d.onclick = () => { setSeed(t.seed); $("gallery").classList.add("hidden"); };
      grid.appendChild(d);
    });
  } catch (e) { grid.textContent = "Error: " + e.message; }
}

function setSeed(s) { state.seed = Math.max(0, Math.floor(s)); $("seed").value = state.seed; schedule(0); }

async function refreshPresets() {
  const list = await api("/api/presets"), sel = $("presetList");
  sel.innerHTML = "<option value=''>Presets…</option>";
  list.forEach((n) => sel.add(new Option(n, n)));
}

// ---------------------------------------------------------------- start
async function init() {
  const info = await api("/api/stages");
  const saved = load();
  state.stages = info.stages;
  state.defaults = info.defaults;
  state.params = {};
  // Keep only the values the user actually changed: a saved value equal to the default it was saved
  // with follows the current default (defaults evolve while the stages are tuned). Saves from before
  // defaults were recorded cannot tell, so they start from the current defaults.
  for (const st of info.stages) {
    const now = { ...info.defaults[st.id] };
    const was = saved.defaults?.[st.id], mine = saved.params?.[st.id];
    if (was && mine) for (const k of Object.keys(now)) if (k in mine && mine[k] !== was[k]) now[k] = mine[k];
    state.params[st.id] = now;
  }
  state.seed = saved.seed ?? 1;
  state.res = info.resolutions.includes(saved.res) ? saved.res : 512;
  state.view = saved.view || {};
  state.selected = info.stages.find((s) => s.id === saved.selected)?.id || info.stages[0]?.id;
  info.resolutions.forEach((r) => $("res").add(new Option(`${r}² (${(info.world_m / r).toFixed(0)} m/px)`, r, false, r === state.res)));
  state.scale = SCALES.includes(saved.scale) ? saved.scale : 1;
  state.heightFollow = saved.heightFollow ?? true;
  SCALES.forEach((s) => $("scale").add(new Option(s === 1 ? "1× (real)" : `${s}×`, s, false, s === state.scale)));
  $("scale").onchange = () => { state.scale = Number($("scale").value); schedule(0); };
  $("heightFollow").checked = state.heightFollow;
  $("heightFollow").onchange = () => { state.heightFollow = $("heightFollow").checked; schedule(0); };
  $("seed").value = state.seed;
  $("seed").onchange = () => setSeed(Number($("seed").value));
  $("randomSeed").onclick = () => setSeed(Math.floor(Math.random() * 1e6));
  $("res").onchange = () => { state.res = Number($("res").value); schedule(0); };
  $("detailBtn").onclick = () => { state.detail = !state.detail; buildTabs(); schedule(0); };
  $("detailSize").onchange = () => { state.detailKm = Number($("detailSize").value); if (state.detail) schedule(0); };
  $("map").onclick = (ev) => {
    const st = current();
    if (!st?.has_detail || state.detail) return;
    const r = $("map").getBoundingClientRect();
    state.cx = (ev.clientX - r.left) / r.width;
    state.cy = (ev.clientY - r.top) / r.height;
    state.detail = true;
    buildTabs();
    schedule(0);
  };
  $("mode2d").onclick = () => setMode("2d");
  $("mode3d").onclick = () => setMode("3d");
  $("exag").oninput = () => update3d($("map").src);
  $("galleryBtn").onclick = gallery;
  $("galleryMore").onclick = gallery;
  $("galleryClose").onclick = () => $("gallery").classList.add("hidden");
  $("resetStage").onclick = () => { const st = current(); state.params[st.id] = { ...state.defaults[st.id] }; buildStages(); schedule(0); };
  $("presetSave").onclick = async () => {
    const name = prompt("Preset name");
    if (!name) return;
    await api("/api/presets", { name, data: { seed: state.seed, scale: state.scale, heightFollow: state.heightFollow, params: state.params } });
    refreshPresets();
  };
  $("presetLoad").onclick = async () => {
    const name = $("presetList").value;
    if (!name) return;
    const p = await api("/api/presets/" + encodeURIComponent(name));
    for (const st of state.stages) state.params[st.id] = { ...state.defaults[st.id], ...(p.params?.[st.id] || {}) };
    if (p.seed != null) { state.seed = p.seed; $("seed").value = p.seed; }
    if (SCALES.includes(p.scale)) { state.scale = p.scale; $("scale").value = p.scale; }
    if (p.heightFollow != null) { state.heightFollow = p.heightFollow; $("heightFollow").checked = p.heightFollow; }
    buildStages(); schedule(0);
  };
  buildStages();
  buildTabs();
  refreshPresets();
  pollExport();
  if (state.stages.length === 0) $("stats").textContent = "No stages yet.";
  else schedule(0);
}

init().catch((e) => { $("stats").textContent = "Could not reach the generator server: " + e.message; });
