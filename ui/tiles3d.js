// 3D view of the exported 2 m tiles around a clicked point (POST /api/tiles): the corners, the
// game's smooth surface over them, each tile's material, soil depth, water and tree density,
// exactly as Export writes them there, without exporting. The surface is the game's
// TerrainSurface (separable bicubic monotone Catmull-Rom over the 4x4 corner window, the one
// the renderer and the server use); tile colours are stage 12's legend colours. Trees are
// markers of the 16 m ecology cells (density and species), not the game's own placement.

const $ = (id) => document.getElementById(id);
const opts = { corners: 64, grid: "2048", color: "tiles", blend: false, lines: true, water: true, trees: true,
  exag: 1, subdiv: 2 };
let view = null, win = null, point = null, bodyFor = null, requestId = 0, reframe = true;

try { Object.assign(opts, JSON.parse(localStorage.getItem("worldgenTiles") || "{}")); } catch (e) {}
const saveOpts = () => { try { localStorage.setItem("worldgenTiles", JSON.stringify(opts)); } catch (e) {} };

// ---------------------------------------------------------------- the game's surface
function limitedTangent(before, after) {
  if (before * after <= 0) return 0;
  const m = 0.5 * (before + after), limit = 3 * Math.min(Math.abs(before), Math.abs(after));
  return m > limit ? limit : m < -limit ? -limit : m;
}
function monotoneCubic(p0, p1, p2, p3, t) {
  const m1 = limitedTangent(p1 - p0, p2 - p1), m2 = limitedTangent(p2 - p1, p3 - p2);
  const t2 = t * t, t3 = t2 * t;
  return (2 * t3 - 3 * t2 + 1) * p1 + (t3 - 2 * t2 + t) * m1 + (-2 * t3 + 3 * t2) * p2 + (t3 - t2) * m2;
}
// height (m) of the surface at (x, z) in corners of the window, rows from the south
function surface(w, x, z) {
  const i = Math.floor(x), j = Math.floor(z), fx = x - i, fz = z - j;
  const row = (cz) => monotoneCubic(w.corner(i - 1, cz), w.corner(i, cz), w.corner(i + 1, cz), w.corner(i + 2, cz), fx);
  return monotoneCubic(row(j - 1), row(j), row(j + 1), row(j + 2), fz);
}

// ---------------------------------------------------------------- data
function bytes(b64) { return Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)); }

function decode(r) {
  const n = r.n, e = n / r.eco;
  const w = { ...r, h: new Uint16Array(bytes(r.h).buffer), t: new Uint16Array(bytes(r.t).buffer), d: bytes(r.d),
    rgb: bytes(r.rgb), ecology: new Uint16Array(bytes(r.ecology).buffer), modifiers: new Uint16Array(bytes(r.modifiers).buffer),
    stillIds: new Uint16Array(bytes(r.still).buffer), cells: e };
  const clamp = (v) => Math.max(0, Math.min(n - 1, v));
  w.raw = (x, z) => w.h[clamp(z) * n + clamp(x)];
  w.corner = (x, z) => w.raw(x, z) * r.unit_m + r.origin_m;
  w.tile = (x, z) => { const t = w.t[clamp(z) * n + clamp(x)]; return { ground: t & 0xff, cover: (t >> 8) & 0xf, growth: t >> 12 }; };
  w.cell = (x, z) => Math.max(0, Math.min(e - 1, Math.floor(z / r.eco))) * e + Math.max(0, Math.min(e - 1, Math.floor(x / r.eco)));
  // the displayed corners: [b, b + corners], so `corners` whole tiles
  w.b = r.border; w.size = n - 2 * r.border;
  return w;
}

// ---------------------------------------------------------------- scene
async function ensureView() {
  if (view) return view;
  const THREE = await import("three");
  const { OrbitControls } = await import("three/addons/controls/OrbitControls.js");
  const el = $("tilesCanvas");
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.localClippingEnabled = true;
  el.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xa9c1d6);
  const camera = new THREE.PerspectiveCamera(50, 1, 0.5, 5000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.49;
  scene.add(new THREE.HemisphereLight(0xdfeaff, 0x5a5040, 1.1));
  const sun = new THREE.DirectionalLight(0xfff2dd, 1.9);
  sun.position.set(-0.5, 0.8, 0.35);
  scene.add(sun);
  const resize = () => {
    const w = el.clientWidth, h = el.clientHeight;
    renderer.setSize(w, h); camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(el);
  resize();
  (function loop() { controls.update(); renderer.render(scene, camera); requestAnimationFrame(loop); })();
  view = { THREE, renderer, scene, camera, controls, group: null, terrain: null };
  renderer.domElement.addEventListener("pointermove", hover);
  renderer.domElement.addEventListener("pointerleave", () => { $("tilesHover").classList.add("hidden"); });
  return view;
}

function dispose(obj) {
  obj.traverse((o) => {
    o.geometry?.dispose();
    [].concat(o.material || []).forEach((m) => { m.map?.dispose(); m.dispose(); });
  });
}

// window corners (x, z) to scene coordinates: metres from the centre, +x east, -z north
function place(w, x, z) { return [(x - w.b) * w.step_m - w.size, w.size - (z - w.b) * w.step_m]; }

function build() {
  if (!view || !win) return;
  const { THREE } = view, w = win, C = w.size, S = opts.subdiv, V = C * S + 1, ex = opts.exag;
  if (view.group) { view.scene.remove(view.group); dispose(view.group); }
  const group = new THREE.Group();
  const half = C * w.step_m / 2;
  const clip = [new THREE.Plane(new THREE.Vector3(1, 0, 0), half), new THREE.Plane(new THREE.Vector3(-1, 0, 0), half),
    new THREE.Plane(new THREE.Vector3(0, 0, 1), half), new THREE.Plane(new THREE.Vector3(0, 0, -1), half)];

  // terrain: the surface sampled S times per tile side
  const pos = new Float32Array(V * V * 3), uv = new Float32Array(V * V * 2), col = new Float32Array(V * V * 3);
  const heights = new Float32Array(V * V);
  for (let j = 0; j < V; j++) for (let i = 0; i < V; i++) {
    const x = w.b + i / S, z = w.b + j / S, k = j * V + i;
    const h = surface(w, x, z), [px, pz] = place(w, x, z);
    heights[k] = h;
    pos.set([px, h * ex, pz], k * 3);
    uv.set([i / (V - 1), j / (V - 1)], k * 2);
    if (opts.color === "soil") col.set(soilColor(w, x, z), k * 3);
  }
  const index = [];
  for (let j = 0; j < V - 1; j++) for (let i = 0; i < V - 1; i++) {
    const a = j * V + i, b = a + 1, c = a + V, d = c + 1;
    index.push(a, b, c, b, d, c);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  if (opts.color === "soil") geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  geo.setIndex(index);
  geo.computeVertexNormals();
  const mat = new THREE.MeshLambertMaterial({ polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
  if (opts.color === "soil") mat.vertexColors = true;
  else mat.map = tileTexture(w, opts.color);
  view.terrain = new THREE.Mesh(geo, mat);
  view.terrain.userData.heights = heights;
  group.add(view.terrain);

  // the 2 m grid: every corner line, following the surface
  if (opts.lines) {
    const seg = [];
    const at = (i, j) => { const k = j * V + i; return [pos[k * 3], pos[k * 3 + 1] + 0.03, pos[k * 3 + 2]]; };
    for (let c = 0; c <= C; c++) for (let s = 0; s < C * S; s++) {
      seg.push(...at(c * S, s), ...at(c * S, s + 1));          // along z
      seg.push(...at(s, c * S), ...at(s + 1, c * S));          // along x
    }
    const lg = new THREE.BufferGeometry();
    lg.setAttribute("position", new THREE.Float32BufferAttribute(seg, 3));
    group.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.28 })));
  }

  // water: the sea at 0 m, lakes and ponds at their level over their 16 m cells, rivers along
  // their centrelines at their surface (the game draws only the sea today)
  if (opts.water) {
    const waterMat = (c) => new THREE.MeshLambertMaterial({ color: c, transparent: true, opacity: 0.62, clippingPlanes: clip,
      side: THREE.DoubleSide, depthWrite: false });
    let low = Infinity;
    for (const h of heights) low = Math.min(low, h);
    if (low < 0) {
      const sea = new THREE.Mesh(new THREE.PlaneGeometry(2 * half, 2 * half), waterMat(0x2f6fae));
      sea.rotation.x = -Math.PI / 2;
      group.add(sea);
    }
    const quads = [];
    for (let cz = 0; cz < w.cells; cz++) for (let cx = 0; cx < w.cells; cx++) {
      const level = w.still_levels[w.stillIds[cz * w.cells + cx]];
      if (level == null) continue;
      const [x0, z0] = place(w, cx * w.eco, cz * w.eco), [x1, z1] = place(w, (cx + 1) * w.eco, (cz + 1) * w.eco);
      const y = level * ex;
      quads.push(x0, y, z0, x1, y, z0, x0, y, z1, x1, y, z0, x1, y, z1, x0, y, z1);
    }
    if (quads.length) {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(quads, 3));
      g.computeVertexNormals();
      group.add(new THREE.Mesh(g, waterMat(0x3a86c8)));
    }
    const rib = [];
    const toCorner = (xm, zm) => [xm / w.step_m - w.x0, zm / w.step_m - w.z0];
    for (const line of w.rivers) for (let k = 0; k + 1 < line.length; k++) {
      const [ax, az] = place(w, ...toCorner(line[k][0], line[k][1])), [bx, bz] = place(w, ...toCorner(line[k + 1][0], line[k + 1][1]));
      const dx = bx - ax, dz = bz - az, len = Math.hypot(dx, dz) || 1, nx = -dz / len, nz = dx / len;
      const ra = Math.max(line[k][3], 1) / 2, rb = Math.max(line[k + 1][3], 1) / 2;
      const ya = line[k][2] * ex, yb = line[k + 1][2] * ex;
      const p = [[ax + nx * ra, ya, az + nz * ra], [ax - nx * ra, ya, az - nz * ra], [bx + nx * rb, yb, bz + nz * rb], [bx - nx * rb, yb, bz - nz * rb]];
      rib.push(...p[0], ...p[1], ...p[2], ...p[1], ...p[3], ...p[2]);
    }
    if (rib.length) {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(rib, 3));
      g.computeVertexNormals();
      group.add(new THREE.Mesh(g, waterMat(0x4a9ad6)));
    }
  }

  if (opts.trees) group.add(trees(w, clip));
  view.group = group;
  view.scene.add(group);
}

// tile colours: stage 12's materials (a texel per tile), or the ecology cell's biome
function tileTexture(w, mode) {
  const { THREE } = view, C = w.size, px = new Uint8Array(C * C * 4);
  for (let j = 0; j < C; j++) for (let i = 0; i < C; i++) {
    const x = w.b + i, z = w.b + j, o = (j * C + i) * 4;
    let c;
    if (mode === "biome") c = w.biomes[w.ecology[w.cell(x, z)] & 0x3f]?.color || [255, 0, 255];
    else { const k = (z * w.n + x) * 3; c = [w.rgb[k], w.rgb[k + 1], w.rgb[k + 2]]; }
    px.set([c[0], c[1], c[2], 255], o);
  }
  const tex = new THREE.DataTexture(px, C, C, THREE.RGBAFormat);
  tex.colorSpace = THREE.SRGBColorSpace;
  // sharp: every 2 m tile one flat colour; blended: colours meet across the tile edges (the game
  // paints its transitions)
  tex.magFilter = opts.blend ? THREE.LinearFilter : THREE.NearestFilter;
  tex.minFilter = opts.blend ? THREE.LinearFilter : THREE.NearestFilter;
  tex.needsUpdate = true;
  return tex;
}

// soil depth over the rock, per corner (bilinear between corners): red at 0 (bare rock), brown
// near 1 m, green from 3 m
const SOIL_RAMP = [[0, [0.72, 0.18, 0.16]], [0.3, [0.85, 0.55, 0.2]], [1, [0.55, 0.4, 0.24]], [3, [0.2, 0.5, 0.22]], [8, [0.1, 0.3, 0.45]]];
function soilAt(w, x, z) {
  const i = Math.floor(x), j = Math.floor(z), fx = x - i, fz = z - j;
  const d = (a, b) => w.d[Math.min(w.n - 1, b) * w.n + Math.min(w.n - 1, a)] * w.soil_unit_m;
  return (d(i, j) * (1 - fx) + d(i + 1, j) * fx) * (1 - fz) + (d(i, j + 1) * (1 - fx) + d(i + 1, j + 1) * fx) * fz;
}
function soilColor(w, x, z) {
  const v = soilAt(w, x, z);
  for (let k = 1; k < SOIL_RAMP.length; k++) {
    const [a, ca] = SOIL_RAMP[k - 1], [b, cb] = SOIL_RAMP[k];
    if (v <= b) { const t = (v - a) / (b - a); return ca.map((c, m) => c + (cb[m] - c) * t); }
  }
  return SOIL_RAMP[SOIL_RAMP.length - 1][1];
}

// tree markers: per 16 m ecology cell, a count from its canopy density (0-15) at spots hashed
// from the cell's world position; cones for conifers, balls for broadleaves, in the species' colour
function trees(w, clip) {
  const { THREE } = view, ex = opts.exag;
  const spots = { 1: [], 2: [] };
  let seed = 0;
  const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
  const first = Math.floor(w.b / w.eco), last = Math.ceil((w.b + w.size) / w.eco);
  for (let cz = first; cz < last; cz++) for (let cx = first; cx < last; cx++) {
    const e = w.ecology[cz * w.cells + cx], dens = (e >> 6) & 0xf, sp = w.species[(e >> 10) & 0x3f];
    if (!dens || !sp || !sp.kind) continue;
    seed = (((w.x0 / w.eco + cx) * 73856093) ^ ((w.z0 / w.eco + cz) * 19349663)) >>> 0;
    const count = Math.round(dens / 15 * 5 + rnd() * 0.8);
    for (let k = 0; k < count; k++) {
      const x = (cx + rnd()) * w.eco, z = (cz + rnd()) * w.eco;
      if (x < w.b || z < w.b || x > w.b + w.size || z > w.b + w.size) continue;
      const h = surface(w, x, z), level = w.still_levels[w.stillIds[w.cell(x, z)]];
      if (h <= 0.2 || (level != null && h < level)) continue;
      spots[sp.kind].push([...place(w, x, z), h * ex, sp.color, 0.8 + rnd() * 0.4]);
    }
  }
  const group = new THREE.Group();
  const shapes = { 1: new THREE.SphereGeometry(3.2, 10, 8).translate(0, 5.5, 0), 2: new THREE.ConeGeometry(2.6, 11, 8).translate(0, 6.5, 0) };
  const m = new THREE.Matrix4(), c = new THREE.Color();
  for (const kind of [1, 2]) {
    if (!spots[kind].length) continue;
    const mesh = new THREE.InstancedMesh(shapes[kind], new THREE.MeshLambertMaterial({ clippingPlanes: clip }), spots[kind].length);
    spots[kind].forEach(([x, z, y, rgb, s], k) => {
      m.makeScale(s, s, s).setPosition(x, y, z);
      mesh.setMatrixAt(k, m);
      mesh.setColorAt(k, c.setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, THREE.SRGBColorSpace));
    });
    group.add(mesh);
  }
  const trunk = spots[1].concat(spots[2]);
  if (trunk.length) {
    const mesh = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.3, 0.4, 3, 6).translate(0, 1.5, 0),
      new THREE.MeshLambertMaterial({ color: 0x5a4632, clippingPlanes: clip }), trunk.length);
    trunk.forEach(([x, z, y, , s], k) => { m.makeScale(s, s, s).setPosition(x, y, z); mesh.setMatrixAt(k, m); });
    group.add(mesh);
  }
  return group;
}

// ---------------------------------------------------------------- hover
let hoverQueued = false, lastEvent = null;
function hover(ev) {
  lastEvent = ev;
  if (hoverQueued) return;
  hoverQueued = true;
  requestAnimationFrame(() => { hoverQueued = false; showHover(lastEvent); });
}

function showHover(ev) {
  if (!view?.terrain || !win) return;
  const { THREE, camera, renderer } = view, w = win;
  const r = renderer.domElement.getBoundingClientRect();
  const ray = new THREE.Raycaster();
  ray.setFromCamera(new THREE.Vector2((ev.clientX - r.left) / r.width * 2 - 1, -(ev.clientY - r.top) / r.height * 2 + 1), camera);
  const hit = ray.intersectObject(view.terrain)[0];
  const box = $("tilesHover");
  if (!hit) { box.classList.add("hidden"); return; }
  const x = w.b + (hit.point.x + w.size) / w.step_m, z = w.b + (w.size - hit.point.z) / w.step_m;
  const ix = Math.round(x), iz = Math.round(z), tx = Math.floor(x), tz = Math.floor(z);
  const t = w.tile(tx, tz), e = w.ecology[w.cell(x, z)], mods = w.modifiers[w.cell(x, z)];
  const sp = w.species[(e >> 10) & 0x3f], still = w.stillIds[w.cell(x, z)];
  const depth = w.d[iz * w.n + ix] * w.soil_unit_m, raw = w.raw(ix, iz), h = w.corner(ix, iz);
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const rows = [
    ["tile", `${w.x0 + tx}, ${w.z0 + tz}  (${((w.x0 + x) * w.step_m).toFixed(1)}, ${((w.z0 + z) * w.step_m).toFixed(1)} m)`],
    ["ground", w.ground_names[t.ground] ?? t.ground],
    ["cover", `${w.cover_names[t.cover] ?? "cover " + t.cover}${t.cover ? `, growth ${t.growth}/15` : ""}`],
    ["surface here", `${surface(w, x, z).toFixed(2)} m`],
    [`corner ${w.x0 + ix}, ${w.z0 + iz}`, `${h.toFixed(1)} m (raw ${raw})`],
    ["soil depth", `${depth.toFixed(1)} m · rock at ${(h - depth).toFixed(1)} m`],
    ["16 m cell", `${w.biomes[e & 0x3f]?.name ?? "?"} · trees ${(e >> 6) & 0xf}/15${sp?.kind ? " " + sp.name : ""}`],
    ["modifiers", w.modifier_names.map((n, k) => `${n.replace("_", " ")} ${(mods >> (4 * k)) & 0xf}`).join(", ")],
  ];
  if (still) rows.push(["still water", `id ${still} at ${w.still_levels[still]?.toFixed(2) ?? "?"} m`]);
  box.innerHTML = rows.map(([k, v]) => `<div><span class="muted">${esc(k)}</span> ${esc(v)}</div>`).join("");
  box.classList.remove("hidden");
}

// ---------------------------------------------------------------- panel
function setBusy(text) {
  const el = $("tilesBusy");
  el.textContent = text || "";
  el.classList.toggle("hidden", !text);
}

async function fetchWindow() {
  const id = ++requestId;
  const slow = opts.grid !== "preview";
  setBusy(`Computing the 2 m tiles${slow ? ` on the ${opts.grid}² grid (the first time after a slider change runs every stage: minutes)` : ""}…`);
  const body = bodyFor();
  if (opts.grid !== "preview") body.res = Number(opts.grid);
  try {
    const r = await fetch("/api/tiles", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...body, cx: point.cx, cy: point.cy, corners: opts.corners }) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    if (id !== requestId) return;
    await ensureView();
    win = decode(j);
    build();
    // a new point or window size: the camera looks at it again; other changes keep the view
    if (reframe || win.size !== view.lastSize) frame();
    reframe = false;
    view.lastSize = win.size;
    $("tilesStats").textContent = Object.entries(j.stats).map(([k, v]) => `${k}: ${v}`).join("   ·   ");
    $("tilesTiming").textContent = `${j.ms} ms (2 m block ${j.block_ms} ms` +
      (j.timings.length ? ", " + j.timings.filter((t) => !t.cached).map((t) => `${t.stage} ${t.ms} ms`).join(", ") : "") + ")";
    setBusy("");
  } catch (e) {
    if (id === requestId) setBusy("Error: " + e.message);
  }
}

function frame() {
  const s = win.size * win.step_m;
  let mid = 0;
  const hs = view.terrain.userData.heights;
  for (const h of hs) mid += h;
  mid = mid / hs.length * opts.exag;
  view.controls.target.set(0, mid, 0);
  view.camera.position.set(-s * 0.35, mid + s * 0.55, s * 0.75);
  view.controls.update();
}

function controls() {
  const bind = (id, key, parse = (v) => v, refetch = false) => {
    const el = $(id);
    if (el.type === "checkbox") el.checked = !!opts[key]; else el.value = String(opts[key]);
    el.onchange = el.oninput = () => {
      opts[key] = parse(el.type === "checkbox" ? el.checked : el.value);
      saveOpts();
      if (refetch) fetchWindow(); else build();
    };
  };
  bind("tilesSize", "corners", Number, true);
  bind("tilesGrid", "grid", String, true);
  bind("tilesColor", "color");
  bind("tilesBlend", "blend");
  bind("tilesLines", "lines");
  bind("tilesWater", "water");
  bind("tilesTrees", "trees");
  bind("tilesExag", "exag", Number);
  bind("tilesSubdiv", "subdiv", Number);
  $("tilesReload").onclick = fetchWindow;
  $("tilesClose").onclick = () => $("tilesPanel").classList.add("hidden");
}

let bound = false;

/** Open the 3D tile view at (cx, cy), fractions of the whole map with row 0 = north. `body()`
 *  returns the page's seed, grid, scale and parameters, read at every fetch. */
export function openTiles(cx, cy, body) {
  point = { cx, cy };
  bodyFor = body;
  reframe = true;
  if (!bound) { controls(); bound = true; }
  $("tilesPanel").classList.remove("hidden");
  fetchWindow();
}
