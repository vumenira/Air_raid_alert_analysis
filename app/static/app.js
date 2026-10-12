/* Air Raid Forecast dashboard: map + ranking + region detail + test-period statistics. */
const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

const state = { meta: null, mode: "replay", idx: 0, snap: null, selected: null, timer: null, nowTimer: null, geo: null, view: "today" };

const fmtKyiv = new Intl.DateTimeFormat("en-GB", { timeZone: "Europe/Kyiv", weekday: "short", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
const fmtShort = new Intl.DateTimeFormat("en-GB", { timeZone: "Europe/Kyiv", day: "2-digit", month: "short" });
const fmtHM = new Intl.DateTimeFormat("en-GB", { timeZone: "Europe/Kyiv", hour: "2-digit", minute: "2-digit" });
const pct = (p) => `${Math.round(p * 100)}%`;
const mins = (m) => (m >= 90 ? `${Math.floor(m / 60)}h ${String(Math.round(m % 60)).padStart(2, "0")}m` : `${Math.round(m)} min`);
const endTime = (d) => `~${fmtHM.format(new Date(d.expected_end))}`;
const label = (r) => state.meta.regions.find((x) => x.region === r)?.label ?? r;

// Discrete sequential scale (one hue, dark -> light on a dark surface)
const RAMP = ["--seq-0", "--seq-1", "--seq-2", "--seq-3", "--seq-4", "--seq-5"].map(css);
const riskColor = d3.scaleThreshold().domain([0.1, 0.2, 0.35, 0.5, 0.65]).range(RAMP);

/* ---------- tooltip ---------- */
const tip = $("tooltip");
function showTip(ev, html) {
  tip.innerHTML = html; tip.hidden = false;
  const { innerWidth: W, innerHeight: H } = window;
  const r = tip.getBoundingClientRect();
  tip.style.left = `${Math.min(ev.clientX + 14, W - r.width - 8)}px`;
  tip.style.top = `${Math.min(ev.clientY + 14, H - r.height - 8)}px`;
}
const hideTip = () => (tip.hidden = true);

/* ---------- data ---------- */
const getJSON = async (url) => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status} ${await r.text()}`);
  return r.json();
};

let requestSeq = 0;
async function loadSnapshot() {
  // Responses can arrive out of order (/api/now is slow); only the latest request may render.
  const seq = ++requestSeq;
  if (state.mode === "now") $("status-line").textContent = "fetching live state…";
  let snap;
  try {
    snap = state.mode === "replay"
      ? await getJSON(`/api/replay?t=${encodeURIComponent(state.meta.times[state.idx])}`)
      : await getJSON("/api/now");
  } catch (e) {
    if (seq === requestSeq) $("status-line").innerHTML = `<span class="warn">${e.message}</span>`;
    return;
  }
  if (seq !== requestSeq) return;
  state.snap = snap;
  render();
}

/* ---------- map ---------- */
let mapSel = null;
function drawMap() {
  const el = $("map");
  el.innerHTML = "";
  const { width: w, height: h } = el.getBoundingClientRect();
  // Planar projection (robust to polygon winding): x scaled by cos(latitude) to keep proportions.
  const c = Math.cos((48.5 * Math.PI) / 180);
  const flat = d3.geoTransform({ point(x, y) { this.stream.point(x * c, -y); } });
  const [[x0, y0], [x1, y1]] = d3.geoPath(flat).bounds(state.geo);
  const k = Math.min((w - 20) / (x1 - x0), (h - 20) / (y1 - y0));
  const ox = (w - k * (x1 - x0)) / 2, oy = (h - k * (y1 - y0)) / 2;
  const proj = d3.geoTransform({ point(x, y) { this.stream.point(ox + k * (x * c - x0), oy + k * (-y - y0)); } });
  const path = d3.geoPath(proj);

  // draw Kyiv City last so it stays clickable on top of Kyiv oblast
  const feats = [...state.geo.features].sort((a, b) => (a.properties.region === "Kyiv City") - (b.properties.region === "Kyiv City"));
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${h}`);
  const shapes = svg.append("g").selectAll("path").data(feats).join("path")
    .attr("class", (d) => `oblast${d.properties.permanent ? " perm" : ""}`)
    .attr("d", path)
    .on("mousemove", (ev, d) => mapTip(ev, d))
    .on("mouseleave", hideTip)
    .on("click", (ev, d) => !d.properties.permanent && select(d.properties.region));
  const selected = svg.append("g"), truth = svg.append("g");
  // one number per oblast: P(start within 1 h), or the expected end time while under alert
  const labels = svg.append("g").selectAll("text")
    .data(feats.filter((d) => !d.properties.permanent && d.properties.region !== "Kyiv City"))
    .join("text").attr("class", "map-value").attr("dy", 4).attr("transform", (d) => `translate(${path.centroid(d)})`);
  mapSel = { shapes, truth, selected, labels, path, feats };
}

function mapTip(ev, d) {
  const r = d.properties.region;
  if (d.properties.permanent) return showTip(ev, `<b>${r}</b><br><span class="mono">Permanent alert since 2022 - not modelled</span>`);
  const s = state.snap?.regions.find((x) => x.region === r);
  if (!s) return;
  let html = `<b>${label(r)}</b> ${s.active ? '<span class="tag alert">ALERT</span>' : ""}`;
  if (s.duration) html += `<br>Most likely ends <b>${endTime(s.duration)}</b><br><span class="mono">80% range ${mins(s.duration.p10_min)} – ${mins(s.duration.p90_min)} total</span>`;
  else {
    html += `<br>P(start ≤ 1 h): <b>${pct(s.p_start)}</b>`;
    if (s.p_baseline != null) html += `<br><span class="mono">baseline ${pct(s.p_baseline)} · actual: ${s.actual_start_next_hour ? "started" : "no start"}</span>`;
  }
  showTip(ev, html);
}

function renderMap() {
  const byRegion = Object.fromEntries(state.snap.regions.map((r) => [r.region, r]));
  const { shapes, truth, selected, labels, path, feats } = mapSel;
  const regionOf = (d) => byRegion[d.properties.region] ?? {};
  shapes.filter((d) => !d.properties.permanent)
    .attr("fill", (d) => (regionOf(d).active ? css("--critical") : riskColor(regionOf(d).p_start ?? 0)));
  const outline = (g, cls, pred) => g.selectAll("path").data(feats.filter(pred), (d) => d.properties.region)
    .join("path").attr("class", cls).attr("d", path);
  outline(truth, "outline-truth", (d) => !regionOf(d).active && regionOf(d).actual_start_next_hour);
  outline(selected, "outline-selected", (d) => d.properties.region === state.selected);
  labels.text((d) => (regionOf(d).duration ? endTime(regionOf(d).duration) : regionOf(d).active ? "" : pct(regionOf(d).p_start ?? 0)));
}

/* ---------- ranking ---------- */
function renderRanking() {
  const endKey = (d) => (d.duration ? new Date(d.duration.expected_end).getTime() : Infinity);
  const rows = [...state.snap.regions].sort((a, b) => (b.active - a.active) || (a.active ? endKey(a) - endKey(b) : b.p_start - a.p_start));
  d3.select("#ranking").selectAll("li").data(rows, (d) => d.region).join("li")
    .attr("class", (d) => (d.region === state.selected ? "selected" : null))
    .on("click", (ev, d) => select(d.region))
    .html((d, i) => `
      <span class="rank">${String(i + 1).padStart(2, "0")}</span>
      <span class="name"><span>${label(d.region)}${d.active ? '<span class="tag alert">ALERT</span>' : ""}${!d.active && d.actual_start_next_hour ? '<span class="tag hit">STARTED</span>' : ""}</span>
        ${d.active ? '<span class="note">ends most likely at</span>' : `<span class="bar"><div style="width:${d.p_start * 100}%;background:${riskColor(d.p_start)}"></div></span>`}</span>
      <span class="val">${d.active ? (d.duration ? endTime(d.duration) : "—") : pct(d.p_start)}</span>`)
    .order();
}

/* ---------- region detail ---------- */
async function renderDetail() {
  const r = state.selected;
  const s = state.snap.regions.find((x) => x.region === r);
  $("detail-title").innerHTML = `${label(r)} <span class="muted">${r}</span>`;
  const isReplay = state.mode === "replay";
  let html = s.active
    ? `<div class="hero"><span class="num alert-num">${s.duration ? endTime(s.duration) : "—"}</span><span class="lbl">alert most likely<br>ends at (Kyiv time)</span></div>
       <div><span class="state alert">ALERT ACTIVE</span></div>`
    : `<div class="hero"><span class="num">${pct(s.p_start)}</span><span class="lbl">chance an alert starts<br>within the next hour</span></div>
       <div><span class="state calm">NO ACTIVE ALERT</span></div>
       <dl class="kv">
        ${isReplay ? `<dt>Baseline (oblast × hour rate)</dt><dd>${pct(s.p_baseline)}</dd>
        <dt>Actual: alert started within the hour</dt><dd>${s.actual_start_next_hour ? "yes" : "no"}</dd>` : ""}
       </dl>`;
  if (s.duration) {
    const d = s.duration;
    html += `<div><div class="section-title">How long will it last?</div>
      <dl class="kv">
        ${d.started_at ? `<dt>Started</dt><dd>${fmtHM.format(new Date(d.started_at))} (${mins(d.elapsed_min)} ago)</dd>` : `<dt>Started</dt><dd>unknown (live feed)</dd>`}
        <dt>Most likely total (median)</dt><dd>${mins(d.median_min)}</dd>
        <dt>80% range</dt><dd>${mins(d.p10_min)} – ${mins(d.p90_min)}</dd>
        ${d.actual_min != null ? `<dt>Actual (revealed)</dt><dd>${mins(d.actual_min)}</dd>` : ""}
      </dl>
      <div id="survival" class="chart" style="height:150px"></div>
      ${!d.started_at ? '<div class="note">Live feed has no start time, so the distribution is not conditioned on elapsed time.</div>' : ""}</div>`;
  }
  html += `<div><div class="section-title">Last 7 days <span id="history-count" class="muted"></span></div><div id="history" class="chart" style="height:46px"></div></div>`;
  if (isReplay) html += `<div><div class="section-title">Forecast vs actual · test period</div>
    <div class="chart-legend"><span><i style="background:var(--series-model)"></i>P(start ≤ 1 h)</span><span><i style="background:var(--text-primary);width:2px;height:8px"></i>alert started</span></div>
    <div id="series" class="chart" style="height:120px"></div></div>`;
  $("detail").innerHTML = html;
  if (s.duration) drawSurvival(s.duration);

  const t = state.snap.t;
  const hist = await getJSON(`/api/region?region=${encodeURIComponent(r)}&t=${encodeURIComponent(t)}&live=${!isReplay}`);
  if (state.selected !== r || state.snap.t !== t) return; // stale response
  drawHistory(hist.history, new Date(t));
  if (isReplay) drawSeries(hist.series, new Date(t));
}

function frame(el, m = { t: 8, r: 12, b: 20, l: 34 }) {
  const { width: w, height: h } = el.getBoundingClientRect();
  const svg = d3.select(el).html("").append("svg").attr("viewBox", `0 0 ${w} ${h}`);
  return { svg, w, h, m, iw: w - m.l - m.r, ih: h - m.t - m.b, g: svg.append("g").attr("transform", `translate(${m.l},${m.t})`) };
}

function drawSurvival(d) {
  const f = frame($("survival"));
  const grid = state.meta.curve_grid;
  const x = d3.scaleLinear([0, grid.at(-1)], [0, f.iw]);
  const y = d3.scaleLinear([0, 1], [f.ih, 0]);
  f.g.append("g").attr("class", "axis").attr("transform", `translate(0,${f.ih})`).call(d3.axisBottom(x).ticks(6).tickFormat((v) => `${v}m`));
  f.g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(3).tickFormat(d3.format(".0%")));
  f.g.append("path").datum(grid.map((g, i) => [g, d.curve[i]])).attr("fill", "none")
    .attr("stroke", css("--series-model")).attr("stroke-width", 2).attr("d", d3.line((p) => x(p[0]), (p) => y(p[1])));
  // labels stacked on separate rows so close markers don't collide
  const vline = (v, color, dash, text, row) => {
    if (v == null || v > grid.at(-1)) return;
    f.g.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", 0).attr("y2", f.ih).attr("stroke", color).attr("stroke-dasharray", dash);
    f.g.append("text").attr("class", "series-label").attr("x", x(v) + 3).attr("y", 9 + row * 12).text(text);
  };
  vline(d.median_min, css("--series-model"), "4 3", `median ${mins(d.median_min)}`, 0);
  if (d.actual_min != null) vline(d.actual_min, css("--text-primary"), "1 2", `actual ${mins(d.actual_min)}`, 1);
  if (d.elapsed_min > 0) vline(d.elapsed_min, css("--text-muted"), "2 2", "now", 2);
  // hover: crosshair + P(still active)
  const hover = f.g.append("line").attr("class", "cursor").attr("y1", 0).attr("y2", f.ih).style("display", "none");
  f.g.append("rect").attr("width", f.iw).attr("height", f.ih).attr("fill", "transparent")
    .on("mousemove", (ev) => {
      const m = Math.max(1, Math.min(grid.at(-1), Math.round(x.invert(d3.pointer(ev)[0]))));
      hover.style("display", null).attr("x1", x(m)).attr("x2", x(m));
      showTip(ev, `${m} min after start<br>P(still active): <b>${pct(d.curve[m - 1])}</b>`);
    })
    .on("mouseleave", () => { hover.style("display", "none"); hideTip(); });
}

function drawHistory(intervals, t) {
  const f = frame($("history"), { t: 4, r: 12, b: 18, l: 12 });
  const x = d3.scaleTime([new Date(t - 7 * 864e5), t], [0, f.iw]);
  f.g.append("rect").attr("width", f.iw).attr("height", f.ih).attr("fill", css("--panel-2")).attr("rx", 2);
  f.g.selectAll("rect.a").data(intervals).join("rect").attr("class", "a")
    .attr("x", (d) => x(new Date(d.start))).attr("width", (d) => Math.max(1.5, x(new Date(d.end)) - x(new Date(d.start))))
    .attr("height", f.ih).attr("fill", css("--critical"))
    .on("mousemove", (ev, d) => showTip(ev, `${fmtKyiv.format(new Date(d.start))} → ${fmtHM.format(new Date(d.end))}`))
    .on("mouseleave", hideTip);
  f.g.append("g").attr("class", "axis").attr("transform", `translate(0,${f.ih})`).call(d3.axisBottom(x).ticks(7).tickFormat((v) => fmtShort.format(v)));
  $("history-count").textContent = `${intervals.length} alerts`;
}

function drawSeries(series, t) {
  const f = frame($("series"));
  const ts = series.t.map((v) => new Date(v));
  const x = d3.scaleTime(d3.extent(ts), [0, f.iw]);
  const y = d3.scaleLinear([0, 1], [f.ih, 0]);
  f.g.append("g").attr("class", "axis").attr("transform", `translate(0,${f.ih})`).call(d3.axisBottom(x).ticks(4).tickFormat((v) => fmtShort.format(v)));
  f.g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(3).tickFormat(d3.format(".0%")));
  f.g.selectAll("line.hit").data(ts.filter((_, i) => series.y[i])).join("line").attr("class", "hit")
    .attr("x1", (d) => x(d)).attr("x2", (d) => x(d)).attr("y1", f.ih).attr("y2", f.ih - 6).attr("stroke", css("--text-primary")).attr("stroke-opacity", 0.6);
  f.g.append("path").datum(ts.map((d, i) => [d, series.p[i]])).attr("fill", "none").attr("stroke", css("--series-model"))
    .attr("stroke-width", 1.2).attr("d", d3.line((p) => x(p[0]), (p) => y(p[1])).defined((p) => p[1] != null));
  f.g.append("line").attr("class", "cursor").attr("x1", x(t)).attr("x2", x(t)).attr("y1", 0).attr("y2", f.ih);
  hoverTime(f, x, ts, (i) => `${fmtKyiv.format(ts[i])}<br>${series.p[i] == null ? "under alert - no start forecast" : `P(start ≤ 1 h): <b>${pct(series.p[i])}</b>`}<br><span class="mono">actual: ${series.y[i] ? "started" : "no start"}</span>`);
}

function hoverTime(f, x, ts, html, onClick) {
  const bis = d3.bisector((d) => d).center;
  const line = f.g.append("line").attr("class", "cursor").attr("stroke-opacity", 0.5).attr("y1", 0).attr("y2", f.ih).style("display", "none");
  f.g.append("rect").attr("width", f.iw).attr("height", f.ih).attr("fill", "transparent").style("cursor", onClick ? "pointer" : null)
    .on("mousemove", (ev) => {
      const i = bis(ts, x.invert(d3.pointer(ev)[0]));
      line.style("display", null).attr("x1", x(ts[i])).attr("x2", x(ts[i]));
      showTip(ev, html(i));
    })
    .on("mouseleave", () => { line.style("display", "none"); hideTip(); })
    .on("click", (ev) => onClick && onClick(bis(ts, x.invert(d3.pointer(ev)[0]))));
}

/* ---------- bottom row: Today / Last 7 days ---------- */
const tile = (k, v, d = "", cls = "") => `<div class="kpi"><span class="k">${k}</span><span class="v">${v}</span><span class="d ${cls}">${d}</span></div>`;
const cards = () => [1, 2, 3].map((i) => ({ head: $(`c${i}-head`), el: $(`c${i}`) }));

function setView(v) {
  state.view = v;
  document.querySelectorAll("#bottom-tabs .seg").forEach((b) => b.classList.toggle("active", b.dataset.view === v));
  renderBottom();
}

let bottomSeq = 0;
async function renderBottom() {
  const seq = ++bottomSeq;
  if (!state.snap) return;
  const live = state.mode === "now";
  const period = state.view === "today" ? "today" : "week";
  let g;
  try {
    g = await getJSON(`/api/glance?t=${encodeURIComponent(state.snap.t)}&period=${period}&live=${live}`);
  } catch (e) { $("glance-note").innerHTML = `<span class="warn">${e.message}</span>`; return; }
  if (seq !== bottomSeq) return; // a newer request (time step / tab switch) superseded this one
  renderGlance(g);
}

function renderGlance(g) {
  const [c1, c2, c3] = cards();
  const span = g.period === "today" ? `since 00:00 Kyiv` : `${fmtShort.format(new Date(g.start))} – ${fmtShort.format(new Date(g.end))}`;
  $("glance-note").innerHTML = span + (g.data_until ? ` · <span class="warn">history only until ${fmtKyiv.format(new Date(g.data_until))}</span>` : "");
  const L = g.longest, P = g.peak;
  $("kpis").innerHTML =
    tile("Alerts started", g.alerts_started, g.period === "today" ? "today" : "last 7 days") +
    tile("Alert-hours", `${g.alert_hours}h`, "summed over oblasts") +
    tile("Oblasts affected", `${g.oblasts_affected}/${g.oblasts_total}`, "had at least one alert") +
    tile("Median duration", g.median_duration_min == null ? "—" : mins(g.median_duration_min), "finished alerts") +
    tile("Longest alert", L ? mins(L.minutes) : "—", L ? `${label(L.region)}${L.ongoing ? " · ongoing" : ""}` : "") +
    tile("Peak at once", P ? `${P.oblasts} obl.` : "—", P ? fmtKyiv.format(new Date(P.t)) : "");

  c1.head.innerHTML = `Most affected oblasts <span class="muted">alert-hours</span>`;
  drawBars(c1.el, g.by_region.slice(0, 8).map((d) => ({ k: d.label, v: d.hours, tip: `<b>${d.label}</b><br>${d.hours.toFixed(1)} alert-hours · ${d.alerts} alerts started` })),
    (v) => `${v.toFixed(1)}h`, 105);

  c2.head.innerHTML = `Oblasts under alert <span class="muted">${g.period === "today" ? "every 10 min" : "hourly"}</span>`;
  drawTimeline(c2.el, g.timeline);

  c3.head.innerHTML = `How long alerts lasted <span class="muted">finished alerts</span>`;
  drawColumns(c3.el, g.durations);
}

function drawBars(el, data, fmt, labelWidth) {
  const f = frame(el, { t: 6, r: 50, b: 6, l: labelWidth });
  const y = d3.scaleBand(data.map((d) => d.k), [0, f.ih]).padding(0.3), x = d3.scaleLinear([0, d3.max(data, (d) => d.v) || 1], [0, f.iw]);
  f.g.append("g").attr("class", "axis").call(d3.axisLeft(y).tickSize(0).tickPadding(8)).select(".domain").remove();
  f.g.selectAll("rect").data(data).join("rect").attr("y", (d) => y(d.k)).attr("height", y.bandwidth()).attr("width", (d) => Math.max(1, x(d.v)))
    .attr("rx", 2).attr("fill", css("--series-model"))
    .on("mousemove", (ev, d) => showTip(ev, d.tip)).on("mouseleave", hideTip);
  f.g.selectAll("text.v").data(data).join("text").attr("class", "series-label").attr("x", (d) => x(d.v) + 4)
    .attr("y", (d) => y(d.k) + y.bandwidth() / 2 + 3).text((d) => fmt(d.v));
}

function drawColumns(el, data) {
  const f = frame(el, { t: 14, r: 10, b: 20, l: 30 });
  const x = d3.scaleBand(data.map((d) => d.bin), [0, f.iw]).padding(0.25), y = d3.scaleLinear([0, d3.max(data, (d) => d.n) || 1], [f.ih, 0]).nice();
  const total = d3.sum(data, (d) => d.n) || 1;
  f.g.append("g").attr("class", "axis").attr("transform", `translate(0,${f.ih})`).call(d3.axisBottom(x).tickSize(0).tickPadding(6));
  f.g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(3));
  f.g.selectAll("rect").data(data).join("rect").attr("x", (d) => x(d.bin)).attr("width", x.bandwidth())
    .attr("y", (d) => y(d.n)).attr("height", (d) => f.ih - y(d.n)).attr("rx", 2).attr("fill", css("--series-model"))
    .on("mousemove", (ev, d) => showTip(ev, `<b>${d.bin}</b><br>${d.n} alerts · ${pct(d.n / total)}`)).on("mouseleave", hideTip);
  f.g.selectAll("text.v").data(data).join("text").attr("class", "series-label").attr("text-anchor", "middle")
    .attr("x", (d) => x(d.bin) + x.bandwidth() / 2).attr("y", (d) => y(d.n) - 3).text((d) => pct(d.n / total));
}

function drawTimeline(el, tl) {
  const f = frame(el, { t: 10, r: 14, b: 22, l: 30 });
  const ts = tl.t.map((v) => new Date(v));
  if (ts.length < 2) return;
  const x = d3.scaleTime(d3.extent(ts), [0, f.iw]), y = d3.scaleLinear([0, Math.max(4, d3.max(tl.active))], [f.ih, 0]).nice();
  const short = ts.at(-1) - ts[0] <= 864e5;
  f.g.append("g").attr("class", "axis").attr("transform", `translate(0,${f.ih})`)
    .call(d3.axisBottom(x).ticks(Math.max(2, Math.floor(f.iw / 80))).tickFormat((v) => (short ? fmtHM : fmtShort).format(v)));
  f.g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4));
  const pts = ts.map((d, i) => [d, tl.active[i]]);
  f.g.append("path").datum(pts).attr("fill", css("--critical")).attr("fill-opacity", 0.2).attr("d", d3.area((p) => x(p[0]), f.ih, (p) => y(p[1])).curve(d3.curveStepAfter));
  f.g.append("path").datum(pts).attr("fill", "none").attr("stroke", css("--critical")).attr("stroke-width", 1.5)
    .attr("d", d3.line((p) => x(p[0]), (p) => y(p[1])).curve(d3.curveStepAfter));
  hoverTime(f, x, ts, (i) => `${fmtKyiv.format(ts[i])}<br><b>${tl.active[i]}</b> oblasts under alert`);
}

/* ---------- orchestration ---------- */
function render() {
  const t = new Date(state.snap.t);
  $("clock-kyiv").textContent = `${fmtKyiv.format(t)} Kyiv`;
  $("clock-sub").textContent = state.mode === "replay"
    ? `forecast issued at ${t.toISOString().slice(0, 16).replace("T", " ")} UTC`
    : `live · refreshes every 60 s`;
  $("status-line").innerHTML = state.mode === "replay"
    ? `test period ${fmtShort.format(new Date(state.meta.times[0]))} – ${fmtShort.format(new Date(state.meta.times.at(-1)))}`
    : `live state: ${state.snap.live_source ? "ubilling feed" : '<span class="warn">unavailable</span>'} · history until ${fmtKyiv.format(new Date(state.snap.history_until))}`;
  if (!state.selected) state.selected = [...state.snap.regions].sort((a, b) => (b.p_start ?? -1) - (a.p_start ?? -1))[0].region;
  renderMap();
  renderRanking();
  renderDetail();
  renderBottom(); // situation stats follow the selected time
}

function select(region) {
  state.selected = region;
  renderMap(); renderRanking(); renderDetail();
}

function setIdx(i) {
  state.idx = Math.max(0, Math.min(state.meta.times.length - 1, i));
  $("slider").value = state.idx;
  loadSnapshot();
}

function setMode(mode) {
  state.mode = mode;
  document.body.classList.toggle("now", mode === "now");
  $("mode-replay").classList.toggle("active", mode === "replay");
  $("mode-now").classList.toggle("active", mode === "now");
  $("timeline").classList.toggle("disabled", mode === "now");
  stopPlay();
  clearInterval(state.nowTimer);
  if (mode === "now") state.nowTimer = setInterval(loadSnapshot, 60_000);
  loadSnapshot();
}

function stopPlay() { clearInterval(state.timer); state.timer = null; $("play").textContent = "▶"; }
function togglePlay() {
  if (state.timer) return stopPlay();
  $("play").textContent = "❚❚";
  state.timer = setInterval(() => (state.idx >= state.meta.times.length - 1 ? stopPlay() : setIdx(state.idx + 1)), 700);
}

async function init() {
  [state.meta, state.geo] = await Promise.all([getJSON("/api/meta"), getJSON("/static/oblasts.geojson")]);
  $("ramp").style.background = `linear-gradient(90deg, ${RAMP.map((c, i) => `${c} ${(i / RAMP.length) * 100}% ${((i + 1) / RAMP.length) * 100}%`).join(",")})`;
  const slider = $("slider");
  slider.max = state.meta.times.length - 1;
  state.idx = state.meta.times.length - 1;
  slider.value = state.idx;
  let deb;
  slider.addEventListener("input", () => { stopPlay(); clearTimeout(deb); state.idx = +slider.value; deb = setTimeout(loadSnapshot, 80); });
  $("play").addEventListener("click", togglePlay);
  $("mode-replay").addEventListener("click", () => setMode("replay"));
  $("mode-now").addEventListener("click", () => setMode("now"));
  document.querySelectorAll("#bottom-tabs .seg").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  drawMap();
  window.addEventListener("resize", () => { drawMap(); if (state.snap) render(); });
  await loadSnapshot();
}

init().catch((e) => { document.body.insertAdjacentHTML("afterbegin", `<pre class="warn" style="padding:16px">${e.message}</pre>`); });
