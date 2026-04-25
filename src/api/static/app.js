/* ============================================================
   V5 dashboard - vanilla JS + Chart.js
   ------------------------------------------------------------
   No build step. State held in a single `state` object; every
   render function pulls straight from it. Charts are recreated
   on each render (Chart.js handles destroy via .destroy()).
   ============================================================ */

const state = {
  runs: [],
  activeRun: null,
  metrics: null,
  portfolio: null,
  fills: [],
  prompts: [],
  equity: [],
  perfBySymbol: [],
  perfByDay: [],
  aiUsage: null,
  universe: [],
  watchlist: null,
  positions: [],
  promptDetails: {},
  sse: null,
  liveLog: [],
  liveCounts: { trigger: 0, intent: 0, fill: 0, review: 0 },
};

const charts = {};

const COLORS = {
  good: "#3ddc84",
  bad: "#ff5d6c",
  accent: "#4ea1ff",
  accent2: "#7ed3ff",
  purple: "#a48cff",
  teal: "#38d9b9",
  warn: "#f0b429",
  grid: "rgba(255,255,255,0.06)",
  text: "#b0b8c4",
};

const TYPE_COLORS = {
  watchlist: COLORS.accent,
  deep: COLORS.purple,
  trigger: COLORS.purple,
  review: COLORS.warn,
  unknown: COLORS.teal,
};

const MAX_LIVE_LINES = 250;
const FILLS_LIMIT = 500;

/* ---------- Chart.js global defaults ---------- */
Chart.defaults.color = COLORS.text;
Chart.defaults.borderColor = COLORS.grid;
Chart.defaults.font.family = "JetBrains Mono, SF Mono, Consolas, monospace";
Chart.defaults.font.size = 11;
Chart.defaults.animation = false;

/* ---------- fetch helpers ---------- */
async function fetchJSON(url, { silent = false } = {}) {
  let r;
  try {
    r = await fetch(url);
  } catch (e) {
    if (!silent) toast("network error", `${url}\n${e.message}`, "error");
    throw e;
  }
  if (!r.ok) {
    let body = "";
    try { body = (await r.text()).slice(0, 200); } catch {}
    if (!silent) toast(`${r.status} ${r.statusText}`, `${url}\n${body}`, "error");
    throw new Error(`${url} -> ${r.status}`);
  }
  try {
    return await r.json();
  } catch (e) {
    if (!silent) toast("bad JSON", `${url}\n${e.message}`, "error");
    throw e;
  }
}
const q = (id) => document.getElementById(id);

/* ---------- toast notifications ---------- */
function toast(title, body, kind = "info", ttl = 5000) {
  const host = q("toast-host");
  if (!host) return;
  const el = document.createElement("div");
  el.className = `toast toast-${kind}`;
  el.innerHTML =
    `<div class="toast-title">${escapeHtml(title)}</div>` +
    (body ? `<div class="toast-body">${escapeHtml(body)}</div>` : "");
  el.addEventListener("click", () => dismiss());
  host.appendChild(el);
  const t = setTimeout(dismiss, ttl);
  function dismiss() {
    clearTimeout(t);
    el.classList.add("dismiss");
    setTimeout(() => el.remove(), 200);
  }
}

/* ---------- last-updated indicator ---------- */
let lastUpdatedAt = 0;
function markUpdated() {
  lastUpdatedAt = Date.now();
  renderLastUpdated();
}
function renderLastUpdated() {
  const el = q("last-updated");
  if (!el) return;
  if (!lastUpdatedAt) { el.textContent = "never updated"; return; }
  const sec = Math.round((Date.now() - lastUpdatedAt) / 1000);
  let s;
  if (sec < 5) s = "updated just now";
  else if (sec < 60) s = `updated ${sec}s ago`;
  else if (sec < 3600) s = `updated ${Math.round(sec / 60)}m ago`;
  else s = `updated ${Math.round(sec / 3600)}h ago`;
  el.textContent = s;
}
setInterval(renderLastUpdated, 5000);

/* ============================================================
   Init
   ============================================================ */
window.addEventListener("DOMContentLoaded", async () => {
  setupNav();
  setupSidebarToggle();
  setupKeyboardShortcuts();
  setupSortableTables();
  setupSymbolSearch();
  initChartView();
  q("refresh-btn").addEventListener("click", refreshAll);
  q("run-select").addEventListener("change", (e) => switchRun(e.target.value));
  q("trade-filter-sym").addEventListener("input", renderTrades);
  q("trade-filter-kind").addEventListener("change", renderTrades);
  q("prompt-filter-type").addEventListener("change", () => loadPrompts().then(renderPrompts));
  q("prompt-filter-limit").addEventListener("change", () => loadPrompts().then(renderPrompts));
  document.addEventListener("visibilitychange", onVisibilityChange);
  await refreshAll();
  schedulePolling();
});

/* ----- visibility-aware polling: pause when tab hidden ----- */
let pollTimer = null;
function schedulePolling() {
  if (pollTimer) clearInterval(pollTimer);
  if (document.visibilityState === "hidden") return;
  pollTimer = setInterval(refreshLight, 15000);
}
function onVisibilityChange() {
  if (document.visibilityState === "visible") {
    refreshLight().catch(() => {});
    schedulePolling();
    ensureLiveStream();
  } else {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }
}

function setupNav() {
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => activateView(btn.dataset.view));
  });
}
function activateView(view) {
  if (!view) return;
  document.querySelectorAll(".nav-item").forEach((b) =>
    b.classList.toggle("active", b.dataset.view === view));
  document.querySelectorAll(".view").forEach((v) =>
    v.classList.toggle("active", v.id === `view-${view}`));
  // resize charts that became visible
  requestAnimationFrame(() => {
    Object.values(charts).forEach((c) => c && c.resize && c.resize());
    if (chartState.chart && view === "chart") {
      const w = q("tv-chart").clientWidth;
      chartState.chart.applyOptions({ width: w });
      chartState.volumeChart.applyOptions({ width: w });
    }
  });
  if (view === "chart" && chartState.current && !chartState.data) loadChart();
  // close sidebar on mobile after picking a view
  q("sidebar")?.classList.remove("open");
}

function setupSidebarToggle() {
  q("sidebar-toggle")?.addEventListener("click", () => {
    q("sidebar")?.classList.toggle("open");
  });
}

function setupKeyboardShortcuts() {
  document.addEventListener("keydown", (ev) => {
    // Skip when typing in inputs.
    const t = ev.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" ||
              t.tagName === "TEXTAREA" || t.isContentEditable)) {
      return;
    }
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const map = document.querySelector(`.nav-item[data-key="${ev.key}"]`);
    if (map) { activateView(map.dataset.view); ev.preventDefault(); return; }
    if (ev.key === "r" || ev.key === "R") { refreshAll(); ev.preventDefault(); }
    if (ev.key === "/") { q("trade-filter-sym")?.focus(); ev.preventDefault(); }
  });
}

/* ============================================================
   Data refresh
   ============================================================ */
async function refreshAll() {
  try {
    const health = await fetchJSON("/api/health", { silent: true });
    setHealth(health);
    const runs = await fetchJSON("/api/runs", { silent: true });
    state.runs = runs.runs || [];
    if (!state.activeRun) state.activeRun = runs.active;
    populateRunSelect();
    renderRunState();
    if (!state.activeRun) {
      renderEmptyOverall();
      return;
    }
    const results = await Promise.allSettled([
      loadMetrics(), loadPortfolio(), loadEquity(),
      loadFills(), loadPrompts(), loadPerfSymbol(),
      loadPerfDay(), loadAIUsage(),
      loadUniverse(), loadWatchlist(), loadPositions(),
      loadSymbols(),
    ]);
    const failed = results.filter((r) => r.status === "rejected");
    if (failed.length) {
      toast(`${failed.length} endpoint(s) failed`,
        failed.slice(0, 3).map((r) => String(r.reason).slice(0, 80)).join("\n"),
        "warn");
    }
    renderAll();
    markUpdated();
    ensureLiveStream();
  } catch (e) {
    console.error(e);
    setHealth({ status: "error" });
  }
}

function renderEmptyOverall() {
  // Show empty cards rather than leaving stale numbers when no runs exist.
  ["kpi-equity","kpi-cash","kpi-realized","kpi-winrate","kpi-open","kpi-budget"]
    .forEach((id) => setText(id, "—"));
  const dl = q("risk-dl"); if (dl) dl.innerHTML =
    `<div class="empty-state"><strong>no runs found</strong>
     start the scanner via <code>scripts/run_exec.py</code></div>`;
}

async function refreshLight() {
  if (!state.activeRun) return;
  try {
    await Promise.all([loadMetrics(), loadPortfolio(), loadPositions()]);
    renderTopKpis();
    renderRiskDl();
    renderPositions();
    renderRunState();
    markUpdated();
  } catch (e) { console.warn(e); }
}

async function switchRun(rid) {
  state.activeRun = rid;
  if (state.sse) { state.sse.close(); state.sse = null; setLiveStatus(false); }
  state.liveLog = [];
  state.liveCounts = { trigger: 0, intent: 0, fill: 0, review: 0 };
  await refreshAll();
}

const rid = () => encodeURIComponent(state.activeRun);
async function loadMetrics()    { state.metrics    = await fetchJSON(`/api/metrics?run_id=${rid()}`); }
async function loadPortfolio()  { state.portfolio  = await fetchJSON(`/api/portfolio?run_id=${rid()}`); }
async function loadEquity()     { state.equity     = (await fetchJSON(`/api/equity_curve?run_id=${rid()}`)).points || []; }
async function loadFills()      { state.fills      = (await fetchJSON(`/api/runs/${rid()}/fills?limit=${FILLS_LIMIT}`)).rows || []; }
async function loadPerfSymbol() { state.perfBySymbol = (await fetchJSON(`/api/performance/by_symbol?run_id=${rid()}`)).rows || []; }
async function loadPerfDay()    { state.perfByDay  = (await fetchJSON(`/api/performance/by_day?run_id=${rid()}&days=30`)).rows || []; }
async function loadAIUsage()    { state.aiUsage    = await fetchJSON(`/api/ai/usage?run_id=${rid()}&days=14`); }
async function loadUniverse()   { state.universe   = await fetchJSON(`/api/runs/${rid()}/universe`); }
async function loadWatchlist()  { state.watchlist  = await fetchJSON(`/api/runs/${rid()}/watchlist`); }
async function loadPositions()  { state.positions  = (await fetchJSON(`/api/positions?run_id=${rid()}`)).rows || []; }
async function loadPrompts() {
  const t = q("prompt-filter-type").value;
  const lim = q("prompt-filter-limit").value;
  const url = `/api/ai/calls?run_id=${rid()}&limit=${lim}` + (t ? `&call_type=${t}` : "");
  state.prompts = (await fetchJSON(url)).rows || [];
}

async function loadPromptDetail(callId) {
  if (!callId) return null;
  if (state.promptDetails[callId]) return state.promptDetails[callId];
  const detail = await fetchJSON(`/api/ai/calls/${encodeURIComponent(callId)}?run_id=${rid()}`);
  state.promptDetails[callId] = detail;
  return detail;
}

/* ============================================================
   Renderers
   ============================================================ */
function renderAll() {
  renderTopKpis();
  renderRiskDl();
  renderEquityChart();
  renderDailyChart("chart-daily");
  renderDailyChart("chart-daily-perf");
  renderBySymbolChart();
  renderOutcomesChart();
  renderPerfTable();
  renderPositions();
  renderTrades();
  renderPrompts();
  renderAISummary();
  renderAIDailyChart();
  renderAIByTypeChart();
  renderUniverse();
  renderWatchlist();
  reapplySorts();
}

function setHealth(h) {
  const el = q("health");
  if (h.status === "ok") {
    el.className = "pill pill-ok";
    el.textContent = "ok";
    q("brand-run").textContent = h.active_run || "—";
  } else {
    el.className = "pill pill-bad";
    el.textContent = "offline";
  }
}
function renderRunState() {
  const el = q("run-state");
  if (!el) return;
  const r = state.runs.find((x) => x.run_id === state.activeRun);
  if (!r || !r.mtime) { el.textContent = ""; el.className = "pill"; return; }
  const ageMin = Math.round((Date.now() / 1000 - r.mtime) / 60);
  if (ageMin < 5) {
    el.className = "pill pill-fresh";
    el.textContent = `fresh ${ageMin}m`;
  } else if (ageMin < 60) {
    el.className = "pill pill-stale";
    el.textContent = `stale ${ageMin}m`;
  } else {
    el.className = "pill pill-stale";
    el.textContent = `stale ${Math.round(ageMin / 60)}h`;
  }
}
function populateRunSelect() {
  const sel = q("run-select"); sel.innerHTML = "";
  for (const r of state.runs) {
    const opt = document.createElement("option");
    opt.value = r.run_id; opt.textContent = r.run_id;
    if (r.run_id === state.activeRun) opt.selected = true;
    sel.appendChild(opt);
  }
}

function renderTopKpis() {
  const w = state.metrics?.fills_window || {};
  const p = state.metrics?.portfolio || {};
  const b = state.metrics?.budget || {};
  setText("kpi-equity", fmtUsd(p.equity_usd));
  setText("kpi-cash", fmtUsd(p.cash_usd));
  setColored("kpi-realized", w.realized_pnl_usd, fmtUsd(w.realized_pnl_usd));
  setText("kpi-winrate", w.win_rate != null ? (w.win_rate * 100).toFixed(1) + "%" : "—");
  setText("kpi-open", p.open_positions ?? "—");
  setText("kpi-budget", fmtUsd(b.spent_usd ?? b.cost_usd));
}

function renderRiskDl() {
  const p = state.portfolio || {};
  const m = state.metrics?.fills_window || {};
  const b = state.metrics?.budget || {};
  const dl = q("risk-dl"); dl.innerHTML = "";
  appendDl(dl, [
    ["equity", fmtUsd(p.equity_usd)],
    ["cash", fmtUsd(p.cash_usd)],
    ["realized total", fmtUsd(p.realized_pnl_usd)],
    ["fees total", fmtUsd(p.fees_paid_usd)],
    ["loser streak", p.loser_streak ?? "—"],
    ["risk multiplier", fmtNum(p.risk_multiplier, 2)],
    ["closes (7d)", m.closes ?? 0],
    ["winners / losers", `${m.winners ?? 0} / ${m.losers ?? 0}`],
    ["AI cost (today)", fmtUsd(b.spent_usd)],
    ["AI cap (day)", fmtUsd(b.cap_usd ?? b.daily_cap_usd)],
  ]);
}
function appendDl(dl, rows) {
  for (const [k, v] of rows) {
    const dt = document.createElement("dt"); dt.textContent = k;
    const dd = document.createElement("dd"); dd.textContent = v;
    dl.appendChild(dt); dl.appendChild(dd);
  }
}

/* ---------- Charts ---------- */
function destroy(name) {
  if (charts[name]) { charts[name].destroy(); delete charts[name]; }
}

function renderEquityChart() {
  destroy("equity");
  const ctx = q("chart-equity");
  if (!ctx) return;
  const pts = state.equity || [];
  q("equity-meta").textContent = pts.length
    ? `${pts.length} closes · ${fmtUsd(pts[pts.length - 1].realized_usd)}`
    : "no closed trades yet";
  if (pts.length < 1) return;
  const data = pts.map((p) => ({ x: p.ts, y: p.realized_usd }));
  const lastY = pts.length ? pts[pts.length - 1].realized_usd : 0;
  const isPos = lastY >= 0;
  charts.equity = new Chart(ctx, {
    type: "line",
    data: {
      datasets: [{
        label: "realized PnL",
        data,
        borderColor: isPos ? COLORS.good : COLORS.bad,
        backgroundColor: isPos
          ? "rgba(61,220,132,0.12)" : "rgba(255,93,108,0.12)",
        fill: true, tension: 0.18,
        pointRadius: 0, borderWidth: 2,
      }],
    },
    options: {
      maintainAspectRatio: false, responsive: true,
      interaction: { mode: "nearest", intersect: false },
      plugins: { legend: { display: false },
        tooltip: { callbacks: { label: (c) => fmtUsd(c.parsed.y) } } },
      scales: {
        x: { type: "time", time: { tooltipFormat: "yyyy-MM-dd HH:mm" },
             grid: { color: COLORS.grid }, ticks: { maxRotation: 0 } },
        y: { grid: { color: COLORS.grid },
             ticks: { callback: (v) => fmtUsd(v) } },
      },
    },
  });
}

function renderDailyChart(canvasId) {
  const key = canvasId === "chart-daily" ? "daily" : "dailyPerf";
  destroy(key);
  const ctx = q(canvasId);
  if (!ctx) return;
  const rows = state.perfByDay || [];
  if (!rows.length) return;
  const labels = rows.map((r) => r.day);
  const values = rows.map((r) => r.realized_usd);
  const colors = values.map((v) => v >= 0 ? COLORS.good : COLORS.bad);
  charts[key] = new Chart(ctx, {
    type: "bar",
    data: { labels, datasets: [{
      label: "realized USD", data: values, backgroundColor: colors,
      borderWidth: 0,
    }] },
    options: {
      maintainAspectRatio: false, responsive: true,
      plugins: { legend: { display: false },
        tooltip: { callbacks: { label: (c) => fmtUsd(c.parsed.y) } } },
      scales: { x: { grid: { display: false } },
                y: { grid: { color: COLORS.grid },
                     ticks: { callback: (v) => fmtUsd(v) } } },
    },
  });
}

function renderBySymbolChart() {
  destroy("bySymbol");
  const ctx = q("chart-by-symbol");
  if (!ctx) return;
  const rows = (state.perfBySymbol || []).slice(0, 12);
  if (!rows.length) return;
  const labels = rows.map((r) => r.symbol);
  const values = rows.map((r) => r.realized_usd);
  const colors = values.map((v) => v >= 0 ? COLORS.good : COLORS.bad);
  charts.bySymbol = new Chart(ctx, {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: colors,
                                 borderWidth: 0 }] },
    options: {
      indexAxis: "y", maintainAspectRatio: false, responsive: true,
      plugins: { legend: { display: false },
        tooltip: { callbacks: { label: (c) => fmtUsd(c.parsed.x) } } },
      scales: { x: { grid: { color: COLORS.grid },
                     ticks: { callback: (v) => fmtUsd(v) } },
                y: { grid: { display: false } } },
    },
  });
}

function renderOutcomesChart() {
  destroy("outcomes");
  const ctx = q("chart-outcomes");
  if (!ctx) return;
  const closes = state.metrics?.fills_window?.closes_by_kind || {};
  const labels = Object.keys(closes);
  const values = Object.values(closes);
  if (!labels.length) return;
  const palette = [COLORS.bad, COLORS.good, COLORS.warn,
                   COLORS.purple, COLORS.teal, COLORS.accent2];
  charts.outcomes = new Chart(ctx, {
    type: "doughnut",
    data: { labels, datasets: [{
      data: values, backgroundColor: labels.map((_, i) => palette[i % palette.length]),
      borderColor: "#0b0e14", borderWidth: 2,
    }] },
    options: { maintainAspectRatio: false, responsive: true,
               plugins: { legend: { position: "right" } } },
  });
}

/* ---------- Performance table ---------- */
function renderPerfTable() {
  const tbody = document.querySelector("#tbl-perf tbody");
  tbody.innerHTML = "";
  const rows = state.perfBySymbol || [];
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="9"><div class="empty-state">
      <strong>no closed trades yet</strong>
      per-symbol stats appear once positions close.</div></td></tr>`;
    return;
  }
  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${escapeHtml(r.symbol)}</strong></td>
      <td class="num">${r.entries}</td>
      <td class="num">${r.closes}</td>
      <td class="num num-pos">${r.winners}</td>
      <td class="num num-neg">${r.losers}</td>
      <td class="num">${(r.win_rate * 100).toFixed(1)}%</td>
      <td class="num ${r.realized_usd >= 0 ? "num-pos" : "num-neg"}">${fmtUsd(r.realized_usd)}</td>
      <td class="num">${fmtUsd(r.fees_usd)}</td>
      <td class="num">${fmtUsd(r.volume_usd)}</td>`;
    tbody.appendChild(tr);
  }
}

/* ---------- Positions ---------- */
function renderPositions() {
  const tbody = document.querySelector("#tbl-positions tbody");
  tbody.innerHTML = "";
  const rows = state.positions || [];
  q("positions-count").textContent = rows.length ? `(${rows.length})` : "";
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="9"><div class="empty-state">
      <strong>no open positions</strong>
      the trader will fill this once a trigger passes the gate.</div></td></tr>`;
    return;
  }
  for (const p of rows) {
    const sideCls = p.side === "long" ? "pill-side-long" : "pill-side-short";
    const dist = (p.entry_price && p.stop_loss)
      ? Math.abs((p.entry_price - p.stop_loss) / p.entry_price * 100).toFixed(2) + "%"
      : "—";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${escapeHtml(p.symbol || "")}</strong></td>
      <td><span class="pill ${sideCls}">${escapeHtml(p.side || "")}</span></td>
      <td class="num">${fmtNum(p.qty)}</td>
      <td class="num">${fmtNum(p.entry_price)}</td>
      <td class="num">${fmtNum(p.stop_loss)}</td>
      <td class="num">${fmtNum(p.tp1)}</td>
      <td class="num">${fmtNum(p.tp2)}</td>
      <td class="num">${dist}</td>
      <td class="dim">${escapeHtml(p.opened_at || p.entry_ts || "")}</td>`;
    tbody.appendChild(tr);
  }
}

/* ---------- Trades ---------- */
function renderTrades() {
  const tbody = document.querySelector("#tbl-trades tbody");
  tbody.innerHTML = "";
  const symF = (q("trade-filter-sym").value || "").toUpperCase().trim();
  const kindF = q("trade-filter-kind").value;
  let rows = [...(state.fills || [])].reverse();
  if (symF) rows = rows.filter((r) => (r.symbol || "").toUpperCase().includes(symF));
  if (kindF) rows = rows.filter((r) => r.kind === kindF);
  q("trades-count").textContent = `${rows.length} rows`;
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="8"><div class="empty-state">
      <strong>no fills match the filters</strong>
      clear the filters or wait for the next entry.</div></td></tr>`;
    return;
  }
  for (const f of rows) {
    const pnl = Number(f.pnl_usd ?? 0);
    const pnlCls = pnl > 0 ? "num-pos" : pnl < 0 ? "num-neg" : "";
    const sideCls = f.side === "long" ? "pill-side-long" : "pill-side-short";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="dim">${escapeHtml(fmtTs(f.ts))}</td>
      <td><strong>${escapeHtml(f.symbol || "")}</strong></td>
      <td><span class="pill ${sideCls}">${escapeHtml(f.side || "")}</span></td>
      <td><span class="pill pill-kind">${escapeHtml(f.kind || "")}</span></td>
      <td class="num">${fmtNum(f.price)}</td>
      <td class="num">${fmtNum(f.qty)}</td>
      <td class="num ${pnlCls}">${f.pnl_usd != null && f.pnl_usd !== 0 ? fmtUsd(f.pnl_usd) : "—"}</td>
      <td class="num dim">${fmtUsd(f.fee_usd)}</td>`;
    tbody.appendChild(tr);
  }
}

/* ---------- AI ---------- */
function renderAISummary() {
  const u = state.aiUsage || {};
  const t = u.totals || {};
  setText("ai-calls", t.calls ?? "—");
  setText("ai-cost", fmtUsd(t.cost_usd));
  setText("ai-tokens", `${fmtInt(t.tokens_in)} / ${fmtInt(t.tokens_out)}`);
}

function renderAIDailyChart() {
  destroy("aiDaily");
  const ctx = q("chart-ai-daily");
  if (!ctx) return;
  const series = state.aiUsage?.series || [];
  if (!series.length) return;
  const labels = series.map((d) => d.day);
  const types = new Set();
  series.forEach((d) => Object.keys(d.by_type || {}).forEach((t) => types.add(t)));
  const datasets = [...types].map((t) => ({
    label: t,
    data: series.map((d) => (d.by_type && d.by_type[t]) || 0),
    backgroundColor: TYPE_COLORS[t] || COLORS.teal,
    borderWidth: 0,
    stack: "calls",
  }));
  charts.aiDaily = new Chart(ctx, {
    type: "bar",
    data: { labels, datasets },
    options: {
      maintainAspectRatio: false, responsive: true,
      plugins: { legend: { position: "bottom" } },
      scales: { x: { stacked: true, grid: { display: false } },
                y: { stacked: true, grid: { color: COLORS.grid } } },
    },
  });
}

function renderAIByTypeChart() {
  destroy("aiByType");
  const ctx = q("chart-ai-bytype");
  if (!ctx) return;
  const rows = state.aiUsage?.by_type || [];
  if (!rows.length) return;
  const labels = rows.map((r) => r.call_type);
  charts.aiByType = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "cost USD",
          data: rows.map((r) => r.cost_usd),
          backgroundColor: COLORS.warn,
          yAxisID: "y", borderWidth: 0,
        },
        {
          label: "calls",
          data: rows.map((r) => r.calls),
          backgroundColor: COLORS.accent,
          yAxisID: "y1", borderWidth: 0,
        },
      ],
    },
    options: {
      maintainAspectRatio: false, responsive: true,
      plugins: { legend: { position: "bottom" } },
      scales: {
        x: { grid: { display: false } },
        y: { position: "left", grid: { color: COLORS.grid },
             ticks: { callback: (v) => fmtUsd(v) },
             title: { display: true, text: "cost (USD)" } },
        y1: { position: "right", grid: { display: false },
              title: { display: true, text: "calls" } },
      },
    },
  });
}

function renderPrompts() {
  const tbody = document.querySelector("#tbl-prompts tbody");
  tbody.innerHTML = "";
  const rows = [...(state.prompts || [])].reverse();
  q("prompts-count").textContent = `${rows.length} rows`;
  for (const p of rows) {
    const ds = p.decision_summary || {};
    const ct = p.call_type || p.type || "";
    const ctColor = TYPE_COLORS[ct] || COLORS.teal;
    const action = ds.action || "";
    const actionCls = action ? `pill-action-${action}` : "";
    const conf = ds.confidence != null ? (ds.confidence * 100).toFixed(0) + "%" : "—";
    const summary = ds.rationale
      || (Array.isArray(ds.reasoning) && ds.reasoning.length ? ds.reasoning[0] : "")
      || (Array.isArray(ds.selections) && ds.selections.length
            ? `${ds.selections.length} picks` : "");
    const tin = p.tokens_in ?? "—";
    const tout = p.tokens_out ?? "—";

    const tr = document.createElement("tr");
    tr.className = "expandable";
    tr.innerHTML = `
      <td class="dim">${escapeHtml(fmtTs(p.ts))}</td>
      <td><span class="pill" style="border-color:${ctColor};color:${ctColor}">${escapeHtml(ct)}</span></td>
      <td><strong>${escapeHtml(p.symbol || "—")}</strong></td>
      <td class="dim">${escapeHtml(p.model || "")}</td>
      <td class="num">${tin} / ${tout}</td>
      <td class="num">${fmtUsd(p.cost_usd)}</td>
      <td>${action ? `<span class="pill ${actionCls}">${escapeHtml(action)}</span>` : "—"}</td>
      <td class="num">${conf}</td>
      <td>${escapeHtml(truncate(summary, 80))}</td>`;
    tbody.appendChild(tr);

    const det = document.createElement("tr");
    det.className = "detail-row";
    det.style.display = "none";
    det.innerHTML = `<td colspan="9">${renderPromptDetailHtml(p, state.promptDetails[p.call_id])}</td>`;
    tbody.appendChild(det);
    wirePromptDetailActions(det);

    tr.addEventListener("click", async () => {
      const opening = det.style.display === "none";
      det.style.display = opening ? "" : "none";
      if (!opening || !p.call_id || state.promptDetails[p.call_id]) return;
      det.querySelector("td").innerHTML = `<p class="dim">loading full prompt / response…</p>`;
      try {
        const detail = await loadPromptDetail(p.call_id);
        det.querySelector("td").innerHTML = renderPromptDetailHtml(p, detail);
      } catch (e) {
        det.querySelector("td").innerHTML = renderPromptDetailHtml(p, { error: e.message });
      }
      wirePromptDetailActions(det);
    });
  }
}

function renderPromptDetailHtml(p, detail) {
  const ds = (detail && detail.decision_summary) || p.decision_summary || {};
  const parts = [];
  const selectedSymbols = new Set();
  if (p.symbol) selectedSymbols.add(p.symbol);
  for (const sel of ds.selections || []) {
    if (sel && sel.symbol) selectedSymbols.add(sel.symbol);
  }
  if (selectedSymbols.size) {
    parts.push(`<div class="prompt-actions">${[...selectedSymbols].map((symbol) =>
      `<button type="button" class="btn-inline jump-chart" data-chart-symbol="${escapeHtml(symbol)}">chart ${escapeHtml(symbol)}</button>`
    ).join("")}</div>`);
  }
  if (ds.market_regime) {
    parts.push(`<div class="field"><span>regime</span><span>${escapeHtml(ds.market_regime)}</span></div>`);
  }
  if (ds.entry != null || ds.stop_loss != null || ds.take_profit_1 != null) {
    parts.push(`<div class="field"><span>entry / stop / tp1 / tp2</span><span>${
      [ds.entry, ds.stop_loss, ds.take_profit_1, ds.take_profit_2]
        .map((v) => v == null ? "—" : fmtNum(v)).join(" / ")
    }</span></div>`);
  }
  if (ds.time_horizon_bars != null) {
    parts.push(`<div class="field"><span>horizon (bars)</span><span>${ds.time_horizon_bars}</span></div>`);
  }
  if (ds.expected_move_pct != null) {
    parts.push(`<div class="field"><span>expected move</span><span>${ds.expected_move_pct}%</span></div>`);
  }
  if (Array.isArray(ds.reasoning) && ds.reasoning.length) {
    parts.push(`<h3>reasoning</h3><ul>${ds.reasoning.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>`);
  }
  if (Array.isArray(ds.key_confluences) && ds.key_confluences.length) {
    parts.push(`<h3>confluences</h3><ul>${ds.key_confluences.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>`);
  }
  if (Array.isArray(ds.selections) && ds.selections.length) {
    parts.push(`<h3>selections</h3><ul>${ds.selections.map((s) =>
      `<li><button type="button" class="chip chip-action jump-chart" data-chart-symbol="${escapeHtml(s.symbol || "")}"><strong>${escapeHtml(s.symbol || "?")}</strong></button> ${escapeHtml(s.side || "")}` +
      ` · conf ${s.confidence != null ? (s.confidence * 100).toFixed(0) + "%" : "?"}` +
      ` · ${escapeHtml(s.thesis || "")}</li>`).join("")}</ul>`);
  }
  if (ds.rationale) parts.push(`<h3>rationale</h3><p>${escapeHtml(ds.rationale)}</p>`);
  if (ds.invalidation) parts.push(`<h3>invalidation</h3><p>${escapeHtml(ds.invalidation)}</p>`);
  if (detail && detail.error) {
    parts.push(`<p class="bad">${escapeHtml(detail.error)}</p>`);
  }
  const req = detail && typeof detail.request === "object" ? detail.request : null;
  if (req) {
    parts.push(`<h3>request.system</h3><pre>${escapeHtml(req.system ?? "")}</pre>`);
    parts.push(`<h3>request.user</h3><pre>${escapeHtml(req.user ?? "")}</pre>`);
  }
  const resp = detail && typeof detail.response === "object" ? detail.response : null;
  if (resp) {
    if (resp.raw_text != null) {
      parts.push(`<h3>response.raw_text</h3><pre>${escapeHtml(resp.raw_text)}</pre>`);
    }
    const parsed = resp.parsed != null ? resp.parsed : (resp.json != null ? resp.json : null);
    if (parsed != null) {
      parts.push(`<h3>response.parsed</h3><pre>${escapeHtml(JSON.stringify(parsed, null, 2))}</pre>`);
    }
  }
  parts.push(`<h3>raw summary</h3><pre>${escapeHtml(JSON.stringify(detail || p, null, 2))}</pre>`);
  return parts.join("");
}

function wirePromptDetailActions(host) {
  host.querySelectorAll("[data-chart-symbol]").forEach((el) => {
    el.addEventListener("click", async (event) => {
      event.preventDefault();
      event.stopPropagation();
      await selectChartSymbol(el.dataset.chartSymbol, { openView: true });
    });
  });
}

/* ---------- Universe / watchlist ---------- */
function renderUniverse() {
  const box = q("universe-box");
  box.innerHTML = "";
  const rows = Array.isArray(state.universe) ? state.universe : [];
  q("universe-count").textContent = rows.length ? `(${rows.length})` : "";
  if (!rows.length) {
    box.innerHTML = `<div class="empty-state"><strong>no universe yet</strong>
      the scanner builds the universe on its first cycle.</div>`;
    return;
  }
  for (const r of rows) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip chip-action";
    const score = r.score != null ? Number(r.score).toFixed(2) : "";
    chip.innerHTML = `<strong>${escapeHtml(r.symbol || "?")}</strong>` +
      (score ? `<span class="chip-meta">${score}</span>` : "");
    chip.title = JSON.stringify(r, null, 2);
    chip.addEventListener("click", () => { void selectChartSymbol(r.symbol, { openView: true }); });
    box.appendChild(chip);
  }
}

function renderWatchlist() {
  const box = q("watchlist-box");
  box.innerHTML = "";
  const wl = state.watchlist || {};
  // Prefer the canonical schema selections (rich data); fall back to
  // the flat symbols array when the payload doesn't carry selections.
  const sels = Array.isArray(wl.selections) ? wl.selections : [];
  const symbols = Array.isArray(wl.symbols) ? wl.symbols : [];
  if (!sels.length && !symbols.length) {
    box.innerHTML = `<div class="empty-state"><strong>no watchlist</strong>
      run the watchlist step to populate symbols.</div>`;
    return;
  }
  if (sels.length) {
    if (wl.market_regime) {
      const meta = document.createElement("div");
      meta.className = "dim small";
      meta.style.flexBasis = "100%";
      meta.style.marginBottom = "6px";
      meta.textContent = `regime: ${wl.market_regime}` +
        (wl.as_of ? ` · as of ${fmtTs(wl.as_of)}` : "");
      box.appendChild(meta);
    }
    for (const s of sels) {
      const sym = s.symbol || "";
      const side = s.side || "";
      const conf = s.confidence != null
        ? `${Math.round(s.confidence * 100)}%` : "";
      const move = s.expected_move_pct != null
        ? `${Number(s.expected_move_pct).toFixed(1)}%` : "";
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip chip-action" +
        (side === "long" ? " chip-long" : side === "short" ? " chip-short" : "");
      chip.innerHTML = `<strong>${escapeHtml(sym)}</strong>` +
        (side ? `<span class="chip-meta">${escapeHtml(side)}</span>` : "") +
        (conf ? `<span class="chip-meta">${conf}</span>` : "") +
        (move ? `<span class="chip-meta">Δ${move}</span>` : "");
      if (s.thesis) chip.title = s.thesis;
      chip.addEventListener("click", () => { void selectChartSymbol(sym, { openView: true }); });
      box.appendChild(chip);
    }
    return;
  }
  for (const s of symbols) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip chip-action";
    chip.innerHTML = `<strong>${escapeHtml(s)}</strong>`;
    chip.addEventListener("click", () => { void selectChartSymbol(s, { openView: true }); });
    box.appendChild(chip);
  }
}

/* ============================================================
   Live SSE — auto-connect on boot, with exponential backoff.
   ============================================================ */
let sseRetryMs = 1000;
let sseRetryTimer = null;
function ensureLiveStream() {
  if (state.sse || !state.activeRun) return;
  if (document.visibilityState === "hidden") return;
  const url = `/api/live?run_id=${rid()}`;
  let es;
  try {
    es = new EventSource(url);
  } catch (e) {
    setLiveStatus(false);
    return;
  }
  state.sse = es;
  setLiveStatus(true);
  es.addEventListener("open", () => { sseRetryMs = 1000; });
  for (const k of ["hello", "trigger", "intent", "fill", "review",
                   "triggers", "intents", "fills", "reviews"]) {
    es.addEventListener(k, (ev) => pushLive(k, ev.data));
  }
  es.onerror = () => {
    setLiveStatus(false);
    try { es.close(); } catch {}
    state.sse = null;
    if (sseRetryTimer) clearTimeout(sseRetryTimer);
    sseRetryTimer = setTimeout(() => {
      sseRetryMs = Math.min(sseRetryMs * 2, 30000);
      ensureLiveStream();
    }, sseRetryMs);
  };
}
function pushLive(kind, data) {
  const k = kind.replace(/s$/, "");
  if (state.liveCounts[k] != null) state.liveCounts[k]++;
  const ts = new Date().toISOString().slice(11, 19);
  const cls = `ev-${k}`;
  state.liveLog.unshift(`<span class="${cls}">[${ts}] ${k.padEnd(7)}</span> ${escapeHtml(data)}`);
  if (state.liveLog.length > MAX_LIVE_LINES) state.liveLog.length = MAX_LIVE_LINES;
  q("live-log").innerHTML = state.liveLog.join("\n");
  q("live-stats").innerHTML = ["trigger", "intent", "fill", "review"]
    .map((k) => `<span class="stat">${k}: ${state.liveCounts[k]}</span>`).join("");
}
function setLiveStatus(on) {
  const el = q("live-status");
  el.className = on ? "pill pill-ok" : "pill pill-warn";
  el.textContent = on ? "live: on" : "live: off";
}

/* ============================================================
   Format helpers
   ============================================================ */
function setText(id, v) { const el = q(id); if (el) el.textContent = v; }
function setColored(id, v, txt) {
  const el = q(id); if (!el) return;
  el.textContent = txt;
  el.classList.remove("num-pos", "num-neg");
  if (typeof v === "number" && v !== 0) {
    el.classList.add(v > 0 ? "num-pos" : "num-neg");
  }
}
function fmtUsd(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  const n = Number(v);
  const sign = n < 0 ? "-" : "";
  const abs = Math.abs(n);
  if (abs >= 1000) return `${sign}$${abs.toLocaleString(undefined, {maximumFractionDigits: 0})}`;
  return `${sign}$${abs.toFixed(2)}`;
}
function fmtNum(v, digits) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  const n = Number(v);
  if (digits != null) return n.toFixed(digits);
  if (Math.abs(n) >= 1000) return n.toFixed(2);
  return n.toPrecision(6).replace(/0+$/, "").replace(/\.$/, "");
}
function fmtInt(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  return Number(v).toLocaleString();
}
function fmtTs(v) {
  if (!v) return "";
  // Strip 'T' and trailing fractional+timezone for compactness.
  return String(v).replace("T", " ").replace(/\.\d+/, "").replace("+00:00", "Z");
}
function truncate(s, n) {
  if (s == null) return "";
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}
function tfSeconds(tf) {
  if (tf === "D") return 86400;
  const n = parseInt(tf, 10);
  return Number.isFinite(n) && n > 0 ? n * 60 : 60;
}

/* ============================================================
   Chart view (TradingView lightweight-charts)
   ============================================================ */
const chartState = {
  symbols: [],
  current: null,
  tf: "15",
  limit: 500,
  data: null,            // { rows, indicators, flags, ... }
  events: null,          // { triggers, fills, intents, reviews, ai_calls }
  indicatorsOn: new Set(["ema_21", "ema_50", "bb_upper", "bb_lower"]),
  flagsOn: new Set(["volume_climax", "sweep_up", "sweep_down"]),
  eventsOn: new Set(["trigger", "ai_call", "fill", "review"]),
  chart: null,
  candleSeries: null,
  volumeChart: null,
  volumeSeries: null,
  indicatorSeries: {},   // name -> series
  flagSeries: {},        // name -> series (for markers we use main series.setMarkers; here we keep refs)
  initialized: false,
};

const INDICATOR_COLORS = {
  ema_8:    "#7ed3ff",
  ema_21:   "#4ea1ff",
  ema_50:   "#a48cff",
  bb_mid:   "#6e7787",
  bb_upper: "#38d9b9",
  bb_lower: "#38d9b9",
  vwap_20:  "#f0b429",
  rsi_14:   "#ff5d6c",
  atr_pct_14: "#ff5d6c",
};

const FLAG_STYLE = {
  volume_climax:   { color: "#f0b429", shape: "circle", text: "V" },
  sweep_up:        { color: "#3ddc84", shape: "arrowUp", text: "SU" },
  sweep_down:      { color: "#ff5d6c", shape: "arrowDown", text: "SD" },
  macd_cross_up:   { color: "#7ed3ff", shape: "arrowUp", text: "M+" },
  macd_cross_down: { color: "#a48cff", shape: "arrowDown", text: "M-" },
  rsi_overbought:  { color: "#ff5d6c", shape: "circle", text: "OB" },
  rsi_oversold:    { color: "#3ddc84", shape: "circle", text: "OS" },
};

const EVENT_STYLE = {
  trigger: { color: "#a48cff", shape: "circle", text: "T", position: "aboveBar" },
  ai_call: { color: "#f0b429", shape: "square", text: "AI", position: "aboveBar" },
  fill:    { color: "#3ddc84", shape: "arrowUp", text: "F", position: "belowBar" },
  review:  { color: "#7ed3ff", shape: "square", text: "R", position: "aboveBar" },
};

function initChartView() {
  if (chartState.initialized) return;
  chartState.initialized = true;

  buildToggle("chart-indicators", [
    "ema_8", "ema_21", "ema_50",
    "bb_upper", "bb_lower", "bb_mid",
    "vwap_20",
  ], chartState.indicatorsOn, () => applyIndicators());

  buildToggle("chart-flags", [
    "volume_climax", "sweep_up", "sweep_down",
    "macd_cross_up", "macd_cross_down",
    "rsi_overbought", "rsi_oversold",
  ], chartState.flagsOn, () => applyMarkers());

  document.querySelectorAll("#chart-events input[data-event]").forEach((cb) => {
    cb.addEventListener("change", () => {
      if (cb.checked) chartState.eventsOn.add(cb.dataset.event);
      else chartState.eventsOn.delete(cb.dataset.event);
      applyMarkers();
    });
  });

  q("chart-symbol").addEventListener("change", (e) => {
    chartState.current = e.target.value;
    loadChart();
  });
  q("chart-tf").addEventListener("change", (e) => {
    chartState.tf = e.target.value;
    loadChart();
  });
  q("chart-limit").addEventListener("change", (e) => {
    chartState.limit = parseInt(e.target.value, 10) || 500;
    loadChart();
  });
  q("chart-reload").addEventListener("click", loadChart);
}

function buildToggle(hostId, names, set, onChange) {
  const host = q(hostId);
  host.innerHTML = "";
  for (const name of names) {
    const lbl = document.createElement("label");
    lbl.className = "toggle" + (set.has(name) ? " on" : "");
    lbl.innerHTML = `<input type="checkbox" ${set.has(name) ? "checked" : ""}/>${name}`;
    const cb = lbl.querySelector("input");
    cb.addEventListener("change", () => {
      if (cb.checked) { set.add(name); lbl.classList.add("on"); }
      else { set.delete(name); lbl.classList.remove("on"); }
      onChange();
    });
    host.appendChild(lbl);
  }
}

async function loadSymbols() {
  try {
    const data = await fetchJSON(`/api/symbols?run_id=${rid()}`);
    chartState.symbols = data.rows || [];
    const sel = q("chart-symbol");
    if (!sel) return;
    sel.innerHTML = "";
    if (!chartState.symbols.length) {
      const opt = document.createElement("option");
      opt.textContent = "(no symbols yet)"; opt.value = "";
      sel.appendChild(opt); return;
    }
    for (const r of chartState.symbols) {
      const opt = document.createElement("option");
      opt.value = r.symbol;
      const tags = [];
      if (r.open_position) tags.push("●");
      if (r.watchlist) tags.push("wl");
      if (r.fills) tags.push(`${r.fills}f`);
      if (r.triggers) tags.push(`${r.triggers}t`);
      if (r.ai_calls) tags.push(`${r.ai_calls}ai`);
      if (!r.has_candles) tags.push("no-candles");
      opt.textContent = `${r.symbol}  ${tags.join(" ")}`;
      sel.appendChild(opt);
    }
    const availableSymbols = new Set(chartState.symbols.map((row) => row.symbol));
    if (!chartState.current || !availableSymbols.has(chartState.current)) {
      chartState.current = chartState.symbols[0].symbol;
    }
    sel.value = chartState.current;
  } catch (e) { console.warn(e); }
}

async function selectChartSymbol(symbol, { openView = false } = {}) {
  if (!symbol) return;
  if (!chartState.symbols.length) await loadSymbols();
  initChartView();
  const sel = q("chart-symbol");
  if (sel && !Array.from(sel.options).some((opt) => opt.value === symbol)) {
    const opt = document.createElement("option");
    opt.value = symbol;
    opt.textContent = `${symbol}  manual`;
    sel.appendChild(opt);
  }
  chartState.current = symbol;
  if (sel) sel.value = symbol;
  if (openView) {
    document.querySelector('.nav-item[data-view="chart"]')?.click();
  }
  await loadChart();
}

function ensureChart() {
  if (chartState.chart) return;
  const opts = {
    layout: { background: { color: "#0b0e14" }, textColor: "#b0b8c4" },
    grid: { vertLines: { color: "rgba(255,255,255,0.04)" },
            horzLines: { color: "rgba(255,255,255,0.04)" } },
    rightPriceScale: { borderColor: "#232a36" },
    timeScale: { borderColor: "#232a36", timeVisible: true, secondsVisible: false },
    crosshair: { mode: 1 },
  };
  chartState.chart = LightweightCharts.createChart(q("tv-chart"), opts);
  chartState.candleSeries = chartState.chart.addCandlestickSeries({
    upColor: "#3ddc84", downColor: "#ff5d6c",
    borderUpColor: "#3ddc84", borderDownColor: "#ff5d6c",
    wickUpColor: "#3ddc84", wickDownColor: "#ff5d6c",
  });
  chartState.volumeChart = LightweightCharts.createChart(q("tv-volume"), {
    ...opts, timeScale: { ...opts.timeScale, visible: false },
  });
  chartState.volumeSeries = chartState.volumeChart.addHistogramSeries({
    priceFormat: { type: "volume" },
    color: "rgba(78,161,255,0.4)",
  });
  // Sync time scales.
  chartState.chart.timeScale().subscribeVisibleLogicalRangeChange((r) => {
    if (r) chartState.volumeChart.timeScale().setVisibleLogicalRange(r);
  });
  // Click marker -> detail.
  chartState.candleSeries.subscribeDataChanged?.(() => {});
  chartState.chart.subscribeClick((p) => onChartClick(p));

  window.addEventListener("resize", () => {
    if (!chartState.chart) return;
    const w = q("tv-chart").clientWidth;
    chartState.chart.applyOptions({ width: w });
    chartState.volumeChart.applyOptions({ width: w });
  });
}

async function loadChart() {
  if (!chartState.current) return;
  ensureChart();
  q("chart-meta").textContent = "loading…";
  try {
    const [data, events] = await Promise.all([
      fetchJSON(`/api/candles?symbol=${encodeURIComponent(chartState.current)}` +
                `&tf=${encodeURIComponent(chartState.tf)}&limit=${chartState.limit}`),
      fetchJSON(`/api/symbol_events?symbol=${encodeURIComponent(chartState.current)}` +
                `&run_id=${rid()}`),
    ]);
    chartState.data = data;
    chartState.events = events;
    if (!data.rows.length) {
      q("chart-meta").innerHTML = `<span class="bad">no candles cached for ${escapeHtml(chartState.current)} ${chartState.tf}</span>`;
      chartState.candleSeries.setData([]);
      chartState.volumeSeries.setData([]);
      return;
    }
    // Set candles.
    chartState.candleSeries.setData(data.rows.map((r) => ({
      time: r.time, open: r.open, high: r.high, low: r.low, close: r.close,
    })));
    chartState.volumeSeries.setData(data.rows.map((r) => ({
      time: r.time, value: r.volume,
      color: r.close >= r.open ? "rgba(61,220,132,0.5)" : "rgba(255,93,108,0.5)",
    })));
    const first = data.rows[0].time;
    const last = data.rows[data.rows.length - 1].time;
    q("chart-meta").innerHTML =
      `${data.rows.length} bars · <span class="dim">${
        new Date(first * 1000).toISOString().slice(0,16).replace("T"," ")
      } → ${new Date(last * 1000).toISOString().slice(0,16).replace("T"," ")}</span>`;
    chartState.chart.timeScale().fitContent();
    applyIndicators();
    applyMarkers();
    renderSymbolTables();
    q("event-detail").innerHTML =
      `<p class="dim">${data.rows.length} bars loaded. Click a marker on the chart to inspect.</p>`;
  } catch (e) {
    console.error(e);
    q("chart-meta").innerHTML = `<span class="bad">${escapeHtml(e.message)}</span>`;
  }
}

function applyIndicators() {
  if (!chartState.data) return;
  const data = chartState.data;
  // Remove series no longer toggled on.
  for (const name of Object.keys(chartState.indicatorSeries)) {
    if (!chartState.indicatorsOn.has(name)) {
      try { chartState.chart.removeSeries(chartState.indicatorSeries[name]); } catch {}
      delete chartState.indicatorSeries[name];
    }
  }
  for (const name of chartState.indicatorsOn) {
    const series = data.indicators?.[name];
    if (!series) continue;
    let s = chartState.indicatorSeries[name];
    if (!s) {
      s = chartState.chart.addLineSeries({
        color: INDICATOR_COLORS[name] || "#4ea1ff",
        lineWidth: name.startsWith("bb_") ? 1 : 2,
        lineStyle: name === "bb_mid" ? 2 : 0,
        priceLineVisible: false,
        lastValueVisible: false,
      });
      chartState.indicatorSeries[name] = s;
    }
    const points = [];
    for (let i = 0; i < series.length; i++) {
      const v = series[i];
      if (v == null) continue;
      points.push({ time: data.rows[i].time, value: v });
    }
    s.setData(points);
  }
}

function applyMarkers() {
  if (!chartState.data || !chartState.candleSeries) return;
  const markers = [];

  // Flag markers (computed from candles).
  for (const flag of chartState.flagsOn) {
    const list = chartState.data.flags?.[flag] || [];
    const style = FLAG_STYLE[flag] || { color: "#fff", shape: "circle", text: flag };
    for (const m of list) {
      markers.push({
        time: m.time, position: "aboveBar",
        color: style.color, shape: style.shape, text: style.text,
        _kind: "flag", _flag: flag, _meta: m,
      });
    }
  }

  // Event markers (from run-time JSONL).
  if (chartState.events) {
    if (chartState.eventsOn.has("trigger")) {
      for (const t of chartState.events.triggers || []) {
        if (!t.time) continue;
        markers.push({
          time: t.time, position: "aboveBar",
          color: t.fired ? EVENT_STYLE.trigger.color : "#6e7787",
          shape: EVENT_STYLE.trigger.shape,
          text: t.fired ? "T" : "·",
          _kind: "trigger", _meta: t,
        });
      }
    }
    if (chartState.eventsOn.has("ai_call")) {
      for (const c of chartState.events.ai_calls || []) {
        if (!c.time) continue;
        const col = c.action === "long" ? "#3ddc84"
                  : c.action === "short" ? "#ff5d6c"
                  : EVENT_STYLE.ai_call.color;
        markers.push({
          time: c.time, position: "aboveBar",
          color: col, shape: EVENT_STYLE.ai_call.shape,
          text: c.call_type ? c.call_type[0].toUpperCase() : "AI",
          _kind: "ai_call", _meta: c,
        });
      }
    }
    if (chartState.eventsOn.has("fill")) {
      for (const f of chartState.events.fills || []) {
        if (!f.time) continue;
        const isEntry = f.kind === "entry";
        const col = isEntry
          ? (f.side === "long" ? "#3ddc84" : "#ff5d6c")
          : (f.pnl_usd > 0 ? "#3ddc84" : f.pnl_usd < 0 ? "#ff5d6c" : "#f0b429");
        markers.push({
          time: f.time,
          position: isEntry ? "belowBar" : "aboveBar",
          color: col,
          shape: isEntry ? "arrowUp" : "square",
          text: f.kind || "",
          _kind: "fill", _meta: f,
        });
      }
    }
    if (chartState.eventsOn.has("review")) {
      for (const r of chartState.events.reviews || []) {
        if (!r.time) continue;
        markers.push({
          time: r.time, position: "aboveBar",
          color: EVENT_STYLE.review.color, shape: EVENT_STYLE.review.shape,
          text: "R", _kind: "review", _meta: r,
        });
      }
    }
  }

  markers.sort((a, b) => a.time - b.time);
  chartState._markers = markers;
  chartState.candleSeries.setMarkers(markers.map(({_kind, _flag, _meta, ...m}) => m));
}

function onChartClick(param) {
  if (!param || !param.time || !chartState._markers) return;
  // Tolerance = half a bar interval, so clicks anywhere on the bar
  // resolve to its marker(s) regardless of tf (was a hard-coded 60s
  // before, which silently broke everything beyond 1m).
  const tfSec = tfSeconds(chartState.tf);
  const tol = Math.max(30, tfSec / 2);
  const t = param.time;
  const hits = chartState._markers.filter((m) => Math.abs(m.time - t) < tol);
  if (!hits.length) {
    // Fallback: bar info
    const bar = chartState.data.rows.find((r) => r.time === t);
    if (!bar) return;
    q("event-detail").innerHTML = `
      <h3>bar @ ${new Date(t * 1000).toISOString().slice(0,16).replace("T"," ")}</h3>
      <div class="field"><span>open</span><span>${fmtNum(bar.open)}</span></div>
      <div class="field"><span>high</span><span>${fmtNum(bar.high)}</span></div>
      <div class="field"><span>low</span><span>${fmtNum(bar.low)}</span></div>
      <div class="field"><span>close</span><span>${fmtNum(bar.close)}</span></div>
      <div class="field"><span>volume</span><span>${fmtInt(bar.volume)}</span></div>`;
    return;
  }
  q("event-detail").innerHTML = hits.map(renderMarkerDetail).join("<hr/>");
}

function renderMarkerDetail(m) {
  const ts = new Date(m.time * 1000).toISOString().slice(0,19).replace("T"," ") + "Z";
  if (m._kind === "flag") {
    const meta = m._meta;
    const extra = meta.ratio ? `<div class="field"><span>ratio</span><span>${meta.ratio}x</span></div>` : "";
    return `<h3>${escapeHtml(m._flag)} @ ${ts}</h3>
      <div class="field"><span>price</span><span>${fmtNum(meta.price)}</span></div>${extra}`;
  }
  if (m._kind === "trigger") {
    const t = m._meta;
    return `<h3>trigger @ ${ts}</h3>
      <div class="field"><span>fired</span><span>${t.fired ? "yes" : "no"}</span></div>
      <div class="field"><span>flag</span><span>${escapeHtml(t.flag || "—")}</span></div>
      <div class="field"><span>decision</span><span>${escapeHtml(t.decision || "—")}</span></div>
      <div class="field"><span>close</span><span>${fmtNum(t.close)}</span></div>
      <div class="field"><span>atr%</span><span>${fmtNum(t.atr_pct, 2)}</span></div>
      <div class="field"><span>move%</span><span>${fmtNum(t.move_pct ? t.move_pct * 100 : null, 2)}</span></div>
      <p class="dim">${escapeHtml(t.reason || "")}</p>`;
  }
  if (m._kind === "fill") {
    const f = m._meta;
    return `<h3>fill · ${escapeHtml(f.kind || "")} @ ${ts}</h3>
      <div class="field"><span>side</span><span>${escapeHtml(f.side || "")}</span></div>
      <div class="field"><span>price</span><span>${fmtNum(f.price)}</span></div>
      <div class="field"><span>qty</span><span>${fmtNum(f.qty)}</span></div>
      <div class="field"><span>pnl</span><span>${fmtUsd(f.pnl_usd)}</span></div>
      <div class="field"><span>fee</span><span>${fmtUsd(f.fee_usd)}</span></div>
      <p class="dim">${escapeHtml(f.reason || "")}</p>`;
  }
  if (m._kind === "review") {
    const r = m._meta;
    return `<h3>review @ ${ts}</h3>
      <div class="field"><span>action</span><span>${escapeHtml(r.action || "—")}</span></div>
      <div class="field"><span>confidence</span><span>${r.confidence != null ? (r.confidence * 100).toFixed(0) + "%" : "—"}</span></div>
      <div class="field"><span>hook</span><span>${escapeHtml(r.hook_reason || "—")}</span></div>
      <div class="field"><span>new stop</span><span>${fmtNum(r.new_stop_loss)}</span></div>
      <p>${escapeHtml(r.rationale || "")}</p>`;
  }
  if (m._kind === "ai_call") {
    const c = m._meta;
    return `<h3>AI · ${escapeHtml(c.call_type || "")} @ ${ts}</h3>
      <div class="field"><span>action</span><span>${escapeHtml(c.action || "—")}</span></div>
      <div class="field"><span>confidence</span><span>${c.confidence != null ? (c.confidence * 100).toFixed(0) + "%" : "—"}</span></div>
      <div class="field"><span>cost</span><span>${fmtUsd(c.cost_usd)}</span></div>
      <div class="field"><span>model</span><span>${escapeHtml(c.model || "")}</span></div>
      <p class="dim">Open the AI calls tab and search for call_id <code>${escapeHtml(c.call_id || "")}</code> for the full prompt.</p>`;
  }
  return `<pre>${escapeHtml(JSON.stringify(m, null, 2))}</pre>`;
}

function renderSymbolTables() {
  const ev = chartState.events || {};
  // Triggers
  const tt = document.querySelector("#tbl-sym-triggers tbody");
  tt.innerHTML = "";
  const trigs = (ev.triggers || []).slice().reverse();
  q("sym-trig-count").textContent = `(${trigs.length})`;
  for (const t of trigs) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="dim">${escapeHtml(t.time ? new Date(t.time * 1000).toISOString().slice(0,16).replace("T"," ") : "—")}</td>
      <td>${escapeHtml(t.flag || "—")}</td>
      <td>${t.fired ? `<span class="pill pill-ok">${escapeHtml(t.decision || "")}</span>` : `<span class="pill">${escapeHtml(t.decision || "")}</span>`}</td>
      <td class="num">${fmtNum(t.close)}</td>
      <td class="num">${fmtNum(t.atr_pct, 2)}</td>
      <td class="num">${fmtNum(t.move_pct ? t.move_pct * 100 : null, 2)}</td>
      <td class="dim">${escapeHtml(t.reason || "")}</td>`;
    tt.appendChild(tr);
  }
  // AI calls
  const at = document.querySelector("#tbl-sym-ai tbody");
  at.innerHTML = "";
  const ais = (ev.ai_calls || []).slice().reverse();
  q("sym-ai-count").textContent = `(${ais.length})`;
  for (const c of ais) {
    const actionCls = c.action ? `pill-action-${c.action}` : "";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="dim">${escapeHtml(c.time ? new Date(c.time * 1000).toISOString().slice(0,16).replace("T"," ") : "—")}</td>
      <td>${escapeHtml(c.call_type || "")}</td>
      <td>${c.action ? `<span class="pill ${actionCls}">${escapeHtml(c.action)}</span>` : "—"}</td>
      <td class="num">${c.confidence != null ? (c.confidence * 100).toFixed(0) + "%" : "—"}</td>
      <td class="num">${fmtUsd(c.cost_usd)}</td>
      <td class="dim">${escapeHtml(c.model || "")}</td>`;
    at.appendChild(tr);
  }
}

// Hook chart view into nav: load symbols on first activation.
document.querySelector('.nav-item[data-view="chart"]')?.addEventListener("click", async () => {
  if (!chartState.symbols.length) await loadSymbols();
  if (chartState.current && !chartState.data) loadChart();
});

/* ============================================================
   Sortable tables — clickable headers with data-sort=text|num.
   Sort runs against the rendered DOM, so it survives every
   re-render automatically (idempotent: re-applies on each render).
   ============================================================ */
const sortState = new WeakMap(); // table -> {col, dir}
function setupSortableTables() {
  document.querySelectorAll("table[data-sortable]").forEach((tbl) => {
    tbl.querySelectorAll("thead th[data-sort]").forEach((th, idx) => {
      th.addEventListener("click", () => sortTable(tbl, idx, th.dataset.sort));
    });
  });
}
function sortTable(tbl, col, kind) {
  const cur = sortState.get(tbl);
  const dir = cur && cur.col === col && cur.dir === "asc" ? "desc" : "asc";
  sortState.set(tbl, { col, dir, kind });
  applySort(tbl);
  tbl.querySelectorAll("thead th").forEach((th, i) => {
    th.classList.remove("sort-asc", "sort-desc");
    if (i === col) th.classList.add(dir === "asc" ? "sort-asc" : "sort-desc");
  });
}
function applySort(tbl) {
  const s = sortState.get(tbl);
  if (!s) return;
  const tbody = tbl.querySelector("tbody");
  if (!tbody) return;
  const rows = Array.from(tbody.querySelectorAll("tr:not(.detail-row)"));
  const factor = s.dir === "asc" ? 1 : -1;
  const parseCell = (tr) => {
    const td = tr.cells[s.col];
    if (!td) return s.kind === "num" ? -Infinity : "";
    const txt = td.textContent.trim();
    if (s.kind === "num") {
      const n = parseFloat(txt.replace(/[^0-9.\-]/g, ""));
      return Number.isFinite(n) ? n : -Infinity;
    }
    return txt.toLowerCase();
  };
  rows.sort((a, b) => {
    const va = parseCell(a), vb = parseCell(b);
    if (va < vb) return -1 * factor;
    if (va > vb) return 1 * factor;
    return 0;
  });
  // Re-append in order; preserve any expand/detail rows attached
  // immediately after their parent (prompts table).
  for (const tr of rows) {
    tbody.appendChild(tr);
    const next = tr.nextSibling;
    if (next && next.classList && next.classList.contains("detail-row")) {
      tbody.appendChild(next);
    }
  }
}
// Re-apply sort after each table render.
function reapplySorts() {
  document.querySelectorAll("table[data-sortable]").forEach(applySort);
}

/* ============================================================
   Symbol search — filter the chart-view symbol dropdown.
   ============================================================ */
function setupSymbolSearch() {
  const input = q("chart-symbol-search");
  const sel = q("chart-symbol");
  if (!input || !sel) return;
  input.addEventListener("input", () => {
    const needle = input.value.trim().toUpperCase();
    let firstVisible = null;
    Array.from(sel.options).forEach((opt) => {
      const match = !needle || opt.value.toUpperCase().includes(needle);
      opt.hidden = !match;
      if (match && firstVisible === null) firstVisible = opt;
    });
    if (firstVisible && needle) sel.value = firstVisible.value;
  });
}
