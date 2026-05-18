const RUNS = [
  {
    id: "benchmark",
    label: "Benchmark",
    file: "data/benchmark.json",
    color: "#4f6fd8",
    env: "SpreadBenchmark-v0",
    psych: false,
  },
  {
    id: "f1_baseline",
    label: "F1 baseline",
    file: "data/f1_baseline.json",
    color: "#6b7280",
    env: "F1 spread (10 agents)",
    psych: false,
  },
  {
    id: "f1_psych",
    label: "F1 psych-aware",
    file: "data/f1_psych.json",
    color: "#e8002d",
    env: "F1 spread (10 agents)",
    psych: true,
  },
];

const DRIVER_CSV = `code,season_points,n_laps,lap_std,resilience
NOR,394,1413,14.0956,0.35
VER,389,1356,13.7549,0.537747
PIA,381,1376,13.8328,0.493987
RUS,289,1424,13.8747,0.470679
LEC,225,1340,13.9034,0.454769
ANT,135,1291,13.4528,0.712209
HAM,135,1341,13.9005,0.456372
ALB,70,1289,13.6606,0.591356
SAI,54,1231,13.0617,0.95
HUL,51,1269,14.0617,0.36827`;

function parseDrivers(csv) {
  const lines = csv.trim().split("\n");
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const vals = line.split(",");
    const row = {};
    headers.forEach((h, i) => {
      row[h] = vals[i];
    });
    return row;
  });
}

function fmtNum(n, digits = 2) {
  return Number(n).toFixed(digits);
}

function fmtReturn(n) {
  return fmtNum(n, 1);
}

function setMetric(name, text) {
  document.querySelectorAll(`[data-metric="${name}"]`).forEach((el) => {
    el.textContent = text;
  });
}

function populateDriverTable() {
  const tbody = document.querySelector("#driver-table tbody");
  if (!tbody) return;
  parseDrivers(DRIVER_CSV).forEach((d) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${d.code}</strong></td>
      <td>${d.season_points}</td>
      <td>${d.n_laps}</td>
      <td>${fmtNum(d.lap_std, 2)}</td>
      <td>${fmtNum(d.resilience, 3)}</td>`;
    tbody.appendChild(tr);
  });
}

function populateResultsTable(results) {
  const tbody = document.getElementById("results-body");
  if (!tbody) return;
  tbody.innerHTML = "";
  results.forEach((r) => {
    const tr = document.createElement("tr");
    const iters = r.iterations ?? r.data?.length ?? 150;
    tr.innerHTML = `
      <td><strong>${r.label}</strong></td>
      <td>${iters}</td>
      <td>${fmtReturn(r.finalReturn)}</td>
      <td>${fmtNum(r.srs, 3)}</td>`;
    tbody.appendChild(tr);
  });
}

function buildLegend(container, runs) {
  container.innerHTML = "";
  runs.forEach((r) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="swatch" style="background:${r.color}"></span>${r.label}`;
    container.appendChild(li);
  });
}

function chartTheme() {
  const root = getComputedStyle(document.documentElement);
  return {
    bg: root.getPropertyValue("--chart-bg").trim() || "#ffffff",
    grid: root.getPropertyValue("--chart-grid").trim() || "#e2e8f0",
    muted: root.getPropertyValue("--text-muted").trim() || "#5c6478",
    text: root.getPropertyValue("--text").trim() || "#1a1d26",
  };
}

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = rect.width || canvas.width;
  const h = (rect.width ? (canvas.height / canvas.width) * w : canvas.height) || 360;
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  return { ctx, w, h };
}

function drawLineChart(canvas, seriesList) {
  const { ctx, w, h } = setupCanvas(canvas);
  const theme = chartTheme();
  const pad = { top: 24, right: 20, bottom: 36, left: 56 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  let ymin = Infinity;
  let ymax = -Infinity;
  let maxLen = 0;
  seriesList.forEach((s) => {
    maxLen = Math.max(maxLen, s.data.length);
    s.data.forEach((v) => {
      ymin = Math.min(ymin, v);
      ymax = Math.max(ymax, v);
    });
  });
  const yPad = (ymax - ymin) * 0.08 || 10;
  ymin -= yPad;
  ymax += yPad;

  ctx.fillStyle = theme.bg;
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = theme.grid;
  ctx.lineWidth = 1;
  const gridLines = 5;
  for (let i = 0; i <= gridLines; i++) {
    const y = pad.top + (plotH * i) / gridLines;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();
    const val = ymax - ((ymax - ymin) * i) / gridLines;
    ctx.fillStyle = theme.muted;
    ctx.font = "11px Inter, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(fmtNum(val, 0), pad.left - 8, y + 4);
  }

  ctx.fillStyle = theme.muted;
  ctx.textAlign = "center";
  ctx.fillText("Iteration", pad.left + plotW / 2, h - 8);
  ctx.save();
  ctx.translate(14, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Mean episode return", 0, 0);
  ctx.restore();

  seriesList.forEach((s) => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    s.data.forEach((v, i) => {
      const x = pad.left + (i / Math.max(1, s.data.length - 1)) * plotW;
      const y = pad.top + ((ymax - v) / (ymax - ymin)) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
}

function drawSrsChart(canvas, results) {
  const { ctx, w, h } = setupCanvas(canvas);
  const theme = chartTheme();
  const pad = { top: 24, right: 20, bottom: 48, left: 48 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  const maxSrs = 1;

  ctx.fillStyle = theme.bg;
  ctx.fillRect(0, 0, w, h);

  const barW = plotW / results.length;
  results.forEach((r, i) => {
    const barH = (r.srs / maxSrs) * plotH;
    const x = pad.left + i * barW + barW * 0.2;
    const bw = barW * 0.6;
    const y = pad.top + plotH - barH;

    ctx.fillStyle = r.color;
    ctx.beginPath();
    ctx.roundRect(x, y, bw, barH, 4);
    ctx.fill();

    ctx.fillStyle = theme.text;
    ctx.font = "600 12px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(fmtNum(r.srs, 3), x + bw / 2, y - 8);

    ctx.fillStyle = theme.muted;
    ctx.font = "11px Inter, sans-serif";
    const label = r.label.replace(" ", "\n");
    const lines = label.split("\n");
    lines.forEach((line, li) => {
      ctx.fillText(line, x + bw / 2, pad.top + plotH + 16 + li * 14);
    });
  });

  ctx.strokeStyle = theme.grid;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();
}

async function loadRun(run) {
  const res = await fetch(run.file);
  if (!res.ok) throw new Error(`Failed to load ${run.file}`);
  const data = await res.json();
  const rets = data.episode_return_mean_per_iter || [];
  return {
    ...run,
    srs: data.srs_on_iter_means ?? 0,
    finalReturn: rets.length ? rets[rets.length - 1] : 0,
    data: rets,
    iterations: data.iterations ?? rets.length,
  };
}

async function init() {
  populateDriverTable();

  const results = await Promise.all(RUNS.map(loadRun));

  const benchmark = results.find((r) => r.id === "benchmark");
  const baseline = results.find((r) => r.id === "f1_baseline");
  const psych = results.find((r) => r.id === "f1_psych");

  if (benchmark) setMetric("benchmark-srs", fmtNum(benchmark.srs, 3));
  if (baseline) {
    setMetric("baseline-srs", fmtNum(baseline.srs, 3));
  }
  if (psych) {
    setMetric("psych-srs", fmtNum(psych.srs, 3));
    setMetric("psych-return", fmtReturn(psych.finalReturn));
  }

  populateResultsTable(results);

  const legend = document.getElementById("legend-returns");
  if (legend) buildLegend(legend, results);

  const returnsCanvas = document.getElementById("chart-returns");
  const srsCanvas = document.getElementById("chart-srs");

  const draw = () => {
    if (returnsCanvas) {
      drawLineChart(
        returnsCanvas,
        results.map((r) => ({ color: r.color, data: r.data, label: r.label }))
      );
    }
    if (srsCanvas) drawSrsChart(srsCanvas, results);
  };

  draw();
  window.addEventListener("resize", draw);
}

init().catch((err) => {
  console.error(err);
});
