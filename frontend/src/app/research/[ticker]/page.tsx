"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import Script from "next/script";
import clsx from "clsx";

import { fetchResearch } from "@/lib/api";
import type { ResearchData } from "@/lib/types";

/* eslint-disable @typescript-eslint/no-explicit-any */
declare global {
  interface Window {
    Plotly: any;
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────

function fmt(v: number | null | undefined, decimals = 2): string {
  if (v == null) return "N/A";
  return v.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

function fmtLarge(v: number | null | undefined): string {
  if (v == null) return "N/A";
  if (v >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
  if (v >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
  return `$${v.toLocaleString()}`;
}

function fmtVol(v: number): string {
  if (v >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(0)}K`;
  return v.toLocaleString();
}

const PLOTLY_DARK = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(15,23,42,0.4)",
  font: { color: "#94a3b8", family: "system-ui, sans-serif", size: 11 },
  margin: { l: 55, r: 15, t: 30, b: 35 },
  legend: {
    bgcolor: "rgba(0,0,0,0.4)",
    bordercolor: "rgba(255,255,255,0.08)",
    font: { size: 10 },
  },
  xaxis: { gridcolor: "rgba(255,255,255,0.04)", zeroline: false },
  yaxis: { gridcolor: "rgba(255,255,255,0.04)", zeroline: false },
};

// ── Main Component ───────────────────────────────────────────────────────

export default function ResearchPage() {
  const params = useParams<{ ticker: string }>();
  const ticker = decodeURIComponent(params.ticker ?? "").toUpperCase();

  const [data, setData] = useState<ResearchData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [plotlyReady, setPlotlyReady] = useState(false);

  const mainChartRef = useRef<HTMLDivElement>(null);
  const rsiChartRef = useRef<HTMLDivElement>(null);
  const macdChartRef = useRef<HTMLDivElement>(null);

  // Fetch research data
  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchResearch(ticker)
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load data"))
      .finally(() => setLoading(false));
  }, [ticker]);

  // ── Chart Rendering ────────────────────────────────────────────────────

  const renderMainChart = useCallback(
    (d: ResearchData) => {
      const el = mainChartRef.current;
      if (!el || !window.Plotly) return;

      const { dates, open, high, low, close, volume } = d.ohlcv;
      const { sma50, sma200, bollinger } = d.indicators;

      // Volume colors: green if close >= open, red otherwise
      const volColors = close.map((c, i) =>
        (c ?? 0) >= (open[i] ?? 0) ? "rgba(16,185,129,0.45)" : "rgba(239,68,68,0.45)"
      );

      // Fibonacci shapes
      const fibEntries = [
        { key: "0%", val: d.fibonacci.level_0 },
        { key: "23.6%", val: d.fibonacci.level_236 },
        { key: "38.2%", val: d.fibonacci.level_382 },
        { key: "50%", val: d.fibonacci.level_500 },
        { key: "61.8%", val: d.fibonacci.level_618 },
        { key: "78.6%", val: d.fibonacci.level_786 },
        { key: "100%", val: d.fibonacci.level_1000 },
      ];

      const fibShapes = fibEntries.map((f) => ({
        type: "line",
        x0: 0, x1: 1, xref: "paper",
        y0: f.val, y1: f.val, yref: "y",
        line: { color: "rgba(251,191,36,0.22)", width: 1, dash: "dash" },
      }));

      const fibAnnotations = fibEntries.map((f) => ({
        x: 1.01, xref: "paper", y: f.val, yref: "y",
        text: `${f.key}`, showarrow: false,
        font: { size: 8, color: "rgba(251,191,36,0.55)" },
        xanchor: "left",
      }));

      const traces: any[] = [
        // Candlestick
        {
          type: "candlestick", x: dates,
          open, high, low, close,
          name: d.ticker,
          increasing: { line: { color: "#10b981" } },
          decreasing: { line: { color: "#ef4444" } },
          yaxis: "y",
        },
        // Bollinger upper
        {
          type: "scatter", x: dates, y: bollinger.upper,
          mode: "lines", name: "BB Upper",
          line: { color: "rgba(255,255,255,0.15)", width: 1, dash: "dot" },
          yaxis: "y", showlegend: false,
        },
        // Bollinger lower (with fill to upper)
        {
          type: "scatter", x: dates, y: bollinger.lower,
          mode: "lines", name: "BB Lower",
          line: { color: "rgba(255,255,255,0.15)", width: 1, dash: "dot" },
          fill: "tonexty", fillcolor: "rgba(255,255,255,0.02)",
          yaxis: "y", showlegend: false,
        },
        // 50-DMA
        {
          type: "scatter", x: dates, y: sma50,
          mode: "lines", name: "50-DMA",
          line: { color: "#06b6d4", width: 1.5 },
          yaxis: "y",
        },
        // 200-DMA
        {
          type: "scatter", x: dates, y: sma200,
          mode: "lines", name: "200-DMA",
          line: { color: "#f97316", width: 1.5 },
          yaxis: "y",
        },
        // Volume bars
        {
          type: "bar", x: dates, y: volume,
          name: "Volume", marker: { color: volColors },
          yaxis: "y2", showlegend: false,
        },
      ];

      const layout = {
        ...PLOTLY_DARK,
        height: 520,
        yaxis: {
          ...PLOTLY_DARK.yaxis,
          domain: [0.22, 1],
          title: { text: "Price ($)", font: { size: 10 } },
        },
        yaxis2: {
          ...PLOTLY_DARK.yaxis,
          domain: [0, 0.17],
          title: { text: "Volume", font: { size: 10 } },
        },
        xaxis: {
          ...PLOTLY_DARK.xaxis,
          rangeslider: { visible: false },
        },
        shapes: fibShapes,
        annotations: fibAnnotations,
        margin: { l: 55, r: 45, t: 15, b: 35 },
      };

      window.Plotly.newPlot(el, traces, layout, { responsive: true, displayModeBar: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] });
    },
    []
  );

  const renderRsiChart = useCallback(
    (d: ResearchData) => {
      const el = rsiChartRef.current;
      if (!el || !window.Plotly) return;

      const { dates } = d.ohlcv;
      const { rsi } = d.indicators;

      const traces: any[] = [
        {
          type: "scatter", x: dates, y: rsi,
          mode: "lines", name: "RSI (14)",
          line: { color: "#06b6d4", width: 1.5 },
        },
        // Overbought line
        {
          type: "scatter", x: [dates[0], dates[dates.length - 1]], y: [70, 70],
          mode: "lines", name: "Overbought",
          line: { color: "rgba(239,68,68,0.5)", width: 1, dash: "dash" },
          showlegend: false,
        },
        // Oversold line
        {
          type: "scatter", x: [dates[0], dates[dates.length - 1]], y: [30, 30],
          mode: "lines", name: "Oversold",
          line: { color: "rgba(16,185,129,0.5)", width: 1, dash: "dash" },
          showlegend: false,
        },
      ];

      const shapes = [
        // Overbought zone fill
        {
          type: "rect", x0: dates[0], x1: dates[dates.length - 1],
          y0: 70, y1: 100, fillcolor: "rgba(239,68,68,0.05)",
          line: { width: 0 },
        },
        // Oversold zone fill
        {
          type: "rect", x0: dates[0], x1: dates[dates.length - 1],
          y0: 0, y1: 30, fillcolor: "rgba(16,185,129,0.05)",
          line: { width: 0 },
        },
      ];

      const layout = {
        ...PLOTLY_DARK,
        height: 200,
        yaxis: { ...PLOTLY_DARK.yaxis, range: [0, 100], title: { text: "RSI", font: { size: 10 } } },
        xaxis: { ...PLOTLY_DARK.xaxis },
        shapes,
        showlegend: false,
      };

      window.Plotly.newPlot(el, traces, layout, { responsive: true, displayModeBar: false });
    },
    []
  );

  const renderMacdChart = useCallback(
    (d: ResearchData) => {
      const el = macdChartRef.current;
      if (!el || !window.Plotly) return;

      const { dates } = d.ohlcv;
      const { macd } = d.indicators;

      const histColors = macd.histogram.map((v) =>
        (v ?? 0) >= 0 ? "rgba(16,185,129,0.6)" : "rgba(239,68,68,0.6)"
      );

      const traces: any[] = [
        {
          type: "bar", x: dates, y: macd.histogram,
          name: "Histogram", marker: { color: histColors },
        },
        {
          type: "scatter", x: dates, y: macd.macdLine,
          mode: "lines", name: "MACD",
          line: { color: "#06b6d4", width: 1.5 },
        },
        {
          type: "scatter", x: dates, y: macd.signalLine,
          mode: "lines", name: "Signal",
          line: { color: "#f97316", width: 1.5 },
        },
      ];

      const layout = {
        ...PLOTLY_DARK,
        height: 200,
        yaxis: { ...PLOTLY_DARK.yaxis, title: { text: "MACD", font: { size: 10 } } },
        xaxis: { ...PLOTLY_DARK.xaxis },
        barmode: "relative",
      };

      window.Plotly.newPlot(el, traces, layout, { responsive: true, displayModeBar: false });
    },
    []
  );

  // Render all charts when data and Plotly are ready
  useEffect(() => {
    if (!data || !plotlyReady) return;
    renderMainChart(data);
    renderRsiChart(data);
    renderMacdChart(data);
  }, [data, plotlyReady, renderMainChart, renderRsiChart, renderMacdChart]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (typeof window !== "undefined" && window.Plotly) {
        [mainChartRef, rsiChartRef, macdChartRef].forEach((ref) => {
          if (ref.current) {
            try { window.Plotly.purge(ref.current); } catch { /* ignore */ }
          }
        });
      }
    };
  }, []);

  // ── Loading State ──────────────────────────────────────────────────────

  if (loading) {
    return (
      <div className="mx-auto max-w-[1600px] px-4 py-8">
        <Link href="/" className="inline-flex items-center gap-1.5 text-sm text-slate-400 hover:text-slate-200 transition-colors">
          <ArrowLeftIcon /> Back to Dashboard
        </Link>
        <div className="mt-16 flex flex-col items-center justify-center gap-4">
          <div className="h-10 w-10 animate-spin rounded-full border-2 border-slate-700 border-t-amber-400" />
          <p className="text-sm text-slate-400">Loading research data for <span className="font-semibold text-slate-200">{ticker}</span>...</p>
          <p className="text-xs text-slate-500">Fetching 365 days of historical data and computing indicators</p>
        </div>
      </div>
    );
  }

  // ── Error State ────────────────────────────────────────────────────────

  if (error) {
    return (
      <div className="mx-auto max-w-[1600px] px-4 py-8">
        <Link href="/" className="inline-flex items-center gap-1.5 text-sm text-slate-400 hover:text-slate-200 transition-colors">
          <ArrowLeftIcon /> Back to Dashboard
        </Link>
        <div className="mt-16 rounded-xl border border-rose-900/60 bg-rose-950/20 p-8 text-center">
          <p className="text-lg font-semibold text-rose-300">Failed to load research for {ticker}</p>
          <p className="mt-2 text-sm text-rose-400/80">{error}</p>
          <button
            className="mt-4 rounded-md bg-slate-800 px-4 py-2 text-sm font-medium text-slate-200 hover:bg-slate-700 transition-colors"
            onClick={() => {
              setLoading(true);
              setError(null);
              fetchResearch(ticker, true)
                .then(setData)
                .catch((e) => setError(e instanceof Error ? e.message : "Failed"))
                .finally(() => setLoading(false));
            }}
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!data) return null;

  // ── Derived values ─────────────────────────────────────────────────────

  const isPositive = data.change >= 0;
  const crossoverLabel: Record<string, { text: string; color: string }> = {
    golden_cross: { text: "Golden Cross Active", color: "text-amber-300" },
    death_cross: { text: "Death Cross Active", color: "text-violet-300" },
    near_golden_cross: { text: "Near Golden Cross", color: "text-amber-400" },
    near_death_cross: { text: "Near Death Cross", color: "text-violet-400" },
    none: { text: "No Crossover Signal", color: "text-slate-400" },
  };
  const cs = crossoverLabel[data.crossover.signal] ?? crossoverLabel.none;

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div className="mx-auto max-w-[1600px] px-4 py-8">
      {/* Plotly CDN */}
      <Script
        src="https://cdn.plot.ly/plotly-latest.min.js"
        strategy="afterInteractive"
        onLoad={() => setPlotlyReady(true)}
      />

      {/* Back nav */}
      <Link href="/" className="inline-flex items-center gap-1.5 text-sm text-slate-400 hover:text-slate-200 transition-colors">
        <ArrowLeftIcon /> Back to Dashboard
      </Link>

      {/* ── Header ── */}
      <header className="mt-5 flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-100">{data.ticker}</h1>
          <p className="text-sm text-slate-400">
            {data.companyName}{data.sector ? ` · ${data.sector}` : ""}
          </p>
        </div>
        <div className="flex items-baseline gap-3">
          <span className="text-2xl font-bold text-slate-100">${fmt(data.currentPrice)}</span>
          <span className={clsx("text-lg font-semibold", isPositive ? "text-emerald-400" : "text-rose-400")}>
            {isPositive ? "+" : ""}{fmt(data.change)} ({isPositive ? "+" : ""}{fmt(data.changePct)}%)
          </span>
        </div>
      </header>

      {/* ── Stats Bar ── */}
      <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-7">
        <StatCard label="Volume" value={fmtVol(data.volume)} />
        <StatCard label="Avg Volume" value={fmtVol(data.avgVolume)} />
        <StatCard label="Market Cap" value={fmtLarge(data.fundamentals.marketCap)} />
        <StatCard label="P/E (Trailing)" value={data.fundamentals.trailingPE != null ? fmt(data.fundamentals.trailingPE) : "N/A"} />
        <StatCard label="P/E (Forward)" value={data.fundamentals.forwardPE != null ? fmt(data.fundamentals.forwardPE) : "N/A"} />
        <StatCard label="Beta" value={data.fundamentals.beta != null ? fmt(data.fundamentals.beta) : "N/A"} />
        <StatCard label="RSI (14)" value={data.latestRSI != null ? fmt(data.latestRSI) : "N/A"}
          valueClass={data.latestRSI != null ? (data.latestRSI > 70 ? "text-rose-400" : data.latestRSI < 30 ? "text-emerald-400" : "text-slate-100") : undefined}
        />
      </div>

      {/* ── Crossover Status ── */}
      <div className="mt-5 rounded-xl border border-slate-800 bg-slate-900/30 p-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-6">
            <div>
              <div className="text-[10px] uppercase tracking-wider text-slate-500">50-DMA</div>
              <div className="text-lg font-bold text-cyan-400">${fmt(data.crossover.dma50)}</div>
            </div>
            <div className="text-slate-600">←→</div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-slate-500">200-DMA</div>
              <div className="text-lg font-bold text-orange-400">${fmt(data.crossover.dma200)}</div>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-slate-500">Gap</div>
              <div className="text-lg font-bold text-slate-200">
                {data.crossover.gapPct != null ? `${data.crossover.gapPct > 0 ? "+" : ""}${fmt(data.crossover.gapPct)}%` : "N/A"}
              </div>
            </div>
          </div>
          <div className={clsx("rounded-full border px-4 py-1.5 text-sm font-semibold", {
            "border-amber-500/40 bg-amber-500/10 text-amber-300": data.crossover.signal.includes("golden"),
            "border-violet-500/40 bg-violet-500/10 text-violet-300": data.crossover.signal.includes("death"),
            "border-slate-700 bg-slate-800/50 text-slate-400": data.crossover.signal === "none",
          })}>
            {cs.text}
          </div>
        </div>
      </div>

      {/* ── Main Price Chart ── */}
      <div className="mt-5 rounded-xl border border-slate-800 bg-slate-900/30 p-4">
        <h2 className="mb-2 text-sm font-semibold text-slate-200">
          Price Chart · 50-DMA · 200-DMA · Bollinger Bands · Fibonacci
        </h2>
        <div ref={mainChartRef} className="w-full" style={{ minHeight: 520 }}>
          {!plotlyReady && <div className="flex h-[520px] items-center justify-center text-sm text-slate-500">Loading chart engine...</div>}
        </div>
      </div>

      {/* ── Fibonacci Levels ── */}
      <div className="mt-5 rounded-xl border border-amber-500/20 bg-amber-950/5 p-4">
        <h2 className="mb-3 text-sm font-semibold text-amber-300">Fibonacci Retracement Levels</h2>
        <div className="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-7">
          {[
            { label: "0% (High)", val: data.fibonacci.level_0 },
            { label: "23.6%", val: data.fibonacci.level_236 },
            { label: "38.2%", val: data.fibonacci.level_382 },
            { label: "50%", val: data.fibonacci.level_500 },
            { label: "61.8%", val: data.fibonacci.level_618 },
            { label: "78.6%", val: data.fibonacci.level_786 },
            { label: "100% (Low)", val: data.fibonacci.level_1000 },
          ].map((f) => (
            <div key={f.label} className="rounded-lg border border-amber-500/10 bg-slate-950/40 px-3 py-2 text-center">
              <div className="text-xs text-amber-400/70">{f.label}</div>
              <div className="text-sm font-semibold text-slate-200">${fmt(f.val)}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── RSI & MACD Charts ── */}
      <div className="mt-5 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-4">
          <h2 className="mb-2 text-sm font-semibold text-slate-200">RSI (Relative Strength Index)</h2>
          <div ref={rsiChartRef} className="w-full" style={{ minHeight: 200 }}>
            {!plotlyReady && <div className="flex h-[200px] items-center justify-center text-sm text-slate-500">Loading...</div>}
          </div>
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-4">
          <h2 className="mb-2 text-sm font-semibold text-slate-200">MACD (Moving Average Convergence Divergence)</h2>
          <div ref={macdChartRef} className="w-full" style={{ minHeight: 200 }}>
            {!plotlyReady && <div className="flex h-[200px] items-center justify-center text-sm text-slate-500">Loading...</div>}
          </div>
        </div>
      </div>

      {/* ── 52-Week Range ── */}
      {(data.fundamentals.fiftyTwoWeekLow != null && data.fundamentals.fiftyTwoWeekHigh != null) && (
        <div className="mt-5 rounded-xl border border-slate-800 bg-slate-900/30 p-4">
          <h2 className="mb-3 text-sm font-semibold text-slate-200">52-Week Range</h2>
          <div className="flex items-center gap-3">
            <span className="text-sm text-slate-400">${fmt(data.fundamentals.fiftyTwoWeekLow)}</span>
            <div className="relative flex-1 h-2 rounded-full bg-slate-800">
              {(() => {
                const lo = data.fundamentals.fiftyTwoWeekLow!;
                const hi = data.fundamentals.fiftyTwoWeekHigh!;
                const pct = hi > lo ? ((data.currentPrice - lo) / (hi - lo)) * 100 : 50;
                return (
                  <>
                    <div className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-emerald-500 to-cyan-500" style={{ width: `${Math.min(100, Math.max(0, pct))}%` }} />
                    <div className="absolute top-1/2 -translate-y-1/2 h-4 w-1 rounded-full bg-white shadow" style={{ left: `${Math.min(100, Math.max(0, pct))}%` }} />
                  </>
                );
              })()}
            </div>
            <span className="text-sm text-slate-400">${fmt(data.fundamentals.fiftyTwoWeekHigh)}</span>
          </div>
        </div>
      )}

      {/* ── Quantitative Trading Strategies ── */}
      <div className="mt-8">
        <h2 className="text-lg font-semibold tracking-tight text-slate-100">Quantitative Trading Strategies</h2>
        <p className="mt-1 text-sm text-slate-400">Nine algorithmic models analyzing {data.ticker} from different angles.</p>

        <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {data.strategies.map((s) => (
            <StrategyCard key={s.name} strategy={s} />
          ))}
        </div>
      </div>

      {/* Footer spacer */}
      <div className="h-12" />
    </div>
  );
}

// ── Sub-Components ───────────────────────────────────────────────────────

function ArrowLeftIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
      <path fillRule="evenodd" d="M17 10a.75.75 0 01-.75.75H5.612l4.158 3.96a.75.75 0 11-1.04 1.08l-5.5-5.25a.75.75 0 010-1.08l5.5-5.25a.75.75 0 111.04 1.08L5.612 9.25H16.25A.75.75 0 0117 10z" clipRule="evenodd" />
    </svg>
  );
}

function StatCard({ label, value, valueClass }: { label: string; value: string; valueClass?: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-2.5">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={clsx("mt-0.5 text-sm font-semibold", valueClass ?? "text-slate-100")}>{value}</div>
    </div>
  );
}

function StrategyCard({ strategy }: { strategy: ResearchData["strategies"][number] }) {
  const s = strategy;
  const [expanded, setExpanded] = useState(false);

  const signalColor: Record<string, string> = {
    BUY: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
    SELL: "bg-rose-500/20 text-rose-300 border-rose-500/30",
    NEUTRAL: "bg-amber-500/20 text-amber-300 border-amber-500/30",
  };
  const barColor: Record<string, string> = {
    BUY: "bg-emerald-500",
    SELL: "bg-rose-500",
    NEUTRAL: "bg-amber-500",
  };
  const reasonBorder: Record<string, string> = {
    BUY: "border-emerald-500/20 bg-emerald-950/20",
    SELL: "border-rose-500/20 bg-rose-950/20",
    NEUTRAL: "border-amber-500/20 bg-amber-950/20",
  };

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-5 transition-all hover:border-slate-700 hover:shadow-lg hover:shadow-slate-900/30">
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="text-sm font-semibold text-slate-100">
          {s.icon} {s.name}
        </div>
        <span className={clsx("rounded-full border px-2.5 py-0.5 text-[11px] font-bold uppercase tracking-wide", signalColor[s.signal])}>
          {s.signal}
        </span>
      </div>

      {/* Description */}
      <p className="mt-2 text-xs text-slate-400 leading-relaxed">{s.description}</p>

      {/* Confidence bar */}
      <div className="mt-3">
        <div className="h-1.5 w-full rounded-full bg-slate-800">
          <div
            className={clsx("h-full rounded-full transition-all", barColor[s.signal])}
            style={{ width: `${s.confidence}%` }}
          />
        </div>
        <div className="mt-1 text-center text-[11px] font-semibold text-slate-300">{s.confidence.toFixed(1)}% Confidence</div>
      </div>

      {/* Reasoning toggle */}
      {s.reasoning && (
        <div className="mt-3">
          <button
            className="flex w-full items-center justify-between rounded-lg border border-slate-700/50 bg-slate-950/30 px-3 py-2 text-left text-[11px] font-medium text-slate-300 hover:bg-slate-950/50 transition-colors"
            onClick={() => setExpanded((v) => !v)}
          >
            <span>Why {s.signal}?</span>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
              className={clsx("h-3.5 w-3.5 text-slate-500 transition-transform", expanded && "rotate-180")}
            >
              <path fillRule="evenodd" d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z" clipRule="evenodd" />
            </svg>
          </button>
          {expanded && (
            <div className={clsx("mt-2 rounded-lg border p-3 text-xs leading-relaxed text-slate-300", reasonBorder[s.signal])}>
              {s.reasoning}
            </div>
          )}
        </div>
      )}

      {/* Metrics */}
      <div className="mt-3 grid grid-cols-2 gap-2">
        {Object.entries(s.metrics).map(([key, value]) => (
          <div key={key} className="rounded-md bg-slate-950/40 px-2 py-1.5 text-center">
            <div className="text-[10px] text-slate-500">{key}</div>
            <div className="text-xs font-semibold text-slate-200">{value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
