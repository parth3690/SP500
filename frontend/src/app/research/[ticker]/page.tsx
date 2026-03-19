"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import Script from "next/script";
import clsx from "clsx";

import { fetchResearch } from "@/lib/api";
import {
  getOptionSuggestion,
  getLeapsSuggestion,
  getFactorBasedSuggestion,
  getFactorBasedLeapsSuggestion,
  type FactorInputs,
} from "@/lib/optionSuggestions";
import { formatNumber, formatLarge as formatLargeMoney, formatVol } from "@/lib/format";
import { toISODate, daysAgo } from "@/lib/date";
import { getVolatilityAndPathSignals, type VolatilityPathSignals } from "@/lib/screening";
import type { ResearchData } from "@/lib/types";

/* eslint-disable @typescript-eslint/no-explicit-any */
declare global {
  interface Window {
    Plotly: any;
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────

const fmt = formatNumber;
const fmtLarge = formatLargeMoney;
const fmtVol = formatVol;

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

const DATE_PRESETS: { label: string; days: number }[] = [
  { label: "1M", days: 30 },
  { label: "3M", days: 90 },
  { label: "6M", days: 180 },
  { label: "1Y", days: 365 },
  { label: "2Y", days: 730 },
  { label: "5Y", days: 1825 },
];

// ── Main Component ───────────────────────────────────────────────────────

export default function ResearchPage() {
  const params = useParams<{ ticker: string }>();
  const ticker = decodeURIComponent(params.ticker ?? "").toUpperCase();

  const [data, setData] = useState<ResearchData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [plotlyReady, setPlotlyReady] = useState(false);

  // Date range state
  const [startDate, setStartDate] = useState(() => daysAgo(365));
  const [endDate, setEndDate] = useState(() => toISODate(new Date()));
  const [activePreset, setActivePreset] = useState<number | null>(365);

  // Search state
  const router = useRouter();
  const [searchQuery, setSearchQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);

  // Close search dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setSearchOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const mainChartRef = useRef<HTMLDivElement>(null);
  const rsiChartRef = useRef<HTMLDivElement>(null);
  const macdChartRef = useRef<HTMLDivElement>(null);

  // Price overlays
  const [showEma20, setShowEma20] = useState(false);
  const [showEma50, setShowEma50] = useState(false);
  const [showEma200, setShowEma200] = useState(false);

  // Single computation for GBM/Monte Carlo (shared by factor inputs + screening).
  // Must be called unconditionally to keep React hook order stable.
  const volatilitySignals = useMemo(() => {
    if (!data?.ohlcv?.close) return null;
    const validClose = data.ohlcv.close.filter((c): c is number => c != null);
    return getVolatilityAndPathSignals(validClose, data.currentPrice);
  }, [data]);

  // ── Data fetcher ────────────────────────────────────────────────────────
  const loadData = useCallback(
    async (start: string, end: string, refresh?: boolean) => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetchResearch(ticker, {
          start,
          end,
          refresh: refresh === true,
        });
        setData(res);
        if (res.dateRangeStart) setStartDate(res.dateRangeStart);
        if (res.dateRangeEnd) setEndDate(res.dateRangeEnd);
      } catch (e) {
        setData(null);
        setError(e instanceof Error ? e.message : "Failed to load data");
      } finally {
        setLoading(false);
      }
    },
    [ticker]
  );

  // Search submit handler (defined after loadData)
  const handleSearchSubmit = useCallback(
    (q?: string) => {
      const t = (q ?? searchQuery).trim().toUpperCase();
      if (!t) return;
      setSearchQuery("");
      setSearchOpen(false);
      if (t !== ticker) {
        router.push(`/research/${encodeURIComponent(t)}`);
      } else {
        loadData(startDate, endDate, true);
      }
    },
    [searchQuery, ticker, startDate, endDate, router, loadData]
  );

  // Fetch research data on mount / ticker change
  useEffect(() => {
    loadData(startDate, endDate);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticker]);

  // ── Chart Rendering ────────────────────────────────────────────────────

  function computeEma(values: (number | null)[], period: number): (number | null)[] {
    const out: (number | null)[] = new Array(values.length).fill(null);
    let prevEma: number | null = null;
    const k = 2 / (period + 1);
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v == null) {
        out[i] = prevEma;
        continue;
      }
      if (prevEma == null) {
        prevEma = v;
      } else {
        prevEma = v * k + prevEma * (1 - k);
      }
      out[i] = prevEma;
    }
    return out;
  }

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
      ];

      // Optional EMA overlays
      if (showEma20 || showEma50 || showEma200) {
        const ema20 = showEma20 ? computeEma(close, 20) : null;
        const ema50 = showEma50 ? computeEma(close, 50) : null;
        const ema200 = showEma200 ? computeEma(close, 200) : null;
        if (ema20) {
          traces.push({
            type: "scatter",
            x: dates,
            y: ema20,
            mode: "lines",
            name: "20-EMA",
            line: { color: "#22c55e", width: 1.2, dash: "dot" },
            yaxis: "y",
          });
        }
        if (ema50) {
          traces.push({
            type: "scatter",
            x: dates,
            y: ema50,
            mode: "lines",
            name: "50-EMA",
            line: { color: "#0ea5e9", width: 1.2, dash: "dot" },
            yaxis: "y",
          });
        }
        if (ema200) {
          traces.push({
            type: "scatter",
            x: dates,
            y: ema200,
            mode: "lines",
            name: "200-EMA",
            line: { color: "#f97316", width: 1.2, dash: "dot" },
            yaxis: "y",
          });
        }
      }

      traces.push(
        // Volume bars
        {
          type: "bar", x: dates, y: volume,
          name: "Volume", marker: { color: volColors },
          yaxis: "y2", showlegend: false,
        }
      );

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
    [showEma20, showEma50, showEma200]
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
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <Link href="/" className="inline-flex items-center gap-1.5 text-sm text-slate-400 hover:text-slate-200 transition-colors">
            <ArrowLeftIcon /> Back to Dashboard
          </Link>
          <div ref={searchRef} className="relative w-full sm:w-auto">
            <form onSubmit={(e) => { e.preventDefault(); handleSearchSubmit(); }} className="flex items-center gap-2">
              <div className="relative flex-1 sm:w-72">
                <SearchIcon />
                <input type="text" value={searchQuery} placeholder="Search any ticker..." className="w-full rounded-lg border border-slate-700 bg-slate-950/60 py-2 pl-9 pr-3 text-sm text-slate-200 placeholder-slate-500 outline-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/25 transition-colors" onChange={(e) => setSearchQuery(e.target.value)} />
              </div>
              <button type="submit" disabled={!searchQuery.trim()} className="rounded-lg bg-amber-500/20 border border-amber-500/30 px-4 py-2 text-sm font-semibold text-amber-300 hover:bg-amber-500/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed">Analyze</button>
            </form>
          </div>
        </div>
        <div className="mt-16 flex flex-col items-center justify-center gap-4">
          <div className="h-10 w-10 animate-spin rounded-full border-2 border-slate-700 border-t-amber-400" />
          <p className="text-sm text-slate-400">Loading research data for <span className="font-semibold text-slate-200">{ticker}</span>...</p>
          <p className="text-xs text-slate-500">Fetching data from {startDate} to {endDate} and computing indicators</p>
        </div>
      </div>
    );
  }

  // ── Error State ────────────────────────────────────────────────────────

  if (error) {
    return (
      <div className="mx-auto max-w-[1600px] px-4 py-8">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <Link href="/" className="inline-flex items-center gap-1.5 text-sm text-slate-400 hover:text-slate-200 transition-colors">
            <ArrowLeftIcon /> Back to Dashboard
          </Link>
          <div ref={searchRef} className="relative w-full sm:w-auto">
            <form onSubmit={(e) => { e.preventDefault(); handleSearchSubmit(); }} className="flex items-center gap-2">
              <div className="relative flex-1 sm:w-72">
                <SearchIcon />
                <input type="text" value={searchQuery} placeholder="Search any ticker..." className="w-full rounded-lg border border-slate-700 bg-slate-950/60 py-2 pl-9 pr-3 text-sm text-slate-200 placeholder-slate-500 outline-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/25 transition-colors" onChange={(e) => setSearchQuery(e.target.value)} />
              </div>
              <button type="submit" disabled={!searchQuery.trim()} className="rounded-lg bg-amber-500/20 border border-amber-500/30 px-4 py-2 text-sm font-semibold text-amber-300 hover:bg-amber-500/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed">Analyze</button>
            </form>
          </div>
        </div>
        <div className="mt-16 rounded-xl border border-rose-900/60 bg-rose-950/20 p-8 text-center">
          <p className="text-lg font-semibold text-rose-300">Failed to load research for {ticker}</p>
          <p className="mt-2 text-sm text-rose-400/80">{error}</p>
          <button
            className="mt-4 rounded-md bg-slate-800 px-4 py-2 text-sm font-medium text-slate-200 hover:bg-slate-700 transition-colors"
            onClick={() => loadData(startDate, endDate, true)}
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

  // Prefer backend-provided latestRSI, but fall back to the most recent non-null
  // value from the RSI indicator series so non-index tickers still get signals.
  const rsiSeries = data.indicators.rsi ?? [];
  let seriesLatestRsi: number | null = null;
  for (let i = rsiSeries.length - 1; i >= 0; i -= 1) {
    const v = rsiSeries[i];
    if (v != null) {
      seriesLatestRsi = v;
      break;
    }
  }
  const latestRsi = data.latestRSI ?? seriesLatestRsi;
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

      {/* Top bar: Back nav + Search */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <Link href="/" className="inline-flex items-center gap-1.5 text-sm text-slate-400 hover:text-slate-200 transition-colors">
          <ArrowLeftIcon /> Back to Dashboard
        </Link>

        {/* Search any ticker */}
        <div ref={searchRef} className="relative w-full sm:w-auto">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSearchSubmit();
            }}
            className="flex items-center gap-2"
          >
            <div className="relative flex-1 sm:w-72">
              <SearchIcon />
              <input
                type="text"
                value={searchQuery}
                placeholder="Search any ticker (e.g. AAPL, MSFT, TSLA)..."
                className="w-full rounded-lg border border-slate-700 bg-slate-950/60 py-2 pl-9 pr-3 text-sm text-slate-200 placeholder-slate-500 outline-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/25 transition-colors"
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  setSearchOpen(e.target.value.trim().length > 0);
                }}
                onFocus={() => {
                  if (searchQuery.trim()) setSearchOpen(true);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Escape") {
                    setSearchOpen(false);
                    setSearchQuery("");
                  }
                }}
              />
            </div>
            <button
              type="submit"
              disabled={!searchQuery.trim()}
              className="rounded-lg bg-amber-500/20 border border-amber-500/30 px-4 py-2 text-sm font-semibold text-amber-300 hover:bg-amber-500/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Analyze
            </button>
          </form>

          {/* Quick suggestions dropdown */}
          {searchOpen && searchQuery.trim().length > 0 && (
            <div className="absolute left-0 right-0 top-full z-50 mt-1 rounded-lg border border-slate-700 bg-slate-900 shadow-xl shadow-black/40 sm:right-auto sm:w-72">
              <div className="p-2">
                <div className="px-2 py-1 text-[10px] uppercase tracking-wider text-slate-500">
                  Press Enter to analyze
                </div>
                <button
                  className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm text-slate-200 hover:bg-slate-800 transition-colors"
                  onClick={() => handleSearchSubmit()}
                >
                  <span className="font-mono font-bold text-amber-400">{searchQuery.trim().toUpperCase()}</span>
                  <span className="text-xs text-slate-400">— Deep research</span>
                </button>
              </div>
              <div className="border-t border-slate-800 p-2">
                <div className="px-2 py-1 text-[10px] uppercase tracking-wider text-slate-500">
                  Popular tickers
                </div>
                {["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM"].filter(
                  (t) => t !== ticker && t.includes(searchQuery.trim().toUpperCase())
                ).slice(0, 5).map((t) => (
                  <button
                    key={t}
                    className="flex w-full items-center gap-2 rounded-md px-3 py-1.5 text-left text-sm text-slate-300 hover:bg-slate-800 transition-colors"
                    onClick={() => handleSearchSubmit(t)}
                  >
                    <span className="font-mono font-semibold text-slate-200">{t}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Header ── */}
      <header className="mt-5 flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-100">{data.ticker}</h1>
          <p className="text-sm text-slate-400">
            {data.companyName}{data.sector ? ` · ${data.sector}` : ""}
          </p>
        </div>
        <div className="flex flex-col items-start gap-0.5 text-right sm:items-end">
          <div className="flex items-baseline gap-3">
            <span className="text-2xl font-bold text-slate-100">${fmt(data.currentPrice)}</span>
            <span className={clsx("text-lg font-semibold", isPositive ? "text-emerald-400" : "text-rose-400")}>
              {isPositive ? "+" : ""}{fmt(data.change)} ({isPositive ? "+" : ""}{fmt(data.changePct)}%)
            </span>
          </div>
          <span className="text-[11px] text-slate-500">
            Last close for selected range · {data.dateRangeEnd}
          </span>
        </div>
      </header>

      {/* ── Date Range Picker ── */}
      <div className="mt-5 rounded-xl border border-slate-800 bg-slate-900/30 p-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          {/* Preset buttons */}
          <div className="flex items-center gap-1.5">
            <span className="mr-1.5 text-xs font-medium uppercase tracking-wider text-slate-500">Range</span>
            {DATE_PRESETS.map((p) => (
              <button
                key={p.days}
                className={clsx(
                  "rounded-md px-2.5 py-1 text-xs font-semibold transition-colors",
                  activePreset === p.days
                    ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                    : "bg-slate-800/60 text-slate-400 border border-slate-700/50 hover:bg-slate-700/60 hover:text-slate-200"
                )}
                onClick={() => {
                  const newStart = daysAgo(p.days);
                  const newEnd = toISODate(new Date());
                  setStartDate(newStart);
                  setEndDate(newEnd);
                  setActivePreset(p.days);
                  loadData(newStart, newEnd);
                }}
              >
                {p.label}
              </button>
            ))}
          </div>

          {/* Custom date inputs */}
          <div className="flex items-center gap-2">
            <label className="text-xs text-slate-500">From</label>
            <input
              type="date"
              value={startDate}
              max={endDate}
              className="rounded-md border border-slate-700 bg-slate-950/60 px-2.5 py-1.5 text-xs text-slate-200 outline-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/25 transition-colors [color-scheme:dark]"
              onChange={(e) => {
                setStartDate(e.target.value);
                setActivePreset(null);
              }}
            />
            <label className="text-xs text-slate-500">To</label>
            <input
              type="date"
              value={endDate}
              min={startDate}
              max={toISODate(new Date())}
              className="rounded-md border border-slate-700 bg-slate-950/60 px-2.5 py-1.5 text-xs text-slate-200 outline-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/25 transition-colors [color-scheme:dark]"
              onChange={(e) => {
                setEndDate(e.target.value);
                setActivePreset(null);
              }}
            />
            <button
              className="ml-1 rounded-md bg-amber-500/20 border border-amber-500/30 px-3 py-1.5 text-xs font-semibold text-amber-300 hover:bg-amber-500/30 transition-colors disabled:opacity-40"
              disabled={loading}
              onClick={() => loadData(startDate, endDate)}
            >
              {loading ? (
                <span className="flex items-center gap-1.5">
                  <span className="inline-block h-3 w-3 animate-spin rounded-full border border-amber-400/40 border-t-amber-400" />
                  Loading...
                </span>
              ) : (
                "Apply"
              )}
            </button>
            <button
              className="rounded-md bg-slate-800/60 border border-slate-700/50 px-3 py-1.5 text-xs font-medium text-slate-400 hover:bg-slate-700/60 hover:text-slate-200 transition-colors disabled:opacity-40"
              disabled={loading}
              onClick={() => loadData(startDate, endDate, true)}
              title="Refresh data (bypass cache)"
            >
              <RefreshIcon />
            </button>
          </div>
        </div>

        {/* Date range summary */}
        {data && (
          <div className="mt-2.5 flex items-center gap-2 text-[11px] text-slate-500">
            <CalendarIcon />
            <span>Showing data from <span className="text-slate-300">{data.dateRangeStart}</span> to <span className="text-slate-300">{data.dateRangeEnd}</span></span>
            <span className="text-slate-600">·</span>
            <span>{data.ohlcv.dates.length} trading days</span>
          </div>
        )}
      </div>

      {/* ── Stats Bar ── */}
      <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-7">
        <StatCard label="Volume" value={fmtVol(data.volume)} />
        <StatCard label="Avg Volume" value={fmtVol(data.avgVolume)} />
        <StatCard label="Market Cap" value={fmtLarge(data.fundamentals.marketCap)} />
        <StatCard label="P/E (Trailing)" value={data.fundamentals.trailingPE != null ? fmt(data.fundamentals.trailingPE) : "N/A"} />
        <StatCard label="P/E (Forward)" value={data.fundamentals.forwardPE != null ? fmt(data.fundamentals.forwardPE) : "N/A"} />
        <StatCard label="Beta" value={data.fundamentals.beta != null ? fmt(data.fundamentals.beta) : "N/A"} />
        <StatCard
          label="RSI (14)"
          value={latestRsi != null ? fmt(latestRsi) : "N/A"}
          valueClass={
            latestRsi != null
              ? latestRsi > 70
                ? "text-rose-400"
                : latestRsi < 30
                  ? "text-emerald-400"
                  : "text-slate-100"
              : undefined
          }
        />
      </div>

      {/* ── Option suggestion (RSI-based or factor-based) ── */}
      {(() => {
        const rsi = latestRsi;
        const context =
          rsi != null && rsi <= 35
            ? "daily_oversold"
            : rsi != null && rsi >= 65
              ? "daily_overbought"
              : null;
        const rsiSuggestion = context ? getOptionSuggestion(context, rsi!, data.currentPrice) : null;
        const factorInputs = getFactorInputs(data, volatilitySignals);
        const factorSuggestion = rsiSuggestion ? null : getFactorBasedSuggestion(factorInputs);
        const suggestion = rsiSuggestion ?? factorSuggestion;
        if (!suggestion) return null;
        const isRsiBased = !!rsiSuggestion;
        const isOversold = isRsiBased && context === "daily_oversold";
        const factorsUsed = "factorsUsed" in suggestion ? (suggestion as { factorsUsed: string[] }).factorsUsed : null;
        return (
          <div
            className={clsx(
              "mt-5 rounded-xl border p-4",
              isRsiBased
                ? isOversold
                  ? "border-emerald-700/50 bg-emerald-950/20"
                  : "border-rose-700/50 bg-rose-950/20"
                : "border-cyan-700/50 bg-cyan-950/20",
            )}
          >
            <h2 className="text-sm font-semibold text-slate-200 mb-2">
              {isRsiBased ? "Option suggestion (RSI-based)" : "Option suggestion (factor-based)"}
            </h2>
            <p className="text-xs text-slate-400 mb-3">
              {isRsiBased
                ? `Strike and expiry ideas where probability of profit is higher, based on current RSI (${rsi != null ? fmt(rsi) : "N/A"}).`
                : "No RSI signal; suggestion from crossover, GBM, Monte Carlo, 52w range. Use backtest key to filter in backtests."}
            </p>
            <div className="grid gap-2 text-sm">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-[10px] uppercase tracking-wider text-slate-500">Strategy</span>
                <span
                  className={clsx(
                    "font-medium",
                    isRsiBased ? (isOversold ? "text-emerald-300" : "text-rose-300") : "text-cyan-300",
                  )}
                >
                  {suggestion.strategy}
                </span>
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-0.5">Strike</div>
                <p className="text-slate-200 text-xs">{suggestion.strikeSuggestion}</p>
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-0.5">Expiry</div>
                <p className="text-slate-200 text-xs">{suggestion.expirySuggestion}</p>
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-0.5">Rationale</div>
                <p className="text-slate-400 text-xs leading-relaxed">{suggestion.rationale}</p>
              </div>
              {factorsUsed && factorsUsed.length > 0 && (
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-0.5">Backtest key</div>
                  <p className="text-xs text-cyan-300/90 font-mono">
                    {factorsUsed.join(" · ")}
                  </p>
                </div>
              )}
            </div>
          </div>
        );
      })()}

      {/* ── LEAPS suggestion (always shown; RSI or factor-based; "No suggestions" only when neither) ── */}
      {(() => {
        const rsiLeaps = getLeapsSuggestion(latestRsi ?? null, data.currentPrice);
        const factorInputs = getFactorInputs(data, volatilitySignals);
        const factorLeaps = rsiLeaps ? null : getFactorBasedLeapsSuggestion(factorInputs);
        const leaps = rsiLeaps ?? factorLeaps;
        const isRsiBased = !!rsiLeaps;
        const isOversold = isRsiBased && (latestRsi ?? 50) <= 35;
        const factorsUsed = leaps && "factorsUsed" in leaps ? (leaps as { factorsUsed: string[] }).factorsUsed : null;
        return (
          <div
            className={clsx(
              "mt-4 rounded-xl border p-4",
              leaps
                ? isRsiBased
                  ? isOversold
                    ? "border-cyan-700/50 bg-cyan-950/20"
                    : "border-amber-700/50 bg-amber-950/20"
                  : "border-cyan-700/50 bg-cyan-950/20"
                : "border-slate-700/50 bg-slate-900/30",
            )}
          >
            <h2 className="text-sm font-semibold text-slate-200 mb-2">LEAPS suggestion</h2>
            <p className="text-xs text-slate-400 mb-3">
              Long-dated (12–18 mo) option ideas. Based on RSI when available; otherwise crossover, GBM, Monte Carlo, 52w. Use with the 9-condition LEAPS checklist.
            </p>
            {leaps ? (
              <div className="grid gap-2 text-sm">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-[10px] uppercase tracking-wider text-slate-500">Strategy</span>
                  <span
                    className={clsx(
                      "font-medium",
                      isRsiBased ? (isOversold ? "text-cyan-300" : "text-amber-300") : "text-cyan-300",
                    )}
                  >
                    {leaps.strategy}
                  </span>
                </div>
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-0.5">Strike</div>
                  <p className="text-slate-200 text-xs">{leaps.strikeSuggestion}</p>
                </div>
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-0.5">Expiry</div>
                  <p className="text-slate-200 text-xs">{leaps.expirySuggestion}</p>
                </div>
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-0.5">Rationale</div>
                  <p className="text-slate-400 text-xs leading-relaxed">{leaps.rationale}</p>
                </div>
                {factorsUsed && factorsUsed.length > 0 && (
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-0.5">Backtest key</div>
                    <p className="text-xs text-cyan-300/90 font-mono">{factorsUsed.join(" · ")}</p>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-sm text-slate-500 italic">
                No suggestions. RSI is neutral (35–65) and factor screen has no clear bullish/bearish tilt. Check back or use quantitative screening for next move.
              </p>
            )}
          </div>
        );
      })()}

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
        <div className="mb-2 flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-sm font-semibold text-slate-200">
            Price Chart · 50-DMA · 200-DMA · Bollinger Bands · Fibonacci
          </h2>
          <div className="flex flex-wrap items-center gap-3 text-[11px] text-slate-400">
            <span className="uppercase tracking-wider text-slate-500">EMAs</span>
            <label className="inline-flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                className="h-3 w-3 rounded border-slate-600 bg-slate-900 text-emerald-400"
                checked={showEma20}
                onChange={(e) => setShowEma20(e.target.checked)}
              />
              <span>20-EMA</span>
            </label>
            <label className="inline-flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                className="h-3 w-3 rounded border-slate-600 bg-slate-900 text-sky-400"
                checked={showEma50}
                onChange={(e) => setShowEma50(e.target.checked)}
              />
              <span>50-EMA</span>
            </label>
            <label className="inline-flex items-center gap-1 cursor-pointer">
              <input
                type="checkbox"
                className="h-3 w-3 rounded border-slate-600 bg-slate-900 text-orange-400"
                checked={showEma200}
                onChange={(e) => setShowEma200(e.target.checked)}
              />
              <span>200-EMA</span>
            </label>
          </div>
        </div>
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

      {/* ── Quantitative Stock Screening Guide ── */}
      <QuantitativeScreeningGuide researchData={data} volatilitySignals={volatilitySignals} />

      {/* Footer spacer */}
      <div className="h-12" />
    </div>
  );
}

// ── Sub-Components ───────────────────────────────────────────────────────

function SearchIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500 pointer-events-none">
      <path fillRule="evenodd" d="M9 3.5a5.5 5.5 0 100 11 5.5 5.5 0 000-11zM2 9a7 7 0 1112.452 4.391l3.328 3.329a.75.75 0 11-1.06 1.06l-3.329-3.328A7 7 0 012 9z" clipRule="evenodd" />
    </svg>
  );
}

function ArrowLeftIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
      <path fillRule="evenodd" d="M17 10a.75.75 0 01-.75.75H5.612l4.158 3.96a.75.75 0 11-1.04 1.08l-5.5-5.25a.75.75 0 010-1.08l5.5-5.25a.75.75 0 111.04 1.08L5.612 9.25H16.25A.75.75 0 0117 10z" clipRule="evenodd" />
    </svg>
  );
}

function RefreshIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-3.5 w-3.5">
      <path fillRule="evenodd" d="M15.312 11.424a5.5 5.5 0 01-9.201 2.466l-.312-.311h2.433a.75.75 0 000-1.5H4.28a.75.75 0 00-.75.75v3.955a.75.75 0 001.5 0v-2.134l.246.245A7 7 0 0016.732 11.5a.75.75 0 10-1.42-.076zm-10.624-2.85A5.5 5.5 0 0113.89 6.11l.311.31h-2.432a.75.75 0 000 1.5h3.955a.75.75 0 00.75-.75V3.214a.75.75 0 10-1.5 0v2.134l-.246-.245A7 7 0 003.268 8.5a.75.75 0 001.42.074z" clipRule="evenodd" />
    </svg>
  );
}

function CalendarIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-3.5 w-3.5">
      <path fillRule="evenodd" d="M5.75 2a.75.75 0 01.75.75V4h7V2.75a.75.75 0 011.5 0V4h.25A2.75 2.75 0 0118 6.75v8.5A2.75 2.75 0 0115.25 18H4.75A2.75 2.75 0 012 15.25v-8.5A2.75 2.75 0 014.75 4H5V2.75A.75.75 0 015.75 2zm-1 5.5c-.69 0-1.25.56-1.25 1.25v6.5c0 .69.56 1.25 1.25 1.25h10.5c.69 0 1.25-.56 1.25-1.25v-6.5c0-.69-.56-1.25-1.25-1.25H4.75z" clipRule="evenodd" />
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

// ── Quantitative Stock Screening (6 formulas, compact + green/red) ─────────

const SCREENING_ROWS: { id: string; formula: string; question: string; green: string; red: string }[] = [
  { id: "bayes", formula: "Bayes", question: "Signal real?", green: "Posterior > 60%", red: "< 40%" },
  { id: "gbm", formula: "GBM", question: "Price range?", green: "Upside 2× downside", red: "10th pct < −20%" },
  { id: "ito", formula: "Itô", question: "Options move?", green: "Gamma + catalyst", red: "Theta > 0.05/day" },
  { id: "bs", formula: "Black-Scholes", question: "Options mispriced?", green: "IV Rank < 20 or > 70", red: "IV 40–60" },
  { id: "markowitz", formula: "Markowitz", question: "Worth adding?", green: "Sharpe > 1, Corr < 0.3", red: "Corr > 0.8" },
  { id: "mc", formula: "Monte Carlo", question: "Real odds?", green: "P(+20%) > 2× P(−20%)", red: "VaR too high" },
];

type ScreeningResult = "green" | "red" | "review";

/** Build factor inputs from research data; uses precomputed volatility signals when provided (avoids duplicate GBM/MC). */
function getFactorInputs(
  data: ResearchData,
  volatilitySignals?: VolatilityPathSignals | null
): FactorInputs {
  const close = data.ohlcv?.close ?? [];
  const validClose = close.filter((c): c is number => c != null);
  const signals = volatilitySignals ?? getVolatilityAndPathSignals(validClose, data.currentPrice);

  const lo = data.fundamentals?.fiftyTwoWeekLow;
  const hi = data.fundamentals?.fiftyTwoWeekHigh;
  let fiftyTwoWeekPct: number | null = null;
  if (lo != null && hi != null && hi > lo) {
    fiftyTwoWeekPct = ((data.currentPrice - lo) / (hi - lo)) * 100;
  }

  return {
    currentPrice: data.currentPrice,
    crossoverSignal: data.crossover?.signal ?? "none",
    gbmBullish: signals?.gbmBullish ?? false,
    gbmBearish: signals?.gbmBearish ?? false,
    mcBullish: signals?.mcBullish ?? false,
    mcBearish: signals?.mcBearish ?? false,
    fiftyTwoWeekPct,
    beta: data.fundamentals?.beta ?? null,
  };
}

/** Uses precomputed volatilitySignals when provided (same as factor inputs — single source of truth). */
function getScreeningResults(
  data: ResearchData | null,
  volatilitySignals?: VolatilityPathSignals | null
): ScreeningResult[] {
  if (!data) return SCREENING_ROWS.map(() => "review");
  const close = data.ohlcv?.close ?? [];
  const validClose = close.filter((c): c is number => c != null);
  const signals = volatilitySignals ?? getVolatilityAndPathSignals(validClose, data.currentPrice);

  let latestRsi: number | null = data.latestRSI ?? null;
  if (latestRsi == null && data.indicators?.rsi) {
    const rsi = data.indicators.rsi;
    for (let i = rsi.length - 1; i >= 0; i--) if (rsi[i] != null) { latestRsi = rsi[i]!; break; }
  }
  const results: ScreeningResult[] = [];

  // Bayes: RSI
  if (latestRsi != null) {
    if (latestRsi <= 35 || latestRsi >= 65) results.push("green");
    else if (latestRsi >= 40 && latestRsi <= 60) results.push("red");
    else results.push("review");
  } else results.push("review");

  // GBM
  if (signals) {
    if (signals.gbmBullish) results.push("green");
    else if (signals.gbmBearish) results.push("red");
    else results.push("review");
  } else results.push("review");

  results.push("review");
  results.push("review");

  const beta = data.fundamentals?.beta;
  if (beta != null) {
    if (beta < 0.9) results.push("green");
    else if (beta > 1.4) results.push("red");
    else results.push("review");
  } else results.push("review");

  // Monte Carlo
  if (signals) {
    if (signals.mcBullish) results.push("green");
    else if (signals.mcBearish) results.push("red");
    else results.push("review");
  } else results.push("review");

  return results;
}

function getScreeningSummary(results: ScreeningResult[]) {
  const pass = results.filter((r) => r === "green").length;
  const fail = results.filter((r) => r === "red").length;
  const review = results.filter((r) => r === "review").length;
  return { pass, fail, review, total: results.length };
}

function getSuggestedNextMove(results: ScreeningResult[]): { text: string; tone: "green" | "red" | "neutral" } {
  const { pass, fail, review } = getScreeningSummary(results);
  if (pass >= 4 && fail <= 1) {
    return {
      text: "Screen leans bullish. Consider adding to watchlist; if RSI supports (oversold), align with LEAPS suggestion and 9-condition checklist before entry.",
      tone: "green",
    };
  }
  if (fail >= 4) {
    return {
      text: "Screen leans bearish or mixed. Avoid new long LEAPS here unless you have a strong catalyst view; consider hedging or waiting for better setup.",
      tone: "red",
    };
  }
  if (pass >= 2 && fail >= 2) {
    return {
      text: "Mixed signals. Review each formula row above; focus on Bayes (RSI) and GBM/Monte Carlo for conviction. Use LEAPS only if your thesis overrides red flags.",
      tone: "neutral",
    };
  }
  return {
    text: "Many results are Review (insufficient or neutral data). Gather more data (e.g. options/IV for Itô/BS) or wait for clearer RSI/volatility before committing.",
    tone: "neutral",
  };
}

function QuantitativeScreeningGuide({
  researchData,
  volatilitySignals,
}: {
  researchData?: ResearchData | null;
  volatilitySignals?: VolatilityPathSignals | null;
}) {
  const results = useMemo(
    () => getScreeningResults(researchData ?? null, volatilitySignals),
    [researchData, volatilitySignals]
  );
  const summary = useMemo(() => getScreeningSummary(results), [results]);
  const nextMove = useMemo(() => getSuggestedNextMove(results), [results]);

  return (
    <div className="mt-10 rounded-xl border border-cyan-800/50 bg-slate-900/40 p-4">
      <h2 className="text-base font-semibold text-slate-100">Quantitative Stock Screening</h2>
      <p className="mt-0.5 text-xs text-slate-400">Bayes · GBM · Itô · Black-Scholes · Markowitz · Monte Carlo — Pass = favorable, Fail = unfavorable, Review = need more data or neutral.</p>
      <div className="mt-3 overflow-x-auto rounded-lg border border-slate-700/60">
        <table className="w-full min-w-[480px] text-left text-xs">
          <thead>
            <tr className="border-b border-slate-700 bg-slate-800/50">
              <th className="px-2 py-1.5 font-semibold text-slate-300">Formula</th>
              <th className="px-2 py-1.5 font-semibold text-slate-300">Question</th>
              <th className="px-2 py-1.5 font-semibold text-emerald-400/90">Green</th>
              <th className="px-2 py-1.5 font-semibold text-rose-400/90">Red</th>
              <th className="px-2 py-1.5 font-semibold text-slate-400 w-20 text-center">Result</th>
            </tr>
          </thead>
          <tbody>
            {SCREENING_ROWS.map((row, i) => {
              const res = results[i] ?? "review";
              return (
                <tr key={row.id} className="border-b border-slate-800/80 hover:bg-slate-800/30">
                  <td className="px-2 py-1.5 font-medium text-cyan-200/90">{row.formula}</td>
                  <td className="px-2 py-1.5 text-slate-400">{row.question}</td>
                  <td className="px-2 py-1.5 text-emerald-300/90">{row.green}</td>
                  <td className="px-2 py-1.5 text-rose-300/90">{row.red}</td>
                  <td className="px-2 py-1.5 text-center">
                    {res === "green" && <span className="inline-flex items-center gap-1 rounded bg-emerald-500/20 px-1.5 py-0.5 text-[10px] font-medium text-emerald-300"><span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />Pass</span>}
                    {res === "red" && <span className="inline-flex items-center gap-1 rounded bg-rose-500/20 px-1.5 py-0.5 text-[10px] font-medium text-rose-300"><span className="h-1.5 w-1.5 rounded-full bg-rose-400" />Fail</span>}
                    {res === "review" && <span className="inline-flex items-center gap-1 rounded bg-slate-600/30 px-1.5 py-0.5 text-[10px] text-slate-400"><span className="h-1.5 w-1.5 rounded-full bg-slate-500" />Review</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-4 rounded-lg border border-slate-700/60 bg-slate-800/30 p-3">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1.5">Results summary</div>
        <p className="text-xs text-slate-300">
          Pass: <span className="text-emerald-400 font-medium">{summary.pass}</span>
          {" · "}
          Fail: <span className="text-rose-400 font-medium">{summary.fail}</span>
          {" · "}
          Review: <span className="text-slate-400 font-medium">{summary.review}</span>
          {" "}(of {summary.total}). Pass = formula favors the trade; Fail = formula argues against; Review = inconclusive or no data.
        </p>
        <div className="mt-3 text-[10px] uppercase tracking-wider text-slate-500 mb-1.5">Suggested next move</div>
        <p
          className={clsx(
            "text-xs leading-relaxed",
            nextMove.tone === "green" && "text-emerald-300/95",
            nextMove.tone === "red" && "text-rose-300/95",
            nextMove.tone === "neutral" && "text-slate-300",
          )}
        >
          {nextMove.text}
        </p>
      </div>
    </div>
  );
}
