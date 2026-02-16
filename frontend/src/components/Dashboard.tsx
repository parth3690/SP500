"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";

import { fetchMovers, fetchCrossovers, fetchOversold, apiBaseUrl } from "@/lib/api";
import { addDays, toLocalISODate } from "@/lib/date";
import type { MoversResponse, MoverRow, CrossoversResponse, OversoldResponse } from "@/lib/types";
import MoversTable from "@/components/MoversTable";
import CrossoverTable from "@/components/CrossoverTable";
import OversoldTable from "@/components/OversoldTable";
import SectorSummary from "@/components/SectorSummary";
import MoversBarChart from "@/components/MoversBarChart";
import Heatmap from "@/components/Heatmap";

type Preset = "1w" | "1m" | "3m" | "custom";

const presetToDays: Record<Exclude<Preset, "custom">, number> = {
  "1w": 7,
  "1m": 30,
  "3m": 90
};

function formatPct(v: number): string {
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}%`;
}

export default function Dashboard() {
  const today = useMemo(() => toLocalISODate(new Date()), []);

  const [preset, setPreset] = useState<Preset>("1m");
  const [end, setEnd] = useState<string>(today);
  const [start, setStart] = useState<string>(addDays(today, -presetToDays["1m"]));
  const [limit, setLimit] = useState<number>(50);

  const [search, setSearch] = useState<string>("");
  const [autoRefresh, setAutoRefresh] = useState<boolean>(false);
  const [refreshEverySec, setRefreshEverySec] = useState<number>(300);

  const [data, setData] = useState<MoversResponse | null>(null);
  const [crossoverData, setCrossoverData] = useState<CrossoversResponse | null>(null);
  const [oversoldData, setOversoldData] = useState<OversoldResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [crossoverLoading, setCrossoverLoading] = useState<boolean>(false);
  const [oversoldLoading, setOversoldLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const intervalRef = useRef<number | null>(null);

  const includeAll = true;

  useEffect(() => {
    if (preset === "custom") return;
    const days = presetToDays[preset];
    setStart(addDays(end, -days));
  }, [preset, end]);

  const runFetch = async (opts?: { refresh?: boolean }) => {
    setLoading(true);
    setError(null);
    try {
      const payload = await fetchMovers({
        start,
        end,
        limit,
        includeAll,
        refresh: opts?.refresh
      });
      setData(payload);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const runCrossoverFetch = async (opts?: { refresh?: boolean }) => {
    setCrossoverLoading(true);
    try {
      const payload = await fetchCrossovers({ threshold: 2.0, refresh: opts?.refresh });
      setCrossoverData(payload);
    } catch (e) {
      console.error("Crossover fetch failed:", e);
    } finally {
      setCrossoverLoading(false);
    }
  };

  const runOversoldFetch = async (opts?: { refresh?: boolean }) => {
    setOversoldLoading(true);
    try {
      const payload = await fetchOversold({ threshold: 30, refresh: opts?.refresh });
      setOversoldData(payload);
    } catch (e) {
      console.error("RSI oversold fetch failed:", e);
    } finally {
      setOversoldLoading(false);
    }
  };

  // Fire ALL fetches in parallel on mount for fastest initial load
  useEffect(() => {
    void runFetch();
    void runCrossoverFetch();
    void runOversoldFetch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [start, end, limit]);

  useEffect(() => {
    if (!autoRefresh) {
      if (intervalRef.current) window.clearInterval(intervalRef.current);
      intervalRef.current = null;
      return;
    }

    if (intervalRef.current) window.clearInterval(intervalRef.current);
    intervalRef.current = window.setInterval(() => void runFetch(), refreshEverySec * 1000);
    return () => {
      if (intervalRef.current) window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRefresh, refreshEverySec, start, end, limit]);

  const allRows = data?.all ?? [];

  const matches = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return [] as MoverRow[];
    return allRows
      .filter(
        (r) => r.ticker.toLowerCase().includes(q) || r.companyName.toLowerCase().includes(q)
      )
      .slice(0, 25);
  }, [allRows, search]);

  const downloadCsvUrl = useMemo(() => {
    const base = apiBaseUrl();
    const url = new URL(`${base}/api/movers.csv`);
    url.searchParams.set("start", start);
    url.searchParams.set("end", end);
    return url.toString();
  }, [start, end]);

  return (
    <div className="mx-auto max-w-[1600px] px-4 py-8">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            S&amp;P 500 Monthly Movers Analyzer
          </h1>
          <p className="text-sm text-slate-400">
            Range: <span className="text-slate-200">{start}</span> →{" "}
            <span className="text-slate-200">{end}</span>
            {data?.asOf ? (
              <>
                {" "}
                · As of <span className="text-slate-200">{new Date(data.asOf).toLocaleString()}</span>
              </>
            ) : null}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <a
            className="rounded-md bg-slate-800 px-3 py-2 text-sm font-medium text-slate-100 hover:bg-slate-700"
            href={downloadCsvUrl}
          >
            Download CSV
          </a>
          <button
            className={clsx(
              "inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-semibold transition-colors",
              loading || crossoverLoading
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/30 cursor-wait"
                : "bg-emerald-600 text-white hover:bg-emerald-500"
            )}
            disabled={loading && crossoverLoading && oversoldLoading}
            onClick={() => {
              void runFetch({ refresh: true });
              void runCrossoverFetch({ refresh: true });
              void runOversoldFetch({ refresh: true });
            }}
          >
            {loading || crossoverLoading || oversoldLoading ? (
              <>
                <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-amber-400/40 border-t-amber-400" />
                Refreshing...
              </>
            ) : (
              <>
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
                  <path fillRule="evenodd" d="M15.312 11.424a5.5 5.5 0 01-9.201 2.466l-.312-.311h2.433a.75.75 0 000-1.5H4.28a.75.75 0 00-.75.75v3.955a.75.75 0 001.5 0v-2.134l.246.245A7 7 0 0016.732 11.5a.75.75 0 10-1.42-.076zm-10.624-2.85A5.5 5.5 0 0113.89 6.11l.311.31h-2.432a.75.75 0 000 1.5h3.955a.75.75 0 00.75-.75V3.214a.75.75 0 10-1.5 0v2.134l-.246-.245A7 7 0 003.268 8.5a.75.75 0 001.42.074z" clipRule="evenodd" />
                </svg>
                Refresh All Data
              </>
            )}
          </button>
        </div>
      </header>

      <section className="mt-6 grid grid-cols-1 gap-3 rounded-xl border border-slate-800 bg-slate-900/30 p-4 md:grid-cols-12">
        <div className="md:col-span-8">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs uppercase tracking-wide text-slate-400">Date range</span>
            {(["1w", "1m", "3m", "custom"] as Preset[]).map((p) => (
              <button
                key={p}
                className={clsx(
                  "rounded-md px-3 py-2 text-sm font-medium",
                  preset === p
                    ? "bg-slate-100 text-slate-900"
                    : "bg-slate-800 text-slate-100 hover:bg-slate-700"
                )}
                onClick={() => setPreset(p)}
              >
                {p === "1w" ? "1W" : p === "1m" ? "1M" : p === "3m" ? "3M" : "Custom"}
              </button>
            ))}

            <div className="ml-2 flex items-center gap-2">
              <label className="text-xs text-slate-400">Start</label>
              <input
                type="date"
                className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-sm"
                value={start}
                onChange={(e) => {
                  setPreset("custom");
                  setStart(e.target.value);
                }}
              />
              <label className="text-xs text-slate-400">End</label>
              <input
                type="date"
                className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-sm"
                value={end}
                onChange={(e) => setEnd(e.target.value)}
              />
            </div>
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-2">
              <label className="text-xs uppercase tracking-wide text-slate-400">Top N</label>
              <select
                className="rounded-md border border-slate-700 bg-slate-950 px-2 py-2 text-sm"
                value={limit}
                onChange={(e) => setLimit(Number(e.target.value))}
              >
                {[25, 50, 100].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex flex-1 items-center gap-2">
              <label className="text-xs uppercase tracking-wide text-slate-400">Search</label>
              <input
                placeholder="Ticker or company name…"
                className="w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
          </div>
        </div>

        <div className="md:col-span-4">
          <div className="flex h-full flex-col justify-between gap-3">
            <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-3">
              <div className="flex items-center justify-between">
                <span className="text-xs uppercase tracking-wide text-slate-400">Auto-refresh</span>
                <button
                  className={clsx(
                    "rounded-md px-3 py-2 text-sm font-medium",
                    autoRefresh ? "bg-emerald-400 text-slate-900" : "bg-slate-800 hover:bg-slate-700"
                  )}
                  onClick={() => setAutoRefresh((v) => !v)}
                >
                  {autoRefresh ? "On" : "Off"}
                </button>
              </div>
              <div className="mt-2 flex items-center gap-2">
                <label className="text-xs text-slate-400">Every</label>
                <select
                  className="rounded-md border border-slate-700 bg-slate-950 px-2 py-2 text-sm"
                  value={refreshEverySec}
                  onChange={(e) => setRefreshEverySec(Number(e.target.value))}
                  disabled={!autoRefresh}
                >
                  <option value={60}>1 min</option>
                  <option value={300}>5 min</option>
                  <option value={900}>15 min</option>
                </select>
              </div>
            </div>

            <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-3">
              <div className="text-xs uppercase tracking-wide text-slate-400">Coverage</div>
              <div className="mt-1 text-sm text-slate-200">
                {data ? (
                  <>
                    Computed for{" "}
                    <span className="font-semibold">{data.meta.computed}</span> /{" "}
                    <span className="font-semibold">{data.meta.total}</span> tickers
                    {data.meta.missingCount ? (
                      <span className="text-slate-400">
                        {" "}
                        ({data.meta.missingCount} missing)
                      </span>
                    ) : null}
                  </>
                ) : (
                  <span className="text-slate-400">—</span>
                )}
              </div>
            </div>
          </div>
        </div>
      </section>

      {error ? (
        <div className="mt-4 rounded-lg border border-rose-900/60 bg-rose-950/30 p-3 text-sm text-rose-200">
          {error}
        </div>
      ) : null}

      {loading && !data ? (
        <div className="mt-8 flex items-center gap-3 text-sm text-slate-400">
          <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-slate-700 border-t-amber-400" />
          Loading market data...
        </div>
      ) : null}

      {/* ─── Golden Cross / Death Cross Section ─── */}
      {crossoverData ? (
        <section className="mt-8">
          <div className="mb-4">
            <h2 className="text-lg font-semibold tracking-tight text-slate-100">
              Moving Average Crossover Signals
            </h2>
            <p className="text-sm text-slate-400">
              Stocks where the 50-DMA and 200-DMA are within{" "}
              <span className="font-medium text-slate-200">{crossoverData.thresholdPct}%</span> of
              each other — potential crossover incoming.
            </p>
          </div>
          <div className="space-y-4">
            <CrossoverTable
              title="Near Golden Cross"
              subtitle="50-DMA approaching 200-DMA from below — bullish signal."
              rows={crossoverData.nearGoldenCross}
              variant="golden"
            />
            <CrossoverTable
              title="Near Death Cross"
              subtitle="50-DMA approaching 200-DMA from above — bearish signal."
              rows={crossoverData.nearDeathCross}
              variant="death"
            />
          </div>
        </section>
      ) : crossoverLoading ? (
        <div className="mt-8 flex items-center gap-3 text-sm text-slate-400">
          <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-slate-700 border-t-amber-400" />
          Computing crossover signals...
        </div>
      ) : null}

      {/* ─── Weekly RSI Oversold Section ─── */}
      {oversoldData ? (
        <section className="mt-8">
          <div className="mb-4">
            <h2 className="text-lg font-semibold tracking-tight text-slate-100">
              Weekly RSI Oversold Stocks
            </h2>
            <p className="text-sm text-slate-400">
              S&amp;P 500 stocks where the 14-period weekly RSI is below{" "}
              <span className="font-medium text-emerald-300">{oversoldData.rsiThreshold}</span>{" "}
              — potential buying opportunities as selling pressure may be exhausted.
            </p>
          </div>
          <OversoldTable
            title="Weekly RSI Below 30"
            subtitle="Stocks with extreme weekly oversold conditions — sorted by most oversold first."
            rows={oversoldData.stocks}
          />
        </section>
      ) : oversoldLoading ? (
        <div className="mt-8 flex items-center gap-3 text-sm text-slate-400">
          <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-slate-700 border-t-emerald-400" />
          Scanning weekly RSI...
        </div>
      ) : null}

      {data ? (
        <>
          {search.trim() ? (
            <section className="mt-8">
              <h2 className="text-sm font-semibold text-slate-200">
                Matches <span className="text-slate-500">({matches.length})</span>
              </h2>
              <div className="mt-3">
                <MoversTable
                  title=""
                  rows={matches}
                  variant="neutral"
                  defaultSort={{ key: "pctChange", dir: "desc" }}
                  footerNote="Matches are filtered from all S&P 500 constituents."
                />
              </div>
            </section>
          ) : null}

          <section className="mt-8 grid grid-cols-1 gap-4 lg:grid-cols-2">
            <MoversTable
              title="📈 Top Gainers"
              rows={data.gainers}
              variant="gainers"
              defaultSort={{ key: "pctChange", dir: "desc" }}
              subtitle={`Sorted by % change (${formatPct(data.gainers[0]?.pctChange ?? 0)} top).`}
            />
            <MoversTable
              title="📉 Top Losers"
              rows={data.losers}
              variant="losers"
              defaultSort={{ key: "pctChange", dir: "asc" }}
              subtitle={`Sorted by % change (${formatPct(data.losers[0]?.pctChange ?? 0)} bottom).`}
            />
          </section>

          <section className="mt-8 grid grid-cols-1 gap-4 lg:grid-cols-3">
            <div className="lg:col-span-2">
              <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-4">
                <div className="flex items-end justify-between gap-3">
                  <div>
                    <h2 className="text-sm font-semibold text-slate-200">Top movers chart</h2>
                    <p className="text-xs text-slate-400">
                      Visualizes top {Math.min(10, data.gainers.length)} gainers and top{" "}
                      {Math.min(10, data.losers.length)} losers.
                    </p>
                  </div>
                </div>
                <div className="mt-3 h-80">
                  <MoversBarChart gainers={data.gainers.slice(0, 10)} losers={data.losers.slice(0, 10)} />
                </div>
              </div>
            </div>
            <div className="lg:col-span-1">
              <SectorSummary rows={data.sectorSummary} allStocks={allRows} />
            </div>
          </section>

          <section className="mt-8">
            <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-4">
              <div className="flex items-end justify-between gap-3">
                <div>
                  <h2 className="text-sm font-semibold text-slate-200">S&amp;P 500 heatmap</h2>
                  <p className="text-xs text-slate-400">
                    Each square is a constituent colored by % change for the selected range.
                  </p>
                </div>
              </div>
              <div className="mt-4">
                <Heatmap rows={allRows} />
              </div>
            </div>
          </section>
        </>
      ) : null}
    </div>
  );
}

