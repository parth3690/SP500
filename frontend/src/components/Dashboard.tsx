"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import clsx from "clsx";

import { fetchMovers, fetchCrossovers, fetchOversold, fetchOverbought, fetchDailyOversold, fetchDailyOverbought, apiBaseUrl } from "@/lib/api";
import { addDays, toLocalISODate, startOfYear } from "@/lib/date";
import { formatPct } from "@/lib/format";
import { getOptionSuggestionShort } from "@/lib/optionSuggestions";
import type { MoversResponse, MoverRow, CrossoversResponse, OversoldResponse, OverboughtResponse } from "@/lib/types";

const MoversTable = dynamic(() => import("@/components/MoversTable"), { ssr: false });
const CrossoverTable = dynamic(() => import("@/components/CrossoverTable"), { ssr: false });
const OversoldTable = dynamic(() => import("@/components/OversoldTable"), { ssr: false });
const OverboughtTable = dynamic(() => import("@/components/OverboughtTable"), { ssr: false });
const SectorSummary = dynamic(() => import("@/components/SectorSummary"), { ssr: false });
const MoversBarChart = dynamic(() => import("@/components/MoversBarChart"), { ssr: false });
const Heatmap = dynamic(() => import("@/components/Heatmap"), { ssr: false });

type Preset = "ytd" | "1w" | "1m" | "3m" | "custom";

const presetToDays: Record<Exclude<Preset, "custom" | "ytd">, number> = {
  "1w": 7,
  "1m": 30,
  "3m": 90
};

function getErrorMessage(e: unknown): string {
  return e instanceof Error ? e.message : "Unknown error";
}

export default function Dashboard() {
  const today = useMemo(() => toLocalISODate(new Date()), []);

  const [preset, setPreset] = useState<Preset>("ytd");
  const [end, setEnd] = useState<string>(today);
  const [start, setStart] = useState<string>(startOfYear(today));
  const [limit, setLimit] = useState<number>(50);

  const [search, setSearch] = useState<string>("");
  const [autoRefresh, setAutoRefresh] = useState<boolean>(false);
  const [refreshEverySec, setRefreshEverySec] = useState<number>(300);

  const [data, setData] = useState<MoversResponse | null>(null);
  const [crossoverData, setCrossoverData] = useState<CrossoversResponse | null>(null);
  const [oversoldData, setOversoldData] = useState<OversoldResponse | null>(null);
  const [overboughtData, setOverboughtData] = useState<OverboughtResponse | null>(null);
  const [dailyOversoldData, setDailyOversoldData] = useState<OversoldResponse | null>(null);
  const [dailyOverboughtData, setDailyOverboughtData] = useState<OverboughtResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [crossoverLoading, setCrossoverLoading] = useState<boolean>(false);
  const [oversoldLoading, setOversoldLoading] = useState<boolean>(false);
  const [overboughtLoading, setOverboughtLoading] = useState<boolean>(false);
  const [dailyOversoldLoading, setDailyOversoldLoading] = useState<boolean>(false);
  const [dailyOverboughtLoading, setDailyOverboughtLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [leapsWeeklyExpanded, setLeapsWeeklyExpanded] = useState(false);
  const [leapsDailyExpanded, setLeapsDailyExpanded] = useState(false);
  const [leapsWeeklyOverboughtExpanded, setLeapsWeeklyOverboughtExpanded] = useState(false);
  const [leapsDailyOverboughtExpanded, setLeapsDailyOverboughtExpanded] = useState(false);

  const intervalRef = useRef<number | null>(null);

  const includeAll = true;

  useEffect(() => {
    if (preset === "custom") return;
    if (preset === "ytd") {
      setStart(startOfYear(end));
      return;
    }
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
        refresh: opts?.refresh === true,
      });
      setData(payload);
    } catch (e) {
      setData(null);
      setError(getErrorMessage(e));
    } finally {
      setLoading(false);
    }
  };

  const runCrossoverFetch = async (opts?: { refresh?: boolean }) => {
    setCrossoverLoading(true);
    try {
      const payload = await fetchCrossovers({ threshold: 2.0, refresh: opts?.refresh === true });
      setCrossoverData(payload);
    } catch (e) {
      setCrossoverData(null);
      console.error("Crossover fetch failed:", getErrorMessage(e));
    } finally {
      setCrossoverLoading(false);
    }
  };

  const runOversoldFetch = async (opts?: { refresh?: boolean }) => {
    setOversoldLoading(true);
    try {
      const payload = await fetchOversold({ threshold: 30, refresh: opts?.refresh === true });
      setOversoldData(payload);
    } catch (e) {
      setOversoldData(null);
      console.error("RSI oversold fetch failed:", getErrorMessage(e));
    } finally {
      setOversoldLoading(false);
    }
  };

  const runOverboughtFetch = async (opts?: { refresh?: boolean }) => {
    setOverboughtLoading(true);
    try {
      const payload = await fetchOverbought({ threshold: 70, refresh: opts?.refresh === true });
      setOverboughtData(payload);
    } catch (e) {
      setOverboughtData(null);
      console.error("RSI overbought fetch failed:", getErrorMessage(e));
    } finally {
      setOverboughtLoading(false);
    }
  };

  const runDailyOversoldFetch = async (opts?: { refresh?: boolean }) => {
    setDailyOversoldLoading(true);
    try {
      const payload = await fetchDailyOversold({ threshold: 30, refresh: opts?.refresh === true });
      setDailyOversoldData(payload);
    } catch (e) {
      setDailyOversoldData(null);
      console.error("Daily RSI oversold fetch failed:", getErrorMessage(e));
    } finally {
      setDailyOversoldLoading(false);
    }
  };

  const runDailyOverboughtFetch = async (opts?: { refresh?: boolean }) => {
    setDailyOverboughtLoading(true);
    try {
      const payload = await fetchDailyOverbought({ threshold: 70, refresh: opts?.refresh === true });
      setDailyOverboughtData(payload);
    } catch (e) {
      setDailyOverboughtData(null);
      console.error("Daily RSI overbought fetch failed:", getErrorMessage(e));
    } finally {
      setDailyOverboughtLoading(false);
    }
  };

  // Fire ALL fetches in parallel on mount for fastest initial load
  useEffect(() => {
    void runFetch();
    void runCrossoverFetch();
    void runOversoldFetch();
    void runOverboughtFetch();
    void runDailyOversoldFetch();
    void runDailyOverboughtFetch();
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

  const hasWeeklyLeapsCandidates = useMemo(
    () => (oversoldData?.stocks?.length ?? 0) > 0,
    [oversoldData],
  );

  const hasDailyLeapsCandidates = useMemo(
    () => (dailyOversoldData?.stocks?.length ?? 0) > 0,
    [dailyOversoldData],
  );

  const hasWeeklyOverbought = useMemo(
    () => (overboughtData?.stocks?.length ?? 0) > 0,
    [overboughtData],
  );

  const hasDailyOverbought = useMemo(
    () => (dailyOverboughtData?.stocks?.length ?? 0) > 0,
    [dailyOverboughtData],
  );

  const hasAnyRadarCandidates = hasWeeklyLeapsCandidates || hasDailyLeapsCandidates || hasWeeklyOverbought || hasDailyOverbought;

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
            disabled={loading && crossoverLoading && oversoldLoading && overboughtLoading && dailyOversoldLoading && dailyOverboughtLoading}
            onClick={() => {
              void runFetch({ refresh: true });
              void runCrossoverFetch({ refresh: true });
              void runOversoldFetch({ refresh: true });
              void runOverboughtFetch({ refresh: true });
              void runDailyOversoldFetch({ refresh: true });
              void runDailyOverboughtFetch({ refresh: true });
            }}
          >
            {loading || crossoverLoading || oversoldLoading || overboughtLoading || dailyOversoldLoading || dailyOverboughtLoading ? (
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
            {(["ytd", "1w", "1m", "3m", "custom"] as Preset[]).map((p) => (
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
                {p === "ytd" ? "YTD" : p === "1w" ? "1W" : p === "1m" ? "1M" : p === "3m" ? "3M" : "Custom"}
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

            <div className="space-y-3">
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

              <div className="rounded-lg border border-emerald-700/60 bg-emerald-950/20 p-3">
                <div className="flex items-center justify-between gap-2">
                  <div>
                    <div className="text-xs uppercase tracking-wide text-emerald-300">
                      LEAPS setup radar
                    </div>
                    <p className="mt-1 text-xs text-emerald-200/80">
                      Oversold and overbought scans — expand each list for tickers worth a deeper LEAPS checklist review.
                    </p>
                  </div>
                  <div
                    className={clsx(
                      "inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs font-medium",
                      hasAnyRadarCandidates
                        ? "bg-emerald-500/20 text-emerald-200 ring-1 ring-emerald-500/60"
                        : "bg-slate-800 text-slate-300 ring-1 ring-slate-700",
                    )}
                  >
                    <span
                      className={clsx(
                        "h-1.5 w-1.5 rounded-full",
                        hasAnyRadarCandidates
                          ? "bg-emerald-400 animate-pulse"
                          : "bg-slate-500",
                      )}
                    />
                    {hasAnyRadarCandidates
                      ? "Candidates detected"
                      : "No candidates"}
                  </div>
                </div>

                <div className="mt-2 space-y-1">
                  <button
                    type="button"
                    onClick={() => setLeapsWeeklyExpanded((v) => !v)}
                    className={clsx(
                      "flex w-full items-center justify-between gap-2 rounded-md border px-2 py-1.5 text-left text-[11px] transition-colors",
                      hasWeeklyLeapsCandidates
                        ? "border-emerald-400/70 bg-emerald-500/10 text-emerald-100 hover:bg-emerald-500/20"
                        : "border-slate-700 bg-slate-900/40 text-slate-300 hover:bg-slate-800/60",
                    )}
                  >
                    <span className="inline-flex items-center gap-1.5">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                      Weekly RSI oversold
                      <span className="font-semibold">
                        {(oversoldData?.stocks?.length ?? 0).toLocaleString()}
                      </span>
                    </span>
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                      className={clsx("h-3.5 w-3.5 flex-shrink-0 transition-transform", leapsWeeklyExpanded && "rotate-180")}
                    >
                      <path fillRule="evenodd" d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z" clipRule="evenodd" />
                    </svg>
                  </button>
                  {leapsWeeklyExpanded && (
                    <div className="max-h-40 overflow-y-auto rounded-md border border-slate-700/60 bg-slate-950/60 py-1">
                      {(oversoldData?.stocks?.length ?? 0) > 0 ? (
                        <ul className="space-y-0.5 px-2 text-[11px]">
                          {oversoldData!.stocks.map((s) => {
                            const suggestion = getOptionSuggestionShort("weekly_oversold", s.weeklyRSI, s.currentPrice);
                            return (
                              <li key={s.ticker} className="rounded px-1.5 py-1 hover:bg-slate-800/40">
                                <Link
                                  href={`/research/${encodeURIComponent(s.ticker)}`}
                                  className="flex items-center justify-between gap-2 text-emerald-200 hover:text-emerald-100"
                                >
                                  <span className="font-medium">{s.ticker}</span>
                                  {s.weeklyRSI != null && (
                                    <span className="text-slate-400">RSI {s.weeklyRSI.toFixed(1)}</span>
                                  )}
                                </Link>
                                {suggestion && (
                                  <p className="mt-0.5 pl-0.5 text-[10px] text-slate-500 leading-tight">
                                    {suggestion}
                                  </p>
                                )}
                              </li>
                            );
                          })}
                        </ul>
                      ) : (
                        <p className="px-2 py-2 text-[11px] text-slate-500">No weekly oversold candidates.</p>
                      )}
                    </div>
                  )}

                  <button
                    type="button"
                    onClick={() => setLeapsDailyExpanded((v) => !v)}
                    className={clsx(
                      "flex w-full items-center justify-between gap-2 rounded-md border px-2 py-1.5 text-left text-[11px] transition-colors",
                      hasDailyLeapsCandidates
                        ? "border-emerald-400/70 bg-emerald-500/10 text-emerald-100 hover:bg-emerald-500/20"
                        : "border-slate-700 bg-slate-900/40 text-slate-300 hover:bg-slate-800/60",
                    )}
                  >
                    <span className="inline-flex items-center gap-1.5">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                      Daily RSI oversold
                      <span className="font-semibold">
                        {(dailyOversoldData?.stocks?.length ?? 0).toLocaleString()}
                      </span>
                    </span>
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                      className={clsx("h-3.5 w-3.5 flex-shrink-0 transition-transform", leapsDailyExpanded && "rotate-180")}
                    >
                      <path fillRule="evenodd" d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z" clipRule="evenodd" />
                    </svg>
                  </button>
                  {leapsDailyExpanded && (
                    <div className="max-h-40 overflow-y-auto rounded-md border border-slate-700/60 bg-slate-950/60 py-1">
                      {(dailyOversoldData?.stocks?.length ?? 0) > 0 ? (
                        <ul className="space-y-0.5 px-2 text-[11px]">
                          {dailyOversoldData!.stocks.map((s) => {
                            const suggestion = getOptionSuggestionShort("daily_oversold", s.dailyRSI, s.currentPrice);
                            return (
                              <li key={s.ticker} className="rounded px-1.5 py-1 hover:bg-slate-800/40">
                                <Link
                                  href={`/research/${encodeURIComponent(s.ticker)}`}
                                  className="flex items-center justify-between gap-2 text-emerald-200 hover:text-emerald-100"
                                >
                                  <span className="font-medium">{s.ticker}</span>
                                  {s.dailyRSI != null && (
                                    <span className="text-slate-400">RSI {s.dailyRSI.toFixed(1)}</span>
                                  )}
                                </Link>
                                {suggestion && (
                                  <p className="mt-0.5 pl-0.5 text-[10px] text-slate-500 leading-tight">
                                    {suggestion}
                                  </p>
                                )}
                              </li>
                            );
                          })}
                        </ul>
                      ) : (
                        <p className="px-2 py-2 text-[11px] text-slate-500">No daily oversold candidates.</p>
                      )}
                    </div>
                  )}

                  <button
                    type="button"
                    onClick={() => setLeapsWeeklyOverboughtExpanded((v) => !v)}
                    className={clsx(
                      "flex w-full items-center justify-between gap-2 rounded-md border px-2 py-1.5 text-left text-[11px] transition-colors",
                      hasWeeklyOverbought
                        ? "border-rose-400/70 bg-rose-500/10 text-rose-100 hover:bg-rose-500/20"
                        : "border-slate-700 bg-slate-900/40 text-slate-300 hover:bg-slate-800/60",
                    )}
                  >
                    <span className="inline-flex items-center gap-1.5">
                      <span className="h-1.5 w-1.5 rounded-full bg-rose-400" />
                      Weekly RSI overbought
                      <span className="font-semibold">
                        {(overboughtData?.stocks?.length ?? 0).toLocaleString()}
                      </span>
                    </span>
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                      className={clsx("h-3.5 w-3.5 flex-shrink-0 transition-transform", leapsWeeklyOverboughtExpanded && "rotate-180")}
                    >
                      <path fillRule="evenodd" d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z" clipRule="evenodd" />
                    </svg>
                  </button>
                  {leapsWeeklyOverboughtExpanded && (
                    <div className="max-h-40 overflow-y-auto rounded-md border border-slate-700/60 bg-slate-950/60 py-1">
                      {(overboughtData?.stocks?.length ?? 0) > 0 ? (
                        <ul className="space-y-0.5 px-2 text-[11px]">
                          {overboughtData!.stocks.map((s) => {
                            const suggestion = getOptionSuggestionShort("weekly_overbought", s.weeklyRSI, s.currentPrice);
                            return (
                              <li key={s.ticker} className="rounded px-1.5 py-1 hover:bg-slate-800/40">
                                <Link
                                  href={`/research/${encodeURIComponent(s.ticker)}`}
                                  className="flex items-center justify-between gap-2 text-rose-200 hover:text-rose-100"
                                >
                                  <span className="font-medium">{s.ticker}</span>
                                  {s.weeklyRSI != null && (
                                    <span className="text-slate-400">RSI {s.weeklyRSI.toFixed(1)}</span>
                                  )}
                                </Link>
                                {suggestion && (
                                  <p className="mt-0.5 pl-0.5 text-[10px] text-slate-500 leading-tight">
                                    {suggestion}
                                  </p>
                                )}
                              </li>
                            );
                          })}
                        </ul>
                      ) : (
                        <p className="px-2 py-2 text-[11px] text-slate-500">No weekly overbought candidates.</p>
                      )}
                    </div>
                  )}

                  <button
                    type="button"
                    onClick={() => setLeapsDailyOverboughtExpanded((v) => !v)}
                    className={clsx(
                      "flex w-full items-center justify-between gap-2 rounded-md border px-2 py-1.5 text-left text-[11px] transition-colors",
                      hasDailyOverbought
                        ? "border-rose-400/70 bg-rose-500/10 text-rose-100 hover:bg-rose-500/20"
                        : "border-slate-700 bg-slate-900/40 text-slate-300 hover:bg-slate-800/60",
                    )}
                  >
                    <span className="inline-flex items-center gap-1.5">
                      <span className="h-1.5 w-1.5 rounded-full bg-rose-400" />
                      Daily RSI overbought
                      <span className="font-semibold">
                        {(dailyOverboughtData?.stocks?.length ?? 0).toLocaleString()}
                      </span>
                    </span>
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                      className={clsx("h-3.5 w-3.5 flex-shrink-0 transition-transform", leapsDailyOverboughtExpanded && "rotate-180")}
                    >
                      <path fillRule="evenodd" d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z" clipRule="evenodd" />
                    </svg>
                  </button>
                  {leapsDailyOverboughtExpanded && (
                    <div className="max-h-40 overflow-y-auto rounded-md border border-slate-700/60 bg-slate-950/60 py-1">
                      {(dailyOverboughtData?.stocks?.length ?? 0) > 0 ? (
                        <ul className="space-y-0.5 px-2 text-[11px]">
                          {dailyOverboughtData!.stocks.map((s) => {
                            const suggestion = getOptionSuggestionShort("daily_overbought", s.dailyRSI, s.currentPrice);
                            return (
                              <li key={s.ticker} className="rounded px-1.5 py-1 hover:bg-slate-800/40">
                                <Link
                                  href={`/research/${encodeURIComponent(s.ticker)}`}
                                  className="flex items-center justify-between gap-2 text-rose-200 hover:text-rose-100"
                                >
                                  <span className="font-medium">{s.ticker}</span>
                                  {s.dailyRSI != null && (
                                    <span className="text-slate-400">RSI {s.dailyRSI.toFixed(1)}</span>
                                  )}
                                </Link>
                                {suggestion && (
                                  <p className="mt-0.5 pl-0.5 text-[10px] text-slate-500 leading-tight">
                                    {suggestion}
                                  </p>
                                )}
                              </li>
                            );
                          })}
                        </ul>
                      ) : (
                        <p className="px-2 py-2 text-[11px] text-slate-500">No daily overbought candidates.</p>
                      )}
                    </div>
                  )}
                </div>
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

      {/* ─── Weekly RSI Overbought Section ─── */}
      <section className="mt-8">
        <div className="mb-4">
          <h2 className="text-lg font-semibold tracking-tight text-slate-100">
            Weekly RSI Overbought Stocks
          </h2>
          <p className="text-sm text-slate-400">
            S&amp;P 500 stocks where the 14-period weekly RSI is above{" "}
            <span className="font-medium text-rose-300">70</span>{" "}
            — overbought territory where buying pressure may be exhausted and a pullback could occur.
          </p>
        </div>
        {overboughtData ? (
          <OverboughtTable
            title="Weekly RSI Above 70"
            subtitle="Stocks with extreme weekly overbought conditions — sorted by most overbought first."
            rows={overboughtData.stocks}
          />
        ) : overboughtLoading ? (
          <div className="flex items-center gap-3 rounded-xl border border-rose-500/30 bg-slate-900/30 px-4 py-6 text-sm text-slate-400">
            <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-slate-700 border-t-rose-400" />
            Scanning weekly RSI...
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 rounded-xl border border-rose-500/30 bg-slate-900/30 px-4 py-6 text-sm text-slate-400">
            <span>Could not load overbought data.</span>
            <button
              type="button"
              className="rounded-md bg-rose-600 px-3 py-2 text-sm font-medium text-white hover:bg-rose-500"
              onClick={() => void runOverboughtFetch({ refresh: true })}
            >
              Retry
            </button>
          </div>
        )}
      </section>

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

      {/* ─── Daily RSI Oversold Section ─── */}
      {dailyOversoldData ? (
        <section className="mt-8">
          <div className="mb-4">
            <h2 className="text-lg font-semibold tracking-tight text-slate-100">
              Daily RSI Oversold Stocks
            </h2>
            <p className="text-sm text-slate-400">
              S&amp;P 500 stocks where the 14-period daily RSI is below{" "}
              <span className="font-medium text-emerald-300">{dailyOversoldData.rsiThreshold}</span>{" "}
              — short-term oversold conditions.
            </p>
          </div>
          <OversoldTable
            title="Daily RSI Below 30"
            subtitle="Stocks with daily oversold conditions — sorted by most oversold first."
            rows={dailyOversoldData.stocks}
            initialSortKey="dailyRSI"
          />
        </section>
      ) : dailyOversoldLoading ? (
        <div className="mt-8 flex items-center gap-3 text-sm text-slate-400">
          <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-slate-700 border-t-emerald-400" />
          Scanning daily RSI...
        </div>
      ) : null}

      {/* ─── Daily RSI Overbought Section ─── */}
      {dailyOverboughtData ? (
        <section className="mt-8">
          <div className="mb-4">
            <h2 className="text-lg font-semibold tracking-tight text-slate-100">
              Daily RSI Overbought Stocks
            </h2>
            <p className="text-sm text-slate-400">
              S&amp;P 500 stocks where the 14-period daily RSI is above{" "}
              <span className="font-medium text-rose-300">{dailyOverboughtData.rsiThreshold}</span>{" "}
              — short-term overbought conditions.
            </p>
          </div>
          <OverboughtTable
            title="Daily RSI Above 70"
            subtitle="Stocks with daily overbought conditions — sorted by most overbought first."
            rows={dailyOverboughtData.stocks}
            initialSortKey="dailyRSI"
          />
        </section>
      ) : dailyOverboughtLoading ? (
        <div className="mt-8 flex items-center gap-3 text-sm text-slate-400">
          <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-slate-700 border-t-rose-400" />
          Scanning daily RSI...
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

