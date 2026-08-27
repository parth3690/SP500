"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import clsx from "clsx";

import { formatMoney, formatPct } from "@/lib/format";
import type { WeeklyMaWatchResponse, WeeklyMaWatchRow, WeeklyMaWatchSignal } from "@/lib/types";

type Filter = "all" | WeeklyMaWatchSignal;
type SortKey = "ticker" | "companyName" | "sector" | "currentPrice" | "dailySma" | "dailyDistancePct" | "weeklySma" | "distancePct" | "signal";
type SortDir = "asc" | "desc";

const SIGNAL_ORDER: Record<WeeklyMaWatchSignal, number> = {
  crossed_below: 0,
  below: 1,
  near: 2,
  reclaimed: 3,
};

const SIGNAL_LABEL: Record<WeeklyMaWatchSignal, string> = {
  crossed_below: "Crossed below",
  below: "At / below",
  near: "Near",
  reclaimed: "Reclaimed",
};

function signalClass(signal: WeeklyMaWatchSignal): string {
  if (signal === "crossed_below") return "border-rose-400/40 bg-rose-500/15 text-rose-200";
  if (signal === "below") return "border-red-400/30 bg-red-500/10 text-red-200";
  if (signal === "reclaimed") return "border-emerald-400/40 bg-emerald-500/15 text-emerald-200";
  return "border-amber-400/40 bg-amber-500/15 text-amber-200";
}

function compareRows(a: WeeklyMaWatchRow, b: WeeklyMaWatchRow, key: SortKey): number {
  if (key === "signal") return SIGNAL_ORDER[a.signal] - SIGNAL_ORDER[b.signal];
  const av = a[key];
  const bv = b[key];
  if (typeof av === "number" && typeof bv === "number") return av - bv;
  return String(av).localeCompare(String(bv));
}

export default function WeeklyMaWatchTable({ data }: { data: WeeklyMaWatchResponse }) {
  const [filter, setFilter] = useState<Filter>("all");
  const [sortKey, setSortKey] = useState<SortKey>("signal");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  const filters: Array<{ key: Filter; label: string; count: number }> = [
    { key: "all", label: "All", count: data.stocks.length },
    { key: "crossed_below", label: "Crossed below", count: data.meta.crossedBelowCount },
    { key: "below", label: "Below", count: data.meta.belowCount },
    { key: "near", label: "Near", count: data.meta.nearCount },
    { key: "reclaimed", label: "Reclaimed", count: data.meta.reclaimedCount },
  ];

  const rows = useMemo(() => {
    const selected = filter === "all" ? data.stocks : data.stocks.filter((row) => row.signal === filter);
    return [...selected].sort((a, b) => {
      const comparison = compareRows(a, b, sortKey);
      return sortDir === "asc" ? comparison : -comparison;
    });
  }, [data.stocks, filter, sortDir, sortKey]);

  const header = (key: SortKey, label: string, alignRight = false) => (
    <button
      type="button"
      className={clsx(
        "flex w-full items-center gap-1 px-3 py-2 text-left text-xs font-semibold text-slate-300 hover:text-white",
        alignRight && "justify-end text-right",
      )}
      onClick={() => {
        if (sortKey === key) setSortDir((current) => (current === "asc" ? "desc" : "asc"));
        else {
          setSortKey(key);
          setSortDir("asc");
        }
      }}
    >
      <span>{label}</span>
      {sortKey === key ? <span className="text-[10px] text-slate-500">{sortDir === "asc" ? "▲" : "▼"}</span> : null}
    </button>
  );

  return (
    <div className="overflow-hidden rounded-lg border border-cyan-500/30 bg-slate-900/30">
      <div className="flex flex-col gap-3 border-b border-cyan-500/20 bg-cyan-950/10 px-4 py-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div className="text-sm font-semibold text-slate-100">
            200 DMA and {data.maLength}-Week {data.maType}
          </div>
          <div className="mt-1 text-xs text-slate-400">
            {data.meta.computed} of {data.meta.total} stocks have enough weekly history · {data.meta.skipped} skipped
          </div>
        </div>
        <div className="flex max-w-full overflow-x-auto rounded-md border border-slate-700 bg-slate-950/60 p-1">
          {filters.map((item) => (
            <button
              key={item.key}
              type="button"
              className={clsx(
                "whitespace-nowrap rounded px-2.5 py-1.5 text-xs font-medium transition-colors",
                filter === item.key ? "bg-cyan-500/20 text-cyan-100" : "text-slate-400 hover:bg-slate-800 hover:text-slate-200",
              )}
              onClick={() => setFilter(item.key)}
            >
              {item.label} {item.count}
            </button>
          ))}
        </div>
      </div>

      <div className="overflow-auto">
        <table className="w-full min-w-[920px]">
          <thead className="bg-slate-950/40">
            <tr className="border-b border-slate-800">
              <th>{header("ticker", "Ticker")}</th>
              <th>{header("companyName", "Company")}</th>
              <th>{header("sector", "Sector")}</th>
              <th>{header("signal", "Status")}</th>
              <th>{header("currentPrice", "Price", true)}</th>
              <th>{header("dailySma", "200 DMA", true)}</th>
              <th>{header("weeklySma", `${data.maLength} Weekly MA`, true)}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.ticker} className="border-b border-slate-900/70 hover:bg-slate-950/40">
                <td className="px-3 py-2 text-sm font-semibold">
                  <Link
                    href={`/research/${encodeURIComponent(row.ticker)}`}
                    className="text-cyan-200 underline decoration-dotted underline-offset-2 hover:text-cyan-100"
                  >
                    {row.ticker}
                  </Link>
                </td>
                <td className="px-3 py-2 text-sm text-slate-200">{row.companyName}</td>
                <td className="px-3 py-2 text-sm text-slate-300">{row.sector}</td>
                <td className="px-3 py-2">
                  <span className={clsx("inline-flex rounded border px-2 py-1 text-[11px] font-semibold", signalClass(row.signal))}>
                    {SIGNAL_LABEL[row.signal]}
                  </span>
                </td>
                <td className="px-3 py-2 text-right text-sm tabular-nums text-slate-200">
                  {formatMoney(row.currentPrice)}
                  <div className="text-[10px] text-slate-500">{row.priceDate}</div>
                </td>
                <td className="px-3 py-2 text-right text-sm tabular-nums text-slate-200">
                  {formatMoney(row.dailySma)}
                  <div className={clsx("text-[10px] font-semibold", row.dailyDistancePct < 0 ? "text-rose-300" : "text-emerald-300")}>
                    {formatPct(row.dailyDistancePct)}
                  </div>
                </td>
                <td className="px-3 py-2 text-right text-sm tabular-nums text-slate-200">
                  {formatMoney(row.weeklySma)}
                  <div className={clsx(
                    "text-[10px] font-semibold",
                    row.distancePct <= 0 ? "text-rose-300" : row.signal === "reclaimed" ? "text-emerald-300" : "text-amber-300",
                  )}>
                    {formatPct(row.distancePct)}
                  </div>
                </td>
              </tr>
            ))}
            {rows.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center text-sm text-slate-400">
                  No S&amp;P 500 stocks currently match this status.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>

      <div className="border-t border-cyan-500/20 px-4 py-3 text-xs text-slate-400">
        The 200 DMA shows the primary daily trend beside the longer 200-week level. Alert categories remain based on the weekly SMA from the uploaded TradingView rules.
      </div>
    </div>
  );
}
