"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import clsx from "clsx";

import type { OversoldRow } from "@/lib/types";

type SortKey = "ticker" | "companyName" | "sector" | "currentPrice" | "weeklyRSI" | "dailyRSI";
type SortDir = "asc" | "desc";

function compare(a: unknown, b: unknown): number {
  if (a == null && b == null) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b));
}

function formatMoney(v: number): string {
  return v >= 1000 ? v.toFixed(0) : v >= 100 ? v.toFixed(2) : v.toFixed(3);
}

function rsiColor(rsi: number | null): string {
  if (rsi == null) return "text-slate-400";
  if (rsi <= 20) return "text-emerald-400 font-bold";
  if (rsi <= 25) return "text-emerald-300";
  return "text-emerald-200";
}

function rsiBadge(rsi: number): string {
  if (rsi <= 20) return "Extremely Oversold";
  if (rsi <= 25) return "Very Oversold";
  return "Oversold";
}

export default function OversoldTable(props: {
  title: string;
  subtitle?: string;
  rows: OversoldRow[];
}) {
  const [sortKey, setSortKey] = useState<SortKey>("weeklyRSI");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  const sorted = useMemo(() => {
    const copy = [...props.rows];
    copy.sort((r1, r2) => {
      const v1 = r1[sortKey];
      const v2 = r2[sortKey];
      const c = compare(v1, v2);
      return sortDir === "asc" ? c : -c;
    });
    return copy;
  }, [props.rows, sortKey, sortDir]);

  const headerCell = (key: SortKey, label: string, alignRight?: boolean) => (
    <button
      className={clsx(
        "flex w-full items-center gap-1 px-3 py-2 text-left text-xs font-semibold text-slate-300 hover:text-slate-100",
        alignRight && "justify-end text-right"
      )}
      onClick={() => {
        if (sortKey === key) {
          setSortDir((d) => (d === "asc" ? "desc" : "asc"));
        } else {
          setSortKey(key);
          setSortDir("asc");
        }
      }}
    >
      <span>{label}</span>
      {sortKey === key ? (
        <span className="text-[10px] text-slate-400">{sortDir === "asc" ? "▲" : "▼"}</span>
      ) : null}
    </button>
  );

  return (
    <div className="rounded-xl border border-emerald-500/30 bg-slate-900/30">
      {(props.title || props.subtitle) && (
        <div className="border-b border-emerald-500/30 bg-emerald-950/10 px-4 py-3">
          <div className="flex items-center gap-2">
            {props.title ? (
              <h2 className="text-sm font-semibold text-slate-200">{props.title}</h2>
            ) : null}
            <span className="inline-flex items-center rounded-full border border-emerald-400/30 bg-emerald-400/20 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-emerald-300">
              {props.rows.length} stocks
            </span>
          </div>
          {props.subtitle ? (
            <p className="mt-1 text-xs text-slate-400">{props.subtitle}</p>
          ) : null}
        </div>
      )}

      <div className="overflow-auto">
        <table className="w-full">
          <thead className="bg-slate-950/40">
            <tr className="border-b border-emerald-500/30">
              <th className="w-[8%]">{headerCell("ticker", "Ticker")}</th>
              <th className="w-[24%]">{headerCell("companyName", "Company")}</th>
              <th className="w-[18%]">{headerCell("sector", "Sector")}</th>
              <th className="w-[12%]">{headerCell("currentPrice", "Price", true)}</th>
              <th className="w-[14%]">{headerCell("weeklyRSI", "Weekly RSI", true)}</th>
              <th className="w-[14%]">{headerCell("dailyRSI", "Daily RSI", true)}</th>
              <th className="w-[10%]">
                <span className="px-3 py-2 text-xs font-semibold text-slate-300">Status</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => (
              <tr
                key={r.ticker}
                className="border-b border-slate-900/60 hover:bg-slate-950/30"
              >
                <td className="px-3 py-2 text-sm font-semibold">
                  <Link
                    href={`/research/${encodeURIComponent(r.ticker)}`}
                    className="text-emerald-200 underline decoration-dotted underline-offset-2 transition-colors hover:text-emerald-100"
                  >
                    {r.ticker}
                  </Link>
                </td>
                <td className="px-3 py-2 text-sm text-slate-200">{r.companyName}</td>
                <td className="px-3 py-2 text-sm text-slate-300">{r.sector}</td>
                <td className="px-3 py-2 text-right text-sm text-slate-200 tabular-nums">
                  ${formatMoney(r.currentPrice)}
                  <div className="text-[10px] text-slate-500">{r.priceDate}</div>
                </td>
                <td className={clsx("px-3 py-2 text-right text-sm tabular-nums", rsiColor(r.weeklyRSI))}>
                  {r.weeklyRSI.toFixed(2)}
                  {/* RSI bar */}
                  <div className="mt-0.5 h-1 w-full rounded-full bg-slate-800">
                    <div
                      className="h-full rounded-full bg-emerald-500 transition-all"
                      style={{ width: `${Math.min(100, r.weeklyRSI)}%` }}
                    />
                  </div>
                </td>
                <td className={clsx("px-3 py-2 text-right text-sm tabular-nums", rsiColor(r.dailyRSI))}>
                  {r.dailyRSI != null ? r.dailyRSI.toFixed(2) : "N/A"}
                </td>
                <td className="px-3 py-2">
                  <span
                    className={clsx(
                      "inline-block rounded-full border px-2 py-0.5 text-[10px] font-semibold",
                      r.weeklyRSI <= 20
                        ? "border-emerald-400/40 bg-emerald-400/20 text-emerald-300"
                        : r.weeklyRSI <= 25
                        ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-400"
                        : "border-emerald-500/20 bg-emerald-500/10 text-emerald-400/80"
                    )}
                  >
                    {rsiBadge(r.weeklyRSI)}
                  </span>
                </td>
              </tr>
            ))}
            {sorted.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-4 py-6 text-sm text-slate-400">
                  No stocks with weekly RSI below the threshold.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>

      <div className="border-t border-emerald-500/30 px-4 py-3 text-xs text-slate-400">
        Weekly RSI below 30 indicates the stock has been heavily sold on a weekly timeframe — a potential buying opportunity as the selling pressure may be exhausted.
      </div>
    </div>
  );
}
