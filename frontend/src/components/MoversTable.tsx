"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import clsx from "clsx";

import type { MoverRow } from "@/lib/types";

type SortKey = "rank" | "ticker" | "companyName" | "sector" | "currentPrice" | "pastPrice" | "pctChange";
type SortDir = "asc" | "desc";

function compare(a: unknown, b: unknown): number {
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b));
}

function formatMoney(v: number): string {
  return v >= 1000 ? v.toFixed(0) : v >= 100 ? v.toFixed(2) : v.toFixed(3);
}

function formatPct(v: number): string {
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}%`;
}

export default function MoversTable(props: {
  title: string;
  subtitle?: string;
  footerNote?: string;
  rows: MoverRow[];
  variant: "gainers" | "losers" | "neutral";
  defaultSort: { key: SortKey; dir: SortDir };
}) {
  const [sortKey, setSortKey] = useState<SortKey>(props.defaultSort.key);
  const [sortDir, setSortDir] = useState<SortDir>(props.defaultSort.dir);

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

  const pctClass = (v: number) =>
    v > 0 ? "text-emerald-300" : v < 0 ? "text-rose-300" : "text-slate-200";

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
          setSortDir(key === "pctChange" ? props.defaultSort.dir : "asc");
        }
      }}
    >
      <span>{label}</span>
      {sortKey === key ? <span className="text-[10px] text-slate-400">{sortDir === "asc" ? "▲" : "▼"}</span> : null}
    </button>
  );

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/30">
      {(props.title || props.subtitle) && (
        <div className="border-b border-slate-800 px-4 py-3">
          {props.title ? <h2 className="text-sm font-semibold text-slate-200">{props.title}</h2> : null}
          {props.subtitle ? <p className="mt-1 text-xs text-slate-400">{props.subtitle}</p> : null}
        </div>
      )}

      <div className="overflow-auto">
        <table className="w-full min-w-[720px]">
          <thead className="bg-slate-950/40">
            <tr className="border-b border-slate-800">
              <th className="w-14">{headerCell("rank", "Rank")}</th>
              <th className="w-24">{headerCell("ticker", "Ticker")}</th>
              <th>{headerCell("companyName", "Company")}</th>
              <th className="w-64">{headerCell("sector", "Sector")}</th>
              <th className="w-28">{headerCell("currentPrice", "Current", true)}</th>
              <th className="w-28">{headerCell("pastPrice", "Past", true)}</th>
              <th className="w-28">{headerCell("pctChange", "% Change", true)}</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => (
              <tr key={`${r.ticker}-${r.rank}`} className="border-b border-slate-900/60 hover:bg-slate-950/30">
                <td className="px-3 py-2 text-sm text-slate-300">{r.rank}</td>
                <td className="px-3 py-2 text-sm font-semibold">
                  <Link
                    href={`/research/${encodeURIComponent(r.ticker)}`}
                    className="text-sky-300 underline decoration-dotted underline-offset-2 hover:text-sky-200 transition-colors"
                  >
                    {r.ticker}
                  </Link>
                </td>
                <td className="px-3 py-2 text-sm text-slate-200">{r.companyName}</td>
                <td className="px-3 py-2 text-sm text-slate-300">{r.sector}</td>
                <td className="px-3 py-2 text-right text-sm text-slate-200 tabular-nums">
                  {formatMoney(r.currentPrice)}
                  <div className="text-[10px] text-slate-500">{r.currentPriceDate}</div>
                </td>
                <td className="px-3 py-2 text-right text-sm text-slate-200 tabular-nums">
                  {formatMoney(r.pastPrice)}
                  <div className="text-[10px] text-slate-500">{r.pastPriceDate}</div>
                </td>
                <td className={clsx("px-3 py-2 text-right text-sm font-semibold tabular-nums", pctClass(r.pctChange))}>
                  {formatPct(r.pctChange)}
                </td>
              </tr>
            ))}
            {sorted.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-4 py-6 text-sm text-slate-400">
                  No rows.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>

      {props.footerNote ? (
        <div className="border-t border-slate-800 px-4 py-3 text-xs text-slate-400">{props.footerNote}</div>
      ) : null}
    </div>
  );
}

