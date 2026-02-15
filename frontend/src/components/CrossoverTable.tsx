"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import clsx from "clsx";

import type { CrossoverRow } from "@/lib/types";

type SortKey = "ticker" | "companyName" | "sector" | "currentPrice" | "dma50" | "dma200" | "gapPct";
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

export default function CrossoverTable(props: {
  title: string;
  subtitle?: string;
  rows: CrossoverRow[];
  variant: "golden" | "death";
}) {
  const [sortKey, setSortKey] = useState<SortKey>("gapPct");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  const sorted = useMemo(() => {
    const copy = [...props.rows];
    copy.sort((r1, r2) => {
      let v1: unknown = r1[sortKey];
      let v2: unknown = r2[sortKey];
      // For gapPct, sort by absolute value by default
      if (sortKey === "gapPct") {
        v1 = Math.abs(v1 as number);
        v2 = Math.abs(v2 as number);
      }
      const c = compare(v1, v2);
      return sortDir === "asc" ? c : -c;
    });
    return copy;
  }, [props.rows, sortKey, sortDir]);

  const isGolden = props.variant === "golden";

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

  const accentBorder = isGolden ? "border-amber-500/30" : "border-violet-500/30";
  const accentBg = isGolden ? "bg-amber-950/10" : "bg-violet-950/10";
  const badgeClass = isGolden
    ? "bg-amber-400/20 text-amber-300 border-amber-400/30"
    : "bg-violet-400/20 text-violet-300 border-violet-400/30";

  return (
    <div className={clsx("rounded-xl border bg-slate-900/30", accentBorder)}>
      {(props.title || props.subtitle) && (
        <div className={clsx("border-b px-4 py-3", accentBorder, accentBg)}>
          <div className="flex items-center gap-2">
            {props.title ? (
              <h2 className="text-sm font-semibold text-slate-200">{props.title}</h2>
            ) : null}
            <span
              className={clsx(
                "inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider",
                badgeClass
              )}
            >
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
            <tr className={clsx("border-b", accentBorder)}>
              <th className="w-[8%]">{headerCell("ticker", "Ticker")}</th>
              <th className="w-[28%]">{headerCell("companyName", "Company")}</th>
              <th className="w-[22%]">{headerCell("sector", "Sector")}</th>
              <th className="w-[12%]">{headerCell("currentPrice", "Price", true)}</th>
              <th className="w-[10%]">{headerCell("dma50", "50-DMA", true)}</th>
              <th className="w-[10%]">{headerCell("dma200", "200-DMA", true)}</th>
              <th className="w-[10%]">{headerCell("gapPct", "Gap %", true)}</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => {
              const gapColor = isGolden ? "text-amber-300" : "text-violet-300";
              return (
                <tr
                  key={r.ticker}
                  className={clsx("border-b border-slate-900/60 hover:bg-slate-950/30")}
                >
                  <td className="px-3 py-2 text-sm font-semibold">
                    <Link
                      href={`/research/${encodeURIComponent(r.ticker)}`}
                      className={clsx(
                        "underline decoration-dotted underline-offset-2 transition-colors",
                        isGolden
                          ? "text-amber-200 hover:text-amber-100"
                          : "text-violet-200 hover:text-violet-100"
                      )}
                    >
                      {r.ticker}
                    </Link>
                  </td>
                  <td className="px-3 py-2 text-sm text-slate-200">{r.companyName}</td>
                  <td className="px-3 py-2 text-sm text-slate-300">{r.sector}</td>
                  <td className="px-3 py-2 text-right text-sm text-slate-200 tabular-nums">
                    {formatMoney(r.currentPrice)}
                    <div className="text-[10px] text-slate-500">{r.priceDate}</div>
                  </td>
                  <td className="px-3 py-2 text-right text-sm text-slate-200 tabular-nums">
                    {formatMoney(r.dma50)}
                  </td>
                  <td className="px-3 py-2 text-right text-sm text-slate-200 tabular-nums">
                    {formatMoney(r.dma200)}
                  </td>
                  <td
                    className={clsx(
                      "px-3 py-2 text-right text-sm font-semibold tabular-nums",
                      gapColor
                    )}
                  >
                    {formatPct(r.gapPct)}
                  </td>
                </tr>
              );
            })}
            {sorted.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-4 py-6 text-sm text-slate-400">
                  No stocks near {isGolden ? "golden" : "death"} cross.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>

      <div className={clsx("border-t px-4 py-3 text-xs text-slate-400", accentBorder)}>
        {isGolden
          ? "50-DMA is below 200-DMA but within threshold — a bullish crossover may be approaching."
          : "50-DMA is above 200-DMA but within threshold — a bearish crossover may be approaching."}
      </div>
    </div>
  );
}
