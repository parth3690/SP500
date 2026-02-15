"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import clsx from "clsx";

import type { SectorSummaryRow, MoverRow } from "@/lib/types";

function formatPct(v: number): string {
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}%`;
}

function formatMoney(v: number): string {
  return v >= 1000 ? `$${v.toFixed(0)}` : v >= 100 ? `$${v.toFixed(2)}` : `$${v.toFixed(3)}`;
}

export default function SectorSummary({
  rows,
  allStocks = [],
}: {
  rows: SectorSummaryRow[];
  allStocks?: MoverRow[];
}) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  // Group stocks by sector for quick lookup
  const stocksBySector = useMemo(() => {
    const map: Record<string, MoverRow[]> = {};
    for (const s of allStocks) {
      if (!map[s.sector]) map[s.sector] = [];
      map[s.sector].push(s);
    }
    // Sort each sector's stocks by pctChange descending
    for (const key of Object.keys(map)) {
      map[key].sort((a, b) => b.pctChange - a.pctChange);
    }
    return map;
  }, [allStocks]);

  const toggle = (sector: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(sector)) {
        next.delete(sector);
      } else {
        next.add(sector);
      }
      return next;
    });
  };

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-4">
      <h2 className="text-sm font-semibold text-slate-200">Sector summary</h2>
      <p className="mt-1 text-xs text-slate-400">
        Click a sector to see individual stocks.
      </p>

      <div className="mt-3 overflow-auto">
        <table className="w-full min-w-[360px]">
          <thead className="bg-slate-950/40">
            <tr className="border-b border-slate-800">
              <th className="px-2 py-2 text-left text-xs font-semibold text-slate-300">Sector</th>
              <th className="px-2 py-2 text-right text-xs font-semibold text-slate-300">Avg</th>
              <th className="px-2 py-2 text-right text-xs font-semibold text-slate-300">Median</th>
              <th className="px-2 py-2 text-right text-xs font-semibold text-slate-300">Count</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 11).map((r) => {
              const isOpen = expanded.has(r.sector);
              const stocks = stocksBySector[r.sector] ?? [];
              const hasStocks = stocks.length > 0;

              return (
                <SectorRowGroup
                  key={r.sector}
                  row={r}
                  stocks={stocks}
                  isOpen={isOpen}
                  hasStocks={hasStocks}
                  onToggle={() => toggle(r.sector)}
                />
              );
            })}
            {rows.length === 0 ? (
              <tr>
                <td colSpan={4} className="px-2 py-6 text-xs text-slate-400">
                  No sector data.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Sector Row + Expandable Stock List ───────────────────────────────────

function SectorRowGroup({
  row,
  stocks,
  isOpen,
  hasStocks,
  onToggle,
}: {
  row: SectorSummaryRow;
  stocks: MoverRow[];
  isOpen: boolean;
  hasStocks: boolean;
  onToggle: () => void;
}) {
  return (
    <>
      {/* Sector summary row */}
      <tr
        className={clsx(
          "border-b border-slate-900/60 transition-colors",
          hasStocks ? "cursor-pointer hover:bg-slate-950/40" : "",
          isOpen && "bg-slate-950/30"
        )}
        onClick={hasStocks ? onToggle : undefined}
      >
        <td className="px-2 py-2 text-xs text-slate-200">
          <div className="flex items-center gap-1.5">
            {hasStocks && (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 20 20"
                fill="currentColor"
                className={clsx(
                  "h-3 w-3 flex-shrink-0 text-slate-500 transition-transform duration-200",
                  isOpen && "rotate-90"
                )}
              >
                <path
                  fillRule="evenodd"
                  d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z"
                  clipRule="evenodd"
                />
              </svg>
            )}
            <span className={clsx(hasStocks && "font-medium")}>{row.sector}</span>
          </div>
        </td>
        <td
          className={clsx(
            "px-2 py-2 text-right text-xs font-semibold tabular-nums",
            row.avgPctChange > 0 ? "text-emerald-400" : row.avgPctChange < 0 ? "text-rose-400" : "text-slate-200"
          )}
        >
          {formatPct(row.avgPctChange)}
        </td>
        <td className="px-2 py-2 text-right text-xs text-slate-300 tabular-nums">
          {formatPct(row.medianPctChange)}
        </td>
        <td className="px-2 py-2 text-right text-xs text-slate-400 tabular-nums">{row.count}</td>
      </tr>

      {/* Expanded stock list */}
      {isOpen && stocks.length > 0 && (
        <tr>
          <td colSpan={4} className="p-0">
            <div className="border-b border-slate-800 bg-slate-950/50">
              {/* Mini header */}
              <div className="flex items-center gap-2 border-b border-slate-800/60 px-3 py-1.5">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                  {row.sector}
                </span>
                <span className="text-[10px] text-slate-600">·</span>
                <span className="text-[10px] text-slate-500">{stocks.length} stocks</span>
              </div>

              {/* Stock rows */}
              <div className="max-h-64 overflow-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-slate-800/40">
                      <th className="px-3 py-1 text-left text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                        Ticker
                      </th>
                      <th className="px-3 py-1 text-left text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                        Company
                      </th>
                      <th className="px-3 py-1 text-right text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                        Price
                      </th>
                      <th className="px-3 py-1 text-right text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                        Change
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {stocks.map((s) => (
                      <tr
                        key={s.ticker}
                        className="border-b border-slate-900/30 hover:bg-slate-900/40 transition-colors"
                      >
                        <td className="px-3 py-1.5 text-xs font-semibold">
                          <Link
                            href={`/research/${encodeURIComponent(s.ticker)}`}
                            className="text-cyan-300 underline decoration-dotted underline-offset-2 hover:text-cyan-200 transition-colors"
                            onClick={(e) => e.stopPropagation()}
                          >
                            {s.ticker}
                          </Link>
                        </td>
                        <td className="px-3 py-1.5 text-xs text-slate-300 truncate max-w-[140px]">
                          {s.companyName}
                        </td>
                        <td className="px-3 py-1.5 text-right text-xs text-slate-200 tabular-nums">
                          {formatMoney(s.currentPrice)}
                        </td>
                        <td
                          className={clsx(
                            "px-3 py-1.5 text-right text-xs font-semibold tabular-nums",
                            s.pctChange > 0
                              ? "text-emerald-400"
                              : s.pctChange < 0
                              ? "text-rose-400"
                              : "text-slate-300"
                          )}
                        >
                          {formatPct(s.pctChange)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}
