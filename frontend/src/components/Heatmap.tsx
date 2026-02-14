"use client";

import clsx from "clsx";
import { useMemo, useState } from "react";

import type { MoverRow } from "@/lib/types";

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

function colorForPct(pct: number): string {
  // Clamp to reduce outlier dominance.
  const v = clamp(pct, -20, 20) / 20; // [-1..1]
  if (v === 0) return "bg-slate-800";
  if (v > 0) {
    const i = Math.round(v * 4); // 1..4
    return ["bg-emerald-950", "bg-emerald-900", "bg-emerald-800", "bg-emerald-700", "bg-emerald-600"][i];
  }
  const i = Math.round(Math.abs(v) * 4);
  return ["bg-rose-950", "bg-rose-900", "bg-rose-800", "bg-rose-700", "bg-rose-600"][i];
}

export default function Heatmap({ rows }: { rows: MoverRow[] }) {
  const [sector, setSector] = useState<string>("All");

  const sectors = useMemo(() => {
    const set = new Set(rows.map((r) => r.sector).filter(Boolean));
    return ["All", ...Array.from(set).sort()];
  }, [rows]);

  const filtered = useMemo(() => {
    if (sector === "All") return rows;
    return rows.filter((r) => r.sector === sector);
  }, [rows, sector]);

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs text-slate-400">
          Showing <span className="text-slate-200">{filtered.length}</span> tickers
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-400">Sector</label>
          <select
            className="rounded-md border border-slate-700 bg-slate-950 px-2 py-2 text-sm"
            value={sector}
            onChange={(e) => setSector(e.target.value)}
          >
            {sectors.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="mt-3 grid grid-cols-6 gap-2 sm:grid-cols-8 md:grid-cols-10 lg:grid-cols-12">
        {filtered.map((r) => (
          <div
            key={r.ticker}
            title={`${r.ticker} · ${r.companyName} · ${r.pctChange.toFixed(2)}%`}
            className={clsx(
              "flex h-10 items-center justify-center rounded-md border border-slate-900/60 text-[11px] font-semibold text-slate-100",
              colorForPct(r.pctChange)
            )}
          >
            {r.ticker}
          </div>
        ))}
        {filtered.length === 0 ? (
          <div className="col-span-full rounded-md border border-slate-800 bg-slate-950/40 p-4 text-sm text-slate-400">
            No tickers.
          </div>
        ) : null}
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-slate-400">
        <span className="inline-flex items-center gap-1">
          <span className="h-3 w-3 rounded bg-rose-700" /> Down
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="h-3 w-3 rounded bg-slate-800" /> Flat
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="h-3 w-3 rounded bg-emerald-700" /> Up
        </span>
        <span className="ml-auto text-[11px] text-slate-500">Color scale clamped to ±20%.</span>
      </div>
    </div>
  );
}

