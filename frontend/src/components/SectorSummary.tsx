"use client";

import type { SectorSummaryRow } from "@/lib/types";

function formatPct(v: number): string {
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}%`;
}

export default function SectorSummary({ rows }: { rows: SectorSummaryRow[] }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-4">
      <h2 className="text-sm font-semibold text-slate-200">Sector summary</h2>
      <p className="mt-1 text-xs text-slate-400">Average and median % change by GICS sector.</p>

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
            {rows.slice(0, 11).map((r) => (
              <tr key={r.sector} className="border-b border-slate-900/60 hover:bg-slate-950/30">
                <td className="px-2 py-2 text-xs text-slate-200">{r.sector}</td>
                <td className="px-2 py-2 text-right text-xs font-semibold text-slate-200 tabular-nums">
                  {formatPct(r.avgPctChange)}
                </td>
                <td className="px-2 py-2 text-right text-xs text-slate-300 tabular-nums">
                  {formatPct(r.medianPctChange)}
                </td>
                <td className="px-2 py-2 text-right text-xs text-slate-400 tabular-nums">{r.count}</td>
              </tr>
            ))}
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

