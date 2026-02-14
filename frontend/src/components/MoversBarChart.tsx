"use client";

import {
  Bar,
  BarChart,
  Cell,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

import type { MoverRow } from "@/lib/types";

type ChartRow = { ticker: string; pctChange: number; side: "Gainer" | "Loser" };

function buildSeries(gainers: MoverRow[], losers: MoverRow[]): ChartRow[] {
  const g = gainers.map((r) => ({ ticker: r.ticker, pctChange: r.pctChange, side: "Gainer" as const }));
  const l = losers
    .slice()
    .reverse()
    .map((r) => ({ ticker: r.ticker, pctChange: r.pctChange, side: "Loser" as const }));
  return [...l, ...g];
}

export default function MoversBarChart({ gainers, losers }: { gainers: MoverRow[]; losers: MoverRow[] }) {
  const data = buildSeries(gainers, losers);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
        <XAxis dataKey="ticker" tick={{ fill: "#94a3b8", fontSize: 11 }} interval={0} angle={-45} textAnchor="end" height={70} />
        <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
        <Tooltip
          contentStyle={{ background: "#0b1220", border: "1px solid #1f2937", color: "#e2e8f0" }}
          formatter={(v: number) => [`${v.toFixed(2)}%`, "% change"]}
        />
        <Bar dataKey="pctChange" radius={[4, 4, 0, 0]}>
          {data.map((entry) => (
            <Cell key={entry.ticker} fill={entry.pctChange >= 0 ? "#34d399" : "#fb7185"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
