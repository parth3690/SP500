"use client";

import { useState } from "react";
import Link from "next/link";
import clsx from "clsx";
import { fetchMultibagger } from "@/lib/api";
import type { MultibaggerResponse } from "@/lib/types";

function getErrorMessage(e: unknown): string {
  return e instanceof Error ? e.message : "Unknown error";
}

export default function MultibaggerScanner() {
  const [ticker, setTicker] = useState("");
  const [deep, setDeep] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<MultibaggerResponse | null>(null);

  const runScan = async (opts?: { refresh?: boolean }) => {
    const sym = ticker.trim().toUpperCase();
    if (!sym) {
      setError("Enter a ticker symbol");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await fetchMultibagger(sym, { deep, refresh: opts?.refresh });
      setResult(data);
      setTicker(sym);
    } catch (e) {
      setResult(null);
      setError(getErrorMessage(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-[1000px] px-4 py-8">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight">US Multibagger Scanner</h1>
        <p className="mt-1 text-sm text-slate-400">
          Check whether a US stock passes the fundamental multibagger-style checklist (12 criteria)
        </p>
      </header>

      <section className="mb-6 rounded-xl border border-slate-800 bg-slate-900/40 p-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
          <label className="flex-1 text-xs uppercase tracking-wide text-slate-400">
            Ticker
            <input
              className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 font-mono text-sm uppercase"
              placeholder="e.g. AAPL"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              onKeyDown={(e) => {
                if (e.key === "Enter") void runScan();
              }}
            />
          </label>
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input
              type="checkbox"
              className="rounded border-slate-600"
              checked={deep}
              onChange={(e) => setDeep(e.target.checked)}
            />
            Deep scan (5y ROE / CAGR)
          </label>
          <button
            type="button"
            disabled={loading}
            className={clsx(
              "rounded-md px-5 py-2 text-sm font-semibold",
              loading ? "bg-slate-700 text-slate-400" : "bg-emerald-600 text-white hover:bg-emerald-500",
            )}
            onClick={() => void runScan()}
          >
            {loading ? "Scanning…" : "Check ticker"}
          </button>
        </div>
        {error ? <p className="mt-3 text-sm text-rose-400">{error}</p> : null}
      </section>

      {result ? (
        <>
          <div className="mb-6 grid gap-3 sm:grid-cols-3">
            <ScoreCard
              label="Qualification"
              value={result.passedAll ? "Full pass" : "Partial"}
              accent={result.passedAll}
              detail={result.passedAll ? "All 12 criteria met" : `${result.fails.length} hard miss(es)`}
            />
            <ScoreCard
              label="Green count"
              value={`${result.nGreen} / ${result.nTotal}`}
              detail={`${Math.round((result.nGreen / result.nTotal) * 100)}% of checklist`}
            />
            <ScoreCard label="Sector P/E median" value={fmtNum(result.sectorPeMedian)} detail={result.sector} />
          </div>

          <section className="mb-4 rounded-xl border border-slate-800 bg-slate-900/30 p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h2 className="text-lg font-semibold">
                  {result.ticker}{" "}
                  <span className="text-base font-normal text-slate-400">— {result.name}</span>
                </h2>
                <p className="mt-1 text-xs text-slate-500">
                  As of {new Date(result.asOf).toLocaleString()}
                  {result.deep ? " · deep metrics" : ""}
                </p>
              </div>
              <div className="flex gap-2">
                <Link
                  href={`/research/${encodeURIComponent(result.ticker)}`}
                  className="rounded-md border border-slate-700 px-3 py-1.5 text-sm hover:bg-slate-800"
                >
                  Open research
                </Link>
                <button
                  type="button"
                  className="rounded-md border border-slate-700 px-3 py-1.5 text-sm hover:bg-slate-800"
                  onClick={() => void runScan({ refresh: true })}
                >
                  Refresh
                </button>
              </div>
            </div>

            {!result.passedAll && result.fails.length > 0 ? (
              <p className="mt-3 text-sm text-amber-300/90">
                Missed: <span className="font-mono">{result.fails.join(", ")}</span>
              </p>
            ) : null}
            {result.skipped.length > 0 ? (
              <p className="mt-1 text-sm text-slate-500">
                No data (soft): <span className="font-mono">{result.skipped.join(", ")}</span>
              </p>
            ) : null}
          </section>

          <div className="overflow-x-auto rounded-xl border border-slate-800">
            <table className="w-full min-w-[640px] text-sm">
              <thead>
                <tr className="border-b border-slate-800 bg-slate-900/60 text-left text-xs uppercase tracking-wide text-slate-400">
                  <th className="px-3 py-2">Criterion</th>
                  <th className="px-3 py-2">Threshold</th>
                  <th className="px-3 py-2">Reading</th>
                  <th className="px-3 py-2">Result</th>
                </tr>
              </thead>
              <tbody>
                {result.criteria.map((row) => (
                  <tr key={row.id} className="border-b border-slate-800/80 hover:bg-slate-900/30">
                    <td className="px-3 py-2">
                      <span className="inline-flex items-center gap-2">
                        <StatusDot status={row.status} />
                        {row.name}
                        {row.soft ? (
                          <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-500">soft</span>
                        ) : null}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-xs text-slate-400">{row.threshold}</td>
                    <td className="px-3 py-2 font-mono text-xs text-slate-300">{row.valueDisplay ?? "—"}</td>
                    <td className="px-3 py-2">
                      <StatusBadge status={row.status} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="mt-4 text-xs text-slate-500">
            Research tool only — not investment advice. Soft criteria with missing Yahoo data are skipped, not failed.
            Sector P/E median is computed from current S&amp;P 500 peers in the same sector.
          </p>
        </>
      ) : (
        <div className="rounded-xl border border-dashed border-slate-700 bg-slate-900/20 p-8 text-center text-sm text-slate-500">
          Enter a ticker and run the scan to see which multibagger criteria it passes or misses.
        </div>
      )}
    </div>
  );
}

function ScoreCard(props: { label: string; value: string; detail?: string; accent?: boolean }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
      <div className="text-xs uppercase tracking-wide text-slate-400">{props.label}</div>
      <div
        className={clsx(
          "mt-1 text-xl font-semibold",
          props.accent ? "text-emerald-400" : "text-slate-100",
        )}
      >
        {props.value}
      </div>
      {props.detail ? <div className="mt-0.5 text-xs text-slate-500">{props.detail}</div> : null}
    </div>
  );
}

function StatusDot(props: { status: string }) {
  return (
    <span
      className={clsx(
        "h-2 w-2 rounded-full",
        props.status === "pass" && "bg-emerald-400",
        props.status === "fail" && "bg-rose-500",
        props.status === "skip" && "border border-slate-600 bg-slate-800",
      )}
    />
  );
}

function StatusBadge(props: { status: string }) {
  return (
    <span
      className={clsx(
        "rounded px-2 py-0.5 text-xs font-medium uppercase",
        props.status === "pass" && "bg-emerald-500/15 text-emerald-300",
        props.status === "fail" && "bg-rose-500/15 text-rose-300",
        props.status === "skip" && "bg-slate-800 text-slate-500",
      )}
    >
      {props.status === "pass" ? "Pass" : props.status === "fail" ? "Fail" : "No data"}
    </span>
  );
}

function fmtNum(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return v.toFixed(2);
}
