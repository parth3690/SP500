"use client";

import { useState, useEffect } from "react";
import { fetchNyseSmidAgent } from "@/lib/api";
import type { InstitutionalScannerResponse, InstitutionalCandidate } from "@/lib/types";
import { formatNumber, formatPercent, formatPrice } from "@/lib/format";

export default function NyseSmidAgent() {
  const [data, setData] = useState<InstitutionalScannerResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [limit, setLimit] = useState(20);
  const [minScore, setMinScore] = useState(65);
  const [riskMode, setRiskMode] = useState<"balanced" | "aggressive" | "defensive">("balanced");
  const [regime, setRegime] = useState<"auto" | "risk_on" | "neutral" | "risk_off">("auto");
  const [tickerInput, setTickerInput] = useState("");

  const load = async (refresh = false) => {
    setLoading(true);
    setError(null);
    try {
      const tickers = tickerInput.trim() 
        ? tickerInput.split(/[,\s]+/).filter(t => t.length > 0)
        : undefined;
      
      const result = await fetchNyseSmidAgent({
        tickers,
        limit,
        minScore,
        riskMode,
        regime,
        refresh,
      });
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const takeCandidates = data?.candidates.filter((c) => c.tradeGate.decision === "TAKE") ?? [];
  const passCandidates = data?.candidates.filter((c) => c.tradeGate.decision === "PASS") ?? [];

  return (
    <div className="mx-auto max-w-[1600px] space-y-6 px-4 py-6">
      {/* Header */}
      <div className="flex flex-col gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">NYSE SMID Agent</h1>
          <p className="mt-1 text-sm text-slate-400">
            Institutional scanner for NYSE-listed small-mid cap stocks ($100M-$2B market cap)
          </p>
          <p className="mt-1 text-xs text-slate-500">
            Exchange: NYSE • Cap Range: $100M-$2B • Reuses S&P 500 data pipeline
          </p>
        </div>
      </div>

      {/* Controls */}
      <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
        <div className="mb-4">
          <label className="mb-2 block text-sm font-medium text-slate-300">
            Ticker List (optional)
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={tickerInput}
              onChange={(e) => setTickerInput(e.target.value)}
              placeholder="e.g., AAPL, MSFT, GOOGL (leave empty to scan all)"
              className="flex-1 rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-100 placeholder-slate-500"
            />
            <button
              onClick={() => load(true)}
              disabled={loading}
              className="rounded-md bg-emerald-600 px-6 py-2 text-sm font-medium text-white transition-colors hover:bg-emerald-700 disabled:opacity-50"
            >
              {loading ? "Running..." : "Run Agent"}
            </button>
          </div>
          <p className="mt-1 text-xs text-slate-500">
            Enter specific tickers or leave empty to scan the entire NYSE SMID universe
          </p>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-4">
          <div>
            <label className="mb-1 block text-xs font-medium text-slate-400">Limit</label>
            <input
              type="number"
              min={5}
              max={50}
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
              className="w-full rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-100"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs font-medium text-slate-400">Min Alpha Score</label>
            <input
              type="number"
              min={0}
              max={100}
              step={5}
              value={minScore}
              onChange={(e) => setMinScore(Number(e.target.value))}
              className="w-full rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-100"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs font-medium text-slate-400">Risk Mode</label>
            <select
              value={riskMode}
              onChange={(e) => setRiskMode(e.target.value as typeof riskMode)}
              className="w-full rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-100"
            >
              <option value="balanced">Balanced</option>
              <option value="aggressive">Aggressive</option>
              <option value="defensive">Defensive</option>
            </select>
          </div>
          <div>
            <label className="mb-1 block text-xs font-medium text-slate-400">Market Regime</label>
            <select
              value={regime}
              onChange={(e) => setRegime(e.target.value as typeof regime)}
              className="w-full rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-100"
            >
              <option value="auto">Auto</option>
              <option value="risk_on">Risk On</option>
              <option value="neutral">Neutral</option>
              <option value="risk_off">Risk Off</option>
            </select>
          </div>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-800 bg-red-950/50 p-4">
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      {data && (
        <>
          {/* Universe Info */}
          <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
            <h2 className="mb-3 text-lg font-bold text-slate-100">Universe Info</h2>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <div>
                <p className="text-xs text-slate-400">Universe</p>
                <p className="text-sm font-medium text-slate-100">{data.meta.universe}</p>
              </div>
              <div>
                <p className="text-xs text-slate-400">Exchange</p>
                <p className="text-sm font-medium text-slate-100">NYSE</p>
              </div>
              <div>
                <p className="text-xs text-slate-400">Cap Range</p>
                <p className="text-sm font-medium text-slate-100">$100M - $2B</p>
              </div>
              <div>
                <p className="text-xs text-slate-400">Names Scanned</p>
                <p className="text-sm font-medium text-slate-100">{data.meta.computed || 0}</p>
              </div>
            </div>
          </div>

          {/* Market Regime */}
          <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
            <h2 className="mb-3 text-lg font-bold text-slate-100">Market Regime</h2>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <div>
                <p className="text-xs text-slate-400">State</p>
                <p className="text-sm font-medium text-slate-100">
                  {data.marketRegime.effectiveState?.replace("_", " ")}
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-400">SPY Trend</p>
                <p className="text-sm font-medium text-slate-100">
                  {data.marketRegime.spyTrend?.replace("_", " ")}
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-400">SPY Drawdown</p>
                <p className="text-sm font-medium text-slate-100">
                  {data.marketRegime.spyDrawdownPct ? formatPercent(data.marketRegime.spyDrawdownPct) : "N/A"}
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-400">Risk Mode</p>
                <p className="text-sm font-medium text-slate-100">{data.marketRegime.riskMode}</p>
              </div>
            </div>
          </div>

          {/* Trade Gate Summary */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="rounded-lg border border-emerald-800 bg-emerald-950/30 p-4">
              <h3 className="mb-2 text-lg font-bold text-emerald-400">TAKE ({takeCandidates.length})</h3>
              <p className="text-sm text-slate-300">
                High-confidence trades that pass all institutional gate conditions.
              </p>
            </div>
            <div className="rounded-lg border border-slate-700 bg-slate-800/30 p-4">
              <h3 className="mb-2 text-lg font-bold text-slate-400">PASS ({passCandidates.length})</h3>
              <p className="text-sm text-slate-300">
                Ideas below the institutional-grade threshold.
              </p>
            </div>
          </div>

          {/* TAKE Candidates */}
          {takeCandidates.length > 0 && (
            <div>
              <h2 className="mb-4 text-xl font-bold text-emerald-400">TAKE - High-Confidence Trades</h2>
              <div className="space-y-4">
                {takeCandidates.map((candidate) => (
                  <CandidateCard key={candidate.ticker} candidate={candidate} />
                ))}
              </div>
            </div>
          )}

          {/* PASS Candidates */}
          {passCandidates.length > 0 && (
            <div>
              <h2 className="mb-4 text-xl font-bold text-slate-400">PASS - Below Threshold</h2>
              <div className="space-y-4">
                {passCandidates.map((candidate) => (
                  <CandidateCard key={candidate.ticker} candidate={candidate} />
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {!data && !loading && !error && (
        <div className="py-12 text-center">
          <p className="text-slate-400">Click "Run Agent" to scan NYSE SMID stocks</p>
        </div>
      )}
    </div>
  );
}

// Reuse the CandidateCard component from InstitutionalScanner
function CandidateCard({ candidate }: { candidate: InstitutionalCandidate }) {
  const [expanded, setExpanded] = useState(false);
  const isTake = candidate.tradeGate.decision === "TAKE";

  return (
    <div
      className={`rounded-lg border p-4 transition-colors ${
        isTake ? "border-emerald-700 bg-emerald-950/20" : "border-slate-700 bg-slate-800/30"
      }`}
    >
      {/* Header */}
      <div className="mb-3 flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-bold text-slate-100">{candidate.ticker}</h3>
            <span
              className={`rounded px-2 py-1 text-xs font-bold ${
                isTake ? "bg-emerald-600 text-white" : "bg-slate-700 text-slate-300"
              }`}
            >
              {candidate.tradeGate.decision}
            </span>
          </div>
          <p className="text-sm text-slate-400">{candidate.companyName}</p>
          <p className="text-xs text-slate-500">{candidate.sector}</p>
        </div>
        <div className="text-right">
          <p className="text-2xl font-bold text-slate-100">{formatPrice(candidate.currentPrice)}</p>
          <p className="text-sm text-slate-400">Alpha: {formatNumber(candidate.alphaScore)}</p>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="mb-3 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <div>
          <p className="text-xs text-slate-400">Confidence</p>
          <p
            className={`text-sm font-bold ${
              candidate.confidence.trustworthy ? "text-emerald-400" : "text-amber-400"
            }`}
          >
            {formatNumber(candidate.confidence.confidence)}%
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-400">Win Rate</p>
          <p className="text-sm font-bold text-slate-100">
            {candidate.backtest.valid && candidate.backtest.winRate
              ? formatNumber(candidate.backtest.winRate) + "%"
              : "N/A"}
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-400">Alpha vs SPY</p>
          <p
            className={`text-sm font-bold ${
              (candidate.backtest.alphaAvgReturn ?? 0) > 0 ? "text-emerald-400" : "text-red-400"
            }`}
          >
            {candidate.backtest.alphaAvgReturn ? formatPercent(candidate.backtest.alphaAvgReturn) : "N/A"}
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-400">Samples</p>
          <p className="text-sm font-bold text-slate-100">{candidate.backtest.sampleSize}</p>
        </div>
      </div>

      {/* Gate Reasons */}
      <div className="mb-3">
        <p className="mb-1 text-xs font-bold text-slate-400">Gate Decision</p>
        <ul className="space-y-1">
          {candidate.tradeGate.reasons.map((reason, idx) => (
            <li key={idx} className="text-xs text-slate-300">
              • {reason}
            </li>
          ))}
        </ul>
      </div>

      <button
        onClick={() => setExpanded(!expanded)}
        className="text-xs font-medium text-slate-400 hover:text-slate-200"
      >
        {expanded ? "Show Less ▲" : "Show Details ▼"}
      </button>

      {expanded && (
        <div className="mt-4 space-y-4 border-t border-slate-700 pt-4">
          {/* Expanded details (same as InstitutionalScanner) */}
          <div>
            <h4 className="mb-2 text-sm font-bold text-slate-300">Backtest Results</h4>
            <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-3">
              <div>
                <span className="text-slate-500">Avg Return:</span>{" "}
                <span className="text-slate-100">
                  {candidate.backtest.avgReturn ? formatPercent(candidate.backtest.avgReturn) : "N/A"}
                </span>
              </div>
              <div>
                <span className="text-slate-500">Max Drawdown:</span>{" "}
                <span className="text-red-400">
                  {candidate.backtest.maxDrawdown ? formatPercent(candidate.backtest.maxDrawdown) : "N/A"}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
