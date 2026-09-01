"use client";

import { useState, useEffect } from "react";
import { fetchInstitutionalScanner } from "@/lib/api";
import type { InstitutionalScannerResponse, InstitutionalCandidate } from "@/lib/types";
import { formatNumber, formatPercent, formatPrice } from "@/lib/format";

export default function InstitutionalScanner() {
  const [data, setData] = useState<InstitutionalScannerResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [limit, setLimit] = useState(20);
  const [minScore, setMinScore] = useState(65);
  const [riskMode, setRiskMode] = useState<"balanced" | "aggressive" | "defensive">("balanced");
  const [regime, setRegime] = useState<"auto" | "risk_on" | "neutral" | "risk_off">("auto");

  const load = async (refresh = false) => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchInstitutionalScanner({
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

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [limit, minScore, riskMode, regime]);

  const takeCandidates = data?.candidates.filter((c) => c.tradeGate.decision === "TAKE") ?? [];
  const passCandidates = data?.candidates.filter((c) => c.tradeGate.decision === "PASS") ?? [];

  return (
    <div className="mx-auto max-w-[1600px] space-y-6 px-4 py-6">
      {/* Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Institutional-Grade Scanner</h1>
          <p className="mt-1 text-sm text-slate-400">
            Walk-forward backtests • Simulation validation • Calibrated confidence • Hard trade gate
          </p>
        </div>
        <button
          onClick={() => load(true)}
          disabled={loading}
          className="rounded-md bg-emerald-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-emerald-700 disabled:opacity-50"
        >
          {loading ? "Loading..." : "Refresh"}
        </button>
      </div>

      {/* Convexity Alerts */}
      {data && data.convexityAlerts.length > 0 && (
        <div className="rounded-lg border-2 border-amber-500 bg-amber-950/50 p-4">
          <div className="mb-2 flex items-center gap-2">
            <span className="text-2xl">🚨</span>
            <h2 className="text-lg font-bold text-amber-400">High-Convexity Opportunities Detected</h2>
          </div>
          <div className="space-y-3">
            {data.convexityAlerts.map((alert) => (
              <div key={alert.ticker} className="rounded-md bg-slate-900/50 p-3">
                <div className="mb-1 flex items-start justify-between">
                  <div>
                    <span className="font-bold text-amber-400">{alert.ticker}</span>
                    <span className="ml-2 text-sm text-slate-400">{alert.type}</span>
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-bold text-amber-400">{alert.probability}% probability</div>
                    <div className="text-xs text-slate-400">{alert.expectedReturn} return potential</div>
                  </div>
                </div>
                <p className="text-sm text-slate-300">{alert.message}</p>
                <div className="mt-2 flex flex-wrap gap-4 text-xs text-slate-400">
                  <span>Stock Move Required: {formatPercent(alert.requiredStockMove)}</span>
                  <span>Current Price: {formatPrice(alert.currentPrice)}</span>
                  <span>Volatility: {formatPercent(alert.volatility)}</span>
                  <span>Alpha Score: {formatNumber(alert.alphaScore)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="grid grid-cols-1 gap-4 rounded-lg border border-slate-800 bg-slate-900/50 p-4 sm:grid-cols-4">
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

      {error && (
        <div className="rounded-lg border border-red-800 bg-red-950/50 p-4">
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      {loading && !data && (
        <div className="py-12 text-center">
          <p className="text-slate-400">Loading institutional scanner...</p>
        </div>
      )}

      {data && (
        <>
          {/* Market Regime Summary */}
          <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
            <h2 className="mb-3 text-lg font-bold text-slate-100">Market Regime</h2>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <div>
                <p className="text-xs text-slate-400">State</p>
                <p className="text-sm font-medium text-slate-100">{data.marketRegime.effectiveState.replace("_", " ")}</p>
              </div>
              <div>
                <p className="text-xs text-slate-400">SPY Trend</p>
                <p className="text-sm font-medium text-slate-100">{data.marketRegime.spyTrend.replace("_", " ")}</p>
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
                High-confidence trades that pass all gate conditions. These ideas have strong backtests,
                survive simulation scenarios, and meet minimum thresholds.
              </p>
            </div>
            <div className="rounded-lg border border-slate-700 bg-slate-800/30 p-4">
              <h3 className="mb-2 text-lg font-bold text-slate-400">PASS ({passCandidates.length})</h3>
              <p className="text-sm text-slate-300">
                Ideas that don't meet the institutional-grade threshold. May have weak backtests,
                insufficient samples, or failed simulation validation.
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

          {/* Meta Info */}
          <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
            <h3 className="mb-2 text-sm font-bold text-slate-400">Scanner Metadata</h3>
            <div className="space-y-1 text-xs text-slate-500">
              <p>Version: {data.meta.scannerVersion}</p>
              <p>Backtest: {data.meta.backtestHorizon}</p>
              <p>Scenarios: {data.meta.simulationScenarios.join(", ")}</p>
              <p>
                Gate: Confidence ≥ {data.meta.tradeGate.minConfidence}%, Win Rate ≥{" "}
                {data.meta.tradeGate.minWinRate}%, Sample ≥ {data.meta.tradeGate.minSampleSize}, Alpha ≥{" "}
                {data.meta.tradeGate.minAlphaVsBenchmark}%
              </p>
              <p>
                Convexity Alert: P ≥ {data.meta.convexityAlert.minProbability}%, Return ≥{" "}
                {data.meta.convexityAlert.minReturn}
              </p>
              <p>Returned: {data.meta.returned} of {data.meta.computed} computed</p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

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
                isTake
                  ? "bg-emerald-600 text-white"
                  : "bg-slate-700 text-slate-300"
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
          <p className="text-sm text-slate-400">Alpha Score: {formatNumber(candidate.alphaScore)}</p>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="mb-3 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <div>
          <p className="text-xs text-slate-400">Confidence</p>
          <p className={`text-sm font-bold ${candidate.confidence.trustworthy ? "text-emerald-400" : "text-amber-400"}`}>
            {formatNumber(candidate.confidence.confidence)}%
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-400">Backtest Win Rate</p>
          <p className="text-sm font-bold text-slate-100">
            {candidate.backtest.valid && candidate.backtest.winRate ? formatNumber(candidate.backtest.winRate) + "%" : "N/A"}
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-400">Alpha vs Benchmark</p>
          <p className={`text-sm font-bold ${(candidate.backtest.alphaAvgReturn ?? 0) > 0 ? "text-emerald-400" : "text-red-400"}`}>
            {candidate.backtest.alphaAvgReturn ? formatPercent(candidate.backtest.alphaAvgReturn) : "N/A"}
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-400">Sample Size</p>
          <p className="text-sm font-bold text-slate-100">{candidate.backtest.sampleSize}</p>
        </div>
      </div>

      {/* Trade Plan */}
      <div className="mb-3 rounded-md bg-slate-900/50 p-3">
        <p className="mb-2 text-xs font-bold text-slate-400">Trade Plan</p>
        <div className="grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
          <div>
            <span className="text-slate-500">Action:</span>{" "}
            <span className="font-medium text-slate-100">{candidate.tradePlan.action}</span>
          </div>
          <div>
            <span className="text-slate-500">Entry:</span>{" "}
            <span className="font-medium text-slate-100">{formatPrice(candidate.tradePlan.entry)}</span>
          </div>
          {candidate.tradePlan.stop && (
            <div>
              <span className="text-slate-500">Stop:</span>{" "}
              <span className="font-medium text-red-400">{formatPrice(candidate.tradePlan.stop)}</span>
            </div>
          )}
          {candidate.tradePlan.target1 && (
            <div>
              <span className="text-slate-500">Target:</span>{" "}
              <span className="font-medium text-emerald-400">{formatPrice(candidate.tradePlan.target1)}</span>
            </div>
          )}
        </div>
      </div>

      {/* Gate Reasons */}
      <div className="mb-3">
        <p className="mb-1 text-xs font-bold text-slate-400">Gate Decision Reasons</p>
        <ul className="space-y-1">
          {candidate.tradeGate.reasons.map((reason, idx) => (
            <li key={idx} className="text-xs text-slate-300">
              • {reason}
            </li>
          ))}
        </ul>
      </div>

      {/* Expand/Collapse */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="text-xs font-medium text-slate-400 hover:text-slate-200"
      >
        {expanded ? "Show Less ▲" : "Show More Details ▼"}
      </button>

      {/* Expanded Details */}
      {expanded && (
        <div className="mt-4 space-y-4 border-t border-slate-700 pt-4">
          {/* Backtest Details */}
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
                <span className="text-slate-500">Median Return:</span>{" "}
                <span className="text-slate-100">
                  {candidate.backtest.medianReturn ? formatPercent(candidate.backtest.medianReturn) : "N/A"}
                </span>
              </div>
              <div>
                <span className="text-slate-500">Benchmark Avg:</span>{" "}
                <span className="text-slate-100">
                  {candidate.backtest.benchmarkAvgReturn ? formatPercent(candidate.backtest.benchmarkAvgReturn) : "N/A"}
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

          {/* Simulation Results */}
          <div>
            <h4 className="mb-2 text-sm font-bold text-slate-300">Simulation Scenarios</h4>
            <div className="space-y-2">
              {Object.entries(candidate.simulation.scenarios).map(([name, scenario]) => (
                <div key={name} className="flex items-center justify-between text-xs">
                  <span className="capitalize text-slate-400">{name.replace("_", " ")}:</span>
                  <div className="flex items-center gap-4">
                    <span className="text-slate-100">
                      WR {formatNumber(scenario.winRate)}% | Avg {formatPercent(scenario.avgReturn)}
                    </span>
                    <span className={scenario.survives ? "text-emerald-400" : "text-red-400"}>
                      {scenario.survives ? "✓ Survives" : "✗ Fails"}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Confidence Details */}
          <div>
            <h4 className="mb-2 text-sm font-bold text-slate-300">Confidence Assessment</h4>
            <p className="text-xs text-slate-400">{candidate.confidence.reason}</p>
            <p className="mt-1 text-xs text-slate-500">
              Trustworthy: {candidate.confidence.trustworthy ? "Yes" : "No"} | Sample Size:{" "}
              {candidate.confidence.sampleSize}
            </p>
          </div>

          {/* Gate Conditions */}
          <div>
            <h4 className="mb-2 text-sm font-bold text-slate-300">Gate Conditions</h4>
            <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-3">
              {Object.entries(candidate.tradeGate.gateConditions).map(([key, pass]) => (
                <div key={key} className="flex items-center gap-2">
                  <span className={pass ? "text-emerald-400" : "text-red-400"}>{pass ? "✓" : "✗"}</span>
                  <span className="text-slate-400">{key.replace(/([A-Z])/g, " $1").trim()}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
