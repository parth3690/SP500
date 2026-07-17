"use client";

import { Fragment, useEffect, useMemo, useState, type ReactNode } from "react";
import clsx from "clsx";

import { fetchAgentBotRun, type AgentBotHistoryItem } from "@/lib/api";
import { formatMoney, formatPct } from "@/lib/format";
import type { AgentBotRunResponse, AgentBotRecommendation, AgentBotOutcome, AgentBotForwardEntry, AgentBotCatalyst } from "@/lib/types";

type RiskMode = "balanced" | "aggressive" | "defensive";
type RegimeMode = "auto" | "risk_on" | "neutral" | "risk_off";
type Mode = "sp500" | "watchlist";

type SavedRecommendation = {
  id: string;
  ticker: string;
  action: string;
  entryPrice: number;
  recommendedAt: string;
  closed: boolean;
  closedAt?: string;
  exitPrice?: number;
};

const AGENT_HISTORY_KEY = "agent-bot-history-v1";
const STOCK_LISTS_KEY = "alpha-stock-lists-v1";
const MAX_STOCK_LIST_SIZE = 100;
const FORWARD_LABELS = ["1W", "1M", "3M", "6M"];

function getErrorMessage(e: unknown): string {
  return e instanceof Error ? e.message : "Unknown error";
}

function pctClass(v: number) {
  return v > 0 ? "text-emerald-300" : v < 0 ? "text-rose-300" : "text-slate-300";
}

function actionClass(action: string) {
  if (action === "BUY") return "border-emerald-500/50 bg-emerald-500/10 text-emerald-300";
  if (action === "SELL") return "border-rose-500/50 bg-rose-500/10 text-rose-300";
  if (action === "WATCH") return "border-amber-500/50 bg-amber-500/10 text-amber-300";
  return "border-slate-700 bg-slate-900 text-slate-400";
}

function riskClass(level: string) {
  if (level === "Extreme" || level === "Elevated") return "text-rose-300";
  if (level === "Watch") return "text-amber-300";
  if (level === "Normal") return "text-emerald-300";
  return "text-slate-300";
}

function loadHistory(): SavedRecommendation[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(AGENT_HISTORY_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((item) => ({
        id: String(item.id || crypto.randomUUID()),
        ticker: String(item.ticker || "").trim().toUpperCase(),
        action: String(item.action || "").toUpperCase(),
        entryPrice: Number(item.entryPrice),
        recommendedAt: String(item.recommendedAt || ""),
        closed: Boolean(item.closed),
        closedAt: item.closedAt ? String(item.closedAt) : undefined,
        exitPrice: Number.isFinite(Number(item.exitPrice)) ? Number(item.exitPrice) : undefined,
      }))
      .filter((item) => item.ticker && item.recommendedAt && item.entryPrice > 0);
  } catch {
    return [];
  }
}

function saveHistory(items: SavedRecommendation[]) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(AGENT_HISTORY_KEY, JSON.stringify(items));
}

function loadStockLists(): { id: string; name: string; tickers: string[] }[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STOCK_LISTS_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((list) => ({
        id: String(list.id || crypto.randomUUID()),
        name: String(list.name || "Watchlist"),
        tickers: Array.isArray(list.tickers) ? list.tickers.map((t: string) => String(t).trim().toUpperCase()).filter(Boolean) : [],
      }))
      .filter((list) => list.tickers.length > 0);
  } catch {
    return [];
  }
}

function normalizeTickers(raw: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const item of raw.split(/[\s,;]+/)) {
    const t = item.trim().toUpperCase();
    if (!t || seen.has(t)) continue;
    seen.add(t);
    out.push(t);
    if (out.length >= MAX_STOCK_LIST_SIZE) break;
  }
  return out;
}

export default function AgentBot() {
  const [data, setData] = useState<AgentBotRunResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [mode, setMode] = useState<Mode>("sp500");
  const [riskMode, setRiskMode] = useState<RiskMode>("balanced");
  const [regime, setRegime] = useState<RegimeMode>("auto");
  const [topN, setTopN] = useState(10);
  const [minScore, setMinScore] = useState(55);
  const [tickerInput, setTickerInput] = useState("");
  const [watchlistTickers, setWatchlistTickers] = useState<string[]>([]);
  const [history, setHistory] = useState<SavedRecommendation[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    setHistory(loadHistory());
    const lists = loadStockLists();
    if (lists.length > 0) {
      setWatchlistTickers(lists[0].tickers);
      setTickerInput(lists[0].tickers.join(", "));
    }
  }, []);

  useEffect(() => {
    saveHistory(history);
  }, [history]);

  const tickersForRun = useMemo(() => {
    if (mode === "sp500") return [];
    return watchlistTickers;
  }, [mode, watchlistTickers]);

  const historyPayload: AgentBotHistoryItem[] = useMemo(
    () => history.map((h) => ({ ...h, closed: h.closed ?? false })),
    [history]
  );

  const runAgent = async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = await fetchAgentBotRun({
        mode,
        tickers: tickersForRun,
        riskMode,
        regime,
        topN,
        minScore,
        history: historyPayload,
      });
      setData(payload);
    } catch (e) {
      setError(getErrorMessage(e));
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  const applyWatchlistInput = () => {
    const tickers = normalizeTickers(tickerInput);
    setWatchlistTickers(tickers);
    if (mode !== "watchlist") setMode("watchlist");
  };

  const saveRecommendations = () => {
    if (!data) return;
    const incoming = data.recommendations
      .filter((r) => r.action === "BUY" || r.action === "SELL")
      .map((r) => ({
      id: crypto.randomUUID(),
      ticker: r.ticker,
      action: r.action,
      entryPrice: r.entry,
      recommendedAt: data.asOf,
      closed: false,
    }));
    setHistory((prev) => {
      const seen = new Set(prev.map((p) => `${p.ticker}|${p.action}|${p.recommendedAt}`));
      return [
        ...prev,
        ...incoming.filter((rec) => !seen.has(`${rec.ticker}|${rec.action}|${rec.recommendedAt}`)),
      ];
    });
  };

  const closeRecommendation = (id: string, exitPrice: number) => {
    setHistory((prev) => prev.map((p) => (
      p.id === id
        ? { ...p, closed: true, closedAt: new Date().toISOString(), exitPrice }
        : p
    )));
  };

  const removeRecommendation = (id: string) => {
    setHistory((prev) => prev.filter((p) => p.id !== id));
  };

  const marketConditions = data?.meta?.marketConditions;
  const adjustedFilters = data?.meta?.adjustedMinScore != null
    ? `score ≥ ${data.meta.adjustedMinScore}${data.meta.adjustedMaxBeta ? `, beta ≤ ${data.meta.adjustedMaxBeta}` : ""}`
    : null;

  return (
    <div className="mx-auto max-w-[1600px] px-4 py-8">
      <header className="mb-6 flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Agent Bot</h1>
          <p className="mt-1 text-sm text-slate-400">
            Autonomous recommendations and outcome tracking for your watchlist or the S&P 500.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-2 sm:flex sm:flex-wrap sm:items-end">
          <Control label="Mode">
            <select
              className="w-36 rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm"
              value={mode}
              onChange={(e) => setMode(e.target.value as Mode)}
            >
              <option value="sp500">S&P 500</option>
              <option value="watchlist">Watchlist</option>
            </select>
          </Control>
          <Control label="Risk">
            <select
              className="w-32 rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm"
              value={riskMode}
              onChange={(e) => setRiskMode(e.target.value as RiskMode)}
            >
              <option value="balanced">Balanced</option>
              <option value="aggressive">Aggressive</option>
              <option value="defensive">Defensive</option>
            </select>
          </Control>
          <Control label="Regime">
            <select
              className="w-32 rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm"
              value={regime}
              onChange={(e) => setRegime(e.target.value as RegimeMode)}
            >
              <option value="auto">Auto</option>
              <option value="risk_on">Risk on</option>
              <option value="neutral">Neutral</option>
              <option value="risk_off">Risk off</option>
            </select>
          </Control>
          <Control label="Top N">
            <input
              className="w-20 rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm"
              type="number"
              min={1}
              max={50}
              value={topN}
              onChange={(e) => setTopN(Number(e.target.value))}
            />
          </Control>
          <Control label="Min score">
            <input
              className="w-20 rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm"
              type="number"
              min={0}
              max={100}
              value={minScore}
              onChange={(e) => setMinScore(Number(e.target.value))}
            />
          </Control>
          <button
            type="button"
            disabled={loading || (mode === "watchlist" && watchlistTickers.length === 0)}
            className="rounded-md bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-500 disabled:opacity-50"
            onClick={() => void runAgent()}
          >
            {loading ? "Running..." : "Run Agent"}
          </button>
        </div>
      </header>

      {mode === "watchlist" ? (
        <section className="mb-5 rounded-xl border border-slate-800 bg-slate-900/35 p-4">
          <label className="block text-xs uppercase tracking-wide text-slate-500">
            Watchlist tickers
            <div className="mt-1 flex gap-2">
              <input
                className="min-w-0 flex-1 rounded-md border border-slate-700 bg-slate-950 px-3 py-2 font-mono text-sm uppercase"
                placeholder="AAPL, MSFT, NVDA"
                value={tickerInput}
                onChange={(e) => setTickerInput(e.target.value.toUpperCase())}
                onKeyDown={(e) => {
                  if (e.key === "Enter") applyWatchlistInput();
                }}
              />
              <button
                type="button"
                className="rounded-md border border-slate-700 px-3 py-2 text-sm hover:bg-slate-800"
                onClick={applyWatchlistInput}
              >
                Apply
              </button>
            </div>
          </label>
          <p className="mt-2 text-xs text-slate-500">{watchlistTickers.length} / {MAX_STOCK_LIST_SIZE} tickers</p>
        </section>
      ) : null}

      {error ? <p className="mb-4 text-sm text-rose-300">{error}</p> : null}

      {data ? (
        <>
          {String(data.meta.status ?? "ok") !== "ok" ? (
            <p className="mb-4 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-200">
              Data status: {String(data.meta.status).replaceAll("_", " ")}.
              {typeof data.meta.error === "string" ? ` ${data.meta.error}` : " Some inputs are unavailable."}
            </p>
          ) : null}
          <section className="mb-5 rounded-xl border border-slate-800 bg-slate-900/40 p-4">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
              <div>
                <h2 className="text-lg font-semibold">Daily Briefing</h2>
                <p className="mt-1 text-sm text-slate-300">{data.briefing.summary}</p>
              </div>
              <div className="grid grid-cols-4 gap-2 text-center">
                <BriefMetric label="BUY" value={data.briefing.counts.buy} />
                <BriefMetric label="SELL" value={data.briefing.counts.sell} />
                <BriefMetric label="WATCH" value={data.briefing.counts.watch} />
                <BriefMetric label="AVOID" value={data.briefing.counts.avoid} />
              </div>
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              <span className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-400">
                Regime: <b className="text-slate-200 capitalize">{data.briefing.regime}</b>
              </span>
              {marketConditions ? (
                <span className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-400">
                  Macro risk: <b className={riskClass(marketConditions.riskLevel)}>{marketConditions.riskLevel}</b>
                </span>
              ) : null}
              {adjustedFilters ? (
                <span className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-400">
                  Filters: <b className="text-slate-200">{adjustedFilters}</b>
                </span>
              ) : null}
              <span className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-400">
                As of: {new Date(data.asOf).toLocaleString()}
              </span>
              {typeof data.meta.priceCoveragePct === "number" ? (
                <span className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-400">
                  Price coverage: <b className="text-slate-200">{data.meta.priceCoveragePct}%</b>
                </span>
              ) : null}
              <button
                type="button"
                onClick={saveRecommendations}
                className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-300 hover:bg-slate-800"
              >
                Save recommendations to journal
              </button>
            </div>
          </section>

          {data.alerts.length > 0 ? (
            <section className="mb-5">
              <h2 className="mb-2 text-lg font-semibold">Alerts</h2>
              <div className="grid gap-2 md:grid-cols-2 lg:grid-cols-3">
                {data.alerts.map((alert) => (
                  <div
                    key={`${alert.ticker}-${alert.type}`}
                    className={clsx(
                      "rounded-md border px-3 py-2 text-sm",
                      alert.severity === "high"
                        ? "border-rose-500/40 bg-rose-500/10 text-rose-300"
                        : alert.severity === "medium"
                          ? "border-amber-500/40 bg-amber-500/10 text-amber-300"
                          : "border-slate-700 bg-slate-900 text-slate-400"
                    )}
                  >
                    <b>{alert.ticker}</b> — {alert.message}
                  </div>
                ))}
              </div>
            </section>
          ) : null}

          <section className="mb-5 overflow-hidden rounded-xl border border-slate-800 bg-slate-900/30">
            <div className="px-4 py-3">
              <h2 className="text-lg font-semibold">Top Recommendations</h2>
            </div>
            <div className="overflow-auto">
              <table className="w-full min-w-[1100px] text-sm">
                <thead className="border-b border-slate-800 bg-slate-950/50 text-xs uppercase tracking-wide text-slate-400">
                  <tr>
                    <th className="px-3 py-2 text-left">Rank</th>
                    <th className="px-3 py-2 text-left">Ticker</th>
                    <th className="px-3 py-2 text-left">Action</th>
                    <th className="px-3 py-2 text-right">Alpha</th>
                    <th className="px-3 py-2 text-right">Entry</th>
                    <th className="px-3 py-2 text-right">Stop</th>
                    <th className="px-3 py-2 text-right">Target 1</th>
                    <th className="px-3 py-2 text-right">Target 2</th>
                    <th className="px-3 py-2 text-right">R/R</th>
                    <th className="px-3 py-2 text-left">Why now</th>
                    <th className="px-3 py-2 text-center">Catalyst</th>
                  </tr>
                </thead>
                <tbody>
                  {data.recommendations.map((rec) => {
                    const isOpen = expanded === rec.ticker;
                    return (
                      <Fragment key={rec.ticker}>
                        <tr className="border-b border-slate-800/80">
                          <td className="px-3 py-2 text-slate-300">{rec.rank}</td>
                          <td className="px-3 py-2">
                            <div className="font-semibold text-slate-100">{rec.ticker}</div>
                            <div className="text-xs text-slate-500">{rec.companyName}</div>
                          </td>
                          <td className="px-3 py-2">
                            <span className={clsx("rounded border px-2 py-1 text-xs font-semibold", actionClass(rec.action))}>
                              {rec.action}
                            </span>
                          </td>
                          <td className={clsx("px-3 py-2 text-right font-semibold tabular-nums", rec.alphaScore >= 65 ? "text-emerald-300" : "text-slate-300")}>
                            {rec.alphaScore.toFixed(1)}
                          </td>
                          <td className="px-3 py-2 text-right tabular-nums text-slate-300">{formatMoney(rec.entry)}</td>
                          <td className="px-3 py-2 text-right tabular-nums text-rose-300">{rec.stop == null ? "N/A" : formatMoney(rec.stop)}</td>
                          <td className="px-3 py-2 text-right tabular-nums text-emerald-300">{rec.target1 == null ? "N/A" : formatMoney(rec.target1)}</td>
                          <td className="px-3 py-2 text-right tabular-nums text-emerald-300">{rec.target2 == null ? "N/A" : formatMoney(rec.target2)}</td>
                          <td className="px-3 py-2 text-right tabular-nums text-slate-300">{rec.riskReward == null ? "N/A" : `${rec.riskReward.toFixed(2)}x`}</td>
                          <td className="px-3 py-2 text-xs text-slate-400 max-w-[260px]">{rec.whyNow}</td>
                          <td className="px-3 py-2 text-center">
                            <button
                              type="button"
                              className="rounded border border-slate-700 px-2 py-0.5 text-xs text-slate-300 hover:bg-slate-800"
                              onClick={() => setExpanded(isOpen ? null : rec.ticker)}
                            >
                              {isOpen ? "Hide" : "View"}
                            </button>
                          </td>
                        </tr>
                        {isOpen ? (
                          <tr className="border-b border-slate-800 bg-slate-950/30">
                            <td colSpan={11} className="px-4 py-3">
                              <CatalystPanel ticker={rec.ticker} catalyst={rec.catalyst} />
                            </td>
                          </tr>
                        ) : null}
                      </Fragment>
                    );
                  })}
                  {data.recommendations.length === 0 ? (
                    <tr>
                      <td colSpan={11} className="px-4 py-6 text-sm text-slate-500">No recommendations match the current filters.</td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </div>
          </section>

          <section className="mb-5 overflow-hidden rounded-xl border border-slate-800 bg-slate-900/30">
            <div className="px-4 py-3">
              <h2 className="text-lg font-semibold">Active Tracking</h2>
            </div>
            <div className="overflow-auto">
              <table className="w-full min-w-[760px] text-sm">
                <thead className="border-b border-slate-800 bg-slate-950/50 text-xs uppercase tracking-wide text-slate-400">
                  <tr>
                    <th className="px-3 py-2 text-left">Ticker</th>
                    <th className="px-3 py-2 text-right">Entry</th>
                    <th className="px-3 py-2 text-right">Current</th>
                    <th className="px-3 py-2 text-right">Unrealized</th>
                    <th className="px-3 py-2 text-right">Stop</th>
                    <th className="px-3 py-2 text-right">Target 1</th>
                    <th className="px-3 py-2 text-right">Score</th>
                    <th className="px-3 py-2 text-left">Why now</th>
                  </tr>
                </thead>
                <tbody>
                  {data.activeTracking.map((t) => (
                    <tr key={t.id ?? `${t.ticker}-${t.entry}`} className="border-b border-slate-800/80">
                      <td className="px-3 py-2 font-semibold text-slate-100">{t.ticker}</td>
                      <td className="px-3 py-2 text-right tabular-nums text-slate-300">{formatMoney(t.entry)}</td>
                      <td className="px-3 py-2 text-right tabular-nums text-slate-300">{formatMoney(t.currentPrice)}</td>
                      <td className={clsx("px-3 py-2 text-right font-semibold tabular-nums", pctClass(t.unrealizedReturnPct ?? 0))}>
                        {t.unrealizedReturnPct == null ? "N/A" : formatPct(t.unrealizedReturnPct)}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-rose-300">{t.stop == null ? "N/A" : formatMoney(t.stop)}</td>
                      <td className="px-3 py-2 text-right tabular-nums text-emerald-300">{t.target1 == null ? "N/A" : formatMoney(t.target1)}</td>
                      <td className="px-3 py-2 text-right tabular-nums text-slate-300">{t.alphaScore.toFixed(1)}</td>
                      <td className="px-3 py-2 text-xs text-slate-400 max-w-[260px]">{t.whyNow}</td>
                    </tr>
                  ))}
                  {data.activeTracking.length === 0 ? (
                    <tr>
                      <td colSpan={8} className="px-4 py-6 text-sm text-slate-500">No open BUY or SELL recommendations.</td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </div>
          </section>

          <ForwardJournal journal={data.forwardJournal} />

          <div className="mt-5">
            <OutcomeHistory outcomes={data.outcomes} history={history} onClose={closeRecommendation} onRemove={removeRecommendation} />
          </div>
        </>
      ) : (
        <section className="rounded-xl border border-slate-800 bg-slate-900/30 p-8 text-center text-slate-400">
          {loading ? "Running agent..." : "Click \"Run Agent\" to generate today\'s briefing and recommendations."}
        </section>
      )}
    </div>
  );
}

function CatalystPanel({ ticker, catalyst }: { ticker: string; catalyst: AgentBotCatalyst }) {
  if (!catalyst.available) {
    return <div className="text-xs text-slate-500">No catalyst data available for {ticker}.</div>;
  }
  return (
    <div className="grid gap-2 text-xs text-slate-400 sm:grid-cols-2 lg:grid-cols-4">
      <div>Earnings date: <b className="text-slate-200">{catalyst.earningsDate ?? "N/A"}</b></div>
      <div>Ex-dividend: <b className="text-slate-200">{catalyst.exDividendDate ?? "N/A"}</b></div>
      <div>Analyst target: <b className="text-slate-200">{catalyst.targetMeanPrice == null ? "N/A" : formatMoney(catalyst.targetMeanPrice)}</b></div>
      <div>Consensus: <b className="text-slate-200">{catalyst.analystRecommendation ?? "N/A"}</b></div>
      <div>Analysts: <b className="text-slate-200">{catalyst.analystCount ?? "N/A"}</b></div>
      <div>Revenue growth: <b className="text-slate-200">{catalyst.revenueGrowth == null ? "N/A" : formatPct(catalyst.revenueGrowth * 100)}</b></div>
      <div>Earnings growth: <b className="text-slate-200">{catalyst.earningsGrowth == null ? "N/A" : formatPct(catalyst.earningsGrowth * 100)}</b></div>
      <div>EPS growth: <b className="text-slate-200">{catalyst.epsGrowth == null ? "N/A" : formatPct(catalyst.epsGrowth * 100)}</b></div>
      <div className="sm:col-span-2 lg:col-span-4">
        Revision notes: <b className="text-slate-200">{catalyst.revisionNotes.join("; ") || "N/A"}</b>
      </div>
    </div>
  );
}

function ForwardJournal({ journal }: { journal: AgentBotRunResponse["forwardJournal"] }) {
  const aggregates = journal.aggregates;
  return (
    <section className="mb-5 overflow-hidden rounded-xl border border-slate-800 bg-slate-900/30">
      <div className="flex items-center justify-between px-4 py-3">
        <h2 className="text-lg font-semibold">Forward Performance Journal</h2>
        <div className="flex gap-2">
          {FORWARD_LABELS.map((label) => (
            <span key={label} className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-400">
              {label} avg: <b className={pctClass(aggregates[label]?.avgReturn ?? 0)}>
                {aggregates[label]?.avgReturn == null ? "N/A" : formatPct(aggregates[label].avgReturn)}
              </b>
            </span>
          ))}
        </div>
      </div>
      <div className="overflow-auto">
        <table className="w-full min-w-[760px] text-sm">
          <thead className="border-b border-slate-800 bg-slate-950/50 text-xs uppercase tracking-wide text-slate-400">
            <tr>
              <th className="px-3 py-2 text-left">Ticker</th>
              <th className="px-3 py-2 text-left">Action</th>
              <th className="px-3 py-2 text-right">Entry</th>
              <th className="px-3 py-2 text-left">Recommended</th>
              <th className="px-3 py-2 text-right">1W</th>
              <th className="px-3 py-2 text-right">1M</th>
              <th className="px-3 py-2 text-right">3M</th>
              <th className="px-3 py-2 text-right">6M</th>
              <th className="px-3 py-2 text-left">Status</th>
            </tr>
          </thead>
          <tbody>
            {journal.entries.map((e) => (
              <ForwardRow key={e.id ?? `${e.ticker}-${e.recommendedAt}`} entry={e} />
            ))}
            {journal.entries.length === 0 ? (
              <tr>
                <td colSpan={9} className="px-4 py-6 text-sm text-slate-500">No saved recommendations yet. Save recommendations to compute forward returns.</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ForwardRow({ entry }: { entry: AgentBotForwardEntry }) {
  return (
    <tr className="border-b border-slate-800/80">
      <td className="px-3 py-2 font-semibold text-slate-100">{entry.ticker}</td>
      <td className="px-3 py-2">
        <span className={clsx("rounded border px-2 py-0.5 text-xs font-semibold", actionClass(entry.action))}>
          {entry.action}
        </span>
      </td>
      <td className="px-3 py-2 text-right tabular-nums text-slate-300">{formatMoney(entry.entryPrice)}</td>
      <td className="px-3 py-2 text-xs text-slate-500">{new Date(entry.recommendedAt).toLocaleDateString()}</td>
      {FORWARD_LABELS.map((label) => (
        <td key={label} className={clsx("px-3 py-2 text-right tabular-nums", pctClass(entry.forwardReturns[label] ?? 0))}>
          {entry.forwardReturns[label] == null ? "N/A" : formatPct(entry.forwardReturns[label])}
        </td>
      ))}
      <td className="px-3 py-2 text-xs capitalize text-slate-400">{entry.closed ? "closed" : "open"}</td>
    </tr>
  );
}

function BriefMetric({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded border border-slate-800 bg-slate-950 px-3 py-2">
      <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
      <div className="text-lg font-semibold text-slate-100">{value}</div>
    </div>
  );
}

function OutcomeHistory({
  outcomes,
  history,
  onClose,
  onRemove,
}: {
  outcomes: AgentBotOutcome[];
  history: SavedRecommendation[];
  onClose: (id: string, exitPrice: number) => void;
  onRemove: (id: string) => void;
}) {
  const display = outcomes.length > 0 ? outcomes : history.map((h) => ({
    id: h.id,
    ticker: h.ticker,
    action: h.action,
    entryPrice: h.entryPrice,
    currentPrice: h.entryPrice,
    returnPct: 0,
    recommendedAt: h.recommendedAt,
    status: h.closed ? "closed" as const : "open" as const,
  }));

  const avgReturn = display.length > 0 ? display.reduce((sum, o) => sum + o.returnPct, 0) / display.length : 0;

  return (
    <section className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900/30">
      <div className="flex items-center justify-between px-4 py-3">
        <h2 className="text-lg font-semibold">Outcome Journal</h2>
        {display.length > 0 ? (
          <span className={clsx("text-sm font-semibold", pctClass(avgReturn))}>
            Avg return: {formatPct(avgReturn)}
          </span>
        ) : null}
      </div>
      <div className="overflow-auto">
        <table className="w-full min-w-[720px] text-sm">
          <thead className="border-b border-slate-800 bg-slate-950/50 text-xs uppercase tracking-wide text-slate-400">
            <tr>
              <th className="px-3 py-2 text-left">Ticker</th>
              <th className="px-3 py-2 text-left">Action</th>
              <th className="px-3 py-2 text-right">Entry</th>
              <th className="px-3 py-2 text-right">Current</th>
              <th className="px-3 py-2 text-right">Return</th>
              <th className="px-3 py-2 text-left">Recommended</th>
              <th className="px-3 py-2 text-left">Status</th>
              <th className="px-3 py-2 text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {display.map((o) => (
              <tr key={o.id ?? `${o.ticker}-${o.recommendedAt}`} className="border-b border-slate-800/80">
                <td className="px-3 py-2 font-semibold text-slate-100">{o.ticker}</td>
                <td className="px-3 py-2">
                  <span className={clsx("rounded border px-2 py-0.5 text-xs font-semibold", actionClass(o.action))}>
                    {o.action}
                  </span>
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-slate-300">{formatMoney(o.entryPrice)}</td>
                <td className="px-3 py-2 text-right tabular-nums text-slate-300">{formatMoney(o.currentPrice)}</td>
                <td className={clsx("px-3 py-2 text-right font-semibold tabular-nums", pctClass(o.returnPct))}>
                  {formatPct(o.returnPct)}
                </td>
                <td className="px-3 py-2 text-xs text-slate-500">{o.recommendedAt ? new Date(o.recommendedAt).toLocaleDateString() : "N/A"}</td>
                <td className="px-3 py-2 text-xs capitalize text-slate-400">{o.status}</td>
                <td className="px-3 py-2 text-right">
                  {o.status !== "closed" ? (
                    <button
                      type="button"
                      className="mr-2 rounded border border-slate-700 px-2 py-0.5 text-xs text-slate-300 hover:bg-slate-800"
                      onClick={() => onClose(o.id ?? `${o.ticker}-${o.recommendedAt}`, o.currentPrice)}
                    >
                      Close
                    </button>
                  ) : null}
                  <button
                    type="button"
                    className="rounded border border-rose-500/50 px-2 py-0.5 text-xs text-rose-300 hover:bg-rose-500/10"
                    onClick={() => onRemove(o.id ?? `${o.ticker}-${o.recommendedAt}`)}
                  >
                    Remove
                  </button>
                </td>
              </tr>
            ))}
            {display.length === 0 ? (
              <tr>
                <td colSpan={8} className="px-4 py-6 text-sm text-slate-500">No saved recommendations yet. Run the agent and save recommendations to start tracking outcomes.</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Control(props: { label: string; children: ReactNode }) {
  return (
    <label className="text-xs uppercase tracking-wide text-slate-500">
      {props.label}
      <div className="mt-1">{props.children}</div>
    </label>
  );
}
