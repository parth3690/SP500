"use client";

import { Fragment, useEffect, useMemo, useState, type ReactNode } from "react";
import Link from "next/link";
import clsx from "clsx";

import { fetchAlphaCandidates, fetchAlphaWatchlist } from "@/lib/api";
import { ASK_ALPHA_EXAMPLES, analyzeAlphaQuestion } from "@/lib/alphaAsk";
import { formatMoney, formatPct } from "@/lib/format";
import type { AlphaCandidateRow, AlphaCandidatesResponse } from "@/lib/types";

type RiskMode = "balanced" | "aggressive" | "defensive";
type RegimeMode = "auto" | "risk_on" | "neutral" | "risk_off";
type UniverseMode = "sp500" | "watchlist";
type OptionFilter = "all" | "premium_30d" | "directional" | "no_option";
type InstitutionalFilter = "all" | "ownership_20" | "adding_10" | "ownership_20_adding_10" | "missing";

type WatchItem = {
  ticker: string;
  companyName: string;
  sector: string;
  entryPrice: number;
  addedAt: string;
  entryScore: number;
  thesis: string;
  invalidation: string;
};

type StockWatchlist = {
  id: string;
  name: string;
  tickers: string[];
};

type AskMessage = {
  role: "user" | "assistant";
  content: string;
  createdAt: string;
};

const WATCHLIST_KEY = "alpha-watchlist-v1";
const STOCK_LISTS_KEY = "alpha-stock-lists-v1";
const MAX_STOCK_LIST_SIZE = 100;
const INST_OWNERSHIP_MIN = 20;
const INST_TRANSACTION_MIN = 10;
const SECTOR_OPTIONS = [
  "Communication Services",
  "Consumer Discretionary",
  "Consumer Staples",
  "Energy",
  "Financials",
  "Health Care",
  "Industrials",
  "Information Technology",
  "Materials",
  "Real Estate",
  "Utilities",
];

function getErrorMessage(e: unknown): string {
  return e instanceof Error ? e.message : "Unknown error";
}

function pctClass(v: number) {
  return v > 0 ? "text-emerald-300" : v < 0 ? "text-rose-300" : "text-slate-300";
}

function scoreClass(v: number) {
  if (v >= 75) return "text-emerald-300";
  if (v >= 65) return "text-sky-300";
  if (v >= 55) return "text-amber-300";
  return "text-slate-300";
}

function stateClass(state: string) {
  if (state === "bullish") return "border-emerald-500/40 bg-emerald-500/10 text-emerald-300";
  if (state === "bearish") return "border-rose-500/40 bg-rose-500/10 text-rose-300";
  return "border-slate-700 bg-slate-900 text-slate-400";
}

function actionClass(action: string) {
  if (action === "BUY") return "border-emerald-500/50 bg-emerald-500/10 text-emerald-300";
  if (action === "SELL") return "border-rose-500/50 bg-rose-500/10 text-rose-300";
  if (action === "WATCH") return "border-amber-500/50 bg-amber-500/10 text-amber-300";
  return "border-slate-700 bg-slate-900 text-slate-400";
}

function institutionalPass(row: AlphaCandidateRow) {
  return (
    row.institutionalOwnershipPct != null &&
    row.institutionalOwnershipPct >= INST_OWNERSHIP_MIN &&
    row.institutionalTransactionPct != null &&
    row.institutionalTransactionPct >= INST_TRANSACTION_MIN
  );
}

function institutionalFilterLabel(filter: InstitutionalFilter) {
  if (filter === "ownership_20") return "Ownership 20%+";
  if (filter === "adding_10") return "Adding +10%";
  if (filter === "ownership_20_adding_10") return "Own 20% + adding";
  if (filter === "missing") return "Missing inst data";
  return "All institutional";
}

function pctValue(v: number | null) {
  return v == null ? "N/A" : `${v.toFixed(1)}%`;
}

function signedPctValue(v: number | null) {
  return v == null ? "N/A" : formatPct(v, 1);
}

function loadWatchlist(): WatchItem[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(WATCHLIST_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function loadStockLists(): StockWatchlist[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STOCK_LISTS_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(parsed)) return [];
    const cleaned = parsed
      .map((list) => ({
        id: String(list.id || crypto.randomUUID()),
        name: String(list.name || "Watchlist"),
        tickers: normalizeTickerList(Array.isArray(list.tickers) ? list.tickers : []),
      }))
      .filter((list) => list.tickers.length > 0);
    return cleaned.length ? cleaned : defaultStockLists();
  } catch {
    return defaultStockLists();
  }
}

function saveStockLists(items: StockWatchlist[]) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(STOCK_LISTS_KEY, JSON.stringify(items));
}

function defaultStockLists(): StockWatchlist[] {
  return [
    {
      id: "core-growth",
      name: "Core Growth",
      tickers: ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA"],
    },
  ];
}

function normalizeTickerList(raw: string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const item of raw) {
    const ticker = item.trim().toUpperCase().replace(/\s+/g, "");
    if (!ticker || seen.has(ticker)) continue;
    seen.add(ticker);
    out.push(ticker);
    if (out.length >= MAX_STOCK_LIST_SIZE) break;
  }
  return out;
}

function parseTickerInput(raw: string): string[] {
  return normalizeTickerList(raw.split(/[\s,;]+/));
}

function saveWatchlist(items: WatchItem[]) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(WATCHLIST_KEY, JSON.stringify(items));
}

function defaultThesis(row: AlphaCandidateRow): string {
  const positives = row.signals
    .filter((s) => s.state === "bullish")
    .slice(0, 3)
    .map((s) => s.label.toLowerCase());
  return positives.length
    ? `${row.ticker}: ${positives.join(", ")} support the setup.`
    : `${row.ticker}: monitor alpha score and signal mix.`;
}

function defaultInvalidation(row: AlphaCandidateRow): string {
  return `Review if alpha score drops below 55 or relative strength vs SPY turns negative.`;
}

export default function AlphaCandidates() {
  const [data, setData] = useState<AlphaCandidatesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [limit, setLimit] = useState(50);
  const [minScore, setMinScore] = useState(55);
  const [maxBeta, setMaxBeta] = useState("");
  const [sector, setSector] = useState("");
  const [riskMode, setRiskMode] = useState<RiskMode>("balanced");
  const [regime, setRegime] = useState<RegimeMode>("auto");
  const [optionFilter, setOptionFilter] = useState<OptionFilter>("all");
  const [institutionalFilter, setInstitutionalFilter] = useState<InstitutionalFilter>("all");
  const [expanded, setExpanded] = useState<string | null>(null);
  const [watchlist, setWatchlist] = useState<WatchItem[]>([]);
  const [universeMode, setUniverseMode] = useState<UniverseMode>("sp500");
  const [stockLists, setStockLists] = useState<StockWatchlist[]>([]);
  const [selectedListId, setSelectedListId] = useState("");
  const [tickerInput, setTickerInput] = useState("");
  const [newListName, setNewListName] = useState("");
  const [askInput, setAskInput] = useState("");
  const [askMessages, setAskMessages] = useState<AskMessage[]>([]);

  useEffect(() => {
    setWatchlist(loadWatchlist());
    const lists = loadStockLists();
    setStockLists(lists);
    setSelectedListId(lists[0]?.id ?? "");
  }, []);

  useEffect(() => {
    saveWatchlist(watchlist);
  }, [watchlist]);

  useEffect(() => {
    if (stockLists.length > 0) saveStockLists(stockLists);
  }, [stockLists]);

  const selectedList = useMemo(
    () => stockLists.find((list) => list.id === selectedListId) ?? stockLists[0],
    [selectedListId, stockLists],
  );

  const runFetch = async (opts?: { refresh?: boolean }) => {
    setLoading(true);
    setError(null);
    try {
      const requestedLimit = universeMode === "watchlist" && selectedList
        ? Math.min(limit, selectedList.tickers.length)
        : Math.max(5, limit);
      const common = {
        limit: requestedLimit,
        minScore: universeMode === "watchlist" ? 0 : minScore,
        maxBeta: maxBeta.trim() ? Number(maxBeta) : undefined,
        riskMode,
        regime,
        enrichTop: institutionalFilter === "all" ? Math.min(30, requestedLimit) : Math.min(50, requestedLimit),
        refresh: opts?.refresh,
      };
      const payload = universeMode === "watchlist"
        ? await fetchAlphaWatchlist({
          ...common,
          tickers: selectedList?.tickers ?? [],
        })
        : await fetchAlphaCandidates({
          ...common,
          sector: sector || undefined,
        });
      setData(payload);
    } catch (e) {
      setError(getErrorMessage(e));
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void runFetch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const allRows = data?.candidates ?? [];
  const rows = useMemo(() => {
    return allRows.filter((row) => {
      const category = row.tradePlan.optionCategory;
      if (optionFilter === "premium_30d") {
        if (!(category === "Premium 30D" && row.tradePlan.optionIvGate === "pass")) return false;
      } else if (optionFilter === "directional") {
        if (category !== "Directional") return false;
      } else if (optionFilter === "no_option") {
        if (row.tradePlan.optionStrategy) return false;
      }

      const ownershipPass = row.institutionalOwnershipPct != null && row.institutionalOwnershipPct >= INST_OWNERSHIP_MIN;
      const transactionPass = row.institutionalTransactionPct != null && row.institutionalTransactionPct >= INST_TRANSACTION_MIN;
      if (institutionalFilter === "ownership_20") return ownershipPass;
      if (institutionalFilter === "adding_10") return transactionPass;
      if (institutionalFilter === "ownership_20_adding_10") return ownershipPass && transactionPass;
      if (institutionalFilter === "missing") return row.institutionalOwnershipPct == null || row.institutionalTransactionPct == null;
      return true;
    });
  }, [allRows, optionFilter, institutionalFilter]);
  const rowByTicker = useMemo(
    () => new Map(allRows.map((r) => [r.ticker, r])),
    [allRows],
  );
  const watched = useMemo(() => new Set(watchlist.map((w) => w.ticker)), [watchlist]);
  const institutionalStats = useMemo(() => {
    const ownershipCoverage = allRows.filter((row) => row.institutionalOwnershipPct != null).length;
    const transactionCoverage = allRows.filter((row) => row.institutionalTransactionPct != null).length;
    const passCount = allRows.filter(institutionalPass).length;
    return { ownershipCoverage, transactionCoverage, passCount };
  }, [allRows]);

  const askAlpha = (prompt?: string) => {
    const question = (prompt ?? askInput).trim();
    if (!question) return;
    const answer = analyzeAlphaQuestion(question, {
      data,
      visibleRows: rows,
      watchlist,
      filters: {
        option: optionFilter,
        institutional: institutionalFilter,
        universe: universeMode,
      },
    });
    const now = new Date().toISOString();
    const nextMessages: AskMessage[] = [
      { role: "user", content: question, createdAt: now },
      { role: "assistant", content: answer, createdAt: now },
    ];
    setAskMessages((prev) => [...prev, ...nextMessages].slice(-10));
    setAskInput("");
  };

  const addWatch = (row: AlphaCandidateRow) => {
    if (watched.has(row.ticker)) return;
    setWatchlist((prev) => [
      {
        ticker: row.ticker,
        companyName: row.companyName,
        sector: row.sector,
        entryPrice: row.currentPrice,
        addedAt: new Date().toISOString(),
        entryScore: row.alphaScore,
        thesis: defaultThesis(row),
        invalidation: defaultInvalidation(row),
      },
      ...prev,
    ]);
  };

  const updateWatch = (ticker: string, patch: Partial<WatchItem>) => {
    setWatchlist((prev) => prev.map((w) => (w.ticker === ticker ? { ...w, ...patch } : w)));
  };

  const removeWatch = (ticker: string) => {
    setWatchlist((prev) => prev.filter((w) => w.ticker !== ticker));
  };

  const createStockList = () => {
    const name = newListName.trim() || `Watchlist ${stockLists.length + 1}`;
    const id = `${Date.now()}-${name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
    const next = { id, name, tickers: [] };
    setStockLists((prev) => [...prev, next]);
    setSelectedListId(id);
    setUniverseMode("watchlist");
    setNewListName("");
  };

  const renameStockList = (id: string, name: string) => {
    setStockLists((prev) => prev.map((list) => (list.id === id ? { ...list, name } : list)));
  };

  const deleteStockList = (id: string) => {
    setStockLists((prev) => {
      const next = prev.filter((list) => list.id !== id);
      if (selectedListId === id) setSelectedListId(next[0]?.id ?? "");
      return next;
    });
  };

  const addTickersToSelectedList = () => {
    if (!selectedList) return;
    const incoming = parseTickerInput(tickerInput);
    if (incoming.length === 0) return;
    setStockLists((prev) =>
      prev.map((list) => {
        if (list.id !== selectedList.id) return list;
        return {
          ...list,
          tickers: normalizeTickerList([...list.tickers, ...incoming]),
        };
      }),
    );
    setTickerInput("");
    setUniverseMode("watchlist");
  };

  const removeTickerFromSelectedList = (ticker: string) => {
    if (!selectedList) return;
    setStockLists((prev) =>
      prev.map((list) =>
        list.id === selectedList.id
          ? { ...list, tickers: list.tickers.filter((t) => t !== ticker) }
          : list,
      ),
    );
  };

  return (
    <div className="mx-auto max-w-[1600px] px-4 py-8">
      <header className="mb-5 flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Alpha Candidates</h1>
          <p className="mt-1 text-sm text-slate-400">
            {data?.asOf ? `As of ${new Date(data.asOf).toLocaleString()}` : "Ranked alpha candidates"}
          </p>
        </div>
        <div className="grid grid-cols-2 gap-2 sm:flex sm:flex-wrap sm:items-end">
          <Control label="Min score">
            <input
              className="w-24 rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm"
              type="number"
              min={0}
              max={100}
              value={minScore}
              onChange={(e) => setMinScore(Number(e.target.value))}
            />
          </Control>
          <Control label="Limit">
            <input
              className="w-20 rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm"
              type="number"
              min={5}
              max={150}
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
            />
          </Control>
          <Control label="Max beta">
            <input
              className="w-24 rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm"
              placeholder="Any"
              value={maxBeta}
              onChange={(e) => setMaxBeta(e.target.value)}
            />
          </Control>
          <Control label="Sector">
            <select
              className="w-44 rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm"
              value={sector}
              disabled={universeMode === "watchlist"}
              onChange={(e) => setSector(e.target.value)}
            >
              <option value="">All sectors</option>
              {SECTOR_OPTIONS.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
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
          <Control label="Options">
            <select
              className="w-44 rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm"
              value={optionFilter}
              onChange={(e) => setOptionFilter(e.target.value as OptionFilter)}
            >
              <option value="all">All setups</option>
              <option value="premium_30d">Premium 30D / IV 50+</option>
              <option value="directional">Directional only</option>
              <option value="no_option">No option idea</option>
            </select>
          </Control>
          <Control label="Institutional">
            <select
              className="w-52 rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm"
              value={institutionalFilter}
              onChange={(e) => setInstitutionalFilter(e.target.value as InstitutionalFilter)}
            >
              <option value="all">All institutional</option>
              <option value="ownership_20">Ownership &gt;20%</option>
              <option value="adding_10">Adding &gt;+10%</option>
              <option value="ownership_20_adding_10">Own 20% + adding 10%</option>
              <option value="missing">Missing inst data</option>
            </select>
          </Control>
          <button
            type="button"
            disabled={loading || (universeMode === "watchlist" && (!selectedList || selectedList.tickers.length === 0))}
            className="rounded-md bg-slate-100 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-white disabled:opacity-50"
            onClick={() => void runFetch()}
          >
            {loading ? "Scanning..." : "Scan"}
          </button>
          <button
            type="button"
            disabled={loading || (universeMode === "watchlist" && (!selectedList || selectedList.tickers.length === 0))}
            className="rounded-md border border-slate-700 px-3 py-2 text-sm hover:bg-slate-800 disabled:opacity-50"
            onClick={() => void runFetch({ refresh: true })}
          >
            Refresh
          </button>
        </div>
      </header>

      <section className="mb-5 rounded-xl border border-slate-800 bg-slate-900/35 p-4">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
          <div className="min-w-0 flex-1">
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <button
                type="button"
                className={clsx(
                  "rounded-md px-3 py-1.5 text-sm font-medium",
                  universeMode === "sp500" ? "bg-slate-100 text-slate-950" : "border border-slate-700 text-slate-300 hover:bg-slate-800",
                )}
                onClick={() => setUniverseMode("sp500")}
              >
                S&P 500
              </button>
              <button
                type="button"
                className={clsx(
                  "rounded-md px-3 py-1.5 text-sm font-medium",
                  universeMode === "watchlist" ? "bg-slate-100 text-slate-950" : "border border-slate-700 text-slate-300 hover:bg-slate-800",
                )}
                onClick={() => setUniverseMode("watchlist")}
              >
                Custom watchlist
              </button>
              {universeMode === "watchlist" && selectedList ? (
                <span className="text-xs text-slate-500">
                  {selectedList.tickers.length} / {MAX_STOCK_LIST_SIZE} stocks / scans all scores
                </span>
              ) : null}
            </div>

            {universeMode === "watchlist" ? (
              <div className="grid gap-3 lg:grid-cols-[260px_1fr]">
                <div className="space-y-2">
                  <label className="block text-xs uppercase tracking-wide text-slate-500">
                    Watchlist
                    <select
                      className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-2 text-sm"
                      value={selectedList?.id ?? ""}
                      onChange={(e) => setSelectedListId(e.target.value)}
                    >
                      {stockLists.map((list) => (
                        <option key={list.id} value={list.id}>
                          {list.name} ({list.tickers.length})
                        </option>
                      ))}
                    </select>
                  </label>
                  <div className="flex gap-2">
                    <input
                      className="min-w-0 flex-1 rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm"
                      placeholder="New list name"
                      value={newListName}
                      onChange={(e) => setNewListName(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") createStockList();
                      }}
                    />
                    <button
                      type="button"
                      className="rounded-md border border-slate-700 px-3 py-1.5 text-sm hover:bg-slate-800"
                      onClick={createStockList}
                    >
                      Create
                    </button>
                  </div>
                  {selectedList ? (
                    <div className="flex gap-2">
                      <input
                        className="min-w-0 flex-1 rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm"
                        value={selectedList.name}
                        onChange={(e) => renameStockList(selectedList.id, e.target.value)}
                      />
                      <button
                        type="button"
                        disabled={stockLists.length <= 1}
                        className="rounded-md border border-rose-500/50 px-3 py-1.5 text-sm text-rose-300 hover:bg-rose-500/10 disabled:opacity-40"
                        onClick={() => deleteStockList(selectedList.id)}
                      >
                        Delete
                      </button>
                    </div>
                  ) : null}
                </div>

                <div>
                  <label className="block text-xs uppercase tracking-wide text-slate-500">
                    Add tickers
                    <div className="mt-1 flex gap-2">
                      <input
                        className="min-w-0 flex-1 rounded-md border border-slate-700 bg-slate-950 px-3 py-2 font-mono text-sm uppercase"
                        placeholder="AAPL, MSFT, NVDA"
                        value={tickerInput}
                        onChange={(e) => setTickerInput(e.target.value.toUpperCase())}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") addTickersToSelectedList();
                        }}
                      />
                      <button
                        type="button"
                        disabled={!selectedList || selectedList.tickers.length >= MAX_STOCK_LIST_SIZE}
                        className="rounded-md bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-500 disabled:opacity-40"
                        onClick={addTickersToSelectedList}
                      >
                        Add
                      </button>
                    </div>
                  </label>
                  <div className="mt-3 flex max-h-24 flex-wrap gap-2 overflow-auto rounded-md border border-slate-800 bg-slate-950/50 p-2">
                    {selectedList?.tickers.map((ticker) => (
                      <span key={ticker} className="inline-flex items-center gap-1 rounded border border-slate-700 bg-slate-900 px-2 py-1 font-mono text-xs text-slate-300">
                        {ticker}
                        <button
                          type="button"
                          className="text-slate-500 hover:text-rose-300"
                          onClick={() => removeTickerFromSelectedList(ticker)}
                        >
                          x
                        </button>
                      </span>
                    ))}
                    {!selectedList || selectedList.tickers.length === 0 ? (
                      <span className="px-1 py-1 text-xs text-slate-500">Add up to 100 tickers to scan this list.</span>
                    ) : null}
                  </div>
                  <div className="mt-3">
                    <button
                      type="button"
                      disabled={loading || !selectedList || selectedList.tickers.length === 0}
                      className="rounded-md bg-slate-100 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-white disabled:opacity-50"
                      onClick={() => void runFetch()}
                    >
                      {loading ? "Loading..." : "Load watchlist in Alpha table"}
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-sm text-slate-400">
                Scanning the full S&P 500 universe. Switch to Custom watchlist to rank only your saved ticker lists.
              </p>
            )}
          </div>
        </div>
      </section>

      {data ? (
        <div className="mb-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
          <Metric label="Regime" value={String(data.marketRegime.effectiveState).replace("_", " ")} detail={data.marketRegime.spyTrend} />
          <Metric label="SPY drawdown" value={data.marketRegime.spyDrawdownPct == null ? "N/A" : formatPct(data.marketRegime.spyDrawdownPct)} />
          <Metric label="Candidates" value={`${data.meta.returned} / ${data.meta.computed}`} detail={`${data.meta.total} total`} />
          <Metric
            label="Option search"
            value={optionFilter === "premium_30d" ? `${rows.length} premium 30D` : optionFilter.replace("_", " ")}
            detail={optionFilter === "all" ? "All option categories" : `${rows.length} of ${allRows.length} visible`}
          />
          <Metric
            label="Institutional"
            value={institutionalFilter === "ownership_20_adding_10" ? `${institutionalStats.passCount} pass` : institutionalFilterLabel(institutionalFilter)}
            detail={`${institutionalStats.transactionCoverage}/${allRows.length} 13F change · ${institutionalStats.passCount} pass`}
          />
          <Metric
            label="Price coverage"
            value={data.meta.coveragePct == null ? "N/A" : `${data.meta.coveragePct}%`}
            detail={data.meta.status ?? data.marketRegime.riskMode}
          />
        </div>
      ) : null}

      {error ? <p className="mb-4 text-sm text-rose-300">{error}</p> : null}
      {data?.meta.warnings?.length ? (
        <p className="mb-4 text-xs text-amber-300">{data.meta.warnings.join(" ")}</p>
      ) : null}

      <section className="mb-5 rounded-xl border border-cyan-700/40 bg-cyan-950/15 p-4">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0 flex-1">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <div>
                <h2 className="text-lg font-semibold text-slate-100">Ask Alpha</h2>
                <p className="mt-0.5 text-xs text-slate-500">
                  {data ? `${rows.length} visible / ${allRows.length} loaded` : "Load a scan first"}
                </p>
              </div>
              <span className="rounded border border-cyan-500/30 bg-cyan-500/10 px-2 py-1 text-xs text-cyan-200">
                Local analyst
              </span>
            </div>
            <form
              className="flex flex-col gap-2 sm:flex-row"
              onSubmit={(event) => {
                event.preventDefault();
                askAlpha();
              }}
            >
              <input
                className="min-w-0 flex-1 rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-cyan-500/60 focus:ring-1 focus:ring-cyan-500/20"
                value={askInput}
                placeholder="Ask about candidates, options, institutions, risk, backtests, targets..."
                onChange={(event) => setAskInput(event.target.value)}
              />
              <button
                type="submit"
                disabled={!askInput.trim()}
                className="rounded-md bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-cyan-300 disabled:opacity-40"
              >
                Ask
              </button>
            </form>
            <div className="mt-3 flex flex-wrap gap-2">
              {ASK_ALPHA_EXAMPLES.map((example) => (
                <button
                  key={example}
                  type="button"
                  className="rounded-md border border-slate-700 bg-slate-950/60 px-2.5 py-1.5 text-xs text-slate-300 hover:border-cyan-500/50 hover:text-cyan-200"
                  onClick={() => askAlpha(example)}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
          <div className="max-h-72 min-h-36 overflow-auto rounded-md border border-slate-800 bg-slate-950/50 p-3 lg:w-[520px]">
            {askMessages.length ? (
              <div className="space-y-3">
                {askMessages.map((message, index) => (
                  <div key={`${message.createdAt}-${index}`} className={message.role === "user" ? "text-right" : "text-left"}>
                    <div
                      className={clsx(
                        "inline-block max-w-full rounded-md px-3 py-2 text-sm leading-relaxed",
                        message.role === "user"
                          ? "bg-slate-100 text-slate-950"
                          : "border border-slate-800 bg-slate-900 text-slate-200",
                      )}
                    >
                      <div className="whitespace-pre-wrap text-left">{message.content}</div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="grid h-full place-items-center text-center text-sm text-slate-500">
                Ask a ticker, setup, or ranking question.
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900/30">
        <div className="overflow-auto">
          <table className="w-full min-w-[1260px] text-sm">
            <thead className="border-b border-slate-800 bg-slate-950/50 text-xs uppercase tracking-wide text-slate-400">
              <tr>
                <th className="px-3 py-2 text-left">Rank</th>
                <th className="px-3 py-2 text-left">Ticker</th>
                <th className="px-3 py-2 text-left">Sector</th>
                <th className="px-3 py-2 text-left">Signal</th>
                <th className="px-3 py-2 text-right">Alpha</th>
                <th className="px-3 py-2 text-right">Risk adj</th>
                <th className="px-3 py-2 text-right">20d exp</th>
                <th className="px-3 py-2 text-right">RS vs SPY</th>
                <th className="px-3 py-2 text-right">RS vs sector</th>
                <th className="px-3 py-2 text-right">Beta</th>
                <th className="px-3 py-2 text-right">Inst</th>
                <th className="px-3 py-2 text-right">Backtest</th>
                <th className="px-3 py-2 text-left">Plan</th>
                <th className="px-3 py-2 text-right">Action</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const bt20 = row.backtests.find((b) => b.horizonDays === 20);
                const isOpen = expanded === row.ticker;
                return (
                  <Fragment key={row.ticker}>
                    <tr key={row.ticker} className="border-b border-slate-800/80 hover:bg-slate-950/30">
                      <td className="px-3 py-2 text-slate-300">{row.rank}</td>
                      <td className="px-3 py-2">
                        <button
                          type="button"
                          className="mr-2 rounded border border-slate-700 px-1.5 py-0.5 text-xs text-slate-400 hover:bg-slate-800"
                          onClick={() => setExpanded(isOpen ? null : row.ticker)}
                        >
                          {isOpen ? "-" : "+"}
                        </button>
                        <Link href={`/research/${encodeURIComponent(row.ticker)}`} className="font-semibold text-sky-300 underline decoration-dotted underline-offset-2 hover:text-sky-200">
                          {row.ticker}
                        </Link>
                        <div className="max-w-[220px] truncate text-xs text-slate-500">{row.companyName}</div>
                      </td>
                      <td className="px-3 py-2 text-slate-300">{row.sector}</td>
                      <td className="px-3 py-2">
                        <span className={clsx("rounded border px-2 py-1 text-xs font-semibold", actionClass(row.tradePlan.action))}>
                          {row.tradePlan.action}
                        </span>
                        <div className="mt-1 text-[11px] text-slate-500">
                          {row.tradePlan.confidence.toFixed(0)}% confidence
                        </div>
                      </td>
                      <td className={clsx("px-3 py-2 text-right font-semibold tabular-nums", scoreClass(row.alphaScore))}>{row.alphaScore.toFixed(1)}</td>
                      <td className="px-3 py-2 text-right tabular-nums text-slate-300">{row.riskAdjustedScore.toFixed(1)}</td>
                      <td className={clsx("px-3 py-2 text-right tabular-nums", pctClass(row.expectedReturn20d))}>{formatPct(row.expectedReturn20d)}</td>
                      <td className={clsx("px-3 py-2 text-right tabular-nums", pctClass(row.rsVsSpy20d))}>{formatPct(row.rsVsSpy20d)}</td>
                      <td className={clsx("px-3 py-2 text-right tabular-nums", pctClass(row.rsVsSector20d))}>{formatPct(row.rsVsSector20d)}</td>
                      <td className="px-3 py-2 text-right tabular-nums text-slate-300">{row.betaVsSpy.toFixed(2)}</td>
                      <td className="px-3 py-2 text-right text-xs tabular-nums">
                        <div className={institutionalPass(row) ? "font-semibold text-emerald-300" : "text-slate-300"}>
                          {pctValue(row.institutionalOwnershipPct)}
                        </div>
                        <div className={row.institutionalTransactionPct != null && row.institutionalTransactionPct >= INST_TRANSACTION_MIN ? "text-emerald-300" : "text-slate-500"}>
                          {signedPctValue(row.institutionalTransactionPct)}
                        </div>
                      </td>
                      <td className="px-3 py-2 text-right text-xs tabular-nums text-slate-300">
                        {bt20 ? `${formatPct(bt20.alphaAvgReturn)} / ${bt20.winRate.toFixed(0)}%` : "N/A"}
                      </td>
                      <td className="px-3 py-2 text-xs text-slate-400">
                        <div>Entry {formatMoney(row.tradePlan.entry)}</div>
                        <div>
                          T1 {row.tradePlan.target1 == null ? "N/A" : formatMoney(row.tradePlan.target1)}
                          {" "} / Stop {row.tradePlan.stop == null ? "N/A" : formatMoney(row.tradePlan.stop)}
                        </div>
                      </td>
                      <td className="px-3 py-2 text-right">
                        <button
                          type="button"
                          disabled={watched.has(row.ticker)}
                          className="rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-200 hover:bg-slate-800 disabled:opacity-40"
                          onClick={() => addWatch(row)}
                        >
                          {watched.has(row.ticker) ? "Watching" : "Watch"}
                        </button>
                      </td>
                    </tr>
                    {isOpen ? (
                      <tr className="border-b border-slate-800 bg-slate-950/30">
                        <td colSpan={14} className="px-4 py-4">
                          <CandidateDetail row={row} />
                        </td>
                      </tr>
                    ) : null}
                  </Fragment>
                );
              })}
              {rows.length === 0 ? (
                <tr>
                  <td colSpan={14} className="px-4 py-8 text-sm text-slate-400">
                    {loading ? "Scanning..." : "No candidates match the current filters."}
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mt-8">
        <div className="mb-3 flex items-end justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold">Watchlist Journal</h2>
            <p className="mt-1 text-xs text-slate-500">{watchlist.length} tracked setup(s)</p>
          </div>
          {watchlist.length ? (
            <button
              type="button"
              className="rounded-md border border-rose-500/50 px-3 py-1.5 text-xs text-rose-300 hover:bg-rose-500/10"
              onClick={() => setWatchlist([])}
            >
              Clear all
            </button>
          ) : null}
        </div>
        <div className="overflow-hidden rounded-xl border border-slate-800">
          <table className="w-full min-w-[980px] text-sm">
            <thead className="border-b border-slate-800 bg-slate-950/50 text-xs uppercase tracking-wide text-slate-400">
              <tr>
                <th className="px-3 py-2 text-left">Ticker</th>
                <th className="px-3 py-2 text-right">Entry</th>
                <th className="px-3 py-2 text-right">Current</th>
                <th className="px-3 py-2 text-right">Return</th>
                <th className="px-3 py-2 text-left">Thesis</th>
                <th className="px-3 py-2 text-left">Invalidation</th>
                <th className="px-3 py-2 text-right">Action</th>
              </tr>
            </thead>
            <tbody>
              {watchlist.map((item) => {
                const live = rowByTicker.get(item.ticker);
                const current = live?.currentPrice ?? item.entryPrice;
                const ret = (current / item.entryPrice - 1) * 100;
                return (
                  <tr key={item.ticker} className="border-b border-slate-800/80">
                    <td className="px-3 py-2">
                      <Link href={`/research/${encodeURIComponent(item.ticker)}`} className="font-semibold text-sky-300 underline decoration-dotted underline-offset-2">
                        {item.ticker}
                      </Link>
                      <div className="text-xs text-slate-500">
                        {new Date(item.addedAt).toLocaleDateString()} / score {item.entryScore.toFixed(1)}
                      </div>
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-slate-300">{formatMoney(item.entryPrice)}</td>
                    <td className="px-3 py-2 text-right tabular-nums text-slate-300">{formatMoney(current)}</td>
                    <td className={clsx("px-3 py-2 text-right font-semibold tabular-nums", pctClass(ret))}>{formatPct(ret)}</td>
                    <td className="px-3 py-2">
                      <input
                        className="w-full rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-300"
                        value={item.thesis}
                        onChange={(e) => updateWatch(item.ticker, { thesis: e.target.value })}
                      />
                    </td>
                    <td className="px-3 py-2">
                      <input
                        className="w-full rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-300"
                        value={item.invalidation}
                        onChange={(e) => updateWatch(item.ticker, { invalidation: e.target.value })}
                      />
                    </td>
                    <td className="px-3 py-2 text-right">
                      <button
                        type="button"
                        className="rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-300 hover:bg-slate-800"
                        onClick={() => removeWatch(item.ticker)}
                      >
                        Remove
                      </button>
                    </td>
                  </tr>
                );
              })}
              {watchlist.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-4 py-6 text-sm text-slate-500">
                    No watched setups yet.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>
    </div>
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

function Metric(props: { label: string; value: string; detail?: string }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
      <div className="text-xs uppercase tracking-wide text-slate-500">{props.label}</div>
      <div className="mt-1 text-lg font-semibold capitalize text-slate-100">{props.value}</div>
      {props.detail ? <div className="mt-0.5 text-xs text-slate-500">{props.detail}</div> : null}
    </div>
  );
}

function CandidateDetail(props: { row: AlphaCandidateRow }) {
  const row = props.row;
  const instPass = institutionalPass(row);
  return (
    <div className="grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
      <div>
        <div className="mb-4 rounded-md border border-slate-800 bg-slate-900/60 p-3">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <span className={clsx("rounded border px-2 py-1 text-xs font-semibold", actionClass(row.tradePlan.action))}>
              {row.tradePlan.action}
            </span>
            <span className="text-xs text-slate-500">{row.tradePlan.horizon}</span>
          </div>
          <p className="text-sm text-slate-300">{row.tradePlan.rationale}</p>
          <div className="mt-3 grid gap-2 text-xs text-slate-400 sm:grid-cols-3">
            <span>Entry: <b className="text-slate-200">{formatMoney(row.tradePlan.entry)}</b></span>
            <span>Buy below: <b className="text-slate-200">{row.tradePlan.buyBelow == null ? "N/A" : formatMoney(row.tradePlan.buyBelow)}</b></span>
            <span>Sell/short above: <b className="text-slate-200">{row.tradePlan.sellAbove == null ? "N/A" : formatMoney(row.tradePlan.sellAbove)}</b></span>
            <span>Target 1: <b className="text-emerald-300">{row.tradePlan.target1 == null ? "N/A" : formatMoney(row.tradePlan.target1)}</b></span>
            <span>Target 2: <b className="text-emerald-300">{row.tradePlan.target2 == null ? "N/A" : formatMoney(row.tradePlan.target2)}</b></span>
            <span>Stop: <b className="text-rose-300">{row.tradePlan.stop == null ? "N/A" : formatMoney(row.tradePlan.stop)}</b></span>
            <span>Risk/reward: <b className="text-slate-200">{row.tradePlan.riskReward == null ? "N/A" : `${row.tradePlan.riskReward.toFixed(2)}x`}</b></span>
            <span>Current: <b className="text-slate-200">{formatMoney(row.currentPrice)}</b></span>
            <span>Price date: <b className="text-slate-200">{row.priceDate}</b></span>
          </div>
          {row.tradePlan.optionStrategy ? (
            <div className="mt-3 rounded border border-sky-500/30 bg-sky-500/10 p-2 text-xs text-sky-200">
              <div className="mb-1 flex flex-wrap items-center gap-1.5">
                {row.tradePlan.optionCategory ? (
                  <span className="rounded border border-sky-400/30 bg-sky-400/10 px-1.5 py-0.5 font-semibold text-sky-100">
                    {row.tradePlan.optionCategory}
                  </span>
                ) : null}
                <span>{row.tradePlan.optionDte == null ? "DTE N/A" : `${row.tradePlan.optionDte} DTE`}</span>
                <span>
                  IV proxy {row.tradePlan.optionIvProxy == null ? "N/A" : `${row.tradePlan.optionIvProxy.toFixed(1)}%`}
                </span>
                <span className={row.tradePlan.optionIvGate === "pass" ? "text-emerald-300" : "text-amber-300"}>
                  {row.tradePlan.optionIvGate === "pass" ? "IV gate pass" : "IV gate below 50"}
                </span>
              </div>
              <b>{row.tradePlan.optionStrategy}</b>
              {" "} / {row.tradePlan.optionDirection}
              {" "} / strike {row.tradePlan.optionStrike == null ? "N/A" : formatMoney(row.tradePlan.optionStrike)}
              {" "} / expiry {row.tradePlan.optionExpiry ?? "N/A"}
              <div className="mt-1 text-sky-200/80">{row.tradePlan.optionRationale}</div>
              {row.tradePlan.optionRules.length ? (
                <div className="mt-2 flex flex-wrap gap-1">
                  {row.tradePlan.optionRules.map((rule) => (
                    <span key={rule} className="rounded border border-slate-700 bg-slate-950/50 px-1.5 py-0.5 text-[11px] text-slate-300">
                      {rule}
                    </span>
                  ))}
                </div>
              ) : null}
            </div>
          ) : null}
          <div className="mt-3 rounded border border-violet-500/30 bg-violet-500/10 p-2 text-xs text-violet-100">
            <div className="mb-1 flex flex-wrap items-center gap-1.5">
              <span className={clsx(
                "rounded border px-1.5 py-0.5 font-semibold",
                instPass
                  ? "border-emerald-400/40 bg-emerald-400/10 text-emerald-200"
                  : "border-violet-400/30 bg-violet-400/10 text-violet-100",
              )}>
                {instPass ? "Institutional pass" : "Institutional watch"}
              </span>
              <span>Own {pctValue(row.institutionalOwnershipPct)}</span>
              <span>Adding {signedPctValue(row.institutionalTransactionPct)}</span>
              {row.institutionalDataSource ? <span>{row.institutionalDataSource}</span> : null}
              {row.institutionalSourceDate ? <span>{row.institutionalSourceDate}</span> : null}
            </div>
            <div className="text-violet-100/80">
              Scanner: institutions own at least {INST_OWNERSHIP_MIN}% and added at least +{INST_TRANSACTION_MIN}% in 13F share-change data.
            </div>
            {row.institutionalNotes.length ? (
              <div className="mt-2 flex flex-wrap gap-1">
                {row.institutionalNotes.map((note) => (
                  <span key={note} className="rounded border border-slate-700 bg-slate-950/50 px-1.5 py-0.5 text-[11px] text-slate-300">
                    {note}
                  </span>
                ))}
              </div>
            ) : null}
          </div>
        </div>
        <div className="mb-2 text-xs uppercase tracking-wide text-slate-500">Signals</div>
        <div className="grid gap-2 md:grid-cols-2">
          {row.signals.map((signal) => (
            <div key={signal.id} className={clsx("rounded-md border px-3 py-2", stateClass(signal.state))}>
              <div className="flex items-center justify-between gap-2">
                <span className="font-medium">{signal.label}</span>
                <span className="font-mono text-xs">{signal.contribution > 0 ? "+" : ""}{signal.contribution.toFixed(1)}</span>
              </div>
              <div className="mt-1 text-xs opacity-80">{signal.detail || "N/A"}</div>
            </div>
          ))}
        </div>
        <div className="mt-3 text-xs text-slate-400">
          {row.catalystNotes.join(" ")}
        </div>
      </div>
      <div>
        <div className="mb-2 text-xs uppercase tracking-wide text-slate-500">Forward Tests</div>
        <div className="overflow-hidden rounded-md border border-slate-800">
          <table className="w-full text-xs">
            <thead className="bg-slate-950/50 text-slate-500">
              <tr>
                <th className="px-2 py-1 text-left">Horizon</th>
                <th className="px-2 py-1 text-right">Samples</th>
                <th className="px-2 py-1 text-right">Win</th>
                <th className="px-2 py-1 text-right">Avg</th>
                <th className="px-2 py-1 text-right">Alpha</th>
              </tr>
            </thead>
            <tbody>
              {row.backtests.map((bt) => (
                <tr key={bt.horizonDays} className="border-t border-slate-800">
                  <td className="px-2 py-1">{bt.horizonDays}d</td>
                  <td className="px-2 py-1 text-right tabular-nums">{bt.sampleSize}</td>
                  <td className="px-2 py-1 text-right tabular-nums">{bt.winRate.toFixed(0)}%</td>
                  <td className={clsx("px-2 py-1 text-right tabular-nums", pctClass(bt.avgReturn))}>{formatPct(bt.avgReturn)}</td>
                  <td className={clsx("px-2 py-1 text-right tabular-nums", pctClass(bt.alphaAvgReturn))}>{formatPct(bt.alphaAvgReturn)}</td>
                </tr>
              ))}
              {row.backtests.length === 0 ? (
                <tr>
                  <td colSpan={5} className="px-2 py-3 text-slate-500">Not enough history for a local test.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
        <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-slate-400">
          <span>20d momentum: <b className={pctClass(row.momentum20d)}>{formatPct(row.momentum20d)}</b></span>
          <span>63d momentum: <b className={pctClass(row.momentum63d)}>{formatPct(row.momentum63d)}</b></span>
          <span>Volatility: <b>{formatPct(row.volatility20d)}</b></span>
          <span>Drawdown: <b className={pctClass(row.maxDrawdown63d)}>{formatPct(row.maxDrawdown63d)}</b></span>
          <span>Factor: <b>{row.factorExposure}</b></span>
          <span>Regime: <b>{row.regimeFit}</b></span>
        </div>
      </div>
    </div>
  );
}
