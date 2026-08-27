import { formatMoney, formatPct } from "@/lib/format";
import type { AlphaCandidateRow, AlphaCandidatesResponse } from "@/lib/types";

export const ASK_ALPHA_EXAMPLES = [
  "Which are the best buy candidates?",
  "Show premium 30D option setups with IV 50+",
  "Which stocks have institutions adding?",
  "Compare the top 3 by risk adjusted score",
  "What has strong relative strength vs SPY and sector?",
];

type WatchItemLike = {
  ticker: string;
  entryPrice: number;
  entryScore: number;
  thesis: string;
  invalidation: string;
};

type AskAlphaContext = {
  data: AlphaCandidatesResponse | null;
  visibleRows: AlphaCandidateRow[];
  watchlist: WatchItemLike[];
  filters: {
    option: string;
    institutional: string;
    universe: string;
  };
};

const SECTOR_TERMS = [
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

function cleanQuestion(question: string) {
  return question.trim().replace(/\s+/g, " ");
}

function lower(question: string) {
  return question.toLowerCase();
}

function includesAny(text: string, terms: string[]) {
  return terms.some((term) => text.includes(term));
}

function requestedLimit(question: string, fallback = 5) {
  const match = question.match(/\btop\s+(\d{1,2})\b|\b(\d{1,2})\s+(?:stocks|names|candidates|setups)\b/i);
  const raw = match?.[1] ?? match?.[2];
  if (!raw) return fallback;
  return Math.max(1, Math.min(10, Number(raw)));
}

function tickerMatches(question: string, rows: AlphaCandidateRow[]) {
  const tokens = new Set(question.toUpperCase().match(/\b[A-Z]{1,5}\b/g) ?? []);
  return rows.filter((row) => tokens.has(row.ticker));
}

function sectorMatch(question: string) {
  const q = lower(question);
  return SECTOR_TERMS.find((sector) => q.includes(sector.toLowerCase()));
}

function rowBacktest(row: AlphaCandidateRow, horizonDays = 20) {
  return row.backtests.find((bt) => bt.horizonDays === horizonDays) ?? row.backtests[0] ?? null;
}

function optionLabel(row: AlphaCandidateRow) {
  if (!row.tradePlan.optionStrategy) return "No option idea";
  const category = row.tradePlan.optionCategory ?? "Option";
  const dte = row.tradePlan.optionDte == null ? "DTE N/A" : `${row.tradePlan.optionDte} DTE`;
  const iv = row.tradePlan.optionIvProxy == null ? "IV N/A" : `IV proxy ${row.tradePlan.optionIvProxy.toFixed(1)}%`;
  return `${category}, ${dte}, ${iv}, ${row.tradePlan.optionStrategy}`;
}

function institutionalLabel(row: AlphaCandidateRow) {
  const pass = row.institutionalScannerPass ? "pass" : "watch";
  const ownership = row.institutionalOwnershipPct == null ? "own N/A" : `own ${row.institutionalOwnershipPct.toFixed(1)}%`;
  const change = row.institutionalTransactionPct == null ? "adding N/A" : `adding ${formatPct(row.institutionalTransactionPct, 1)}`;
  return `${pass}, ${ownership}, ${change}`;
}

function planLabel(row: AlphaCandidateRow) {
  const plan = row.tradePlan;
  const target = plan.target1 == null ? "T1 N/A" : `T1 $${formatMoney(plan.target1)}`;
  const stop = plan.stop == null ? "stop N/A" : `stop $${formatMoney(plan.stop)}`;
  return `${plan.action} at $${formatMoney(plan.entry)} (${target}, ${stop}, ${plan.confidence.toFixed(0)}% confidence)`;
}

function compactRow(row: AlphaCandidateRow) {
  const bt = rowBacktest(row);
  const btText = bt ? `20d test ${formatPct(bt.alphaAvgReturn)} alpha / ${bt.winRate.toFixed(0)}% win` : "backtest N/A";
  return [
    `${row.ticker}: alpha ${row.alphaScore.toFixed(1)}, risk-adj ${row.riskAdjustedScore.toFixed(1)}`,
    `RS SPY ${formatPct(row.rsVsSpy20d)}, RS sector ${formatPct(row.rsVsSector20d)}`,
    planLabel(row),
    optionLabel(row),
    institutionalLabel(row),
    btText,
  ].join(" | ");
}

function explainRow(row: AlphaCandidateRow) {
  const bullishSignals = row.signals.filter((signal) => signal.state === "bullish").slice(0, 4);
  const bearishSignals = row.signals.filter((signal) => signal.state === "bearish").slice(0, 3);
  const bt = rowBacktest(row);
  const lines = [
    `${row.ticker} readout`,
    compactRow(row),
    `Why it ranks here: ${row.tradePlan.rationale}`,
  ];
  if (bullishSignals.length) {
    lines.push(`Bullish supports: ${bullishSignals.map((signal) => `${signal.label} (${signal.detail})`).join("; ")}`);
  }
  if (bearishSignals.length) {
    lines.push(`Risks/drag: ${bearishSignals.map((signal) => `${signal.label} (${signal.detail})`).join("; ")}`);
  }
  if (bt) {
    lines.push(`Forward test: ${bt.horizonDays}d sample ${bt.sampleSize}, ${bt.winRate.toFixed(0)}% win rate, ${formatPct(bt.alphaAvgReturn)} average alpha.`);
  }
  if (row.institutionalNotes.length) {
    lines.push(`Institutional notes: ${row.institutionalNotes.slice(0, 3).join(" ")}`);
  }
  return lines.join("\n");
}

function topList(title: string, rows: AlphaCandidateRow[], limit: number) {
  if (!rows.length) return `${title}\nNo rows match that request in the currently loaded scan.`;
  return [
    title,
    ...rows.slice(0, limit).map((row, index) => `${index + 1}. ${compactRow(row)}`),
  ].join("\n");
}

function useAllRows(question: string) {
  const q = lower(question);
  return includesAny(q, ["all rows", "all candidates", "ignore filter", "ignore filters", "full scan", "entire scan"]);
}

function scopedRows(question: string, context: AskAlphaContext) {
  const dataRows = context.data?.candidates ?? [];
  const base = useAllRows(question) ? dataRows : context.visibleRows;
  const sector = sectorMatch(question);
  return sector ? base.filter((row) => row.sector === sector) : base;
}

function marketSummary(context: AskAlphaContext) {
  const data = context.data;
  if (!data) return "Scan first, then I can summarize the market regime and candidates.";
  const rows = context.visibleRows;
  const buys = rows.filter((row) => row.tradePlan.action === "BUY").length;
  const sells = rows.filter((row) => row.tradePlan.action === "SELL").length;
  const premium = rows.filter((row) => row.tradePlan.optionCategory === "Premium 30D" && row.tradePlan.optionIvGate === "pass").length;
  const inst = rows.filter((row) => row.institutionalScannerPass).length;
  return [
    `Market read: ${String(data.marketRegime.effectiveState).replace("_", " ")} (${data.marketRegime.spyTrend}), SPY drawdown ${data.marketRegime.spyDrawdownPct == null ? "N/A" : formatPct(data.marketRegime.spyDrawdownPct)}.`,
    `Visible scan: ${rows.length} candidates, ${buys} BUY, ${sells} SELL, ${premium} premium 30D option setup(s), ${inst} institutional accumulation pass(es).`,
    rows[0] ? `Top candidate: ${compactRow(rows[0])}` : "No visible rows under current filters.",
  ].join("\n");
}

function watchlistAnswer(context: AskAlphaContext) {
  if (!context.watchlist.length) return "No watchlist journal entries yet. Add candidates with the Watch button and I can compare entry vs current scan data.";
  const liveByTicker = new Map((context.data?.candidates ?? []).map((row) => [row.ticker, row]));
  const lines = context.watchlist.slice(0, 10).map((item, index) => {
    const live = liveByTicker.get(item.ticker);
    const current = live?.currentPrice ?? item.entryPrice;
    const ret = (current / item.entryPrice - 1) * 100;
    const liveText = live ? `current alpha ${live.alphaScore.toFixed(1)}, ${live.tradePlan.action}` : "not in current scan";
    return `${index + 1}. ${item.ticker}: entry $${formatMoney(item.entryPrice)}, current $${formatMoney(current)}, ${formatPct(ret)} since add, ${liveText}. Thesis: ${item.thesis}`;
  });
  return ["Watchlist journal readout", ...lines].join("\n");
}

function unavailableValuationAnswer() {
  return [
    "This Alpha table does not carry valuation fields like trailing P/E, forward P/E, P/S, or EV/EBITDA in the loaded row data.",
    "I can answer alpha/risk/RS/options/institutional/backtest questions here. For valuation, open the stock research page or S&P dashboard where those fields are loaded.",
  ].join("\n");
}

export function analyzeAlphaQuestion(question: string, context: AskAlphaContext): string {
  const cleaned = cleanQuestion(question);
  if (!cleaned) return "Ask me about buy candidates, premium 30D options, institutions adding, risk-adjusted rank, relative strength, backtests, targets, or a ticker like NUE.";
  if (!context.data) return "Load or refresh the Alpha scan first, then I can answer from the current candidate data.";

  const q = lower(cleaned);
  const rows = scopedRows(cleaned, context);
  const limit = requestedLimit(cleaned);
  const mentioned = tickerMatches(cleaned, context.data.candidates);

  if (includesAny(q, ["help", "what can you", "examples", "how do i ask"])) {
    return `Try: ${ASK_ALPHA_EXAMPLES.join(" / ")}`;
  }
  if (mentioned.length === 1 && (includesAny(q, ["why", "explain", "read", "analysis", "analyze", "tell me"]) || cleaned.toUpperCase().includes(mentioned[0].ticker))) {
    return explainRow(mentioned[0]);
  }
  if (mentioned.length > 1) {
    return topList("Comparison", mentioned, limit);
  }
  if (includesAny(q, ["watchlist", "journal", "tracked"])) {
    return watchlistAnswer(context);
  }
  if (includesAny(q, ["valuation", "p/e", "pe ", "forward pe", "trailing pe", "price sales", "ev/ebitda", "undervalued"])) {
    return unavailableValuationAnswer();
  }
  if (includesAny(q, ["market", "regime", "overview", "summary", "condition"])) {
    return marketSummary(context);
  }
  if (includesAny(q, ["institution", "institutions", "ownership", "adding", "13f", "smart money", "accumulation"])) {
    const institutional = rows
      .filter((row) => row.institutionalScannerPass || row.institutionalOwnershipPct != null || row.institutionalTransactionPct != null)
      .sort((a, b) => Number(b.institutionalScannerPass) - Number(a.institutionalScannerPass)
        || (b.institutionalTransactionPct ?? -999) - (a.institutionalTransactionPct ?? -999)
        || (b.institutionalOwnershipPct ?? -999) - (a.institutionalOwnershipPct ?? -999));
    return topList("Institutional accumulation readout", institutional, limit);
  }
  if (includesAny(q, ["option", "options", "30 dte", "premium", "iv", "credit spread", "theta"])) {
    const optionRows = rows
      .filter((row) => row.tradePlan.optionStrategy)
      .sort((a, b) => Number(b.tradePlan.optionCategory === "Premium 30D" && b.tradePlan.optionIvGate === "pass")
        - Number(a.tradePlan.optionCategory === "Premium 30D" && a.tradePlan.optionIvGate === "pass")
        || (b.tradePlan.optionIvProxy ?? 0) - (a.tradePlan.optionIvProxy ?? 0)
        || b.alphaScore - a.alphaScore);
    return topList("Options readout", optionRows, limit);
  }
  if (includesAny(q, ["relative strength", "rs vs", "spy", "sector strength", "strong vs sector", "strong vs spy"])) {
    const rsRows = rows
      .filter((row) => row.rsVsSpy20d > 0 && row.rsVsSector20d > 0)
      .sort((a, b) => (b.rsVsSpy20d + b.rsVsSector20d) - (a.rsVsSpy20d + a.rsVsSector20d));
    return topList("Relative strength leaders", rsRows, limit);
  }
  if (includesAny(q, ["risk", "safe", "safer", "risk adjusted", "low beta", "defensive"])) {
    const riskRows = [...rows]
      .sort((a, b) => b.riskAdjustedScore - a.riskAdjustedScore || a.betaVsSpy - b.betaVsSpy);
    return topList("Risk-adjusted leaders", riskRows, limit);
  }
  if (includesAny(q, ["backtest", "win rate", "forward test", "worked before", "historical"])) {
    const btRows = rows
      .filter((row) => rowBacktest(row))
      .sort((a, b) => {
        const ab = rowBacktest(a);
        const bb = rowBacktest(b);
        return (bb?.alphaAvgReturn ?? -999) - (ab?.alphaAvgReturn ?? -999)
          || (bb?.winRate ?? -999) - (ab?.winRate ?? -999);
      });
    return topList("Best forward-test evidence", btRows, limit);
  }
  if (includesAny(q, ["sell", "short", "avoid", "bearish"])) {
    const bearish = rows
      .filter((row) => row.tradePlan.action === "SELL" || row.tradePlan.action === "AVOID")
      .sort((a, b) => a.alphaScore - b.alphaScore);
    return topList("Bearish / avoid list", bearish, limit);
  }
  if (includesAny(q, ["target", "entry", "stop", "buy below", "price", "plan"])) {
    const planRows = [...rows].sort((a, b) => b.tradePlan.confidence - a.tradePlan.confidence || b.alphaScore - a.alphaScore);
    return topList("Trade plan readout", planRows, limit);
  }
  if (includesAny(q, ["compare", "versus", " vs "])) {
    return topList("Comparison", rows, limit);
  }

  const buyRows = rows
    .filter((row) => row.tradePlan.action === "BUY")
    .sort((a, b) => b.alphaScore - a.alphaScore || b.riskAdjustedScore - a.riskAdjustedScore);
  return topList("Best current buy candidates", buyRows.length ? buyRows : rows, limit);
}
