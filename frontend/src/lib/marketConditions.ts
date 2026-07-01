export type ConditionState = "triggered" | "not_triggered" | "unknown";

export type MarketCondition = {
  id: string;
  name: string;
  category: string;
  threshold: string;
  source: string;
  value: string | number | null;
  state: ConditionState;
  notes?: string;
  evaluate?: (value: string | number | null) => ConditionState;
};

export type HistoricalPeak = {
  date: string;
  pctTriggered: number;
  spLevel: number;
};

function num(value: string | number | null): number | null {
  if (value === null || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

// ===== EDIT HERE: add or change conditions =====
export const SEED_CONDITIONS: MarketCondition[] = [
  {
    id: "cb_consumer_confidence",
    name: "Conf Board Consumer Confidence > 110 (prior 6m)",
    category: "Sentiment",
    threshold: "index > 110",
    source: "Conference Board (manual)",
    value: null,
    state: "unknown",
    evaluate(value) {
      const n = num(value);
      if (n === null) return "unknown";
      return n > 110 ? "triggered" : "not_triggered";
    },
  },
  {
    id: "cb_net_pct_stocks_higher",
    name: "Conf Board: Net % Expecting Stocks Higher > 20",
    category: "Sentiment",
    threshold: "net % > 20",
    source: "Conference Board (manual)",
    value: null,
    state: "unknown",
    evaluate(value) {
      const n = num(value);
      if (n === null) return "unknown";
      return n > 20 ? "triggered" : "not_triggered";
    },
  },
  {
    id: "sell_side_indicator",
    name: 'Sell Side Indicator: "Sell" signal triggered',
    category: "Sentiment",
    threshold: "signal = Sell",
    source: "BofA (manual)",
    value: null,
    state: "unknown",
    evaluate(value) {
      if (value === null || value === "") return "unknown";
      const s = String(value).trim().toLowerCase();
      if (s === "sell" || s === "1" || s === "true" || s === "yes") return "triggered";
      if (s === "hold" || s === "buy" || s === "0" || s === "false" || s === "no") return "not_triggered";
      return "unknown";
    },
  },
  {
    id: "ltg_5yr_z",
    name: "S&P 500 LT growth expectations (LTG): 5yr Z > 1",
    category: "Sentiment",
    threshold: "z-score > 1",
    source: "IBES (manual)",
    value: null,
    state: "unknown",
    evaluate(value) {
      const n = num(value);
      if (n === null) return "unknown";
      return n > 1 ? "triggered" : "not_triggered";
    },
  },
  {
    id: "mna_10yr_z",
    name: "10yr Z of # of M&A deals (3m sum) > 1",
    category: "Sentiment",
    threshold: "z-score > 1",
    source: "Deal data (manual)",
    value: null,
    state: "unknown",
    evaluate(value) {
      const n = num(value);
      if (n === null) return "unknown";
      return n > 1 ? "triggered" : "not_triggered";
    },
  },
  {
    id: "valuation_z",
    name: "10yr Z of (trailing S&P 500 PE + YoY CPI) > 1",
    category: "Valuation",
    threshold: "z-score > 1",
    source: "Computed (FRED + PE)",
    value: null,
    state: "unknown",
    evaluate(value) {
      const n = num(value);
      if (n === null) return "unknown";
      return n > 1 ? "triggered" : "not_triggered";
    },
  },
  {
    id: "low_minus_high_pe_6m",
    name: "Low PE underperforms High PE by 2.5ppt over 6m",
    category: "Valuation",
    threshold: "low − high ≤ −2.5 ppt",
    source: "Factor data (manual)",
    value: null,
    state: "unknown",
    evaluate(value) {
      const n = num(value);
      if (n === null) return "unknown";
      return n <= -2.5 ? "triggered" : "not_triggered";
    },
  },
  {
    id: "inverted_curve",
    name: "Inverted yield curve (prior 6m)",
    category: "Macro",
    threshold: "10y–2y < 0 in last 6m",
    source: "FRED (T10Y2Y)",
    value: null,
    state: "unknown",
    evaluate(value) {
      if (value === null || value === "") return "unknown";
      const s = String(value).trim().toLowerCase();
      if (s === "yes" || s === "true" || s === "1" || s === "inverted") return "triggered";
      if (s === "no" || s === "false" || s === "0" || s === "normal") return "not_triggered";
      const n = num(value);
      if (n === null) return "unknown";
      return n < 0 ? "triggered" : "not_triggered";
    },
  },
  {
    id: "credit_stress_indicator",
    name: "Credit Stress Indicator drops below 0.25",
    category: "Macro",
    threshold: "level < 0.25",
    source: "BofA (manual)",
    value: null,
    state: "unknown",
    evaluate(value) {
      const n = num(value);
      if (n === null) return "unknown";
      return n < 0.25 ? "triggered" : "not_triggered";
    },
  },
  {
    id: "sloos_tightening",
    name: "Tightening credit conditions (SLOOS)",
    category: "Macro",
    threshold: "net % tightening > 0",
    source: "FRED (DRTSCILM)",
    value: null,
    state: "unknown",
    evaluate(value) {
      const n = num(value);
      if (n === null) return "unknown";
      return n > 0 ? "triggered" : "not_triggered";
    },
  },
];

export const HISTORICAL_PEAKS: HistoricalPeak[] = [
  { date: "Jul-90", pctTriggered: 88, spLevel: 369 },
  { date: "Mar-00", pctTriggered: 90, spLevel: 1527 },
  { date: "Oct-07", pctTriggered: 80, spLevel: 1565 },
  { date: "Sep-18", pctTriggered: 60, spLevel: 2931 },
  { date: "Feb-20", pctTriggered: 50, spLevel: 3386 },
  { date: "Jan-22", pctTriggered: 50, spLevel: 4797 },
  { date: "Feb-25", pctTriggered: 70, spLevel: 6144 },
  { date: "Mar-26", pctTriggered: 40, spLevel: 6529 },
  { date: "Apr-26", pctTriggered: 50, spLevel: 7209 },
  { date: "May-26", pctTriggered: 70, spLevel: 7580 },
];

export const CATEGORY_ORDER = ["Sentiment", "Valuation", "Macro"];

export type RuntimeCondition = MarketCondition & { manualState?: boolean };

export function cloneSeedConditions(): RuntimeCondition[] {
  return SEED_CONDITIONS.map((c) => ({
    id: c.id,
    name: c.name,
    category: c.category,
    threshold: c.threshold,
    source: c.source,
    value: c.value,
    state: c.state,
    notes: c.notes ?? "",
    evaluate: c.evaluate,
    manualState: false,
  }));
}

export function resolveState(c: RuntimeCondition): ConditionState {
  if (c.manualState) return c.state;
  if (c.evaluate) {
    const derived = c.evaluate(c.value);
    if (derived === "triggered" || derived === "not_triggered" || derived === "unknown") {
      return derived;
    }
  }
  return c.state;
}

export function computeSummary(conditions: RuntimeCondition[]) {
  let triggered = 0;
  let known = 0;
  let unknown = 0;
  for (const c of conditions) {
    const state = resolveState(c);
    if (state === "unknown") unknown++;
    else {
      known++;
      if (state === "triggered") triggered++;
    }
  }
  const pct = known > 0 ? Math.round((triggered / known) * 100) : null;
  return { triggered, known, unknown, pct, total: conditions.length };
}

export function sortedCategories(conditions: RuntimeCondition[]): string[] {
  const seen = new Set<string>();
  const ordered: string[] = [];
  for (const cat of CATEGORY_ORDER) {
    if (conditions.some((c) => c.category === cat)) {
      ordered.push(cat);
      seen.add(cat);
    }
  }
  const rest = [...new Set(conditions.map((c) => c.category))]
    .filter((c) => !seen.has(c))
    .sort();
  return [...ordered, ...rest];
}

const templateById = new Map(SEED_CONDITIONS.map((c) => [c.id, c]));

export function mergeImportedConditions(
  incoming: Array<Partial<RuntimeCondition> & { id: string }>,
): RuntimeCondition[] {
  return incoming.map((item) => {
    const template = templateById.get(item.id);
    const merged: RuntimeCondition = {
      id: item.id,
      name: item.name ?? template?.name ?? item.id,
      category: item.category ?? template?.category ?? "Other",
      threshold: item.threshold ?? template?.threshold ?? "",
      source: item.source ?? template?.source ?? "",
      value: item.value ?? null,
      state: (item.state as ConditionState) ?? "unknown",
      notes: item.notes ?? "",
      manualState: !!item.manualState,
      evaluate: template?.evaluate,
    };
    return merged;
  });
}

export function applyFetchedConditions(
  conditions: RuntimeCondition[],
  fetched: Array<{
    id: string;
    value: string | number | null;
    state: ConditionState;
    fetched?: boolean;
    note?: string | null;
  }>,
): RuntimeCondition[] {
  const byId = new Map(fetched.map((r) => [r.id, r]));
  return conditions.map((c) => {
    const row = byId.get(c.id);
    if (!row || !row.fetched) return c;
    const next: RuntimeCondition = {
      ...c,
      value: row.value,
      state: row.state,
      manualState: false,
      notes: row.note ? String(row.note) : c.notes,
    };
    return next;
  });
}
