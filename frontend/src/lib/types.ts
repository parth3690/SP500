export type MoverRow = {
  rank: number;
  ticker: string;
  companyName: string;
  sector: string;
  currentPrice: number;
  currentPriceDate: string;
  pastPrice: number;
  pastPriceDate: string;
  pctChange: number;
  trailingPE: number | null;
  forwardPE: number | null;
};

export type SectorSummaryRow = {
  sector: string;
  count: number;
  avgPctChange: number;
  medianPctChange: number;
  positiveCount: number;
  negativeCount: number;
};

export type MoversResponse = {
  start: string;
  end: string;
  asOf: string;
  gainers: MoverRow[];
  losers: MoverRow[];
  sectorSummary: SectorSummaryRow[];
  meta: {
    total: number;
    computed: number;
    missingCount: number;
    missingTickers: string[];
    computedAt: string;
  };
  all?: MoverRow[];
};

export type CrossoverRow = {
  ticker: string;
  companyName: string;
  sector: string;
  currentPrice: number;
  priceDate: string;
  dma50: number;
  dma200: number;
  gapPct: number;
  signal: "near_golden_cross" | "near_death_cross";
};

export type CrossoversResponse = {
  asOf: string;
  thresholdPct: number;
  nearGoldenCross: CrossoverRow[];
  nearDeathCross: CrossoverRow[];
  meta: {
    total: number;
    computed: number;
    skipped: number;
    nearGoldenCross: number;
    nearDeathCross: number;
    thresholdPct: number;
    computedAt: string;
  };
};

export type OversoldRow = {
  ticker: string;
  companyName: string;
  sector: string;
  currentPrice: number;
  priceDate: string;
  weeklyRSI: number | null;
  dailyRSI: number | null;
};

export type OversoldResponse = {
  asOf: string;
  rsiThreshold: number;
  stocks: OversoldRow[];
  meta: {
    total: number;
    computed: number;
    skipped: number;
    oversoldCount: number;
    rsiThreshold: number;
    computedAt: string;
  };
};

export type OverboughtRow = {
  ticker: string;
  companyName: string;
  sector: string;
  currentPrice: number;
  priceDate: string;
  weeklyRSI: number | null;
  dailyRSI: number | null;
};

export type OverboughtResponse = {
  asOf: string;
  rsiThreshold: number;
  stocks: OverboughtRow[];
  meta: {
    total: number;
    computed: number;
    skipped: number;
    overboughtCount: number;
    rsiThreshold: number;
    computedAt: string;
  };
};

/** Present when backend merged a Financial Modeling Prep live quote (see FMP_API_KEY). */
export type LiveQuoteMeta = {
  source: string;
  providerUrl?: string;
  symbol?: string;
  asOf?: string | null;
};

export type ResearchData = {
  ticker: string;
  companyName: string;
  sector: string;
  dateRangeStart: string;
  dateRangeEnd: string;
  currentPrice: number;
  previousClose: number;
  change: number;
  changePct: number;
  volume: number;
  avgVolume: number;
  /** Last daily close in the OHLCV series (Yahoo); set when live quote overrides header price. */
  chartLastClose?: number;
  liveQuote?: LiveQuoteMeta | null;
  latestRSI: number | null;
  fundamentals: {
    trailingPE: number | null;
    forwardPE: number | null;
    marketCap: number | null;
    fiftyTwoWeekHigh: number | null;
    fiftyTwoWeekLow: number | null;
    beta: number | null;
    dividendYield: number | null;
  };
  ohlcv: {
    dates: string[];
    open: (number | null)[];
    high: (number | null)[];
    low: (number | null)[];
    close: (number | null)[];
    volume: number[];
  };
  indicators: {
    sma50: (number | null)[];
    sma200: (number | null)[];
    rsi: (number | null)[];
    macd: {
      macdLine: (number | null)[];
      signalLine: (number | null)[];
      histogram: (number | null)[];
    };
    bollinger: {
      upper: (number | null)[];
      middle: (number | null)[];
      lower: (number | null)[];
    };
  };
  fibonacci: {
    high: number;
    low: number;
    level_0: number;
    level_236: number;
    level_382: number;
    level_500: number;
    level_618: number;
    level_786: number;
    level_1000: number;
  };
  crossover: {
    dma50: number | null;
    dma200: number | null;
    gapPct: number | null;
    signal: string;
  };
  strategies: {
    name: string;
    icon: string;
    description: string;
    signal: "BUY" | "SELL" | "NEUTRAL";
    confidence: number;
    reasoning: string;
    metrics: Record<string, string>;
  }[];
};

export type MultibaggerCriterion = {
  id: string;
  name: string;
  threshold: string;
  valueDisplay: string | null;
  status: "pass" | "fail" | "skip";
  soft: boolean;
};

export type MultibaggerResponse = {
  asOf: string;
  ticker: string;
  name: string;
  sector: string;
  deep: boolean;
  metrics: {
    marketCap: number | null;
    pe: number | null;
    peg: number | null;
    roe: number | null;
    roic: number | null;
    debtToEquity: number | null;
    insider: number | null;
    revGrowth: number | null;
    earnGrowth: number | null;
    opMargin: number | null;
    priceToSales: number | null;
    evEbitda: number | null;
  };
  sectorPeMedian: number | null;
  nGreen: number;
  nTotal: number;
  passedAll: boolean;
  green: string[];
  fails: string[];
  skipped: string[];
  criteria: MultibaggerCriterion[];
};

export type MarketConditionFetchRow = {
  id: string;
  value: string | number | null;
  state: "triggered" | "not_triggered" | "unknown";
  fetched: boolean;
  note?: string | null;
};

export type MarketConditionsFetchResponse = {
  asOf: string;
  conditions: MarketConditionFetchRow[];
  meta: {
    fredConfigured: boolean;
    fetchedCount: number;
    unknownCount: number;
    triggeredCount?: number;
    coveragePct?: number;
    riskLevel?: string;
    warnings: string[];
  };
};

export type AlphaSignal = {
  id: string;
  label: string;
  state: "bullish" | "neutral" | "bearish";
  contribution: number;
  detail: string;
};

export type AlphaBacktestMetric = {
  signal: string;
  horizonDays: number;
  sampleSize: number;
  winRate: number;
  avgReturn: number;
  medianReturn: number;
  benchmarkAvgReturn: number;
  alphaAvgReturn: number;
};

export type AlphaTradePlan = {
  action: "BUY" | "SELL" | "WATCH" | "AVOID";
  confidence: number;
  horizon: string;
  entry: number;
  buyBelow: number | null;
  sellAbove: number | null;
  stop: number | null;
  target1: number | null;
  target2: number | null;
  riskReward: number | null;
  optionStrategy: string | null;
  optionDirection: string | null;
  optionStrike: number | null;
  optionExpiry: string | null;
  optionRationale: string | null;
  rationale: string;
};

export type AlphaCandidateRow = {
  rank: number;
  ticker: string;
  companyName: string;
  sector: string;
  currentPrice: number;
  priceDate: string;
  alphaScore: number;
  technicalScore: number;
  riskAdjustedScore: number;
  expectedReturn20d: number;
  momentum20d: number;
  momentum63d: number;
  rsVsSpy20d: number;
  rsVsSector20d: number;
  sectorStrength20d: number;
  volatility20d: number;
  betaVsSpy: number;
  maxDrawdown63d: number;
  trendState: string;
  factorExposure: string;
  regimeFit: string;
  catalystScore: number;
  revisionScore: number;
  catalystNotes: string[];
  tradePlan: AlphaTradePlan;
  signals: AlphaSignal[];
  backtests: AlphaBacktestMetric[];
};

export type AlphaCandidatesResponse = {
  asOf: string;
  marketRegime: {
    state: string;
    effectiveState: string;
    riskMode: string;
    spyTrend: string;
    spyDrawdownPct: number | null;
  };
  candidates: AlphaCandidateRow[];
  meta: {
    total: number;
    computed: number;
    returned: number;
    skipped: number;
    minScore: number;
    sector: string | null;
    maxBeta: number | null;
    signals: string[];
    warnings: string[];
  };
};
