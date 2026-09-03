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

export type WeeklyMaWatchSignal = "crossed_below" | "below" | "near" | "reclaimed";

export type WeeklyMaWatchRow = {
  ticker: string;
  companyName: string;
  sector: string;
  currentPrice: number;
  priceDate: string;
  dailySma: number;
  dailyDistancePct: number;
  weeklySma: number;
  distancePct: number;
  signal: WeeklyMaWatchSignal;
  weeklyObservations: number;
};

export type WeeklyMaWatchResponse = {
  asOf: string;
  maLength: number;
  maType: "SMA";
  nearPct: number;
  stocks: WeeklyMaWatchRow[];
  meta: {
    total: number;
    computed: number;
    skipped: number;
    candidateCount: number;
    crossedBelowCount: number;
    belowCount: number;
    nearCount: number;
    reclaimedCount: number;
    maLength: number;
    nearPct: number;
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
    source?: string | null;
  };
  dataQuality?: {
    status: "complete" | "degraded";
    priceSource: string;
    priceBars: number;
    fundamentalsAvailable: number;
    fundamentalsTotal: number;
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
  optionCategory: string | null;
  optionDte: number | null;
  optionIvProxy: number | null;
  optionIvGate: string | null;
  optionRules: string[];
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
  institutionalOwnershipPct: number | null;
  institutionalTransactionPct: number | null;
  institutionalScannerPass: boolean;
  institutionalSourceDate: string | null;
  institutionalDataSource: string | null;
  institutionalNotes: string[];
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
    eligible?: number;
    computed: number;
    returned: number;
    skipped: number;
    requested?: number;
    available?: number;
    coveragePct?: number;
    priceCoveragePct?: number;
    missingTickers?: string[];
    status?: "complete" | "partial";
    minScore: number;
    sector: string | null;
    maxBeta: number | null;
    signals: string[];
    warnings: string[];
  };
};

export type AgentBotCatalyst = {
  available: boolean;
  earningsDate: string | null;
  exDividendDate: string | null;
  targetMeanPrice: number | null;
  analystRecommendation: string | null;
  analystCount: number | null;
  revenueGrowth: number | null;
  earningsGrowth: number | null;
  epsGrowth: number | null;
  dividendYield: number | null;
  revisionNotes: string[];
};

export type AgentBotRecommendation = {
  rank: number;
  ticker: string;
  companyName: string;
  sector: string;
  action: "BUY" | "SELL" | "WATCH" | "AVOID";
  confidence: number;
  alphaScore: number;
  riskAdjustedScore: number;
  expectedReturn20d: number;
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
  optionCategory: string | null;
  optionDte: number | null;
  optionIvProxy: number | null;
  optionIvGate: string | null;
  optionRules: string[];
  rationale: string;
  whyNow: string;
  signals: AlphaSignal[];
  backtests: AlphaBacktestMetric[];
  catalyst: AgentBotCatalyst;
};

export type AgentBotTracking = {
  id: string | null;
  ticker: string;
  companyName: string;
  action: "BUY" | "SELL" | "WATCH" | "AVOID";
  entry: number;
  currentPrice: number;
  priceDate: string;
  stop: number | null;
  target1: number | null;
  target2: number | null;
  unrealizedReturnPct: number | null;
  alphaScore: number;
  whyNow: string;
};

export type AgentBotAlert = {
  ticker: string;
  type: string;
  severity: "high" | "medium" | "low";
  message: string;
};

export type AgentBotOutcome = {
  id: string | null;
  ticker: string;
  action: string;
  entryPrice: number;
  currentPrice: number;
  returnPct: number;
  recommendedAt: string | null;
  status: "open" | "closed";
};

export type AgentBotForwardEntry = {
  id: string | null;
  ticker: string;
  action: string;
  entryPrice: number;
  recommendedAt: string;
  closed: boolean;
  forwardReturns: Record<string, number | null>;
};

export type AgentBotForwardAggregate = {
  count: number;
  avgReturn: number | null;
};

export type AgentBotForwardJournal = {
  entries: AgentBotForwardEntry[];
  aggregates: Record<string, AgentBotForwardAggregate>;
};

export type AgentBotBriefing = {
  summary: string;
  regime: string;
  riskLevel: string;
  topBuy: AgentBotRecommendation | null;
  topSell: AgentBotRecommendation | null;
  topWatch: AgentBotRecommendation | null;
  topAvoid: AgentBotRecommendation | null;
  counts: {
    buy: number;
    sell: number;
    watch: number;
    avoid: number;
    alerts: number;
    outcomes: number;
  };
};

export type AgentBotRunResponse = {
  asOf: string;
  mode: "sp500" | "watchlist";
  briefing: AgentBotBriefing;
  recommendations: AgentBotRecommendation[];
  activeTracking: AgentBotTracking[];
  alerts: AgentBotAlert[];
  outcomes: AgentBotOutcome[];
  forwardJournal: AgentBotForwardJournal;
  catalysts: Record<string, AgentBotCatalyst>;
  meta: {
    mode: string;
    riskMode: string;
    regime: string;
    topN: number;
    minScore: number;
    adjustedMinScore?: number;
    adjustedMaxBeta?: number | null;
    marketConditions?: {
      riskLevel: string;
      coveragePct: number;
      triggeredCount: number;
      asOf: string | null;
    };
    [key: string]: unknown;
  };
};

export type InstitutionalBacktest = {
  winRate: number | null;
  avgReturn: number | null;
  medianReturn: number | null;
  benchmarkAvgReturn: number | null;
  alphaAvgReturn: number | null;
  maxDrawdown: number | null;
  sampleSize: number;
  valid: boolean;
};

export type InstitutionalSimulationScenario = {
  winRate: number;
  avgReturn: number;
  survives: boolean;
};

export type InstitutionalSimulation = {
  scenarios: Record<string, InstitutionalSimulationScenario>;
  allScenariosSurvive: boolean;
};

export type InstitutionalConfidence = {
  confidence: number;
  sampleSize: number;
  trustworthy: boolean;
  reason: string;
  calibrationDetails?: {
    baseConfidence: number;
    samplePenalty: number;
    simulationMultiplier: number;
    riskAdjustment: number;
  };
};

export type InstitutionalGateDelta = {
  actual: any;
  required: any;
  delta?: any;
  pass: boolean;
  failedScenarios?: string[];
};

export type InstitutionalTradeGate = {
  decision: "TAKE" | "PASS";
  reasons: string[];
  gateConditions: {
    confidence: boolean;
    winRate: boolean;
    sampleSize: boolean;
    alphaVsBenchmark: boolean;
    simulationSurvival: boolean;
  };
  gateDeltas: Record<string, InstitutionalGateDelta>;
  numFailures: number;
  watchTier: boolean;
};

export type InstitutionalConvexityAlert = {
  ticker: string;
  type: string;
  probability: number;
  expectedReturn: string;
  requiredStockMove: number;
  currentPrice: number;
  volatility: number;
  alphaScore: number;
  message: string;
};

export type InstitutionalCandidate = {
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
  tradePlan: AlphaTradePlan;
  signals: AlphaSignal[];
  backtests: AlphaBacktestMetric[];
  backtest: InstitutionalBacktest;
  simulation: InstitutionalSimulation;
  confidence: InstitutionalConfidence;
  tradeGate: InstitutionalTradeGate;
  convexityAlert: InstitutionalConvexityAlert | null;
};

export type InstitutionalScannerResponse = {
  asOf: string;
  marketRegime: {
    state: string;
    spyTrend: string;
    spyDrawdownPct: number | null;
    effectiveState: string;
    riskMode: string;
  };
  candidates: InstitutionalCandidate[];
  convexityAlerts: InstitutionalConvexityAlert[];
  meta: {
    total: number;
    eligible: number;
    computed: number;
    returned: number;
    skipped: number;
    priceCoveragePct: number;
    minScore: number;
    sector: string | null;
    maxBeta: number | null;
    signals: string[];
    warnings: string[];
    scannerVersion: string;
    status: string;
    backtestHorizon: string;
    simulationScenarios: string[];
    tradeGate: {
      minConfidence: number;
      minWinRate: number;
      minSampleSize: number;
      minAlphaVsBenchmark: number;
    };
    convexityAlert: {
      minProbability: number;
      minReturn: string;
    };
    [key: string]: unknown;
  };
};
