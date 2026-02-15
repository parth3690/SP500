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

export type ResearchData = {
  ticker: string;
  companyName: string;
  sector: string;
  currentPrice: number;
  previousClose: number;
  change: number;
  changePct: number;
  volume: number;
  avgVolume: number;
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
    metrics: Record<string, string>;
  }[];
};

