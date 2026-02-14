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

