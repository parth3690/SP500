from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class Constituent(BaseModel):
    ticker: str = Field(..., description="Ticker symbol as shown in the S&P 500 list.")
    yahooTicker: str = Field(..., description="Ticker symbol compatible with Yahoo Finance.")
    companyName: str
    sector: str
    subIndustry: Optional[str] = None


class MoverRow(BaseModel):
    rank: int
    ticker: str
    companyName: str
    sector: str
    currentPrice: float
    currentPriceDate: date
    pastPrice: float
    pastPriceDate: date
    pctChange: float
    trailingPE: Optional[float] = None
    forwardPE: Optional[float] = None


class SectorSummaryRow(BaseModel):
    sector: str
    count: int
    avgPctChange: float
    medianPctChange: float
    positiveCount: int
    negativeCount: int


class MoversResponse(BaseModel):
    start: date
    end: date
    asOf: datetime
    gainers: list[MoverRow]
    losers: list[MoverRow]
    sectorSummary: list[SectorSummaryRow]
    meta: dict[str, Any]
    all: Optional[list[MoverRow]] = None


class CrossoverRow(BaseModel):
    ticker: str
    companyName: str
    sector: str
    currentPrice: float
    priceDate: date
    dma50: float
    dma200: float
    gapPct: float = Field(..., description="Gap between 50-DMA and 200-DMA as % of 200-DMA. Negative = 50-DMA below 200-DMA.")
    signal: str = Field(..., description="'near_golden_cross' or 'near_death_cross'")


class CrossoversResponse(BaseModel):
    asOf: datetime
    thresholdPct: float
    nearGoldenCross: list[CrossoverRow]
    nearDeathCross: list[CrossoverRow]
    meta: dict[str, Any]


class OversoldRow(BaseModel):
    ticker: str
    companyName: str
    sector: str
    currentPrice: float
    priceDate: date
    weeklyRSI: Optional[float] = Field(None, description="14-period weekly RSI value")
    dailyRSI: Optional[float] = Field(None, description="14-period daily RSI for reference")


class OversoldResponse(BaseModel):
    asOf: datetime
    rsiThreshold: float
    stocks: list[OversoldRow]
    meta: dict[str, Any]


class OverboughtRow(BaseModel):
    ticker: str
    companyName: str
    sector: str
    currentPrice: float
    priceDate: date
    weeklyRSI: Optional[float] = Field(None, description="14-period weekly RSI value")
    dailyRSI: Optional[float] = Field(None, description="14-period daily RSI for reference")


class OverboughtResponse(BaseModel):
    asOf: datetime
    rsiThreshold: float
    stocks: list[OverboughtRow]
    meta: dict[str, Any]


class MultibaggerCriterion(BaseModel):
    id: str
    name: str
    threshold: str
    valueDisplay: Optional[str] = None
    status: str = Field(..., description="'pass' | 'fail' | 'skip'")
    soft: bool = False


class MultibaggerMetrics(BaseModel):
    marketCap: Optional[float] = None
    pe: Optional[float] = None
    peg: Optional[float] = None
    roe: Optional[float] = None
    roic: Optional[float] = None
    debtToEquity: Optional[float] = None
    insider: Optional[float] = None
    revGrowth: Optional[float] = None
    earnGrowth: Optional[float] = None
    opMargin: Optional[float] = None
    priceToSales: Optional[float] = None
    evEbitda: Optional[float] = None


class MultibaggerResponse(BaseModel):
    asOf: datetime
    ticker: str
    name: str
    sector: str
    deep: bool
    metrics: MultibaggerMetrics
    sectorPeMedian: Optional[float] = None
    nGreen: int
    nTotal: int
    passedAll: bool
    green: list[str]
    fails: list[str]
    skipped: list[str]
    criteria: list[MultibaggerCriterion]


class MarketConditionFetchRow(BaseModel):
    id: str
    value: Optional[Any] = None
    state: str = Field(..., description="'triggered' | 'not_triggered' | 'unknown'")
    fetched: bool = False
    note: Optional[str] = None


class MarketConditionsFetchResponse(BaseModel):
    asOf: datetime
    conditions: list[MarketConditionFetchRow]
    meta: dict[str, Any]


class AlphaSignal(BaseModel):
    id: str
    label: str
    state: str = Field(..., description="'bullish' | 'neutral' | 'bearish'")
    contribution: float
    detail: str


class AlphaBacktestMetric(BaseModel):
    signal: str
    horizonDays: int
    sampleSize: int
    winRate: float
    avgReturn: float
    medianReturn: float
    benchmarkAvgReturn: float
    alphaAvgReturn: float


class AlphaTradePlan(BaseModel):
    action: str = Field(..., description="'BUY' | 'SELL' | 'WATCH' | 'AVOID'")
    confidence: float
    horizon: str
    entry: float
    buyBelow: Optional[float] = None
    sellAbove: Optional[float] = None
    stop: Optional[float] = None
    target1: Optional[float] = None
    target2: Optional[float] = None
    riskReward: Optional[float] = None
    optionStrategy: Optional[str] = None
    optionDirection: Optional[str] = None
    optionStrike: Optional[float] = None
    optionExpiry: Optional[str] = None
    optionRationale: Optional[str] = None
    rationale: str


class AlphaCandidateRow(BaseModel):
    rank: int
    ticker: str
    companyName: str
    sector: str
    currentPrice: float
    priceDate: date
    alphaScore: float
    technicalScore: float
    riskAdjustedScore: float
    expectedReturn20d: float
    momentum20d: float
    momentum63d: float
    rsVsSpy20d: float
    rsVsSector20d: float
    sectorStrength20d: float
    volatility20d: float
    betaVsSpy: float
    maxDrawdown63d: float
    trendState: str
    factorExposure: str
    regimeFit: str
    catalystScore: float
    revisionScore: float
    catalystNotes: list[str]
    tradePlan: AlphaTradePlan
    signals: list[AlphaSignal]
    backtests: list[AlphaBacktestMetric]


class AlphaCandidatesResponse(BaseModel):
    asOf: datetime
    marketRegime: dict[str, Any]
    candidates: list[AlphaCandidateRow]
    meta: dict[str, Any]


class AlphaWatchlistRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1, max_length=100)
    limit: int = Field(50, ge=1, le=100)
    minScore: float = Field(0.0, ge=0.0, le=100.0)
    maxBeta: Optional[float] = Field(None, ge=0.1, le=5.0)
    riskMode: str = Field("balanced", pattern="^(balanced|aggressive|defensive)$")
    regime: str = Field("auto", pattern="^(auto|risk_on|neutral|risk_off)$")
    enrichTop: int = Field(20, ge=0, le=50)
    refresh: bool = False
