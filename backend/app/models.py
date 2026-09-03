from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal, Optional

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


class WeeklyMaWatchRow(BaseModel):
    ticker: str
    companyName: str
    sector: str
    currentPrice: float
    priceDate: date
    dailySma: float = Field(..., description="Current 200-day simple moving average")
    dailyDistancePct: float = Field(..., description="Price distance from the daily moving average")
    weeklySma: float = Field(..., description="Current moving average of weekly closes")
    distancePct: float = Field(..., description="Price distance from the weekly moving average")
    signal: str = Field(..., description="'crossed_below' | 'below' | 'near' | 'reclaimed'")
    weeklyObservations: int


class WeeklyMaWatchResponse(BaseModel):
    asOf: datetime
    maLength: int
    maType: str
    nearPct: float
    stocks: list[WeeklyMaWatchRow]
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
    optionCategory: Optional[str] = None
    optionDte: Optional[int] = None
    optionIvProxy: Optional[float] = None
    optionIvGate: Optional[str] = None
    optionRules: list[str] = Field(default_factory=list)
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
    institutionalOwnershipPct: Optional[float] = None
    institutionalTransactionPct: Optional[float] = None
    institutionalScannerPass: bool = False
    institutionalSourceDate: Optional[str] = None
    institutionalDataSource: Optional[str] = None
    institutionalNotes: list[str] = Field(default_factory=list)
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


class AgentBotHistoryItem(BaseModel):
    id: Optional[str] = Field(None, max_length=64)
    ticker: str = Field(..., min_length=1, max_length=20)
    action: Literal["BUY", "SELL", "WATCH", "AVOID"]
    entryPrice: float = Field(..., gt=0)
    recommendedAt: datetime
    closed: bool = False
    closedAt: Optional[datetime] = None
    exitPrice: Optional[float] = Field(None, gt=0)


class AgentBotRunRequest(BaseModel):
    mode: str = Field("watchlist", pattern="^(sp500|watchlist)$")
    tickers: list[str] = Field(default_factory=list, max_length=100)
    riskMode: str = Field("balanced", pattern="^(balanced|aggressive|defensive)$")
    regime: str = Field("auto", pattern="^(auto|risk_on|neutral|risk_off)$")
    topN: int = Field(10, ge=1, le=50)
    minScore: float = Field(55.0, ge=0.0, le=100.0)
    history: list[AgentBotHistoryItem] = Field(default_factory=list, max_length=1000)
    refresh: bool = False


class AgentBotCatalyst(BaseModel):
    available: bool = False
    earningsDate: Optional[str] = None
    exDividendDate: Optional[str] = None
    targetMeanPrice: Optional[float] = None
    analystRecommendation: Optional[str] = None
    analystCount: Optional[int] = None
    revenueGrowth: Optional[float] = None
    earningsGrowth: Optional[float] = None
    epsGrowth: Optional[float] = None
    dividendYield: Optional[float] = None
    revisionNotes: list[str] = Field(default_factory=list)


class AgentBotRecommendation(BaseModel):
    rank: int
    ticker: str
    companyName: str
    sector: str
    action: str
    confidence: float
    alphaScore: float
    riskAdjustedScore: float
    expectedReturn20d: float
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
    optionCategory: Optional[str] = None
    optionDte: Optional[int] = None
    optionIvProxy: Optional[float] = None
    optionIvGate: Optional[str] = None
    optionRules: list[str] = Field(default_factory=list)
    rationale: str
    whyNow: str
    signals: list[AlphaSignal]
    backtests: list[AlphaBacktestMetric]
    catalyst: AgentBotCatalyst = Field(default_factory=AgentBotCatalyst)


class AgentBotTracking(BaseModel):
    id: Optional[str] = None
    ticker: str
    companyName: str
    action: str
    entry: float
    currentPrice: float
    priceDate: date
    stop: Optional[float] = None
    target1: Optional[float] = None
    target2: Optional[float] = None
    unrealizedReturnPct: Optional[float] = None
    alphaScore: float
    whyNow: str = ""


class AgentBotAlert(BaseModel):
    ticker: str
    type: str
    severity: str
    message: str


class AgentBotOutcome(BaseModel):
    id: Optional[str] = None
    ticker: str
    action: str
    entryPrice: float
    currentPrice: float
    returnPct: float
    recommendedAt: Optional[str] = None
    status: str


class AgentBotForwardEntry(BaseModel):
    id: Optional[str] = None
    ticker: str
    action: str
    entryPrice: float
    recommendedAt: str
    closed: bool
    forwardReturns: dict[str, Optional[float]]


class AgentBotForwardAggregate(BaseModel):
    count: int
    avgReturn: Optional[float] = None


class AgentBotForwardJournal(BaseModel):
    entries: list[AgentBotForwardEntry]
    aggregates: dict[str, AgentBotForwardAggregate]


class AgentBotBriefing(BaseModel):
    summary: str
    regime: str
    riskLevel: str
    topBuy: Optional[AgentBotRecommendation] = None
    topSell: Optional[AgentBotRecommendation] = None
    topWatch: Optional[AgentBotRecommendation] = None
    topAvoid: Optional[AgentBotRecommendation] = None
    counts: dict[str, int]


class AgentBotRunResponse(BaseModel):
    asOf: datetime
    mode: str
    briefing: AgentBotBriefing
    recommendations: list[AgentBotRecommendation]
    activeTracking: list[AgentBotTracking]
    alerts: list[AgentBotAlert]
    outcomes: list[AgentBotOutcome]
    forwardJournal: AgentBotForwardJournal
    catalysts: dict[str, AgentBotCatalyst]
    meta: dict[str, Any]


class InstitutionalBacktest(BaseModel):
    winRate: Optional[float] = None
    avgReturn: Optional[float] = None
    medianReturn: Optional[float] = None
    benchmarkAvgReturn: Optional[float] = None
    alphaAvgReturn: Optional[float] = None
    maxDrawdown: Optional[float] = None
    sampleSize: int
    valid: bool


class InstitutionalSimulationScenario(BaseModel):
    winRate: float
    avgReturn: float
    survives: bool


class InstitutionalSimulation(BaseModel):
    scenarios: dict[str, InstitutionalSimulationScenario]
    allScenariosSurvive: bool


class InstitutionalConfidence(BaseModel):
    confidence: float
    sampleSize: int
    trustworthy: bool
    reason: str
    calibrationDetails: Optional[dict[str, Any]] = None


class InstitutionalGateDelta(BaseModel):
    actual: Any
    required: Any
    delta: Optional[Any] = None
    pass_: bool = Field(..., alias="pass")
    failedScenarios: Optional[list[str]] = None


class InstitutionalTradeGate(BaseModel):
    decision: str = Field(..., description="'TAKE' or 'PASS'")
    reasons: list[str]
    gateConditions: dict[str, bool]
    gateDeltas: dict[str, InstitutionalGateDelta]
    numFailures: int
    watchTier: bool = Field(..., description="True if failing on exactly one dimension")


class InstitutionalConvexityAlert(BaseModel):
    ticker: str
    type: str
    probability: float
    expectedReturn: str
    requiredStockMove: float
    currentPrice: float
    volatility: float
    alphaScore: float
    message: str


class InstitutionalCandidate(BaseModel):
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
    tradePlan: AlphaTradePlan
    signals: list[AlphaSignal]
    backtests: list[AlphaBacktestMetric]
    backtest: InstitutionalBacktest
    simulation: InstitutionalSimulation
    confidence: InstitutionalConfidence
    tradeGate: InstitutionalTradeGate
    convexityAlert: Optional[InstitutionalConvexityAlert] = None


class InstitutionalScannerResponse(BaseModel):
    asOf: datetime
    marketRegime: dict[str, Any]
    candidates: list[InstitutionalCandidate]
    convexityAlerts: list[InstitutionalConvexityAlert]
    meta: dict[str, Any]
