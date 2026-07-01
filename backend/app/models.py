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

