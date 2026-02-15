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
    weeklyRSI: float = Field(..., description="14-period weekly RSI value")
    dailyRSI: Optional[float] = Field(None, description="14-period daily RSI for reference")


class OversoldResponse(BaseModel):
    asOf: datetime
    rsiThreshold: float
    stocks: list[OversoldRow]
    meta: dict[str, Any]

