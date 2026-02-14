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

