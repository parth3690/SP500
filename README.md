# S&P 500 Analyzer

Full-stack web app for S&P 500 analysis featuring real-time market data, technical indicators, quantitative trading strategies, and deep research tools.

## Features

- **Dashboard**
  - Top gainers/losers with filtering, search, sorting, trailing/forward P/E, CSV export, charts, and a full heatmap
  - LEAPS radar: weekly/daily oversold & overbought lists with compact option suggestions for each ticker
- **Institutional-Grade Scanner** 🆕
  - Walk-forward backtests with no lookahead bias (20-day forward returns)
  - Simulation validation under bull, base, bear, and high-volatility scenarios with transaction costs
  - Calibrated confidence estimates from backtest performance and simulation survival
  - Hard trade gate: only TAKE when confidence ≥ 75%, win rate ≥ 62%, sample ≥ 20, alpha ≥ 3% vs benchmark
  - Rooftop alert for high-convexity option opportunities (10%+ probability of 100x return)
  - TAKE/PASS decision framework that rejects trades a skeptical desk wouldn't size
- **Alpha Candidates**
  - Unified ranked alpha score across momentum, relative strength, trend, risk, factor exposure, and market regime
  - Lightweight signal backtests, SPY/sector relative strength, risk controls, catalyst/revision proxies, and a local watchlist journal
  - Multiple custom watchlists with up to 100 tickers each, scanned through the same alpha-candidate view
- **Golden/Death Cross Detection**
  - Highlights stocks where the 50-DMA and 200-DMA are converging or crossing, with a dedicated crossovers table
- **Deep Research Page (any ticker, not just S&P 500)**
  - Interactive candlestick charts (Plotly.js) with Bollinger Bands, Fibonacci retracements
  - Moving Averages (50-DMA, 200-DMA), RSI, MACD and crossover status
  - Editable date range (1M / 3M / 6M / 1Y / 2Y / 5Y presets or custom)
  - 9 Quantitative Trading Strategies with BUY/SELL/NEUTRAL signals and plain-English reasoning
  - Fundamental data (P/E ratios, market cap, beta, 52-week range)
  - **Option suggestion card (RSI-based or factor-based)** – strategy, strike, expiry and rationale you can backtest
  - **LEAPS suggestion card** – long-dated (12–18 month) call/put ideas using RSI when available, otherwise factor-based
  - **Quantitative Stock Screening (6-formula checklist)** – Bayes, GBM, Itô, Black‑Scholes, Markowitz, Monte Carlo with pass/fail/review and a suggested next move

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.9+, FastAPI, Pandas, yfinance |
| Frontend | Next.js 14, React 18, TypeScript, Tailwind CSS |
| Charts | Plotly.js (CDN), Recharts |
| Caching | In-memory TTL caches (cachetools) |

## Repo Layout

```
backend/          FastAPI API server
  app/
    main.py       Routes & endpoints
    models.py     Pydantic models
    services/     Business logic (movers, crossovers, research, prices, caching)
frontend/         Next.js dashboard
  src/
    app/          Pages (dashboard, research/[ticker])
    components/   UI components (tables, heatmap, charts)
    lib/
      api.ts      Typed API client & fetch helpers
      types.ts    Shared response types (mirrors backend models)
      date.ts     Date utilities for ISO ranges and presets
      format.ts   Centralized numeric/price/percent formatting
      screening.ts  Shared GBM/Monte Carlo volatility pipeline
      optionSuggestions.ts  RSI + factor-based option/LEAPS logic
```

## Quantitative Trading Strategies

| # | Strategy | What It Does |
|---|----------|-------------|
| 1 | Time-Series Trend Following | SMA 20/50 crossover + MACD histogram |
| 2 | Multi-Factor Equity Model | Momentum + volume quality + volatility composite |
| 3 | Cross-Sectional Momentum | RSI + weighted multi-timeframe returns |
| 4 | Statistical Arbitrage | Mean reversion via z-score from 20-day SMA |
| 5 | Bollinger Band Squeeze | Volatility contraction + breakout direction |
| 6 | Stochastic Oscillator | %K/%D overbought/oversold crossovers |
| 7 | ADX Trend Strength | Trend strength (ADX) + direction (+DI/-DI) |
| 8 | OBV Volume Trend | Confirms price trends via cumulative volume flow |
| 9 | Machine Learning Alpha | AI-driven composite of 5 weighted features |

Each strategy returns a **BUY / SELL / NEUTRAL** signal with a confidence score and a detailed plain-English explanation of *why*.

---

## Local Development

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local   # sets NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Environment Variables

**Backend** (`backend/.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `DEFAULT_RANGE_DAYS` | `30` | Default date range for movers |
| `MAX_RANGE_DAYS` | `366` | Maximum allowed date range |
| `CONSTITUENTS_TTL_SECONDS` | `86400` | Constituents cache TTL (24h) |
| `MOVERS_TTL_SECONDS` | `900` | Movers cache TTL (15m) |
| `FMP_API_KEY` | _(empty)_ | [Financial Modeling Prep](https://site.financialmodelingprep.com/developer/docs/dashboard) API key; when set, research header uses **live quote** from FMP while charts stay on Yahoo daily OHLCV |
| `FMP_API_BASE` | `https://financialmodelingprep.com/stable` | Override FMP base URL if needed |

**Frontend** (`frontend/.env.local`):

| Variable | Example | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_BASE_URL` | `http://localhost:8000` | Backend API URL |

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/constituents` | S&P 500 constituent list |
| `GET` | `/api/movers` | Top gainers/losers with `?start=&end=&limit=&includeAll=` |
| `GET` | `/api/movers.csv` | CSV export of movers |
| `GET` | `/api/institutional-scanner` | Institutional-grade scanner with walk-forward backtests, simulation validation, confidence, and trade gate |
| `GET` | `/api/alpha-candidates` | Ranked alpha candidates with filters for score, sector, beta, risk mode, and market regime |
| `POST` | `/api/alpha-watchlist` | Ranked alpha candidates for a supplied watchlist of up to 100 tickers |
| `GET` | `/api/crossovers` | Golden/death cross detection with `?threshold=` |
| `GET` | `/api/research/{ticker}` | Deep research with `?start=&end=` date range |

---

## Free Deployment Guide

Deploy the entire app for free using **Render** (backend) + **Vercel** (frontend).

### Step 1: Deploy Backend on Render

1. Sign up at [render.com](https://render.com) with your GitHub account
2. Click **New +** > **Web Service**
3. Connect the repository and configure:

| Setting | Value |
|---------|-------|
| Name | `sp500-api` |
| Root Directory | `backend` |
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Instance Type | **Free** |

4. Add environment variables:

| Variable | Value |
|----------|-------|
| `ALLOWED_ORIGINS` | `https://your-app.vercel.app` (update after Step 2) |
| `PYTHON_VERSION` | `3.11` |

5. Click **Create Web Service** -- you'll get a URL like `https://sp500-api.onrender.com`

> **Note:** Render free tier spins down after 15 min of inactivity. First request after idle takes ~30-60s to cold-start.

### Step 2: Deploy Frontend on Vercel

1. Sign up at [vercel.com](https://vercel.com) with your GitHub account
2. Click **Add New Project** > import the repository
3. Configure:

| Setting | Value |
|---------|-------|
| Framework Preset | Next.js (auto-detected) |
| Root Directory | `frontend` |

4. Add environment variable:

| Variable | Value |
|----------|-------|
| `NEXT_PUBLIC_API_BASE_URL` | `https://sp500-api.onrender.com` (your Render URL) |

5. Click **Deploy** -- you'll get a URL like `https://sp500-analyzer.vercel.app`

### Step 3: Update CORS

Go back to Render > your service > **Environment** > update `ALLOWED_ORIGINS` to your actual Vercel URL. Render will auto-redeploy.

### Free Tier Summary

| Service | Hosts | Free Limits |
|---------|-------|-------------|
| **Vercel** | Next.js frontend | 100 GB bandwidth/mo, unlimited deploys |
| **Render** | FastAPI backend | 750 hrs/mo, sleeps after 15 min idle |

Both auto-deploy on every `git push` to main.

---

## Institutional-Grade Scanner Philosophy

The Institutional Scanner is designed to behave like a top-tier quantitative analyst: skeptical, rigorous, and conservative. It operates under the principle that **no trade is better than a bad trade**.

### How It Works

1. **Walk-Forward Backtests**
   - Uses only data available at signal time (no lookahead bias)
   - Measures 20-day forward returns from historical signals
   - Computes win rate, average return, alpha vs SPY, and max drawdown
   - Requires minimum 20 samples for statistical significance

2. **Simulation Validation**
   - Tests candidates under four scenarios: bull, base, bear, high-volatility
   - Includes realistic transaction costs (20 bps round-trip)
   - Adds slippage for realistic fills (5 bps adverse)
   - Edge must survive all scenarios to pass

3. **Calibrated Confidence**
   - Computed from backtest performance, simulation survival, alpha score, and risk score
   - Penalized for small sample sizes or failed simulation scenarios
   - Trustworthiness flag indicates whether confidence is reliable

4. **Hard Trade Gate**
   - Only emit TAKE when ALL conditions pass:
     - Confidence ≥ 75%
     - Win rate ≥ 62%
     - Sample size ≥ 20
     - Alpha vs benchmark ≥ 3%
     - All simulation scenarios survive
   - Otherwise PASS (most candidates will PASS)

5. **High-Convexity Options Alert**
   - Detects far-OTM option setups with ≥10% probability of 100x return
   - Based on stock volatility, momentum, and technical setup
   - Only alerts when both probability threshold AND strong alpha score (≥70) are met
   - Rooftop shout (🚨) in UI when detected

### Design Principles

- **Conservative by default**: Prefers fewer, better trades over volume
- **No fabricated numbers**: All metrics computed from actual data; missing data degrades honestly
- **Transparent reasoning**: Every PASS decision includes specific reasons
- **Walk-forward validation**: Historical analysis prevents overfitting
- **Real-world costs**: Transaction costs and slippage are always included

**Disclaimer**: This is research and screening software for educational purposes, not investment advice.

---

## NYSE SMID Universe ($100M-$2B)

The NYSE SMID Agent extends the institutional scanner to small-mid cap stocks listed on the NYSE.

### Universe Definition

- **Exchange**: NYSE only (not Nasdaq, not AMEX)
- **Market Cap**: Greater than $100M and less than $2B
- **Exclusions**: ETFs, funds, ADRs, preferreds, warrants
- **Liquidity Filter**: Minimum $2.00 price
- **Data Source**: Financial Modeling Prep (requires `FMP_API_KEY`)

### How It Works

The NYSE SMID Agent **reuses the existing S&P 500 data pipeline**:

1. **Constituents**: Fetched from FMP stock screener with market cap and exchange filters
2. **Prices**: Same `fetch_close_prices` function (Yahoo Finance + FMP fallback)
3. **Alpha Engine**: Same `compute_alpha_candidates` scoring model
4. **Institutional Gate**: Same walk-forward backtest + simulation + confidence framework

### Usage

**API**: 
```bash
# Scan entire NYSE SMID universe
GET /api/nyse-smid-agent?limit=20&minScore=65

# Scan specific tickers within the universe (example: $100M-$2B market cap NYSE stocks)
GET /api/nyse-smid-agent?tickers=TICKER1,TICKER2,TICKER3&limit=20
```

**Frontend**: Navigate to "NYSE SMID Agent" tab

**Requirements**: Set `FMP_API_KEY` in environment for NYSE listings + market cap data

### Performance

- Universe is larger than S&P 500 (~800-1200 stocks vs 500)
- Min alpha score pre-filters before expensive walk-forward backtests
- Price data cached with 15-minute TTL
- Constituents cached with 24-hour TTL

---

## How Data & Signals Work

- **Constituents**: Scraped from Wikipedia's "List of S&P 500 companies"
- **Ticker normalization**: Share classes like `BRK.B` are converted to Yahoo format (`BRK-B`)
- **Prices**: Historical OHLCV from Yahoo Finance via `yfinance`; optional **live** header price/change/volume from [Financial Modeling Prep](https://site.financialmodelingprep.com/developer/docs) when `FMP_API_KEY` is set
- **Caching**: Constituents cached 24h, movers cached 15m, research cached per ticker+date range
- **Indicators**:
  - All standard indicators (SMA, EMA, RSI, MACD, Bollinger, Fibonacci, Stochastic, ADX, OBV) are computed server-side
  - RSI for research and RSI scans share the same Wilder formula via a common `indicators.py` helper for consistency
- **Quantitative Stock Screening**:
  - Uses research OHLCV data to compute GBM and Monte Carlo once, then reuses that pipeline for:
    - The 6-formula screening table (Bayes, GBM, Itô, Black‑Scholes, Markowitz, Monte Carlo)
    - Factor-based option and LEAPS suggestions when RSI is missing or neutral
  - Screening results also output a **Results summary** and a **Suggested next move** (bullish / bearish / review)
- **Option & LEAPS Suggestions**:
  - **RSI-based suggestions**: Daily/weekly RSI drives bullish (oversold) vs bearish (overbought) strategies, strike range, and expiry
  - **Factor-based suggestions**: When RSI is missing/neutral, the app uses crossover signal, GBM, Monte Carlo, 52‑week range, and beta
  - Both near-term and LEAPS suggestions include a **Backtest key** (e.g. `crossover:golden_cross · gbm:bullish · 52w:near_low`) so you can filter and analyze results offline

## Performance Notes

- Constituents are cached in-memory for 24h by default
- Movers results are cached per (start, end) for 15 minutes
- Research results are cached per (ticker, start, end) with configurable TTL
- Price downloads are batched to reduce Yahoo Finance throttling risk
- GBM and Monte Carlo volatility signals are computed once per research payload and reused across the screening table and option/LEAPS suggestion logic
