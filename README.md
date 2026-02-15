# S&P 500 Analyzer

Full-stack web app for S&P 500 analysis featuring real-time market data, technical indicators, quantitative trading strategies, and deep research tools.

## Features

- **Dashboard** -- Top gainers/losers with filtering, search, sorting, CSV export, charts, and a full heatmap
- **Golden/Death Cross Detection** -- Highlights stocks where the 50-DMA and 200-DMA are converging
- **Deep Research Page** -- Click any ticker for comprehensive single-stock analysis:
  - Interactive candlestick charts (Plotly.js) with Bollinger Bands, Fibonacci retracements
  - Moving Averages (50-DMA, 200-DMA), RSI, MACD
  - Editable date range (1M / 3M / 6M / 1Y / 2Y / 5Y presets or custom)
  - 9 Quantitative Trading Strategies with BUY/SELL/NEUTRAL signals and plain-English reasoning
  - Fundamental data (P/E ratios, market cap, beta, 52-week range)

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
    lib/          API client, types
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

## How Data Works

- **Constituents**: Scraped from Wikipedia's "List of S&P 500 companies"
- **Ticker normalization**: Share classes like `BRK.B` are converted to Yahoo format (`BRK-B`)
- **Prices**: Fetched from Yahoo Finance via `yfinance` in batched downloads
- **Caching**: Constituents cached 24h, movers cached 15m, research cached per ticker+date range
- **Indicators**: All computed server-side (SMA, EMA, RSI, MACD, Bollinger, Fibonacci, Stochastic, ADX, OBV)

## Performance Notes

- Constituents are cached in-memory for 24h by default
- Movers results are cached per (start, end) for 15 minutes
- Research results are cached per (ticker, start, end) with configurable TTL
- Price downloads are batched to reduce Yahoo Finance throttling risk
