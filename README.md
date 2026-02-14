# S&P 500 Monthly Movers Analyzer

Full-stack web app that:

- Retrieves current S&P 500 constituents dynamically (Wikipedia).
- Pulls historical daily close prices (Yahoo Finance via `yfinance`).
- Calculates percent change over a selected date range (default: last 30 calendar days).
- Displays top gainers/losers with filtering, search, sorting, CSV export, charts, and a heatmap.

## Repo layout

- `backend/`: FastAPI + Pandas + yfinance API
- `frontend/`: Next.js + Tailwind dashboard UI

## How data is retrieved

- Constituents: scraped from Wikipedia’s “List of S&P 500 companies”.
- Ticker normalization: share classes like `BRK.B` are converted to Yahoo format (`BRK-B`) for pricing calls.
- Prices: fetched from Yahoo Finance via `yfinance` in batched downloads.

## How calculations work

For each ticker:

- Fetch daily close prices for the selected range **with a small buffer**.
- `pastPrice` = last available close **on or before** `start`
- `currentPrice` = last available close **on or before** `end`
- `% Change = (currentPrice / pastPrice - 1) * 100`

This matches the “30 calendar days” requirement while handling weekends/market holidays gracefully.

## Backend (FastAPI)

### Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Environment variables

Create `backend/.env` (optional):

- `ALLOWED_ORIGINS` (default `*`): comma-separated origins (e.g. `https://your-vercel-app.vercel.app`)
- `DEFAULT_RANGE_DAYS` (default `30`)
- `MAX_RANGE_DAYS` (default `366`)
- `CONSTITUENTS_TTL_SECONDS` (default `86400`)
- `MOVERS_TTL_SECONDS` (default `900`)

### API endpoints

- `GET /health`
- `GET /api/constituents?refresh=false`
- `GET /api/movers?start=YYYY-MM-DD&end=YYYY-MM-DD&limit=50&includeAll=true&refresh=false`
- `GET /api/movers.csv?start=YYYY-MM-DD&end=YYYY-MM-DD`

## Frontend (Next.js)

### Setup

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

### Environment variables

Frontend uses:

- `NEXT_PUBLIC_API_BASE_URL` (example: `http://localhost:8000`)

## Deployment

### Frontend (Vercel)

- Project root: `frontend/`
- Build command: `npm run build`
- Output: Next.js default
- Set env var `NEXT_PUBLIC_API_BASE_URL` to your backend URL.

### Backend (Render / Heroku)

Render:

- Root directory: `backend/`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Set `ALLOWED_ORIGINS` to your Vercel domain.

Heroku:

- Deploy `backend/` and use the included `Procfile`.
- Set config vars (`ALLOWED_ORIGINS`, etc.).

## Notes on performance

- Constituents are cached in-memory for 24h by default.
- Movers results are cached in-memory per (start,end) for 15 minutes by default.
- Price downloads are batched (chunked) to reduce throttling risk.
