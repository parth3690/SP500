# Project Review: Alpha Market Scanner

## Current Product Shape

The project is a full-stack market scanner with three live app areas:

- S&P 500 dashboard: movers, sector summary, heatmap, RSI oversold/overbought radar, crossovers, CSV export.
- Market conditions: macro/sentiment/valuation peak checklist with live and manual inputs.
- Multibagger scanner: ticker-level fundamental checklist with optional deeper historical scan.

The product is useful as a dashboard, but it is not yet a true alpha-discovery engine. Most current features describe conditions after they happen. The next layer should rank ideas, test whether signals worked historically, and track forward outcomes.

## Cleanup Completed

- Removed local generated clutter: `.DS_Store` and `frontend/.next`.
- Consolidated duplicated RSI scanner logic in `backend/app/services/rsi_scan.py`.
- Kept the existing public RSI functions unchanged so current API routes still work.
- Preserved existing uncommitted changes in the market-condition files.

## Redundant Or Low-Value Features

1. Standalone HTML prototypes

   Files such as `sp500-standalone.html`, `quantelite-standalone.html`, `market-conditions.html`, and `frontend/leaps-evaluator.html` overlap with the Next/FastAPI app. Keep them only as references, or archive/remove them once the app versions are confirmed complete.

2. Separate daily and weekly RSI panels

   Daily and weekly oversold/overbought lists are useful, but the current UI treats them as separate sections. A combined RSI radar with timeframe filters would reduce noise and make candidates easier to compare.

3. Overbought LEAPS suggestions

   LEAPS are usually most useful for long-duration directional conviction. Overbought put ideas may belong in a bearish/options module, not mixed into a multibagger/alpha discovery flow.

4. Too many research-page strategy signals

   The research page lists many technical strategies. Without historical hit rate, drawdown, average forward return, or regime context, multiple BUY/SELL/NEUTRAL labels can look precise without proving edge.

5. Moving average crossovers as a standalone alpha signal

   Golden/death crosses are common and lagging. They are better as supporting trend/context features inside a ranked model rather than a primary discovery feature.

6. Market-peak checklist mixed with stock discovery

   The market-conditions page is valuable for regime awareness, but it is not directly connected to stock ranking, sizing, or signal gating yet. Until connected, it functions as a separate dashboard.

7. `/api/move-finder` endpoint without frontend

   The backend has an explosive move finder endpoint, but no visible UI uses it. Either add a scanner tab or remove/park the endpoint.

8. CLI scanner scripts outside the app flow

   `backend/us_multibagger_scanner.py` and `backend/market_peak_tracker.py` are currently untracked and overlap with app services. Fold useful logic into app services or archive them as experiments.

## Features Needed To Find Market Alpha

1. Ranked alpha score

   Create one scoring model that combines momentum, relative strength, volume acceleration, fundamentals, valuation, quality, market regime, and event catalysts. The dashboard should answer: "What are the top ideas today, and why?"

2. Backtesting for every signal

   Each signal should show forward 5-day, 20-day, 60-day, and 120-day returns, win rate, max drawdown, and benchmark-relative return. Without this, the app finds activity, not alpha.

3. Relative strength versus benchmark and sector

   Add stock vs SPY, stock vs sector ETF, and sector vs SPY strength. Alpha usually comes from relative outperformance, not absolute price movement alone.

4. Factor exposure controls

   Separate real stock-specific alpha from broad factor beta: market beta, size, value, quality, momentum, volatility, and sector exposure.

5. Catalyst tracking

   Add earnings dates, guidance changes, analyst revisions, insider buying, buybacks, unusual volume, short interest changes, and news/sentiment summaries. Price signals are stronger when paired with catalysts.

6. Earnings and revision momentum

   Add revenue/EPS estimate revisions, earnings surprise history, margin trend, and forward growth acceleration. This is especially important for multibagger discovery.

7. Volume and liquidity quality

   Add relative volume, dollar volume, spread/liquidity filters, accumulation/distribution, and institutional ownership changes. Big moves without liquidity can be traps.

8. Risk-adjusted ranking

   Rank ideas by expected return divided by risk: volatility, beta, drawdown, ATR stop distance, liquidity risk, and event risk.

9. Regime-aware filters

   Connect the market-condition page to the stock scanner. Example: in risk-off regimes, prefer quality/low leverage/relative strength; in risk-on regimes, allow high-growth momentum and smaller caps.

10. Portfolio/watchlist workflow

   Add saved watchlists, candidate status, entry price, stop level, thesis, invalidation rule, and follow-up reminders. Alpha discovery needs tracking after the first signal.

11. Forward performance journal

   Automatically track every generated idea after 1 week, 1 month, 3 months, and 6 months. This creates feedback loops and reveals which signals are actually useful.

12. Explainable "why now" summary

   For each candidate, show the few reasons that matter most: trend, relative strength, valuation, growth, catalyst, risk, and regime alignment.

## Suggested Next Build Order

1. Add a unified "Alpha Candidates" table.
2. Create a first alpha score from current available data.
3. Add benchmark/sector relative strength.
4. Add simple forward-return backtests.
5. Connect market regime to ranking filters.
6. Add watchlist and performance journal.

