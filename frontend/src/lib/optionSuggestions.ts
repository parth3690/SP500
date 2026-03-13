/**
 * Option-picking suggestions based on daily/weekly RSI (oversold vs overbought).
 * Aims to suggest strike and expiry where probability of profit is higher (mean reversion).
 */

import { formatStrike } from "@/lib/format";

export type OptionSuggestionContext = "weekly_oversold" | "daily_oversold" | "weekly_overbought" | "daily_overbought";

export type OptionSuggestion = {
  strategy: string;
  strikeSuggestion: string;
  expirySuggestion: string;
  rationale: string;
  /** Approximate strike price if we have currentPrice (e.g. 0.95 = 5% OTM) */
  strikeMultiplier?: number;
};

/**
 * Get ideal option suggestion for a stock given RSI and context (oversold/overbought, weekly/daily).
 * Uses mean-reversion logic: oversold → bullish strategies; overbought → bearish strategies.
 */
export function getOptionSuggestion(
  context: OptionSuggestionContext,
  rsi: number | null,
  currentPrice: number
): OptionSuggestion | null {
  if (rsi == null) return null;

  const isOversold = context === "weekly_oversold" || context === "daily_oversold";
  const isWeekly = context === "weekly_oversold" || context === "weekly_overbought";

  if (isOversold) {
    if (rsi > 35) return null; // Not oversold enough
    // Deep oversold: stronger bounce signal → closer to ATM
    if (rsi <= 22) {
      return {
        strategy: "Buy calls",
        strikeSuggestion: `ATM or 2–5% OTM (≈ $${formatStrike(currentPrice, 1.02)}–$${formatStrike(currentPrice, 1.05)})`,
        expirySuggestion: isWeekly ? "45–90 DTE or LEAPS 12–18 mo" : "30–45 DTE",
        rationale: "Deep oversold: mean reversion bounce likely. Shorter DTE for daily, LEAPS if weekly confirms.",
        strikeMultiplier: 1.03,
      };
    }
    if (rsi <= 28) {
      return {
        strategy: "Buy calls or sell cash-secured puts",
        strikeSuggestion: `5–10% OTM (≈ $${formatStrike(currentPrice, 1.05)}–$${formatStrike(currentPrice, 1.10)})`,
        expirySuggestion: isWeekly ? "60–90 DTE or LEAPS 9–15 mo" : "30–60 DTE",
        rationale: "Oversold bounce setup. OTM calls capture upside; CSPs reduce cost basis if assigned.",
        strikeMultiplier: 1.075,
      };
    }
    return {
      strategy: "Buy calls",
      strikeSuggestion: `5–10% OTM (≈ $${formatStrike(currentPrice, 1.05)}–$${formatStrike(currentPrice, 1.10)})`,
      expirySuggestion: isWeekly ? "90 DTE–LEAPS" : "45–60 DTE",
      rationale: "Moderate oversold. Favor 45–60 DTE for daily; extend to LEAPS if weekly also oversold.",
      strikeMultiplier: 1.08,
    };
  }

  // Overbought
  if (rsi < 65) return null;
  if (rsi >= 82) {
    return {
      strategy: "Buy puts",
      strikeSuggestion: `5–10% OTM puts (≈ $${formatStrike(currentPrice, 0.90)}–$${formatStrike(currentPrice, 0.95)})`,
      expirySuggestion: isWeekly ? "45–90 DTE or LEAPS" : "30–45 DTE",
      rationale: "Extreme overbought: pullback likely. OTM puts for cost efficiency.",
      strikeMultiplier: 0.92,
    };
  }
  if (rsi >= 75) {
    return {
      strategy: "Buy puts or sell covered calls",
      strikeSuggestion: `10% OTM puts (≈ $${formatStrike(currentPrice, 0.90)})`,
      expirySuggestion: isWeekly ? "60–90 DTE" : "30–45 DTE",
      rationale: "Overbought: mean reversion pullback. Puts for downside; CC if holding.",
      strikeMultiplier: 0.90,
    };
  }
  return {
    strategy: "Consider puts or reduce exposure",
    strikeSuggestion: `10–15% OTM puts (≈ $${formatStrike(currentPrice, 0.85)}–$${formatStrike(currentPrice, 0.90)})`,
    expirySuggestion: isWeekly ? "90 DTE" : "30–45 DTE",
    rationale: "Moderate overbought. Shorter DTE puts for pullback; avoid long-dated unless weekly confirms.",
    strikeMultiplier: 0.88,
  };
}

/**
 * One-line summary for use in compact lists (e.g. LEAPS radar).
 */
export function getOptionSuggestionShort(
  context: OptionSuggestionContext,
  rsi: number | null,
  currentPrice: number
): string {
  const s = getOptionSuggestion(context, rsi, currentPrice);
  if (!s) return "";
  return `${s.strategy} · ${s.strikeSuggestion.split(" (")[0]} · ${s.expirySuggestion}`;
}

/** LEAPS = long-dated options (12–18 mo). Use same RSI logic but expiry fixed to LEAPS. */
export function getLeapsSuggestion(
  rsi: number | null,
  currentPrice: number
): OptionSuggestion | null {
  if (rsi == null) return null;
  const isOversold = rsi <= 35;
  const isOverbought = rsi >= 65;
  if (!isOversold && !isOverbought) return null;

  if (isOversold) {
    if (rsi <= 22) {
      return {
        strategy: "LEAPS calls",
        strikeSuggestion: `ATM or 2–5% OTM (≈ $${formatStrike(currentPrice, 1.02)}–$${formatStrike(currentPrice, 1.05)})`,
        expirySuggestion: "12–18 months (e.g. Jan 2026 – Jun 2026)",
        rationale: "Deep oversold: time for thesis to play out. LEAPS give 360+ DTE for mean reversion and qualify for long-term cap gains.",
        strikeMultiplier: 1.03,
      };
    }
    return {
      strategy: "LEAPS calls or CSPs",
      strikeSuggestion: `5–10% OTM (≈ $${formatStrike(currentPrice, 1.05)}–$${formatStrike(currentPrice, 1.10)})`,
      expirySuggestion: "12–18 months",
      rationale: "Oversold setup. LEAPS reduce theta drag; size for conviction and run 9-condition checklist.",
      strikeMultiplier: 1.075,
    };
  }

  return {
    strategy: "LEAPS puts (hedge) or reduce",
    strikeSuggestion: `10% OTM puts (≈ $${formatStrike(currentPrice, 0.90)})`,
    expirySuggestion: "12–18 months if hedging",
    rationale: "Overbought: LEAPS puts for long-dated hedge. Prefer shorter DTE for pure pullback plays.",
    strikeMultiplier: 0.90,
  };
}

// ─── Factor-based suggestions (when RSI missing or neutral) ───────────────────

/** Inputs derived from research page: crossover, GBM, Monte Carlo, 52w, beta. */
export type FactorInputs = {
  currentPrice: number;
  crossoverSignal: string;
  gbmBullish: boolean;
  gbmBearish: boolean;
  mcBullish: boolean;
  mcBearish: boolean;
  fiftyTwoWeekPct: number | null;
  beta: number | null;
};

export type FactorBasedSuggestion = OptionSuggestion & {
  /** Backtest key: filter or segment by these factors. */
  factorsUsed: string[];
};

/**
 * Option suggestion from crossover, GBM, Monte Carlo, 52w range, beta (no RSI).
 * Returns strategy + strike + expiry + rationale and factorsUsed for backtesting.
 */
export function getFactorBasedSuggestion(inputs: FactorInputs): FactorBasedSuggestion | null {
  const factors: string[] = [];
  let bullish = 0;
  let bearish = 0;

  if (inputs.crossoverSignal.includes("golden")) {
    factors.push("crossover:golden_cross");
    bullish++;
  } else if (inputs.crossoverSignal.includes("death")) {
    factors.push("crossover:death_cross");
    bearish++;
  }

  if (inputs.gbmBullish) {
    factors.push("gbm:bullish");
    bullish++;
  }
  if (inputs.gbmBearish) {
    factors.push("gbm:bearish");
    bearish++;
  }
  if (inputs.mcBullish) {
    factors.push("mc:bullish");
    bullish++;
  }
  if (inputs.mcBearish) {
    factors.push("mc:bearish");
    bearish++;
  }

  const pct = inputs.fiftyTwoWeekPct;
  if (pct != null) {
    if (pct < 25) {
      factors.push("52w:near_low");
      bullish++;
    } else if (pct > 75) {
      factors.push("52w:near_high");
      bearish++;
    }
  }

  if (bullish === 0 && bearish === 0) return null;

  const price = inputs.currentPrice;
  const highBeta = inputs.beta != null && inputs.beta > 1.3;
  const expiryShort = highBeta ? "45–60 DTE" : "60–90 DTE";
  const expiryLeaps = "12–18 months";

  if (bullish > bearish) {
    const strike = `5–10% OTM (≈ $${formatStrike(price, 1.05)}–$${formatStrike(price, 1.10)})`;
    return {
      strategy: "Buy calls or sell cash-secured puts",
      strikeSuggestion: strike,
      expirySuggestion: expiryShort,
      rationale: `Factor-based: ${factors.join(", ")}. No RSI signal; use for backtest. Favor calls or CSPs; size small until RSI confirms.`,
      strikeMultiplier: 1.075,
      factorsUsed: factors,
    };
  }

  if (bearish > bullish) {
    const strike = `5–10% OTM puts (≈ $${formatStrike(price, 0.90)}–$${formatStrike(price, 0.95)})`;
    return {
      strategy: "Buy puts or reduce exposure",
      strikeSuggestion: strike,
      expirySuggestion: expiryShort,
      rationale: `Factor-based: ${factors.join(", ")}. No RSI signal; use for backtest. Puts for hedge or pullback; avoid long-dated calls.`,
      strikeMultiplier: 0.92,
      factorsUsed: factors,
    };
  }

  // tie: mixed
  const strike = `ATM to 5% OTM (≈ $${formatStrike(price, 1.0)}–$${formatStrike(price, 1.05)})`;
  return {
    strategy: "Consider small call or put (direction from catalyst)",
    strikeSuggestion: strike,
    expirySuggestion: "45–60 DTE",
    rationale: `Factor-based mixed: ${factors.join(", ")}. Backtest both sides or wait for clearer RSI.`,
    strikeMultiplier: 1.02,
    factorsUsed: factors,
  };
}

/**
 * LEAPS version of factor-based suggestion (same logic, 12–18 mo expiry).
 */
export function getFactorBasedLeapsSuggestion(inputs: FactorInputs): FactorBasedSuggestion | null {
  const base = getFactorBasedSuggestion(inputs);
  if (!base) return null;
  return {
    ...base,
    expirySuggestion: "12–18 months (LEAPS)",
    rationale: base.rationale.replace(/Favor calls or CSPs; size small until RSI confirms\.?/i, "LEAPS reduce theta; run 9-condition checklist before entry.")
      .replace(/Puts for hedge or pullback; avoid long-dated calls\.?/i, "LEAPS puts for long-dated hedge.")
      .replace(/Backtest both sides or wait for clearer RSI\.?/i, "LEAPS only if catalyst view overrides mixed signals."),
    factorsUsed: base.factorsUsed,
  };
}
