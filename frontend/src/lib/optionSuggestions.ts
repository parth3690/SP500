/**
 * Option-picking suggestions based on daily/weekly RSI (oversold vs overbought).
 * Aims to suggest strike and expiry where probability of profit is higher (mean reversion).
 */

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

function formatStrike(price: number, mult: number): string {
  const s = price * mult;
  if (s >= 1000) return s.toFixed(0);
  if (s >= 100) return s.toFixed(1);
  if (s >= 10) return s.toFixed(2);
  return s.toFixed(2);
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
