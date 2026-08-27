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
  optionCategory?: "Premium 30D" | "Directional" | "LEAPS";
  dte?: number;
  ivProxy?: number | null;
  ivGate?: "pass" | "below_50" | "not_applicable";
  rules?: string[];
  /** Approximate strike price if we have currentPrice (e.g. 0.95 = 5% OTM) */
  strikeMultiplier?: number;
};

export const PREMIUM_SELL_IV_MIN = 50;
export const PREMIUM_SELL_DTE = 30;

function ivGate(ivProxy?: number | null): "pass" | "below_50" | "not_applicable" {
  if (ivProxy == null || !Number.isFinite(ivProxy)) return "not_applicable";
  return ivProxy >= PREMIUM_SELL_IV_MIN ? "pass" : "below_50";
}

function withOptionRules<T extends OptionSuggestion>(
  suggestion: T,
  ivProxy: number | null | undefined,
  optionCategory: OptionSuggestion["optionCategory"],
  dte?: number,
): T {
  return {
    ...suggestion,
    optionCategory,
    dte,
    ivProxy: ivProxy ?? null,
    ivGate: optionCategory === "LEAPS" ? "not_applicable" : ivGate(ivProxy),
    rules: [
      "Use about 30 DTE for short-premium theta edge.",
      "Only sell option premium when IV or IV proxy is 50+.",
    ],
  };
}

function premiumSellingAllowed(ivProxy?: number | null): boolean {
  return ivGate(ivProxy) === "pass";
}

/**
 * Get ideal option suggestion for a stock given RSI and context (oversold/overbought, weekly/daily).
 * Uses mean-reversion logic: oversold → bullish strategies; overbought → bearish strategies.
 */
export function getOptionSuggestion(
  context: OptionSuggestionContext,
  rsi: number | null,
  currentPrice: number,
  ivProxy?: number | null,
): OptionSuggestion | null {
  if (rsi == null) return null;

  const isOversold = context === "weekly_oversold" || context === "daily_oversold";
  const isWeekly = context === "weekly_oversold" || context === "weekly_overbought";
  const canSellPremium = premiumSellingAllowed(ivProxy);

  if (isOversold) {
    if (rsi > 35) return null; // Not oversold enough
    if (canSellPremium && rsi <= 28) {
      return withOptionRules({
        strategy: "Sell cash-secured puts or bull put spread",
        strikeSuggestion: `Short put 5–10% OTM (≈ $${formatStrike(currentPrice, 0.90)}–$${formatStrike(currentPrice, 0.95)})`,
        expirySuggestion: "30 DTE short-premium window",
        rationale: `IV proxy is ${ivProxy?.toFixed(1)}%, above the 50+ gate. Premium is rich enough to favor 30 DTE put-selling/bull-put-spread structures instead of buying expensive calls.`,
        strikeMultiplier: 0.95,
      }, ivProxy, "Premium 30D", PREMIUM_SELL_DTE);
    }
    // Deep oversold: stronger bounce signal → closer to ATM
    if (rsi <= 22) {
      return withOptionRules({
        strategy: "Buy calls",
        strikeSuggestion: `ATM or 2–5% OTM (≈ $${formatStrike(currentPrice, 1.02)}–$${formatStrike(currentPrice, 1.05)})`,
        expirySuggestion: isWeekly ? "45–90 DTE or LEAPS 12–18 mo" : "30–45 DTE",
        rationale: "Deep oversold: mean reversion bounce likely. IV proxy is below the 50+ premium-selling gate, so avoid short-premium income trades.",
        strikeMultiplier: 1.03,
      }, ivProxy, "Directional", 45);
    }
    if (rsi <= 28) {
      return withOptionRules({
        strategy: "Buy calls",
        strikeSuggestion: `5–10% OTM (≈ $${formatStrike(currentPrice, 1.05)}–$${formatStrike(currentPrice, 1.10)})`,
        expirySuggestion: isWeekly ? "60–90 DTE or LEAPS 9–15 mo" : "30–60 DTE",
        rationale: "Oversold bounce setup. OTM calls capture upside; sell puts only if live IV/IV rank confirms 50+.",
        strikeMultiplier: 1.075,
      }, ivProxy, "Directional", 45);
    }
    return withOptionRules({
      strategy: "Buy calls",
      strikeSuggestion: `5–10% OTM (≈ $${formatStrike(currentPrice, 1.05)}–$${formatStrike(currentPrice, 1.10)})`,
      expirySuggestion: isWeekly ? "90 DTE–LEAPS" : "45–60 DTE",
      rationale: "Moderate oversold. Favor 45–60 DTE for daily; extend to LEAPS if weekly also oversold. Avoid short premium until IV is 50+.",
      strikeMultiplier: 1.08,
    }, ivProxy, "Directional", 60);
  }

  // Overbought
  if (rsi < 65) return null;
  if (canSellPremium && rsi >= 75) {
    return withOptionRules({
      strategy: "Sell covered calls or bear call spread",
      strikeSuggestion: `Short call 5–10% OTM (≈ $${formatStrike(currentPrice, 1.05)}–$${formatStrike(currentPrice, 1.10)})`,
      expirySuggestion: "30 DTE short-premium window",
      rationale: `IV proxy is ${ivProxy?.toFixed(1)}%, above the 50+ gate. Premium is rich enough to sell 30 DTE calls/call spreads for time-decay edge.`,
      strikeMultiplier: 1.05,
    }, ivProxy, "Premium 30D", PREMIUM_SELL_DTE);
  }
  if (rsi >= 82) {
    return withOptionRules({
      strategy: "Buy puts",
      strikeSuggestion: `5–10% OTM puts (≈ $${formatStrike(currentPrice, 0.90)}–$${formatStrike(currentPrice, 0.95)})`,
      expirySuggestion: isWeekly ? "45–90 DTE or LEAPS" : "30–45 DTE",
      rationale: "Extreme overbought: pullback likely. OTM puts for cost efficiency; avoid selling premium until IV is 50+.",
      strikeMultiplier: 0.92,
    }, ivProxy, "Directional", 45);
  }
  if (rsi >= 75) {
    return withOptionRules({
      strategy: "Buy puts",
      strikeSuggestion: `10% OTM puts (≈ $${formatStrike(currentPrice, 0.90)})`,
      expirySuggestion: isWeekly ? "60–90 DTE" : "30–45 DTE",
      rationale: "Overbought: mean reversion pullback. Puts for downside; sell covered calls only if live IV/IV rank confirms 50+.",
      strikeMultiplier: 0.90,
    }, ivProxy, "Directional", 45);
  }
  return withOptionRules({
    strategy: "Consider puts or reduce exposure",
    strikeSuggestion: `10–15% OTM puts (≈ $${formatStrike(currentPrice, 0.85)}–$${formatStrike(currentPrice, 0.90)})`,
    expirySuggestion: isWeekly ? "90 DTE" : "30–45 DTE",
    rationale: "Moderate overbought. Shorter DTE puts for pullback; avoid short premium until IV is 50+.",
    strikeMultiplier: 0.88,
  }, ivProxy, "Directional", 45);
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
      return withOptionRules({
        strategy: "LEAPS calls",
        strikeSuggestion: `ATM or 2–5% OTM (≈ $${formatStrike(currentPrice, 1.02)}–$${formatStrike(currentPrice, 1.05)})`,
        expirySuggestion: "12–18 months (e.g. Jan 2026 – Jun 2026)",
        rationale: "Deep oversold: time for thesis to play out. LEAPS give 360+ DTE for mean reversion and qualify for long-term cap gains.",
        strikeMultiplier: 1.03,
      }, null, "LEAPS");
    }
    return withOptionRules({
      strategy: "LEAPS calls or CSPs",
      strikeSuggestion: `5–10% OTM (≈ $${formatStrike(currentPrice, 1.05)}–$${formatStrike(currentPrice, 1.10)})`,
      expirySuggestion: "12–18 months",
      rationale: "Oversold setup. LEAPS reduce theta drag; size for conviction and run 9-condition checklist.",
      strikeMultiplier: 1.075,
    }, null, "LEAPS");
  }

  return withOptionRules({
    strategy: "LEAPS puts (hedge) or reduce",
    strikeSuggestion: `10% OTM puts (≈ $${formatStrike(currentPrice, 0.90)})`,
    expirySuggestion: "12–18 months if hedging",
    rationale: "Overbought: LEAPS puts for long-dated hedge. Prefer shorter DTE for pure pullback plays.",
    strikeMultiplier: 0.90,
  }, null, "LEAPS");
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
  ivProxy: number | null;
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
  const canSellPremium = premiumSellingAllowed(inputs.ivProxy);

  if (bullish > bearish) {
    if (canSellPremium) {
      const strike = `Short put 5–10% OTM (≈ $${formatStrike(price, 0.90)}–$${formatStrike(price, 0.95)})`;
      return withOptionRules({
        strategy: "Sell cash-secured puts or bull put spread",
        strikeSuggestion: strike,
        expirySuggestion: "30 DTE short-premium window",
        rationale: `Factor-based: ${factors.join(", ")}. IV proxy is ${inputs.ivProxy?.toFixed(1)}%, so premium is rich enough for 30 DTE put-selling or bull-put-spread structures.`,
        strikeMultiplier: 0.95,
        factorsUsed: [...factors, "options:premium_30d", "iv_proxy:50_plus"],
      }, inputs.ivProxy, "Premium 30D", PREMIUM_SELL_DTE);
    }
    const strike = `5–10% OTM (≈ $${formatStrike(price, 1.05)}–$${formatStrike(price, 1.10)})`;
    return withOptionRules({
      strategy: "Buy calls or sell cash-secured puts",
      strikeSuggestion: strike,
      expirySuggestion: expiryShort,
      rationale: `Factor-based: ${factors.join(", ")}. No RSI signal; use for backtest. Favor calls; sell cash-secured puts only if live IV/IV rank confirms 50+.`,
      strikeMultiplier: 1.075,
      factorsUsed: factors,
    }, inputs.ivProxy, "Directional", highBeta ? 60 : 90);
  }

  if (bearish > bullish) {
    if (canSellPremium) {
      const strike = `Short call 5–10% OTM (≈ $${formatStrike(price, 1.05)}–$${formatStrike(price, 1.10)})`;
      return withOptionRules({
        strategy: "Bear call credit spread",
        strikeSuggestion: strike,
        expirySuggestion: "30 DTE short-premium window",
        rationale: `Factor-based: ${factors.join(", ")}. IV proxy is ${inputs.ivProxy?.toFixed(1)}%, so premium is rich enough for a 30 DTE call-credit-spread setup.`,
        strikeMultiplier: 1.05,
        factorsUsed: [...factors, "options:premium_30d", "iv_proxy:50_plus"],
      }, inputs.ivProxy, "Premium 30D", PREMIUM_SELL_DTE);
    }
    const strike = `5–10% OTM puts (≈ $${formatStrike(price, 0.90)}–$${formatStrike(price, 0.95)})`;
    return withOptionRules({
      strategy: "Buy puts or reduce exposure",
      strikeSuggestion: strike,
      expirySuggestion: expiryShort,
      rationale: `Factor-based: ${factors.join(", ")}. No RSI signal; use for backtest. Puts for hedge or pullback; avoid selling calls unless live IV/IV rank confirms 50+.`,
      strikeMultiplier: 0.92,
      factorsUsed: factors,
    }, inputs.ivProxy, "Directional", highBeta ? 60 : 90);
  }

  // tie: mixed
  const strike = `ATM to 5% OTM (≈ $${formatStrike(price, 1.0)}–$${formatStrike(price, 1.05)})`;
  return withOptionRules({
    strategy: "Consider small call or put (direction from catalyst)",
    strikeSuggestion: strike,
    expirySuggestion: "45–60 DTE",
    rationale: `Factor-based mixed: ${factors.join(", ")}. Backtest both sides or wait for clearer RSI. Avoid short premium until directional edge and IV 50+ align.`,
    strikeMultiplier: 1.02,
    factorsUsed: factors,
  }, inputs.ivProxy, "Directional", 60);
}

/**
 * LEAPS version of factor-based suggestion (same logic, 12–18 mo expiry).
 */
export function getFactorBasedLeapsSuggestion(inputs: FactorInputs): FactorBasedSuggestion | null {
  const base = getFactorBasedSuggestion(inputs);
  if (!base) return null;
  const bearish = /bear|put|reduce/i.test(base.strategy) && !/bull put|cash-secured put/i.test(base.strategy);
  return {
    ...base,
    strategy: bearish ? "LEAPS puts or reduce exposure" : "LEAPS calls",
    optionCategory: "LEAPS",
    dte: undefined,
    ivGate: "not_applicable",
    expirySuggestion: "12–18 months (LEAPS)",
    rationale: base.rationale
      .replace(/Premium is rich enough for 30 DTE .*? structures\./i, "Use LEAPS only when the directional thesis is stronger than the short-premium setup.")
      .replace(/premium is rich enough for a 30 DTE call-credit-spread setup\./i, "Use LEAPS puts only when the directional hedge thesis is stronger than the short-premium setup.")
      .replace(/Favor calls or CSPs; size small until RSI confirms\.?/i, "LEAPS reduce theta; run 9-condition checklist before entry.")
      .replace(/Favor calls; sell cash-secured puts only if live IV\/IV rank confirms 50\+\./i, "LEAPS reduce theta; run 9-condition checklist before entry.")
      .replace(/Puts for hedge or pullback; avoid long-dated calls\.?/i, "LEAPS puts for long-dated hedge.")
      .replace(/Puts for hedge or pullback; avoid selling calls unless live IV\/IV rank confirms 50\+\./i, "LEAPS puts for long-dated hedge.")
      .replace(/Backtest both sides or wait for clearer RSI\.?/i, "LEAPS only if catalyst view overrides mixed signals."),
    factorsUsed: base.factorsUsed.filter((factor) => !factor.startsWith("options:") && !factor.startsWith("iv_proxy:")),
  };
}
