/**
 * Shared GBM / Monte Carlo logic for quantitative screening and factor-based option suggestions.
 * Single source of truth for log-return volatility and path signals (data accuracy + no duplicate work).
 */

export type VolatilityPathSignals = {
  gbmBullish: boolean;
  gbmBearish: boolean;
  mcBullish: boolean;
  mcBearish: boolean;
};

const MC_PATHS = 500;
const MC_SEED_MULT = 9301;
const MC_SEED_ADD = 49297;
const MC_SEED_MOD = 233280;

/**
 * Compute 1m GBM upside/downside and 500-path Monte Carlo signals from close prices.
 * Uses log returns, annualized vol/drift, deterministic MC seed for reproducibility.
 * Returns null if fewer than 22 observations.
 */
export function getVolatilityAndPathSignals(
  validClose: number[],
  currentPrice: number
): VolatilityPathSignals | null {
  const n = validClose.length;
  if (n < 22) return null;

  const logR: number[] = [];
  for (let i = 1; i < validClose.length; i++) {
    logR.push(Math.log(validClose[i] / validClose[i - 1]));
  }
  const mean = logR.reduce((a, b) => a + b, 0) / logR.length;
  const variance = logR.reduce((a, b) => a + (b - mean) ** 2, 0) / logR.length;
  const vol = Math.sqrt(Math.max(0, variance) * 252);
  const drift = mean * 252;
  const S0 = currentPrice;
  const t = 1 / 12;

  // GBM: 90th / 10th percentile (z ≈ ±1.28)
  const up = S0 * Math.exp((drift - 0.5 * vol * vol) * t + vol * Math.sqrt(t) * 1.28);
  const down = S0 * Math.exp((drift - 0.5 * vol * vol) * t - vol * Math.sqrt(t) * 1.28);
  const upside = (up - S0) / S0;
  const downside = S0 - down > 0 ? (S0 - down) / S0 : 0;
  const gbmBullish = downside > 0 && upside >= 2 * downside;
  const gbmBearish = down < S0 * 0.8;

  // Monte Carlo: deterministic seed from data
  let above = 0;
  let below = 0;
  let seed = (validClose[0] ?? 0) * 1e4 + n;
  const next = () => {
    seed = (seed * MC_SEED_MULT + MC_SEED_ADD) % MC_SEED_MOD;
    return seed / MC_SEED_MOD;
  };
  for (let i = 0; i < MC_PATHS; i++) {
    const u = next() + next() + next() + next() - 2;
    const z = Math.max(-2.5, Math.min(2.5, u));
    const st = S0 * Math.exp((drift - 0.5 * vol * vol) * t + vol * Math.sqrt(t) * z);
    if (st >= S0 * 1.2) above++;
    if (st <= S0 * 0.8) below++;
  }
  const mcBullish = below > 0 && above >= 2 * below;
  const mcBearish = below >= above;

  return { gbmBullish, gbmBearish, mcBullish, mcBearish };
}
