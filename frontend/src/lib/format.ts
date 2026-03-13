/**
 * Centralized number/price/volume formatting for consistency and reuse.
 */

/** Locale-aware number with fixed decimals; "N/A" for null/undefined. */
export function formatNumber(v: number | null | undefined, decimals = 2): string {
  if (v == null) return "N/A";
  return v.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

/** Large numbers as $B / $M / $T. */
export function formatLarge(v: number | null | undefined): string {
  if (v == null) return "N/A";
  if (v >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
  if (v >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
  return `$${v.toLocaleString()}`;
}

/** Volume as K / M / B (no $). */
export function formatVol(v: number): string {
  if (v >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(0)}K`;
  return v.toLocaleString();
}

/** Percent with optional sign prefix. */
export function formatPct(v: number | null | undefined, decimals = 2): string {
  if (v == null) return "N/A";
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}%`;
}

/** Price for display (>=1000 no decimals, >=100 one, else two). */
export function formatMoney(v: number): string {
  if (v >= 1000) return v.toFixed(0);
  if (v >= 100) return v.toFixed(2);
  if (v >= 10) return v.toFixed(2);
  return v.toFixed(2);
}

/** Strike price from current price × multiplier (option suggestions). */
export function formatStrike(price: number, mult: number): string {
  const s = price * mult;
  if (s >= 1000) return s.toFixed(0);
  if (s >= 100) return s.toFixed(1);
  if (s >= 10) return s.toFixed(2);
  return s.toFixed(2);
}
