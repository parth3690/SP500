const pad2 = (n: number) => String(n).padStart(2, "0");

/** YYYY-MM-DD in local time. */
export function toLocalISODate(d: Date): string {
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
}

/** UTC YYYY-MM-DD (for API params). */
export function toISODate(d: Date): string {
  return d.toISOString().slice(0, 10);
}

/** ISO date string for n days ago (from today). */
export function daysAgo(n: number): string {
  const d = new Date();
  d.setDate(d.getDate() - n);
  return toISODate(d);
}

export function addDays(isoDate: string, days: number): string {
  const [y, m, d] = isoDate.split("-").map(Number);
  const dt = new Date(y, m - 1, d);
  dt.setDate(dt.getDate() + days);
  return toLocalISODate(dt);
}

/** Start of year for an ISO date (YYYY-MM-DD). Returns YYYY-01-01. */
export function startOfYear(isoDate: string): string {
  return isoDate.slice(0, 4) + "-01-01";
}

