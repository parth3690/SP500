import type { MoversResponse, CrossoversResponse, OversoldResponse, ResearchData } from "@/lib/types";

const DEFAULT_BASE_URL = "http://localhost:8000";

export function apiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? DEFAULT_BASE_URL;
}

export async function fetchMovers(params: {
  start: string;
  end: string;
  limit: number;
  includeAll: boolean;
  refresh?: boolean;
}): Promise<MoversResponse> {
  const base = apiBaseUrl();
  const url = new URL(`${base}/api/movers`);
  url.searchParams.set("start", params.start);
  url.searchParams.set("end", params.end);
  url.searchParams.set("limit", String(params.limit));
  url.searchParams.set("includeAll", String(params.includeAll));
  if (params.refresh) url.searchParams.set("refresh", "true");

  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API error (${res.status}): ${text || res.statusText}`);
  }
  return (await res.json()) as MoversResponse;
}

export async function fetchCrossovers(params?: {
  threshold?: number;
  refresh?: boolean;
}): Promise<CrossoversResponse> {
  const base = apiBaseUrl();
  const url = new URL(`${base}/api/crossovers`);
  if (params?.threshold != null) url.searchParams.set("threshold", String(params.threshold));
  if (params?.refresh) url.searchParams.set("refresh", "true");

  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API error (${res.status}): ${text || res.statusText}`);
  }
  return (await res.json()) as CrossoversResponse;
}

export async function fetchOversold(params?: {
  threshold?: number;
  refresh?: boolean;
}): Promise<OversoldResponse> {
  const base = apiBaseUrl();
  const url = new URL(`${base}/api/rsi-oversold`);
  if (params?.threshold != null) url.searchParams.set("threshold", String(params.threshold));
  if (params?.refresh) url.searchParams.set("refresh", "true");

  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API error (${res.status}): ${text || res.statusText}`);
  }
  return (await res.json()) as OversoldResponse;
}

export async function fetchResearch(
  ticker: string,
  params?: { start?: string; end?: string; refresh?: boolean },
): Promise<ResearchData> {
  const base = apiBaseUrl();
  const url = new URL(`${base}/api/research/${encodeURIComponent(ticker)}`);
  if (params?.start) url.searchParams.set("start", params.start);
  if (params?.end) url.searchParams.set("end", params.end);
  if (params?.refresh) url.searchParams.set("refresh", "true");

  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API error (${res.status}): ${text || res.statusText}`);
  }
  return (await res.json()) as ResearchData;
}

