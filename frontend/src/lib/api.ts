import type { MoversResponse, CrossoversResponse, OversoldResponse, OverboughtResponse, ResearchData } from "@/lib/types";

const DEFAULT_BASE_URL = "http://localhost:8000";

export function apiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? DEFAULT_BASE_URL;
}

function parseJson<T>(text: string, status: number): T {
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new Error(`Invalid JSON response (${status})`);
  }
}

async function fetchApi(url: string): Promise<string> {
  const res = await fetch(url, { cache: "no-store" });
  const text = await res.text().catch(() => "");
  if (!res.ok) {
    throw new Error(text ? `${res.status}: ${text}` : `API error ${res.status}`);
  }
  return text;
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
  if (params.refresh === true) url.searchParams.set("refresh", "true");

  const text = await fetchApi(url.toString());
  return parseJson<MoversResponse>(text, 200);
}

export async function fetchCrossovers(params?: {
  threshold?: number;
  refresh?: boolean;
}): Promise<CrossoversResponse> {
  const base = apiBaseUrl();
  const url = new URL(`${base}/api/crossovers`);
  if (params?.threshold != null) url.searchParams.set("threshold", String(params.threshold));
  if (params?.refresh === true) url.searchParams.set("refresh", "true");

  const text = await fetchApi(url.toString());
  return parseJson<CrossoversResponse>(text, 200);
}

export async function fetchOversold(params?: {
  threshold?: number;
  refresh?: boolean;
}): Promise<OversoldResponse> {
  const base = apiBaseUrl();
  const url = new URL(`${base}/api/rsi-oversold`);
  if (params?.threshold != null) url.searchParams.set("threshold", String(params.threshold));
  if (params?.refresh === true) url.searchParams.set("refresh", "true");

  const text = await fetchApi(url.toString());
  return parseJson<OversoldResponse>(text, 200);
}

export async function fetchOverbought(params?: {
  threshold?: number;
  refresh?: boolean;
}): Promise<OverboughtResponse> {
  const base = apiBaseUrl();
  const url = new URL(`${base}/api/rsi-overbought`);
  if (params?.threshold != null) url.searchParams.set("threshold", String(params.threshold));
  if (params?.refresh === true) url.searchParams.set("refresh", "true");

  const text = await fetchApi(url.toString());
  return parseJson<OverboughtResponse>(text, 200);
}

export async function fetchDailyOversold(params?: {
  threshold?: number;
  refresh?: boolean;
}): Promise<OversoldResponse> {
  const base = apiBaseUrl();
  const url = new URL(`${base}/api/rsi-daily-oversold`);
  if (params?.threshold != null) url.searchParams.set("threshold", String(params.threshold));
  if (params?.refresh === true) url.searchParams.set("refresh", "true");

  const text = await fetchApi(url.toString());
  return parseJson<OversoldResponse>(text, 200);
}

export async function fetchDailyOverbought(params?: {
  threshold?: number;
  refresh?: boolean;
}): Promise<OverboughtResponse> {
  const base = apiBaseUrl();
  const url = new URL(`${base}/api/rsi-daily-overbought`);
  if (params?.threshold != null) url.searchParams.set("threshold", String(params.threshold));
  if (params?.refresh === true) url.searchParams.set("refresh", "true");

  const text = await fetchApi(url.toString());
  return parseJson<OverboughtResponse>(text, 200);
}

export async function fetchResearch(
  ticker: string,
  params?: { start?: string; end?: string; refresh?: boolean },
): Promise<ResearchData> {
  const base = apiBaseUrl();
  const url = new URL(`${base}/api/research/${encodeURIComponent(ticker.trim().toUpperCase())}`);
  if (params?.start) url.searchParams.set("start", params.start);
  if (params?.end) url.searchParams.set("end", params.end);
  if (params?.refresh === true) url.searchParams.set("refresh", "true");

  const text = await fetchApi(url.toString());
  return parseJson<ResearchData>(text, 200);
}

