import type { MoversResponse, CrossoversResponse, OversoldResponse, OverboughtResponse, ResearchData, MultibaggerResponse, MarketConditionsFetchResponse, AlphaCandidatesResponse, AgentBotRunResponse } from "@/lib/types";

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

function apiError(text: string, status: number): Error {
  try {
    const payload = JSON.parse(text) as { detail?: unknown };
    if (typeof payload.detail === "string") return new Error(payload.detail);
  } catch {
    // Use the raw response below when the server did not return JSON.
  }
  return new Error(text || `API error ${status}`);
}

async function fetchApi(url: string): Promise<string> {
  const res = await fetch(url, { cache: "no-store" });
  const text = await res.text().catch(() => "");
  if (!res.ok) {
    throw apiError(text, res.status);
  }
  return text;
}

async function fetchApiJson(url: string, body: unknown): Promise<string> {
  const res = await fetch(url, {
    method: "POST",
    cache: "no-store",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await res.text().catch(() => "");
  if (!res.ok) {
    throw apiError(text, res.status);
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

export async function fetchMultibagger(
  ticker: string,
  params?: { deep?: boolean; refresh?: boolean },
): Promise<MultibaggerResponse> {
  const base = apiBaseUrl();
  const sym = ticker.trim().toUpperCase();
  const url = new URL(`${base}/api/multibagger/${encodeURIComponent(sym)}`);
  if (params?.deep === true) url.searchParams.set("deep", "true");
  if (params?.refresh === true) url.searchParams.set("refresh", "true");

  const text = await fetchApi(url.toString());
  return parseJson<MultibaggerResponse>(text, 200);
}

export async function fetchMarketConditions(params?: { refresh?: boolean }): Promise<MarketConditionsFetchResponse> {
  const base = apiBaseUrl();
  const url = new URL(`${base}/api/market-conditions/fetch`);
  if (params?.refresh === true) url.searchParams.set("refresh", "true");

  const text = await fetchApi(url.toString());
  return parseJson<MarketConditionsFetchResponse>(text, 200);
}

export async function fetchAlphaCandidates(params?: {
  limit?: number;
  minScore?: number;
  sector?: string;
  maxBeta?: number;
  riskMode?: "balanced" | "aggressive" | "defensive";
  regime?: "auto" | "risk_on" | "neutral" | "risk_off";
  enrichTop?: number;
  refresh?: boolean;
}): Promise<AlphaCandidatesResponse> {
  const base = apiBaseUrl();
  const url = new URL(`${base}/api/alpha-candidates`);
  if (params?.limit != null) url.searchParams.set("limit", String(params.limit));
  if (params?.minScore != null) url.searchParams.set("minScore", String(params.minScore));
  if (params?.sector) url.searchParams.set("sector", params.sector);
  if (params?.maxBeta != null) url.searchParams.set("maxBeta", String(params.maxBeta));
  if (params?.riskMode) url.searchParams.set("riskMode", params.riskMode);
  if (params?.regime) url.searchParams.set("regime", params.regime);
  if (params?.enrichTop != null) url.searchParams.set("enrichTop", String(params.enrichTop));
  if (params?.refresh === true) url.searchParams.set("refresh", "true");

  const text = await fetchApi(url.toString());
  return parseJson<AlphaCandidatesResponse>(text, 200);
}

export async function fetchAlphaWatchlist(params: {
  tickers: string[];
  limit?: number;
  minScore?: number;
  maxBeta?: number;
  riskMode?: "balanced" | "aggressive" | "defensive";
  regime?: "auto" | "risk_on" | "neutral" | "risk_off";
  enrichTop?: number;
  refresh?: boolean;
}): Promise<AlphaCandidatesResponse> {
  const base = apiBaseUrl();
  const url = new URL(`${base}/api/alpha-watchlist`);
  const text = await fetchApiJson(url.toString(), {
    tickers: params.tickers,
    limit: params.limit ?? 50,
    minScore: params.minScore ?? 0,
    maxBeta: params.maxBeta,
    riskMode: params.riskMode ?? "balanced",
    regime: params.regime ?? "auto",
    enrichTop: params.enrichTop ?? 20,
    refresh: params.refresh === true,
  });
  return parseJson<AlphaCandidatesResponse>(text, 200);
}

export type AgentBotHistoryItem = {
  id?: string;
  ticker: string;
  action: string;
  entryPrice: number;
  recommendedAt: string;
  closed?: boolean;
  closedAt?: string;
  exitPrice?: number;
};

export async function fetchAgentBotRun(params: {
  mode: "sp500" | "watchlist";
  tickers: string[];
  riskMode?: "balanced" | "aggressive" | "defensive";
  regime?: "auto" | "risk_on" | "neutral" | "risk_off";
  topN?: number;
  minScore?: number;
  history?: AgentBotHistoryItem[];
  refresh?: boolean;
}): Promise<AgentBotRunResponse> {
  const base = apiBaseUrl();
  const url = new URL(`${base}/api/agent-bot/run`);
  const text = await fetchApiJson(url.toString(), {
    mode: params.mode,
    tickers: params.tickers,
    riskMode: params.riskMode ?? "balanced",
    regime: params.regime ?? "auto",
    topN: params.topN ?? 10,
    minScore: params.minScore ?? 55,
    history: params.history ?? [],
    refresh: params.refresh === true,
  });
  return parseJson<AgentBotRunResponse>(text, 200);
}
