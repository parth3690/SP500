import type { MoversResponse } from "@/lib/types";

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

