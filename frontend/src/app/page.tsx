"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import clsx from "clsx";

const Dashboard = dynamic(() => import("@/components/Dashboard"), { ssr: false });
const MarketIndicators = dynamic(() => import("@/components/MarketIndicators"), { ssr: false });
const MultibaggerScanner = dynamic(() => import("@/components/MultibaggerScanner"), { ssr: false });
const AlphaCandidates = dynamic(() => import("@/components/AlphaCandidates"), { ssr: false });
const AgentBot = dynamic(() => import("@/components/AgentBot"), { ssr: false });
const InstitutionalScanner = dynamic(() => import("@/components/InstitutionalScanner"), { ssr: false });
const NyseSmidAgent = dynamic(() => import("@/components/NyseSmidAgent"), { ssr: false });

type AppTab = "agent" | "institutional" | "nyse_smid" | "alpha" | "sp500" | "indicators" | "multibagger";

const TABS: { id: AppTab; label: string }[] = [
  { id: "agent", label: "Agent Bot" },
  { id: "institutional", label: "Institutional Scanner" },
  { id: "nyse_smid", label: "NYSE SMID Agent" },
  { id: "alpha", label: "Alpha Candidates" },
  { id: "sp500", label: "S&P 500 Dashboard" },
  { id: "indicators", label: "Market Indicators" },
  { id: "multibagger", label: "US Multibagger Scanner" },
];

export default function Page() {
  const [tab, setTab] = useState<AppTab>("alpha");

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <nav className="sticky top-0 z-20 border-b border-slate-800 bg-slate-950/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center gap-2 px-4 py-3">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setTab(t.id)}
              className={clsx(
                "rounded-md px-3 py-2 text-sm font-medium transition-colors",
                tab === t.id
                  ? "bg-slate-100 text-slate-900"
                  : "text-slate-300 hover:bg-slate-800 hover:text-slate-100",
              )}
            >
              {t.label}
            </button>
          ))}
        </div>
      </nav>

      {tab === "agent" ? <AgentBot /> : null}
      {tab === "institutional" ? <InstitutionalScanner /> : null}
      {tab === "nyse_smid" ? <NyseSmidAgent /> : null}
      {tab === "alpha" ? <AlphaCandidates /> : null}
      {tab === "sp500" ? <Dashboard /> : null}
      {tab === "indicators" ? <MarketIndicators /> : null}
      {tab === "multibagger" ? <MultibaggerScanner /> : null}
    </main>
  );
}
