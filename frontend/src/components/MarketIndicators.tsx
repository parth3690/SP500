"use client";

import { useMemo, useRef, useState } from "react";
import clsx from "clsx";
import {
  HISTORICAL_PEAKS,
  cloneSeedConditions,
  computeSummary,
  mergeImportedConditions,
  resolveState,
  sortedCategories,
  type ConditionState,
  type RuntimeCondition,
} from "@/lib/marketConditions";

function slugify(text: string): string {
  return (
    text
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_|_$/g, "")
      .slice(0, 48) || "condition"
  );
}

function uniqueId(base: string, existing: Set<string>): string {
  let id = base;
  let i = 2;
  while (existing.has(id)) {
    id = `${base}_${i++}`;
  }
  return id;
}

const STATE_BUTTONS: { state: ConditionState; label: string }[] = [
  { state: "triggered", label: "Trig" },
  { state: "not_triggered", label: "Clear" },
  { state: "unknown", label: "n/a" },
];

export default function MarketIndicators() {
  const [conditions, setConditions] = useState<RuntimeCondition[]>(() => cloneSeedConditions());
  const [showAdd, setShowAdd] = useState(false);
  const [addName, setAddName] = useState("");
  const [addCategory, setAddCategory] = useState("");
  const [addThreshold, setAddThreshold] = useState("");
  const [addSource, setAddSource] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const summary = useMemo(() => computeSummary(conditions), [conditions]);
  const categories = useMemo(() => sortedCategories(conditions), [conditions]);
  const categoryOptions = useMemo(
    () => [...new Set(conditions.map((c) => c.category))].sort(),
    [conditions],
  );

  const updateCondition = (id: string, patch: Partial<RuntimeCondition>) => {
    setConditions((prev) =>
      prev.map((c) => (c.id === id ? { ...c, ...patch } : c)),
    );
  };

  const onValueChange = (id: string, raw: string) => {
    setConditions((prev) =>
      prev.map((c) => {
        if (c.id !== id) return c;
        const value = raw.trim() === "" ? null : raw;
        const next: RuntimeCondition = { ...c, value, manualState: false };
        if (next.evaluate) next.state = resolveState(next);
        return next;
      }),
    );
  };

  const onStateChange = (id: string, state: ConditionState) => {
    updateCondition(id, { state, manualState: true });
  };

  const exportJson = () => {
    const payload = conditions.map(({ evaluate, ...rest }) => ({
      ...rest,
      ...(rest.manualState ? { manualState: true } : {}),
    }));
    const blob = new Blob([JSON.stringify({ conditions: payload }, null, 2)], {
      type: "application/json",
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `market-conditions-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const onImport = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(String(reader.result));
        const incoming = Array.isArray(data) ? data : data.conditions;
        if (!Array.isArray(incoming)) throw new Error("Expected { conditions: [...] }");
        setConditions(mergeImportedConditions(incoming));
      } catch (e) {
        alert(`Import failed: ${e instanceof Error ? e.message : "unknown error"}`);
      }
    };
    reader.readAsText(file);
  };

  const onAdd = (e: React.FormEvent) => {
    e.preventDefault();
    if (!addName.trim() || !addCategory.trim() || !addThreshold.trim() || !addSource.trim()) return;
    const existing = new Set(conditions.map((c) => c.id));
    const id = uniqueId(slugify(addName), existing);
    setConditions((prev) => [
      ...prev,
      {
        id,
        name: addName.trim(),
        category: addCategory.trim(),
        threshold: addThreshold.trim(),
        source: addSource.trim(),
        value: null,
        state: "unknown",
        notes: "",
        manualState: false,
      },
    ]);
    setAddName("");
    setAddCategory("");
    setAddThreshold("");
    setAddSource("");
    setShowAdd(false);
  };

  return (
    <div className="mx-auto max-w-[1200px] px-4 py-8">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight">Market Conditions</h1>
        <p className="mt-1 text-sm text-slate-400">
          Peak signpost checklist — compare current signals against historical S&amp;P 500 tops
        </p>
      </header>

      <div className="mb-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <SummaryCard label="% Triggered" value={summary.pct !== null ? `${summary.pct}%` : "—"} accent />
        <SummaryCard label="Triggered" value={String(summary.triggered)} detail={`of ${summary.known} known`} />
        <SummaryCard label="Unknown" value={String(summary.unknown)} detail="excluded from %" />
        <SummaryCard label="Total signals" value={String(summary.total)} />
      </div>

      <section className="mb-6 rounded-xl border border-slate-800 bg-slate-900/40 p-4">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
          Historical reference (% triggered at prior peaks)
        </h2>
        <div className="mt-3 flex flex-wrap gap-2">
          {HISTORICAL_PEAKS.map((p) => {
            const near = summary.pct !== null && Math.abs(p.pctTriggered - summary.pct) <= 5;
            return (
              <span
                key={p.date}
                title={`S&P ${p.spLevel.toLocaleString()}`}
                className={clsx(
                  "rounded border px-2 py-1 font-mono text-[11px]",
                  near
                    ? "border-rose-500/60 bg-rose-500/10 text-rose-300"
                    : "border-slate-700 text-slate-400",
                )}
              >
                {p.date} {p.pctTriggered}% / {p.spLevel.toLocaleString()}
              </span>
            );
          })}
        </div>
      </section>

      <div className="mb-6 flex flex-wrap gap-2">
        <button type="button" className="rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm hover:bg-slate-800" onClick={exportJson}>
          Export JSON
        </button>
        <button
          type="button"
          className="rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm hover:bg-slate-800"
          onClick={() => fileRef.current?.click()}
        >
          Import JSON
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".json,application/json"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onImport(file);
            e.target.value = "";
          }}
        />
        <button
          type="button"
          className="rounded-md border border-rose-500/50 px-3 py-2 text-sm text-rose-300 hover:bg-rose-500/10"
          onClick={() => setShowAdd((v) => !v)}
        >
          + Add condition
        </button>
      </div>

      {showAdd && (
        <form onSubmit={onAdd} className="mb-6 grid gap-3 rounded-xl border border-slate-800 bg-slate-900/50 p-4 sm:grid-cols-2 lg:grid-cols-5">
          <Field label="Name" value={addName} onChange={setAddName} required />
          <Field label="Category" value={addCategory} onChange={setAddCategory} list="mc-categories" required />
          <datalist id="mc-categories">
            {categoryOptions.map((c) => (
              <option key={c} value={c} />
            ))}
          </datalist>
          <Field label="Threshold" value={addThreshold} onChange={setAddThreshold} required />
          <Field label="Source" value={addSource} onChange={setAddSource} required />
          <div className="flex items-end">
            <button type="submit" className="rounded-md bg-slate-100 px-4 py-2 text-sm font-medium text-slate-900">
              Add
            </button>
          </div>
        </form>
      )}

      {categories.map((category) => {
        const rows = conditions.filter((c) => c.category === category);
        return (
          <section key={category} className="mb-8">
            <h2 className="mb-3 border-b border-slate-800 pb-2 text-sm font-semibold uppercase tracking-wide">
              {category}
            </h2>
            <div className="overflow-x-auto rounded-xl border border-slate-800">
              <table className="w-full min-w-[720px] text-sm">
                <thead>
                  <tr className="border-b border-slate-800 bg-slate-900/60 text-left text-xs uppercase tracking-wide text-slate-400">
                    <th className="px-3 py-2">Signal</th>
                    <th className="px-3 py-2">Threshold</th>
                    <th className="hidden px-3 py-2 sm:table-cell">Source</th>
                    <th className="px-3 py-2">Value</th>
                    <th className="px-3 py-2">State</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((c) => {
                    const effective = resolveState(c);
                    return (
                      <tr key={c.id} className="border-b border-slate-800/80 hover:bg-slate-900/30">
                        <td className="px-3 py-2">
                          <span className="inline-flex items-start gap-2">
                            <span
                              className={clsx(
                                "mt-1.5 h-2 w-2 flex-shrink-0 rounded-full",
                                effective === "triggered" && "bg-rose-500",
                                effective === "not_triggered" && "bg-slate-500",
                                effective === "unknown" && "border border-slate-600 bg-slate-800",
                              )}
                            />
                            <span>{c.name}</span>
                          </span>
                        </td>
                        <td className="px-3 py-2 text-xs text-slate-400">{c.threshold}</td>
                        <td className="hidden px-3 py-2 text-xs text-slate-500 sm:table-cell">{c.source}</td>
                        <td className="px-3 py-2">
                          <input
                            className="w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 font-mono text-xs"
                            value={c.value === null ? "" : String(c.value)}
                            placeholder="—"
                            onChange={(e) => onValueChange(c.id, e.target.value)}
                          />
                        </td>
                        <td className="px-3 py-2">
                          <div className="inline-flex border border-slate-700">
                            {STATE_BUTTONS.map(({ state, label }) => (
                              <button
                                key={state}
                                type="button"
                                className={clsx(
                                  "px-2 py-1 text-[10px] uppercase tracking-wide",
                                  effective === state
                                    ? state === "triggered"
                                      ? "bg-rose-500/20 text-rose-300"
                                      : state === "not_triggered"
                                        ? "bg-slate-700 text-slate-200"
                                        : "bg-slate-800 text-slate-500"
                                    : "text-slate-500 hover:bg-slate-800",
                                )}
                                onClick={() => onStateChange(c.id, state)}
                              >
                                {label}
                              </button>
                            ))}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        );
      })}
    </div>
  );
}

function SummaryCard(props: {
  label: string;
  value: string;
  detail?: string;
  accent?: boolean;
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
      <div className="text-xs uppercase tracking-wide text-slate-400">{props.label}</div>
      <div className={clsx("mt-1 font-mono text-2xl font-semibold tabular-nums", props.accent && "text-rose-400")}>
        {props.value}
      </div>
      {props.detail ? <div className="mt-0.5 text-xs text-slate-500">{props.detail}</div> : null}
    </div>
  );
}

function Field(props: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  required?: boolean;
  list?: string;
}) {
  return (
    <label className="block text-xs text-slate-400">
      {props.label}
      <input
        className="mt-1 w-full rounded border border-slate-700 bg-slate-950 px-2 py-2 text-sm text-slate-100"
        value={props.value}
        onChange={(e) => props.onChange(e.target.value)}
        required={props.required}
        list={props.list}
      />
    </label>
  );
}
