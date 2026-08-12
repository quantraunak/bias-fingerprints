"use client";

import { useMemo, useState } from "react";
import {
  analyze, ASSET_LABEL, ACCOUNT_LABEL,
  type AccountType, type AssetClass, type Holding, type Profile, type Style,
} from "@/lib/analysis";

const CLASSES = Object.keys(ASSET_LABEL) as AssetClass[];
const ACCOUNTS = Object.keys(ACCOUNT_LABEL) as AccountType[];

const SAMPLE: Holding[] = [
  { id: "1", ticker: "NVDA", assetClass: "us_equity", account: "taxable", value: 8_400_000, shares: 46_666, costBasis: 1_612_000, singleName: true, edge: 0.06, vol: 0.45 },
  { id: "2", ticker: "VTI", assetClass: "us_equity", account: "taxable", value: 5_200_000, shares: 18_245, costBasis: 4_185_000, singleName: false },
  { id: "3", ticker: "VXUS", assetClass: "intl_equity", account: "taxable", value: 1_600_000, shares: 25_806, costBasis: 1_755_000, singleName: false },
  { id: "4", ticker: "BND", assetClass: "bonds", account: "taxable", value: 2_100_000, shares: 28_767, costBasis: 2_272_000, singleName: false },
  { id: "5", ticker: "Cash", assetClass: "cash", account: "taxable", value: 3_300_000, costBasis: 3_300_000, singleName: false },
  { id: "6", ticker: "AGG", assetClass: "bonds", account: "ira", value: 1_900_000, costBasis: 1_900_000, singleName: false },
  { id: "7", ticker: "BND", assetClass: "bonds", account: "roth", value: 900_000, costBasis: 900_000, singleName: false },
];

const money = (n: number) => {
  const sign = n < 0 ? "−" : "";
  const v = Math.abs(n);
  if (v >= 1_000_000) return `${sign}$${(v / 1_000_000).toFixed(2)}M`;
  if (v >= 1000) return `${sign}$${Math.round(v / 1000)}K`;
  return `${sign}$${Math.round(v)}`;
};
const pct = (n: number) => `${(n * 100).toFixed(1)}%`;

const blank = (): Holding => ({
  id: Math.random().toString(36).slice(2),
  ticker: "", assetClass: "us_equity", account: "taxable",
  value: 0, shares: 0, costBasis: 0, singleName: false, edge: 0, vol: 0.42,
});

export default function Tool() {
  const [holdings, setHoldings] = useState<Holding[]>(SAMPLE);
  const [quotes, setQuotes] = useState<Record<string, { price: number; asOf: string | null }>>({});
  const [loadingQuotes, setLoadingQuotes] = useState(false);
  const [quoteNote, setQuoteNote] = useState<string | null>(null);

  async function refreshPrices() {
    const tickers = holdings.map((h) => h.ticker).filter(Boolean);
    if (tickers.length === 0) return;
    setLoadingQuotes(true);
    setQuoteNote(null);
    try {
      const res = await fetch("/api/quote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tickers }),
      });
      const data = await res.json();
      const next: Record<string, { price: number; asOf: string | null }> = {};
      let missing: string[] = [];
      for (const q of data.quotes ?? []) {
        if (q.price) next[q.ticker] = { price: q.price, asOf: q.asOf };
        else missing.push(q.ticker);
      }
      setQuotes(next);
      setQuoteNote(
        missing.length
          ? `No price for ${missing.join(", ")} — those keep the value you typed.`
          : `Prices updated. Only tickers were sent; your values never left this page.`,
      );
      // Re-mark positions to market where we have both a price and share count.
      setHoldings((hs) =>
        hs.map((h) => {
          const q = next[h.ticker];
          return q && h.shares ? { ...h, value: Math.round(q.price * h.shares) } : h;
        }),
      );
    } catch {
      setQuoteNote("Could not reach the price service. Values are unchanged.");
    } finally {
      setLoadingQuotes(false);
    }
  }
  const [profile, setProfile] = useState<Profile>({
    drawdownTolerance: 0.35, horizonYears: 30, ltcgRate: 0.371, ordinaryRate: 0.541,
    style: "barbell", floorYears: 5, annualSpending: 500_000,
  });

  const a = useMemo(
    () => analyze(holdings.filter((h) => h.value > 0), profile),
    [holdings, profile],
  );

  const set = (id: string, patch: Partial<Holding>) =>
    setHoldings((hs) => hs.map((h) => (h.id === id ? { ...h, ...patch } : h)));

  const sev = {
    critical: { dot: "#8c3a1e", label: "Act now" },
    warning: { dot: "#9a6f2c", label: "Worth fixing" },
    ok: { dot: "#47654f", label: "Fine as is" },
  } as const;

  return (
    <>
      <section className="border-b border-rule bg-sunk">
        <div className="mx-auto max-w-6xl px-6 pb-14 pt-16 md:pt-20">
          <div className="flex items-center gap-4">
            <span className="eyebrow">Portfolio review</span>
            <span className="h-px flex-1 bg-rule" aria-hidden="true" />
          </div>
          <h1 className="mt-7 max-w-3xl text-balance font-display text-4xl font-light leading-[1.1] tracking-[-0.02em] text-ink md:text-5xl">
            Put your holdings in. See what to change and why.
          </h1>
          <p className="mt-6 max-w-2xl leading-relaxed text-ink-2">
            Everything is calculated in your browser. Your numbers are never
            sent anywhere, never stored, and never seen by us — close the tab
            and they are gone. Start from the sample below or clear it and
            enter your own.
          </p>
        </div>
      </section>

      <div className="mx-auto max-w-6xl px-6 py-14">
        {/* ---------------- inputs ---------------- */}
        <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
          <h2 className="font-display text-2xl font-normal text-ink">
            Your holdings
          </h2>
          <div className="flex gap-3">
            <button
              onClick={refreshPrices}
              disabled={loadingQuotes}
              className="border border-ochre px-4 py-2 font-mono text-[0.7rem] uppercase tracking-[0.13em] text-ochre transition-colors hover:bg-ochre hover:text-white disabled:opacity-50"
            >
              {loadingQuotes ? "Fetching…" : "Get live prices"}
            </button>
            <button
              onClick={() => setHoldings((h) => [...h, blank()])}
              className="border border-veld px-4 py-2 font-mono text-[0.7rem] uppercase tracking-[0.13em] text-veld transition-colors hover:bg-veld hover:text-white"
            >
              Add holding
            </button>
            <button
              onClick={() => setHoldings([blank()])}
              className="border border-rule-strong px-4 py-2 font-mono text-[0.7rem] uppercase tracking-[0.13em] text-muted transition-colors hover:border-ink hover:text-ink"
            >
              Clear
            </button>
          </div>
        </div>

        {quoteNote && (
          <p className="mb-4 border-l-2 border-ochre bg-surface px-4 py-3 font-mono text-[0.7rem] leading-relaxed text-ink-2">
            {quoteNote}
          </p>
        )}

        <div className="overflow-x-auto border border-rule bg-surface">
          <table className="w-full min-w-[52rem] border-collapse">
            <thead>
              <tr className="border-b border-rule-strong">
                {["Ticker", "Type", "Account", "Shares", "Value", "Cost basis", "One company?", "Your edge", ""].map((h) => (
                  <th key={h} className="px-3 py-3 text-left font-mono text-[0.62rem] font-medium uppercase tracking-[0.11em] text-muted">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {holdings.map((h) => (
                <tr key={h.id} className="border-b border-rule last:border-0">
                  <td className="px-3 py-2">
                    <input
                      value={h.ticker}
                      onChange={(e) => set(h.id, { ticker: e.target.value.toUpperCase() })}
                      placeholder="AAPL"
                      aria-label="Ticker"
                      className="w-24 border border-rule bg-paper px-2 py-1.5 font-mono text-sm text-ink focus:border-veld focus:outline-none"
                    />
                  </td>
                  <td className="px-3 py-2">
                    <select
                      value={h.assetClass}
                      onChange={(e) => set(h.id, { assetClass: e.target.value as AssetClass })}
                      aria-label="Asset type"
                      className="border border-rule bg-paper px-2 py-1.5 text-sm text-ink focus:border-veld focus:outline-none"
                    >
                      {CLASSES.map((c) => <option key={c} value={c}>{ASSET_LABEL[c]}</option>)}
                    </select>
                  </td>
                  <td className="px-3 py-2">
                    <select
                      value={h.account}
                      onChange={(e) => set(h.id, { account: e.target.value as AccountType })}
                      aria-label="Account"
                      className="border border-rule bg-paper px-2 py-1.5 text-sm text-ink focus:border-veld focus:outline-none"
                    >
                      {ACCOUNTS.map((c) => <option key={c} value={c}>{ACCOUNT_LABEL[c]}</option>)}
                    </select>
                  </td>
                  <td className="px-3 py-2">
                    <input
                      type="number" value={h.shares || ""}
                      onChange={(e) => {
                        const shares = Number(e.target.value);
                        const q = quotes[h.ticker];
                        set(h.id, q ? { shares, value: Math.round(q.price * shares) } : { shares });
                      }}
                      placeholder="—" aria-label="Shares"
                      className="tnum w-24 border border-rule bg-paper px-2 py-1.5 text-right font-mono text-sm text-ink focus:border-veld focus:outline-none"
                    />
                    {quotes[h.ticker] && (
                      <div className="tnum mt-1 text-right font-mono text-[0.6rem] text-muted">
                        ${quotes[h.ticker].price.toFixed(2)}
                      </div>
                    )}
                  </td>
                  <td className="px-3 py-2">
                    <input
                      type="number" value={h.value || ""}
                      onChange={(e) => set(h.id, { value: Number(e.target.value) })}
                      placeholder="0" aria-label="Current value"
                      className="tnum w-32 border border-rule bg-paper px-2 py-1.5 text-right font-mono text-sm text-ink focus:border-veld focus:outline-none"
                    />
                  </td>
                  <td className="px-3 py-2">
                    <input
                      type="number" value={h.costBasis || ""}
                      onChange={(e) => set(h.id, { costBasis: Number(e.target.value) })}
                      placeholder="0" aria-label="Cost basis"
                      className="tnum w-32 border border-rule bg-paper px-2 py-1.5 text-right font-mono text-sm text-ink focus:border-veld focus:outline-none"
                    />
                  </td>
                  <td className="px-3 py-2 text-center">
                    <input
                      type="checkbox" checked={h.singleName}
                      onChange={(e) => set(h.id, { singleName: e.target.checked })}
                      aria-label="Is a single company"
                      className="h-4 w-4 accent-[var(--color-veld)]"
                    />
                  </td>
                  <td className="px-3 py-2">
                    {h.singleName ? (
                      <select
                        value={h.edge ?? 0}
                        onChange={(e) => set(h.id, { edge: Number(e.target.value) })}
                        aria-label="Expected outperformance"
                        className="border border-rule bg-paper px-2 py-1.5 text-sm text-ink focus:border-veld focus:outline-none"
                      >
                        <option value={0}>No view</option>
                        <option value={0.03}>Beats market by 3%/yr</option>
                        <option value={0.06}>by 6%/yr</option>
                        <option value={0.1}>by 10%/yr</option>
                        <option value={0.15}>by 15%/yr</option>
                      </select>
                    ) : (
                      <span className="font-mono text-[0.7rem] text-muted">—</span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-right">
                    <button
                      onClick={() => setHoldings((hs) => hs.filter((x) => x.id !== h.id))}
                      aria-label={`Remove ${h.ticker || "holding"}`}
                      className="px-2 font-mono text-sm text-muted transition-colors hover:text-ink"
                    >
                      ×
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-6 flex flex-wrap items-center gap-8 border border-rule bg-surface p-5">
          <label className="flex items-center gap-3">
            <span className="font-mono text-[0.7rem] uppercase tracking-[0.12em] text-muted">
              How you invest
            </span>
            <select
              value={profile.style}
              onChange={(e) => setProfile({ ...profile, style: e.target.value as Style })}
              className="border border-rule bg-paper px-2 py-1.5 text-sm text-ink focus:border-veld focus:outline-none"
            >
              <option value="barbell">Safe floor + concentrated bets</option>
              <option value="diversified">Spread across everything</option>
            </select>
          </label>
          <label className="flex items-center gap-3">
            <span className="font-mono text-[0.7rem] uppercase tracking-[0.12em] text-muted">
              You spend per year
            </span>
            <select
              value={profile.annualSpending}
              onChange={(e) => setProfile({ ...profile, annualSpending: Number(e.target.value) })}
              className="border border-rule bg-paper px-2 py-1.5 text-sm text-ink focus:border-veld focus:outline-none"
            >
              {[250_000, 500_000, 750_000, 1_000_000, 2_000_000].map((v) => (
                <option key={v} value={v}>{money(v)}</option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-3">
            <span className="font-mono text-[0.7rem] uppercase tracking-[0.12em] text-muted">
              Worst drop you could sit through
            </span>
            <select
              value={profile.drawdownTolerance}
              onChange={(e) => setProfile({ ...profile, drawdownTolerance: Number(e.target.value) })}
              className="border border-rule bg-paper px-2 py-1.5 text-sm text-ink focus:border-veld focus:outline-none"
            >
              {[0.15, 0.25, 0.35, 0.45, 0.55, 0.65].map((v) => (
                <option key={v} value={v}>{Math.round(v * 100)}%</option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-3">
            <span className="font-mono text-[0.7rem] uppercase tracking-[0.12em] text-muted">
              Your tax rate on gains
            </span>
            <select
              value={profile.ltcgRate}
              onChange={(e) => setProfile({ ...profile, ltcgRate: Number(e.target.value) })}
              className="border border-rule bg-paper px-2 py-1.5 text-sm text-ink focus:border-veld focus:outline-none"
            >
              <option value={0.238}>23.8% — no state tax</option>
              <option value={0.291}>29.1% — mid-tax state</option>
              <option value={0.371}>37.1% — California / NY top rate</option>
            </select>
          </label>
        </div>

        {/* ---------------- summary ---------------- */}
        <div className="mt-12 grid gap-px border border-rule bg-rule sm:grid-cols-2 lg:grid-cols-4">
          {[
            { v: money(a.total), l: "Total portfolio" },
            { v: pct(a.currentVol), l: `Risk today — expect a ${pct(-a.expectedWorstDrawdown)} drop in a bad market` },
            { v: money(a.unrealizedGain), l: "Unrealized gain" },
            { v: money(a.harvestBenefit + a.locationSavings), l: "Found in tax savings" },
          ].map((s) => (
            <div key={s.l} className="bg-surface p-6">
              <div className="tnum font-mono text-2xl font-medium tracking-tight text-veld">{s.v}</div>
              <div className="mt-2 text-sm leading-snug text-muted">{s.l}</div>
            </div>
          ))}
        </div>

        {/* ---------------- allocation ---------------- */}
        <h2 className="mt-16 font-display text-2xl font-normal text-ink">
          What you own vs what you should own
        </h2>
        <p className="mt-3 max-w-2xl leading-relaxed text-ink-2">
          The target comes from the worst drop you said you could sit through. A
          portfolio you abandon at the bottom returns nothing, however good it
          looked on paper.
        </p>

        <div className="mt-8 border border-rule bg-surface p-6 md:p-8">
          <ul className="flex flex-col gap-5">
            {a.gaps.map((g) => {
              const max = Math.max(...a.gaps.map((x) => Math.max(x.current, x.target)), 0.01);
              return (
                <li key={g.assetClass}>
                  <div className="flex flex-wrap items-baseline justify-between gap-3">
                    <span className="text-[0.95rem] text-ink">{ASSET_LABEL[g.assetClass]}</span>
                    <span className="tnum font-mono text-sm text-ink-2">
                      {pct(g.current)} <span className="text-muted">now</span> · {pct(g.target)}{" "}
                      <span className="text-muted">target</span>
                      {Math.abs(g.deltaDollars) > 1000 && (
                        <span className={g.deltaDollars > 0 ? "text-veld" : "text-ochre"}>
                          {" "}· {g.deltaDollars > 0 ? "buy" : "sell"} {money(Math.abs(g.deltaDollars))}
                        </span>
                      )}
                    </span>
                  </div>
                  <div className="mt-2 flex flex-col gap-1">
                    <div className="h-2 w-full bg-[#e6e4dc]">
                      <div className="h-full bg-[#24352b]" style={{ width: `${(g.current / max) * 100}%` }} />
                    </div>
                    <div className="h-2 w-full bg-[#e6e4dc]">
                      <div className="h-full bg-[#6f9079]" style={{ width: `${(g.target / max) * 100}%` }} />
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
          <div className="mt-7 flex flex-wrap gap-6 border-t border-rule pt-5 font-mono text-[0.68rem] text-muted">
            <span className="flex items-center gap-2">
              <span className="inline-block h-2.5 w-2.5 rounded-[2px] bg-[#24352b]" /> What you own now
            </span>
            <span className="flex items-center gap-2">
              <span className="inline-block h-2.5 w-2.5 rounded-[2px] bg-[#6f9079]" /> Target
            </span>
            <span className="ml-auto">
              Today: {pct(a.currentReturn)} expected return at {pct(a.currentVol)} risk · Target:{" "}
              {pct(a.targetReturn)} at {pct(a.targetVol)}
            </span>
          </div>
        </div>

        {/* ---------------- per position ---------------- */}
        <h2 className="mt-16 font-display text-2xl font-normal text-ink">
          Every holding, and why
        </h2>
        <div className="mt-8 flex flex-col gap-px bg-rule">
          {a.positions.map((p) => (
            <div key={p.ticker + p.account} className="bg-surface p-6 md:p-8">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div className="flex items-start gap-3">
                  <span
                    className="mt-2 inline-block h-2.5 w-2.5 shrink-0 rounded-full"
                    style={{ background: sev[p.severity].dot }}
                    aria-hidden="true"
                  />
                  <div>
                    <div className="flex flex-wrap items-baseline gap-x-3">
                      <span className="font-mono text-base font-medium text-ink">{p.ticker}</span>
                      <span className="font-mono text-[0.68rem] uppercase tracking-[0.1em] text-muted">
                        {ACCOUNT_LABEL[p.account]} · {ASSET_LABEL[p.assetClass]}
                      </span>
                    </div>
                    <p className="mt-1.5 font-display text-lg leading-snug text-ink">{p.headline}</p>
                  </div>
                </div>
                <div className="text-right">
                  <div className="tnum font-mono text-lg font-medium text-ink">{money(p.value)}</div>
                  <div className="tnum font-mono text-[0.7rem] text-muted">
                    {pct(p.weight)} of portfolio
                  </div>
                </div>
              </div>

              <div className="mt-5 grid gap-5 md:grid-cols-[1.35fr_1fr] md:pl-6">
                <div className="flex flex-col gap-3">
                  {p.reasons.map((r, i) => (
                    <p key={i} className="text-[0.95rem] leading-relaxed text-ink-2">{r}</p>
                  ))}
                </div>
                {p.actions.length > 0 && (
                  <div className="border-l-2 border-ochre pl-5">
                    <div className="font-mono text-[0.66rem] uppercase tracking-[0.13em] text-muted">
                      What to do
                    </div>
                    <ul className="mt-3 flex flex-col gap-2">
                      {p.actions.map((x, i) => (
                        <li key={i} className="text-[0.95rem] leading-snug text-ink">{x}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              <div className="mt-5 flex flex-wrap gap-x-8 gap-y-1 border-t border-rule pt-4 font-mono text-[0.68rem] text-muted md:pl-6">
                <span>Gain: <span className="tnum text-ink-2">{money(p.gain)}</span></span>
                <span>Tax if sold today: <span className="tnum text-ink-2">{money(p.taxIfSold)}</span></span>
                {p.harvestable > 0 && (
                  <span>Loss available: <span className="tnum text-ink-2">{money(p.harvestable)}</span></span>
                )}
              </div>
            </div>
          ))}
        </div>

        <p className="mt-12 max-w-3xl border-t border-rule pt-6 font-mono text-[0.7rem] leading-relaxed text-muted">
          This is a calculator, not advice. It applies general rules to numbers
          you typed in and cannot know your full circumstances. Marula is not a
          registered investment adviser and is not paid for anything shown here.
          Nothing on this page is a recommendation to buy or sell any security.
          Talk to a qualified adviser and a CPA before acting on any of it.
        </p>
      </div>
    </>
  );
}
