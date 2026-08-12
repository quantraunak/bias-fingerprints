"use client";

import { useEffect, useRef, useState } from "react";
import {
  FINDINGS,
  FINDINGS_TOTAL,
  FINDINGS_HIGH,
  FINDINGS_ASSETS,
} from "@/app/findings";

/**
 * What the engine found on one real balance sheet.
 *
 * This is the product, so it gets the clearest possible form: one bar per
 * action, sorted by size, in dollars a year. Colour encodes how much the
 * estimate can be trusted — a three-step ramp of one hue, because confidence
 * is ordered, not categorical.
 */
// Three steps of one hue, dark to light. Every step has to stay clearly
// visible against the track behind it — a pale tint reads as an empty bar,
// which is worse than no chart at all.
const TIERS = {
  high: {
    fill: "#24352b",
    label: "Certain — tax rules and fee schedules",
  },
  medium: {
    fill: "#47654f",
    label: "Likely — depends on markets behaving normally",
  },
  low: {
    fill: "#6f9079",
    label: "Uncertain — depends on picking good managers",
  },
} as const;

const money = (n: number) =>
  n >= 1_000_000
    ? `$${(n / 1_000_000).toFixed(2)}M`
    : `$${Math.round(n / 1000)}K`;

export default function Findings() {
  const ref = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(false);
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      // No transition at all — a stagger DELAY survives the global
      // reduced-motion duration override, so bars would still appear one by
      // one for someone who asked for no motion.
      setReduced(true);
      setShown(true);
      return;
    }
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setShown(true);
          io.disconnect();
        }
      },
      { threshold: 0.2 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const max = Math.max(...FINDINGS.map((f) => f.annual));

  return (
    <div ref={ref} className="border border-rule bg-surface">
      <div className="border-b border-rule p-6 md:p-8">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <h3 className="font-mono text-[0.72rem] font-medium uppercase tracking-[0.14em] text-ink">
            A $28M balance sheet, reviewed
          </h3>
          <span className="font-mono text-[0.66rem] uppercase tracking-[0.1em] text-ochre">
            Sample household
          </span>
        </div>
        <div className="mt-6 flex flex-wrap items-end gap-x-12 gap-y-6">
          <div>
            <div className="tnum font-mono text-5xl font-medium leading-none tracking-tight text-veld">
              {money(FINDINGS_HIGH)}
            </div>
            <p className="mt-3 max-w-xs text-sm leading-snug text-ink-2">
              a year, from changes that do not depend on predicting anything —{" "}
              <span className="tnum">
                {((FINDINGS_HIGH / FINDINGS_ASSETS) * 100).toFixed(1)}%
              </span>{" "}
              of the portfolio.
            </p>
          </div>
          <div>
            <div className="tnum font-mono text-2xl font-medium leading-none text-ink-2">
              {money(FINDINGS_TOTAL)}
            </div>
            <p className="mt-2 max-w-[13rem] text-sm leading-snug text-muted">
              a year including the parts that carry real uncertainty.
            </p>
          </div>
        </div>
      </div>

      <div className="p-6 md:p-8">
        <ul className="flex flex-col gap-4">
          {FINDINGS.map((f, i) => {
            const tier = TIERS[f.conf as keyof typeof TIERS];
            return (
              <li key={f.label} className="grid gap-2">
                <div className="flex items-baseline justify-between gap-4">
                  <span className="text-[0.95rem] leading-snug text-ink">
                    {f.label}
                  </span>
                  <span className="tnum shrink-0 font-mono text-sm font-medium text-ink">
                    {money(f.annual)}
                    <span className="text-muted">/yr</span>
                  </span>
                </div>
                <div className="h-2.5 w-full bg-[#e6e4dc]">
                  <div
                    className="h-full rounded-r-[3px]"
                    style={{
                      width: shown ? `${(f.annual / max) * 100}%` : "0%",
                      background: tier.fill,
                      transition: reduced
                        ? "none"
                        : `width 0.9s cubic-bezier(0.22,1,0.36,1) ${i * 80}ms`,
                    }}
                  />
                </div>
              </li>
            );
          })}
        </ul>

        <div className="mt-8 flex flex-wrap gap-x-7 gap-y-2 border-t border-rule pt-5">
          {Object.entries(TIERS).map(([k, t]) => (
            <span
              key={k}
              className="flex items-center gap-2 font-mono text-[0.68rem] text-muted"
            >
              <span
                className="inline-block h-2.5 w-2.5 rounded-[2px]"
                style={{ background: t.fill }}
                aria-hidden="true"
              />
              {t.label}
            </span>
          ))}
        </div>

        <p className="mt-5 font-mono text-[0.68rem] leading-relaxed text-muted">
          Output of our engine run against a representative household: $28M
          across five accounts, a founder stock position at 30% of net worth,
          and an estate heading past the exemption. Every family is different
          and these figures will not be yours.
        </p>
      </div>
    </div>
  );
}
