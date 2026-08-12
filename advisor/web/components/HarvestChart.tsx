"use client";

import { useEffect, useRef, useState } from "react";
import { HARVEST_LONG_ONLY, HARVEST_LONG_SHORT } from "@/app/data";

/**
 * Harvesting capacity by year — the single clearest argument for the extension.
 *
 * Two series, so a legend is always present and both are direct-labeled at the
 * point of maximum separation rather than at every data point.
 */
export default function HarvestChart() {
  const ref = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
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
      { threshold: 0.35 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const years = HARVEST_LONG_ONLY.length;
  const max = 0.26;
  const W = 720;
  const H = 300;
  const padL = 46;
  const padB = 34;
  const padT = 12;

  const x = (i: number) => padL + (i / (years - 1)) * (W - padL - 16);
  const y = (v: number) => padT + (1 - v / max) * (H - padT - padB);

  const line = (data: number[]) =>
    data.map((v, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(v)}`).join(" ");

  const area = (data: number[]) =>
    `${line(data)} L${x(years - 1)},${y(0)} L${x(0)},${y(0)} Z`;

  return (
    <div ref={ref} className="border border-rule bg-surface p-6 md:p-8">
      <div className="flex flex-wrap items-baseline justify-between gap-3 border-b border-rule pb-4">
        <h3 className="font-mono text-[0.72rem] font-medium uppercase tracking-[0.14em] text-ink">
          Harvestable losses by year
        </h3>
        <span className="font-mono text-[0.66rem] uppercase tracking-[0.1em] text-ochre">
          Simulated · % of portfolio
        </span>
      </div>

      <div className="mt-6 overflow-x-auto">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="h-auto w-full min-w-[32rem]"
          role="img"
          aria-label="Harvestable losses decay from 15.5% to 1% of portfolio value over twelve years in a long-only portfolio, while a long/short extension starts at 24% and remains above 6% throughout."
        >
          {[0, 0.05, 0.1, 0.15, 0.2, 0.25].map((g) => (
            <g key={g}>
              <line
                x1={padL}
                x2={W - 16}
                y1={y(g)}
                y2={y(g)}
                stroke="var(--color-rule)"
                strokeWidth="1"
              />
              <text
                x={padL - 10}
                y={y(g) + 4}
                textAnchor="end"
                className="fill-[var(--color-muted)] font-mono text-[10px]"
              >
                {Math.round(g * 100)}%
              </text>
            </g>
          ))}

          {[0, 3, 6, 9, 11].map((i) => (
            <text
              key={i}
              x={x(i)}
              y={H - 12}
              textAnchor="middle"
              className="fill-[var(--color-muted)] font-mono text-[10px]"
            >
              {`Y${i + 1}`}
            </text>
          ))}

          <g
            style={{
              opacity: shown ? 1 : 0,
              transition: "opacity 1s ease 0.1s",
            }}
          >
            <path d={area(HARVEST_LONG_SHORT)} fill="var(--color-veld)" opacity="0.09" />
            <path
              d={line(HARVEST_LONG_SHORT)}
              fill="none"
              stroke="var(--color-veld)"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              style={{
                strokeDasharray: 1400,
                strokeDashoffset: shown ? 0 : 1400,
                transition: "stroke-dashoffset 1.6s cubic-bezier(0.22,1,0.36,1)",
              }}
            />
            <path
              d={line(HARVEST_LONG_ONLY)}
              fill="none"
              stroke="var(--color-ochre)"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeDasharray="5 4"
              style={{
                opacity: shown ? 1 : 0,
                transition: "opacity 1.1s ease 0.5s",
              }}
            />

            <text
              x={x(6)}
              y={y(HARVEST_LONG_SHORT[6]) - 12}
              className="fill-[var(--color-veld)] font-mono text-[11px] font-medium"
            >
              Long/short extension
            </text>
            <text
              x={x(6)}
              y={y(HARVEST_LONG_ONLY[6]) + 20}
              className="fill-[var(--color-ochre)] font-mono text-[11px] font-medium"
            >
              Long only
            </text>
          </g>
        </svg>
      </div>

      <p className="mt-5 border-t border-rule pt-4 font-mono text-[0.68rem] leading-relaxed text-muted">
        Lot-level simulation across 250 names, 5% harvest threshold, 31-day wash-sale
        block. A long-only portfolio exhausts its capacity as positions
        appreciate above basis; a short book keeps generating losses in rising
        markets. Losses have value only against realized gains.
      </p>
    </div>
  );
}
