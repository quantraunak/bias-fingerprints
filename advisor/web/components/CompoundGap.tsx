"use client";

import { useEffect, useRef, useState } from "react";

/**
 * The gap between a gross return and the one that actually compounded.
 *
 * A single headline idea, so it gets a hero number alongside the plot rather
 * than a dense multi-series chart.
 */
const YEARS = 30;
const GROSS = 0.075;
const DRAG = 0.019; // typical all-in tax and fee drag for this cohort

const gross = Array.from({ length: YEARS + 1 }, (_, i) => (1 + GROSS) ** i);
const net = Array.from({ length: YEARS + 1 }, (_, i) => (1 + GROSS - DRAG) ** i);

export default function CompoundGap() {
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
      { threshold: 0.3 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const W = 520;
  const H = 260;
  const padL = 34;
  const padB = 28;
  const max = 9.5;

  const x = (i: number) => padL + (i / YEARS) * (W - padL - 12);
  const y = (v: number) => H - padB - (v / max) * (H - padB - 10);
  const line = (d: number[]) =>
    d.map((v, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(v)}`).join(" ");
  const band = `${line(gross)} L${x(YEARS)},${y(net[YEARS])} ${net
    .slice()
    .reverse()
    .map((v, i) => `L${x(YEARS - i)},${y(v)}`)
    .join(" ")} Z`;

  const lost = (gross[YEARS] - net[YEARS]) / gross[YEARS];

  return (
    <div ref={ref} className="border border-rule bg-surface p-6 md:p-8">
      <div className="flex flex-wrap items-baseline justify-between gap-3 border-b border-rule pb-4">
        <h3 className="font-mono text-[0.72rem] font-medium uppercase tracking-[0.14em] text-ink">
          Growth of $1 over 30 years
        </h3>
        <span className="font-mono text-[0.66rem] uppercase tracking-[0.1em] text-ochre">
          Illustrative
        </span>
      </div>

      <div className="mt-6 grid gap-8 md:grid-cols-[1fr_auto] md:items-center">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="h-auto w-full"
          role="img"
          aria-label={`At a 7.5% gross return, $1 grows to $${gross[YEARS].toFixed(2)} over thirty years. After 1.9 points of annual tax and fee drag it reaches only $${net[YEARS].toFixed(2)} — ${Math.round(lost * 100)}% less.`}
        >
          {[2, 4, 6, 8].map((g) => (
            <g key={g}>
              <line
                x1={padL}
                x2={W - 12}
                y1={y(g)}
                y2={y(g)}
                stroke="var(--color-rule)"
              />
              <text
                x={padL - 8}
                y={y(g) + 4}
                textAnchor="end"
                className="fill-[var(--color-muted)] font-mono text-[10px]"
              >
                {g}×
              </text>
            </g>
          ))}
          {[0, 10, 20, 30].map((i) => (
            <text
              key={i}
              x={x(i)}
              y={H - 8}
              textAnchor="middle"
              className="fill-[var(--color-muted)] font-mono text-[10px]"
            >
              {i === 0 ? "Now" : `Y${i}`}
            </text>
          ))}

          <path
            d={band}
            fill="var(--color-ochre)"
            opacity={shown ? 0.16 : 0}
            style={{ transition: "opacity 1s ease 0.6s" }}
          />
          <path
            d={line(gross)}
            fill="none"
            stroke="var(--color-muted)"
            strokeWidth="1.5"
            strokeDasharray="4 4"
            style={{
              opacity: shown ? 1 : 0,
              transition: "opacity 0.9s ease 0.3s",
            }}
          />
          <path
            d={line(net)}
            fill="none"
            stroke="var(--color-veld)"
            strokeWidth="2.5"
            strokeLinecap="round"
            style={{
              strokeDasharray: 900,
              strokeDashoffset: shown ? 0 : 900,
              transition: "stroke-dashoffset 1.7s cubic-bezier(0.22,1,0.36,1)",
            }}
          />
          <text
            x={x(YEARS) - 4}
            y={y(gross[YEARS]) - 8}
            textAnchor="end"
            className="fill-[var(--color-muted)] font-mono text-[11px]"
          >
            Gross {gross[YEARS].toFixed(1)}×
          </text>
          <text
            x={x(YEARS) - 4}
            y={y(net[YEARS]) + 26}
            textAnchor="end"
            className="fill-[var(--color-veld)] font-mono text-[11px] font-medium"
          >
            Kept {net[YEARS].toFixed(1)}×
          </text>
        </svg>

        <div className="md:w-40">
          <div className="tnum font-mono text-5xl font-medium leading-none tracking-tight text-ochre">
            {Math.round(lost * 100)}%
          </div>
          <p className="mt-3 text-sm leading-snug text-ink-2">
            of terminal wealth lost to 1.9 points of annual drag.
          </p>
        </div>
      </div>

      <p className="mt-5 border-t border-rule pt-4 font-mono text-[0.68rem] leading-relaxed text-muted">
        7.5% gross versus 5.6% after a representative all-in tax and fee load.
        Illustrative arithmetic, not a projection of any portfolio.
      </p>
    </div>
  );
}
