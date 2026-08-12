import type { ReactNode } from "react";

export function Eyebrow({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-center gap-4">
      <span className="eyebrow">{children}</span>
      <span className="h-px flex-1 bg-rule" aria-hidden="true" />
    </div>
  );
}

export function Section({
  children,
  bordered = false,
}: {
  children: ReactNode;
  bordered?: boolean;
}) {
  return (
    <section className={bordered ? "border-t border-rule" : undefined}>
      <div className="mx-auto max-w-6xl px-6 py-24 md:py-28">{children}</div>
    </section>
  );
}

export function StatGrid({
  items,
}: {
  items: { v: string; l: string }[];
}) {
  return (
    <div className="mt-12 grid gap-px border border-rule bg-rule sm:grid-cols-2 lg:grid-cols-4">
      {items.map((s) => (
        <div key={s.l} className="bg-surface p-7">
          <div className="tnum font-mono text-2xl font-medium tracking-tight text-veld">
            {s.v}
          </div>
          <div className="mt-3 text-sm leading-snug text-muted">{s.l}</div>
        </div>
      ))}
    </div>
  );
}

export function DataTable({
  caption,
  head,
  rows,
}: {
  caption?: string;
  head: string[];
  rows: ReactNode[][];
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[36rem] border-collapse">
        {caption ? (
          <caption className="pb-4 text-left font-mono text-[0.72rem] font-medium uppercase tracking-[0.13em] text-ink">
            {caption}
          </caption>
        ) : null}
        <thead>
          <tr>
            {head.map((h, i) => (
              <th
                key={h}
                className={`border-b border-rule-strong pb-3 pr-6 font-mono text-[0.66rem] font-medium uppercase tracking-[0.11em] text-muted ${
                  i === 0 ? "text-left" : "text-right"
                }`}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, ri) => (
            <tr key={ri}>
              {r.map((cell, ci) => (
                <td
                  key={ci}
                  className={`tnum border-b border-rule py-4 pr-6 align-top text-sm ${
                    ci === 0
                      ? "font-medium text-ink"
                      : "text-right font-mono text-ink-2"
                  }`}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
