import Link from "next/link";
import { Section, Eyebrow, StatGrid } from "@/components/primitives";
import Ridgeline from "@/components/Ridgeline";
import Findings from "@/components/Findings";
import HarvestChart from "@/components/HarvestChart";
import Reveal from "@/components/Reveal";

const PILLARS = [
  {
    title: "Stock picking that knows your tax bill",
    body: "Holding an index through individual stocks, with a small short book alongside, means something is always below what you paid for it — so there are always losses to bank against your gains. Every trade gets scored against the tax it creates.",
    detail: "130/30 to 150/50 · full market exposure · rebalanced monthly",
  },
  {
    title: "One portfolio, not six accounts",
    body: "Your brokerage account, your IRA, your Roth and your trust are one pool of money. Which asset belongs in which account, and what gets sold first, are decisions nobody managing a single account can even see.",
    detail: "Account placement · what to sell first · after-tax optimization",
  },
  {
    title: "The position that made the money",
    body: "Most families here got wealthy from one holding. Every way out — hold it, sell in stages, swap into a fund, hedge it, give it away — gets priced against your whole balance sheet after tax, not just the position.",
    detail: "Founder stock · after an IPO or sale · §1202 qualification",
  },
  {
    title: "Getting money to the next generation",
    body: "If your estate is heading past the exemption, the transfer tax is larger than anything an investment decision can add. Each structure gets priced against what it costs you in lost basis step-up.",
    detail: "GRAT · SLAT · IDGT · CLAT · annual gifting",
  },
];

const STEPS = [
  {
    title: "Enter what you own",
    body: "Every holding, which account it sits in, and what you paid for it. Cost basis matters more than anything else you type — it is what decides the tax on every move you might make.",
  },
  {
    title: "It reads them as one portfolio",
    body: "Not five accounts scored separately. One picture of what you own, what you paid, which account holds it, and how much risk the whole thing carries.",
  },
  {
    title: "You get a list, in dollars",
    body: "What to change, what each change is worth per year, and the reasoning behind every one. Take it to your adviser or your accountant and check it.",
  },
];

const RESEARCH = [
  { k: "Sharpe", v: "1.46", note: "95% CI [0.90, 2.07]" },
  { k: "Annualized", v: "35.3%", note: "gross of tax" },
  { k: "Max drawdown", v: "−24.7%", note: "monthly underwater" },
  { k: "Beta to SPY", v: "0.14", note: "beta-constrained" },
];

export default function Home() {
  return (
    <>
      {/* ---------------- Hero ---------------- */}
      <section className="relative isolate overflow-hidden bg-night">
        <Ridgeline />
        {/* Scrim keeps the headline legible over the brightest part of the sky. */}
        <div
          className="absolute inset-0 -z-0"
          style={{
            background:
              "linear-gradient(100deg, rgba(19,24,20,0.94) 0%, rgba(19,24,20,0.76) 40%, rgba(19,24,20,0.14) 74%, rgba(19,24,20,0.0) 100%)",
          }}
          aria-hidden="true"
        />
        <div className="relative mx-auto max-w-6xl px-6 pb-32 pt-28 md:pb-44 md:pt-40">
          <div className="rise max-w-3xl">
            <div className="flex items-center gap-4">
              <span className="font-mono text-[0.7rem] font-medium uppercase tracking-[0.19em] text-dusk-muted">
                Systematic investment management
              </span>
              <span className="h-px w-16 bg-dusk-muted/40" aria-hidden="true" />
            </div>
            <h1 className="mt-8 text-balance font-display text-5xl font-light leading-[1.05] tracking-[-0.02em] text-dusk-ink md:text-7xl">
              The return you keep is the only one that{" "}
              <em className="font-normal italic text-ochre-2">compounded</em>.
            </h1>
            <p className="mt-8 max-w-2xl font-display text-xl leading-relaxed text-dusk-ink/80 md:text-2xl">
              A free tool that reviews a family's whole balance sheet the way
              an institution would — every account at once, after tax. Put your
              holdings in and see what to change, and why.
            </p>
            <div className="mt-12 flex flex-wrap items-center gap-4">
              <Link
                href="/tool"
                className="bg-ochre px-7 py-3.5 font-mono text-[0.72rem] uppercase tracking-[0.15em] text-white transition-colors hover:bg-ochre-2"
              >
                Review your portfolio
              </Link>
              <Link
                href="/strategy"
                className="border border-dusk-muted/50 px-7 py-3.5 font-mono text-[0.72rem] uppercase tracking-[0.15em] text-dusk-ink transition-colors hover:border-dusk-ink hover:bg-dusk-ink/10"
              >
                How it works
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* ---------------- The problem ---------------- */}
      <Section>
        <div className="grid gap-14 md:grid-cols-[0.85fr_1.15fr]">
          <div>
            <Eyebrow>The premise</Eyebrow>
            <h2 className="mt-6 text-balance font-display text-3xl font-light leading-[1.15] text-ink md:text-4xl">
              Most wealth at this level is managed pre-tax, one account at a
              time.
            </h2>
          </div>
          <div className="space-y-6 text-[1.0625rem] leading-relaxed text-ink-2">
            <p>
              A family with $30 million typically holds it across a taxable
              brokerage account, a rollover IRA, a Roth, a trust, and two
              private funds — each managed to its own benchmark, by people who
              cannot see the others. The reporting is consolidated. The
              decisions are not.
            </p>
            <p>
              What falls between them is the money. Selling down a concentrated
              stock creates gains — and those gains are exactly what makes loss
              harvesting worth doing, if anyone connects the two. Which account
              holds an asset can change its return by half a point a year.
              Whether to gift appreciated stock or hold it depends on an estate
              projection nobody is running.
            </p>
            <p>
              Each of these is worth more than the manager selection that takes
              up the whole conversation. None of them can be seen from inside a
              single account.
            </p>
            <p className="border-l-2 border-ochre pl-6 font-display text-lg italic leading-relaxed text-ink">
              The reliable edge for a family this size is not a better forecast.
              It is arithmetic nobody is doing.
            </p>
          </div>
        </div>

        <Reveal>
          <div className="mt-16">
            <Findings />
          </div>
        </Reveal>
      </Section>

      {/* ---------------- How it works ---------------- */}
      <Section bordered>
        <Eyebrow>How it works</Eyebrow>
        <h2 className="mt-6 max-w-3xl text-balance font-display text-3xl font-light leading-[1.15] text-ink md:text-4xl">
          Three steps, about ten minutes.
        </h2>

        <ol className="mt-14 grid gap-px bg-rule md:grid-cols-3">
          {STEPS.map((s, i) => (
            <Reveal key={s.title} delay={i * 80}>
              <li className="h-full bg-paper p-8 md:p-10">
                <div className="tnum font-mono text-sm text-ochre">
                  {String(i + 1).padStart(2, "0")}
                </div>
                <h3 className="mt-5 font-display text-2xl font-normal leading-snug text-ink">
                  {s.title}
                </h3>
                <p className="mt-4 leading-relaxed text-ink-2">{s.body}</p>
              </li>
            </Reveal>
          ))}
        </ol>
      </Section>

      {/* ---------------- What we do ---------------- */}
      <Section bordered>
        <Eyebrow>What we do</Eyebrow>
        <h2 className="mt-6 max-w-3xl text-balance font-display text-3xl font-light leading-[1.15] text-ink md:text-4xl">
          Four mandates, run as one portfolio.
        </h2>

        <div className="mt-14 grid gap-px bg-rule md:grid-cols-2">
          {PILLARS.map((p, i) => (
            <Reveal key={p.title} delay={i * 70}>
              <div className="h-full bg-paper p-8 transition-colors duration-300 hover:bg-surface md:p-10">
                <h3 className="font-display text-2xl font-normal leading-snug text-ink">
                  {p.title}
                </h3>
                <p className="mt-4 leading-relaxed text-ink-2">{p.body}</p>
                <p className="mt-6 border-t border-rule pt-4 font-mono text-[0.7rem] leading-relaxed tracking-wide text-muted">
                  {p.detail}
                </p>
              </div>
            </Reveal>
          ))}
        </div>

        <Reveal>
          <div className="mt-16">
            <HarvestChart />
          </div>
        </Reveal>
      </Section>

      {/* ---------------- Research ---------------- */}
      <Section bordered>
        <div className="grid gap-14 md:grid-cols-[1fr_1fr]">
          <div>
            <Eyebrow>Research</Eyebrow>
            <h2 className="mt-6 text-balance font-display text-3xl font-light leading-[1.15] text-ink md:text-4xl">
              The signal engine, and what we are willing to claim for it.
            </h2>
            <div className="mt-8 space-y-5 leading-relaxed text-ink-2">
              <p>
                Our equity model ranks a large-cap universe cross-sectionally on
                momentum, volatility, trend, and microstructure factors using
                gradient-boosted trees, trained with purged walk-forward
                validation and a 21-day embargo so no future information reaches
                the training set. Positions are sized by a convex optimizer
                under dollar-neutral and beta-neutral constraints with
                Ledoit-Wolf shrinkage on the covariance matrix.
              </p>
              <p>
                The figures beside this are from that research backtest. They
                are simulated, gross of tax, and run on a universe with
                survivorship bias — which flatters them. We publish the
                confidence interval rather than the point estimate because the
                interval is the honest object, and we size the strategy off its
                lower bound.
              </p>
            </div>
          </div>

          <div>
            <div className="border border-rule bg-surface p-8">
              <div className="flex items-baseline justify-between gap-4 border-b border-rule pb-4">
                <div className="font-mono text-[0.72rem] font-medium uppercase tracking-[0.14em] text-ink">
                  Research backtest
                </div>
                <div className="font-mono text-[0.66rem] uppercase tracking-[0.1em] text-ochre">
                  Simulated · not live
                </div>
              </div>
              <dl className="mt-2 divide-y divide-rule">
                {RESEARCH.map((r) => (
                  <div
                    key={r.k}
                    className="flex items-baseline justify-between gap-6 py-5"
                  >
                    <dt className="font-mono text-[0.72rem] uppercase tracking-[0.12em] text-muted">
                      {r.k}
                    </dt>
                    <dd className="text-right">
                      <div className="tnum font-mono text-2xl font-medium text-ink">
                        {r.v}
                      </div>
                      <div className="tnum mt-1 font-mono text-[0.68rem] text-muted">
                        {r.note}
                      </div>
                    </dd>
                  </div>
                ))}
              </dl>
              <p className="mt-2 border-t border-rule pt-4 font-mono text-[0.66rem] leading-relaxed text-muted">
                2010–2024 · monthly rebalance · net of 1bp commission and 5bp
                slippage on turnover · Sharpe interval from a 10,000-sample
                block bootstrap.
              </p>
            </div>
          </div>
        </div>
      </Section>

      {/* ---------------- Who we work with ---------------- */}
      <Section bordered>
        <Eyebrow>Who we work with</Eyebrow>
        <h2 className="mt-6 max-w-3xl text-balance font-display text-3xl font-light leading-[1.15] text-ink md:text-4xl">
          Families for whom the tax and structure decisions have become larger
          than the investment decisions.
        </h2>

        <StatGrid
          items={[
            { v: "$10–50M", l: "Typical investable balance sheet" },
            { v: "Post-liquidity", l: "Founders, operators, and early employees" },
            { v: "Single-family", l: "Offices without an in-house quant team" },
            { v: "Multi-generational", l: "Horizons measured in decades" },
          ]}
        />

        <div className="mt-14 max-w-2xl space-y-5 leading-relaxed text-ink-2">
          <p>
            The tool reviews the whole balance sheet: every account, every
            holding, the embedded gain, and the risk you are actually carrying.
            It returns a ranked set of actions with a dollar value attached to
            each, and separates the ones that are pure arithmetic from the ones
            that depend on a forecast.
          </p>
          <p>
            Most of the value in that first report is mechanical. That is the
            point.
          </p>
        </div>

        <div className="mt-12">
          <Link
            href="/tool"
            className="inline-block bg-veld px-7 py-3.5 font-mono text-[0.72rem] uppercase tracking-[0.15em] text-white transition-colors hover:bg-veld-2"
          >
            Review your portfolio
          </Link>
        </div>
      </Section>
    </>
  );
}
