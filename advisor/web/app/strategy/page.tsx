import type { Metadata } from "next";
import Link from "next/link";
import { Section, Eyebrow, DataTable } from "@/components/primitives";

export const metadata: Metadata = {
  title: "Strategy — Marula",
  description:
    "How Marula builds an after-tax portfolio: capital market assumptions, the tax-aware long/short extension, asset location, and structure.",
};

const STAGES = [
  {
    label: "Assumptions",
    title: "Forward-looking, built from observable inputs",
    body: "Historical average returns are the worst available forecast — they peak exactly when valuations are richest. Every expected return we use is built bottom-up: equities from dividend and buyback yield, real growth, inflation, and a valuation drift term; bonds from starting yield, which explains the overwhelming majority of subsequent ten-year returns. Private-market correlations are de-smoothed, because appraisal-based marks understate true equity correlation by roughly half.",
  },
  {
    label: "Policy",
    title: "A volatility budget, not a risk questionnaire",
    body: "A stated drawdown tolerance converts into an annual volatility ceiling, and that ceiling binds the optimizer. Illiquid assets are capped by what the family can genuinely fund through a stress — spending, plus unfunded capital calls, against liquid assets marked down. A policy the family abandons at the bottom has a realized return of zero, whatever its expected return was.",
  },
  {
    label: "Implementation",
    title: "Tax enters the optimizer, not the post-mortem",
    body: "The equity sleeve runs as a long/short extension holding full market exposure. Because there are positions on both sides, losses are available to harvest in rising and falling markets alike — long-only direct indexing exhausts its capacity after a few years of appreciation. Lot-level basis is a constraint in the optimization, so every trade is scored against the tax it creates.",
  },
  {
    label: "Location",
    title: "Same holdings, different wrappers",
    body: "High-drag assets belong where drag is not paid. That is usually not the conventional answer: at current yields, high-turnover alternatives and credit deserve the shelter more than low-coupon core bonds. Solved as an assignment problem across every account the family holds, subject to what each wrapper can actually contain.",
  },
  {
    label: "Structure",
    title: "The largest number on most balance sheets",
    body: "For an estate projected through the exemption, transfer tax dwarfs any plausible alpha. We model the exposure on the projected balance sheet rather than today's, and price each structure against the basis step-up it costs — because gifted assets carry over basis, and that is a real expense set against the transfer-tax saving.",
  },
];

export default function Strategy() {
  return (
    <>
      <section className="border-b border-rule">
        <div className="mx-auto max-w-6xl px-6 pb-20 pt-20 md:pt-28">
          <Eyebrow>Strategy</Eyebrow>
          <h1 className="mt-8 max-w-4xl text-balance font-display text-4xl font-light leading-[1.1] tracking-[-0.02em] text-ink md:text-6xl">
            Five decisions, in the order they actually matter.
          </h1>
          <p className="mt-8 max-w-2xl font-display text-xl leading-relaxed text-ink-2">
            Manager selection is the conversation most families have. It is
            close to the least important thing on this list.
          </p>
        </div>
      </section>

      <Section>
        <ol className="grid gap-px bg-rule">
          {STAGES.map((s, i) => (
            <li
              key={s.label}
              className="grid gap-6 bg-paper p-8 md:grid-cols-[auto_10rem_1fr] md:gap-10 md:p-10"
            >
              <div className="tnum font-mono text-sm text-ochre">
                {String(i + 1).padStart(2, "0")}
              </div>
              <div className="font-mono text-[0.72rem] uppercase tracking-[0.15em] text-muted">
                {s.label}
              </div>
              <div>
                <h2 className="font-display text-2xl font-normal leading-snug text-ink">
                  {s.title}
                </h2>
                <p className="mt-4 max-w-2xl leading-relaxed text-ink-2">
                  {s.body}
                </p>
              </div>
            </li>
          ))}
        </ol>
      </Section>

      <Section bordered>
        <div className="grid gap-14 md:grid-cols-[0.85fr_1.15fr]">
          <div>
            <Eyebrow>The extension</Eyebrow>
            <h2 className="mt-6 text-balance font-display text-3xl font-light leading-[1.15] text-ink md:text-4xl">
              Why a long/short structure harvests more.
            </h2>
          </div>
          <div className="space-y-6 leading-relaxed text-ink-2">
            <p>
              A long-only portfolio can only harvest a loss where a position has
              fallen below its basis. After several strong years, very few have
              — the capacity dries up precisely when the family has the largest
              embedded gains to offset.
            </p>
            <p>
              Adding a short book restores it. Shorts appreciate when the market
              falls and depreciate when it rises, so in any market some sleeve of
              the portfolio is generating losses. Holding 130% long against 30%
              short keeps net exposure at 100% while roughly doubling the
              positions available to harvest from.
            </p>
            <p>
              The trade-offs are real and we state them plainly: financing costs
              on the short book, borrow availability and recall risk, higher
              tracking error, and greater operational complexity. The pre-tax
              alpha of the strategy should cover its costs before any tax
              benefit is counted. If it does not, the structure is not worth
              owning.
            </p>
          </div>
        </div>

        <div className="mt-16">
          <DataTable
            caption="Harvesting capacity by structure"
            head={["Structure", "Net exposure", "Harvestable positions", "Capacity over time"]}
            rows={[
              ["Index fund or ETF", "100%", "1", "Exhausted after the first decline"],
              ["Direct indexing, long only", "100%", "~250", "Decays sharply after 3–5 years"],
              ["Long/short extension 130/30", "100%", "~325", "Persists — both legs contribute"],
              ["Long/short extension 150/50", "100%", "~375", "Highest, at higher cost and tracking error"],
            ]}
          />
          <p className="mt-6 max-w-2xl font-mono text-[0.7rem] leading-relaxed text-muted">
            Position counts are illustrative for a 250-name implementation.
            Higher extensions increase financing cost, tracking error, and risk
            of loss. Harvested losses have value only to the extent the family
            realizes gains against which to apply them.
          </p>
        </div>
      </Section>

      <Section bordered>
        <div className="max-w-3xl">
          <Eyebrow>What we will not claim</Eyebrow>
          <h2 className="mt-6 text-balance font-display text-3xl font-light leading-[1.15] text-ink md:text-4xl">
            Deferral is not forgiveness.
          </h2>
          <div className="mt-8 space-y-5 leading-relaxed text-ink-2">
            <p>
              Systematic harvesting drives a portfolio's embedded gain steadily
              upward. The tax is deferred, not erased, and a portfolio run this
              way for two decades carries a substantial latent liability. We
              model that liability explicitly and report it alongside the
              benefit, because a strategy whose costs only appear at the end is
              not a strategy.
            </p>
            <p>
              Where deferral becomes permanent — a basis step-up at death, a
              charitable structure, a qualifying small-business exclusion — that
              is a structural outcome we plan toward deliberately, not a
              byproduct we assume.
            </p>
            <p>
              We also do not forecast our way to a target return. Expected
              returns are distributions, and we would rather show a family the
              probability of reaching its goal than a number it will remember as
              a promise.
            </p>
          </div>
          <div className="mt-12">
            <Link
              href="/access"
              className="inline-block bg-veld px-7 py-3.5 font-mono text-[0.72rem] uppercase tracking-[0.15em] text-white transition-colors hover:bg-veld-2"
            >
              Request a portfolio review
            </Link>
          </div>
        </div>
      </Section>
    </>
  );
}
