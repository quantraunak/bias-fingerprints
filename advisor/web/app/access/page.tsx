import type { Metadata } from "next";
import { Section, Eyebrow } from "@/components/primitives";

export const metadata: Metadata = {
  title: "Access — Marula",
  description:
    "Request a balance-sheet review. What we need, what you receive, and how the engagement works.",
};

const DELIVERABLES = [
  {
    title: "Diagnostic",
    body: "Concentration, liquidity coverage, embedded gain, spending sustainability, and the drag your current structure is paying. Stated in dollars.",
  },
  {
    title: "Policy portfolio",
    body: "A strategic allocation solved against your drawdown tolerance and genuine liquidity capacity, with the gap to your current holdings priced by the tax cost of closing it.",
  },
  {
    title: "Ranked actions",
    body: "Every recommendation carries an annual value, a present value, a time horizon, and a confidence tier — separating what is arithmetic from what depends on a forecast.",
  },
  {
    title: "Projection",
    body: "Your balance sheet run forward under regime-aware, fat-tailed, after-tax simulation. Goal funding probabilities, sustainable spending, and the drawdown you should expect to live through.",
  },
];

const NEEDED = [
  ["Account statements", "Every account, including trusts and retirement wrappers"],
  ["Cost basis by lot", "The single most valuable input — averages understate what is harvestable"],
  ["Spending and income", "Net draw on the portfolio, and any outside income"],
  ["Existing structures", "Trusts, entities, prior gifts, and remaining exemption used"],
  ["Unfunded commitments", "Capital calls still outstanding to private funds"],
];

export default function Access() {
  return (
    <>
      <section className="border-b border-rule">
        <div className="mx-auto max-w-6xl px-6 pb-20 pt-20 md:pt-28">
          <Eyebrow>Access</Eyebrow>
          <h1 className="mt-8 max-w-4xl text-balance font-display text-4xl font-light leading-[1.1] tracking-[-0.02em] text-ink md:text-6xl">
            Start with the review.
          </h1>
          <p className="mt-8 max-w-2xl font-display text-xl leading-relaxed text-ink-2">
            We do not take a mandate before we have seen the whole balance
            sheet. The first deliverable is the analysis itself — you are free
            to take it and implement elsewhere.
          </p>
        </div>
      </section>

      <Section>
        <div className="grid gap-16 lg:grid-cols-[1.05fr_0.95fr]">
          {/* ---- form ---- */}
          <div>
            <Eyebrow>Request a review</Eyebrow>
            <form
              className="mt-8 space-y-6"
              action="mailto:raunak.sood@gmail.com"
              method="post"
              encType="text/plain"
            >
              <Field id="name" label="Name" type="text" autoComplete="name" />
              <Field
                id="email"
                label="Email"
                type="email"
                autoComplete="email"
              />
              <Field
                id="entity"
                label="Family office or entity"
                type="text"
                required={false}
              />

              <div>
                <label
                  htmlFor="size"
                  className="block font-mono text-[0.7rem] uppercase tracking-[0.14em] text-muted"
                >
                  Approximate investable balance sheet
                </label>
                <select
                  id="size"
                  name="size"
                  className="mt-2.5 w-full border border-rule-strong bg-surface px-4 py-3 text-[0.95rem] text-ink transition-colors focus:border-veld focus:outline-none"
                  defaultValue=""
                >
                  <option value="" disabled>
                    Select a range
                  </option>
                  <option>Under $10M</option>
                  <option>$10M – $25M</option>
                  <option>$25M – $50M</option>
                  <option>$50M – $100M</option>
                  <option>Over $100M</option>
                </select>
              </div>

              <div>
                <label
                  htmlFor="context"
                  className="block font-mono text-[0.7rem] uppercase tracking-[0.14em] text-muted"
                >
                  What prompted this
                </label>
                <textarea
                  id="context"
                  name="context"
                  rows={5}
                  placeholder="A liquidity event, a concentrated position, an estate deadline, or a review that is overdue."
                  className="mt-2.5 w-full resize-y border border-rule-strong bg-surface px-4 py-3 text-[0.95rem] leading-relaxed text-ink placeholder:text-muted/70 transition-colors focus:border-veld focus:outline-none"
                />
              </div>

              <button
                type="submit"
                className="w-full bg-veld px-7 py-4 font-mono text-[0.72rem] uppercase tracking-[0.15em] text-white transition-colors hover:bg-veld-2 sm:w-auto"
              >
                Send request
              </button>

              <p className="font-mono text-[0.68rem] leading-relaxed text-muted">
                Submitting this form opens your mail client. Connect it to a
                form endpoint or CRM before launch — do not collect statements
                or account data over unencrypted email.
              </p>
            </form>
          </div>

          {/* ---- what you receive ---- */}
          <div className="space-y-12">
            <div>
              <Eyebrow>What you receive</Eyebrow>
              <div className="mt-8 grid gap-px bg-rule">
                {DELIVERABLES.map((d) => (
                  <div key={d.title} className="bg-surface p-6">
                    <h3 className="font-display text-xl font-normal text-ink">
                      {d.title}
                    </h3>
                    <p className="mt-2 text-[0.95rem] leading-relaxed text-ink-2">
                      {d.body}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <Eyebrow>What we need</Eyebrow>
              <dl className="mt-8 divide-y divide-rule border-y border-rule">
                {NEEDED.map(([k, v]) => (
                  <div key={k} className="py-5">
                    <dt className="font-mono text-[0.72rem] uppercase tracking-[0.12em] text-ink">
                      {k}
                    </dt>
                    <dd className="mt-1.5 text-[0.95rem] leading-relaxed text-ink-2">
                      {v}
                    </dd>
                  </div>
                ))}
              </dl>
            </div>
          </div>
        </div>
      </Section>
    </>
  );
}

function Field({
  id,
  label,
  type,
  autoComplete,
  required = true,
}: {
  id: string;
  label: string;
  type: string;
  autoComplete?: string;
  required?: boolean;
}) {
  return (
    <div>
      <label
        htmlFor={id}
        className="block font-mono text-[0.7rem] uppercase tracking-[0.14em] text-muted"
      >
        {label}
      </label>
      <input
        id={id}
        name={id}
        type={type}
        autoComplete={autoComplete}
        required={required}
        className="mt-2.5 w-full border border-rule-strong bg-surface px-4 py-3 text-[0.95rem] text-ink transition-colors focus:border-veld focus:outline-none"
      />
    </div>
  );
}
