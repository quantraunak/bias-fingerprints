/**
 * Portfolio analysis — runs entirely in the browser.
 *
 * Nothing here touches a network. The user's holdings never leave their
 * machine, which is both the right privacy posture for financial data and the
 * reason this is a calculator rather than a service that receives client
 * information.
 *
 * Ported from the Python research engine (engine/cma.py, engine/tax/*).
 */

export type AccountType = "taxable" | "ira" | "roth" | "trust";

export type AssetClass =
  | "us_equity"
  | "intl_equity"
  | "em_equity"
  | "bonds"
  | "credit"
  | "real_assets"
  | "cash";

export interface Holding {
  id: string;
  ticker: string;
  assetClass: AssetClass;
  account: AccountType;
  value: number;
  costBasis: number;
  /** Share count, so live prices can mark the position to market. */
  shares?: number;
  singleName: boolean;
  /** How much you think this beats the market by, per year. 0 = no view. */
  edge?: number;
  /** Annualized volatility of this name. Defaults to a large-cap 42%. */
  vol?: number;
}

export type Style = "diversified" | "barbell";

export interface Profile {
  /** Worst peak-to-trough the family can hold through without selling. */
  drawdownTolerance: number;
  horizonYears: number;
  /** Combined federal + state marginal rate on long-term gains. */
  ltcgRate: number;
  /** Combined marginal rate on interest and short-term gains. */
  ordinaryRate: number;
  /**
   * How the family actually invests.
   *
   * "diversified" is the textbook answer. "barbell" is what most people who
   * built this kind of money actually do: keep a floor of safe assets large
   * enough that you can never be forced to sell, then take real conviction
   * bets with everything above it. Mean-variance cannot express the second
   * one, because it assumes you have no view.
   */
  style: Style;
  /** Years of spending that must stay untouchable. Sets the floor. */
  floorYears: number;
  annualSpending: number;
}

/**
 * Half-Kelly position size for a bet you actually have a view on.
 *
 * Full Kelly maximizes long-run growth but is famously wild — a 50% drawdown
 * is routine. Half-Kelly gives up about a quarter of the growth for
 * dramatically less pain, which is why essentially every professional who
 * uses Kelly uses a fraction of it.
 *
 *   f* = edge / variance     (full)
 *
 * `edge` is how much you think this beats the market by, per year. If you have
 * no view, edge is zero and the answer is the market weight.
 */
export function kellySize(edge: number, vol: number, fraction = 0.5): number {
  if (edge <= 0 || vol <= 0) return 0;
  return Math.max(0, Math.min((edge / vol ** 2) * fraction, 1));
}

/**
 * How far from the Kelly optimum you should tolerate before trading — the
 * piece that standard Kelly gets wrong for a taxable investor.
 *
 * Textbook Kelly assumes rebalancing is free, so it says move to the optimum
 * immediately. For a family holding a position with an 80% embedded gain,
 * moving there costs 30% of every dollar traded. That tax is certain and paid
 * today; the benefit of being perfectly sized is small and arrives slowly.
 *
 * Being off-target by d costs roughly (d^2 * sigma^2 / 2) of growth per year.
 * Over H years the benefit of closing the gap is:
 *
 *     benefit = d^2 * sigma^2 * H / 2
 *
 * and the tax paid to close it is:
 *
 *     cost = d * gainPct * taxRate
 *
 * Setting them equal gives the width of the region where trading destroys
 * value:
 *
 *     d* = 2 * gainPct * taxRate / (sigma^2 * H)
 *
 * Inside that band, the right action is to do nothing. Outside it, the right
 * action is to trade to the EDGE of the band — not to the optimum, because
 * the last stretch costs more tax than the sizing improvement is worth.
 */
export function noTradeBand(
  gainPct: number,
  taxRate: number,
  vol: number,
  horizonYears: number,
): number {
  if (vol <= 0 || horizonYears <= 0) return 0;
  const g = Math.max(gainPct, 0);
  return Math.min((2 * g * taxRate) / (vol ** 2 * horizonYears), 0.5);
}

export interface SizingVerdict {
  kelly: number;
  ruinCap: number;
  target: number;
  band: number;
  lower: number;
  upper: number;
  action: "trim" | "add" | "hold";
  tradeTo: number;
  /** Which constraint set the answer. */
  binding: "kelly" | "ruin" | "band";
}

/** Combine the growth-optimal size, the survivability cap, and the tax band. */
export function sizePosition(
  weight: number,
  edge: number,
  vol: number,
  gainPct: number,
  taxRate: number,
  horizonYears: number,
  ruinCap: number,
): SizingVerdict {
  const kelly = kellySize(edge, vol);
  const target = Math.min(kelly > 0 ? kelly : 0.1, ruinCap);
  const binding: SizingVerdict["binding"] = ruinCap < kelly ? "ruin" : "kelly";

  const band = noTradeBand(gainPct, taxRate, vol, horizonYears);
  const lower = Math.max(target - band, 0);
  // Never let the tax band excuse a position that could end the family.
  const upper = Math.min(target + band, ruinCap);

  let action: SizingVerdict["action"] = "hold";
  let tradeTo = weight;
  if (weight > upper) {
    action = "trim";
    tradeTo = upper;
  } else if (weight < lower) {
    action = "add";
    tradeTo = lower;
  }

  return {
    kelly, ruinCap, target, band, lower, upper,
    action, tradeTo,
    binding: action === "hold" ? "band" : binding,
  };
}

/**
 * The most you can bet without risking the floor.
 *
 * Ruin is not "the number went down" — it is being forced to sell at the
 * bottom. As long as the floor covers spending, a concentrated position can
 * fall 60% and the family simply waits. This is the constraint that actually
 * matters, and it is the one no risk questionnaire asks about.
 */
export function maxBetSize(
  total: number,
  annualSpending: number,
  floorYears: number,
  stressLoss = 0.6,
  safeWithdrawal = 0.035,
): number {
  if (total <= 0 || annualSpending <= 0) return 0.6;

  // Test 1 — the position drops `stressLoss` and never recovers. Can what is
  // LEFT still support the family permanently? Checking only that a few years
  // of cash survive is far too weak: it would clear a 100% single-stock
  // portfolio, which is obviously not a survivable bet.
  const capitalNeeded = annualSpending / safeWithdrawal;
  const permanent = (1 - capitalNeeded / total) / stressLoss;

  // Test 2 — enough liquid floor that nothing has to be sold during the fall.
  const floor = annualSpending * floorYears;
  const liquidity = Math.max(total - floor, 0) / total;

  return Math.max(0, Math.min(permanent, liquidity, 0.75));
}

export const ASSET_LABEL: Record<AssetClass, string> = {
  us_equity: "US equity",
  intl_equity: "International equity",
  em_equity: "Emerging markets",
  bonds: "Bonds",
  credit: "Credit",
  real_assets: "Real assets",
  cash: "Cash",
};

export const ACCOUNT_LABEL: Record<AccountType, string> = {
  taxable: "Taxable",
  ira: "IRA / 401(k)",
  roth: "Roth",
  trust: "Trust",
};

/**
 * Forward-looking assumptions, built from yields and growth rather than past
 * returns. `income` is the share of return arriving as taxable income each
 * year; `turnover` drives realized gains.
 */
interface Assumption {
  ret: number;
  vol: number;
  income: number;
  incomeIsOrdinary: boolean;
  turnover: number;
}

export const CMA: Record<AssetClass, Assumption> = {
  us_equity: { ret: 0.065, vol: 0.158, income: 0.013, incomeIsOrdinary: false, turnover: 0.05 },
  intl_equity: { ret: 0.071, vol: 0.17, income: 0.031, incomeIsOrdinary: false, turnover: 0.08 },
  em_equity: { ret: 0.09, vol: 0.215, income: 0.029, incomeIsOrdinary: false, turnover: 0.12 },
  bonds: { ret: 0.046, vol: 0.058, income: 0.046, incomeIsOrdinary: true, turnover: 0.3 },
  credit: { ret: 0.053, vol: 0.098, income: 0.053, incomeIsOrdinary: true, turnover: 0.4 },
  real_assets: { ret: 0.068, vol: 0.145, income: 0.038, incomeIsOrdinary: true, turnover: 0.1 },
  cash: { ret: 0.037, vol: 0.006, income: 0.037, incomeIsOrdinary: true, turnover: 0 },
};

/** Correlations, de-smoothed for real assets. */
const CORR: Record<AssetClass, Partial<Record<AssetClass, number>>> = {
  us_equity: { intl_equity: 0.84, em_equity: 0.72, bonds: 0.1, credit: 0.72, real_assets: 0.55, cash: 0 },
  intl_equity: { em_equity: 0.8, bonds: 0.12, credit: 0.68, real_assets: 0.5, cash: 0 },
  em_equity: { bonds: 0.1, credit: 0.65, real_assets: 0.45, cash: 0 },
  bonds: { credit: 0.28, real_assets: 0.18, cash: 0.15 },
  credit: { real_assets: 0.48, cash: 0.02 },
  real_assets: { cash: 0.05 },
  cash: {},
};

const CLASSES = Object.keys(CMA) as AssetClass[];

function corr(a: AssetClass, b: AssetClass): number {
  if (a === b) return 1;
  return CORR[a]?.[b] ?? CORR[b]?.[a] ?? 0;
}

/** Annual return lost to tax if this asset sits in a taxable account. */
export function taxDrag(ac: AssetClass, p: Profile): number {
  const a = CMA[ac];
  const incomeRate = a.incomeIsOrdinary ? p.ordinaryRate : p.ltcgRate;
  const incomeDrag = a.income * incomeRate;
  const appreciation = Math.max(a.ret - a.income, 0);
  const gainDrag = appreciation * Math.min(a.turnover, 1) * p.ltcgRate;
  return incomeDrag + gainDrag;
}

/** How much of that drag is actually paid inside a given wrapper. */
export function wrapperMultiplier(acct: AccountType): number {
  if (acct === "roth") return 0;
  if (acct === "ira") return 0.35; // deferred, but taxed as ordinary on exit
  return 1;
}

export function portfolioVol(weights: Record<AssetClass, number>): number {
  let v = 0;
  for (const a of CLASSES) {
    for (const b of CLASSES) {
      v += (weights[a] ?? 0) * (weights[b] ?? 0) * CMA[a].vol * CMA[b].vol * corr(a, b);
    }
  }
  return Math.sqrt(Math.max(v, 0));
}

export function portfolioReturn(weights: Record<AssetClass, number>): number {
  return CLASSES.reduce((s, a) => s + (weights[a] ?? 0) * CMA[a].ret, 0);
}

/**
 * Target allocation for a given drawdown tolerance.
 *
 * The tolerance converts to a volatility ceiling (a diversified portfolio's
 * worst drawdown runs roughly 2.4x its annual vol), then we scale between a
 * defensive and a growth mix to hit that ceiling. This is a glidepath rather
 * than an optimizer — a solver would give false precision on seven inputs.
 */
export function targetAllocation(
  p: Profile,
  total = 0,
): Record<AssetClass, number> {
  if (p.style === "barbell") {
    // Floor first: enough safe assets that spending is covered through any
    // drawdown, so nothing ever has to be sold at the bottom. Everything above
    // the floor goes to work. No bonds "for balance" — bonds here have one
    // job, and once it is done they stop earning their place.
    const floor = p.annualSpending * p.floorYears;
    const floorPct = total > 0 ? Math.min(floor / total, 0.6) : 0.15;
    const risk = 1 - floorPct;
    return {
      us_equity: risk * 0.7,
      intl_equity: risk * 0.13,
      em_equity: risk * 0.07,
      real_assets: risk * 0.1,
      bonds: floorPct * 0.55,
      credit: 0,
      cash: floorPct * 0.45,
    };
  }

  const volBudget = Math.max(p.drawdownTolerance / 2.4, 0.02);

  const defensive: Record<AssetClass, number> = {
    us_equity: 0.18, intl_equity: 0.08, em_equity: 0.02,
    bonds: 0.45, credit: 0.05, real_assets: 0.07, cash: 0.15,
  };
  const growth: Record<AssetClass, number> = {
    us_equity: 0.42, intl_equity: 0.2, em_equity: 0.09,
    bonds: 0.1, credit: 0.04, real_assets: 0.13, cash: 0.02,
  };

  const volD = portfolioVol(defensive);
  const volG = portfolioVol(growth);
  let t = (volBudget - volD) / (volG - volD);
  t = Math.min(Math.max(t, 0), 1);

  const out = {} as Record<AssetClass, number>;
  for (const c of CLASSES) out[c] = defensive[c] + (growth[c] - defensive[c]) * t;
  return out;
}

export interface PositionFinding {
  ticker: string;
  value: number;
  weight: number;
  account: AccountType;
  assetClass: AssetClass;
  gain: number;
  gainPct: number;
  taxIfSold: number;
  /** Positive = a harvestable loss sitting in a taxable account. */
  harvestable: number;
  severity: "critical" | "warning" | "ok";
  headline: string;
  reasons: string[];
  actions: string[];
}

/**
 * Required annual outperformance to justify keeping a concentrated position,
 * measured at the PORTFOLIO level. Scoring the position in isolation produces
 * required-alpha numbers in the tens of percent, which nobody believes.
 */
export function requiredAlpha(
  weight: number,
  gainPct: number,
  p: Profile,
  restVol: number,
  singleVol = 0.42,
  gamma = 3,
): number {
  if (weight <= 0) return 0;
  const rho = 0.6;
  const volHold = Math.sqrt(
    (weight * singleVol) ** 2 +
      ((1 - weight) * restVol) ** 2 +
      2 * weight * (1 - weight) * rho * singleVol * restVol,
  );
  const varPenalty = (gamma * (volHold ** 2 - restVol ** 2)) / 2;
  const wealthCost = weight * gainPct * p.ltcgRate;
  const taxHurdle = wealthCost < 1 ? -Math.log(1 - wealthCost) / p.horizonYears : Infinity;
  return (varPenalty + taxHurdle) / weight;
}

export interface Analysis {
  total: number;
  byClass: Record<AssetClass, number>;
  byAccount: Record<AccountType, number>;
  currentWeights: Record<AssetClass, number>;
  target: Record<AssetClass, number>;
  currentReturn: number;
  currentVol: number;
  targetReturn: number;
  targetVol: number;
  expectedWorstDrawdown: number;
  unrealizedGain: number;
  harvestableNow: number;
  harvestBenefit: number;
  locationSavings: number;
  positions: PositionFinding[];
  gaps: { assetClass: AssetClass; current: number; target: number; deltaDollars: number }[];
}

export function analyze(holdings: Holding[], p: Profile): Analysis {
  const total = holdings.reduce((s, h) => s + h.value, 0);

  const byClass = Object.fromEntries(CLASSES.map((c) => [c, 0])) as Record<AssetClass, number>;
  const byAccount = { taxable: 0, ira: 0, roth: 0, trust: 0 } as Record<AccountType, number>;
  for (const h of holdings) {
    byClass[h.assetClass] += h.value;
    byAccount[h.account] += h.value;
  }

  const currentWeights = Object.fromEntries(
    CLASSES.map((c) => [c, total > 0 ? byClass[c] / total : 0]),
  ) as Record<AssetClass, number>;

  const target = targetAllocation(p, total);
  const currentVol = portfolioVol(currentWeights);
  const currentReturn = portfolioReturn(currentWeights);
  const targetVol = portfolioVol(target);
  const targetReturn = portfolioReturn(target);

  const unrealizedGain = holdings.reduce((s, h) => s + (h.value - h.costBasis), 0);

  // Losses only count where they can actually be used — a loss inside an IRA
  // is worth nothing.
  let harvestableNow = 0;
  for (const h of holdings) {
    if (h.account !== "taxable" && h.account !== "trust") continue;
    const loss = h.costBasis - h.value;
    if (loss > 0) harvestableNow += loss;
  }
  const harvestBenefit = harvestableNow * p.ltcgRate;

  // Asset location: what you'd save by moving the highest-drag assets into the
  // sheltered accounts you already have.
  const shelterCapacity = byAccount.ira + byAccount.roth;
  const ranked = [...CLASSES].sort((a, b) => taxDrag(b, p) - taxDrag(a, p));
  let remaining = shelterCapacity;
  let idealDrag = 0;
  for (const c of ranked) {
    const amt = byClass[c];
    const shelter = Math.min(amt, remaining);
    remaining -= shelter;
    idealDrag += (amt - shelter) * taxDrag(c, p);
  }
  let actualDrag = 0;
  for (const h of holdings) {
    actualDrag += h.value * taxDrag(h.assetClass, p) * wrapperMultiplier(h.account);
  }
  const locationSavings = Math.max(actualDrag - idealDrag, 0);

  const positions = holdings
    .map((h) => buildFinding(h, total, p, currentVol))
    .sort((a, b) => rank(b.severity) - rank(a.severity) || b.value - a.value);

  const gaps = CLASSES.map((c) => ({
    assetClass: c,
    current: currentWeights[c],
    target: target[c],
    deltaDollars: (target[c] - currentWeights[c]) * total,
  })).sort((a, b) => Math.abs(b.deltaDollars) - Math.abs(a.deltaDollars));

  return {
    total, byClass, byAccount, currentWeights, target,
    currentReturn, currentVol, targetReturn, targetVol,
    expectedWorstDrawdown: -currentVol * 2.4,
    unrealizedGain, harvestableNow, harvestBenefit, locationSavings,
    positions, gaps,
  };
}

function rank(s: PositionFinding["severity"]) {
  return s === "critical" ? 2 : s === "warning" ? 1 : 0;
}

function pct(n: number) {
  return `${(n * 100).toFixed(1)}%`;
}
function money(n: number) {
  const sign = n < 0 ? "−" : "";
  return `${sign}$${Math.abs(Math.round(n)).toLocaleString()}`;
}

function buildFinding(
  h: Holding,
  total: number,
  p: Profile,
  portVol: number,
): PositionFinding {
  const weight = total > 0 ? h.value / total : 0;
  const gain = h.value - h.costBasis;
  const gainPct = h.value > 0 ? gain / h.value : 0;
  const isTaxable = h.account === "taxable" || h.account === "trust";
  const taxIfSold = isTaxable && gain > 0 ? gain * p.ltcgRate : 0;
  const harvestable = isTaxable && gain < 0 ? -gain : 0;

  const reasons: string[] = [];
  const actions: string[] = [];
  let severity: PositionFinding["severity"] = "ok";
  let headline = "No action needed.";

  // --- concentration ----------------------------------------------------
  if (h.singleName && weight > 0.08) {
    const restVol = Math.max(portVol, 0.06);
    const vol = h.vol ?? 0.42;
    const edge = h.edge ?? 0;

    const ruinCap = maxBetSize(total, p.annualSpending, p.floorYears);
    const v = sizePosition(
      weight, edge, vol, Math.max(gainPct, 0), isTaxable ? p.ltcgRate : 0,
      p.horizonYears, ruinCap,
    );

    if (edge > 0) {
      reasons.push(
        `You expect ${h.ticker} to beat the market by ${pct(edge)} a year at ${pct(vol)} volatility. Kelly — the formula for how much to bet when you have a real edge — puts the growth-maximizing size at ${pct(v.kelly)} at half-Kelly. Bet too small and you leave compounding on the table; bet too big and volatility eats the returns even when you are right about the company.`,
      );
    } else {
      const alpha = requiredAlpha(weight, Math.max(gainPct, 0), p, restVol);
      reasons.push(
        `You have not said you expect ${h.ticker} to beat the market, so the math treats it as a coin flip carrying extra risk. On that basis it would need to beat the market by ${pct(alpha)} a year to earn this size. Enter your actual view and the answer changes completely.`,
      );
    }

    reasons.push(
      `Ruin is not the number going down — it is going down and staying down while you still need to live off it. If ${h.ticker} fell 60% and never recovered, you could hold up to ${pct(ruinCap)} and what remained would still cover ${money(p.annualSpending)} a year forever.`,
    );

    if (v.band > 0.005) {
      reasons.push(
        `Because this sits on a ${pct(gainPct)} gain, moving it costs ${pct(p.ltcgRate)} of every dollar traded. That tax is certain and paid today; better sizing pays off slowly. Netting the two gives a no-trade band of ±${pct(v.band)} around ${pct(v.target)} — anywhere between ${pct(v.lower)} and ${pct(v.upper)} is close enough that trading destroys more value than it creates.`,
      );
    }

    if (v.action === "trim") {
      severity = weight > ruinCap ? "critical" : "warning";
      headline = `Above the range — trim to ${pct(v.upper)}, not to ${pct(v.target)}.`;
      const frac = (weight - v.tradeTo) / weight;
      reasons.push(
        `Trimming to ${pct(v.upper)} realizes about ${money(frac * gain)} of gain and costs roughly ${money(frac * taxIfSold)} in tax. Going all the way to ${pct(v.target)} would cost more tax than the extra sizing precision is worth — this is the part textbook Kelly gets wrong for a taxable holder.`,
      );
      actions.push(`Sell down to ${pct(v.upper)} of the portfolio — about ${money((weight - v.tradeTo) * total)}.`);
      actions.push(`Stage it across two or three tax years and harvest losses in the same years.`);
    } else if (v.action === "add") {
      severity = "warning";
      headline = `Below what your own conviction supports.`;
      actions.push(`You could add up to ${pct(v.lower)} — about ${money((v.tradeTo - weight) * total)} — before sizing is optimal.`);
    } else {
      severity = "ok";
      headline = `Inside the range. Leave it alone.`;
      actions.push(`Hold. Revisit if it drifts past ${pct(v.upper)} or your view on the company changes.`);
    }
  }

  // --- harvestable loss --------------------------------------------------
  if (harvestable > 0) {
    if (severity === "ok") severity = "warning";
    if (headline === "No action needed.") {
      headline = `Sitting on a ${money(harvestable)} loss you can use.`;
    }
    reasons.push(
      `This is worth ${money(harvestable)} less than you paid. Selling it banks the loss against gains elsewhere and saves roughly ${money(harvestable * p.ltcgRate)} in tax. You can buy something similar immediately to stay invested — just not the identical security for 31 days.`,
    );
    actions.push(`Sell and replace with a similar fund to bank ${money(harvestable * p.ltcgRate)}.`);
  }

  // --- account placement --------------------------------------------------
  const drag = taxDrag(h.assetClass, p);
  if (isTaxable && drag > 0.015 && h.assetClass !== "us_equity") {
    if (severity === "ok") severity = "warning";
    if (headline === "No action needed.") headline = `In the wrong account.`;
    reasons.push(
      `This throws off income that is taxed every year, costing about ${pct(drag)} annually — roughly ${money(h.value * drag)} on this position. Held in an IRA or Roth instead, that cost disappears entirely.`,
    );
    actions.push(`Move ${ASSET_LABEL[h.assetClass].toLowerCase()} into your IRA or Roth, and hold stocks here instead.`);
  }

  if (h.account === "roth" && CMA[h.assetClass].ret < 0.05) {
    if (severity === "ok") severity = "warning";
    if (headline === "No action needed.") headline = `Low-growth asset in your best account.`;
    reasons.push(
      `A Roth is the one account where growth is never taxed, so it should hold whatever you expect to grow the most. ${ASSET_LABEL[h.assetClass].toLowerCase()} is close to the opposite of that.`,
    );
    actions.push(`Swap this for stocks inside the Roth.`);
  }

  if (severity === "ok" && gain > 0) {
    reasons.push(
      `Held in the right place with a ${money(gain)} unrealized gain. Leave it alone — selling would cost ${money(taxIfSold)} for no reason.`,
    );
  }

  return {
    ticker: h.ticker, value: h.value, weight, account: h.account,
    assetClass: h.assetClass, gain, gainPct, taxIfSold, harvestable,
    severity, headline, reasons, actions,
  };
}
