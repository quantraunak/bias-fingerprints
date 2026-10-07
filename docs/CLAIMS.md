# Claims ledger

Every number this project states, where it appears, the artifact it comes from, whether
it was pre-registered, and its status after the 2026-10-07 consistency pass. Artifacts:
`PVN` = `project/reports/pit_vs_naive.csv` (written 2026-09-11 by `scripts/pit_vs_naive.py`);
`SURV` = `project/reports/survivorship.csv` (2026-09-11, `scripts/survivorship.py`);
`GATES` = `project/reports/gates_1_to_3.txt` (2026-10-05, `scripts/gate_ic_series.py` + `scripts/gates.py`);
`RUN` = `project/reports/20260820_175632/` (`signal.json`, `performance.json`, `quantile_returns.csv`, `universe_coverage.csv`);
`QJ` = `public/data/quant.json` in the site repo, generated from RUN; "site" = the raunaksood repo
(`app/work/quant/page.tsx`, `app/writing/testing-for-leakage/page.tsx`, `lib/content.ts`).
Nothing in this study was pre-registered as a hypothesis test; the protocol order in
`docs/FINGERPRINT.md` is the only pre-specified element, and the Gate 3 pass criterion
(every class recovered above 50%) is stated in `scripts/gates.py`, not in the protocol.

| claim | where it appears | artifact | pre-registered? | status |
|---|---|---|---|---|
| Period-end join inflates mean IC by 59% among the 11 filing-sensitive factors | README (3×); BIAS.md; paper abstract, §3, conclusion; site quant page; leakage post; content.ts | PVN: mean(naive − PIT) / mean\|PIT\| = 0.00476 / 0.00813 = 58.5%, which `pit_vs_naive.py` prints as 59% | no (measurement) | consistent; definition now in this ledger |
| 4 of 11 cross t = 2 spuriously: earnings_yield, cash_flow_yield, roe, accruals | README; BIAS.md; paper; site; leakage post | PVN (2.54, 2.72, 2.30, 2.99 vs 1.56, 1.70, 1.22, 1.23) | no | consistent |
| 10 of 11 look better under NAIVE | BIAS.md | PVN | no | consistent |
| Fixed 45-day lag: +9%, 8 of 11 better, 0 of 11 cross | BIAS.md; README; paper Table | PVN (9.5%, 8, 0) | no | consistent |
| Price-only factors unchanged, max drift 0.0e+00 | README; BIAS.md; paper; site; leakage post | PVN (11 factors identical across conventions) | no | consistent |
| turnover_1m is the one price-family factor that moves | README; BIAS.md; leakage post | PVN (−0.481 → −0.586) | no | consistent |
| Accruals t 1.23 → 2.99; shifts accruals +1.77 ± 0.31, roe +1.07 ± 0.15, earnings_yield +0.98 ± 0.15, cash_flow_yield +1.02 ± 0.14 | README; BIAS.md; paper §6; GATES | PVN + GATES Gate 2 | no | consistent |
| Universe conditioning: mean IC −0.0043 across 22; 8 of 22 look better | README; BIAS.md; paper abstract | SURV (−0.00432; 8) | no | consistent |
| amihud −1.11 → +2.80 (+3.91 ± 0.44); turnover −0.48 → −2.53 (−2.05); idio_vol 0.49 → −1.53; vol_60d −0.08 → −1.49; earnings_yield −1.28; roe −1.24 | README; BIAS.md; paper §3, §6; GATES; QJ signatures | SURV + GATES | no | consistent |
| Two universe false positives, one with the wrong sign | README | SURV (amihud +2.80, turnover −2.53) | no | consistent |
| Signature correlation −0.088, block-bootstrap 95% CI [−0.225, +0.036] | README; paper; GATES; QJ | GATES Gate 2 | no | consistent |
| Gate 1: subsample correlations +0.928 … +0.995; amihud +2.05 … +3.55 in every subsample; exclusion restriction exact | README; paper §6 | GATES | protocol step 1 (FINGERPRINT.md) | consistent |
| Gate 3 realistic τ: universe 99.2% / worst 98.5%; dating 63.9% / 29.5%; joint 64.3% / 29.2%; 70% of clean tables labelled defective | README; paper abstract, Table 5; site ("99%") | GATES | protocol step 3; pass criterion from `gates.py` | consistent |
| Gate 3 exchangeable τ: dating 76.7%, universe 99.7%, joint 77.3% | paper | GATES | same | consistent |
| ‖τ̃‖ 2.19, ‖δ̃_dating‖ 1.40, ‖δ̃_universe‖ 5.17; corr +0.543 / −0.268; projection +0.85 / −0.11 | README; paper | GATES | no | consistent |
| Earlier gate run used undeflated t, ≈ 4.6× too large; verdicts unchanged | README; paper | `project/reports/gates_1_to_3_naive_t_20260911.txt` vs GATES | — | consistent |
| 3,142 dates deflated to ~150 independent; 150 out-of-sample months; 3,143 of 3,261 P&L days | GATES; paper; README; QJ | GATES; RUN performance.json (3,143 days; IC needs a forward return, hence 3,142) | — | consistent |
| Universe "727 names"; "278 of 990 historical members could not be priced"; "648 issuers" | README (2×); paper §4 | `data/raw/universe/sp500_spells.csv`: 990 tickers with a spell from 2004 on; 725 have a price parquet, 265 do not; `universe_coverage.csv` peaks at 645 issuers with facts | — | **CONTRADICTION** with the current cache (727 vs 725; 278 vs 265; 648 vs 645): fixed to the artifact counts |
| Mean IC 0.0165, t = 2.00, ICIR 0.164 / 0.567, decile spread 68.1bp t = 3.08, selection turnover 0.24 | README; site | RUN signal.json; QJ signal | — | consistent |
| Pooled decile table top − bottom = 73.3bp | (not stated) | RUN quantile_returns.csv | — | note: a second definition; README now names it |
| Beta decomposition: raw spread 43.7bp t 2.11, beta 0.218, alpha 17.7bp t 0.92 (2.12%/yr), 59% of edge is beta, long/short beta 1.10 / 0.99, market CAGR 13.7%, "positive in 73% of months" | README; site quant page; dashboard | constants hard-coded in `scripts/build_dashboard.py`; no regression output committed; 13.7% market CAGR is in QJ | — | **UNTRACEABLE**: marked unverified in README and on the site; "73% of months" removed |
| Seeds: IC 0.0164 ± 0.0008; Sharpe 0.275 ± 0.138 (0.09 … 0.49); CAGR 1.67% ± 0.91% | README; site | QJ seeds (from `scripts/robustness.py`) | — | consistent |
| Re-ordering the factor list moved Sharpe 0.07 → 0.29 | README | none | — | **UNTRACEABLE**: removed |
| Model selection table (LightGBM 0.0124 / 1.51 / 46.1bp …) | README | QJ models (`scripts/compare_models.py`) | — | consistent |
| Smoothing: IC 0.0124 → 0.0165, turnover 0.48 → 0.24 | README | QJ smoothing (0.01235 → 0.01639; 0.482 → 0.243) | — | consistent |
| risk_neutral residualisation: IC 0.0165 → 0.0062 | README | `src/factors/registry.py` docstring says 0.0164 → 0.0062; output not committed | — | minor mismatch fixed to 0.0164; marked as recorded in code |
| Portfolio table: CAGR 1.85%, Sharpe 0.29, Sortino 0.44, vol 7.4%, MDD −18.9%, beta 0.011, turnover 72%, leverage 1.45×, 67 names, 96.4% coverage, Sharpe 0.30 / 0.24 / 0.12 by slippage | README; site (1.9%, 0.29) | RUN performance.json; QJ | — | consistent |
| PORTFOLIO_FINDINGS variant A: Sharpe 0.240, CAGR 1.47% (vs README 0.29 / 1.85%) | docs/PORTFOLIO_FINDINGS.md | that document (includes 40bp borrow) | — | **CONTRADICTION** unlabelled: note added explaining the cost model |
| 28% of the label's variance is the market component | README; site (bugs) | QJ bugs | — | consistent |
| 55 of 180 calendar month-ends are not trading days | README | `tests/test_no_lookahead.py` asserts the property and its docstring says 30% (55/180 = 30.6%) | — | consistent |
| Median filing lag 34 days; 35 → 25 days 2010 → 2025; >90-day share 22% → 1.6%; era table (61% / 45% / 33%) | README; BIAS.md; paper §3 | `scripts/filing_lag_study.py` prints the median; no committed script prints the era split; outputs not committed | — | **UNTRACEABLE**: marked in BIAS.md and paper |
| Cohort table: 386 / 203 / 88 names per day; $95M / $28M / $54M volume; $10.1B / $3.5B / $5.3B cap | README; BIAS.md; paper Table 3 | none (no script prints it) | — | **UNTRACEABLE**: marked in BIAS.md and the paper caption |
| Apple: 56 of 72 quarters of operating cash flow dropped | README | none | — | **UNTRACEABLE**: number removed, point kept |
| Paper "14pp" | README | PDF is 15 pages | — | **CONTRADICTION**: fixed |
| "126 tests + 1 xfail" | README | `pytest -q project/tests` → 7 passed | — | **CONTRADICTION**: fixed |
| Layout lists docs/HYPOTHESIS.md, SPECIFICATIONS.md, EXTRACTION.md, RESULT_01.md, RELATED_WORK.md, DATASHEET.md, RELEASE.md, gold_exclusions.md, scripts/build_data.py, benchmark_table.py, src/graph | README | none of these exist here (moved to filing-links) | — | **CONTRADICTION**: layout and run instructions rewritten |
| Factor list: 22 = 12 price-family + 10 fundamental; 11 price-only + 11 filing-sensitive | README; BIAS.md; paper | `src/factors/registry.py` (trend_200d was missing from the README list) | — | fixed |
| Gates 1–3 run; Gate 4 unrun and applies only to the universe signature | README; paper | GATES | protocol | consistent (an earlier intro sentence saying gates were unrun was fixed 2026-10-05) |

## What needs a new experiment or a re-run, not an edit

- Re-run and commit the beta decomposition (decile long–short series regressed on the market) so the 59%-of-edge claim on the site has an artifact.
- Re-run `scripts/filing_lag_study.py` and the era split of `pit_vs_naive.py`, and commit their output.
- Reproduce the cohort table (names per day and medians for genuine / wrongly included / wrongly excluded) as a script with committed output, or drop it from the paper.
- Model τ in the Gate 3 classifier (the paper already names this as the next step).
- CRSP/Compustat rebuild with delisting returns.
