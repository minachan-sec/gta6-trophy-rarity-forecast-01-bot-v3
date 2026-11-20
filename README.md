# GTA VI Trophy Rarity Forecaster v2

[![Bounty](https://img.shields.io/badge/make.fun-bounty%20%24250-7c3aed)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Entry for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn
rate on Exophase at **2026-11-26T00:00:00Z** (five quantiles per trophy:
5%, 25%, 50%, 75%, 95%).

`main.py` is the whole entry: Python 3.13, standard library only, no
network, deterministic (fixed-seed PRNG — two runs are byte-identical),
~0.3 s per case, and it never crashes (layered fallbacks down to a
persistence forecast, then to fixed valid quantiles).

## How it works

**Earners and players are modelled separately, then divided.** The earn
rate is earners / players-tracked, and the two sides move on different
logic: the denominator jumps when Exophase refreshes and new players sync,
while the numerator only grows as fast as people play. Modelling the ratio
directly is blinded by refresh jumps in the denominator.

**Players** follow a recency-weighted power law in time (mean of the
all-history and last-24h fits, clipped to [0.3, 1.3]), anchored on the
last read, with a weekend boost (the final horizon covers Sat–Sun 21–22
Nov, which no history contains) and a soft regularisation band.

**Earners** follow a local log-log fit over the last 36 hours — on **net**
earners, total minus the launch-night reviewer base (see below) — shrunk
toward a per-class prior slope, and capped by a per-class day-7 rate cap.
Platinum and 100% completion get a hard cap instead of a fit: ~200 hours
of play cannot fit in a 168–181 hour week, so their earners are
essentially the pre-launch reviewers plus a small Poisson trickle.

**The answer is the page as of Exophase's last refresh**, so each Monte
Carlo path forecasts at the target minus a random refresh lag (0–6 h).
Quantiles come from 2,000 deterministic Monte Carlo paths over the real
sources of uncertainty (players exponent and level, weekend factor,
refresh lag, per-trophy slope and earner noise), with a spread floor so
intervals stay honest over a 4.5-day horizon. No sharpening.

**Structure is enforced per quantile level**: platinum below every other
trophy, chapters nested (final ≤ Ch4 ≤ Ch3 ≤ Ch2 ≤ Ch1 ≤ gateway), with a
15% slack on the 95% quantile. The gateway trophy (first mission) is
forecast at ~99.95%: Exophase only counts players who earned ≥1 trophy.

**A frozen history** (≤2 distinct page states, as in the practice cases)
falls back to a refresh mixture: either the page stays put, or it
refreshes with a lognormal jump in players and class-dependent newcomer
earners.

## Two things validation forced us to fix

Local validation against synthetic finals (below) exposed two real
defects in the naive version of this design; both fixes are principled,
not tuned to any score:

1. **Fitting total earners dilutes the growth slope.** Launch-night
   earners are mostly pre-launch reviewers — a constant base added to
   every later read. Early on the base is most of E, so a log-log fit on
   total E understates newcomer growth by ~1 unit of slope and the
   forecast under-shoots day-7 rates by ~1.2 logit almost everywhere.
   The fit therefore runs on net earners (total minus the earliest
   observed base) and the base is added back at projection time.
2. **The class slope priors must agree with the class median priors.**
   The day-7 median rates (from GTA V / RDR2 analogs + day-1 live rates)
   imply net-earner growth of 50–400× over the week, i.e. slopes of
   2.3–3.2 — so a shrinkage target of ~1.0 contradicts the very medians
   it is supposed to support. Each `gamma_class` is derived from its
   class median: ln(E_new(day 7) / E_new(day 1)) / ln(168/26), rounded
   to 1 decimal and capped at 3.0.

A third, smaller fix: in the frozen-history branch, newcomer earn shares
are capped at a small multiple of the trophy's current rate (relaxing
with horizon). Without the cap, the refresh branch forecasts rates
*higher* than the frozen page — contradicting every observed refresh,
where the denominator jumps far faster than earners.

## Validation (local, no submission)

**Practice cases** (p00/p01, horizon 90 min, frozen history): scored
against reconstructed answers anchored to the one published number (the
house baseline scores ~1.51 on the practice). v2 scores **0.46–0.62**
across four reconstruction scenarios vs **1.32–1.56** for both house
baselines. The real practice answers are server-side, so these are
scenarios, not ground truth — and practice is a 90-minute proxy for a
4.5-day final in any case.

**Synthetic finals** (the main validation): final-shaped cases
(checkpoint 21 Nov 11:30 UTC, horizon 6,510 min) built from a
ground-truth generator deliberately different from this model
(log-linear N through anchors in weekend-stretched time; reviewer base +
per-trophy newcomer power law calibrated to observed day-1 earners and
scenario day-7 rates; refresh freezes every 3–8 h; rates rounded to 2
decimals; answer at target minus a scenario lag). v2 beats **both**
house baselines in **8/8 scenarios** (5 core + 3 stress):

| scenario | v2 | persistence | curve |
|---|---:|---:|---:|
| base | 0.183 | 0.927 | 0.505 |
| N low (130k) | 0.182 | 0.935 | 0.461 |
| N high (500k) | 0.261 | 0.846 | 0.353 |
| fast takeoff (×1.8 rates) | 0.175 | 1.191 | 0.429 |
| slow takeoff (×0.55 rates) | 0.184 | 0.674 | 0.481 |
| takeoff surge after deadline (stress) | 0.918 | 2.247 | 1.564 |
| page frozen 24 h (stress) | 0.193 | 0.811 | 0.437 |
| no weekend boost (stress) | 0.172 | 0.965 | 0.550 |

90% coverage on the five core scenarios is 0.96–1.00. The synthetic
truth is smoother than reality (no regime changes except the stress
scenarios), so real coverage will be lower — the spread is kept honest
rather than narrowed to hit 0.90 on a too-smooth generator.

**Robustness**: 22 edge cases (empty/corrupt input, empty history,
all-null rates, N = 0, rates 0/100, zero/negative horizon, target before
checkpoint, missing fields, 10-year horizon, single trophy) — all pass:
exit 0, valid output or clean skip. Three runs on the same cases are
byte-identical (sha256). Output ~3.2 KB per case (limit 65,536).

## Honest limits

- The day-7 medians for the fastest takeoffs (Chapter 1, fishing) are
  the largest uncertainty: if the story takeoff is much faster than the
  priors (guides + weekend marathons), the medians under-shoot and the
  loss concentrates there. The takeoff-surge stress scenario (loss
  0.92) shows the cost; no model with a 6-hour lag prior and a
  history-only fit can fully cover a post-deadline regime change.
- If Exophase's page freezes for much longer than 6 hours before the
  target, every forecast that projects growth to the target
  over-shoots; this risk is shared by all entrants.
- The house model forecast (superbot, data to 21 Nov 09:30 UTC) may do
  the same gate reasoning. The edge here is the structured Monte Carlo +
  explicit calibration, not any single insight.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input dir, writes one `*.out` per case.
