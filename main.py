#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster (v2) for make.fun gta6-trophy-rarity-forecast-01.

Forecasts, for every PS5 trophy of Grand Theft Auto VI, five quantiles of its
earn rate on Exophase at the case target (final: 2026-11-26T00:00:00Z).

Method in brief (the full rationale is in the repo README):
  * Earners and players-tracked are modelled separately, then divided: the
    ratio alone is blinded by Exophase refresh jumps in the denominator.
  * Players follow a recency-weighted power law in time, with a weekend
    boost (the final horizon covers Sat-Sun 21-22 Nov, which no history
    contains) and a soft regularisation band.
  * Earners follow a local log-log fit over the last 36h -- on NET earners
    (total minus the launch-night reviewer base, which would otherwise
    dilute the growth slope) -- shrunk toward a per-class prior slope that
    is derived from the class day-7 median itself, and capped by a
    per-class day-7 rate cap. Platinum
    and 100% completion get a hard cap instead of a fit: ~200h of play
    cannot fit in a 168-181h week, so their earners are essentially the
    pre-launch reviewers plus a small Poisson trickle.
  * The answer is the page as of Exophase's last refresh before the target,
    so each Monte Carlo path forecasts at target minus a random refresh lag.
  * Quantiles come from 2,000 deterministic Monte Carlo paths (fixed seed),
    with a spread floor so intervals stay honest over a 4.5-day horizon.
  * Structural ordering is enforced per quantile level: platinum below every
    other trophy, chapters nested (final <= Ch4 <= Ch3 <= Ch2 <= Ch1 <=
    gateway), with a small slack on the 95% quantile.
  * A frozen history (<=2 distinct page states) falls back to a refresh
    mixture: either the page stays put, or it refreshes with a lognormal
    jump in players and class-dependent newcomer earners.

Standard library only, no network, deterministic. Never crashes: any
failure inside the pipeline falls back to a persistence forecast, and any
failure there falls back to fixed valid quantiles.
"""

import json
import math
import os
import random
import sys
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------- constants

SEED = 20261119
N_PATHS = 2000
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]  # standard normal quantiles at LEVELS
Q_MIN, Q_MAX = 0.01, 99.99  # scorer clamps to this range too
GATEWAY_Q = [99.5, 99.9, 99.95, 99.99, 99.99]
ULTRA_FALLBACK_Q = [0.01, 0.1, 1.0, 10.0, 50.0]

# Per-class parameters, as (gamma_class, cap_pct_day7, prior_median_pct_day7,
# prior_sd_logit, newcomer_share). gamma_class is the "normal" log-log slope
# of NET earners (total minus the launch-night reviewer base) for the class
# in week one; cap is a soft upper bound on the day-7 rate; newcomer_share
# is the share of newly synced players who have earned a trophy of this
# class (used by the frozen-history branch).
#
# gamma_class is DERIVED, not guessed: it is the slope implied by the class
# prior median itself, ln(E_new(day7)/E_new(day1)) / ln(168/26), with
# E_new(day1) from the observed day-1 rates (N~20k) minus launch-night
# earners and E_new(day7) = median x N(day7)~250k. The derivation matters:
# the day-7 medians imply net-earner growth of 50-400x over the week, i.e.
# slopes of 2.3-3.2 -- a shrinkage target of ~1.0 would contradict the very
# medians it is supposed to support. Values are rounded to 1 decimal and
# capped at 3.0 (the noisiest derivations, from 2-3 day-1 earners).
CLASSES = {
    "gateway":            (None, 100.0, 99.95, 0.3, 1.0),
    "platinum":           (0.1,  0.05,  0.01,  0.7, 0.0005),
    "completion100":      (0.15, 0.08,  0.015, 0.7, 0.0005),
    "story_ch1":          (2.4,  30.0,  5.0,   1.0, 0.15),
    "story_ch2":          (2.8,  12.0,  1.5,   1.0, 0.05),
    "story_ch3":          (3.0,  6.0,   0.5,   1.0, 0.03),
    "story_ch4":          (2.7,  3.0,   0.3,   1.0, 0.02),
    "story_final":        (2.5,  2.0,   0.2,   1.0, 0.02),
    "story_mission_early": (2.8, 15.0,  3.0,   1.0, 0.10),
    "story_mission":      (3.0,  6.0,   0.8,   1.0, 0.03),
    "strand_finale":      (3.0,  4.0,   0.5,   1.0, 0.03),
    "strand_bundle":      (2.8,  1.0,   0.15,  0.9, 0.01),
    "activity_quick":     (2.5,  20.0,  3.0,   0.8, 0.3),
    "activity_sampler":   (3.0,  5.0,   0.8,   0.9, 0.05),
    "activity_mid":       (2.9,  3.0,   0.3,   0.9, 0.02),
    "grind_count":        (2.6,  2.0,   0.15,  0.9, 0.005),
    "collectible":        (2.6,  1.0,   0.1,   0.9, 0.005),
    "collectible_light":  (2.6,  8.0,   2.0,   0.9, 0.05),
    "explore_all":        (2.6,  1.0,   0.15,  0.9, 0.005),
    "skill":              (2.3,  3.0,   0.2,   0.9, 0.005),
    "skill_grind":        (2.6,  1.0,   0.1,   0.9, 0.005),
    "economy":            (3.0,  5.0,   0.5,   0.9, 0.01),
    "misc_oneoff":        (3.0,  6.0,   0.7,   0.9, 0.05),
    "misc_mid":           (2.9,  3.0,   0.3,   0.9, 0.02),
    "misc_secret":        (2.6,  3.0,   0.4,   1.1, 0.02),
}
FISHING_PRIOR_MEDIAN = 8.0  # fishing is the fastest non-trivial trophy

# ------------------------------------------------------------ math helpers


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def logit_pct(x):
    """logit of a percent, with the scorer's [0.01, 99.99] clamp."""
    p = clamp(x, Q_MIN, Q_MAX) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(l):
    """Inverse of logit_pct: logit -> percent, overflow-safe."""
    if l >= 0.0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def parse_ts(s):
    """Parse an ISO 8601 timestamp to an aware UTC datetime, or None."""
    if not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def weekend_hours(start, end):
    """Hours of [start, end) falling on Saturday/Sunday (UTC)."""
    if end <= start:
        return 0.0
    total = 0.0
    day = start.date()
    last = end.date()
    while day <= last:
        if day.weekday() >= 5:  # Saturday = 5, Sunday = 6
            d0 = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
            d1 = d0 + timedelta(days=1)
            lo = start if start > d0 else d0
            hi = end if end < d1 else d1
            if hi > lo:
                total += (hi - lo).total_seconds() / 3600.0
        day += timedelta(days=1)
    return total


def wols_slope(xs, ys, ws):
    """Weighted least-squares slope of y on x; None if degenerate."""
    if len(xs) < 2:
        return None
    sw = sum(ws)
    if sw <= 0.0:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    return sxy / sxx


def quantile_sorted(vals, q):
    """Linear-interpolation quantile of an already sorted list."""
    n = len(vals)
    if n == 0:
        return 0.0
    if n == 1:
        return vals[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = lo + 1 if lo + 1 < n else lo
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def poisson_draw(rng, lam):
    """Poisson draw: Knuth for small lambda, normal approximation above."""
    if lam <= 0.0:
        return 0
    if lam < 30.0:
        limit = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            p *= rng.random()
            if p <= limit:
                return k
            k += 1
    return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))


# -------------------------------------------------------------- classifier


def classify_trophy(name, desc, tier, last_rate):
    """Classify one trophy from its text. Returns (class, is_fishing).

    Rules are ordered most-specific-first; anything unmatched falls back to
    misc_secret. No trophy id is used, so the classifier is robust to the
    case listing trophies in any order.
    """
    text = ((name or "") + " " + (desc or "")).lower()
    desc_l = (desc or "").lower().strip()
    tier_l = (tier or "").lower()
    if tier_l == "platinum":
        return "platinum", False
    if "100% completion" in text:
        return "completion100", False
    if "gone fishing" in text or (last_rate is not None and last_rate >= 99.0):
        return "gateway", False
    for chapter, cls in (("chapter 1", "story_ch1"), ("chapter 2", "story_ch2"),
                         ("chapter 3", "story_ch3"), ("chapter 4", "story_ch4")):
        if chapter in text:
            return cls, False
    if "final mission for" in text:
        return "strand_finale", False
    if "complete the final mission" in text:
        return "story_final", False
    if "breakfast run" in text:
        return "story_mission_early", False
    if "mission strands for strangers" in text:
        return "strand_bundle", False
    if "fishing" in text:
        return "activity_quick", True
    if "arcade game" in text or "cocktail" in text or "bowling alley" in text:
        return "activity_quick", False
    if "try each type of activity once" in text:
        return "activity_sampler", False
    if "hunt a unique animal" in text:
        return "activity_mid", False
    if "meditate at every spot" in text or "collectible card sets" in text:
        return "collectible", False
    if "discover 100 vehicles" in text:
        return "collectible_light", False
    if "explore all of leonida" in text:
        return "explore_all", False
    if ("gold medal in all shooting range" in text
            or "take down 5 enemies in a single use" in text
            or "without being detected" in text):
        return "skill", False
    if "top fun drone race at every location" in text:
        return "skill_grind", False
    if ("250 stunts" in text or "20 different races" in text
            or "waink it at 100" in text or "50 challenges" in text
            or "5 bank vaults" in text or "register 10 vehicles" in text):
        return "grind_count", False
    if ("laundering" in text or "spend $100,000" in text
            or "purchase all garage properties" in text):
        return "economy", False
    if ("lose a 4 star wanted" in text or "rob 3 shops" in text
            or "unscheduled departure" in text or "kinda hot in these rhinos" in text
            or "rob an armored truck" in text or "smash your way out" in text
            or "bowl 'em over" in text):
        return "misc_oneoff", False
    if "rob an illegal operation" in text:
        return "misc_mid", False
    if desc_l.startswith("complete "):
        return "story_mission", False
    return "misc_secret", False


# ---------------------------------------------------------- case preparation


def rate_at(state, idx):
    """Rate (percent) of trophy idx in a state, or None if absent/invalid."""
    rates = state["rates"]
    if idx < len(rates):
        r = rates[idx]
        if isinstance(r, (int, float)) and math.isfinite(r):
            return float(r)
    return None


def prepare_case(case):
    """Parse a case into the working structure used by both branches."""
    trophies = case.get("trophies") or []
    at_dt = parse_ts(case.get("at"))
    target_dt = parse_ts(case.get("target"))
    horizon_min = case.get("horizon_minutes")
    if not isinstance(horizon_min, (int, float)):
        horizon_min = None
    if horizon_min is None and at_dt is not None and target_dt is not None:
        horizon_min = (target_dt - at_dt).total_seconds() / 60.0
    if horizon_min is None:
        horizon_min = 0.0

    game = case.get("game") or {}
    release_dt = None
    rel = game.get("release_date")
    if isinstance(rel, str):
        try:
            release_dt = datetime.fromisoformat(rel).replace(tzinfo=timezone.utc)
        except ValueError:
            release_dt = None
    if release_dt is None and at_dt is not None:
        release_dt = at_dt - timedelta(minutes=float(horizon_min))
    if release_dt is None:
        release_dt = datetime(2026, 11, 19, tzinfo=timezone.utc)
    if at_dt is None:
        at_dt = release_dt
    if target_dt is None:
        target_dt = at_dt + timedelta(minutes=float(horizon_min))

    # Distinct page states: consecutive reads with identical players + rates
    # are one state (the page had not refreshed); keep the first read of
    # each run, with its timestamp.
    states = []
    prev_key = None
    for read in case.get("history") or []:
        if not isinstance(read, dict):
            continue
        rdt = parse_ts(read.get("at")) or parse_ts(read.get("slot"))
        if rdt is None:
            continue
        n = read.get("players_tracked")
        n = n if isinstance(n, (int, float)) and math.isfinite(n) else 0
        rates = read.get("rates")
        rates = list(rates) if isinstance(rates, list) else []
        key = (n, tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        states.append({
            "dt": rdt,
            "t": (rdt - release_dt).total_seconds() / 3600.0,
            "n": n,
            "rates": rates,
        })

    n_last = 0
    for s in reversed(states):
        if s["n"] > 0:
            n_last = s["n"]
            break
    if states:
        t_last_dt = states[-1]["dt"]
        t_last = states[-1]["t"]
        t_first = states[0]["t"]
        last_rates = [rate_at(states[-1], i) for i in range(len(trophies))]
    else:
        t_last_dt = at_dt
        t_last = (at_dt - release_dt).total_seconds() / 3600.0
        t_first = t_last
        last_rates = [None] * len(trophies)
    span_hours = max(0.0, t_last - t_first)
    staleness_hours = max(0.0, (at_dt - t_last_dt).total_seconds() / 3600.0)

    classes = []
    fishing = []
    for i, tr in enumerate(trophies):
        cls, is_fish = classify_trophy(tr.get("name"), tr.get("description"),
                                       tr.get("tier"), last_rates[i])
        classes.append(cls)
        fishing.append(is_fish)

    prep = {
        "trophies": trophies,
        "at_dt": at_dt,
        "target_dt": target_dt,
        "horizon_hours": float(horizon_min) / 60.0,
        "horizon_days": float(horizon_min) / 1440.0,
        "release_dt": release_dt,
        "states": states,
        "n_last": n_last,
        "t_last_dt": t_last_dt,
        "t_last": t_last,
        "span_hours": span_hours,
        "staleness_hours": staleness_hours,
        "last_rates": last_rates,
        "classes": classes,
        "fishing": fishing,
    }
    prep["params"] = [trophy_params(prep, i) for i in range(len(trophies))]
    return prep


def trophy_params(prep, idx):
    """Per-trophy fit + prior summary, shared by both branches.

    anchor: the most recent state with a non-null rate for this trophy.
    anchor_E is its earners (floored at 0.25 in log space: a printed 0.00%
    still allows a handful of earners); floor_E is the raw earners there
    (earners never decrease, so the forecast may not go below it).
    """
    states = prep["states"]
    t_last = prep["t_last"]
    cls = prep["classes"][idx]
    gamma_c, cap, median, sd_prior, newcomer = CLASSES[cls]
    if prep["fishing"][idx]:
        median = FISHING_PRIOR_MEDIAN

    obs = []  # (t, E_raw, E_log) over distinct states with a rate
    for s in states:
        r = rate_at(s, idx)
        if r is None or s["n"] <= 0:
            continue
        e_raw = r / 100.0 * s["n"]
        obs.append((s["t"], e_raw, max(e_raw, 0.25)))
    if obs:
        anchor_t, anchor_raw, _ = obs[-1]
        anchor_e = max(anchor_raw, 0.25)
        floor_e = max(anchor_raw, 0.0)
        # Last known rate: most recent state that lists this trophy.
        last_rate = None
        for s in reversed(states):
            rr = rate_at(s, idx)
            if rr is not None:
                last_rate = rr
                break
    else:
        anchor_t, anchor_e, floor_e, last_rate = t_last, 0.0, 0.0, None

    # Net earners: the launch-night cohort (pre-launch reviewers) is a
    # CONSTANT base added to every later read. Fitting total earners
    # dilutes the growth slope badly while the base dominates -- early on
    # it is most of E -- so the fit runs on net earners (total minus the
    # base at the earliest observed state) and the base is added back at
    # projection time. A trophy with no net growth by the last state has
    # no growth signal and falls back to its class prior.
    base_e = obs[0][1] if obs else 0.0
    anchor_new_e = max(anchor_e - base_e, 0.0)
    net_obs = [(t, max(e_raw - base_e, 0.0), max(e_raw - base_e, 0.25))
               for t, e_raw, _ in obs]

    # Local log-log fit on the last 36h (or all states if the span is shorter).
    window = 36.0 if prep["span_hours"] >= 36.0 else prep["span_hours"]
    fit_obs = [o for o in net_obs if o[0] >= t_last - window] if window > 0 else list(net_obs)
    n_eff = min(10, sum(1 for _, e_net, _ in fit_obs if e_net > 0))
    has_fit = False
    gamma_star = gamma_c if gamma_c is not None else 0.0
    w_fit = 0.0
    if n_eff >= 3 and len(fit_obs) >= 3 and gamma_c is not None and anchor_new_e > 0:
        xs = [math.log(max(t, 0.25)) for t, _, _ in fit_obs]
        ys = [math.log(e_log) for _, _, e_log in fit_obs]
        ws = [0.5 ** ((t_last - t) / 18.0) * e_log / (e_log + 5.0)
              for t, _, e_log in fit_obs]
        gamma = wols_slope(xs, ys, ws)
        if gamma is not None:
            # Clip allows the prior-consistent range (<=3.0) while still
            # blocking noise explosions from fits on a handful of earners.
            gamma = clamp(gamma, 0.0, 3.5)
            w_fit = n_eff / (n_eff + 3.0)
            gamma_star = w_fit * gamma + (1.0 - w_fit) * gamma_c
            has_fit = True

    # Platinum / 100%: Poisson trickle around the reviewer base.
    lam = 0.0
    if cls == "platinum":
        lam = max(2.0, 0.5 * anchor_e)
        # Safety valve: if earners doubled in the last 24h, the 200h
        # estimate is wrong and the trickle must be allowed to grow.
        recent = [o for o in obs if o[0] >= t_last - 24.0]
        if len(recent) >= 2 and recent[0][1] > 0 and anchor_e >= 2.0 * recent[0][1]:
            lam = max(lam, min(3.0 * anchor_e, (anchor_e - recent[0][1]) * 2.25))
    elif cls == "completion100":
        lam = max(3.0, 0.75 * anchor_e)

    return {
        "cls": cls,
        "cap": cap,
        "median": median,
        "sd_prior": sd_prior,
        "newcomer": newcomer,
        "anchor_t_h": max(anchor_t, 0.25),
        "anchor_E": anchor_e,
        "base_E": base_e,
        "anchor_new_E": max(anchor_new_e, 0.25),
        "floor_E": floor_e,
        "last_rate": last_rate,
        "has_fit": has_fit,
        "gamma_star": gamma_star,
        "w_fit": w_fit,
        "sd_gamma": 0.25 + 0.35 * (1.0 - w_fit),
        "sigma_E": 0.30 + 0.40 / math.sqrt(max(anchor_new_e, 1.0)),
        "lam": lam,
    }


def fit_players_beta(prep):
    """Power-law exponent for players-tracked: mean of the all-history fit
    (recency-weighted, 24h half-life) and the last-24h fit, clipped."""
    states = prep["states"]
    t_last = prep["t_last"]
    pts = [(s["t"], s["n"]) for s in states if s["n"] > 0]
    if len(pts) < 2:
        return 0.8
    xs = [math.log(max(t, 0.25)) for t, _ in pts]
    ys = [math.log(n) for _, n in pts]
    ws = [0.5 ** ((t_last - t) / 24.0) for t, _ in pts]
    b_all = wols_slope(xs, ys, ws)
    if b_all is None:
        b_all = 0.8
    recent = [(x, y, w) for (t, _), x, y, w in zip(pts, xs, ys, ws)
              if t >= t_last - 24.0]
    b_rec = None
    if len(recent) >= 2:
        b_rec = wols_slope([r[0] for r in recent], [r[1] for r in recent],
                           [r[2] for r in recent])
    if b_rec is None:
        b_rec = b_all
    return clamp(0.5 * b_all + 0.5 * b_rec, 0.3, 1.3)


def soft_regularise_n(n_path, n_last):
    """Soft pull of an extrapolated players count into a sane band: the
    geometric mean with the violated bound. Only a guard against runaway
    extrapolation, never a binding constraint on a sane forecast."""
    lo = n_last * 2.0
    hi = n_last * 60.0
    if n_path < lo:
        n_path = math.sqrt(n_path * lo)
    elif n_path > hi:
        n_path = math.sqrt(n_path * hi)
    if n_path < 50000.0:
        n_path = math.sqrt(n_path * 50000.0)
    elif n_path > 1500000.0:
        n_path = math.sqrt(n_path * 1500000.0)
    return n_path


# --------------------------------------------------- branch A: rich history


def draw_rate_branch_a(rng, p, n_path, t_eff_e):
    """One Monte Carlo draw of a trophy's rate (percent) for branch A."""
    cls = p["cls"]
    if cls in ("platinum", "completion100"):
        e = p["anchor_E"] + poisson_draw(rng, p["lam"])
        rate = 100.0 * e / n_path
        return min(rate, p["cap"])
    if not p["has_fit"]:
        rate = p["median"] * math.exp(rng.gauss(0.0, p["sd_prior"]))
    elif rng.random() < 0.15 * (1.0 - p["w_fit"]):
        # Small chance the local fit is wholly wrong: fall back to the
        # class prior for this path (widens the tail toward the prior).
        rate = p["median"] * math.exp(rng.gauss(0.0, p["sd_prior"]))
    else:
        g = clamp(rng.gauss(p["gamma_star"], p["sd_gamma"]), 0.0, 4.0)
        e_new = (p["anchor_new_E"] * (t_eff_e / p["anchor_t_h"]) ** g
                 * math.exp(rng.gauss(0.0, p["sigma_E"])))
        e = p["base_E"] + e_new
        rate = 100.0 * e / n_path
    if rate > p["cap"]:
        rate = p["cap"]
    floor_rate = 100.0 * p["floor_E"] / n_path
    if rate < floor_rate:
        rate = floor_rate
    return rate


def mc_rows_branch_a(prep, rng):
    """Monte Carlo quantiles for the rich-history branch."""
    n_tr = len(prep["trophies"])
    beta_star = fit_players_beta(prep)
    horizon_days = prep["horizon_days"]
    horizon_hours = prep["horizon_hours"]
    sd_beta = 0.15 + 0.10 * (max(horizon_days, 0.0) / 4.5)
    t_last_h = max(prep["t_last"], 0.25)
    n_last = prep["n_last"]
    t_last_dt = prep["t_last_dt"]
    target_dt = prep["target_dt"]
    draws = [[] for _ in range(n_tr)]
    log_weekend_med = math.log(1.5)

    for _ in range(N_PATHS):
        beta_p = clamp(rng.gauss(beta_star, sd_beta), 0.1, 1.6)
        level = rng.gauss(0.0, 0.25)
        w = math.exp(rng.gauss(log_weekend_med, 0.3))
        w_n = w ** 1.15
        w_e = w ** 0.85
        if horizon_hours > 0:
            lag = (rng.uniform(0.0, 2.0) if rng.random() < 0.5
                   else rng.uniform(2.0, 6.0))
            lag = min(lag, horizon_hours)
        else:
            lag = 0.0
        t_eff_dt = target_dt - timedelta(hours=lag)
        h_path = (t_eff_dt - t_last_dt).total_seconds() / 3600.0
        if h_path <= 0.0:
            h_path = 0.0
            w_hours = 0.0
        else:
            w_hours = weekend_hours(t_last_dt, t_eff_dt)
        t_eff_n = max(t_last_h + h_path + (w_n - 1.0) * w_hours, 0.25)
        t_eff_e = max(t_last_h + h_path + (w_e - 1.0) * w_hours, 0.25)
        n_path = n_last * (t_eff_n / t_last_h) ** beta_p * math.exp(level)
        n_path = max(n_path, 1.0)
        if horizon_days >= 1.0:
            n_path = soft_regularise_n(n_path, n_last)
        for idx in range(n_tr):
            if prep["classes"][idx] == "gateway":
                continue
            draws[idx].append(
                draw_rate_branch_a(rng, prep["params"][idx], n_path, t_eff_e))

    rows = {}
    for idx in range(n_tr):
        if prep["classes"][idx] == "gateway":
            rows[idx] = list(GATEWAY_Q)
            continue
        vals = sorted(draws[idx])
        rows[idx] = [quantile_sorted(vals, q) for q in LEVELS]
    return rows


# ---------------------------------------------- branch B: frozen history


def mc_rows_branch_b(prep, rng):
    """Refresh-mixture quantiles for a frozen history (<=2 distinct states).

    Either the page stays put until the target (rates exactly as last
    read), or it refreshes once: players jump by a lognormal factor and
    earners grow by the newcomers who earned each trophy. The mixture is
    genuinely bimodal, and its spread is honestly wide.
    """
    n_tr = len(prep["trophies"])
    horizon_hours = max(prep["horizon_hours"], 0.0)
    horizon_days = max(prep["horizon_days"], 0.0)
    if len(prep["states"]) <= 1:
        # Frozen since the first read: the page was already stale when the
        # recording started, so it is at least ~4h stale.
        staleness_eff = max(prep["staleness_hours"], 4.0)
    else:
        staleness_eff = prep["staleness_hours"]
    p_refresh = 1.0 - math.exp(-(((horizon_hours + staleness_eff) / 6.0) ** 1.5))
    p_refresh = clamp(p_refresh, 0.0, 1.0)
    n_last = prep["n_last"]
    draws = [[] for _ in range(n_tr)]
    log_jump_med = math.log(4.0)

    for _ in range(N_PATHS):
        refresh = rng.random() < p_refresh
        if refresh:
            jump = math.exp(rng.gauss(log_jump_med, 0.7))
            n_t = max(n_last * jump, 1.0)
        else:
            jump = 1.0
            n_t = max(n_last, 1.0)
        for idx in range(n_tr):
            if prep["classes"][idx] == "gateway":
                continue
            p = prep["params"][idx]
            if not refresh:
                rate = p["last_rate"] if p["last_rate"] is not None else p["median"]
            else:
                # Newcomers have had far less playtime than the cohort that
                # produced the current rate, so their earn share cannot
                # exceed a small multiple of the current rate: without this
                # cap the refresh branch would forecast rates HIGHER than
                # the frozen page, contradicting every observed refresh
                # (denominator jumps far faster than earners). The cap
                # relaxes with horizon: over days, newcomers catch up.
                last_r = p["last_rate"] if p["last_rate"] is not None else p["median"]
                share_cap = max(last_r, 0.05) / 100.0 * 0.15 * (1.0 + horizon_days)
                share = min(p["newcomer"], share_cap)
                e_new = (jump - 1.0) * n_last * share
                e = p["floor_E"] + e_new * math.exp(rng.gauss(0.0, 0.15))
                rate = 100.0 * e / n_t
                if rate > p["cap"]:
                    rate = p["cap"]
                floor_rate = 100.0 * p["floor_E"] / n_t
                if rate < floor_rate:
                    rate = floor_rate
            draws[idx].append(rate)

    rows = {}
    for idx in range(n_tr):
        if prep["classes"][idx] == "gateway":
            rows[idx] = list(GATEWAY_Q)
            continue
        vals = sorted(draws[idx])
        rows[idx] = [quantile_sorted(vals, q) for q in LEVELS]
    return rows


# ------------------------------------------- calibration, structure, output


def calibrate_quantiles(qs, horizon_days, branch_b):
    """Spread floor + cap in logit space, centred on the median.

    The model is never as certain as a narrow Monte Carlo interval claims
    over a multi-day horizon (refresh lag, weekend, S-curve takeoff), so
    intervals narrower than the floor are widened; absurdly wide ones
    (sd > 2.5 logit) are narrowed. No sharpening: the median is not
    accurate enough for it to pay under pinball loss.
    """
    if branch_b:
        s_min = 0.8
    else:
        s_min = 0.35 + 0.15 * math.sqrt(max(horizon_days, 0.0))
    s = (logit_pct(qs[4]) - logit_pct(qs[0])) / (2.0 * 1.645)
    if s_min <= s <= 2.5:
        return list(qs)
    target_s = s_min if s < s_min else 2.5
    mid = logit_pct(qs[2])
    return [inv_logit_pct(mid + z * target_s) for z in Z]


def enforce_structure(rows, classes):
    """Per-level structural ordering (in place).

    Chain, enforced top-down so it is consistent: gateway >= Ch1 >= Ch2 >=
    Ch3 >= Ch4 >= final mission; platinum <= every other trophy. The 95%
    quantile gets a 15% relative slack (small nesting inversions are real
    in tracker data: cheats, save imports, offline play).
    """
    n = len(rows)

    def first_of(cls):
        for i, c in enumerate(classes):
            if c == cls:
                return i
        return None

    plat = first_of("platinum")
    chain = [first_of("gateway"), first_of("story_ch1"), first_of("story_ch2"),
             first_of("story_ch3"), first_of("story_ch4"), first_of("story_final")]
    for tau in range(5):
        slack = 1.15 if tau == 4 else 1.0
        prev = None
        for idx in chain:
            if idx is None:
                continue
            if prev is not None:
                bound = rows[prev][tau] * slack
                if rows[idx][tau] > bound:
                    rows[idx][tau] = bound
            prev = idx
        if plat is not None:
            m = min(rows[i][tau] for i in range(n) if i != plat)
            if rows[plat][tau] > m * slack:
                rows[plat][tau] = m * slack
    return rows


def finalize_row(qs):
    """Clamp to [0.01, 99.99], round to 4 decimals, force non-decreasing."""
    out = []
    for q in qs:
        try:
            v = float(q)
        except (TypeError, ValueError):
            v = 1.0
        if not math.isfinite(v):
            v = 1.0
        out.append(clamp(round(v, 4), Q_MIN, Q_MAX))
    out.sort()
    return out


def fallback_rows(prep):
    """Persistence forecast: last rate as median, random-walk-like spread.

    Used when the main pipeline fails for any reason. Gateway keeps its
    fixed quantiles; a trophy never seen in history centres on its class
    prior median.
    """
    horizon_days = max(prep["horizon_days"], 0.0)
    sd = max(0.5 * math.sqrt(horizon_days),
             0.35 + 0.15 * math.sqrt(horizon_days))
    rows = {}
    for idx in range(len(prep["trophies"])):
        if prep["classes"][idx] == "gateway":
            rows[idx] = list(GATEWAY_Q)
            continue
        center = prep["last_rates"][idx]
        if center is None:
            center = prep["params"][idx]["median"]
        base = logit_pct(center)
        rows[idx] = [inv_logit_pct(base + z * sd) for z in Z]
    return rows


def forecast_case(case):
    """Forecast one case. Never raises: always returns a valid document."""
    prep = prepare_case(case)
    n_tr = len(prep["trophies"])
    if n_tr == 0:
        return {"trophies": []}
    rng = random.Random(SEED)
    rows = None
    try:
        if len(prep["states"]) >= 3 and prep["n_last"] > 0:
            rows = mc_rows_branch_a(prep, rng)
            for idx in range(n_tr):
                if prep["classes"][idx] != "gateway":
                    rows[idx] = calibrate_quantiles(rows[idx], prep["horizon_days"], False)
        elif prep["states"] and prep["n_last"] > 0:
            rows = mc_rows_branch_b(prep, rng)
            for idx in range(n_tr):
                if prep["classes"][idx] != "gateway":
                    rows[idx] = calibrate_quantiles(rows[idx], prep["horizon_days"], True)
    except Exception:
        rows = None
    if rows is None:
        try:
            rows = fallback_rows(prep)
        except Exception:
            rows = {i: list(ULTRA_FALLBACK_Q) for i in range(n_tr)}
    try:
        enforce_structure(rows, prep["classes"])
    except Exception:
        pass
    out_rows = []
    for i, tr in enumerate(prep["trophies"]):
        try:
            q = finalize_row(rows[i])
        except Exception:
            q = list(ULTRA_FALLBACK_Q)
        out_rows.append({"id": tr.get("id"), "q": q})
    return {"trophies": out_rows}


# ------------------------------------------------------------------- entry


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    try:
        names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    except Exception:
        names = []
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        return 0
    for name in names:
        case_id = name[:-3]
        try:
            with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
                case = json.load(f)
        except Exception:
            continue
        if not isinstance(case, dict):
            continue
        try:
            result = forecast_case(case)
        except Exception:
            continue
        try:
            with open(os.path.join(out_dir, case_id + ".out"), "w",
                      encoding="utf-8") as f:
                json.dump(result, f)
        except Exception:
            continue
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)
