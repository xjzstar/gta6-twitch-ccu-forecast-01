"""Hourly quantile forecast of GTA VI's launch-day Twitch viewership.

For every case file in the input directory this program predicts, for each
hour 2..24 of launch day, five quantiles (5/25/50/75/95%) of the concurrent
viewers in the game's Twitch category.

Method: a launch's whole day is summarised by the log-ratio of each hour's
viewers to hour 1's viewers.  The case file carries past launches
("comparables"); only those whose own hour 1 drew at least 1,000 viewers are
pooled, because launches that opened to a near-empty category and exploded
later follow a completely different curve.  For each hour, the median
log-ratio is the geometric-mean blend of two local estimates over
ln(hour-1 viewers): a k-nearest-neighbour weighted average (k = 8, Gaussian
kernel, bandwidth 0.7) and a kernel-weighted local-linear regression.  The
spread comes from the residuals around that regression, widened a little as
the day goes on, and the five quantiles are median * exp(z * sd).  Medians
are lightly smoothed across neighbouring hours, log-ratios are clamped to
the range the pool actually shows, and every failure path falls back to a
persistence forecast so a case always yields a valid file.

Standard library only, offline, deterministic.

Local run:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
HOURS = list(range(2, 25))

MIN_POOL_H1 = 1000      # a comparable's launch-day hour 1 must reach this
NEIGHBOURS = 8          # k for the kNN estimate
BANDWIDTH = 0.7         # Gaussian kernel width, in ln(viewers) units
MIN_SD = 0.05           # never report a tighter spread than this
DRIFT_SD = 0.05         # extra log-sd accumulated per sqrt(hour - 1)
BASE_SD = 0.15          # persistence fallback spread, per sqrt(hour - 1)
MAX_VIEWERS = 1e9


def round2(x):
    """Round to two decimals, halves away from zero; whole values stay ints."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if float(value).is_integer() else value


def persistence_row(p1, h):
    """Flat median at hour 1's print, spread growing like a random walk."""
    sd = BASE_SD * math.sqrt(h - 1)
    return [round2(p1 * math.exp(z * sd)) for z in Z]


def launch_curve(comp):
    """(ln of hour-1 viewers, {hour: ln(viewers_h / viewers_1)}) or None.

    A comparable's launch day starts at the hourly row whose `from` equals
    the comparable's `day_start`; the 24 rows from there are its day.
    """
    try:
        hourly = comp["hourly"]
        start = comp["day_start"]
        idx = None
        for i, row in enumerate(hourly):
            if row.get("from") == start:
                idx = i
                break
        if idx is None or len(hourly) - idx < 24:
            return None
        day = hourly[idx:idx + 24]
        first = max(1.0, float(day[0]["viewers"]))
        if first < MIN_POOL_H1:
            return None
        ratios = {}
        for h in HOURS:
            viewers = max(1.0, float(day[h - 1]["viewers"]))
            ratios[h] = math.log(viewers / first)
        return math.log(first), ratios
    except (KeyError, TypeError, ValueError):
        return None


def build_pool(case):
    """The case's usable comparables, sorted by name for stable order."""
    pool = []
    for comp in sorted(case.get("comparables") or [], key=lambda c: str(c.get("name", ""))):
        curve = launch_curve(comp)
        if curve is not None:
            pool.append(curve)
    return pool


def kernel(dx):
    return math.exp(-0.5 * (dx / BANDWIDTH) ** 2)


def knn_ratio(pool, xt, h):
    """Weighted mean log-ratio at hour h over the NEIGHBOURS closest launches."""
    ranked = sorted(pool, key=lambda entry: abs(entry[0] - xt))
    near = ranked[:NEIGHBOURS]
    wsum = 0.0
    total = 0.0
    for x, ratios in near:
        w = kernel(x - xt)
        wsum += w
        total += w * ratios[h]
    if wsum <= 0.0:
        return sum(ratios[h] for _, ratios in near) / len(near)
    return total / wsum


def local_linear(pool, xt, h):
    """Kernel-weighted line fit of log-ratio on ln(hour 1), at the target.

    Returns (fitted value at xt, residual sd around the fit).
    """
    s0 = s1 = s2 = t0 = t1 = 0.0
    for x, ratios in pool:
        w = kernel(x - xt)
        dx = x - xt
        r = ratios[h]
        s0 += w
        s1 += w * dx
        s2 += w * dx * dx
        t0 += w * r
        t1 += w * dx * r
    if s0 <= 0.0:
        return None, None
    det = s0 * s2 - s1 * s1
    if abs(det) < 1e-12:
        fit = t0 / s0
        slope = 0.0
    else:
        fit = (s2 * t0 - s1 * t1) / det
        slope = (s0 * t1 - s1 * t0) / det
    var = 0.0
    for x, ratios in pool:
        w = kernel(x - xt)
        resid = ratios[h] - (fit + slope * (x - xt))
        var += w * resid * resid
    return fit, math.sqrt(max(var / s0, 0.0))


def model_rows(p1, pool):
    """The blended forecast: {hour: [q05, q25, q50, q75, q95]}."""
    xt = math.log(p1)
    medians = {}
    spreads = {}
    for h in HOURS:
        observed = [ratios[h] for _, ratios in pool]
        lo, hi = min(observed), max(observed)
        near = knn_ratio(pool, xt, h)
        fit, sd = local_linear(pool, xt, h)
        if fit is None:
            blended = near
            sd = 0.3
        else:
            blended = 0.5 * (near + fit)
        medians[h] = min(max(blended, lo), hi)
        spreads[h] = max(sd, MIN_SD)
    # Light smoothing of the median curve: each hour leans on its neighbours.
    smooth = {}
    for pos, h in enumerate(HOURS):
        acc = medians[h] * 2.0
        weight = 2.0
        if pos > 0:
            acc += medians[HOURS[pos - 1]]
            weight += 1.0
        if pos < len(HOURS) - 1:
            acc += medians[HOURS[pos + 1]]
            weight += 1.0
        smooth[h] = acc / weight
    rows = {}
    for h in HOURS:
        sd = math.sqrt(spreads[h] ** 2 + DRIFT_SD ** 2 * (h - 1))
        centre = p1 * math.exp(smooth[h])
        qs = [centre * math.exp(z * sd) for z in Z]
        qs = [min(max(q, 1e-6), MAX_VIEWERS) for q in qs]
        qs.sort()
        rows[h] = [round2(q) for q in qs]
    return rows


def hour_one(case):
    for row in case.get("known") or []:
        if row.get("hour") == 1:
            return max(1.0, float(row["viewers"]))
    return None


def forecast_case(case):
    """Forecast one case; any surprise degrades to persistence, never a crash."""
    p1 = hour_one(case)
    if p1 is None:
        p1 = 1.0
    try:
        pool = build_pool(case)
        if len(pool) >= 3:
            rows = model_rows(p1, pool)
        else:
            rows = {h: persistence_row(p1, h) for h in HOURS}
    except Exception:
        rows = {h: persistence_row(p1, h) for h in HOURS}
    return {"hours": [{"hour": h, "q": rows[h]} for h in HOURS]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            result = forecast_case(case)
        except Exception:
            result = {"hours": [{"hour": h, "q": persistence_row(1.0, h)} for h in HOURS]}
        text = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
