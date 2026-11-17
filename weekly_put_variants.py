"""
FOUR WEEKLY SHORT-PUT VARIANTS ON SPY, EXITED ON THE INDEX
==========================================================
One trade per Monday-week inside each post-midterm window (1-Jan -> 15-Jul of the year after
the midterm election), which is the only span this folder holds a traded SPY price path for.

    1   sell the nearest-ATM weekly put at Monday's open, hold to Friday's close. No stop, no
        target.
    2   same trade, with the stop and target below applied to the UNDERLYING.
    3   sell the put whose strike is nearest the PREVIOUS week's high, buy a ~3% OTM put, same
        Monday open -> Friday close, same stop and target.
    4   same as 3 but the short strike is nearest the previous week's LOW.

STOP AND TARGET
    They are percentages of the entry SPOT, not of the premium. Every one of these structures
    is long delta (a short put, or a short put financed by a cheaper long put), so

        target  = entry spot * (1 + 1.5995%)      the index moves FOR us
        stop    = entry spot * (1 - 1.1306%)      the index moves AGAINST us

    The index decides WHEN the trade is over; the chain decides WHAT it was worth then. No
    spot number is ever applied to a premium. A bar that touches both levels is booked as the
    stop, because the bar does not say which came first. Levels are live on every bar of the
    hold including the last; if neither is touched the trade ends at Friday's close.

FILLS
    Entry  : Open of the first option bar at or after 09:30 on the entry session.
    Touch  : Close of the option inside the SAME bar the index touched on - you see the level
             break and you are out by the end of that bar.
    Time   : Close of the last option bar of the exit session.

PRICES
    The spot path is the clean minute file put back on SPY's traded scale (spot_bars), because
    the adjusted file would pick a 113 strike against a 127 chain. Chains are the cached
    weekly put pulls in ETFs/data/option_cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ETFS = ROOT / "ETFs"
sys.path.insert(0, str(ETFS))

import spot_bars                                     # noqa: E402
import spot_engine as E                              # noqa: E402

CACHE_DIR = ETFS / "data" / "option_cache"
OUT_XLSX = ROOT / "weekly_put_variants.xlsx"

SYMBOL, OPT_CODE = "SPY", "PE"
MULTIPLIER, QUANTITY = 100, 1

TARGET_PCT = 1.5995
STOP_PCT = 1.1306
OTM_PCT = 3.0                     # the long put of variants 3 and 4, below Monday's spot

ENTRY_TIME = "09:30:00"
ENTRY_LATEST = "15:00:00"         # accept a later entry bar if the strike is silent at 09:30
SESSION_END = "16:00:00"
STRIKE_TRIES = 8


# =============================================================================
# CHAINS
# =============================================================================

def chain_for(entry_day: pd.Timestamp, exit_day: pd.Timestamp):
    """The cached put chain for the week, minute bars, one row per strike per minute."""
    e, x = entry_day.strftime("%Y-%m-%d"), exit_day.strftime("%Y-%m-%d")
    hits = sorted(CACHE_DIR.glob(f"{SYMBOL}_{OPT_CODE}_*_{e}_{x}.pkl.gz"))
    if not hits:
        return None
    best = None
    for p in hits:
        expiry = pd.Timestamp(p.stem.replace(".pkl", "").split("_")[2])
        if best is None or abs((expiry - exit_day).days) < abs((best[0] - exit_day).days):
            best = (expiry, p)
    obj = pd.read_pickle(best[1])
    df = obj.get("df")
    if df is None or df.empty:
        return None
    df = df.rename(columns={"StrikePrice": "Strike", "strikePrice": "Strike"})
    df["Strike"] = pd.to_numeric(df["Strike"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Datetime", "Strike", "Open", "Close"])
    if df.empty:
        return None
    return df.sort_values("Datetime").reset_index(drop=True)


class Leg:
    """One contract's minute bars, with the three lookups the trade needs."""

    def __init__(self, g: pd.DataFrame):
        g = g.sort_values("Datetime")
        self.t = g["Datetime"].to_numpy("datetime64[ns]")
        self.o = g["Open"].to_numpy(float)
        self.c = g["Close"].to_numpy(float)

    def entry(self, day: pd.Timestamp):
        lo = np.datetime64(day.normalize() + pd.Timedelta(ENTRY_TIME), "ns")
        hi = np.datetime64(day.normalize() + pd.Timedelta(ENTRY_LATEST), "ns")
        i = int(np.searchsorted(self.t, lo, "left"))
        if i >= len(self.t) or self.t[i] > hi:
            return None, None
        return float(self.o[i]), pd.Timestamp(self.t[i])

    def close_at_or_before(self, ts):
        j = int(np.searchsorted(self.t, np.datetime64(pd.Timestamp(ts), "ns"), "right")) - 1
        if j < 0:
            return None, None
        return float(self.c[j]), pd.Timestamp(self.t[j])

    def time_exit(self, day: pd.Timestamp):
        return self.close_at_or_before(day.normalize() + pd.Timedelta(SESSION_END))


def pick_strike(chain: pd.DataFrame, want: float, entry_day: pd.Timestamp,
                exit_day: pd.Timestamp, exclude=None):
    """The strike nearest `want` that actually has an entry bar and an exit bar. Walking out
    to the next-nearest strike is the same fallback main.py uses: a strike that printed
    nothing on Monday morning is not a trade, it is a gap in the chain."""
    strikes = np.array(sorted(chain["Strike"].unique()))
    if exclude is not None:
        strikes = strikes[np.abs(strikes - exclude) > 1e-9]
    if len(strikes) == 0:
        return None, None, None, None
    order = strikes[np.argsort(np.abs(strikes - want))][:STRIKE_TRIES]
    for k in order:
        leg = Leg(chain[chain["Strike"] == k])
        px, dt = leg.entry(entry_day)
        xpx, _ = leg.time_exit(exit_day)
        if px is None or xpx is None or not (px > 0):
            continue
        return float(k), leg, px, dt
    return None, None, None, None


# =============================================================================
# THE INDEX DECIDES WHEN
# =============================================================================

def bar_grid(bars: pd.DataFrame, freq: str, entry_dt: pd.Timestamp, exit_day: pd.Timestamp):
    """The bars of the hold, with each bar's END timestamp - the moment the option is marked
    if the index touched a level inside that bar."""
    if freq == "1D":
        d = bars[(bars["Datetime"] >= entry_dt.normalize()) &
                 (bars["Datetime"] <= exit_day.normalize())]
        ends = d["Datetime"].dt.normalize() + pd.Timedelta(SESSION_END)
    else:
        d = bars[(bars["Datetime"] >= entry_dt.floor("h")) &
                 (bars["Datetime"] <= exit_day.normalize() + pd.Timedelta(SESSION_END))]
        ends = (d["Datetime"] + pd.Timedelta(hours=1)).clip(
            upper=d["Datetime"].dt.normalize() + pd.Timedelta(SESSION_END))
    return (d["High"].to_numpy(float), d["Low"].to_numpy(float),
            pd.DatetimeIndex(ends.to_numpy()))


def trigger(bars: pd.DataFrame, freq: str, entry_spot: float, entry_dt: pd.Timestamp,
            exit_day: pd.Timestamp):
    """(reason, mark timestamp). TARGET is the index up, STOP is the index down, because every
    structure here is long delta."""
    tgt = entry_spot * (1 + TARGET_PCT / 100.0)
    stp = entry_spot * (1 - STOP_PCT / 100.0)
    H, L, ends = bar_grid(bars, freq, entry_dt, exit_day)
    for k in range(len(H)):
        up, dn = bool(H[k] >= tgt), bool(L[k] <= stp)
        if not (up or dn):
            continue
        return ("BOTH_STOP" if (up and dn) else ("STOP" if dn else "TARGET")), ends[k]
    return "TIME_EXIT", None


# =============================================================================
# ONE VARIANT
# =============================================================================

def run(variant: int, freq, weeks: pd.DataFrame, b1d: pd.DataFrame, bars) -> pd.DataFrame:
    rows = []
    for _, w in weeks.iterrows():
        entry_day, exit_day = pd.Timestamp(w["entry_day"]), pd.Timestamp(w["exit_day"])
        chain = chain_for(entry_day, exit_day)
        if chain is None:
            continue
        spot_row = b1d[b1d["Datetime"] == entry_day.normalize()]
        if spot_row.empty:
            continue
        entry_spot = float(spot_row["Open"].iloc[0])

        prev_high = prev_low = None
        if variant in (3, 4):
            # anchored on the CALENDAR Monday, not the entry session, so a holiday Monday
            # still looks back at the whole previous week rather than sliding a day
            prev = b1d[(b1d["Datetime"] >= pd.Timestamp(w["monday"]) - pd.Timedelta(days=7)) &
                       (b1d["Datetime"] < pd.Timestamp(w["monday"]))]
            if prev.empty:
                continue
            prev_high, prev_low = float(prev["High"].max()), float(prev["Low"].min())
            want_short = prev_high if variant == 3 else prev_low
        else:
            want_short = entry_spot

        k_s, leg_s, ep_s, entry_dt = pick_strike(chain, want_short, entry_day, exit_day)
        if k_s is None:
            continue

        k_l = leg_l = ep_l = None
        if variant in (3, 4):
            k_l, leg_l, ep_l, _ = pick_strike(chain, entry_spot * (1 - OTM_PCT / 100.0),
                                              entry_day, exit_day, exclude=k_s)
            if k_l is None:
                continue

        if freq is None:
            reason, mark = "TIME_EXIT", None
        else:
            reason, mark = trigger(bars, freq, entry_spot, entry_dt, exit_day)

        if reason == "TIME_EXIT":
            xp_s, xdt = leg_s.time_exit(exit_day)
            xp_l = leg_l.time_exit(exit_day)[0] if leg_l is not None else None
        else:
            xp_s, xdt = leg_s.close_at_or_before(mark)
            xp_l = leg_l.close_at_or_before(mark)[0] if leg_l is not None else None
            if xp_s is None or xdt < entry_dt or (leg_l is not None and xp_l is None):
                xp_s, xdt = leg_s.time_exit(exit_day)
                xp_l = leg_l.time_exit(exit_day)[0] if leg_l is not None else None
                reason = reason + "_NO_OPTION_BAR"

        entry_px = ep_s - (ep_l or 0.0)
        exit_px = xp_s - (xp_l or 0.0)
        rec = {"entry_date": entry_dt, "exit_date": xdt,
               "election_year": int(w["midterm_year"]),
               "entry_strike": k_s, "entry_spot": entry_spot,
               "entry_price": entry_px, "exit_price": exit_px,
               "pnl": (entry_px - exit_px) * MULTIPLIER * QUANTITY,
               "exit_reason": reason,
               "target_pct": TARGET_PCT if freq else np.nan,
               "stop_pct": STOP_PCT if freq else np.nan}
        if variant in (3, 4):
            rec.update({"long_strike": k_l, "long_entry_price": ep_l,
                        "long_exit_price": xp_l,
                        "prev_week_high": prev_high, "prev_week_low": prev_low})
        rows.append(rec)
    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    b1d = spot_bars.spy_bars("1D")
    b1h = spot_bars.spy_bars("1h")
    sessions = pd.DatetimeIndex(sorted(b1d["Datetime"].unique()))
    weeks = E.build_weeks(sessions, E.midterm_cycles(2006, 2022))
    print(f"{len(weeks)} weeks, cycles {sorted(weeks['cycle'].unique().tolist())}")

    sheets, chosen = {}, {}
    plans = [(1, "1_ATM_no_SL", None), (2, "2_ATM_SL_TGT", "pick"),
             (3, "3_PrevHigh_spread", "pick"), (4, "4_PrevLow_spread", "pick")]
    for variant, sheet, mode in plans:
        if mode is None:
            df = run(variant, None, weeks, b1d, None)
            print(f"variant {variant}: {len(df)} trades  pnl {df['pnl'].sum():,.0f}")
        else:
            cand = {}
            for freq, bars in (("1h", b1h), ("1D", b1d)):
                d = run(variant, freq, weeks, b1d, bars)
                cand[freq] = d
                print(f"variant {variant} {freq}: {len(d)} trades  pnl {d['pnl'].sum():,.0f}  "
                      + str(d["exit_reason"].value_counts().to_dict()))
            freq = max(cand, key=lambda f: cand[f]["pnl"].sum())
            chosen[variant] = freq
            df = cand[freq]
            print(f"variant {variant}: using {freq}")
        sheets[sheet] = df

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl", datetime_format="yyyy-mm-dd hh:mm") as xw:
        for name, df in sheets.items():
            out = df.copy()
            num = out.select_dtypes("number").columns
            out[num] = out[num].round(4)
            out.to_excel(xw, sheet_name=name, index=False)
    print(f"\nwrote {OUT_XLSX}")
    print("bar size used:", chosen)


if __name__ == "__main__":
    main()
