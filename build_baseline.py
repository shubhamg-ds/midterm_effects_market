"""
THE ONE BOOK, AND THE GATED TRADE SHEET BUILT ON IT
===================================================
Baseline: SHORT the nearest-ATM weekly put at the week's first open, out at the week's last
close, stop and target as percentages of the ENTRY SPOT, on 1D trigger bars. That is
weekly_put_variants.py variant 2, and on 1-Jan -> 15-Jul it reproduces the 5,075 total in the
brief's options pivot exactly - which is the whole point, because the gated sheet is compared
against that pivot.

    BUY_3PCT_OTM_WING = False   the naked short put, the structure the brief's trade sheet shows
                      = True    the same short put with a ~3% OTM long put against it

Nothing else is run. The previous pass also simulated the previous-week-high spread, the
previous-week-low spread and the long spot book; those are gone.

THE SHEET
    One row per Monday-week in the window, INCLUDING the weeks the gate skipped - with what
    those weeks would have paid - because "did we give up profit or avoid a loss" cannot be
    answered from a sheet that only lists the trades that happened.

        pnl         the baseline trade, every week, gate ignored
        size        the gate's position multiplier for that week (1.0 / 0.5 / 0.0)
        pnl_gated   pnl * size, what the gated book actually booked
        pnl_forgone pnl * (1 - size), signed: positive = profit given up,
                                              negative = loss avoided
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ETFs"))

import risk_gate as R                                      # noqa: E402
import spot_bars                                           # noqa: E402
import weekly_put_variants as W                            # noqa: E402

TRIGGER_FREQ = "1D"                 # the grid the 5,075 was produced on
BUY_3PCT_OTM_WING = False
CACHE = ROOT / "data" / "processed" / "short_put_book.pkl.gz"

SHEET_COLS = ["entry_date", "exit_date", "election_year", "entry_strike", "entry_spot",
              "entry_price", "exit_price", "pnl", "exit_reason", "target_pct", "stop_pct"]


def run_short_put(weeks: pd.DataFrame, b1d: pd.DataFrame,
                  wing: bool = BUY_3PCT_OTM_WING) -> pd.DataFrame:
    """The weekly short put, week by week, with the brief's own trade-sheet columns.

    Same machinery as weekly_put_variants: W.chain_for for the week's cached chain,
    W.pick_strike for a strike that actually printed an entry AND an exit bar, W.trigger for
    when the index ended the trade. Fills: entry at the first option bar at or after 09:30;
    on a stop or target, the option's close inside the SAME bar the index touched on; otherwise
    the last option bar of the exit session.
    """
    rows = []
    for _, w in weeks.iterrows():
        entry_day, exit_day = pd.Timestamp(w["entry_day"]), pd.Timestamp(w["exit_day"])
        srow = b1d[b1d["Datetime"] == entry_day.normalize()]
        if srow.empty:
            continue
        entry_spot = float(srow["Open"].iloc[0])
        base = {"monday": pd.Timestamp(w["monday"]), "cycle": int(w["cycle"]),
                "election_year": int(w["midterm_year"]), "entry_spot": entry_spot,
                "target_pct": R.TARGET_PCT, "stop_pct": R.STOP_PCT}

        chain = W.chain_for(entry_day, exit_day)
        if chain is None:
            rows.append({**base, "entry_date": entry_day + pd.Timedelta("09:30:00"),
                         "exit_date": pd.NaT, "entry_strike": np.nan, "entry_price": np.nan,
                         "exit_price": np.nan, "pnl": np.nan, "exit_reason": "NO_CHAIN"})
            continue

        k_s, leg_s, ep_s, entry_dt = W.pick_strike(chain, entry_spot, entry_day, exit_day)
        if k_s is None:
            rows.append({**base, "entry_date": entry_day + pd.Timedelta("09:30:00"),
                         "exit_date": pd.NaT, "entry_strike": np.nan, "entry_price": np.nan,
                         "exit_price": np.nan, "pnl": np.nan, "exit_reason": "NO_STRIKE"})
            continue

        k_l = leg_l = ep_l = None
        if wing:
            k_l, leg_l, ep_l, _ = W.pick_strike(chain, entry_spot * (1 - R.OTM_PCT / 100),
                                                entry_day, exit_day, exclude=k_s)
            if k_l is None:
                rows.append({**base, "entry_date": entry_dt, "exit_date": pd.NaT,
                             "entry_strike": k_s, "entry_price": np.nan, "exit_price": np.nan,
                             "pnl": np.nan, "exit_reason": "NO_WING"})
                continue

        reason, mark = W.trigger(b1d, TRIGGER_FREQ, entry_spot, entry_dt, exit_day)
        if reason == "TIME_EXIT":
            xp_s, xdt = leg_s.time_exit(exit_day)
            xp_l = leg_l.time_exit(exit_day)[0] if leg_l is not None else None
        else:
            xp_s, xdt = leg_s.close_at_or_before(mark)
            xp_l = leg_l.close_at_or_before(mark)[0] if leg_l is not None else None
            if xp_s is None or xdt < entry_dt or (leg_l is not None and xp_l is None):
                xp_s, xdt = leg_s.time_exit(exit_day)
                xp_l = leg_l.time_exit(exit_day)[0] if leg_l is not None else None
                reason += "_NO_OPTION_BAR"

        ep = ep_s - (ep_l or 0.0)
        xp = xp_s - (xp_l or 0.0)
        rec = {**base, "entry_date": entry_dt, "exit_date": xdt, "entry_strike": k_s,
               "entry_price": ep, "exit_price": xp,
               "pnl": (ep - xp) * R.MULTIPLIER * R.QUANTITY, "exit_reason": reason}
        if wing:
            rec.update({"long_strike": k_l, "long_entry_price": ep_l, "long_exit_price": xp_l})
        rows.append(rec)
    return pd.DataFrame(rows)


def build(force: bool = False, wing: bool = BUY_3PCT_OTM_WING) -> dict:
    """The baseline book: 1-Jan -> 15-Jul of each post-midterm year, the only window this
    strategy trades. All four cycles have chains for it, 2011 included - the Nov-Dec gap that
    truncated an earlier pass does not touch this window."""
    if CACHE.exists() and not force:
        out = pd.read_pickle(CACHE)
        if out.get("wing") == wing:
            return out
    b1d = spot_bars.spy_bars("1D")
    sessions = pd.DatetimeIndex(sorted(b1d["Datetime"].unique()))
    weeks = R.build_weeks(sessions, R.cycle_windows())
    weeks = weeks[weeks["cycle"].isin([2011, 2015, 2019, 2023])].reset_index(drop=True)
    d = run_short_put(weeks, b1d, wing=wing)
    ok = d["pnl"].notna()
    print(f"{len(weeks)} weeks, {int(ok.sum())} tradeable, pnl {d.loc[ok, 'pnl'].sum():,.0f}"
          + (f"  missing: {d.loc[~ok, 'exit_reason'].value_counts().to_dict()}"
             if (~ok).any() else ""), flush=True)
    out = {"wing": wing, "weeks": weeks, "book": d}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, CACHE)
    return out


def make_sheet(book: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """Join the gate's per-week decision onto the baseline book and compute the gated P&L.

    A week with no chain carries size 0 and reason NO_CHAIN - not a gate skip, a data gap, and
    the sheet distinguishes the two so the accounting is not quietly credited with it."""
    keep = ["monday", "mname", "p_stop", "p_target", "edge", "edge_cut", "size", "n_train",
            "train_cycles", "dd_w", "mfe_w", "ret_w", "loss_w", "target_hit_w",
            "rv5", "rv10", "rv20", "rv25", "rv5_over_rv20", "nd_ret", "nd_rv", "wk_idx"]
    d = book.merge(decisions[[c for c in keep if c in decisions.columns]],
                   on="monday", how="left")
    d["size"] = d["size"].fillna(1.0)                 # no decision -> the baseline trade
    tradeable = d["pnl"].notna()
    d.loc[~tradeable, "size"] = 0.0
    d["pnl_gated"] = d["pnl"].fillna(0.0) * d["size"]
    d["pnl_forgone"] = d["pnl"].fillna(0.0) * (1.0 - d["size"])
    d["decision"] = np.where(~tradeable, "NO_TRADE_" + d["exit_reason"].astype(str),
                      np.where(d["size"] >= 1, "TAKE",
                        np.where(d["size"] > 0, "REDUCE", "SKIP")))
    d["gate_outcome"] = np.where((d["size"] >= 1) | (~tradeable), "-",
                          np.where(d["pnl_forgone"] > 0, "PROFIT_SACRIFICED",
                            np.where(d["pnl_forgone"] < 0, "LOSS_AVOIDED", "FLAT")))
    d["taken"] = d["size"] > 0
    lead = SHEET_COLS + ["mname", "p_stop", "p_target", "edge", "edge_cut",
                         "decision", "size", "taken", "pnl_gated", "pnl_forgone",
                         "gate_outcome", "train_cycles"]
    rest = [c for c in d.columns if c not in lead]
    return d[[c for c in lead if c in d.columns] + rest].sort_values("entry_date") \
            .reset_index(drop=True)


if __name__ == "__main__":
    build(force=True)
