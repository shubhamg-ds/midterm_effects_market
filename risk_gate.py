"""
POST-MIDTERM WEEKLY SHORT PUT  -  QUANTILE-ONLY DOWNSIDE GATE
=============================================================
One strategy. Short the nearest-ATM weekly put at the week's first open, out at the week's last
close, one trade per Monday-week, 1-Jan -> 15-Jul of the year after a midterm election. Stop and
target are percentages of the ENTRY SPOT and the index decides when the trade is over:

    target = entry_spot * (1 + 1.5995%)      stop = entry_spot * (1 - 1.1306%)
    otherwise the week's last close (Friday)

THE ONE IDEA THAT CHANGED EVERYTHING
    The option chains cover 4 post-midterm cycles. The SPX tape covers SIXTEEN
    (1963, 1967, ... 2023). An earlier pass trained the risk model on general market history and
    found the signal worked everywhere EXCEPT inside post-midterm windows - because that is
    where it had never been trained. So the model here is fitted on POST-MIDTERM WEEKS ONLY,
    expanding by cycle: the 2011 cycle trains on 1963-2007 (336 weeks), 2015 on 1963-2011, and
    so on. Every one of the 112 traded weeks gets a decision, 2011 included.

NO LOGISTIC REGRESSION
    It was tried and it failed (AUC ~0.50 out of sample). The probabilities here come straight
    out of the quantile functions instead, which is what the brief means by "if logistic is
    failing, we stick to quantile":

        p_stop   = the tau at which  Q_tau(dd_w)  = -1.1306      (dd is signed, <= 0)
        p_target = 1 - the tau at which Q_tau(mfe_w) = +1.5995
        edge     = p_target - p_stop

    Fit the quantile regression across a grid of taus, sort each week's predictions to kill
    quantile crossing, and read off the tau where the predicted quantile crosses the actual
    stop / target level. That IS the model's probability of the stop firing, with no second
    model bolted on. `edge` is the quantity the gate ranks weeks by: it compares the two tails
    against the two levels the trade actually has.

FEATURES
    month-of-window dummies (Jan is the base level), fast and slow realized vol, and the
    "deciding months" the brief asks for - the realized return and vol of Nov-Dec of the
    midterm year, which are known before the window opens. The month dummy uses the week's
    SCHEDULED last session, which is known at entry; the pivot attributes P&L by the option's
    ACTUAL exit date, which is the brief's own convention. The two differ only for weeks that
    straddle a month end.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
SPX_CSV = ROOT / "data" / "SP_DLY_SPX, 1D.csv"
CACHE_DIR = ROOT / "data" / "processed"

TARGET_PCT = 1.5995
STOP_PCT = 1.1306
OTM_PCT = 3.0
MULTIPLIER, QUANTITY = 100, 1

MIDTERM_YEARS = [y for y in range(1962, 2027) if y % 4 == 2]
DECISION_TAU = 0.10
MONTH_HORIZON = 21

# the tau grid the probabilities are read off. Dense in the tails, which is where the stop and
# the target sit; a sparse grid there would quantise p_stop into visible steps.
TAUS = np.array([.02, .04, .06, .08, .10, .13, .16, .20, .25, .30, .35, .40, .50,
                 .60, .70, .80, .90, .95, .98])
MONTHS = ["Feb", "Mar", "Apr", "May", "Jun", "Jul"]        # Jan is the base level
MONTH_ORDER = [1, 2, 3, 4, 5, 6, 7]

FAST_FEATS = ["rv5", "rv10", "rv5_over_rv20"]
SLOW_FEATS = ["rv20", "rv25"]
MONTH_DUMMIES = [f"m_{m}" for m in MONTHS]
FEATURE_SETS = {
    "month_only": MONTH_DUMMIES,
    "rv_only": ["rv5", "rv20"],
    "month_rv": MONTH_DUMMIES + ["rv5", "rv20"],
    "month_rv_novdec": MONTH_DUMMIES + ["rv5", "rv20", "nd_ret", "nd_rv"],
}


def cached(key: str, fn, force: bool = False):
    p = CACHE_DIR / f"gate_{key}.pkl.gz"
    if p.exists() and not force:
        return pd.read_pickle(p)
    out = fn()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, p)
    return out


# =============================================================================
# PRICES, WINDOWS, WEEKS
# =============================================================================

def load_spx() -> pd.DataFrame:
    """SPX daily OHLC. A price index, not a total-return series, so a weekly % excursion off it
    is the move a stop would actually have seen."""
    d = pd.read_csv(SPX_CSV)
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce").dt.tz_localize(None)
    d = d.rename(columns={"time": "Datetime", "open": "Open", "high": "High",
                          "low": "Low", "close": "Close"})
    d = d.dropna(subset=["Datetime", "Open", "High", "Low", "Close"])
    d = d[["Datetime", "Open", "High", "Low", "Close"]].sort_values("Datetime")
    # the pre-1962 tape prints open=high=low=close on most days: a zero-range bar carries no
    # excursion information, so those sessions cannot teach a model about drawdown
    return d[d["High"] > d["Low"] + 1e-9].reset_index(drop=True)


def cycle_windows(first_cycle: int = 1963, last_cycle: int = 2023) -> list[dict]:
    """1-Jan -> 15-Jul of the year after each midterm election. This is the ONLY window the
    strategy trades; Nov-Dec of the midterm year is a feature window, never a traded one."""
    return [{"midterm_year": y, "cycle": y + 1,
             "start": pd.Timestamp(y + 1, 1, 1), "end": pd.Timestamp(y + 1, 7, 15)}
            for y in MIDTERM_YEARS if first_cycle <= y + 1 <= last_cycle]


def build_weeks(sessions: pd.DatetimeIndex, windows: list[dict]) -> pd.DataFrame:
    """One row per Monday-week: the session the trade enters on and the one it exits on.
    Positional on the week's sessions, so a Monday holiday moves the entry to Tuesday instead
    of throwing the week away. One trade per week, always."""
    rows = []
    for w in windows:
        start, end = w["start"], w["end"]
        first_monday = start - pd.Timedelta(days=int(start.weekday()))
        for monday in pd.date_range(first_monday, end, freq="W-MON"):
            lo, hi = max(monday, start), min(monday + pd.Timedelta(days=4), end)
            inweek = sessions[(sessions >= lo) & (sessions <= hi)]
            if len(inweek) < 2:
                continue
            rows.append({"cycle": w["cycle"], "midterm_year": w["midterm_year"],
                         "monday": monday, "entry_day": inweek[0], "exit_day": inweek[-1],
                         "sessions_in_week": len(inweek)})
    return pd.DataFrame(rows)


# =============================================================================
# FEATURES AND LABELS
# =============================================================================

def realized_vol_table(px: pd.DataFrame, windows=(5, 10, 15, 20, 25)) -> pd.DataFrame:
    """Annualised realized vol in %, indexed by the session it is MEASURED ON. A feature
    stamped on session t uses returns up to and including t, so a week entering on t+1 is fed
    only history."""
    d = px[["Datetime", "Close"]].copy()
    d["r"] = np.log(d["Close"]).diff()
    for n in windows:
        d[f"rv{n}"] = d["r"].rolling(n).std(ddof=1) * math.sqrt(252) * 100.0
    d["rv5_over_rv20"] = d["rv5"] / d["rv20"]
    return d.drop(columns=["r"])


def excursions(px: pd.DataFrame, weeks: pd.DataFrame) -> pd.DataFrame:
    """Per week, the SIGNED worst drawdown and the favourable excursion, in % off the entry open:

        dd_w  = (min Low  over the week / entry open - 1) * 100   <= 0
        mfe_w = (max High over the week / entry open - 1) * 100   >= 0

    Signed, because the quantile of interest is the LOWER one: Q_0.10(dd) is the drawdown only
    one week in ten is worse than. loss_w / target_hit_w are the two events the stop and the
    target are defined by."""
    dt = px["Datetime"].to_numpy("datetime64[ns]")
    O, H, L, Cl = (px[c].to_numpy(float) for c in ("Open", "High", "Low", "Close"))
    n = len(px)
    rows = []
    for _, w in weeks.iterrows():
        tgt = np.datetime64(pd.Timestamp(w["entry_day"]), "ns")
        i = int(np.searchsorted(dt, tgt))
        j = int(np.searchsorted(dt, np.datetime64(pd.Timestamp(w["exit_day"]), "ns")))
        if i >= n or dt[i] != tgt or j < i:
            continue
        ent, k = O[i], min(j + 1, n)
        km = min(i + MONTH_HORIZON, n)
        rec = dict(w)
        rec["entry_open"] = ent
        rec["dd_w"] = min(0.0, (L[i:k].min() / ent - 1.0) * 100.0)
        rec["mfe_w"] = max(0.0, (H[i:k].max() / ent - 1.0) * 100.0)
        rec["ret_w"] = (Cl[j] / ent - 1.0) * 100.0
        rec["dd_m"] = min(0.0, (L[i:km].min() / ent - 1.0) * 100.0)
        rec["loss_w"] = int(-rec["dd_w"] >= STOP_PCT)
        rec["target_hit_w"] = int(rec["mfe_w"] >= TARGET_PCT)
        rows.append(rec)
    return pd.DataFrame(rows)


def novdec_features(px: pd.DataFrame, cycles: list[int]) -> pd.DataFrame:
    """The brief's "deciding months": what Nov-Dec of the midterm year did. Constant within a
    cycle and fully known before 1-Jan, so it can condition every week of the window.

    Note these are CYCLE-level features - sixteen observations, not 448 - so they can only
    ever carry a little information, and §7 of the notebook tests whether they carry any."""
    rows = []
    for cyc in cycles:
        d = px[(px["Datetime"] >= pd.Timestamp(cyc - 1, 11, 1)) &
               (px["Datetime"] <= pd.Timestamp(cyc - 1, 12, 31))]
        if len(d) < 20:
            rows.append({"cycle": cyc, "nd_ret": np.nan, "nd_rv": np.nan, "nd_dd": np.nan})
            continue
        r = np.log(d["Close"]).diff().dropna()
        rows.append({"cycle": cyc,
                     "nd_ret": (d["Close"].iloc[-1] / d["Open"].iloc[0] - 1) * 100,
                     "nd_rv": float(r.std(ddof=1) * math.sqrt(252) * 100),
                     "nd_dd": (d["Low"].min() / d["Open"].iloc[0] - 1) * 100})
    return pd.DataFrame(rows)


def postmidterm_panel(px: pd.DataFrame, last_cycle: int = 2023) -> pd.DataFrame:
    """Every post-midterm Jan->Jul week the SPX tape can reach, with its labels and features.
    This is the training universe: post-midterm weeks and nothing else."""
    sessions = pd.DatetimeIndex(sorted(px["Datetime"].unique()))
    wins = [w for w in cycle_windows(last_cycle=last_cycle) if w["start"] >= sessions.min()]
    panel = excursions(px, build_weeks(sessions, wins))
    rv = realized_vol_table(px).sort_values("Datetime")
    panel = pd.merge_asof(panel.sort_values("entry_day"),
                          rv.rename(columns={"Datetime": "feat_day"}),
                          left_on="entry_day", right_on="feat_day",
                          direction="backward", allow_exact_matches=False)
    # the month dummy uses the week's SCHEDULED last session, known at entry
    panel["month"] = pd.to_datetime(panel["exit_day"]).dt.month
    panel["mname"] = [pd.Timestamp(2021, m, 1).strftime("%b") for m in panel["month"]]
    for m in MONTHS:
        panel[f"m_{m}"] = (panel["mname"] == m).astype(float)
    panel["wk_idx"] = panel.groupby("cycle").cumcount() + 1
    panel = panel.merge(novdec_features(px, sorted(panel["cycle"].unique())),
                        on="cycle", how="left")
    return panel.reset_index(drop=True)


# =============================================================================
# QUANTILE REGRESSION
# =============================================================================

def pinball(y, yhat, tau: float) -> float:
    """The loss the tau-quantile is the unique minimiser of, so the only metric that scores a
    quantile forecast on its own terms."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    e = y - yhat
    return float(np.mean(np.maximum(tau * e, (tau - 1) * e)))


def qr_metrics(y, yhat, tau: float, y_train=None) -> dict:
    """coverage is the share of actuals AT OR BELOW the forecast. For a tau=0.10 DOWNSIDE line a
    calibrated forecast has 10% of weeks below it - 10% of weeks worse than the warning. Above
    0.10 the line is too shallow and the gate under-warns, which is the costly direction.
    mae/rmse score the CENTRE; a tau=0.10 line is meant to sit below the data, so its bias is
    positive by construction and it is never used to choose."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    if len(y) == 0:
        return {}
    bench = np.quantile(np.asarray(y_train, float), tau) if y_train is not None \
        else np.quantile(y, tau)
    pl, pl0 = pinball(y, yhat, tau), pinball(y, np.full_like(y, bench), tau)
    e = y - yhat
    return {"n": int(len(y)), "tau": tau, "pinball": pl, "pinball_benchmark": pl0,
            "pseudo_r2_KM": float(1 - pl / pl0) if pl0 > 0 else np.nan,
            "coverage": float(np.mean(y <= yhat)),
            "coverage_err": float(np.mean(y <= yhat) - tau),
            "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "bias": float(np.mean(e)),
            "corr": float(np.corrcoef(y, yhat)[0, 1]) if len(y) > 2 and np.std(yhat) > 0
            else np.nan}


def fit_qr(train: pd.DataFrame, feats: list[str], target: str, tau: float):
    import statsmodels.api as sm
    d = train[feats + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 10 * max(1, len(feats)) or d[feats].nunique().min() <= 1:
        return None
    try:
        return sm.QuantReg(d[target].astype(float),
                           sm.add_constant(d[feats].astype(float), has_constant="add")
                           ).fit(q=tau, max_iter=8000)
    except Exception:
        return None


def predict_qr(res, frame: pd.DataFrame, feats: list[str], clip: str | None = None):
    import statsmodels.api as sm
    if res is None:
        return np.full(len(frame), np.nan)
    X = sm.add_constant(frame[feats].astype(float), has_constant="add")
    p = np.asarray(res.predict(X.reindex(columns=res.params.index)), float)
    if clip == "neg":                      # a drawdown cannot be positive
        return np.clip(p, None, 0.0)
    if clip == "pos":                      # an excursion cannot be negative
        return np.clip(p, 0.0, None)
    return p


def vif_table(frame: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    import statsmodels.api as sm
    d = frame[feats].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    rows = []
    for f in feats:
        others = [c for c in feats if c != f]
        if not others:
            rows.append({"feature": f, "vif": 1.0}); continue
        r2 = sm.OLS(d[f], sm.add_constant(d[others])).fit().rsquared
        rows.append({"feature": f, "vif": float(1 / (1 - r2)) if r2 < 1 else np.inf})
    return pd.DataFrame(rows)


def quantile_prob(train: pd.DataFrame, test: pd.DataFrame, feats: list[str], target: str,
                  level: float, side: str, taus=TAUS) -> np.ndarray:
    """THE PROBABILITY, STRAIGHT OUT OF THE QUANTILE FUNCTIONS - no logistic.

    Fit the quantile regression at every tau in `taus`, stack each week's predicted quantiles,
    SORT them (quantile regressions fitted independently can cross, and a crossed quantile
    function has no inverse), then invert: the tau at which the predicted quantile passes
    `level` is P(target <= level).

        side 'below' -> returns that tau            e.g. P(dd_w  <= -stop)   = P(stop fires)
        side 'above' -> returns 1 - tau             e.g. P(mfe_w >= +target) = P(target hit)

    Clamped to the ends of the grid, so a week whose whole predicted distribution sits above
    the level returns the smallest tau rather than extrapolating a quantile nobody fitted.
    """
    P = []
    for tau in taus:
        res = fit_qr(train, feats, target, tau)
        if res is None:
            return np.full(len(test), np.nan)
        P.append(predict_qr(res, test, feats))
    Q = np.sort(np.vstack(P), axis=0)
    out = np.array([np.interp(level, Q[:, j], taus, left=taus[0], right=taus[-1])
                    for j in range(Q.shape[1])])
    return out if side == "below" else 1.0 - out


def qr_walkforward(panel: pd.DataFrame, feats: list[str], target: str, tau: float,
                   clip: str | None = None):
    """Expanding by CYCLE over post-midterm windows: fit on every earlier cycle, score this one.
    The only honest split when the whole universe is 16 cycles."""
    d = (panel.replace([np.inf, -np.inf], np.nan).dropna(subset=feats + [target])
         .sort_values("entry_day"))
    cycles = sorted(d["cycle"].unique())
    rows, preds = [], []
    for i, cyc in enumerate(cycles):
        if i == 0:
            continue
        tr, te = d[d["cycle"] < cyc], d[d["cycle"] == cyc]
        if len(tr) < 60 or len(te) == 0:
            continue
        res = fit_qr(tr, feats, target, tau)
        if res is None:
            continue
        p = predict_qr(res, te, feats, clip=clip)
        preds.append(pd.DataFrame({"cycle": cyc, "entry_day": te["entry_day"].to_numpy(),
                                   "mname": te["mname"].to_numpy(),
                                   "y": te[target].to_numpy(), "pred": p}))
        rows.append({"cycle": cyc, "n_train": len(tr),
                     **qr_metrics(te[target], p, tau, y_train=tr[target])})
    return pd.DataFrame(rows), (pd.concat(preds, ignore_index=True) if preds
                                else pd.DataFrame(columns=["cycle", "y", "pred"]))


# =============================================================================
# THE GATE
# =============================================================================

def postmidterm_gate(panel: pd.DataFrame, feats: list[str], test_cycles: list[int],
                     skip_fraction: float = 0.20, taus=TAUS) -> pd.DataFrame:
    """Expanding by cycle: for each test cycle, fit on every EARLIER post-midterm cycle, then
    score that cycle's weeks.

    The skip threshold is the `skip_fraction` quantile of the edge the model assigns to its own
    TRAINING weeks, so it is set without seeing the test cycle. `skip_fraction` is the dial the
    desk sets - how much of the book it is willing to stand aside from - not something inferred
    from a classification score. The brief's binding constraint is forgone profit, so this dial
    is the honest control and the sweep over it is the deliverable.
    """
    need = feats + ["dd_w", "mfe_w"]
    out = []
    for cyc in test_cycles:
        tr = panel[panel["cycle"] < cyc].replace([np.inf, -np.inf], np.nan).dropna(subset=need)
        te = panel[panel["cycle"] == cyc].replace([np.inf, -np.inf], np.nan).dropna(subset=need)
        if len(tr) < 60 or len(te) == 0:
            continue
        g = te.copy()
        g["p_stop"] = quantile_prob(tr, te, feats, "dd_w", -STOP_PCT, "below", taus)
        g["p_target"] = quantile_prob(tr, te, feats, "mfe_w", TARGET_PCT, "above", taus)
        g["edge"] = g["p_target"] - g["p_stop"]
        # the cut comes from the model's view of its own TRAINING weeks. The whole quantile
        # vector is carried, not one number, so sweeping the dial later needs no refit.
        tr_edge = (quantile_prob(tr, tr, feats, "mfe_w", TARGET_PCT, "above", taus)
                   - quantile_prob(tr, tr, feats, "dd_w", -STOP_PCT, "below", taus))
        g["edge_train_q"] = [np.nanquantile(tr_edge, Q_GRID)] * len(g)
        g["n_train"] = len(tr)
        g["train_cycles"] = f"{int(min(tr.cycle))}-{int(max(tr.cycle))}"
        out.append(g)
    if not out:
        return pd.DataFrame()
    return apply_edge_cut(pd.concat(out, ignore_index=True), skip_fraction)


Q_GRID = np.round(np.arange(0.02, 1.00, 0.02), 2)


def apply_edge_cut(dec: pd.DataFrame, skip_fraction: float) -> pd.DataFrame:
    """Turn the stored training-edge quantiles into this run's skip decision. Separated from the
    fitting so the dial can be swept for free - the edge itself never depends on it."""
    d = dec.copy()
    d["edge_cut"] = [float(np.interp(skip_fraction, Q_GRID, np.asarray(a, float)))
                     if a is not None else np.nan for a in d["edge_train_q"]]
    d["size"] = (d["edge"] > d["edge_cut"]).astype(float)
    d["skip_fraction"] = skip_fraction
    return d


# =============================================================================
# BOOK METRICS, ACCOUNTING, PIVOTS
# =============================================================================

def book_metrics(pnl: pd.Series, label: str, years: float, sizes=None) -> dict:
    """Drawdown and Sharpe are on the FULL weekly series with a skipped week contributing 0,
    because sitting out is a position and a flat week is a real week of the record. Win rate,
    payoff and profit factor are on the weeks actually TRADED, because they are statements about
    the trades. max_drawdown is on cumulative dollars, not a compounded curve: the position is
    one contract a week, not a re-invested fraction."""
    p = pd.Series(pnl, dtype=float).fillna(0.0)
    sz = (pd.Series(sizes, dtype=float).reindex(p.index).fillna(0.0) if sizes is not None
          else pd.Series(1.0, index=p.index))
    taken = p[sz > 0]
    n = int((sz > 0).sum())
    wins, losses = taken[taken > 0], taken[taken < 0]
    gl = float(-losses.sum())
    eq = p.cumsum()
    sd = float(p.std(ddof=1)) if len(p) > 1 else np.nan
    sh = float(p.mean() / sd) if sd and sd > 0 else np.nan
    wpy = len(p) / years if years > 0 else np.nan
    avg_w = float(wins.mean()) if len(wins) else 0.0
    avg_l = float(-losses.mean()) if len(losses) else 0.0
    avg_sz = float(sz.mean()) if len(sz) else np.nan
    return {"book": label, "weeks_in_window": len(p), "trades_taken": n,
            "total_net_pnl": float(p.sum()),
            "expectancy_per_week": float(p.mean()) if len(p) else np.nan,
            "expectancy_per_trade": float(p.sum() / n) if n else np.nan,
            "pnl_per_unit_exposure": float(p.sum() / avg_sz) if avg_sz else np.nan,
            "win_rate_of_trades": len(wins) / n if n else np.nan,
            "avg_win": avg_w, "avg_loss": avg_l,
            "payoff_ratio": avg_w / avg_l if avg_l > 0 else np.nan,
            "profit_factor": float(wins.sum() / gl) if gl > 0 else np.nan,
            "max_drawdown": float((eq - eq.cummax()).min()) if len(eq) else np.nan,
            "std_per_week": sd, "sharpe_per_week": sh,
            "sharpe_annualised": sh * math.sqrt(wpy) if np.isfinite(sh) else np.nan,
            "best_week": float(p.max()) if len(p) else np.nan,
            "worst_week": float(p.min()) if len(p) else np.nan,
            "avg_size": avg_sz, "weeks_per_year": wpy}


def gate_accounting(sheet: pd.DataFrame) -> dict:
    """What the gating cost and what it saved. The two always add back to the baseline exactly.
    `ratio` is the one number the brief's constraint turns on: money saved per pound of profit
    given up. Below 1.0 the gate is paying more in forgone profit than it is saving."""
    f = sheet["pnl_forgone"]
    sac = float(f[f > 0].sum()); avo = float(-f[f < 0].sum())
    sk = sheet[sheet["size"] <= 0]
    return {"weeks_skipped": int(len(sk)),
            "weeks_reduced": int(((sheet["size"] > 0) & (sheet["size"] < 1)).sum()),
            "weeks_full_size": int((sheet["size"] >= 1).sum()),
            "profits_sacrificed": sac, "losses_avoided": avo,
            "saved_per_unit_sacrificed": (avo / sac) if sac > 0 else np.inf,
            "net_effect_of_gate": float(sheet["pnl_gated"].sum() - sheet["pnl"].fillna(0).sum()),
            "skipped_pnl_baseline": float(sk["pnl"].sum()),
            "skipped_win_rate": float((sk["pnl"] > 0).mean()) if len(sk) else np.nan}


def pivot(sheet: pd.DataFrame, value: str = "pnl", how: str = "sum",
          taken: str | None = None, attribute: str = "exit") -> pd.DataFrame:
    """The brief's pivot: rows = election_year, columns = month Jan->Jul, plus Grand Total and
    the summary rows underneath.

    `attribute` = 'exit' BECAUSE THAT IS WHAT THE BRIEF'S PIVOT USES - attributing by the
    option's exit date reproduces it cell for cell (182, 150, -167, 259, 250, -64 for 2011 and
    so on through all 28 cells); entry-date attribution gives the same cycle totals and a
    different monthly split.

    `taken` names a boolean column restricting the win-rate and worst-week rows to the weeks a
    trade was actually on. Without it a gated book's win rate is nonsense: a skipped week books
    exactly 0, 0 is not > 0, and the skips get counted as losses.
    """
    d = sheet.copy()
    datecol = "exit_date" if attribute == "exit" else "entry_date"
    stamp = pd.to_datetime(d[datecol]).fillna(pd.to_datetime(d["entry_date"]))
    d["month_name"] = [pd.Timestamp(2021, m, 1).strftime("%b") for m in stamp.dt.month]
    order = [pd.Timestamp(2021, m, 1).strftime("%b") for m in MONTH_ORDER]
    order = [m for m in order if m in set(d["month_name"])]

    # Everything is built as float and concatenated. Assigning a float summary row into a frame
    # holding even ONE integer column makes pandas 3 cast the whole row to int, so a 0.769 win
    # rate silently prints as 1.00.
    def _row(s: pd.Series, grand) -> pd.Series:
        r = s.reindex(order).astype(float)
        r["Grand Total"] = float(grand) if pd.notna(grand) else np.nan
        return r

    body = (d.pivot_table(index="election_year", columns="month_name", values=value,
                          aggfunc=how, observed=True).reindex(columns=order).astype(float))
    body["Grand Total"] = d.groupby("election_year")[value].agg(how).astype(float)
    t = d[d[taken].astype(bool)] if taken else d[d[value].notna()]
    extra = pd.DataFrame({
        "Grand Total": _row(d.groupby("month_name")[value].agg(how), d[value].agg(how)),
        "win_rate": _row(t.groupby("month_name")[value].agg(lambda s: (s > 0).mean()),
                         (t[value] > 0).mean() if len(t) else np.nan),
        "worst_week": _row(t.groupby("month_name")[value].min(),
                           t[value].min() if len(t) else np.nan),
        "trades_taken": _row(t.groupby("month_name")[value].size(), len(t)),
    }).T.reindex(columns=list(body.columns))
    return pd.concat([body, extra])


# =============================================================================
# LOOK-BACK FEATURES
# =============================================================================
# The brief's point: "why aren't we look-backing - maybe we look at April and then decide what to
# do in May". Everything above conditions on the LEVEL of volatility and the calendar. Nothing
# conditions on what the market just DID. These are the missing family.
#
# All of them are built on a CONTINUOUS weekly series spanning the whole tape, not on the
# post-midterm weeks alone, then joined onto the traded weeks. That matters at the window edge:
# the first week of January must take its lag from late December, not from the previous July.
#
# Every feature is shifted so a week sees only weeks that CLOSED before it. The shift is applied
# before any rolling window, so a 4-week mean never contains the week it is predicting.

LOOKBACK_SETS = {
    "lag1": ["l1_dd", "l1_mfe", "l1_ret", "l1_loss"],
    "prior_month": ["r4_stop_rate", "r4_mean_ret", "r4_worst_dd", "r4_target_rate"],
    "path": ["dist_high20", "mom_4w", "mom_12w"],
    "breach": ["l1_breach", "r4_breach_rate"],
}


def all_weeks_panel(px: pd.DataFrame) -> pd.DataFrame:
    """Every Monday-week on the tape, with the excursion labels. The continuous spine the
    look-back features are computed on."""
    sessions = pd.DatetimeIndex(sorted(px["Datetime"].unique()))
    first = sessions.min() - pd.Timedelta(days=int(sessions.min().weekday()))
    rows = []
    for monday in pd.date_range(first, sessions.max(), freq="W-MON"):
        inweek = sessions[(sessions >= monday) & (sessions <= monday + pd.Timedelta(days=4))]
        if len(inweek) < 2:
            continue
        rows.append({"monday": monday, "entry_day": inweek[0], "exit_day": inweek[-1]})
    return excursions(px, pd.DataFrame(rows))


def lookback_features(px: pd.DataFrame) -> pd.DataFrame:
    """The look-back block, keyed on `monday` so it can be joined onto any week frame.

    l1_*            what last week actually did - its drawdown, its favourable excursion, its
                    return, and whether the stop would have fired
    r4_*            the previous FOUR closed weeks, i.e. "look at April to decide May": the rate
                    at which the stop fired, the mean return, the worst drawdown, the rate at
                    which the target was reached
    dist_high20     how far below its 20-session high the index closed - the drawdown STATE,
                    which is not the same thing as its volatility
    mom_4w/12w      trailing compounded return
    l1_breach       did last week's realized drawdown come in WORSE than the expanding
                    unconditional 10th percentile of weekly drawdowns? A breach says the recent
                    past was in the tail, which is the quantile-regression-native way of asking
                    "has the regime just shifted?" - and it is the only one of these features
                    that could not be built without the quantile machinery
    r4_breach_rate  the same, averaged over the last four closed weeks
    """
    w = all_weeks_panel(px).sort_values("monday").reset_index(drop=True)
    # a breach is judged against the quantile known BEFORE the week, so the expanding quantile is
    # shifted; min_periods keeps the first few years out of the feature rather than guessing
    q10 = w["dd_w"].shift(1).expanding(min_periods=104).quantile(0.10)
    w["breach"] = (w["dd_w"] < q10).astype(float)
    w.loc[q10.isna(), "breach"] = np.nan

    out = pd.DataFrame({"monday": w["monday"]})
    out["l1_dd"] = w["dd_w"].shift(1)
    out["l1_mfe"] = w["mfe_w"].shift(1)
    out["l1_ret"] = w["ret_w"].shift(1)
    out["l1_loss"] = w["loss_w"].shift(1).astype(float)
    out["l1_target"] = w["target_hit_w"].shift(1).astype(float)
    out["l2_loss"] = w["loss_w"].shift(2).astype(float)
    s = w["loss_w"].shift(1)
    out["loss_streak"] = s.groupby((s != s.shift()).cumsum()).cumcount().add(1).where(s == 1, 0)
    out["r4_stop_rate"] = w["loss_w"].shift(1).rolling(4).mean()
    out["r4_mean_ret"] = w["ret_w"].shift(1).rolling(4).mean()
    out["r4_worst_dd"] = w["dd_w"].shift(1).rolling(4).min()
    out["r4_target_rate"] = w["target_hit_w"].shift(1).rolling(4).mean()
    out["r8_stop_rate"] = w["loss_w"].shift(1).rolling(8).mean()
    out["r8_mean_ret"] = w["ret_w"].shift(1).rolling(8).mean()
    out["mom_4w"] = w["ret_w"].shift(1).rolling(4).sum()
    out["mom_12w"] = w["ret_w"].shift(1).rolling(12).sum()
    out["l1_breach"] = w["breach"].shift(1)
    out["r4_breach_rate"] = w["breach"].shift(1).rolling(4).mean()

    # the price-state features come off the daily tape at the last session before entry
    d = px[["Datetime", "Close", "High"]].copy()
    d["hi20"] = d["High"].rolling(20).max()
    d["dist_high20"] = (d["Close"] / d["hi20"] - 1.0) * 100.0
    d["ma50"] = d["Close"].rolling(50).mean()
    d["above_ma50"] = (d["Close"] > d["ma50"]).astype(float)
    ent = w[["monday", "entry_day"]]
    j = pd.merge_asof(ent.sort_values("entry_day"),
                      d[["Datetime", "dist_high20", "above_ma50"]].sort_values("Datetime"),
                      left_on="entry_day", right_on="Datetime",
                      direction="backward", allow_exact_matches=False)
    return out.merge(j[["monday", "dist_high20", "above_ma50"]], on="monday", how="left")


# =============================================================================
# IMPLIED VOL, INVERTED FROM THE PREMIUM WE ACTUALLY RECEIVED
# =============================================================================
# There is no VIX file in this project and the option cache carries prices only, so the implied
# vol has to come out of the ATM put premium at entry. That is the better reading anyway: it is
# not a general market gauge, it is literally what this trade was PAID, on the strike and expiry
# it traded, known the moment the fill happened.
#
# WHY THIS IS THE MISSING PIECE
# Every signal in this project measures the SPOT geometry - how far the index might fall, how
# likely it is to reach the target. The book's P&L is premium MINUS spot geometry. That is why a
# signal with a good target-hit AUC still flagged February and June, the two months where the
# premium more than paid for the risk. IV closes the loop: it says whether we are being paid
# enough for the risk the quantile regression just forecast.

def bs_put_iv(premium: float, spot: float, strike: float, t_years: float,
              r: float = 0.0, lo: float = 0.01, hi: float = 5.0) -> float:
    """Black-Scholes implied vol of a European put, by bisection.

    r = 0 and no dividend: over a four-day hold the carry terms move the answer by far less
    than the bid-ask on a weekly option, and using a constant keeps the series internally
    consistent across fifteen years of rate regimes."""
    from math import log, sqrt, exp
    from statistics import NormalDist
    if not all(np.isfinite([premium, spot, strike, t_years])) or t_years <= 0 or premium <= 0:
        return np.nan
    N = NormalDist().cdf
    intrinsic = max(0.0, strike * exp(-r * t_years) - spot)
    if premium <= intrinsic + 1e-9:          # no time value left, IV is unidentified
        return np.nan

    def price(sig):
        d1 = (log(spot / strike) + (r + 0.5 * sig * sig) * t_years) / (sig * sqrt(t_years))
        d2 = d1 - sig * sqrt(t_years)
        return strike * exp(-r * t_years) * N(-d2) - spot * N(-d1)

    if price(hi) < premium:
        return np.nan
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if price(mid) < premium:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def attach_iv(sheet: pd.DataFrame) -> pd.DataFrame:
    """Per traded week: the ATM put's implied vol at entry, the premium as a share of spot, and
    the IV-minus-realized spread. All three are known at the fill.

    Time to expiry runs from the entry stamp to 16:00 on the week's SCHEDULED last session -
    `exit_day`, not `exit_date`, because the expiry is fixed when the trade is put on while the
    actual exit may come early on a stop."""
    d = sheet.copy()
    ent = pd.to_datetime(d["entry_date"])
    exp_dt = pd.to_datetime(d["exit_day"]).dt.normalize() + pd.Timedelta(hours=16)
    d["t_years"] = (exp_dt - ent).dt.total_seconds() / (365.25 * 24 * 3600)
    d["iv_atm"] = [bs_put_iv(p, s, k, t) * 100 if np.isfinite(t) else np.nan
                   for p, s, k, t in zip(d["entry_price"], d["entry_spot"],
                                         d["entry_strike"], d["t_years"])]
    d["pay_pct"] = d["entry_price"] / d["entry_spot"] * 100.0      # premium, % of spot
    d["iv_minus_rv20"] = d["iv_atm"] - d["rv20"]
    return d


def ev_proxy(sheet: pd.DataFrame, loss_pct_col: str = "loss_pct_est") -> pd.Series:
    """Expected value per unit of spot, in the trade's own currency:

        EV = (1 - p_stop) * premium_received  -  p_stop * expected_loss_if_stopped

    `p_stop` is the quantile regression's probability that the stop fires, `pay_pct` is the IV
    reading, and the loss term is estimated from EARLIER CYCLES only (see `causal_loss_est`).

    EV <= 0 is a decision rule with nothing fitted in it: it says "the premium does not cover
    the forecast risk". That matters because every other threshold in this project had a dial
    on it, and a dial is what let the last pass cherry-pick."""
    return (1.0 - sheet["p_stop"]) * sheet["pay_pct"] - sheet["p_stop"] * sheet[loss_pct_col]


def causal_loss_est(sheet: pd.DataFrame) -> pd.Series:
    """Average loss-if-stopped, as a % of spot, estimated from PRIOR cycles only.

    The first cycle has no prior, so it falls back to the stop distance itself (1.1306%) - the
    move that defines the exit, and a conservative stand-in given a short put loses roughly its
    delta times that move."""
    d = sheet.sort_values("entry_date")
    stopped = d["exit_reason"].astype(str).str.startswith("STOP")
    lp = (-(d["pnl"] / (MULTIPLIER * QUANTITY)) / d["entry_spot"] * 100.0).where(stopped)
    out = pd.Series(np.nan, index=d.index)
    for cyc in sorted(d["cycle"].unique()):
        prior = lp[d["cycle"] < cyc].dropna()
        out[d["cycle"] == cyc] = float(prior.mean()) if len(prior) else STOP_PCT
    return out.reindex(sheet.index)
