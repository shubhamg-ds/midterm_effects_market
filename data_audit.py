from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import numpy as np

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# If this script is inside a subfolder, adjust BASE_DIR accordingly.
DATA_DIR = BASE_DIR / "data"

# If the script is already inside the data folder, use:
# DATA_DIR = BASE_DIR

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_LOG = []


def log(message):
    print(message)
    AUDIT_LOG.append(str(message))


# ============================================================
# 2. FILE LOCATIONS
# ============================================================

FILES = {
    "SPX": DATA_DIR / "SPX_1D.csv",
    "DJI": DATA_DIR / "DJI_1D.csv",
    "NASDAQ": DATA_DIR / "NASDAQ_DLY_IXIC_1D.csv",
    "VIX": DATA_DIR / "CBOE_DLY_VIX_1D.csv",

    "DFF": DATA_DIR / "DFF.csv",
    "DGS10": DATA_DIR / "DGS10.csv",
    "DGS2": DATA_DIR / "DGS2.csv",

    "CPI": DATA_DIR / "CPIAUCSL.csv",
    "UNRATE": DATA_DIR / "UNRATE.csv",
    "USREC": DATA_DIR / "USREC.csv",

    "SWING": DATA_DIR / "us_swing_states_all_elections_1962_2024.csv",
    #"WARS": DATA_DIR / "us_presidents_wars_conflicts_full_1961_2024.csv",
    "PRESIDENTIAL": DATA_DIR / "us_presidential_elections_by_state_1964_2024.csv",

    "QQQ_MIN": DATA_DIR / "QQQ_1min_adj.csv",
    "SPY_MIN": DATA_DIR / "SPY_1min_adj.csv",
}


# ============================================================
# 3. GENERIC DATE PARSER
# ============================================================

def parse_dates(series):
    """
    Parse common date formats, including numeric Unix timestamps.
    """

    if pd.api.types.is_numeric_dtype(series):
        valid = series.dropna()

        if valid.empty:
            return pd.to_datetime(series, errors="coerce")

        magnitude = valid.abs().median()

        if magnitude >= 1e12:
            return pd.to_datetime(
                series, unit="ms", errors="coerce"
            )

        if magnitude >= 1e9:
            return pd.to_datetime(
                series, unit="s", errors="coerce"
            )

    return pd.to_datetime(series, errors="coerce")


def normalise_columns(df):
    df = df.copy()
    df.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in df.columns
    ]
    return df


def find_date_column(df):
    candidates = [
        "dt",
        "datetime",
        "date",
        "time",
        "observation_date",
        "timestamp",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"No recognised date column. Found: {df.columns.tolist()}"
    )


# ============================================================
# 4. AUDIT NEW DAILY MARKET DATA
# ============================================================

def audit_market_file(path, name):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, low_memory=False)
    df = normalise_columns(df)

    date_col = find_date_column(df)
    df[date_col] = parse_dates(df[date_col])

    # Common market column names
    rename_map = {
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "vol": "volume",
    }

    df = df.rename(columns=rename_map)

    required = ["open", "high", "low", "close"]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"{name}: missing columns {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(
            df["volume"], errors="coerce"
        )

    # Standardise date
    df["date"] = df[date_col].dt.normalize()

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    invalid_ohlc = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )

    audit = {
        "dataset": name,
        "rows": len(df),
        "start_date": df["date"].min(),
        "end_date": df["date"].max(),
        "duplicate_dates": int(df["date"].duplicated().sum()),
        "missing_dates": int(df["date"].isna().sum()),
        "missing_close": int(df["close"].isna().sum()),
        "nonpositive_close": int((df["close"] <= 0).sum()),
        "invalid_ohlc": int(invalid_ohlc.sum()),
    }

    log(f"\n{name} audit:")
    for key, value in audit.items():
        log(f"  {key}: {value}")

    return df, audit


# ============================================================
# 5. LOAD AND AUDIT DAILY MACROECONOMIC SERIES
# ============================================================

def load_fred_series(path, name):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, low_memory=False)
    df = normalise_columns(df)

    date_col = find_date_column(df)

    # Find the value column.
    possible_values = [
        col for col in df.columns
        if col != date_col
        and col not in {"realtime_start", "realtime_end"}
    ]

    if len(possible_values) == 0:
        raise ValueError(
            f"{name}: no value column found in {df.columns.tolist()}"
        )

    # Prefer a column matching the series identifier.
    preferred = {
        "DFF": "dff",
        "DGS10": "dgs10",
        "DGS2": "dgs2",
        "CPI": "cpiaucsl",
        "UNRATE": "unrate",
        "USREC": "usrec",
    }.get(name)

    if preferred in possible_values:
        value_col = preferred
    elif len(possible_values) == 1:
        value_col = possible_values[0]
    else:
        raise ValueError(
            f"{name}: ambiguous value columns {possible_values}"
        )

    df["date"] = parse_dates(df[date_col]).dt.normalize()

    df[name] = pd.to_numeric(
        df[value_col].replace(".", np.nan),
        errors="coerce"
    )

    df = df[["date", name]].dropna(subset=["date"])

    # Duplicate dates need investigation rather than silent aggregation.
    duplicate_dates = int(df["date"].duplicated().sum())

    log(f"\n{name} audit:")
    log(f"  rows: {len(df)}")
    log(f"  start: {df['date'].min()}")
    log(f"  end: {df['date'].max()}")
    log(f"  missing values: {df[name].isna().sum()}")
    log(f"  duplicate dates: {duplicate_dates}")

    if duplicate_dates:
        raise ValueError(
            f"{name}: duplicate dates found. Inspect the source file."
        )

    return df.sort_values("date")


# ============================================================
# 6. RESAMPLE QQQ AND SPY FROM MINUTE TO DAILY
# ============================================================

def minute_to_daily(path, name):
    """
    Resample the existing one-minute ETF data.
    This is a transformation, not a full re-audit of the raw files.
    """

    if not path.exists():
        raise FileNotFoundError(path)

    # Read only the columns needed for resampling.
    df = pd.read_csv(path, low_memory=False)
    df = normalise_columns(df)

    required = ["dt", "open", "high", "low", "close", "volume"]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"{name}: missing columns {missing}"
        )

    df = df[required].copy()

    df["dt"] = parse_dates(df["dt"])

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["dt", "close"])
    df = df.sort_values("dt")

    # Calendar date in the timestamp's current timezone.
    df["date"] = df["dt"].dt.normalize()

    daily = (
        df.groupby("date", sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            n_minutes=("close", "count"),
        )
        .reset_index()
    )

    daily["asset"] = name

    log(f"\n{name} resampling:")
    log(f"  minute rows: {len(df)}")
    log(f"  daily rows: {len(daily)}")
    log(f"  start: {daily['date'].min()}")
    log(f"  end: {daily['date'].max()}")

    return daily


# ============================================================
# 7. CREATE MARKET FEATURES
# ============================================================

def add_market_features(df, prefix):
    df = df.sort_values("date").copy()

    df[f"{prefix}_log_return"] = np.log(
        df["close"] / df["close"].shift(1)
    )

    df[f"{prefix}_simple_return"] = (
        df["close"].pct_change()
    )

    r = df[f"{prefix}_log_return"]

    # Realised volatility proxies: square root of the
    # sum of squared daily log returns over each window.
    for window in [5, 21, 63]:
        df[f"{prefix}_rv_{window}d"] = np.sqrt(
            r.pow(2).rolling(
                window=window,
                min_periods=window
            ).sum()
        )

    # Annualised rolling standard deviation of daily returns.
    df[f"{prefix}_volatility_21d_ann"] = (
        r.rolling(21, min_periods=21).std()
        * np.sqrt(252)
    )

    # Daily intraday range, relative to the opening price.
    df[f"{prefix}_intraday_range"] = (
        (df["high"] - df["low"]) / df["open"]
    )

    return df


# ============================================================
# 8. LOAD DAILY MARKET SERIES
# ============================================================

market_frames = []
audit_results = []

for name in ["SPX", "DJI", "NASDAQ"]:
    df, audit = audit_market_file(FILES[name], name)

    audit_results.append(audit)

    keep = ["date", "open", "high", "low", "close"]

    if "volume" in df.columns:
        keep.append("volume")

    df = df[keep].copy()

    rename = {
        col: f"{name.lower()}_{col}"
        for col in keep
        if col != "date"
    }

    df = df.rename(columns=rename)

    market_frames.append(df)


# ============================================================
# 9. ADD QQQ AND SPY DAILY BARS
# ============================================================

qqq_daily = minute_to_daily(FILES["QQQ_MIN"], "QQQ")
spy_daily = minute_to_daily(FILES["SPY_MIN"], "SPY")

for name, df in [
    ("qqq", qqq_daily),
    ("spy", spy_daily),
]:
    df = df.rename(
        columns={
            col: f"{name}_{col}"
            for col in [
                "open", "high", "low", "close",
                "volume", "n_minutes"
            ]
        }
    )

    market_frames.append(
        df[[
            "date",
            f"{name}_open",
            f"{name}_high",
            f"{name}_low",
            f"{name}_close",
            f"{name}_volume",
            f"{name}_n_minutes",
        ]]
    )


# ============================================================
# 10. MERGE MARKET SERIES
# ============================================================

market_daily = market_frames[0]

for frame in market_frames[1:]:
    market_daily = market_daily.merge(
        frame,
        on="date",
        how="outer",
        validate="one_to_one"
    )

market_daily = (
    market_daily
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# 11. ADD VIX
# ============================================================

vix, vix_audit = audit_market_file(FILES["VIX"], "VIX")
audit_results.append(vix_audit)

vix_keep = ["date", "close"]
vix = vix[vix_keep].rename(
    columns={"close": "vix_close"}
)

market_daily = market_daily.merge(
    vix,
    on="date",
    how="outer",
    validate="one_to_one"
)

market_daily = (
    market_daily
    .sort_values("date")
    .reset_index(drop=True)
)

market_daily["vix_log_change"] = np.log(
    market_daily["vix_close"]
    / market_daily["vix_close"].shift(1)
)

# Calculate returns and volatility separately for each asset.
for asset in ["spx", "dji", "nasdaq", "qqq", "spy"]:
    close_col = f"{asset}_close"

    if close_col not in market_daily.columns:
        continue

    temp = market_daily[["date", close_col]].copy()

    temp = temp.rename(columns={close_col: "close"})
    temp["open"] = market_daily.get(f"{asset}_open")
    temp["high"] = market_daily.get(f"{asset}_high")
    temp["low"] = market_daily.get(f"{asset}_low")

    features = add_market_features(temp, asset)

    feature_cols = [
        col for col in features.columns
        if col.startswith(f"{asset}_")
        and col not in {
            f"{asset}_open",
            f"{asset}_high",
            f"{asset}_low",
            f"{asset}_close",
        }
    ]

    for col in feature_cols:
        market_daily[col] = features[col].values


# ============================================================
# 12. LOAD MACROECONOMIC SERIES
# ============================================================

macro_series = {}

for name in [
    "DFF", "DGS10", "DGS2",
    "CPI", "UNRATE", "USREC"
]:
    macro_series[name] = load_fred_series(
        FILES[name], name
    )


# Start with the union of the market and macro dates.
macro_daily = pd.DataFrame({
    "date": pd.Index(
        sorted(
            set(market_daily["date"].dropna())
            | set().union(
                *[
                    set(df["date"].dropna())
                    for df in macro_series.values()
                ]
            )
        )
    )
})

macro_daily = macro_daily.sort_values("date")

# Merge daily macroeconomic series.
for name in ["DFF", "DGS10", "DGS2", "USREC"]:
    macro_daily = macro_daily.merge(
        macro_series[name],
        on="date",
        how="left",
        validate="one_to_one"
    )

# CPI and UNRATE remain at their native monthly observation dates.
# Do not forward-fill them yet: the observation date is not
# necessarily the date the information became publicly available.
for name in ["CPI", "UNRATE"]:
    macro_daily = macro_daily.merge(
        macro_series[name],
        on="date",
        how="left",
        validate="one_to_one"
    )


# ============================================================
# 13. CREATE MIDTERM ELECTION CALENDAR
# ============================================================

def midterm_election_date(year):
    """
    U.S. federal general election day:
    Tuesday after the first Monday in November.
    """

    nov_2 = date(year, 11, 2)

    days_until_tuesday = (
        1 - nov_2.weekday()
    ) % 7

    return nov_2 + timedelta(days=days_until_tuesday)


# Midterm elections from 1962 through 2022.
midterm_years = list(range(1962, 2023, 4))

election_events = pd.DataFrame({
    "election_year": midterm_years,
    "election_date": [
        pd.Timestamp(midterm_election_date(year))
        for year in midterm_years
    ],
})

election_events["event_type"] = "Midterm"
election_events["president_party"] = pd.NA
election_events["house_seat_change_president_party"] = np.nan
election_events["senate_seat_change_president_party"] = np.nan
election_events["house_control_before"] = pd.NA
election_events["house_control_after"] = pd.NA
election_events["senate_control_before"] = pd.NA
election_events["senate_control_after"] = pd.NA

# First available market trading date after election day.
market_dates = pd.DatetimeIndex(
    market_daily["date"].dropna().sort_values().unique()
)

def first_market_session_after(event_date):
    later_dates = market_dates[market_dates > event_date]

    if len(later_dates) == 0:
        return pd.NaT

    return later_dates[0]


election_events["first_market_session_after"] = (
    election_events["election_date"].apply(
        first_market_session_after
    )
)


# ============================================================
# 14. AUDIT EXISTING POLITICAL DATASETS
# ============================================================

political_frames = {}

for name in ["SWING", "PRESIDENTIAL"]:
    path = FILES[name]

    df = pd.read_csv(path, low_memory=False)
    df = normalise_columns(df)

    political_frames[name] = df

    log(f"\n{name} political-data audit:")
    log(f"  rows: {len(df)}")
    log(f"  columns: {df.columns.tolist()}")
    log(f"  duplicate full rows: {df.duplicated().sum()}")


# ============================================================
# 15. FINAL DATASET VALIDATION
# ============================================================

def check_unique_dates(df, name):
    duplicates = df["date"].duplicated().sum()

    if duplicates:
        raise ValueError(
            f"{name}: {duplicates} duplicate dates remain."
        )


check_unique_dates(market_daily, "market_daily")
check_unique_dates(macro_daily, "macro_daily")

log("\nFINAL DATASET SUMMARY")

for name, df in [
    ("market_daily", market_daily),
    ("macro_daily", macro_daily),
    ("election_events", election_events),
]:
    log(f"\n{name}")
    log(f"  rows: {len(df)}")
    log(f"  columns: {len(df.columns)}")

    if "date" in df.columns:
        log(f"  start: {df['date'].min()}")
        log(f"  end: {df['date'].max()}")

    if "election_date" in df.columns:
        log(f"  first election: {df['election_date'].min()}")
        log(f"  last election: {df['election_date'].max()}")




for filename in ["QQQ_1min_adj.csv", "SPY_1min_adj.csv"]:
    df = pd.read_csv(DATA_DIR / filename)

    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

    print(f"\n{filename}")
    print("Rows:", len(df))
    print("Invalid timestamps:", df["dt"].isna().sum())
    print("Minimum timestamp:", df["dt"].min())
    print("Maximum timestamp:", df["dt"].max())
    print("Duplicate timestamps:", df["dt"].duplicated().sum())


# --------------------------------------------------
# 2. Locate missing macroeconomic observations
# --------------------------------------------------

macro_files = {
    "DGS10": "DGS10.csv",
    "DGS2": "DGS2.csv",
    "CPIAUCSL": "CPIAUCSL.csv",
    "UNRATE": "UNRATE.csv",
}

for name, filename in macro_files.items():
    df = pd.read_csv(DATA_DIR / filename)

    print(f"\n{name}")
    print("Columns:", df.columns.tolist())

    # Find the date and value columns.
    normalized = {
        col: col.strip().lower().replace(" ", "_")
        for col in df.columns
    }

    date_col = next(
        (
            col for col, norm in normalized.items()
            if norm in [
                "observation_date", "date", "datetime",
                "dt", "time", "timestamp"
            ]
        ),
        None,
    )

    value_col = next(
        (
            col for col, norm in normalized.items()
            if norm == name.lower()
        ),
        None,
    )

    if date_col is None or value_col is None:
        print("Could not automatically identify date/value columns.")
        continue

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    missing = df.loc[
        df[value_col].isna(),
        [date_col, value_col]
    ]

    print("Missing values:", len(missing))
    print(missing.to_string(index=False))



def add_asset_features(data, asset):
    """
    Calculate features on one asset's own daily observations.
    The input must contain date and close columns.
    """
    df = data[["date", "close"]].copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df = df.rename(columns={"close": f"{asset}_close"})

    close_col = f"{asset}_close"
    returns_col = f"{asset}_log_return"

    # One-session log return
    df[returns_col] = np.log(
        df[close_col] / df[close_col].shift(1)
    )

    # Realized-volatility proxies over the previous 5, 21 and 63 returns.
    # These use daily squared log returns, not intraday returns.
    for window in [5, 21, 63]:
        df[f"{asset}_rv_{window}"] = np.sqrt(
            df[returns_col]
            .pow(2)
            .rolling(window, min_periods=window)
            .sum()
        )

    # Annualized rolling standard deviation of daily returns.
    df[f"{asset}_vol_21_annualized"] = (
        df[returns_col].rolling(21, min_periods=21).std()
        * np.sqrt(252)
    )

    return df











# ============================================================
# 16. SAVE PROCESSED DATASETS
# ============================================================

market_daily.to_csv(
    PROCESSED_DIR / "market_daily.csv",
    index=False
)

macro_daily.to_csv(
    PROCESSED_DIR / "macro_daily.csv",
    index=False
)

election_events.to_csv(
    PROCESSED_DIR / "election_events.csv",
    index=False
)

pd.DataFrame(audit_results).to_csv(
    PROCESSED_DIR / "market_audit_report.csv",
    index=False
)

with open(
    PROCESSED_DIR / "data_audit_log.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write("\n".join(AUDIT_LOG))


log("\nAll three datasets have been saved.")
log(f"Output directory: {PROCESSED_DIR}")
