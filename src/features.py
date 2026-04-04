"""
features.py
-----------
Transforms raw game logs into a model-ready feature matrix.

Design principles:
  - Zero data leakage: every feature is computed from games BEFORE the target game.
  - All rolling windows are computed per-team in strict date order.
  - The final row has one entry per game with home/away mirrored features
    and a binary target (home_win).

Usage:
    from features import build_feature_matrix

    df = build_feature_matrix(
        game_logs_path="data/raw/game_logs_all.csv",
        window=10,           # rolling window size (games)
        min_games=5,         # drop rows where team has fewer than this many prior games
        exclude_covid=True,  # drop the 2020 season
    )

    # df is ready to pass to a PyTorch Dataset.
    # Feature columns: df.drop(columns=["game_date","home_team","away_team",
    #                                    "season","home_win"])
    # Target column:   df["home_win"]

Output columns (prefix h_ = home team, a_ = away team):
    Rolling offensive stats (last N games):
        {h,a}_run_rate          – runs scored per game
        {h,a}_ops               – on-base + slugging proxy (run-rate / opp run-rate)
        {h,a}_scoring_var       – std dev of runs scored (consistency)
        {h,a}_win_rate          – rolling win %
        {h,a}_run_diff_rate     – avg run differential per game

    Rolling defensive/pitching stats:
        {h,a}_ra_rate           – runs allowed per game
        {h,a}_quality_start_rt  – fraction of games allowing ≤3 runs

    Streak & momentum:
        {h,a}_streak            – current win (+) or loss (-) streak
        {h,a}_last3_win_rate    – win rate over last 3 games

    Context:
        home_advantage          – always 1 (acts as a learnable bias anchor)
        is_covid_season         – flag for shortened / no-fans season
        day_of_week_{0..6}      – one-hot day of week
        month_{4..10}           – one-hot month (April=4 … October=10)

    Target:
        home_win                – 1 if home team won, 0 otherwise
"""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RAW_DIR      = Path(__file__).parent.parent / "data" / "raw"
FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Per-team rolling stats ─────────────────────────────────────────────────────

def _build_team_timeseries(games: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape the game log (one row per game, home/away columns) into a
    long-format table with one row per team per game appearance, preserving
    strict date order.  This is the input to all rolling calculations.

    Returned columns:
        game_date, team, opponent,
        runs_scored, runs_allowed,
        won,           # 1 if this team won the game
        is_home        # 1 if this team was the home team
    """
    home = games[["game_date", "home_team", "away_team",
                  "home_score", "away_score", "home_win", "season"]].copy()
    home = home.rename(columns={
        "home_team":  "team",
        "away_team":  "opponent",
        "home_score": "runs_scored",
        "away_score": "runs_allowed",
        "home_win":   "won",
    })
    home["is_home"] = 1

    away = games[["game_date", "away_team", "home_team",
                  "away_score", "home_score", "home_win", "season"]].copy()
    away["won"] = (away["home_win"] == 0).astype(int)
    away = away.drop(columns=["home_win"])
    away = away.rename(columns={
        "away_team":  "team",
        "home_team":  "opponent",
        "away_score": "runs_scored",
        "home_score": "runs_allowed",
    })
    away["is_home"] = 0

    ts = pd.concat([home, away], ignore_index=True)
    ts = ts.sort_values(["team", "game_date", "is_home"]).reset_index(drop=True)
    return ts


def _rolling_team_stats(ts: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    For each team, compute rolling stats over the previous `window` games.
    All shifts are by 1 so the current game is never included (no leakage).

    Returns a DataFrame indexed by (team, game_date) with feature columns.
    We use .shift(1) before .rolling(window) throughout.
    """
    results = []

    for team, grp in ts.groupby("team", sort=False):
        grp = grp.sort_values("game_date").copy()

        rs  = grp["runs_scored"]
        ra  = grp["runs_allowed"]
        won = grp["won"]
        rd  = rs - ra   # run differential per game

        # Shift everything by 1 game so no leakage into the current game
        rs_s  = rs.shift(1)
        ra_s  = ra.shift(1)
        won_s = won.shift(1)
        rd_s  = rd.shift(1)

        roll  = lambda s: s.rolling(window, min_periods=1)
        roll3 = lambda s: s.rolling(3,      min_periods=1)

        grp["run_rate"]         = roll(rs_s).mean()
        grp["ra_rate"]          = roll(ra_s).mean()
        grp["run_diff_rate"]    = roll(rd_s).mean()
        grp["win_rate"]         = roll(won_s).mean()
        grp["scoring_var"]      = roll(rs_s).std().fillna(0)
        grp["last3_win_rate"]   = roll3(won_s).mean()

        # Quality start proxy: games where runs allowed ≤ 3
        qs_s = (ra_s <= 3).astype(float)
        grp["quality_start_rt"] = roll(qs_s).mean()

        # OPS proxy: run_rate / (run_rate + ra_rate), capped to avoid div-by-zero
        denom = grp["run_rate"] + grp["ra_rate"]
        grp["ops"]              = grp["run_rate"] / denom.replace(0, np.nan)
        grp["ops"]              = grp["ops"].fillna(0.5)

        # Streak: positive = win streak, negative = loss streak
        grp["streak"] = _compute_streak(won_s)

        results.append(grp)

    out = pd.concat(results, ignore_index=True)
    feature_cols = [
        "game_date", "team", "opponent", "is_home", "season",
        "run_rate", "ra_rate", "run_diff_rate", "win_rate",
        "scoring_var", "last3_win_rate", "quality_start_rt",
        "ops", "streak",
    ]
    return out[feature_cols]


def _compute_streak(won_shifted: pd.Series) -> pd.Series:
    """
    Given a shifted win series, return the current win/loss streak at each row.
    Positive = win streak, negative = losing streak.
    """
    streak = []
    current = 0
    for w in won_shifted:
        if pd.isna(w):
            streak.append(0)
            continue
        if w == 1:
            current = current + 1 if current > 0 else 1
        else:
            current = current - 1 if current < 0 else -1
        streak.append(current)
    return pd.Series(streak, index=won_shifted.index)


# ── Assemble game-level feature matrix ────────────────────────────────────────

def _merge_team_stats_onto_games(
    games: pd.DataFrame,
    team_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join pre-computed per-team rolling stats back onto the game-level table
    as home (h_) and away (a_) prefixed columns.
    """
    stat_cols = [
        "run_rate", "ra_rate", "run_diff_rate", "win_rate",
        "scoring_var", "last3_win_rate", "quality_start_rt",
        "ops", "streak",
    ]

    # Home side — only rows where the team was the home team
    home_stats = (
        team_stats[team_stats["is_home"] == 1]
        [["game_date", "team"] + stat_cols]
        .rename(columns={"team": "home_team"})
        .rename(columns={c: f"h_{c}" for c in stat_cols})
    )

    # Away side
    away_stats = (
        team_stats[team_stats["is_home"] == 0]
        [["game_date", "team"] + stat_cols]
        .rename(columns={"team": "away_team"})
        .rename(columns={c: f"a_{c}" for c in stat_cols})
    )

    merged = games.merge(home_stats, on=["game_date", "home_team"], how="left")
    merged = merged.merge(away_stats, on=["game_date", "away_team"], how="left")
    return merged


def _add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add date-derived and context features."""
    df = df.copy()

    df["home_advantage"]  = 1                                  # learnable constant
    df["is_covid_season"] = df["is_covid_season"].astype(int)

    # Day-of-week one-hot (Mon=0 … Sun=6)
    dow = df["game_date"].dt.dayofweek
    for d in range(7):
        df[f"dow_{d}"] = (dow == d).astype(int)

    # Month one-hot (April=4 … October=10)
    month = df["game_date"].dt.month
    for m in range(4, 11):
        df[f"month_{m}"] = (month == m).astype(int)

    return df


def _add_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add home-minus-away differential columns.
    These give the model explicit relative signals without
    requiring it to learn the subtraction itself.
    """
    df = df.copy()
    diff_pairs = [
        "run_rate", "ra_rate", "run_diff_rate",
        "win_rate", "ops", "streak", "last3_win_rate",
    ]
    for stat in diff_pairs:
        h_col = f"h_{stat}"
        a_col = f"a_{stat}"
        if h_col in df.columns and a_col in df.columns:
            df[f"diff_{stat}"] = df[h_col] - df[a_col]

    return df


# ── Public API ────────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    # Home rolling stats
    "h_run_rate", "h_ra_rate", "h_run_diff_rate", "h_win_rate",
    "h_scoring_var", "h_last3_win_rate", "h_quality_start_rt",
    "h_ops", "h_streak",
    # Away rolling stats
    "a_run_rate", "a_ra_rate", "a_run_diff_rate", "a_win_rate",
    "a_scoring_var", "a_last3_win_rate", "a_quality_start_rt",
    "a_ops", "a_streak",
    # Differentials
    "diff_run_rate", "diff_ra_rate", "diff_run_diff_rate",
    "diff_win_rate", "diff_ops", "diff_streak", "diff_last3_win_rate",
    # Context
    "home_advantage", "is_covid_season",
    # Day of week
    "dow_0", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6",
    # Month
    "month_4", "month_5", "month_6", "month_7", "month_8", "month_9", "month_10",
]

TARGET_COLUMN = "home_win"

META_COLUMNS = ["game_date", "home_team", "away_team", "season"]


def build_feature_matrix(
    game_logs_path: str | Path = RAW_DIR / "game_logs_all.csv",
    window: int = 10,
    min_games: int = 5,
    exclude_covid: bool = True,
    save: bool = True,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Build a fully feature-engineered, model-ready DataFrame.

    Parameters
    ----------
    game_logs_path : path to the combined game log CSV (from data_collection.py)
    window         : rolling window size in games (default 10)
    min_games      : minimum prior games a team must have for its row to be kept
    exclude_covid  : drop the 2020 season if True
    save           : write the result to data/features/ as CSV
    output_path    : override the default save path

    Returns
    -------
    pd.DataFrame with columns FEATURE_COLUMNS + [TARGET_COLUMN] + META_COLUMNS,
    sorted chronologically. Rows are dropped where any feature is NaN.
    """
    print("Building feature matrix...")
    print(f"  Window:        {window} games")
    print(f"  Min games:     {min_games}")
    print(f"  Exclude COVID: {exclude_covid}")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    games = pd.read_csv(game_logs_path, parse_dates=["game_date"])
    print(f"  Loaded {len(games):,} games from {game_logs_path}")

    if exclude_covid:
        before = len(games)
        games = games[games["is_covid_season"] == False].copy()
        print(f"  Dropped {before - len(games)} COVID-season games → {len(games):,} remain")

    games = games.sort_values("game_date").reset_index(drop=True)

    # ── 2. Build per-team time series ─────────────────────────────────────────
    print("  Computing per-team rolling stats...")
    ts         = _build_team_timeseries(games)
    team_stats = _rolling_team_stats(ts, window=window)

    # ── 3. Merge back onto game rows ──────────────────────────────────────────
    print("  Merging stats onto games...")
    df = _merge_team_stats_onto_games(games, team_stats)

    # ── 4. Context + differential features ───────────────────────────────────
    df = _add_context_features(df)
    df = _add_differential_features(df)

    # ── 5. Drop rows with insufficient history ────────────────────────────────
    # Proxy for "enough prior games": h_run_rate and a_run_rate are based on
    # at least min_games observations when their rolling count hits min_games.
    # We achieve this by requiring the team to have played min_games previously,
    # which we approximate by dropping the first (min_games - 1) home and away
    # appearances per team per season.
    before = len(df)
    df = _drop_early_season_rows(df, ts, min_games=min_games)
    print(f"  Dropped {before - len(df)} early-season rows (< {min_games} prior games)")

    # ── 6. Final column selection & NaN check ─────────────────────────────────
    all_cols = META_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
    missing  = [c for c in all_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns after feature build: {missing}")

    df = df[all_cols].copy()

    nan_rows = df[FEATURE_COLUMNS].isna().any(axis=1).sum()
    if nan_rows:
        print(f"  Dropping {nan_rows} rows with NaN features...")
        df = df.dropna(subset=FEATURE_COLUMNS)

    df = df.sort_values("game_date").reset_index(drop=True)

    # ── 7. Summary ────────────────────────────────────────────────────────────
    _print_summary(df)

    # ── 8. Save ───────────────────────────────────────────────────────────────
    if save:
        out = Path(output_path) if output_path else (
            FEATURES_DIR / f"features_w{window}.csv"
        )
        df.to_csv(out, index=False)
        print(f"\n  Saved → {out}")

    return df


def _drop_early_season_rows(
    df: pd.DataFrame,
    ts: pd.DataFrame,
    min_games: int,
) -> pd.DataFrame:
    """
    For each team in each season, find the date of their (min_games)-th game.
    Drop any game rows where EITHER the home or away team had fewer than
    min_games prior appearances in that season.
    """
    # Count appearances per team per season in chronological order
    ts_sorted = ts.sort_values(["season", "team", "game_date"])
    ts_sorted["game_num"] = ts_sorted.groupby(["season", "team"]).cumcount() + 1

    # Date of the min_games-th game for each team-season
    cutoff = (
        ts_sorted[ts_sorted["game_num"] == min_games]
        .groupby(["season", "team"])["game_date"]
        .first()
        .reset_index()
        .rename(columns={"game_date": "cutoff_date"})
    )

    # Merge cutoff for home team
    df = df.merge(
        cutoff.rename(columns={"team": "home_team", "cutoff_date": "h_cutoff"}),
        on=["season", "home_team"], how="left"
    )
    # Merge cutoff for away team
    df = df.merge(
        cutoff.rename(columns={"team": "away_team", "cutoff_date": "a_cutoff"}),
        on=["season", "away_team"], how="left"
    )

    # Keep only rows where game_date is AFTER both cutoffs
    mask = (df["game_date"] > df["h_cutoff"]) & (df["game_date"] > df["a_cutoff"])
    df = df[mask].drop(columns=["h_cutoff", "a_cutoff"])
    return df


def _print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'─'*60}")
    print(f"  Feature matrix summary")
    print(f"{'─'*60}")
    print(f"  Rows (games):      {len(df):,}")
    print(f"  Feature columns:   {len(FEATURE_COLUMNS)}")
    print(f"  Target balance:    {df[TARGET_COLUMN].mean():.3f} home win rate")
    print(f"  Seasons:           {sorted(df['season'].unique())}")
    print(f"\n  Feature stats (mean ± std):")
    stats = df[FEATURE_COLUMNS].agg(["mean", "std"]).T
    for col in FEATURE_COLUMNS[:12]:   # show first 12 to avoid wall of text
        m, s = stats.loc[col, "mean"], stats.loc[col, "std"]
        print(f"    {col:<28} {m:+.3f} ± {s:.3f}")
    if len(FEATURE_COLUMNS) > 12:
        print(f"    ... and {len(FEATURE_COLUMNS) - 12} more columns")
    print(f"{'─'*60}")


# ── Train / val / test split ──────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    val_season:  int = 2022,
    test_season: int = 2023,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the feature matrix chronologically into train / val / test.
    NEVER shuffle before splitting — that would leak future data into training.

    Parameters
    ----------
    val_season  : season used for hyperparameter tuning
    test_season : held-out season for final evaluation only

    Returns
    -------
    (train_df, val_df, test_df)
    """
    train = df[df["season"] <  val_season].copy()
    val   = df[df["season"] == val_season].copy()
    test  = df[df["season"] == test_season].copy()

    print(f"  Train: {len(train):,} games  ({df[df['season'] < val_season]['season'].min()}–{val_season-1})")
    print(f"  Val:   {len(val):,} games  ({val_season})")
    print(f"  Test:  {len(test):,} games  ({test_season})")

    return train, val, test


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build MLB feature matrix.")
    parser.add_argument("--window",        type=int, default=10,
                        help="Rolling window size in games (default: 10)")
    parser.add_argument("--min_games",     type=int, default=5,
                        help="Min prior games required per team (default: 5)")
    parser.add_argument("--include_covid", action="store_true",
                        help="Include the 2020 COVID season (excluded by default)")
    parser.add_argument("--game_logs",     type=str,
                        default=str(RAW_DIR / "game_logs_all.csv"),
                        help="Path to game_logs_all.csv")
    args = parser.parse_args()

    df = build_feature_matrix(
        game_logs_path=args.game_logs,
        window=args.window,
        min_games=args.min_games,
        exclude_covid=not args.include_covid,
    )

    print("\nFirst 3 rows (meta + first 5 features):")
    preview_cols = META_COLUMNS + FEATURE_COLUMNS[:5] + [TARGET_COLUMN]
    print(df[preview_cols].head(3).to_string(index=False))