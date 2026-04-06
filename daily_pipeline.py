"""
daily_pipeline.py
-----------------
Morning pipeline: refresh game logs → rebuild features → predict today's games.

Run this each morning before games start. It will:
  1. Pull completed game logs for the current season from Baseball Reference
  2. Append them to the historical data (or start fresh if first run of season)
  3. Rebuild the rolling feature matrix through yesterday
  4. Identify today's scheduled games
  5. Run model inference on each matchup
  6. Save predictions to reports/predictions_YYYY-MM-DD.txt and .csv

Usage:
    python src/daily_pipeline.py
    python src/daily_pipeline.py --checkpoint checkpoints/best_mlp.pt
    python src/daily_pipeline.py --window 15  # use 15-game rolling window
    python src/daily_pipeline.py --dry_run    # skip data pull, use cached CSVs

Scheduling (run once per day before first pitch, e.g. 9 AM):
    # macOS / Linux crontab:
    0 9 * * * cd /path/to/mlb_win_probability && python src/daily_pipeline.py

    # Windows Task Scheduler: point to python.exe and this script
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# ── path setup ────────────────────────────────────────────────────────────────
_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.data_collection import MLB_TEAMS, fetch_game_logs
from src.features import (
    FEATURE_COLUMNS as FEATURE_COLS, META_COLUMNS as META_COLS,
    _build_team_timeseries, _rolling_team_stats,
    _merge_team_stats_onto_games, _add_context_features,
    _add_differential_features,
)
from src.predict_game import run_inference, _confidence, _apply_scaler
from src.train import (
    load_checkpoint, PlattWrapper, IsotonicWrapper
)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
RAW_DIR     = ROOT / "data" / "raw"
FEAT_DIR    = ROOT / "data" / "features"
REPORTS_DIR = ROOT / "reports"
CHECKPOINTS_DIR = ROOT / "checkpoints"

for d in (RAW_DIR, FEAT_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_CKPT   = ROOT / "checkpoints" / "best_resnet.pt"
COMBINED_LOGS  = RAW_DIR / "game_logs_all.csv"
UPDATE_TRACKER = RAW_DIR / ".last_update.json"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def _step(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def _load_last_update() -> date | None:
    """Load the date of the last successful game log update."""
    if UPDATE_TRACKER.exists():
        try:
            data = json.loads(UPDATE_TRACKER.read_text())
            return datetime.strptime(data["last_update"], "%Y-%m-%d").date()
        except:
            pass
    return None


def _save_last_update(update_date: date):
    """Save the date of the last successful game log update."""
    UPDATE_TRACKER.write_text(json.dumps({
        "last_update": update_date.strftime("%Y-%m-%d"),
        "timestamp": datetime.now().isoformat()
    }, indent=2))


# ── Step 1: refresh game logs ─────────────────────────────────────────────────

def refresh_game_logs(today: date, dry_run: bool = False) -> pd.DataFrame:
    """
    Pull completed game logs for the current MLB season.
    Merges with historical data if present, deduplicates, and saves.

    Returns the full combined game log DataFrame.
    """
    _step("Step 1/4 — Refreshing game logs")

    current_year = today.year
    last_update = _load_last_update()

    if dry_run:
        _log("Dry run: skipping data pull, loading cached CSVs.")
        if not COMBINED_LOGS.exists():
            raise FileNotFoundError(
                f"No cached game logs found at {COMBINED_LOGS}. "
                "Run without --dry_run first."
            )
        return pd.read_csv(COMBINED_LOGS, parse_dates=["game_date"])

    # Load existing data
    existing_current_season: pd.DataFrame | None = None
    historical: list[pd.DataFrame] = []
    
    if COMBINED_LOGS.exists():
        existing = pd.read_csv(COMBINED_LOGS, parse_dates=["game_date"])
        # Keep historical data (prior seasons)
        historical_only = existing[existing["season"] < current_year]
        if len(historical_only):
            historical.append(historical_only)
            _log(f"Loaded {len(historical_only):,} historical games "
                 f"({historical_only['season'].min()}–{current_year - 1})")
        
        # Keep existing current season data
        existing_current = existing[existing["season"] == current_year]
        if len(existing_current):
            existing_current_season = existing_current
            last_game_date = existing_current["game_date"].max().strftime("%Y-%m-%d")
            _log(f"Loaded {len(existing_current):,} existing {current_year} games (through {last_game_date})")

    # Check if already up-to-date
    if last_update and existing_current_season is not None:
        if last_update >= today:
            _log(f"Data is current (last update: {last_update}). Skipping fetch.")
            all_frames = historical + [existing_current_season]
            combined = pd.concat(all_frames, ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["game_date", "home_team", "away_team"]
            ).sort_values("game_date").reset_index(drop=True)
            return combined

    # Determine date range to fetch
    if last_update and existing_current_season is not None:
        # Only fetch from the day after last update
        fetch_from = last_update + timedelta(days=1)
        _log(f"Fetching games from {fetch_from} to {today}...")
    else:
        # No existing data or no tracker - full season fetch
        fetch_from = date(current_year, 3, 1)  # Opening Day-ish
        _log(f"No existing data found. Fetching full {current_year} season...")
    # Disable pybaseball cache to ensure fresh data
    import pybaseball as pyb
    pyb.cache.disable()

    current_frames: list[pd.DataFrame] = []
    failed_teams: list[str] = []

    for i, team in enumerate(MLB_TEAMS, 1):
        try:
            df = fetch_game_logs(current_year)
            # Filter to games from fetch_from onwards
            df["game_date"] = pd.to_datetime(df["game_date"])
            df = df[df["game_date"] >= pd.Timestamp(fetch_from)]
            # Only completed games
            if "home_win" in df.columns:
                df = df[df["home_win"].notna()]
            if len(df):
                current_frames.append(df)
                _log(f"Retrieved {len(df):,} new games from {fetch_from} onwards.")
            break
        except Exception as e:
            _log(f"  Warning [{team}]: {e}")
            failed_teams.append(team)
            time.sleep(1)

    # fetch_game_logs already loops all teams internally and deduplicates.
    # If it failed, fall back to per-team pulls.
    if not current_frames:
        _log("Bulk fetch failed — falling back to per-team pulls...")
        import pybaseball as pyb
        pyb.cache.enable()
        for team in MLB_TEAMS:
            try:
                raw = pyb.schedule_and_record(current_year, team)
                raw["team"] = team
                current_frames.append(raw)
                time.sleep(0.4)
            except Exception as e:
                _log(f"  Warning [{team}]: {e}")

    if current_frames:
        new_games = pd.concat(current_frames, ignore_index=True)
        _log(f"Retrieved {len(new_games):,} new games.")
    else:
        _log("No new games found.")
        new_games = pd.DataFrame()

    # Combine new games with existing current season data
    all_current_frames: list[pd.DataFrame] = []
    if existing_current_season is not None:
        all_current_frames.append(existing_current_season)
    if len(new_games):
        all_current_frames.append(new_games)
    
    if all_current_frames:
        current_games = pd.concat(all_current_frames, ignore_index=True)
        current_games = current_games.drop_duplicates(
            subset=["game_date", "home_team", "away_team"]
        )
        _log(f"Combined current season: {len(current_games):,} unique games.")
    else:
        current_games = pd.DataFrame()
        _log("Could not retrieve current season data. Using cached data only.")

    # Combine and deduplicate
    all_frames = historical + ([current_games] if len(current_games) else [])
    if not all_frames:
        raise RuntimeError("No game log data available.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["game_date", "home_team", "away_team"]
    ).sort_values("game_date").reset_index(drop=True)

    combined.to_csv(COMBINED_LOGS, index=False)
    _save_last_update(today)
    _log(f"Combined log: {len(combined):,} games → {COMBINED_LOGS}")

    return combined


# ── Step 2: identify today's scheduled games ──────────────────────────────────

def get_todays_schedule(today: date) -> list[dict]:
    """
    Pull today's schedule from Baseball Reference via pybaseball.
    Returns a list of {home_team, away_team, game_date} dicts.
    Falls back to an empty list with a clear message if unavailable.
    """
    _step("Step 2/4 — Fetching today's schedule")
    import pybaseball as pyb

    schedule: list[dict] = []
    date_str = today.strftime("%Y-%m-%d")
    seen: set[tuple] = set()

    for team in MLB_TEAMS:
        try:
            df = pyb.schedule_and_record(today.year, team)
            # Rows without a W/L result are future (or today's) games
            upcoming = df[df["W/L"].isna()].copy()
            upcoming["game_date"] = pd.to_datetime(
                upcoming["Date"].str.replace(r"^\w+, ", "", regex=True)
                + f" {today.year}",
                format="%b %d %Y", errors="coerce"
            )
            today_games = upcoming[
                upcoming["game_date"].dt.strftime("%Y-%m-%d") == date_str
            ]

            for _, row in today_games.iterrows():
                unnamed = [c for c in df.columns if "Unnamed" in str(c)]
                is_away = (
                    row[unnamed[0]].strip() == "@"
                    if unnamed and pd.notna(row[unnamed[0]])
                    else False
                )
                home = row["Opp"] if is_away else team
                away = team     if is_away else row["Opp"]
                key  = (home, away)
                if key not in seen:
                    seen.add(key)
                    schedule.append({
                        "home_team": home,
                        "away_team": away,
                        "game_date": date_str,
                    })
            time.sleep(0.3)
        except Exception:
            continue

    if schedule:
        _log(f"Found {len(schedule)} games scheduled for {date_str}.")
    else:
        _log(f"No schedule data found for {date_str}. "
             f"(Off-day, or Baseball Reference not yet updated?)")

    return schedule


# ── Step 3: rebuild feature matrix ───────────────────────────────────────────

def rebuild_features(
    game_logs:  pd.DataFrame,
    window:     int = 10,
    min_games:  int = 5,
) -> pd.DataFrame:
    """
    Recompute the rolling feature matrix from the latest game logs.
    Only uses completed games, so features are always based on past results.
    """
    _step("Step 3/4 — Rebuilding feature matrix")

    # Exclude COVID season and any incomplete rows
    games = game_logs[
        (game_logs["is_covid_season"] == False) &
        game_logs["home_win"].notna()
    ].copy()
    games = games.sort_values("game_date").reset_index(drop=True)

    _log(f"Building rolling stats (window={window}) on {len(games):,} games...")

    ts         = _build_team_timeseries(games)
    team_stats = _rolling_team_stats(ts, window=window)
    df         = _merge_team_stats_onto_games(games, team_stats)
    df         = _add_context_features(df)
    df         = _add_differential_features(df)

    # Drop early-season rows where teams have insufficient history
    df = df.dropna(subset=FEATURE_COLS)
    df = df.sort_values("game_date").reset_index(drop=True)

    out_path = FEAT_DIR / f"features_w{window}.csv"
    df.to_csv(out_path, index=False)
    _log(f"Feature matrix: {len(df):,} rows → {out_path}")

    return df


# ── Step 4: predict and save ──────────────────────────────────────────────────

def _get_last_known_stats(
    team:      str,
    game_logs: pd.DataFrame,
    feat_df:   pd.DataFrame,
    today:     date,
) -> dict | None:
    """
    Extract rolling stats for a team from their most recent completed game
    in the feature matrix. Used to build feature vectors for today's games,
    which are not yet in the feature matrix.
    """
    today_ts = pd.Timestamp(today)

    # Find the most recent game where this team appeared
    home_rows = feat_df[
        (feat_df["home_team"] == team) & (feat_df["game_date"] < today_ts)
    ].sort_values("game_date")

    away_rows = feat_df[
        (feat_df["away_team"] == team) & (feat_df["game_date"] < today_ts)
    ].sort_values("game_date")

    if len(home_rows) == 0 and len(away_rows) == 0:
        return None

    # Pick whichever is more recent
    last_home = home_rows.iloc[-1] if len(home_rows) else None
    last_away = away_rows.iloc[-1] if len(away_rows) else None

    if last_home is not None and last_away is not None:
        row = last_home if last_home["game_date"] >= last_away["game_date"] else last_away
        prefix = "h_" if row is last_home else "a_"
    elif last_home is not None:
        row, prefix = last_home, "h_"
    else:
        row, prefix = last_away, "a_"

    stat_names = [
        "win_rate", "run_rate", "ra_rate", "run_diff_rate",
        "scoring_var", "last3_win_rate", "quality_start_rt", "ops", "streak",
    ]

    stats = {}
    for s in stat_names:
        col = f"{prefix}{s}"
        if col in row.index:
            stats[s] = float(row[col])

    return stats


def predict_todays_games(
    schedule:   list[dict],
    feat_df:    pd.DataFrame,
    game_logs:  pd.DataFrame,
    model:      torch.nn.Module,
    meta:       dict,
    device:     torch.device,
    today:      date,
) -> list[dict]:
    """
    Build feature vectors for each game in today's schedule and run inference.
    Returns a list of prediction dicts.
    """
    _step("Step 4/4 — Running model inference")

    if not schedule:
        _log("No games to predict.")
        return []

    from predict_game import _build_feature_row

    predictions = []
    date_str    = today.strftime("%Y-%m-%d")

    for game in schedule:
        home = game["home_team"]
        away = game["away_team"]

        h_stats = _get_last_known_stats(home, game_logs, feat_df, today)
        a_stats = _get_last_known_stats(away, game_logs, feat_df, today)

        if h_stats is None or a_stats is None:
            _log(f"  Skipping {away} @ {home}: insufficient history.")
            continue

        # Build unified stats dict with h_/a_ prefixes
        stats = {f"h_{k}": v for k, v in h_stats.items()}
        stats.update({f"a_{k}": v for k, v in a_stats.items()})

        X     = _build_feature_row(stats, date_str)[None, :]
        probs = run_inference(X, model, meta, device)
        prob  = float(probs[0])

        # Convert to American moneyline equivalent
        if prob >= 0.5:
            ml_equiv = -round((prob / (1 - prob)) * 100)
        else:
            ml_equiv = round(((1 - prob) / prob) * 100)

        predictions.append({
            "game_date":    date_str,
            "home_team":    home,
            "away_team":    away,
            "home_win_prob": round(prob, 4),
            "away_win_prob": round(1 - prob, 4),
            "ml_equivalent": ml_equiv,
            "confidence":   _confidence(prob),
            "model_arch":   meta.get("arch", "?"),
            "calibrated":   meta.get("calibrator") is not None,
        })

        _log(f"  {away:>4} @ {home:<4}  home={prob*100:.1f}%  "
             f"ML equiv: {ml_equiv:+d}  [{_confidence(prob)}]")

    return predictions


# ── save reports ──────────────────────────────────────────────────────────────

def save_reports(predictions: list[dict], today: date) -> tuple[Path, Path]:
    """
    Save predictions as both a human-readable .txt and a .csv.
    Returns (txt_path, csv_path).
    """
    date_str = today.strftime("%Y-%m-%d")
    txt_path = PREDICTIONS_DIR / f"predictions_{date_str}.txt"
    csv_path = PREDICTIONS_DIR / f"predictions_{date_str}.csv"

    # ── CSV ───────────────────────────────────────────────────────────────────
    df = pd.DataFrame(predictions)
    df.to_csv(csv_path, index=False)

    # ── Plain text ────────────────────────────────────────────────────────────
    lines = [
        "═" * 62,
        f"  MLB WIN PROBABILITY REPORT  —  {date_str}",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "═" * 62,
        "",
    ]

    if not predictions:
        lines += ["  No games found for today.", ""]
    else:
        lines += [
            f"  {'MATCHUP':<22} {'HOME%':>6}  {'AWAY%':>6}  "
            f"{'ML EQUIV':>9}  CONFIDENCE",
            "  " + "─" * 58,
        ]
        for p in predictions:
            matchup = f"{p['away_team']:>4} @ {p['home_team']:<4}"
            ml_str  = f"{p['ml_equivalent']:+d}"
            lines.append(
                f"  {matchup:<22} {p['home_win_prob']*100:>5.1f}%  "
                f"{p['away_win_prob']*100:>5.1f}%  "
                f"{ml_str:>9}  {p['confidence']}"
            )

        lines += [
            "  " + "─" * 58,
            f"  {len(predictions)} games  |  "
            f"avg home win prob: "
            f"{sum(p['home_win_prob'] for p in predictions)/len(predictions)*100:.1f}%",
            "",
            "  ML EQUIV: American moneyline equivalent of model probability.",
            "  Negative = home team favored. Positive = away team favored.",
        ]

    lines += [
        "",
        "═" * 62,
        f"  Model: {predictions[0]['model_arch'] if predictions else 'N/A'}  |  "
        f"Calibrated: {predictions[0]['calibrated'] if predictions else 'N/A'}",
        "═" * 62,
    ]

    txt_path.write_text("\n".join(lines))

    return txt_path, csv_path


# ── main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    today  = date.today()
    device = torch.device("cpu")

    print(f"\n{'═'*62}")
    print(f"  MLB Daily Pipeline  —  {today.strftime('%A, %B %-d, %Y')}")
    print(f"{'═'*62}")

    # Load model once
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"\nCheckpoint not found: {ckpt_path}")
        print("Train the model first: python src/train.py")
        sys.exit(1)

    _log(f"Loading model from {ckpt_path.name}...")
    model, meta = load_checkpoint(ckpt_path, device=device)

    # Step 1: refresh game logs
    try:
        game_logs = refresh_game_logs(today, dry_run=args.dry_run)
    except Exception as e:
        _log(f"ERROR refreshing game logs: {e}")
        sys.exit(1)

    # Step 2: today's schedule
    if args.dry_run:
        _log("Dry run: schedule fetch skipped. Using --date if provided.")
        schedule = []
        if args.date:
            # Parse a manually specified date for testing
            test_date = datetime.strptime(args.date, "%Y-%m-%d").date()
            today     = test_date
    else:
        schedule = get_todays_schedule(today)

    # Step 3: rebuild features
    try:
        feat_df = rebuild_features(game_logs, window=args.window)
    except Exception as e:
        _log(f"ERROR rebuilding features: {e}")
        sys.exit(1)

    # Step 4: predict
    predictions = predict_todays_games(
        schedule, feat_df, game_logs, model, meta, device, today
    )

    # Save
    if predictions or True:   # always save, even if empty (useful for log history)
        txt_path, csv_path = save_reports(predictions, today)
        print(f"\n{'─'*62}")
        _log(f"Report saved:")
        _log(f"  Text → {txt_path}")
        _log(f"  CSV  → {csv_path}")

    print(f"\n{'═'*62}")
    _log("Pipeline complete.")
    print(f"{'═'*62}\n")

    # Print the text report to stdout too
    print(txt_path.read_text())


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MLB daily morning pipeline.")
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    p.add_argument("--window",     type=int, default=15,
                   help="Rolling window size in games (default 15)")
    p.add_argument("--dry_run",    action="store_true",
                   help="Skip data pull, use cached CSVs")
    p.add_argument("--date",       default=None,
                   help="Override today's date (YYYY-MM-DD) — useful for testing")
    main(p.parse_args())