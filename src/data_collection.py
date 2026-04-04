"""
data_collection.py
------------------
Pulls historical MLB game logs and team stats using pybaseball.
Saves raw data to data/raw/ as CSV files.

Usage:
    python src/data_collection.py --start_year 2015 --end_year 2023

Outputs:
    data/raw/game_logs_{year}.csv   - one row per game, all teams
    data/raw/team_batting_{year}.csv
    data/raw/team_pitching_{year}.csv
"""

import argparse
import os
import time
import warnings
from pathlib import Path

import pandas as pd
import pybaseball as pyb

warnings.filterwarnings("ignore")
pyb.cache.enable()  # cache responses so you don't re-download on reruns

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Teams that existed through our target window (excludes relocated/expansion edge cases)
MLB_TEAMS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE",
    "COL", "DET", "HOU", "KCR", "LAA", "LAD", "MIA", "MIL",
    "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SDP", "SFG",
    "SEA", "STL", "TBR", "TEX", "TOR", "WSN",
]

# 2020 was a 60-game COVID season — flag it but keep it so callers can decide
COVID_YEAR = 2020


def fetch_game_logs(year: int) -> pd.DataFrame:
    """
    Pulls the full season schedule + result for every team,
    deduplicates (each game appears for home AND away team),
    and returns one row per game.

    Columns returned:
        game_date, home_team, away_team, home_score, away_score,
        home_win (bool), is_covid_season (bool)
    """
    print(f"  Fetching game logs for {year}...")
    frames = []

    for team in MLB_TEAMS:
        try:
            df = pyb.schedule_and_record(year, team)
            df["team"] = team
            frames.append(df)
            time.sleep(0.3)  # be polite to Baseball Reference
        except Exception as e:
            print(f"    Warning: could not fetch {team} {year}: {e}")

    if not frames:
        raise RuntimeError(f"No game log data retrieved for {year}.")

    raw = pd.concat(frames, ignore_index=True)
    games = _clean_game_logs(raw, year)
    return games


def _clean_game_logs(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Normalize the schedule_and_record output into a clean, deduplicated
    game-level table. Baseball Reference returns one row per team per game,
    so we pivot to a single row per game keyed on (date, home_team).
    """
    # Rename columns to consistent snake_case
    col_map = {
        "Date": "date_raw",
        "Tm":   "team_original",
        "Opp":  "opponent",
        "R":    "runs_scored",
        "RA":   "runs_allowed",
        "W/L":  "result",
        "Home_Away": "home_away",  # not always present — handled below
    }
    raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

    # Drop rows without a result (future games, postponements)
    raw = raw[raw["result"].notna()]
    raw = raw[raw["result"].str.startswith(("W", "L"))]

    # Parse date — Baseball Reference includes day-of-week prefix e.g. "Monday, Apr 3"
    raw["game_date"] = pd.to_datetime(
        raw["date_raw"].str.replace(r"^\w+, ", "", regex=True) + f" {year}",
        format="%b %d %Y",
        errors="coerce",
    )
    raw = raw[raw["game_date"].notna()]

    # Identify home vs away from the '@' in the raw schedule
    # Baseball Reference encodes away games with '@' in the Home_Away column
    if "home_away" in raw.columns:
        raw["is_away"] = raw["home_away"].fillna("").str.strip() == "@"
    else:
        # Fallback for older scraper versions - check for Unnamed columns
        unnamed_cols = [c for c in raw.columns if "Unnamed" in str(c)]
        if unnamed_cols:
            raw["is_away"] = raw[unnamed_cols[0]].fillna("").str.strip() == "@"
        else:
            raw["is_away"] = False  # fallback — some scraper versions differ

    raw["runs_scored"] = pd.to_numeric(raw["runs_scored"], errors="coerce")
    raw["runs_allowed"] = pd.to_numeric(raw["runs_allowed"], errors="coerce")
    raw = raw.dropna(subset=["runs_scored", "runs_allowed"])

    # Build home/away columns - use vectorized operations instead of apply
    home_mask = ~raw["is_away"]
    away_mask = raw["is_away"]
    
    raw["home_team"] = raw["opponent"].where(home_mask, raw["team_original"])
    raw["away_team"] = raw["team_original"].where(home_mask, raw["opponent"])
    raw["home_score"] = raw["runs_allowed"].where(home_mask, raw["runs_scored"])
    raw["away_score"] = raw["runs_scored"].where(home_mask, raw["runs_allowed"])

    # Deduplicate — keep one row per (game_date, home_team) pair
    games = (
        raw[["game_date", "home_team", "away_team", "home_score", "away_score"]]
        .drop_duplicates(subset=["game_date", "home_team", "away_team"])
        .copy()
    )

    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
    games["season"] = year
    games["is_covid_season"] = (year == COVID_YEAR)
    games = games.sort_values("game_date").reset_index(drop=True)

    return games


def fetch_team_batting(year: int) -> pd.DataFrame:
    """Season-level team batting stats from FanGraphs via pybaseball."""
    print(f"  Fetching team batting stats for {year}...")
    df = pyb.team_batting(year)
    df["season"] = year
    return df


def fetch_team_pitching(year: int) -> pd.DataFrame:
    """Season-level team pitching stats from FanGraphs via pybaseball."""
    print(f"  Fetching team pitching stats for {year}...")
    df = pyb.team_pitching(year)
    df["season"] = year
    return df


def collect_years(start_year: int, end_year: int, skip_covid: bool = False) -> None:
    """
    Main entry point. Pulls all data for [start_year, end_year] inclusive
    and writes CSV files to data/raw/.
    """
    years = list(range(start_year, end_year + 1))
    if skip_covid and COVID_YEAR in years:
        print(f"  Skipping {COVID_YEAR} (COVID shortened season).")
        years.remove(COVID_YEAR)

    all_games = []
    all_batting = []
    all_pitching = []

    for year in years:
        print(f"\n[{year}]")
        try:
            games = fetch_game_logs(year)
            all_games.append(games)
            out = RAW_DIR / f"game_logs_{year}.csv"
            games.to_csv(out, index=False)
            print(f"    Saved {len(games)} games → {out}")
        except Exception as e:
            print(f"    ERROR fetching game logs for {year}: {e}")

        try:
            bat = fetch_team_batting(year)
            all_batting.append(bat)
            bat.to_csv(RAW_DIR / f"team_batting_{year}.csv", index=False)
        except Exception as e:
            print(f"    ERROR fetching batting for {year}: {e}")

        try:
            pit = fetch_team_pitching(year)
            all_pitching.append(pit)
            pit.to_csv(RAW_DIR / f"team_pitching_{year}.csv", index=False)
        except Exception as e:
            print(f"    ERROR fetching pitching for {year}: {e}")

    # Also write combined files spanning all years
    if all_games:
        combined = pd.concat(all_games, ignore_index=True)
        combined.to_csv(RAW_DIR / "game_logs_all.csv", index=False)
        print(f"\nCombined game log: {len(combined)} games across {len(all_games)} seasons.")

    if all_batting:
        pd.concat(all_batting, ignore_index=True).to_csv(
            RAW_DIR / "team_batting_all.csv", index=False
        )

    if all_pitching:
        pd.concat(all_pitching, ignore_index=True).to_csv(
            RAW_DIR / "team_pitching_all.csv", index=False
        )

    print("\nData collection complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect MLB historical data.")
    parser.add_argument("--start_year", type=int, default=2015)
    parser.add_argument("--end_year",   type=int, default=2023)
    parser.add_argument(
        "--skip_covid", action="store_true",
        help="Exclude the 2020 60-game COVID season from collection."
    )
    args = parser.parse_args()

    print(f"Collecting MLB data: {args.start_year}–{args.end_year}")
    collect_years(args.start_year, args.end_year, skip_covid=args.skip_covid)
