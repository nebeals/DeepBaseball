"""
odds_comparison.py
------------------
Full script with fixed header formatting to prevent SyntaxErrors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── The Odds API config ───────────────────────────────────────────────────────
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "baseball_mlb"
MARKET = "h2h"
REGIONS = "us"

TEAM_NAME_MAP = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox", "CHC": "Chicago Cubs", "CHW": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians", "COL": "Colorado Rockies",
    "DET": "Detroit Tigers", "HOU": "Houston Astros", "KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "OAK": "Oakland Athletics", "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates", "SDP": "San Diego Padres", "SFG": "San Francisco Giants",
    "SEA": "Seattle Mariners", "STL": "St. Louis Cardinals", "TBR": "Tampa Bay Rays",
    "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays", "WSN": "Washington Nationals",
}
NAME_TO_ABBR = {v: k for k, v in TEAM_NAME_MAP.items()}


# ── moneyline math ────────────────────────────────────────────────────────────

def american_to_prob(ml: float) -> float:
    return (-ml) / (-ml + 100) if ml < 0 else 100 / (ml + 100)


def prob_to_american(prob: float) -> str:
    """Convert probability to American ML string with explicit + or - sign."""
    prob = max(0.001, min(0.999, prob))
    if prob >= 0.5:
        val = -round((prob / (1 - prob)) * 100)
    else:
        val = round(((1 - prob) / prob) * 100)
    return f"{val:+d}"


def fmt_ml(val: float) -> str:
    """Formats a raw moneyline float to a string with an explicit sign."""
    return f"{int(val):+d}"


def remove_vig(home_p: float, away_p: float) -> tuple[float, float]:
    total = home_p + away_p
    return home_p / total, away_p / total


# ── caching & fetching ────────────────────────────────────────────────────────

def get_cache_path(date_str: str) -> Path:
    return CACHE_DIR / f"odds_raw_{date_str}.json"


def save_cached_data(raw_games: list[dict], date_str: str):
    data = {"date": date_str, "timestamp": datetime.now().isoformat(), "raw_games": raw_games}
    get_cache_path(date_str).write_text(json.dumps(data, indent=2))


def load_cached_data(date_str: str) -> Optional[list[dict]]:
    cp = get_cache_path(date_str)
    if not cp.exists(): return None
    try:
        d = json.loads(cp.read_text())
        if d.get("date") == date_str: return d.get("raw_games")
    except:
        pass
    return None


def fetch_odds(api_key: str) -> list[dict]:
    url = f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds"
    params = {"apiKey": api_key, "regions": REGIONS, "markets": MARKET, "oddsFormat": "american", "dateFormat": "iso"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def parse_odds(raw_games: list[dict], remove_vig_flag: bool = True) -> pd.DataFrame:
    rows = []
    for g in raw_games:
        h_abbr, a_abbr = NAME_TO_ABBR.get(g["home_team"]), NAME_TO_ABBR.get(g["away_team"])
        if not h_abbr or not a_abbr: continue
        for bk in g.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m["key"] != "h2h": continue
                h_ml = next((o["price"] for o in m["outcomes"] if o["name"] == g["home_team"]), None)
                a_ml = next((o["price"] for o in m["outcomes"] if o["name"] == g["away_team"]), None)
                if h_ml is None or a_ml is None: continue
                hp, ap = remove_vig(american_to_prob(h_ml), american_to_prob(a_ml)) if remove_vig_flag else (
                    american_to_prob(h_ml), american_to_prob(a_ml))
                rows.append(
                    {"home_team": h_abbr, "away_team": a_abbr, "home_ml": h_ml, "away_ml": a_ml, "home_mkt_p": hp,
                     "away_mkt_p": ap, "book": bk["key"]})
    return pd.DataFrame(rows)


# ── comparison ────────────────────────────────────────────────────────────────

def calculate_kelly(row: pd.Series) -> float:
    if row['edge_raw'] > 0:  # Home
        p, ml = row['home_win_prob'], row['home_ml']
    else:  # Away
        p, ml = (1 - row['home_win_prob']), row['away_ml']
    b = (ml / 100) if ml > 0 else (100 / abs(ml))
    f = (b * p - (1 - p)) / b
    return max(0, f)


def compare(predictions: pd.DataFrame, odds: pd.DataFrame, min_edge: float) -> pd.DataFrame:
    merged = predictions.merge(odds, on=["home_team", "away_team"], how="inner")
    if merged.empty: return merged

    merged["edge_raw"] = (merged["home_win_prob"] - merged["home_mkt_p"]).round(4)
    merged["EDGE"] = (merged["edge_raw"].abs() * 100).round(1)

    idx = merged.groupby(["home_team", "away_team"])["EDGE"].idxmax()
    df = merged.loc[idx].copy()
    df = df[df["EDGE"] >= (min_edge * 100)]
    if df.empty: return df

    df["kelly_f"] = df.apply(calculate_kelly, axis=1)
    total_k = df["kelly_f"].sum()
    df["STAKE %"] = (df["kelly_f"] / total_k * 100).round(1) if total_k > 0 else 0.0

    df["MOD %"] = df.apply(lambda r: f"{(1 - r['home_win_prob']) * 100:0.1f}% / {r['home_win_prob'] * 100:0.1f}%",
                           axis=1)
    df["MKT %"] = df.apply(lambda r: f"{(1 - r['home_mkt_p']) * 100:0.1f}% / {r['home_mkt_p'] * 100:0.1f}%", axis=1)
    df["MOD ML"] = df["home_win_prob"].apply(prob_to_american)
    df["MKT ML"] = df["home_ml"].apply(fmt_ml)
    df["SIDE"] = df.apply(lambda r: r["home_team"] if r["edge_raw"] > 0 else r["away_team"], axis=1)
    df["BOOK"] = df["book"]

    return df.sort_values("STAKE %", ascending=False).reset_index(drop=True)


# ── output ────────────────────────────────────────────────────────────────────

def save_comparison(result: pd.DataFrame, date_str: str, min_edge: float, remove_vig: bool):
    txt_path, csv_path = REPORTS_DIR / f"odds_{date_str}.txt", REPORTS_DIR / f"odds_{date_str}.csv"

    # CSV Column order
    cols = ["home_team", "away_team", "MOD %", "MKT %", "MOD ML", "MKT ML", "EDGE", "SIDE", "STAKE %", "BOOK"]
    result[cols].to_csv(csv_path, index=False)

    vig_s = "vig removed" if remove_vig else "vig included"

    # Header construction split for safety
    h1 = f"  {'MATCHUP':<18}  {'MOD % (A/H)':<13}  {'MKT % (A/H)':<13}  "
    h2 = f"{'MOD ML':>8}  {'MKT ML':>8}  {'EDGE %':>6}  {'SIDE':<5}  {'STAKE %':>8}  {'BOOK':<12}"

    s1 = f"  {'':─<18}  {'':─<13}  {'':─<13}  "
    s2 = f"{'':─<8}  {'':─<8}  {'':─<6}  {'':─<5}  {'':─<8}  {'':─<12}"

    lines = [
        "═" * 115,
        f"  MLB OPTIMAL ALLOCATION REPORT  —  {date_str}",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Min Edge: {min_edge * 100:.0f}%  |  Market: {vig_s}",
        "═" * 115,
        "",
        h1 + h2,
        s1 + s2,
    ]

    for _, r in result.iterrows():
        matchup = f"{r['away_team']:>4} @ {r['home_team']:<4}"
        row_str = (
            f"  {matchup:<18}  {r['MOD %']:<13}  {r['MKT %']:<13}  "
            f"{r['MOD ML']:>8}  {r['MKT ML']:>8}  {r['EDGE']:>6.1f}%  "
            f"{r['SIDE']:<5}  {r['STAKE %']:>7.1f}%   {r['BOOK'][:12]:<12}"
        )
        lines.append(row_str)

    lines += ["", "  NOTE: MOD ML and MKT ML refer to the HOME team price for direct comparison.", "═" * 115]
    txt_path.write_text("\n".join(lines))
    return txt_path, csv_path


def main(args):
    date_str = args.date or date.today().strftime("%Y-%m-%d")
    api_key = args.api_key or os.environ.get("ODDS_API_KEY")
    if not api_key: return print("Error: No ODDS_API_KEY found.")

    pred_csv = REPORTS_DIR / f"predictions_{date_str}.csv"
    if not pred_csv.exists(): return print(f"Error: {pred_csv} not found.")

    predictions = pd.read_csv(pred_csv)
    raw_games = load_cached_data(date_str) or fetch_odds(api_key)
    if not load_cached_data(date_str): save_cached_data(raw_games, date_str)

    odds = parse_odds(raw_games, remove_vig_flag=args.remove_vig)
    result = compare(predictions, odds, min_edge=args.min_edge)

    if result.empty: return print("No edges found.")
    txt, _ = save_comparison(result, date_str, args.min_edge, args.remove_vig)
    print(txt.read_text())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--api_key", default=None)
    p.add_argument("--date", default=None)
    p.add_argument("--min_edge", type=float, default=0.01)
    p.add_argument("--remove_vig", action="store_true", default=True)
    main(p.parse_args())