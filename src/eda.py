"""
eda.py
------
Exploratory Data Analysis for the collected MLB game log data.
Produces summary statistics and diagnostic plots saved to data/eda/.

Usage:
    python src/eda.py

Expects data/raw/game_logs_all.csv to exist (run data_collection.py first).
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
EDA_DIR = Path(__file__).parent.parent / "data" / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# ── Plotting style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1923",
    "axes.facecolor":   "#0f1923",
    "axes.edgecolor":   "#2a3a4a",
    "axes.labelcolor":  "#c8d8e8",
    "xtick.color":      "#c8d8e8",
    "ytick.color":      "#c8d8e8",
    "text.color":       "#c8d8e8",
    "grid.color":       "#1e2e3e",
    "grid.linewidth":   0.8,
    "font.family":      "monospace",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})
ACCENT   = "#00c8ff"
ACCENT2  = "#ff6b35"
POSITIVE = "#3ddc84"
NEGATIVE = "#ff4560"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_games(path: Path = RAW_DIR / "game_logs_all.csv") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run data_collection.py first."
        )
    df = pd.read_csv(path, parse_dates=["game_date"])
    return df


def print_section(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


# ── Analysis functions ────────────────────────────────────────────────────────

def summarize_dataset(df: pd.DataFrame) -> None:
    print_section("Dataset Overview")
    print(f"  Total games:       {len(df):,}")
    print(f"  Seasons covered:   {sorted(df['season'].unique())}")
    print(f"  Date range:        {df['game_date'].min().date()} → {df['game_date'].max().date()}")
    print(f"  Unique teams:      {sorted(set(df['home_team'].unique()) | set(df['away_team'].unique()))}")
    print(f"  Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    covid = df[df["is_covid_season"]]
    if len(covid):
        print(f"\n  ⚠  COVID season (2020): {len(covid)} games (60-game schedule, ~37% of normal).")


def home_win_rate(df: pd.DataFrame) -> None:
    print_section("Home Field Advantage")
    overall = df["home_win"].mean()
    print(f"  Overall home win rate: {overall:.3f} ({overall*100:.1f}%)")

    by_season = df.groupby("season")["home_win"].mean().reset_index()
    by_season.columns = ["season", "home_win_rate"]
    print(f"\n  By season:\n{by_season.to_string(index=False)}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(by_season["season"], by_season["home_win_rate"], color=ACCENT, alpha=0.85, width=0.6)
    ax.axhline(overall, color=ACCENT2, linewidth=1.5, linestyle="--", label=f"Overall avg ({overall:.3f})")
    ax.axhline(0.5, color="#ffffff", linewidth=0.8, linestyle=":", alpha=0.4, label="0.500 baseline")
    ax.set_title("Home Win Rate by Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Home Win Rate")
    ax.set_ylim(0.40, 0.65)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.legend(fontsize=9)
    ax.grid(axis="y")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "home_win_rate_by_season.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved plot: home_win_rate_by_season.png")


def run_distribution(df: pd.DataFrame) -> None:
    print_section("Run Distribution")
    for col, label in [("home_score", "Home"), ("away_score", "Away")]:
        s = df[col]
        print(f"  {label} score — mean: {s.mean():.2f}, median: {s.median()}, std: {s.std():.2f}, max: {s.max()}")

    total = df["home_score"] + df["away_score"]
    print(f"  Total runs/game  — mean: {total.mean():.2f}, median: {total.median()}")

    # Combined histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, col, color, label in [
        (axes[0], "home_score", ACCENT,  "Home Team Runs"),
        (axes[0], "away_score", ACCENT2, "Away Team Runs"),
    ]:
        ax.hist(df[col], bins=range(0, 21), alpha=0.65, color=color, label=label, edgecolor="#0f1923")
    axes[0].set_title("Run Distribution: Home vs Away")
    axes[0].set_xlabel("Runs Scored")
    axes[0].set_ylabel("Games")
    axes[0].legend()
    axes[0].grid(axis="y")

    axes[1].hist(total, bins=range(0, 35), color=POSITIVE, alpha=0.8, edgecolor="#0f1923")
    axes[1].axvline(total.mean(), color=ACCENT2, linewidth=1.5, linestyle="--",
                    label=f"Mean: {total.mean():.1f}")
    axes[1].set_title("Total Runs Per Game")
    axes[1].set_xlabel("Total Runs")
    axes[1].set_ylabel("Games")
    axes[1].legend()
    axes[1].grid(axis="y")

    plt.tight_layout()
    plt.savefig(EDA_DIR / "run_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved plot: run_distribution.png")


def score_differential(df: pd.DataFrame) -> None:
    print_section("Score Differential Analysis")
    df = df.copy()
    df["margin"] = df["home_score"] - df["away_score"]

    wins   = df[df["home_win"] == 1]["margin"]
    losses = df[df["home_win"] == 0]["margin"]

    print(f"  Home wins  — avg margin: +{wins.mean():.2f}")
    print(f"  Home losses — avg margin: {losses.mean():.2f}")
    print(f"  Walk-off / 1-run games: {(df['margin'].abs() == 1).sum():,} ({(df['margin'].abs() == 1).mean()*100:.1f}%)")
    print(f"  Blowouts (≥7 runs):     {(df['margin'].abs() >= 7).sum():,} ({(df['margin'].abs() >= 7).mean()*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(10, 4))
    bins = range(-20, 21)
    ax.hist(wins,   bins=bins, alpha=0.7, color=POSITIVE, label="Home Win",  edgecolor="#0f1923")
    ax.hist(losses, bins=bins, alpha=0.7, color=NEGATIVE, label="Home Loss", edgecolor="#0f1923")
    ax.axvline(0, color="white", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title("Home Score Margin Distribution")
    ax.set_xlabel("Home Score − Away Score")
    ax.set_ylabel("Games")
    ax.legend()
    ax.grid(axis="y")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "score_differential.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved plot: score_differential.png")


def team_win_rates(df: pd.DataFrame) -> None:
    print_section("Team Win Rates (all seasons combined)")

    # A team wins if it's home and home_win=1, or away and home_win=0
    home_wins  = df.groupby("home_team")["home_win"].agg(["sum", "count"]).rename(
        columns={"sum": "hw", "count": "hg"})
    away_wins  = df.groupby("away_team").apply(
        lambda x: pd.Series({
            "aw": (x["home_win"] == 0).sum(),
            "ag": len(x)
        }), include_groups=False
    )

    teams = home_wins.join(away_wins, how="outer").fillna(0)
    teams["total_wins"]  = teams["hw"] + teams["aw"]
    teams["total_games"] = teams["hg"] + teams["ag"]
    teams["win_rate"]    = teams["total_wins"] / teams["total_games"]
    teams = teams.sort_values("win_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 9))
    colors = [POSITIVE if r >= 0.5 else NEGATIVE for r in teams["win_rate"]]
    ax.barh(teams.index, teams["win_rate"], color=colors, alpha=0.85)
    ax.axvline(0.5, color="white", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title("Overall Win Rate by Team")
    ax.set_xlabel("Win Rate")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlim(0.35, 0.65)
    ax.grid(axis="x")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "team_win_rates.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved plot: team_win_rates.png")
    print(f"\n  Top 5:\n{teams[['win_rate']].tail(5).iloc[::-1].to_string()}")
    print(f"\n  Bottom 5:\n{teams[['win_rate']].head(5).to_string()}")


def games_per_season(df: pd.DataFrame) -> None:
    print_section("Games Per Season (data completeness check)")
    by_season = df.groupby("season").size().reset_index(name="num_games")
    expected  = {2020: 900, **{y: 2430 for y in range(2015, 2024) if y != 2020}}
    by_season["expected"]  = by_season["season"].map(expected)
    by_season["pct_found"] = (by_season["num_games"] / by_season["expected"] * 100).round(1)
    print(by_season.to_string(index=False))

    low = by_season[by_season["pct_found"] < 90]
    if len(low):
        print(f"\n  ⚠  Low coverage seasons: {low['season'].tolist()} — inspect raw CSVs.")


def check_data_leakage_risk(df: pd.DataFrame) -> None:
    """
    Flags any structural issues that could cause future data
    to bleed into training — e.g., duplicate games, NaN dates.
    """
    print_section("Data Leakage / Integrity Checks")

    dupes = df.duplicated(subset=["game_date", "home_team", "away_team"])
    print(f"  Duplicate (date, home, away) rows: {dupes.sum()}")

    null_dates = df["game_date"].isna().sum()
    print(f"  Null game_date rows:               {null_dates}")

    null_scores = (df["home_score"].isna() | df["away_score"].isna()).sum()
    print(f"  Null score rows:                   {null_scores}")

    # Ties should not exist in MLB (extra innings resolve them)
    ties = (df["home_score"] == df["away_score"]).sum()
    print(f"  Tied games (unexpected):           {ties}")

    if dupes.sum() == 0 and null_dates == 0 and null_scores == 0 and ties == 0:
        print("\n  ✓ No critical data integrity issues found.")
    else:
        print("\n  ⚠  Issues found — inspect and clean before feature engineering.")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_eda(games_path: Path = RAW_DIR / "game_logs_all.csv") -> None:
    df = load_games(games_path)

    summarize_dataset(df)
    games_per_season(df)
    check_data_leakage_risk(df)
    home_win_rate(df)
    run_distribution(df)
    score_differential(df)
    team_win_rates(df)

    print(f"\n✓ EDA complete. Plots saved to: {EDA_DIR}\n")


if __name__ == "__main__":
    run_eda()
