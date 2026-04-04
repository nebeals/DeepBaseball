"""
predict_game.py
---------------
Predict the home-team win probability for upcoming MLB games.

This script is the inference counterpart to train.py. It loads a
saved checkpoint (which contains the model weights, scaler, and
calibrator) and produces win probabilities for games that have
not yet taken place.

Two modes
---------
1. From the feature matrix (recommended)
   The feature matrix built by features.py already contains rolling
   stats for every game in the dataset. For a future game, run
   features.py on up-to-date game logs first, then use --lookup to
   find the matchup you want.

2. Manual stat entry (--manual)
   Supply rolling stats directly via command-line flags. Useful when
   you want a quick prediction without rerunning the full pipeline,
   or when you already know the teams' recent numbers.

Examples
--------
# Mode 1: look up a game already in the feature matrix
python src/predict_game.py \\
    --checkpoint checkpoints/best_mlp.pt \\
    --features   data/features/features_w10.csv \\
    --home NYY --away BOS --date 2023-09-15

# Mode 1: predict all games on a given date
python src/predict_game.py \\
    --checkpoint checkpoints/best_mlp.pt \\
    --features   data/features/features_w10.csv \\
    --date 2023-09-15

# Mode 2: supply stats manually
python src/predict_game.py \\
    --checkpoint checkpoints/best_mlp.pt \\
    --manual \\
    --home NYY --away BOS --date 2024-07-04 \\
    --h_win_rate 0.60 --h_run_rate 5.1 --h_ra_rate 3.8 \\
    --a_win_rate 0.52 --a_run_rate 4.6 --a_ra_rate 4.2
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# ── path setup ────────────────────────────────────────────────────────────────
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from train import FEATURE_COLS, load_checkpoint

ROOT     = Path(__file__).parent.parent
CKPT_DIR = ROOT / "checkpoints"
FEAT_DIR = ROOT / "data" / "features"

# League-average defaults for optional stats
LEAGUE_AVG = {
    "h_scoring_var":      2.1,
    "h_last3_win_rate":   0.500,
    "h_quality_start_rt": 0.370,
    "h_ops":              0.500,
    "h_streak":           0,
    "a_scoring_var":      2.1,
    "a_last3_win_rate":   0.500,
    "a_quality_start_rt": 0.370,
    "a_ops":              0.500,
    "a_streak":           0,
}


# ── feature assembly ──────────────────────────────────────────────────────────

def _build_feature_row(stats: dict, game_date: str) -> np.ndarray:
    """
    Convert a flat dict of rolling stats into the ordered FEATURE_COLS
    numpy array, computing derived/context features automatically.
    """
    d = datetime.strptime(game_date, "%Y-%m-%d")

    h_ops = stats.get("h_ops", LEAGUE_AVG["h_ops"])
    a_ops = stats.get("a_ops", LEAGUE_AVG["a_ops"])

    row = {
        # Home rolling stats
        "h_run_rate":          stats["h_run_rate"],
        "h_ra_rate":           stats["h_ra_rate"],
        "h_run_diff_rate":     stats.get("h_run_diff_rate",
                                         stats["h_run_rate"] - stats["h_ra_rate"]),
        "h_win_rate":          stats["h_win_rate"],
        "h_scoring_var":       stats.get("h_scoring_var",      LEAGUE_AVG["h_scoring_var"]),
        "h_last3_win_rate":    stats.get("h_last3_win_rate",   LEAGUE_AVG["h_last3_win_rate"]),
        "h_quality_start_rt":  stats.get("h_quality_start_rt", LEAGUE_AVG["h_quality_start_rt"]),
        "h_ops":               h_ops,
        "h_streak":            stats.get("h_streak",           LEAGUE_AVG["h_streak"]),
        # Away rolling stats
        "a_run_rate":          stats["a_run_rate"],
        "a_ra_rate":           stats["a_ra_rate"],
        "a_run_diff_rate":     stats.get("a_run_diff_rate",
                                         stats["a_run_rate"] - stats["a_ra_rate"]),
        "a_win_rate":          stats["a_win_rate"],
        "a_scoring_var":       stats.get("a_scoring_var",      LEAGUE_AVG["a_scoring_var"]),
        "a_last3_win_rate":    stats.get("a_last3_win_rate",   LEAGUE_AVG["a_last3_win_rate"]),
        "a_quality_start_rt":  stats.get("a_quality_start_rt", LEAGUE_AVG["a_quality_start_rt"]),
        "a_ops":               a_ops,
        "a_streak":            stats.get("a_streak",           LEAGUE_AVG["a_streak"]),
        # Differentials (computed)
        "diff_run_rate":       stats["h_run_rate"]    - stats["a_run_rate"],
        "diff_ra_rate":        stats["h_ra_rate"]     - stats["a_ra_rate"],
        "diff_run_diff_rate":  stats.get("h_run_diff_rate", stats["h_run_rate"] - stats["h_ra_rate"]) -
                               stats.get("a_run_diff_rate", stats["a_run_rate"] - stats["a_ra_rate"]),
        "diff_win_rate":       stats["h_win_rate"]    - stats["a_win_rate"],
        "diff_ops":            h_ops - a_ops,
        "diff_streak":         stats.get("h_streak", 0) - stats.get("a_streak", 0),
        "diff_last3_win_rate": stats.get("h_last3_win_rate", 0.5) -
                               stats.get("a_last3_win_rate", 0.5),
        # Context
        "home_advantage":  1.0,
        "is_covid_season": 0.0,
        # Day of week one-hot (Mon=0, Sun=6)
        **{f"dow_{i}":   float(d.weekday() == i) for i in range(7)},
        # Month one-hot (April=4 … October=10, clamped for edge cases)
        **{f"month_{m}": float(max(4, min(10, d.month)) == m) for m in range(4, 11)},
    }

    return np.array([row[col] for col in FEATURE_COLS], dtype=np.float32)


def _apply_scaler(X: np.ndarray, meta: dict) -> np.ndarray:
    if meta.get("scaler_mean") is None:
        return X
    X = X.copy()
    binary_start = FEATURE_COLS.index("home_advantage")
    X[:, :binary_start] = (
        (X[:, :binary_start] - meta["scaler_mean"])
        / np.clip(meta["scaler_std"], 1e-6, None)
    )
    return X


# ── inference ─────────────────────────────────────────────────────────────────

def run_inference(
    feature_rows: np.ndarray,
    model:        torch.nn.Module,
    meta:         dict,
    device:       torch.device,
) -> np.ndarray:
    """
    Run the model on a (N, 41) feature array and return calibrated
    win probabilities as a (N,) numpy array.
    """
    X = _apply_scaler(feature_rows, meta)
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(tensor)
        probs  = torch.sigmoid(logits).cpu().numpy().ravel()

    calibrator = meta.get("calibrator")
    if calibrator is not None:
        probs = calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]

    return probs.clip(0.0, 1.0)


def _confidence(prob: float) -> str:
    m = abs(prob - 0.5)
    if m < 0.04: return "toss-up"
    if m < 0.08: return "slight edge"
    if m < 0.13: return "moderate edge"
    if m < 0.18: return "clear edge"
    return "strong edge"


def _print_result(home: str, away: str, date: str, prob: float) -> None:
    bar_len   = 40
    home_fill = round(prob * bar_len)
    away_fill = bar_len - home_fill
    bar       = f"[{'█' * home_fill}{'░' * away_fill}]"

    print(f"\n  {'─'*50}")
    print(f"  {away:>4}  @  {home:<4}     {date}")
    print(f"  {'─'*50}")
    print(f"  Home win probability : {prob*100:5.1f}%")
    print(f"  Away win probability : {(1-prob)*100:5.1f}%")
    print(f"  {bar}  {_confidence(prob)}")
    print(f"  {'─'*50}")


# ── mode 1: lookup from feature matrix ───────────────────────────────────────

def predict_from_features(
    features_path: Path,
    model:         torch.nn.Module,
    meta:          dict,
    device:        torch.device,
    date:          str | None = None,
    home:          str | None = None,
    away:          str | None = None,
) -> None:
    """
    Look up game(s) in the pre-built feature matrix and predict.

    If --date is given without --home/--away, predicts every game on that date.
    If all three are given, predicts that specific matchup.
    """
    if not features_path.exists():
        print(f"Feature matrix not found: {features_path}")
        print("Run: python src/features.py  to build it first.")
        sys.exit(1)

    df = pd.read_csv(features_path, parse_dates=["game_date"])

    mask = pd.Series([True] * len(df))
    if date:
        mask &= df["game_date"].dt.strftime("%Y-%m-%d") == date
    if home:
        mask &= df["home_team"].str.upper() == home.upper()
    if away:
        mask &= df["away_team"].str.upper() == away.upper()

    subset = df[mask].copy()

    if len(subset) == 0:
        print(f"\nNo games found matching your filter.")
        print(f"  date={date}  home={home}  away={away}")
        print(f"\nAvailable dates: {df['game_date'].dt.date.min()} → "
              f"{df['game_date'].dt.date.max()}")
        sys.exit(1)

    print(f"\nFound {len(subset)} game(s) matching filter.")

    X = subset[FEATURE_COLS].values.astype(np.float32)
    probs = run_inference(X, model, meta, device)

    for (_, row), prob in zip(subset.iterrows(), probs):
        _print_result(
            home  = row["home_team"],
            away  = row["away_team"],
            date  = row["game_date"].strftime("%Y-%m-%d"),
            prob  = float(prob),
        )

    if len(subset) > 1:
        print(f"\n  Average home win prob across {len(subset)} games: "
              f"{probs.mean()*100:.1f}%\n")


# ── mode 2: manual stat entry ─────────────────────────────────────────────────

def predict_manual(
    args:   argparse.Namespace,
    model:  torch.nn.Module,
    meta:   dict,
    device: torch.device,
) -> None:
    """
    Build a feature vector from command-line stats and predict.
    Only the four core stats per team are required; everything
    else defaults to league average.
    """
    required = ["home", "away", "date",
                "h_win_rate", "h_run_rate", "h_ra_rate",
                "a_win_rate", "a_run_rate", "a_ra_rate"]
    missing = [r for r in required if getattr(args, r, None) is None]
    if missing:
        print(f"\nMissing required arguments for --manual mode: {missing}")
        print("Required: --home --away --date "
              "--h_win_rate --h_run_rate --h_ra_rate "
              "--a_win_rate --a_run_rate --a_ra_rate")
        sys.exit(1)

    stats = {
        "h_win_rate":         args.h_win_rate,
        "h_run_rate":         args.h_run_rate,
        "h_ra_rate":          args.h_ra_rate,
        "h_run_diff_rate":    getattr(args, "h_run_diff_rate", None)
                              or (args.h_run_rate - args.h_ra_rate),
        "h_scoring_var":      getattr(args, "h_scoring_var",      None),
        "h_last3_win_rate":   getattr(args, "h_last3_win_rate",   None),
        "h_quality_start_rt": getattr(args, "h_quality_start_rt", None),
        "h_ops":              getattr(args, "h_ops",              None),
        "h_streak":           getattr(args, "h_streak",           None),
        "a_win_rate":         args.a_win_rate,
        "a_run_rate":         args.a_run_rate,
        "a_ra_rate":          args.a_ra_rate,
        "a_run_diff_rate":    getattr(args, "a_run_diff_rate", None)
                              or (args.a_run_rate - args.a_ra_rate),
        "a_scoring_var":      getattr(args, "a_scoring_var",      None),
        "a_last3_win_rate":   getattr(args, "a_last3_win_rate",   None),
        "a_quality_start_rt": getattr(args, "a_quality_start_rt", None),
        "a_ops":              getattr(args, "a_ops",              None),
        "a_streak":           getattr(args, "a_streak",           None),
    }
    # Strip Nones — _build_feature_row handles defaults
    stats = {k: v for k, v in stats.items() if v is not None}

    X     = _build_feature_row(stats, args.date)[None, :]  # shape (1, 41)
    probs = run_inference(X, model, meta, device)

    _print_result(
        home = args.home.upper(),
        away = args.away.upper(),
        date = args.date,
        prob = float(probs[0]),
    )

    cal_note = "(calibrated)" if meta.get("calibrator") else "(uncalibrated)"
    print(f"  Model: {meta['arch']}  epoch {meta['epoch']}  {cal_note}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Predict MLB home-team win probability for future games.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Common
    p.add_argument("--checkpoint", default=str(CKPT_DIR / "best_mlp.pt"),
                   help="Path to the .pt checkpoint saved by train.py")
    p.add_argument("--home", help="Home team abbreviation (e.g. NYY)")
    p.add_argument("--away", help="Away team abbreviation (e.g. BOS)")
    p.add_argument("--date", help="Game date: YYYY-MM-DD")

    # Mode selection
    p.add_argument("--manual", action="store_true",
                   help="Supply stats manually instead of looking up from feature matrix")

    # Feature matrix path (mode 1)
    p.add_argument("--features", default=str(FEAT_DIR / "features_w10.csv"),
                   help="Path to feature matrix CSV (built by features.py)")

    # Manual stat entry (mode 2) — core required
    p.add_argument("--h_win_rate",  type=float, help="Home win rate  [0–1]")
    p.add_argument("--h_run_rate",  type=float, help="Home runs scored per game")
    p.add_argument("--h_ra_rate",   type=float, help="Home runs allowed per game")
    p.add_argument("--a_win_rate",  type=float, help="Away win rate  [0–1]")
    p.add_argument("--a_run_rate",  type=float, help="Away runs scored per game")
    p.add_argument("--a_ra_rate",   type=float, help="Away runs allowed per game")

    # Manual stat entry — optional (all default to league average)
    p.add_argument("--h_run_diff_rate",    type=float)
    p.add_argument("--h_scoring_var",      type=float)
    p.add_argument("--h_last3_win_rate",   type=float)
    p.add_argument("--h_quality_start_rt", type=float)
    p.add_argument("--h_ops",              type=float)
    p.add_argument("--h_streak",           type=float)
    p.add_argument("--a_run_diff_rate",    type=float)
    p.add_argument("--a_scoring_var",      type=float)
    p.add_argument("--a_last3_win_rate",   type=float)
    p.add_argument("--a_quality_start_rt", type=float)
    p.add_argument("--a_ops",              type=float)
    p.add_argument("--a_streak",           type=float)

    return p.parse_args()


def main() -> None:
    args   = _parse_args()
    device = torch.device("cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    model, meta = load_checkpoint(args.checkpoint, device=device)

    if args.manual:
        predict_manual(args, model, meta, device)
    else:
        if not any([args.date, args.home, args.away]):
            print("\nProvide at least one of --date, --home, --away to filter games.")
            print("Or use --manual to supply stats directly.")
            sys.exit(1)
        predict_from_features(
            features_path = Path(args.features),
            model         = model,
            meta          = meta,
            device        = device,
            date          = args.date,
            home          = args.home,
            away          = args.away,
        )


if __name__ == "__main__":
    main()