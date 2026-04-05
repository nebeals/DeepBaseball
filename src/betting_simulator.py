"""
betting_simulator.py
--------------------
Simulates placing bets on the VALUE BET recommendations from the odds comparison
and tracks results using actual game outcomes.

Usage:
    python src/betting_simulator.py --start_date 2024-07-01 --end_date 2024-07-31
    python src/betting_simulator.py --date 2024-07-04 --bankroll 1000 --unit 100

The simulator:
  1. Reads odds comparison reports for specified date(s)
  2. Places artificial bets on VALUE BET recommendations
  3. Fetches actual game results from the game logs
  4. Calculates wins/losses based on the moneyline odds
  5. Tracks bankroll, ROI, and other metrics over time
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"
ODDS_DIR = REPORTS_DIR / "odds"
SIMULATION_DIR = REPORTS_DIR / "simulation"
DATA_DIR = ROOT / "data" / "raw"

for d in (REPORTS_DIR, ODDS_DIR, SIMULATION_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ── configuration ────────────────────────────────────────────────────────────

DEFAULT_BANKROLL = 1000.0  # Starting bankroll in dollars
DEFAULT_UNIT = 100.0       # Flat bet size per wager


# ── helpers ───────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def _step(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def american_to_decimal(ml: float) -> float:
    """Convert American moneyline to decimal odds."""
    if ml < 0:
        return 1 + (100 / abs(ml))
    else:
        return 1 + (ml / 100)


def calculate_payout(stake: float, ml: float) -> float:
    """
    Calculate payout for a winning bet.
    Returns total payout (stake + profit).
    """
    if ml < 0:
        # Favorite: e.g., -140 means bet $140 to win $100
        profit = stake * (100 / abs(ml))
    else:
        # Underdog: e.g., +120 means bet $100 to win $120
        profit = stake * (ml / 100)
    return stake + profit


# ── data loading ─────────────────────────────────────────────────────────────

def load_game_results(target_date: date) -> pd.DataFrame | None:
    """
    Load game results for a specific date from the game logs.
    Returns DataFrame with columns: home_team, away_team, home_win
    """
    game_logs_path = DATA_DIR / "game_logs_all.csv"
    
    if not game_logs_path.exists():
        return None
    
    try:
        df = pd.read_csv(game_logs_path, parse_dates=["game_date"])
        
        # Filter to the target date
        date_str = target_date.strftime("%Y-%m-%d")
        day_games = df[df["game_date"].dt.strftime("%Y-%m-%d") == date_str].copy()
        
        if day_games.empty:
            return None
        
        # Return relevant columns
        return day_games[["home_team", "away_team", "home_win"]].copy()
        
    except Exception as e:
        _log(f"Error loading game results: {e}")
        return None


def load_odds_comparison(target_date: date) -> pd.DataFrame | None:
    """
    Load odds comparison report for a specific date.
    Returns DataFrame with betting recommendations.
    """
    odds_csv = ODDS_DIR / f"odds_{target_date.strftime('%Y-%m-%d')}.csv"
    
    if not odds_csv.exists():
        return None
    
    try:
        return pd.read_csv(odds_csv)
    except Exception as e:
        _log(f"Error loading odds comparison: {e}")
        return None


# ── bet simulation ───────────────────────────────────────────────────────────

class Bet:
    """Represents a single simulated bet."""
    
    def __init__(
        self,
        date: date,
        home_team: str,
        away_team: str,
        bet_team: str,
        bet_side: str,  # "home" or "away"
        stake: float,
        moneyline: float,
    ):
        self.date = date
        self.home_team = home_team
        self.away_team = away_team
        self.bet_team = bet_team
        self.bet_side = bet_side
        self.stake = stake
        self.moneyline = moneyline
        
        # These get filled in after results are known
        self.won: bool | None = None
        self.payout: float = 0.0
        self.profit: float = -stake  # Default to loss
    
    def resolve(self, home_win: bool) -> float:
        """
        Resolve the bet based on game result.
        Returns profit/loss amount.
        """
        self.won = (self.bet_side == "home" and home_win) or \
                   (self.bet_side == "away" and not home_win)
        
        if self.won:
            self.payout = calculate_payout(self.stake, self.moneyline)
            self.profit = self.payout - self.stake
        else:
            self.payout = 0.0
            self.profit = -self.stake
        
        return self.profit
    
    def to_dict(self) -> dict:
        """Convert bet to dictionary for DataFrame export."""
        return {
            "date": self.date.strftime("%Y-%m-%d"),
            "matchup": f"{self.away_team} @ {self.home_team}",
            "bet_team": self.bet_team,
            "bet_side": self.bet_side,
            "stake": self.stake,
            "moneyline": self.moneyline,
            "result": "WIN" if self.won else ("LOSS" if self.won is False else "PENDING"),
            "payout": round(self.payout, 2) if self.won else 0.0,
            "profit": round(self.profit, 2),
        }


def parse_value_bet(side_str: str) -> tuple[str, str] | None:
    """
    Parse the SIDE string from odds comparison.
    Input: "MIL" or "SDP" (team abbreviation)
    Returns: (team_abbr, side) or None if no value bet
    """
    if not side_str or side_str == "—":
        return None
    
    # SIDE column contains just the team abbreviation
    team = side_str.strip()
    return (team, "unknown")


def simulate_day_bets(
    target_date: date,
    bankroll: float,
    unit_size: float,
) -> list[Bet]:
    """
    Simulate bets for a single day.
    Returns list of Bet objects (unresolved).
    """
    # Load odds comparison
    odds_df = load_odds_comparison(target_date)
    if odds_df is None:
        _log(f"No odds comparison found for {target_date}")
        return []
    
    bets: list[Bet] = []
    
    for _, row in odds_df.iterrows():
        # Check if there's a value bet (SIDE column)
        side = row.get("SIDE", "")
        if not side:
            continue
        
        bet_team = side
        home_team = row["home_team"]
        away_team = row["away_team"]
        
        # Determine if betting on home or away
        if bet_team == home_team:
            bet_side = "home"
            moneyline = row.get("home_ml", 0)
        elif bet_team == away_team:
            bet_side = "away"
            moneyline = row.get("away_ml", 0)
        else:
            continue
        
        # Skip if no valid moneyline
        if moneyline == 0 or pd.isna(moneyline):
            continue
        
        bet = Bet(
            date=target_date,
            home_team=home_team,
            away_team=away_team,
            bet_team=bet_team,
            bet_side=bet_side,
            stake=unit_size,
            moneyline=float(moneyline),
        )
        bets.append(bet)
    
    return bets


def resolve_bets(bets: list[Bet], game_results: pd.DataFrame) -> list[Bet]:
    """
    Resolve bets against actual game results.
    Returns list of resolved Bet objects.
    """
    resolved: list[Bet] = []
    
    for bet in bets:
        # Find the matching game (check both home/away orientations)
        match = game_results[
            ((game_results["home_team"] == bet.home_team) &
             (game_results["away_team"] == bet.away_team)) |
            ((game_results["home_team"] == bet.away_team) &
             (game_results["away_team"] == bet.home_team))
        ]
        
        if match.empty:
            _log(f"  Warning: No result found for {bet.away_team} @ {bet.home_team}")
            continue
        
        # Determine if the bet's home team matches the game result's home team
        result_row = match.iloc[0]
        if result_row["home_team"] == bet.home_team:
            home_win = bool(result_row["home_win"])
        else:
            # Teams are swapped, so invert the result
            home_win = not bool(result_row["home_win"])
        
        bet.resolve(home_win)
        resolved.append(bet)
    
    return resolved


# ── bankroll tracking ──────────────────────────────────────────────────────────

class BankrollTracker:
    """Tracks bankroll over time with daily P&L."""
    
    def __init__(self, initial_bankroll: float):
        self.initial = initial_bankroll
        self.current = initial_bankroll
        self.history: list[dict] = []
    
    def record_day(self, date: date, bets: list[Bet]) -> float:
        """Record day's results and return daily profit/loss."""
        day_profit = sum(bet.profit for bet in bets)
        wins = sum(1 for bet in bets if bet.won)
        losses = sum(1 for bet in bets if bet.won is False)
        
        self.current += day_profit
        
        self.history.append({
            "date": date.strftime("%Y-%m-%d"),
            "bets_placed": len(bets),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(bets) if bets else 0.0,
            "day_profit": round(day_profit, 2),
            "bankroll": round(self.current, 2),
        })
        
        return day_profit
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        if not self.history:
            return {}
        
        total_bets = sum(h["bets_placed"] for h in self.history)
        total_wins = sum(h["wins"] for h in self.history)
        total_losses = sum(h["losses"] for h in self.history)
        total_profit = self.current - self.initial
        
        return {
            "initial_bankroll": self.initial,
            "final_bankroll": round(self.current, 2),
            "total_profit": round(total_profit, 2),
            "roi_pct": round((total_profit / self.initial) * 100, 2),
            "total_bets": total_bets,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "win_rate": round(total_wins / total_bets * 100, 1) if total_bets else 0.0,
            "days_tracked": len(self.history),
        }


# ── reporting ─────────────────────────────────────────────────────────────────

def save_simulation_results(
    all_bets: list[Bet],
    tracker: BankrollTracker,
    start_date: date,
    end_date: date,
) -> Path:
    """Save simulation results to reports directory."""
    date_range = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
    
    # Individual bets CSV
    bets_df = pd.DataFrame([bet.to_dict() for bet in all_bets])
    bets_path = SIMULATION_DIR / f"simulation_bets_{date_range}.csv"
    bets_df.to_csv(bets_path, index=False)
    
    # Daily bankroll history CSV
    history_df = pd.DataFrame(tracker.history)
    history_path = SIMULATION_DIR / f"simulation_bankroll_{date_range}.csv"
    history_df.to_csv(history_path, index=False)
    
    # Summary text report
    summary = tracker.get_summary()
    lines = [
        "=" * 70,
        f"  BETTING SIMULATION RESULTS",
        f"  Period: {start_date} to {end_date}",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 70,
        "",
        "  SUMMARY",
        "  ───────",
        f"  Initial Bankroll:    ${summary.get('initial_bankroll', 0):,.2f}",
        f"  Final Bankroll:      ${summary.get('final_bankroll', 0):,.2f}",
        f"  Total Profit/Loss:   ${summary.get('total_profit', 0):,.2f}",
        f"  ROI:                 {summary.get('roi_pct', 0):+.2f}%",
        "",
        "  BETTING PERFORMANCE",
        "  ───────────────────",
        f"  Total Bets Placed:   {summary.get('total_bets', 0)}",
        f"  Wins:                {summary.get('total_wins', 0)}",
        f"  Losses:              {summary.get('total_losses', 0)}",
        f"  Win Rate:            {summary.get('win_rate', 0):.1f}%",
        f"  Days Simulated:      {summary.get('days_tracked', 0)}",
        "",
    ]
    
    if all_bets:
        # Show recent bets
        lines += [
            "  RECENT BETS (last 10)",
            "  ─────────────────────",
            f"  {'Date':<12} {'Matchup':<20} {'Bet':<10} {'ML':<8} {'Result':<8} {'P/L':>10}",
            "  " + "─" * 68,
        ]
        
        for bet in all_bets[-10:]:
            result = bet.to_dict()
            lines.append(
                f"  {result['date']:<12} {result['matchup']:<20} "
                f"{result['bet_team']:<10} {result['moneyline']:>+7.0f} "
                f"{result['result']:<8} ${result['profit']:>+9.2f}"
            )
        
        lines.append("")
    
    lines += [
        "  FILES SAVED",
        "  ───────────",
        f"  Bets:    {bets_path}",
        f"  Bankroll: {history_path}",
        "",
        "=" * 70,
    ]
    
    report_path = SIMULATION_DIR / f"simulation_report_{date_range}.txt"
    report_path.write_text("\n".join(lines))
    
    return report_path


# ── main simulation ───────────────────────────────────────────────────────────

def run_simulation(
    start_date: date,
    end_date: date,
    initial_bankroll: float,
    unit_size: float,
) -> None:
    """Run betting simulation over date range."""
    _step("Betting Simulator")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Initial Bankroll: ${initial_bankroll:,.2f}")
    print(f"  Unit Size: ${unit_size:,.2f}")
    print(f"{'─'*60}")
    
    tracker = BankrollTracker(initial_bankroll)
    all_bets: list[Bet] = []
    
    current = start_date
    while current <= end_date:
        # Simulate bets for this day
        day_bets = simulate_day_bets(current, tracker.current, unit_size)
        
        if day_bets:
            _log(f"{current}: Placed {len(day_bets)} bet(s)")
            
            # Load game results and resolve bets
            results = load_game_results(current)
            if results is not None:
                resolved = resolve_bets(day_bets, results)
                day_pnl = tracker.record_day(current, resolved)
                all_bets.extend(resolved)
                
                win_count = sum(1 for b in resolved if b.won)
                _log(f"  Results: {win_count}/{len(resolved)} wins, P&L: ${day_pnl:+.2f}")
            else:
                # Still track bets even without results (pending status)
                day_pnl = tracker.record_day(current, day_bets)
                all_bets.extend(day_bets)
                _log(f"  Warning: No game results found for {current} ({len(day_bets)} bets pending)")
        
        current += timedelta(days=1)
    
    # Generate report
    report_path = save_simulation_results(all_bets, tracker, start_date, end_date)
    
    # Print summary
    print(f"\n{'═'*60}")
    print(report_path.read_text())


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Simulate betting on VALUE BET recommendations."
    )
    p.add_argument(
        "--start_date",
        default=None,
        help="Start date (YYYY-MM-DD). Defaults to yesterday.",
    )
    p.add_argument(
        "--end_date",
        default=None,
        help="End date (YYYY-MM-DD). Defaults to start_date.",
    )
    p.add_argument(
        "--date",
        default=None,
        help="Single date to simulate (YYYY-MM-DD). Overrides start/end dates.",
    )
    p.add_argument(
        "--bankroll",
        type=float,
        default=DEFAULT_BANKROLL,
        help=f"Initial bankroll in dollars (default: ${DEFAULT_BANKROLL:,.0f})",
    )
    p.add_argument(
        "--unit",
        type=float,
        default=DEFAULT_UNIT,
        help=f"Flat bet size in dollars (default: ${DEFAULT_UNIT:,.0f})",
    )
    
    args = p.parse_args()
    
    # Determine date range
    if args.date:
        start = end = datetime.strptime(args.date, "%Y-%m-%d").date()
    elif args.start_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else start
    else:
        # Default to yesterday (to ensure results are available)
        yesterday = date.today() - timedelta(days=1)
        start = end = yesterday
    
    # Validate
    if end < start:
        print("ERROR: end_date must be after start_date")
        sys.exit(1)
    
    if args.unit > args.bankroll:
        print("ERROR: unit size cannot exceed bankroll")
        sys.exit(1)
    
    run_simulation(start, end, args.bankroll, args.unit)


if __name__ == "__main__":
    main()
