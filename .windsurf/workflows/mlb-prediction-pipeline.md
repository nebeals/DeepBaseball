---
description: MLB Daily Prediction Pipeline - from data collection to odds comparison
---

# MLB Daily Prediction Pipeline

This workflow runs the complete MLB prediction pipeline: from data collection through model inference to odds comparison.

## Overview

The pipeline consists of three main scripts:

1. **daily_pipeline.py** - Generates win probability predictions for today's games
2. **odds_comparison.py** - Compares model predictions against live betting odds
3. **betting_simulator.py** - Simulates bets on VALUE BET recommendations and tracks results

## Prerequisites

- Model checkpoint must exist at `checkpoints/best_resnet.pt` (or another checkpoint)
- Predictions file for today's date at `reports/predictions_YYYY-MM-DD.csv`
- Odds API key (for odds comparison step)
- Historical game logs at `data/raw/game_logs_all.csv` (for simulator)

## Step 1: Generate Today's Predictions

Run the daily pipeline to fetch game logs, rebuild features, and generate predictions:

```bash
python src/daily_pipeline.py
```

Optional flags:
- `--checkpoint path/to/model.pt` - Use a specific model checkpoint
- `--window 15` - Change rolling window size (default: 15 games)
- `--dry_run` - Skip data pull, use cached CSVs
- `--date 2024-07-04` - Override today's date (for testing historical dates)

Expected output:
- `reports/predictions_YYYY-MM-DD.txt` - Human-readable report
- `reports/predictions_YYYY-MM-DD.csv` - Machine-readable CSV

## Step 2: Compare Against Live Odds

After generating predictions, compare them to live betting odds:

```bash
# Requires ODDS_API_KEY environment variable
export ODDS_API_KEY="your_key_here"
python src/odds_comparison.py
```

Or pass the key directly:

```bash
python src/odds_comparison.py --api_key YOUR_KEY_HERE
```

Optional flags:
- `--date 2024-07-04` - Compare specific date's predictions
- `--min_edge 0.05` - Only show games with ≥5% edge
- `--remove_vig` - Remove vig from market probabilities (default: include vig)
- `--force_refresh` - Force refetch odds from API (ignore cache)

Expected output:
- `reports/odds_YYYY-MM-DD.txt` - Human-readable odds comparison
- `reports/odds_YYYY-MM-DD.csv` - Machine-readable comparison data

## Step 3: Simulate Betting (Optional)

After the games complete, run the betting simulator to see how the VALUE BET recommendations performed:

```bash
python src/betting_simulator.py --date 2024-07-04
```

Optional flags:
- `--start_date 2024-07-01 --end_date 2024-07-31` - Simulate a date range
- `--bankroll 1000` - Starting bankroll (default: $1000)
- `--unit 100` - Flat bet size (default: $100)

Expected output:
- `reports/simulation_report_YYYYMMDD-YYYYMMDD.txt` - Summary report
- `reports/simulation_bets_YYYYMMDD-YYYYMMDD.csv` - Individual bet results
- `reports/simulation_bankroll_YYYYMMDD-YYYYMMDD.csv` - Daily bankroll tracking

**How it works:**
1. Reads the odds comparison CSV for VALUE BET recommendations
2. Places artificial flat bets on each value bet
3. Fetches actual game results from `game_logs_all.csv`
4. Calculates win/loss and profit/loss for each bet
5. Tracks bankroll over time with ROI metrics

## Full Daily Workflow

For a complete daily run:

```bash
# 1. Generate predictions
python src/daily_pipeline.py

# 2. Compare to live odds
export ODDS_API_KEY="your_key_here"
python src/odds_comparison.py
```

## Scheduling (Optional)

To run automatically each morning before games:

**macOS/Linux (crontab):**
```bash
0 9 * * * cd /path/to/DeepBaseball && python src/daily_pipeline.py && python src/odds_comparison.py
```

**Windows (Task Scheduler):**
- Create a task that runs daily at 9:00 AM
- Point to: `python.exe src\daily_pipeline.py`
- Then add second action: `python.exe src\odds_comparison.py`

## Troubleshooting

**"Checkpoint not found" error:**
- Train the model first: `python src/train.py`
- Or specify a checkpoint: `--checkpoint checkpoints/best_mlp.pt`

**"No Odds API key found" error:**
- Get a free key at https://the-odds-api.com
- Set as environment variable: `export ODDS_API_KEY="your_key"`
- Or pass on command line: `--api_key YOUR_KEY`

**"No predictions file found" error:**
- Run daily_pipeline.py first to generate predictions
- Or check the date: predictions are saved with today's date

**"No games scheduled for today" message:**
- Check if today is an MLB off-day
- Verify the date with `--date` flag for testing
- Baseball Reference data may not be updated yet (try later in the morning)

## Output Files

- `data/raw/game_logs_all.csv` - Combined historical game logs
- `data/features/features_w15.csv` - Feature matrix with rolling stats
- `reports/predictions_YYYY-MM-DD.txt` - Daily predictions report
- `reports/predictions_YYYY-MM-DD.csv` - Predictions data
- `reports/odds_YYYY-MM-DD.txt` - Odds comparison report
- `reports/odds_YYYY-MM-DD.csv` - Odds comparison data
- `cache/odds_raw_YYYY-MM-DD.json` - Cached API odds data

## Understanding the Output

**Predictions Report Columns:**
- MATCHUP: Away team @ Home team
- HOME%: Model's home team win probability
- AWAY%: Model's away team win probability  
- ML EQUIV: American moneyline equivalent of the model's probability
- CONFIDENCE: Qualitative confidence level

**Odds Comparison Report Columns:**
- MATCHUP: Away team @ Home team
- MODEL: Model away% / home% win probability
- MARKET: Sportsbook away% / home% implied probability
- MODEL ML: Model's moneyline equivalent for home team
- BOOK ML: Actual moneyline offered by books for home team
- EDGE: Model prob - market prob (home team perspective)
- VALUE BET: Which side has positive edge > 3%

## Notes

- The pipeline is designed to run once per day, ideally in the morning before games start
- Odds data is cached for 24 hours to avoid excessive API usage
- Use `--force_refresh` to get fresh odds if needed
- The model uses a 15-game rolling window by default for team statistics
