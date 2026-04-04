# DeepBaseball: MLB Win Probability & Betting Edge Detection

A PyTorch-based machine learning system that predicts MLB game outcomes and identifies betting edges by comparing model probabilities against live sportsbook odds.

---

## Overview

This project provides a complete pipeline for:
1. **Data Collection**: Fetching historical game logs and statistics
2. **Feature Engineering**: Building rolling team performance metrics
3. **Model Training**: Neural networks (ResNet, MLP) with probability calibration
4. **Daily Predictions**: Automated win probability forecasts for today's games
5. **Odds Comparison**: Real-time comparison against live betting markets
6. **Betting Simulation**: Backtesting VALUE BET recommendations with historical results

---

## Project Structure

```
DeepBaseball/
├── src/
│   ├── data_collection.py      # Download historical data from Baseball Reference
│   ├── features.py             # Build rolling feature matrix
│   ├── train.py                # Model training with calibration
│   ├── predict_game.py         # Single-game inference
│   ├── daily_pipeline.py       # Morning pipeline: data → features → predictions
│   ├── odds_comparison.py      # Compare predictions to live odds
│   └── betting_simulator.py    # Simulate bets on VALUE BET recommendations
├── data/
│   ├── raw/                    # Game logs, batting/pitching stats (git-ignored)
│   └── features/               # Feature matrices (git-ignored)
├── reports/                    # Daily predictions and odds comparisons (git-ignored)
├── cache/                      # Cached API responses (git-ignored)
├── checkpoints/                # Saved model weights (git-ignored)
├── notebooks/                  # Jupyter notebooks for exploration
├── requirements.txt
└── README.md                   # This file
```

---

## Installation

### Prerequisites

- Python 3.10+
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Step 1: Clone and Setup Environment

```bash
cd /path/to/DeepBaseball
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies include:
- `pytorch` — Neural network framework
- `pandas`, `numpy` — Data manipulation
- `pybaseball` — MLB data from Baseball Reference/FanGraphs
- `requests` — API calls to The Odds API
- `scikit-learn` — Calibration and metrics

### Step 3: Get API Keys

#### The Odds API (Free tier: 500 requests/month)

1. Sign up at [the-odds-api.com](https://the-odds-api.com)
2. Get your API key from the dashboard
3. Set as environment variable:

```bash
export ODDS_API_KEY="your_key_here"
# Add to ~/.bashrc or ~/.zshrc to persist
```

Or pass directly on command line: `--api_key YOUR_KEY`

---

## First-Time Setup

### 1. Collect Historical Data

Download 8+ seasons of game logs (needed for feature engineering):

```bash
python src/data_collection.py --start_year 2015 --end_year 2023
```

This creates:
- `data/raw/game_logs_all.csv` — All games with scores and winners
- `data/raw/team_batting_all.csv` — Season batting stats
- `data/raw/team_pitching_all.csv` — Season pitching stats

**Time required**: ~10-15 minutes (uses pybaseball cache)

### 2. Train the Model

```bash
python src/train.py
```

Training details:
- Default: ResNet architecture with 10-game rolling window
- Automatically handles train/val/test split (chronological)
- Applies Platt scaling or isotonic regression for calibration
- Saves best checkpoint to `checkpoints/best_resnet.pt`

**Optional flags**:
- `--window 15` — Use 15-game rolling window
- `--epochs 50` — Train for more epochs
- `--arch mlp` — Use MLP instead of ResNet

**Time required**: ~5-10 minutes on CPU, ~1-2 minutes on GPU

---

## Daily Operation

### Morning Pipeline (Before First Pitch)

Run these two commands each morning to get fresh predictions and odds comparison:

```bash
# Step 1: Update game logs, rebuild features, predict today's games
python src/daily_pipeline.py

# Step 2: Compare predictions to live betting odds
python src/odds_comparison.py
```

**What happens**:
1. `daily_pipeline.py` pulls yesterday's completed games, updates rolling stats, generates predictions
2. `odds_comparison.py` fetches live moneylines, finds edges, recommends value bets

**Output files**:
- `reports/predictions_YYYY-MM-DD.txt/.csv` — Model win probabilities
- `reports/odds_YYYY-MM-DD.txt/.csv` — Odds comparison with edges

### Understanding the Odds Report

Sample output:
```
MATCHUP          MOD % (A/H)   MKT % (A/H)   MOD ML   MKT ML   EDGE %   SIDE   STAKE %   BOOK
KCR @ MIL        45.0%/55.0%   41.7%/58.3%   -122    -140     -3.3%   KCR     12.5%   betmgm
BOS @ NYY        40.0%/60.0%   47.9%/52.1%   -150    -110     +7.9%   NYY     25.0%   draftkings
```

**Columns**:
- `MOD % (A/H)` — Model's away% / home% win probability
- `MKT % (A/H)` — Market implied away% / home% probability (vig removed)
- `MOD ML / MKT ML` — Moneyline equivalents (HOME team perspective)
- `EDGE %` — Absolute edge magnitude |model_prob - market_prob|
- `SIDE` — Team with positive edge (value bet recommendation)
- `STAKE %` — Kelly criterion allocation percentage
- `BOOK` — Sportsbook offering the best edge

### Automation (Optional)

Schedule daily runs with crontab (macOS/Linux):

```bash
# Edit crontab
crontab -e

# Add this line to run at 9:00 AM daily
0 9 * * * cd /path/to/DeepBaseball && python src/daily_pipeline.py && python src/odds_comparison.py
```

Windows: Use Task Scheduler to run the commands at your preferred time.

---

## Manual Pipeline Walkthrough

For learning or debugging, you can run each step manually:

### Step 1: Data Collection

```bash
# Pull specific season
python src/data_collection.py --start_year 2024 --end_year 2024

# Exclude 2020 COVID season
python src/data_collection.py --start_year 2015 --end_year 2023 --skip_covid
```

### Step 2: Feature Engineering

```bash
# Rebuild feature matrix with custom rolling window
python src/daily_pipeline.py --window 20 --dry_run
```

### Step 3: Training

```bash
# Train with custom architecture
python src/train.py --arch resnet --window 15 --epochs 100

# Use specific checkpoint for inference
python src/daily_pipeline.py --checkpoint checkpoints/best_mlp.pt
```

### Step 4: Single Game Prediction

```bash
# Predict specific matchup
python src/predict_game.py --date 2024-07-04 --home NYY --away BOS
```

### Step 5: Odds Comparison with Options

```bash
# Only show strong edges (≥8%)
python src/odds_comparison.py --min_edge 0.08

# Compare specific historical date
python src/odds_comparison.py --date 2024-07-04

# Include vig in market probabilities (default: vig removed)
python src/odds_comparison.py --remove_vig false

# Force refresh (ignore cache)
python src/odds_comparison.py --force_refresh
```

---

## Betting Simulator

After games complete, simulate how the VALUE BET recommendations performed:

```bash
# Simulate single day
python src/betting_simulator.py --date 2024-07-04

# Simulate date range with custom bankroll
python src/betting_simulator.py \
    --start_date 2024-07-01 \
    --end_date 2024-07-31 \
    --bankroll 5000 \
    --unit 250
```

**How it works**:
1. Reads `reports/odds_YYYY-MM-DD.csv` for VALUE BET recommendations
2. Places flat unit bets on each value bet
3. Fetches actual game results from `data/raw/game_logs_all.csv`
4. Calculates win/loss based on moneyline payouts
5. Tracks bankroll, ROI, win rate over time

**Output files**:
- `reports/simulation_report_YYYYMMDD-YYYYMMDD.txt` — Summary with ROI
- `reports/simulation_bets_YYYYMMDD-YYYYMMDD.csv` — Individual bet details
- `reports/simulation_bankroll_YYYYMMDD-YYYYMMDD.csv` — Daily bankroll tracking

---

## Key Features

### No Data Leakage
All features are computed strictly from data available **before** the game's first pitch. Rolling windows and lagged stats maintain temporal ordering.

### Probability Calibration
Models use Platt scaling or isotonic regression to ensure well-calibrated probabilities (predicted 60% should win 60% of the time).

### Kelly Criterion Staking
Odds comparison uses Kelly formula to calculate optimal allocation percentages based on edge size and odds.

### Best Edge Selection
Instead of averaging bookmaker odds, the system selects the single sportsbook offering the largest magnitude edge for each game.

### Caching
Odds API responses are cached for 24 hours to minimize API usage (500 requests/month limit on free tier).

---

## Troubleshooting

### "Checkpoint not found" error
```bash
# Train model first
python src/train.py
```

### "No Odds API key found" error
```bash
# Set environment variable
export ODDS_API_KEY="your_key_here"

# Or pass on command line
python src/odds_comparison.py --api_key YOUR_KEY
```

### "No predictions file found" error
```bash
# Run daily pipeline first
python src/daily_pipeline.py
```

### "No game results found" (simulator)
Games must be completed and in `game_logs_all.csv`. Run `data_collection.py` to update.

### Empty odds report
- Check if MLB season is active (off-season = no games)
- Try `--force_refresh` to bypass cache
- Verify API key has remaining quota

---

## Model Architecture

**ResNet (default)**:
- Residual connections for gradient flow
- 2 hidden layers with skip connections
- Dropout regularization
- Sigmoid output with calibration

**MLP (alternative)**:
- Simple feedforward network
- Faster training, slightly lower capacity
- Good baseline comparison

Features used (20 total):
- Rolling win rate, run rate, runs allowed
- Run differential, scoring variance
- Last-3-game performance
- Quality start rate, OPS, streak
- Home/away differential features

---

## Data Sources

| Source | Data | Library |
|--------|------|---------|
| Baseball Reference | Game logs, schedules, scores | pybaseball |
| FanGraphs | Team batting & pitching stats | pybaseball |
| The Odds API | Live moneylines | requests |

---

## Output Files Reference

### Data Files
- `data/raw/game_logs_all.csv` — Historical games with results
- `data/features/features_w15.csv` — Feature matrix (rolling window)

### Report Files
- `reports/predictions_YYYY-MM-DD.txt/.csv` — Daily model predictions
- `reports/odds_YYYY-MM-DD.txt/.csv` — Odds comparison with edges
- `reports/simulation_*_YYYYMMDD-YYYYMMDD.*` — Betting simulation results

### Model Files
- `checkpoints/best_resnet.pt` — Best ResNet checkpoint
- `checkpoints/best_mlp.pt` — Best MLP checkpoint

### Cache Files
- `cache/odds_raw_YYYY-MM-DD.json` — Cached API responses

---

## Development

### Running Tests

```bash
# Basic validation
python -c "from src.train import load_checkpoint; print('Imports OK')"

# Test data pipeline
python src/daily_pipeline.py --dry_run --date 2024-07-04
```

### Adding New Features

Edit `src/features.py`:
1. Add column name to `FEATURE_COLUMNS`
2. Implement calculation in feature building functions
3. Ensure no data leakage (only use past data)

### Customizing Reports

Edit column formatting in:
- `src/odds_comparison.py` — `save_comparison()` function
- `src/daily_pipeline.py` — `save_reports()` function

---



