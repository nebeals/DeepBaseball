"""
api.py
------
FastAPI inference server for the MLB win probability model.

Loads a self-contained checkpoint (produced by train.py) once at startup
and serves predictions without any retraining.

Endpoints:
    GET  /health                – liveness check + model metadata
    POST /predict               – predict win probability for one game
    POST /predict/batch         – predict for multiple games at once
    GET  /feature_importance    – ranked feature sensitivity scores
    GET  /docs                  – auto-generated Swagger UI (built into FastAPI)

Quick start:
    pip install fastapi uvicorn
    python src/api.py --checkpoint checkpoints/best_mlp.pt

    # Or with uvicorn directly (recommended for production):
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

Example curl:
    curl -X POST http://localhost:8000/predict \\
      -H "Content-Type: application/json" \\
      -d '{
        "home_team": "NYY",
        "away_team": "BOS",
        "game_date": "2024-07-04",
        "h_win_rate": 0.60,
        "h_run_rate": 5.1,
        "h_ra_rate": 3.8,
        "h_run_diff_rate": 1.3,
        "a_win_rate": 0.52,
        "a_run_rate": 4.6,
        "a_ra_rate": 4.2,
        "a_run_diff_rate": 0.4
      }'
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

# ── path setup so `from model import ...` works when run as a script ──────────
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from train import FEATURE_COLS, load_checkpoint, predict as _predict

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, model_validator
except ImportError:
    print("FastAPI not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)


# ── Constants ─────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).parent.parent
CKPT_DIR  = ROOT / "checkpoints"
DEFAULT_CKPT = CKPT_DIR / "best_mlp.pt"

MLB_TEAMS = {
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE",
    "COL", "DET", "HOU", "KCR", "LAA", "LAD", "MIA", "MIL",
    "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SDP", "SFG",
    "SEA", "STL", "TBR", "TEX", "TOR", "WSN",
}

# Features that are computed internally from the inputs
# (not required from the caller)
_INTERNAL_FEATURES = {
    "home_advantage",   # always 1
    "is_covid_season",  # derived from game_date
    # dow_* and month_* derived from game_date
    *[f"dow_{i}"   for i in range(7)],
    *[f"month_{m}" for m in range(4, 11)],
    # diff_* derived from h_* and a_* pairs
    "diff_run_rate", "diff_ra_rate", "diff_run_diff_rate",
    "diff_win_rate", "diff_ops", "diff_streak", "diff_last3_win_rate",
}

# Features the caller must supply (or we use sensible defaults for)
_REQUIRED_ROLLING = [c for c in FEATURE_COLS if c not in _INTERNAL_FEATURES]


# ── Global model state (loaded once at startup) ────────────────────────────────

class _ModelState:
    model      = None
    meta: dict = {}
    device     = torch.device("cpu")
    loaded_at  = None
    ckpt_path  = None

_state = _ModelState()


def _load_model(checkpoint_path: str | Path) -> None:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Train the model first: python src/train.py"
        )
    _state.model, _state.meta = load_checkpoint(path, device=_state.device)
    _state.model.eval()
    _state.loaded_at = datetime.utcnow().isoformat() + "Z"
    _state.ckpt_path = str(path)
    print(f"Model loaded: {_state.meta['arch']} "
          f"(epoch={_state.meta['epoch']}, val_loss={_state.meta['val_loss']:.4f})")


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MLB Win Probability API",
    description=(
        "Neural network inference API for MLB home-team win probability. "
        "Supply rolling team stats for both teams and receive a calibrated "
        "win probability between 0 and 1."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ───────────────────────────────────────────────────────────

_STAT_DESC = {
    "h_run_rate":          "Home: avg runs scored per game (rolling window)",
    "h_ra_rate":           "Home: avg runs allowed per game (rolling window)",
    "h_run_diff_rate":     "Home: avg run differential per game (rolling window)",
    "h_win_rate":          "Home: win rate over rolling window  [0–1]",
    "h_scoring_var":       "Home: std-dev of runs scored (rolling window)",
    "h_last3_win_rate":    "Home: win rate over last 3 games  [0–1]",
    "h_quality_start_rt":  "Home: fraction of games allowing ≤3 runs  [0–1]",
    "h_ops":               "Home: OPS proxy (run_rate / (run_rate + ra_rate))  [0–1]",
    "h_streak":            "Home: current streak (positive=win, negative=loss)",
    "a_run_rate":          "Away: avg runs scored per game (rolling window)",
    "a_ra_rate":           "Away: avg runs allowed per game (rolling window)",
    "a_run_diff_rate":     "Away: avg run differential per game (rolling window)",
    "a_win_rate":          "Away: win rate over rolling window  [0–1]",
    "a_scoring_var":       "Away: std-dev of runs scored (rolling window)",
    "a_last3_win_rate":    "Away: win rate over last 3 games  [0–1]",
    "a_quality_start_rt":  "Away: fraction of games allowing ≤3 runs  [0–1]",
    "a_ops":               "Away: OPS proxy  [0–1]",
    "a_streak":            "Away: current streak (positive=win, negative=loss)",
}

_DEFAULTS = {
    "h_scoring_var":      2.1,
    "h_last3_win_rate":   0.5,
    "h_quality_start_rt": 0.37,
    "h_ops":              0.5,
    "h_streak":           0,
    "a_scoring_var":      2.1,
    "a_last3_win_rate":   0.5,
    "a_quality_start_rt": 0.37,
    "a_ops":              0.5,
    "a_streak":           0,
}


class GameInput(BaseModel):
    """
    Input features for a single MLB game.

    Required fields are the core rolling stats that most strongly drive
    the prediction. Optional fields have sensible league-average defaults
    and can be omitted if unavailable.
    """
    # Identity
    home_team: str = Field(..., description="Home team abbreviation (e.g. 'NYY')")
    away_team: str = Field(..., description="Away team abbreviation (e.g. 'BOS')")
    game_date: str = Field(..., description="Game date in YYYY-MM-DD format")

    # Core home stats (required)
    h_win_rate:      float = Field(..., ge=0, le=1,  description=_STAT_DESC["h_win_rate"])
    h_run_rate:      float = Field(..., ge=0,         description=_STAT_DESC["h_run_rate"])
    h_ra_rate:       float = Field(..., ge=0,         description=_STAT_DESC["h_ra_rate"])
    h_run_diff_rate: float = Field(...,               description=_STAT_DESC["h_run_diff_rate"])

    # Core away stats (required)
    a_win_rate:      float = Field(..., ge=0, le=1,  description=_STAT_DESC["a_win_rate"])
    a_run_rate:      float = Field(..., ge=0,         description=_STAT_DESC["a_run_rate"])
    a_ra_rate:       float = Field(..., ge=0,         description=_STAT_DESC["a_ra_rate"])
    a_run_diff_rate: float = Field(...,               description=_STAT_DESC["a_run_diff_rate"])

    # Optional — defaults to league average if omitted
    h_scoring_var:      Optional[float] = Field(None, ge=0,      description=_STAT_DESC["h_scoring_var"])
    h_last3_win_rate:   Optional[float] = Field(None, ge=0, le=1, description=_STAT_DESC["h_last3_win_rate"])
    h_quality_start_rt: Optional[float] = Field(None, ge=0, le=1, description=_STAT_DESC["h_quality_start_rt"])
    h_ops:              Optional[float] = Field(None, ge=0, le=1, description=_STAT_DESC["h_ops"])
    h_streak:           Optional[float] = Field(None,             description=_STAT_DESC["h_streak"])
    a_scoring_var:      Optional[float] = Field(None, ge=0,      description=_STAT_DESC["a_scoring_var"])
    a_last3_win_rate:   Optional[float] = Field(None, ge=0, le=1, description=_STAT_DESC["a_last3_win_rate"])
    a_quality_start_rt: Optional[float] = Field(None, ge=0, le=1, description=_STAT_DESC["a_quality_start_rt"])
    a_ops:              Optional[float] = Field(None, ge=0, le=1, description=_STAT_DESC["a_ops"])
    a_streak:           Optional[float] = Field(None,             description=_STAT_DESC["a_streak"])

    @model_validator(mode="after")
    def check_teams_and_date(self) -> "GameInput":
        if self.home_team.upper() not in MLB_TEAMS:
            raise ValueError(f"Unknown home_team '{self.home_team}'. Valid: {sorted(MLB_TEAMS)}")
        if self.away_team.upper() not in MLB_TEAMS:
            raise ValueError(f"Unknown away_team '{self.away_team}'. Valid: {sorted(MLB_TEAMS)}")
        if self.home_team == self.away_team:
            raise ValueError("home_team and away_team must be different.")
        try:
            datetime.strptime(self.game_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"game_date must be YYYY-MM-DD, got: '{self.game_date}'")
        return self


class BatchInput(BaseModel):
    games: list[GameInput] = Field(..., min_length=1, max_length=100,
                                   description="List of 1–100 games to predict")


class PredictionResponse(BaseModel):
    home_team:        str
    away_team:        str
    game_date:        str
    home_win_prob:    float = Field(..., description="Probability the home team wins [0–1]")
    away_win_prob:    float = Field(..., description="Probability the away team wins [0–1]")
    confidence:       str   = Field(..., description="Qualitative confidence label")
    model_arch:       str
    calibrated:       bool
    inference_ms:     float


class BatchPredictionResponse(BaseModel):
    predictions:  list[PredictionResponse]
    n_games:      int
    total_ms:     float


class HealthResponse(BaseModel):
    status:       str
    model_arch:   str
    epoch:        int
    val_loss:     float
    calibrated:   bool
    loaded_at:    str
    checkpoint:   str
    feature_cols: list[str]


# ── Feature assembly ───────────────────────────────────────────────────────────

def _game_to_feature_vector(game: GameInput) -> np.ndarray:
    """
    Convert a GameInput into the ordered FEATURE_COLS numpy array,
    filling in optional fields and computing derived features.
    """
    d = datetime.strptime(game.game_date, "%Y-%m-%d")

    # Apply defaults for optional fields
    vals: dict[str, float] = {
        "h_run_rate":          game.h_run_rate,
        "h_ra_rate":           game.h_ra_rate,
        "h_run_diff_rate":     game.h_run_diff_rate,
        "h_win_rate":          game.h_win_rate,
        "h_scoring_var":       game.h_scoring_var      if game.h_scoring_var      is not None else _DEFAULTS["h_scoring_var"],
        "h_last3_win_rate":    game.h_last3_win_rate   if game.h_last3_win_rate   is not None else _DEFAULTS["h_last3_win_rate"],
        "h_quality_start_rt":  game.h_quality_start_rt if game.h_quality_start_rt is not None else _DEFAULTS["h_quality_start_rt"],
        "h_ops":               game.h_ops              if game.h_ops              is not None else _DEFAULTS["h_ops"],
        "h_streak":            game.h_streak           if game.h_streak           is not None else _DEFAULTS["h_streak"],
        "a_run_rate":          game.a_run_rate,
        "a_ra_rate":           game.a_ra_rate,
        "a_run_diff_rate":     game.a_run_diff_rate,
        "a_win_rate":          game.a_win_rate,
        "a_scoring_var":       game.a_scoring_var      if game.a_scoring_var      is not None else _DEFAULTS["a_scoring_var"],
        "a_last3_win_rate":    game.a_last3_win_rate   if game.a_last3_win_rate   is not None else _DEFAULTS["a_last3_win_rate"],
        "a_quality_start_rt":  game.a_quality_start_rt if game.a_quality_start_rt is not None else _DEFAULTS["a_quality_start_rt"],
        "a_ops":               game.a_ops              if game.a_ops              is not None else _DEFAULTS["a_ops"],
        "a_streak":            game.a_streak           if game.a_streak           is not None else _DEFAULTS["a_streak"],
        # Differentials
        "diff_run_rate":       game.h_run_rate      - game.a_run_rate,
        "diff_ra_rate":        game.h_ra_rate       - game.a_ra_rate,
        "diff_run_diff_rate":  game.h_run_diff_rate - game.a_run_diff_rate,
        "diff_win_rate":       game.h_win_rate      - game.a_win_rate,
        "diff_ops":            (game.h_ops if game.h_ops is not None else _DEFAULTS["h_ops"]) -
                               (game.a_ops if game.a_ops is not None else _DEFAULTS["a_ops"]),
        "diff_streak":         (game.h_streak if game.h_streak is not None else 0) -
                               (game.a_streak if game.a_streak is not None else 0),
        "diff_last3_win_rate": (game.h_last3_win_rate if game.h_last3_win_rate is not None else 0.5) -
                               (game.a_last3_win_rate if game.a_last3_win_rate is not None else 0.5),
        # Context
        "home_advantage":  1.0,
        "is_covid_season": 0.0,   # any future game is not COVID season
        # Day-of-week one-hot
        **{f"dow_{i}": float(d.weekday() == i) for i in range(7)},
        # Month one-hot (clamp to [4, 10] for spring training / postseason edge cases)
        **{f"month_{m}": float(max(4, min(10, d.month)) == m) for m in range(4, 11)},
    }

    return np.array([vals[col] for col in FEATURE_COLS], dtype=np.float32)


def _apply_scaler(X: np.ndarray, meta: dict) -> np.ndarray:
    """Normalize continuous features using training-set statistics."""
    if meta.get("scaler_mean") is None:
        return X   # no scaler saved — return as-is
    X = X.copy()
    binary_start = FEATURE_COLS.index("home_advantage")
    mean = meta["scaler_mean"]
    std  = meta["scaler_std"]
    X[:, :binary_start] = (X[:, :binary_start] - mean) / np.clip(std, 1e-6, None)
    return X


def _confidence_label(prob: float) -> str:
    margin = abs(prob - 0.5)
    if margin < 0.04:  return "toss-up"
    if margin < 0.08:  return "slight edge"
    if margin < 0.13:  return "moderate edge"
    if margin < 0.18:  return "clear edge"
    return "strong edge"


def _run_inference(games: list[GameInput]) -> list[PredictionResponse]:
    if _state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0 = time.perf_counter()

    # Build feature matrix
    X = np.stack([_game_to_feature_vector(g) for g in games], axis=0)
    X = _apply_scaler(X, _state.meta)

    # Forward pass
    _state.model.eval()
    with torch.no_grad():
        tensor  = torch.tensor(X, dtype=torch.float32).to(_state.device)
        logits  = _state.model(tensor)
        probs   = torch.sigmoid(logits).cpu().numpy().ravel()

    # Apply calibrator if present
    calibrator  = _state.meta.get("calibrator")
    calibrated  = calibrator is not None
    if calibrated:
        probs = calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]

    elapsed_ms = (time.perf_counter() - t0) * 1000
    per_game   = elapsed_ms / len(games)

    results = []
    for game, prob in zip(games, probs):
        prob = float(np.clip(prob, 0.0, 1.0))
        results.append(PredictionResponse(
            home_team     = game.home_team.upper(),
            away_team     = game.away_team.upper(),
            game_date     = game.game_date,
            home_win_prob = round(prob, 4),
            away_win_prob = round(1 - prob, 4),
            confidence    = _confidence_label(prob),
            model_arch    = _state.meta.get("arch", "unknown"),
            calibrated    = calibrated,
            inference_ms  = round(per_game, 2),
        ))
    return results


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health() -> HealthResponse:
    """Check that the API is running and the model is loaded."""
    if _state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return HealthResponse(
        status       = "ok",
        model_arch   = _state.meta.get("arch", "unknown"),
        epoch        = _state.meta.get("epoch", -1),
        val_loss     = round(_state.meta.get("val_loss", 0), 4),
        calibrated   = _state.meta.get("calibrator") is not None,
        loaded_at    = _state.loaded_at,
        checkpoint   = _state.ckpt_path,
        feature_cols = FEATURE_COLS,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(game: GameInput) -> PredictionResponse:
    """
    Predict home-team win probability for a single game.

    Supply rolling stats computed over the N games *before* this game.
    Optional fields default to league-average values if omitted.
    """
    return _run_inference([game])[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(payload: BatchInput) -> BatchPredictionResponse:
    """
    Predict home-team win probability for up to 100 games in one call.
    More efficient than calling /predict repeatedly.
    """
    t0 = time.perf_counter()
    predictions = _run_inference(payload.games)
    total_ms = (time.perf_counter() - t0) * 1000
    return BatchPredictionResponse(
        predictions = predictions,
        n_games     = len(predictions),
        total_ms    = round(total_ms, 2),
    )


@app.get("/feature_importance", tags=["Meta"])
def feature_importance() -> dict:
    """
    Estimate feature sensitivity using input perturbation.
    Each feature is zeroed out and the change in average predicted
    probability is measured. Larger change = more important feature.

    Note: this is a simple sensitivity measure, not SHAP values.
    """
    if _state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Use a league-average "neutral" game as the reference point
    neutral: dict[str, float] = {col: 0.0 for col in FEATURE_COLS}
    neutral.update({
        "h_win_rate": 0.5,  "a_win_rate": 0.5,
        "h_run_rate": 4.5,  "a_run_rate": 4.5,
        "h_ra_rate":  4.5,  "a_ra_rate":  4.5,
        "h_ops":      0.5,  "a_ops":      0.5,
        "home_advantage": 1.0,
        "month_7": 1.0,     # July — mid-season
        "dow_4": 1.0,       # Friday
    })

    base_vec = np.array([neutral[c] for c in FEATURE_COLS], dtype=np.float32)
    base_scaled = _apply_scaler(base_vec[None, :], _state.meta)
    with torch.no_grad():
        base_prob = torch.sigmoid(
            _state.model(torch.tensor(base_scaled, dtype=torch.float32))
        ).item()

    importances = {}
    for i, col in enumerate(FEATURE_COLS):
        perturbed = base_scaled.copy()
        perturbed[0, i] = 0.0   # zero out one feature
        with torch.no_grad():
            p = torch.sigmoid(
                _state.model(torch.tensor(perturbed, dtype=torch.float32))
            ).item()
        importances[col] = round(abs(p - base_prob), 5)

    ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    return {
        "base_prob":     round(base_prob, 4),
        "method":        "zero-out perturbation",
        "ranked_features": [{"feature": k, "sensitivity": v} for k, v in ranked],
    }


# ── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    ckpt = os.environ.get("MLB_CHECKPOINT", str(DEFAULT_CKPT))
    try:
        _load_model(ckpt)
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("The API will start but /predict will return 503 until a checkpoint exists.")


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="MLB Win Probability API server.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT),
                        help="Path to the .pt checkpoint file from train.py")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true",
                        help="Auto-reload on code changes (dev mode)")
    args = parser.parse_args()

    os.environ["MLB_CHECKPOINT"] = args.checkpoint
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        app_dir=str(_SRC),
    )