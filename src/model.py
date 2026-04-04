"""
model.py
--------
PyTorch model definitions for MLB win probability prediction.

Three architectures are provided, ordered by complexity:

  WinProbMLP        – Standard feedforward network. Good baseline.
                      Best starting point; interpretable, fast to train.

  WinProbResNet     – MLP with residual skip connections. Helps with
                      deeper networks by easing gradient flow.

  WinProbEnsemble   – Wraps multiple sub-models and averages their
                      sigmoid outputs. Improves calibration and reduces
                      variance at the cost of training time.

All models:
  - Accept a float32 tensor of shape (batch, input_dim)
  - Output a float32 tensor of shape (batch, 1) — raw logit (pre-sigmoid)
  - Use sigmoid externally during inference / BCEWithLogitsLoss during training

Usage:
    from model import WinProbMLP, WinProbResNet, WinProbEnsemble, build_model

    model = build_model("mlp", input_dim=41)
    logits = model(x)                        # (B, 1) raw logit
    probs  = torch.sigmoid(logits)           # (B, 1) win probability
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ── Utility blocks ────────────────────────────────────────────────────────────

class FCBlock(nn.Module):
    """
    A single fully-connected block:
        Linear → BatchNorm → Activation → Dropout
    """
    def __init__(
        self,
        in_dim:    int,
        out_dim:   int,
        dropout:   float = 0.3,
        act:       nn.Module | None = None,
        use_bn:    bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn     = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()
        self.act    = act if act is not None else nn.GELU()
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.act(self.bn(self.linear(x))))


class ResBlock(nn.Module):
    """
    Residual block:  x → FCBlock → FCBlock → + x → output
    Requires in_dim == out_dim (same-width residual).
    """
    def __init__(self, dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.block1 = FCBlock(dim, dim, dropout=dropout)
        self.block2 = FCBlock(dim, dim, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block2(self.block1(x))


# ── Architecture 1: Standard MLP ─────────────────────────────────────────────

class WinProbMLP(nn.Module):
    """
    Feedforward neural network for binary win probability.

    Architecture:
        Input(41) → FC(256) → FC(128) → FC(64) → FC(32) → Linear(1)

    Parameters
    ----------
    input_dim   : number of input features (default 41)
    hidden_dims : sequence of hidden layer widths
    dropout     : dropout rate applied after each hidden layer
    """
    def __init__(
        self,
        input_dim:   int = 41,
        hidden_dims: tuple[int, ...] = (256, 128, 64, 32),
        dropout:     float = 0.3,
    ) -> None:
        super().__init__()
        dims   = [input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            # Taper dropout toward the output
            d = dropout * (1 - i / len(dims))
            layers.append(FCBlock(dims[i], dims[i + 1], dropout=d))
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.body(x))   # raw logit

    def predict_proba(self, x: Tensor) -> Tensor:
        """Convenience: returns sigmoid-calibrated probability."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


# ── Architecture 2: Residual MLP ─────────────────────────────────────────────

class WinProbResNet(nn.Module):
    """
    MLP with residual skip connections.

    Architecture:
        Input(41) → Stem FC(128)
                  → ResBlock(128) × n_blocks
                  → Linear(1)

    Residual connections help gradient flow in deeper networks and
    make the model easier to train than a plain deep MLP.

    Parameters
    ----------
    input_dim : number of input features
    width     : hidden width (same throughout — required for residuals)
    n_blocks  : number of residual blocks
    dropout   : dropout rate inside each block
    """
    def __init__(
        self,
        input_dim: int = 41,
        width:     int = 128,
        n_blocks:  int = 4,
        dropout:   float = 0.2,
    ) -> None:
        super().__init__()
        self.stem   = FCBlock(input_dim, width, dropout=dropout)
        self.blocks = nn.Sequential(*[ResBlock(width, dropout=dropout) for _ in range(n_blocks)])
        self.head   = nn.Linear(width, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.blocks(self.stem(x)))

    def predict_proba(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


# ── Architecture 3: Ensemble ──────────────────────────────────────────────────

class WinProbEnsemble(nn.Module):
    """
    Ensemble of N independent sub-models.

    During training, call each sub-model separately (they have their own
    optimizers — see train.py).  During inference, forward() averages the
    sigmoid probabilities of all sub-models and returns the ensemble logit
    (logit of the averaged probability).

    Parameters
    ----------
    model_class : WinProbMLP or WinProbResNet
    n_members   : number of ensemble members (typically 3–7)
    **kwargs    : forwarded to each sub-model constructor
    """
    def __init__(
        self,
        model_class: type,
        n_members:   int = 5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.members = nn.ModuleList([model_class(**kwargs) for _ in range(n_members)])

    def forward(self, x: Tensor) -> Tensor:
        """Returns averaged probability converted back to logit space."""
        probs = torch.stack(
            [torch.sigmoid(m(x)) for m in self.members], dim=0
        ).mean(dim=0)
        # Clamp to avoid log(0) in logit conversion
        probs = probs.clamp(1e-6, 1 - 1e-6)
        return torch.log(probs / (1 - probs))   # logit

    def predict_proba(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def member_probs(self, x: Tensor) -> Tensor:
        """Returns (n_members, batch, 1) tensor of individual member probs."""
        with torch.no_grad():
            return torch.stack([torch.sigmoid(m(x)) for m in self.members], dim=0)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(
    arch:      str = "mlp",
    input_dim: int = 41,
    **kwargs,
) -> nn.Module:
    """
    Instantiate a model by name.

    Parameters
    ----------
    arch      : one of "mlp", "resnet", "ensemble_mlp", "ensemble_resnet"
    input_dim : number of input features
    **kwargs  : forwarded to the model constructor

    Examples
    --------
    build_model("mlp", input_dim=41)
    build_model("resnet", input_dim=41, width=128, n_blocks=4)
    build_model("ensemble_mlp", input_dim=41, n_members=5)
    """
    arch = arch.lower()
    if arch == "mlp":
        return WinProbMLP(input_dim=input_dim, **kwargs)
    elif arch == "resnet":
        return WinProbResNet(input_dim=input_dim, **kwargs)
    elif arch == "ensemble_mlp":
        n = kwargs.pop("n_members", 5)
        return WinProbEnsemble(WinProbMLP, n_members=n, input_dim=input_dim, **kwargs)
    elif arch == "ensemble_resnet":
        n = kwargs.pop("n_members", 5)
        return WinProbEnsemble(WinProbResNet, n_members=n, input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: '{arch}'. "
                         f"Choose from: mlp, resnet, ensemble_mlp, ensemble_resnet")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_dim: int = 41) -> None:
    """Print a concise architecture summary."""
    print(f"\n{'─'*50}")
    print(f"  {model.__class__.__name__}")
    print(f"{'─'*50}")
    print(f"  Trainable parameters: {count_parameters(model):,}")
    x = torch.randn(4, input_dim)
    with torch.no_grad():
        out = model(x)
    print(f"  Input shape:  {tuple(x.shape)}")
    print(f"  Output shape: {tuple(out.shape)}  (raw logit)")
    print(f"  Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    print(f"{'─'*50}\n")


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    INPUT_DIM = 41
    x = torch.randn(32, INPUT_DIM)

    for arch in ("mlp", "resnet", "ensemble_mlp"):
        m = build_model(arch, input_dim=INPUT_DIM)
        model_summary(m, INPUT_DIM)
        probs = m.predict_proba(x)
        assert probs.shape == (32, 1), f"Unexpected output shape: {probs.shape}"
        assert probs.min() >= 0 and probs.max() <= 1, "Probabilities out of range"

    print("✓ All model checks passed.")