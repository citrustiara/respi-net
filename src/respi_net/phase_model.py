"""Causal networks that label every radar frame with a breathing phase.

The input is the A121 chest signal as channels x time at 20 Hz (the
``respi_net.nn_dataset`` channels: chest displacement, its velocity and the
echo strength); the output is one score per class of
:mod:`respi_net.breath_phases` for every frame.  Everything here is causal --
the output at frame t uses frames <= t only -- so the same weights run offline
over a whole recording and live on frames as they arrive
(:class:`StreamingPhaseDetector`), with the same result.

Two networks share the ``[B, C, T] -> [B, num_classes, T]`` interface:

* :class:`CausalTCN`: dilated causal convolutions with a fixed, known
  receptive field (509 frames = 25.4 s at 20 Hz by default -- a whole breath
  at 6/min and a 15 s hold both fit), so streaming needs exactly that much
  history and nothing older can leak in.
* :class:`GRUTagger`: a unidirectional GRU; streaming carries its hidden state.

Offline, per-frame log-probabilities go through
:func:`respi_net.phase_metrics.viterbi_decode`.  Live there is no future to
backtrack from, so :class:`OnlinePhaseSmoother` stops the flicker instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .breath_phases import ALLOWED_TRANSITIONS, CLASS_NAMES, IGNORE, NOISE, NUM_CLASSES

__all__ = [
    "CHECKPOINT_FORMAT",
    "MODELS",
    "CausalConv1d",
    "CausalTCN",
    "ChannelNorm",
    "GRUTagger",
    "OnlinePhaseSmoother",
    "StreamingPhaseDetector",
    "build_model",
    "class_weights",
    "count_parameters",
    "load_checkpoint",
    "masked_cross_entropy",
    "save_checkpoint",
]

CHECKPOINT_FORMAT = "respi_net.phase_model/1"
DEFAULT_DILATIONS = (1, 2, 4, 8, 16, 32, 64)


class ChannelNorm(nn.Module):
    """LayerNorm across channels, separately at every frame.

    BatchNorm and GroupNorm on ``[B, C, T]`` also average over time, so a
    frame's output would depend on the frames after it (and, live, on how many
    frames happen to be in a chunk).  Normalising each frame on its own keeps
    the network causal and streaming exact.
    """

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalConv1d(nn.Module):
    """Dilated ``Conv1d`` whose padding is all on the left: frame t sees frames <= t."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        # How many past frames one output needs; a stream carries exactly these.
        self.context = (kernel_size - 1) * dilation

    def forward(self, x: Tensor, past: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Convolve ``x`` [B, C, T] that follows the ``context`` frames in ``past``.

        ``past=None`` is the start of a sequence and means zeros, i.e. plain
        left padding, so a whole sequence and the same sequence fed chunk by
        chunk go through identical arithmetic.  Returns the output
        [B, C_out, T] and the ``past`` for the next chunk.
        """

        if past is None:
            past = x.new_zeros(x.shape[0], x.shape[1], self.context)
        padded = torch.cat([past, x], dim=-1)
        return self.conv(padded), padded[..., padded.shape[-1] - self.context :]


class _TCNBlock(nn.Module):
    """Residual block: causal dilated conv -> norm -> GELU -> dropout -> 1x1 conv."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm = ChannelNorm(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        # A second (pointwise) layer per block adds depth without widening the
        # receptive field, which the dilations alone decide.
        self.mix = nn.Conv1d(channels, channels, 1)

    def forward(self, x: Tensor, past: Tensor | None = None) -> tuple[Tensor, Tensor]:
        h, past = self.conv(x, past)
        h = self.drop(self.act(self.norm(h)))
        return x + self.mix(h), past


class CausalTCN(nn.Module):
    """Temporal convolutional network: ``[B, C, T]`` -> logits ``[B, num_classes, T]``.

    One dilated causal conv per residual block, so the receptive field is
    ``1 + (kernel_size - 1) * sum(dilations)`` frames.
    """

    model_name = "tcn"

    def __init__(
        self,
        in_channels: int,
        num_classes: int = NUM_CLASSES,
        channels: int = 32,
        kernel_size: int = 5,
        dilations: Sequence[int] = DEFAULT_DILATIONS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dilations = [int(d) for d in dilations]
        if kernel_size < 1 or not dilations or min(dilations) < 1:
            raise ValueError("kernel_size and every dilation must be >= 1, with at least one dilation")
        # Plain values only, so a checkpoint can rebuild the model and still
        # load with torch.load(weights_only=True).
        self.config: dict[str, Any] = {
            "in_channels": int(in_channels),
            "num_classes": int(num_classes),
            "channels": int(channels),
            "kernel_size": int(kernel_size),
            "dilations": dilations,
            "dropout": float(dropout),
        }
        self.inp = nn.Conv1d(in_channels, channels, 1)
        self.blocks = nn.ModuleList(_TCNBlock(channels, kernel_size, d, dropout) for d in dilations)
        # The residual stream is never normalised, so normalise before the head.
        self.head = nn.Sequential(ChannelNorm(channels), nn.GELU(), nn.Conv1d(channels, num_classes, 1))

    def receptive_field_samples(self) -> int:
        """Input frames that can influence one output frame, the current one included."""

        return 1 + sum(block.conv.context for block in self.blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.stream(x)[0]

    def stream(self, x: Tensor, state: Sequence[Tensor] | None = None) -> tuple[Tensor, list[Tensor]]:
        """Logits for a chunk ``x`` that follows the chunk ``state`` was returned for.

        ``state`` holds, per block, the last frames its conv saw (None = the
        start of a sequence).  Chunk after chunk this equals one pass over the
        whole sequence while computing the new frames only.
        """

        h = self.inp(x)
        new_state: list[Tensor] = []
        for index, block in enumerate(self.blocks):
            h, past = block(h, None if state is None else state[index])
            new_state.append(past)
        return self.head(h), new_state


class GRUTagger(nn.Module):
    """Unidirectional GRU: ``[B, C, T]`` -> logits ``[B, num_classes, T]``."""

    model_name = "gru"

    def __init__(
        self,
        in_channels: int,
        num_classes: int = NUM_CLASSES,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.config: dict[str, Any] = {
            "in_channels": int(in_channels),
            "num_classes": int(num_classes),
            "hidden": int(hidden),
            "layers": int(layers),
            "dropout": float(dropout),
        }
        # nn.GRU only applies dropout between stacked layers and warns otherwise.
        self.gru = nn.GRU(in_channels, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.stream(x)[0]

    def stream(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Logits for a chunk; ``state`` is the hidden state [layers, B, hidden] after the previous one."""

        # A contiguous [B, T, C] copy: some GRU backends (cuDNN, MPS) reject the strided view.
        out, state = self.gru(x.transpose(1, 2).contiguous(), state)
        return self.head(out).transpose(1, 2), state


MODELS: dict[str, type[nn.Module]] = {"tcn": CausalTCN, "gru": GRUTagger}


def build_model(name: str, in_channels: int, **kwargs: Any) -> nn.Module:
    """``"tcn"`` -> :class:`CausalTCN`, ``"gru"`` -> :class:`GRUTagger`, with ``kwargs`` passed on."""

    try:
        cls = MODELS[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown phase model {name!r}; choose from {sorted(MODELS)}") from None
    return cls(in_channels, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    channels: Sequence[str],
    fs: float,
    **extra: Any,
) -> None:
    """Weights plus everything needed to rebuild and feed the model.

    ``extra`` must hold plain values (str, numbers, lists, dicts) so the file
    loads with ``weights_only=True``.
    """

    payload = {
        "format": CHECKPOINT_FORMAT,
        "model_name": model.model_name,
        "model_config": dict(model.config),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "channels": [str(name) for name in channels],
        "fs": float(fs),
        "class_names": list(CLASS_NAMES),
        **extra,
    }
    torch.save(payload, Path(path))


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """The model (in eval mode) and the whole checkpoint dict from :func:`save_checkpoint`."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    model = build_model(payload["model_name"], **payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    model.to(map_location)
    return model.eval(), payload


def class_weights(labels: np.ndarray, num_classes: int = NUM_CLASSES, power: float = 1.0) -> np.ndarray:
    """Inverse-frequency ("balanced") weights; IGNORE is skipped, absent classes get 0.

    ``n_labelled / (n_present_classes * n_class)``, raised to ``power`` (1 =
    inverse frequency, 0.5 = its square root for a rare class that would
    otherwise dominate, 0 = unweighted) and scaled so the average weight over
    labelled samples is 1, which keeps the loss on its usual scale.
    """

    labels = np.asarray(labels).ravel()
    counts = np.bincount(labels[(labels >= 0) & (labels < num_classes)].astype(np.int64), minlength=num_classes)
    counts = counts[:num_classes].astype(float)
    present = counts > 0
    weights = np.zeros(num_classes)
    if present.any():
        raw = (counts.sum() / (present.sum() * counts[present])) ** power
        weights[present] = raw * counts.sum() / np.sum(counts[present] * raw)
    return weights


def masked_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    weight: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Cross-entropy over labelled frames only; IGNORE (-1) frames count for nothing.

    ``logits`` [B, K, T], ``targets`` [B, T].  A batch without a single label
    gives a zero loss (still attached to the graph) instead of 0/0 = NaN.
    """

    targets = targets.long()
    if not bool((targets != IGNORE).any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits, targets, weight=weight, ignore_index=IGNORE, reduction=reduction)


class StreamingPhaseDetector:
    """Phase probabilities for frames that arrive a chunk at a time.

    :meth:`push` takes the new frames ``[C, n]`` -- pre-processed exactly like
    the training windows -- and returns probabilities ``[n, num_classes]`` for
    those frames only.

    With ``history_s=None`` the model's own stream state is carried between
    chunks (per-block conv history for the TCN, i.e. its receptive field; the
    hidden state for the GRU), each chunk costs only its new frames, and the
    output equals one pass over the whole sequence.  With ``history_s`` the
    detector keeps that many seconds of input and reruns the model over
    history + chunk instead: exact for a TCN whose receptive field fits in the
    history, and the way to run a causal model without ``stream()`` or to cap
    a GRU's memory at the window length it was trained on.  The model is put
    in eval mode.
    """

    def __init__(self, model: nn.Module, fs: float, history_s: float | None = None) -> None:
        self.model = model.eval()
        self.fs = float(fs)
        self.stateful = history_s is None and hasattr(model, "stream")
        if self.stateful:
            self.history_frames = 0
        elif history_s is not None:
            self.history_frames = max(0, int(round(history_s * self.fs)))
        elif hasattr(model, "receptive_field_samples"):
            self.history_frames = int(model.receptive_field_samples()) - 1
        else:
            raise ValueError("history_s is needed for a model without stream() or receptive_field_samples()")
        config = getattr(model, "config", {})
        self.num_classes = int(config.get("num_classes", NUM_CLASSES))
        self.in_channels = config.get("in_channels")
        self.device = next(model.parameters()).device
        self.reset()

    def reset(self) -> None:
        """Forget the past: the next chunk starts a new sequence."""

        self._state: Any = None
        self._history: Tensor | None = None
        self.frames_seen = 0

    @property
    def time_s(self) -> float:
        """Stream time at the end of the last pushed frame."""

        return self.frames_seen / self.fs

    @torch.no_grad()
    def push(self, frames: np.ndarray | Tensor) -> np.ndarray:
        """Probabilities ``[n, num_classes]`` for new frames ``[C, n]`` (or one frame ``[C]``)."""

        x = torch.as_tensor(frames, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or (self.in_channels is not None and x.shape[0] != self.in_channels):
            raise ValueError(f"expected frames [channels={self.in_channels}, n], got {tuple(x.shape)}")
        n = x.shape[-1]
        if n == 0:
            return np.zeros((0, self.num_classes), dtype=np.float32)
        x = x[None]
        if self.stateful:
            logits, self._state = self.model.stream(x, self._state)
        else:
            window = x if self._history is None else torch.cat([self._history, x], dim=-1)
            logits = self.model(window)[..., -n:]
            keep = min(self.history_frames, window.shape[-1])
            self._history = window[..., window.shape[-1] - keep :] if keep else None
        self.frames_seen += n
        return torch.softmax(logits.float(), dim=1)[0].T.cpu().numpy()


class OnlinePhaseSmoother:
    """Live anti-flicker for per-frame class probabilities.

    The emitted class changes only when a new class has been the confident
    top class (probability >= ``threshold``) for ``confirm_frames``
    consecutive frames, the change is allowed by ``allowed`` (noise may
    interrupt anything and anything may follow noise) and the current phase
    has been emitted for at least ``min_phase_s``.  A switch therefore lands
    on the ``confirm_frames``-th confident frame, ``confirm_frames - 1``
    frames after the first (100 ms at 20 Hz with the defaults).  Until the
    first class is confirmed the output is IGNORE (-1).  Unlike Viterbi it
    never revises the past, which is the price of deciding every frame as it
    arrives.
    """

    def __init__(
        self,
        fs: float,
        min_phase_s: float = 0.3,
        confirm_frames: int = 3,
        threshold: float = 0.5,
        allowed: Mapping[int, frozenset[int]] = ALLOWED_TRANSITIONS,
    ) -> None:
        if confirm_frames < 1:
            raise ValueError("confirm_frames must be >= 1")
        self.fs = float(fs)
        self.min_phase_frames = max(0, int(round(min_phase_s * self.fs)))
        self.confirm_frames = int(confirm_frames)
        self.threshold = float(threshold)
        self.allowed = allowed
        self.reset()

    def reset(self) -> None:
        self.current = IGNORE
        self._held = 0  # frames the current class has been emitted
        self._candidate = IGNORE
        self._streak = 0

    def _may_switch(self, target: int) -> bool:
        if self.current == IGNORE:
            return True
        if self._held < self.min_phase_frames:
            return False
        return target == NOISE or target in self.allowed.get(self.current, frozenset())

    def step(self, probs: np.ndarray) -> int:
        """Emitted class after one frame's probabilities ``[num_classes]``."""

        probs = np.asarray(probs, dtype=float)
        top = int(np.argmax(probs))
        if top == self.current:
            self._candidate, self._streak = IGNORE, 0
        elif probs[top] >= self.threshold:
            self._streak = self._streak + 1 if top == self._candidate else 1
            self._candidate = top
            # A blocked candidate keeps its streak, so it takes over as soon as
            # the current phase is old enough (if it is still confident then).
            if self._streak >= self.confirm_frames and self._may_switch(top):
                self.current, self._held = top, 0
                self._candidate, self._streak = IGNORE, 0
        else:
            # Confirmation needs consecutive confident frames; an unsure one resets it.
            self._candidate, self._streak = IGNORE, 0
        if self.current != IGNORE:
            self._held += 1
        return self.current

    def update(self, probs: np.ndarray) -> np.ndarray:
        """Emitted class per frame for probabilities ``[n, num_classes]`` (or one frame ``[num_classes]``)."""

        rows = np.atleast_2d(np.asarray(probs, dtype=float))
        return np.array([self.step(row) for row in rows], dtype=np.int64)
