"""Configuration for the referential-localization scene generator.

``SceneConfig`` exposes every knob from the spec and ``validate()`` enforces
the well-posedness constraints (attribute-space feasibility, grid capacity,
color/background contrast) *before* any sample is generated, failing loudly
instead of silently producing unwinnable trials.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Literal, Sequence

from .shapes import SHAPES

# Default palette (RGB, 0-255). Extend or override via SceneConfig.palette.
PALETTE: dict[str, tuple[int, int, int]] = {
    "red": (225, 45, 45),
    "green": (45, 180, 70),
    "blue": (50, 95, 230),
    "yellow": (240, 210, 45),
    "magenta": (220, 60, 200),
    "cyan": (60, 200, 220),
    "orange": (245, 140, 30),
    "purple": (130, 60, 200),
    "white": (240, 240, 240),
    "gray": (128, 128, 128),
    "black": (0, 0, 0),
}

DEFAULT_COLORS = ("red", "green", "blue", "yellow", "magenta", "cyan")

# Minimum Euclidean RGB distance between an object color and the background
# for the object to count as visible.
MIN_CONTRAST = 60.0


class FeasibilityError(ValueError):
    """The requested configuration cannot produce well-posed scenes."""


@dataclass
class SceneConfig:
    # --- Scene composition (required) ---
    n_objects: tuple[int, int]
    """[min, max] objects per scene (K). Difficulty dial: chance floor is 1/K."""

    # --- Content ---
    shapes: Sequence[str] = field(default_factory=lambda: tuple(SHAPES))
    colors: Sequence[str] = DEFAULT_COLORS
    size: Literal["same", "variable"] = "same"
    size_value: float | None = None
    """Object extent in px when size == "same". None -> auto from canvas/grid."""
    size_bins: Sequence[float] | None = None
    """Discrete allowed sizes (px) when size == "variable"."""
    size_range: tuple[float, float] | None = None
    """Continuous size range (px) when size == "variable" and no size_bins."""
    n_size_bins: int = 3
    """Bin count used to discretize a continuous size_range for labels/uniqueness."""

    # --- Scene composition ---
    distractor_policy: Literal["iid", "hard_negative"] = "iid"
    n_hard_negatives: int = 1
    overlap: Literal["none", "allow"] = "none"
    min_gap: float = 2.0
    """Minimum gap (px) between object bounding boxes when overlap == "none"."""

    # --- Layout ---
    fix_to_grid: bool = False
    placement_grid: tuple[int, int] = (4, 4)  # (rows, cols)
    label_grid: tuple[int, int] | None = None
    """Grid used for the Receiver's cell label. None -> same as placement_grid."""

    # --- Bookkeeping ---
    n_samples: int = 10_000
    seed: int = 0
    canvas_size: int | tuple[int, int] = 128  # (H, W) if tuple
    background: str | tuple[int, int, int] = "black"
    splits: tuple[float, float, float] = (0.8, 0.1, 0.1)  # train/val/test
    holdout_combo_frac: float = 0.0
    """Fraction of attribute combos held out of train/val entirely; test targets
    are drawn from them (compositional-generalization split). 0 disables."""
    stratify_targets: bool = True
    """Cycle target attributes through a shuffled Cartesian product so target
    marginals are exactly balanced (keeps the MI readout clean)."""

    # --- Sender input ---
    crop_size: int = 64
    """Target crops are resized to (crop_size, crop_size) for batching."""
    crop_mode: Literal["tight", "fixed_window"] = "tight"
    """tight: bbox + crop_pad resized to crop_size — destroys absolute size
    (fine when size is constant). fixed_window: a fixed crop_window x
    crop_window px window centered on the target, so relative size survives
    in the crop. Required whenever size is a distinguishing attribute."""
    crop_window: float | None = None
    """Window extent (px) for crop_mode="fixed_window". None -> 1.4 * max size."""
    crop_pad: float = 2.0
    """Padding (px) around the target bbox when cropping (crop_mode="tight")."""

    palette: dict[str, tuple[int, int, int]] | None = None
    """Extra/overriding color definitions merged over the default PALETTE."""

    # ------------------------------------------------------------------ #

    @property
    def canvas_hw(self) -> tuple[int, int]:
        c = self.canvas_size
        return (c, c) if isinstance(c, int) else (int(c[0]), int(c[1]))

    @property
    def full_palette(self) -> dict[str, tuple[int, int, int]]:
        return {**PALETTE, **(self.palette or {})}

    @property
    def background_rgb(self) -> tuple[int, int, int]:
        bg = self.background
        return self.full_palette[bg] if isinstance(bg, str) else tuple(bg)

    @property
    def cell_hw(self) -> tuple[float, float]:
        """Placement-grid cell size in px."""
        h, w = self.canvas_hw
        rows, cols = self.placement_grid
        return h / rows, w / cols

    @property
    def effective_label_grid(self) -> tuple[int, int]:
        return tuple(self.label_grid) if self.label_grid else tuple(self.placement_grid)

    @property
    def num_size_bins(self) -> int:
        if self.size == "same":
            return 1
        if self.size_bins is not None:
            return len(self.size_bins)
        return self.n_size_bins

    @property
    def effective_size_value(self) -> float:
        if self.size_value is not None:
            return float(self.size_value)
        if self.fix_to_grid:
            return 0.7 * min(self.cell_hw)
        return min(self.canvas_hw) / 7.0

    @property
    def effective_crop_window(self) -> float:
        return float(self.crop_window) if self.crop_window is not None else 1.4 * self.max_size

    @property
    def max_size(self) -> float:
        if self.size == "same":
            return self.effective_size_value
        if self.size_bins is not None:
            return max(self.size_bins)
        return self.size_range[1]

    def validate(self) -> None:
        """Raise FeasibilityError / ValueError on ill-posed configurations."""
        kmin, kmax = self.n_objects
        if not (1 <= kmin <= kmax):
            raise ValueError(f"n_objects must satisfy 1 <= min <= max, got {self.n_objects}")

        unknown = [s for s in self.shapes if s not in SHAPES]
        if unknown:
            raise ValueError(f"Unknown shapes {unknown}; available: {sorted(SHAPES)}")
        unknown = [c for c in self.colors if c not in self.full_palette]
        if unknown:
            raise ValueError(f"Unknown colors {unknown}; available: {sorted(self.full_palette)}")
        if len(set(self.shapes)) != len(self.shapes) or len(set(self.colors)) != len(self.colors):
            raise ValueError("shapes/colors must not contain duplicates")

        if self.size == "variable":
            if (self.size_bins is None) == (self.size_range is None):
                raise ValueError('size == "variable" needs exactly one of size_bins or size_range')
            if self.size_range is not None and not (0 < self.size_range[0] < self.size_range[1]):
                raise ValueError(f"Invalid size_range {self.size_range}")

        # Feasibility: the attribute space must accommodate K all-unique objects,
        # otherwise the target-uniqueness constraint starves the sampler.
        n_combos = len(self.shapes) * len(self.colors) * self.num_size_bins
        if n_combos < kmax:
            raise FeasibilityError(
                f"Attribute space too small: |shapes|*|colors|*|size_bins| = "
                f"{len(self.shapes)}*{len(self.colors)}*{self.num_size_bins} = {n_combos} "
                f"< max n_objects = {kmax}. Enlarge the vocabulary or lower n_objects."
            )
        if self.distractor_policy == "hard_negative" and self.n_hard_negatives > kmax - 1:
            raise ValueError(
                f"n_hard_negatives={self.n_hard_negatives} exceeds max distractors ({kmax - 1})"
            )

        # Layout capacity.
        rows, cols = self.placement_grid
        if self.fix_to_grid:
            if rows * cols < kmax:
                raise FeasibilityError(
                    f"placement_grid {rows}x{cols} has {rows * cols} cells < max n_objects {kmax}"
                )
            if self.max_size > min(self.cell_hw):
                raise FeasibilityError(
                    f"Max object size {self.max_size:.1f}px does not fit in a "
                    f"{self.cell_hw[0]:.1f}x{self.cell_hw[1]:.1f}px grid cell"
                )
        h, w = self.canvas_hw
        if self.max_size > min(h, w) / 2:
            raise FeasibilityError(f"Max object size {self.max_size:.1f}px too large for canvas {h}x{w}")

        if self.crop_mode == "fixed_window":
            if self.effective_crop_window < self.max_size:
                raise FeasibilityError(
                    f"crop_window {self.effective_crop_window:.1f}px smaller than the max "
                    f"object size {self.max_size:.1f}px — the largest objects would be clipped"
                )
        if self.size == "variable" and self.crop_mode == "tight":
            import warnings

            warnings.warn(
                'size == "variable" with crop_mode == "tight": tight crops are resized to '
                "crop_size, so the Sender cannot perceive object size. Use "
                'crop_mode="fixed_window" if size should be communicable.',
                stacklevel=2,
            )

        # Visibility: every allowed color must contrast with the background.
        bg = self.background_rgb
        invisible = [
            c for c in self.colors
            if sum((a - b) ** 2 for a, b in zip(self.full_palette[c], bg)) ** 0.5 < MIN_CONTRAST
        ]
        if invisible:
            raise FeasibilityError(
                f"Colors {invisible} are too close to background {bg} — objects would be invisible"
            )

        if abs(sum(self.splits) - 1.0) > 1e-6 or any(s < 0 for s in self.splits):
            raise ValueError(f"splits must be non-negative and sum to 1, got {self.splits}")
        if not 0.0 <= self.holdout_combo_frac < 1.0:
            raise ValueError(f"holdout_combo_frac must be in [0, 1), got {self.holdout_combo_frac}")

    # --- (De)serialization ------------------------------------------------ #

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "SceneConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        d = dict(d)
        for key in ("n_objects", "placement_grid", "label_grid", "splits", "canvas_size",
                    "size_range", "size_bins", "shapes", "colors", "background"):
            if isinstance(d.get(key), list):
                d[key] = tuple(d[key])
        return cls(**d)

    @classmethod
    def from_json(cls, path) -> "SceneConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
