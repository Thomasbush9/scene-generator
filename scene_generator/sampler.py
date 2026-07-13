"""Attribute and position sampling.

All randomness lives here, on the CPU via ``numpy.random.Generator``, so a
dataset is bit-reproducible from its seed regardless of the rendering device.

Correctness guarantees enforced here:
  * **Target uniqueness** — no distractor ever equals the target's full
    (shape, color, size-bin) tuple; distractors may duplicate each other.
  * **Hard negatives** — under ``distractor_policy="hard_negative"``, the
    first ``n_hard_negatives`` distractors share all-but-one attribute with
    the target, and the flipped attribute CYCLES across them (shuffled order).
    With n_hard_negatives >= the number of flippable attributes, no single
    attribute ever identifies the target — messages must be compositional.
  * **Balanced marginals** — attributes are sampled uniformly and
    independently; positions uniformly (over the canvas or over grid cells).
    With ``stratify_targets=True`` target combos are cycled through shuffled
    permutations of the full Cartesian product, making target marginals
    exactly balanced.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .config import FeasibilityError, SceneConfig

Combo = tuple[int, int, int]  # (shape index, color index, size-bin index)


class PlacementError(RuntimeError):
    """Rejection sampling could not place all objects without overlap."""


@dataclass
class ObjectSpec:
    """Full ground-truth description of one object in a scene."""

    shape: str
    color: str
    size: float          # bounding-box extent, px
    size_bin: int
    cx: float            # center, px
    cy: float
    cell: int            # row-major index in the label grid
    is_target: bool
    combo: Combo = field(repr=False, default=(0, 0, 0))

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        h = self.size / 2.0
        return (self.cx - h, self.cy - h, self.cx + h, self.cy + h)

    def to_dict(self) -> dict:
        return {
            "shape": self.shape,
            "color": self.color,
            "size": round(self.size, 3),
            "size_bin": self.size_bin,
            "bbox": [round(v, 3) for v in self.bbox],
            "cell": self.cell,
            "is_target": self.is_target,
        }


class AttributeSpace:
    """The discrete attribute lattice (shape x color x size-bin) of a config."""

    def __init__(self, config: SceneConfig):
        self.config = config
        self.shapes = list(config.shapes)
        self.colors = list(config.colors)
        self.n_size_bins = config.num_size_bins
        self.combos: list[Combo] = list(
            itertools.product(
                range(len(self.shapes)), range(len(self.colors)), range(self.n_size_bins)
            )
        )

    def size_from_bin(self, b: int, rng: np.random.Generator) -> float:
        """Concrete pixel size for a size-bin (uniform within the bin if continuous)."""
        cfg = self.config
        if cfg.size == "same":
            return cfg.effective_size_value
        if cfg.size_bins is not None:
            return float(cfg.size_bins[b])
        lo, hi = cfg.size_range
        width = (hi - lo) / self.n_size_bins
        return float(rng.uniform(lo + b * width, lo + (b + 1) * width))


class SceneSampler:
    """Samples the symbolic layout of one scene (attributes + positions).

    ``target_combos`` / ``object_combos`` restrict the attribute lattice per
    split, which is how the compositional-generalization holdout is realized.
    """

    MAX_SCENE_RESTARTS = 100
    MAX_PLACEMENT_TRIES = 200

    def __init__(
        self,
        config: SceneConfig,
        space: AttributeSpace,
        rng: np.random.Generator,
        target_combos: list[Combo] | None = None,
        object_combos: list[Combo] | None = None,
    ):
        self.config = config
        self.space = space
        self.rng = rng
        self.target_combos = list(target_combos if target_combos is not None else space.combos)
        self.object_combos = list(object_combos if object_combos is not None else space.combos)
        if not self.target_combos:
            raise FeasibilityError("Sampler has an empty target-combo set")
        if len(self.object_combos) < max(config.n_objects):
            raise FeasibilityError(
                f"Only {len(self.object_combos)} attribute combos available to this split "
                f"but scenes need up to {max(config.n_objects)} objects. "
                f"Lower holdout_combo_frac or n_objects."
            )
        self._target_cycle: list[Combo] = []

    # --- attribute sampling ------------------------------------------------ #

    def _next_target_combo(self) -> Combo:
        if not self.config.stratify_targets:
            return self.target_combos[self.rng.integers(len(self.target_combos))]
        if not self._target_cycle:
            self._target_cycle = list(self.target_combos)
            self.rng.shuffle(self._target_cycle)
        return self._target_cycle.pop()

    def _flippable_dims(self) -> list[int]:
        """Attribute dimensions with >1 value in this split's lattice."""
        return [d for d in range(3) if len({c[d] for c in self.object_combos}) > 1]

    def _hard_negative_combo(self, target: Combo, prefer_dim: int | None = None) -> Combo:
        """A combo sharing all-but-one attribute with the target.

        ``prefer_dim`` requests which attribute to flip; the other dims serve
        as fallback when the split's lattice has no neighbor along it (can
        happen under aggressive holdout).
        """
        dims = self._flippable_dims()
        self.rng.shuffle(dims)
        if prefer_dim in dims:
            dims.remove(prefer_dim)
            dims.insert(0, prefer_dim)
        for dim in dims:
            values = sorted({c[dim] for c in self.object_combos} - {target[dim]})
            self.rng.shuffle(values)
            for v in values:
                cand = list(target)
                cand[dim] = v
                cand = tuple(cand)
                if cand in self._object_combo_set:
                    return cand
        # No single-attribute neighbor available in this split's lattice
        # (can happen under aggressive holdout); fall back to iid.
        return self._iid_distractor_combo(target)

    def _iid_distractor_combo(self, target: Combo) -> Combo:
        while True:
            c = self.object_combos[self.rng.integers(len(self.object_combos))]
            if c != target:  # target uniqueness — the one hard constraint
                return c

    @property
    def _object_combo_set(self) -> set[Combo]:
        return set(self.object_combos)

    # --- position sampling -------------------------------------------------- #

    def _label_cell(self, cx: float, cy: float) -> int:
        h, w = self.config.canvas_hw
        rows, cols = self.config.effective_label_grid
        r = min(int(cy / (h / rows)), rows - 1)
        c = min(int(cx / (w / cols)), cols - 1)
        return r * cols + c

    def _place_grid(self, k: int) -> list[tuple[float, float]]:
        rows, cols = self.config.placement_grid
        h, w = self.config.canvas_hw
        cells = self.rng.choice(rows * cols, size=k, replace=False)
        return [
            ((c % cols + 0.5) * (w / cols), (c // cols + 0.5) * (h / rows))
            for c in cells
        ]

    def _place_free(self, sizes: list[float]) -> list[tuple[float, float]]:
        h, w = self.config.canvas_hw
        gap = self.config.min_gap
        check_overlap = self.config.overlap == "none"
        centers: list[tuple[float, float]] = []
        for size in sizes:
            half = size / 2.0
            for _ in range(self.MAX_PLACEMENT_TRIES):
                cx = self.rng.uniform(half, w - half)
                cy = self.rng.uniform(half, h - half)
                if not check_overlap or all(
                    # axis-aligned bbox separation + gap (Chebyshev distance)
                    max(abs(cx - ox), abs(cy - oy)) >= half + osize / 2.0 + gap
                    for (ox, oy), osize in zip(centers, sizes)
                ):
                    centers.append((cx, cy))
                    break
            else:
                raise PlacementError(
                    f"Could not place {len(sizes)} objects of sizes {sizes} on a "
                    f"{h}x{w} canvas without overlap. Reduce n_objects/size or the canvas density."
                )
        return centers

    # --- scene sampling ----------------------------------------------------- #

    def sample_scene(self) -> list[ObjectSpec]:
        cfg = self.config
        kmin, kmax = cfg.n_objects
        k = int(self.rng.integers(kmin, kmax + 1))

        target = self._next_target_combo()
        combos = [target]
        n_hard = cfg.n_hard_negatives if cfg.distractor_policy == "hard_negative" else 0
        n_hard = min(n_hard, k - 1)
        if n_hard:
            # Cycle the flipped attribute over the scene's hard negatives so
            # they cover distinct dimensions: with enough negatives, naming a
            # single attribute never suffices to identify the target.
            flip = self._flippable_dims()
            self.rng.shuffle(flip)
            combos += [
                self._hard_negative_combo(target, flip[i % len(flip)] if flip else None)
                for i in range(n_hard)
            ]
        combos += [self._iid_distractor_combo(target) for _ in range(k - 1 - n_hard)]

        # Shuffle so the target's index carries no information.
        order = self.rng.permutation(k)
        combos = [combos[i] for i in order]
        target_idx = int(np.argwhere(order == 0)[0, 0])

        sizes = [self.space.size_from_bin(c[2], self.rng) for c in combos]
        for _ in range(self.MAX_SCENE_RESTARTS):
            try:
                centers = self._place_grid(k) if cfg.fix_to_grid else self._place_free(sizes)
                break
            except PlacementError:
                continue
        else:
            raise PlacementError(
                f"Placement failed after {self.MAX_SCENE_RESTARTS} scene restarts "
                f"(k={k}, sizes={sizes}). The configuration is too dense."
            )

        return [
            ObjectSpec(
                shape=self.space.shapes[c[0]],
                color=self.space.colors[c[1]],
                size=sizes[i],
                size_bin=c[2],
                cx=centers[i][0],
                cy=centers[i][1],
                cell=self._label_cell(*centers[i]),
                is_target=(i == target_idx),
                combo=c,
            )
            for i, c in enumerate(combos)
        ]


def split_combos(
    space: AttributeSpace, config: SceneConfig, rng: np.random.Generator
) -> dict[str, dict[str, list[Combo]]]:
    """Per-split target/object combo sets implementing the compositional holdout.

    With ``holdout_combo_frac > 0``: held-out combos never appear in train/val
    scenes (neither as target nor distractor); test targets are drawn from the
    held-out set, with distractors free over the full lattice.
    """
    combos = list(space.combos)
    frac = config.holdout_combo_frac
    if frac <= 0.0:
        full = {"target_combos": combos, "object_combos": combos}
        return {"train": full, "val": dict(full), "test": dict(full)}

    n_held = max(1, round(frac * len(combos)))
    if n_held >= len(combos):
        raise FeasibilityError("holdout_combo_frac leaves no combos for training")
    held_idx = rng.choice(len(combos), size=n_held, replace=False)
    held = [combos[i] for i in held_idx]
    seen = [c for i, c in enumerate(combos) if i not in set(held_idx.tolist())]
    return {
        "train": {"target_combos": seen, "object_combos": seen},
        "val": {"target_combos": seen, "object_combos": seen},
        "test": {"target_combos": held, "object_combos": combos},
    }
