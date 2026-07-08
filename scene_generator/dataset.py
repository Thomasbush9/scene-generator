"""Loading saved datasets back as PyTorch Datasets.

``SceneDataset`` returns the spec's per-sample dict; ``EggSceneDataset`` adapts
it to EGG's ``(sender_input, labels, receiver_input, aux_input)`` convention so
it can be fed straight into an EGG game's DataLoader.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .config import SceneConfig


class SceneDataset(Dataset):
    """Reads a dataset written by ``SceneGenerator.save``.

    Images are stored as uint8 shards and converted to float32 in [0, 1] at
    access time. Shards are memory-cached one at a time (samples within a
    shard are contiguous, so sequential and shuffled-by-shard access is cheap).
    """

    def __init__(self, root: str | Path, split: str = "train"):
        self.root = Path(root)
        self.split = split
        self.config = SceneConfig.from_json(self.root / "config.json")
        split_dir = self.root / split
        self.shard_paths = sorted(split_dir.glob("shard_*.pt"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards found in {split_dir}")
        self._shard_sizes = []
        for p in self.shard_paths:
            shard = torch.load(p, map_location="cpu", weights_only=False)
            self._shard_sizes.append(shard["target_cell"].shape[0])
        self._offsets = [0]
        for s in self._shard_sizes:
            self._offsets.append(self._offsets[-1] + s)
        self._cache_idx: int | None = None
        self._cache: dict | None = None

    def __len__(self) -> int:
        return self._offsets[-1]

    def _locate(self, index: int) -> tuple[dict, int]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_idx = next(i for i in range(len(self.shard_paths)) if self._offsets[i + 1] > index)
        if shard_idx != self._cache_idx:
            self._cache = torch.load(
                self.shard_paths[shard_idx], map_location="cpu", weights_only=False
            )
            self._cache_idx = shard_idx
        return self._cache, index - self._offsets[shard_idx]

    def __getitem__(self, index: int) -> dict:
        shard, i = self._locate(index)
        return {
            "scene": shard["scene"][i].float() / 255.0,
            "target_crop": shard["target_crop"][i].float() / 255.0,
            "target_bbox": shard["target_bbox"][i],
            "target_cell": shard["target_cell"][i],
            "objects": shard["objects"][i],
            "meta": shard["meta"],
        }

    def load_metadata(self) -> list[dict]:
        """All per-sample metadata records (for the MI analysis, no images)."""
        with open(self.root / self.split / "metadata.jsonl") as f:
            return [json.loads(line) for line in f]


class EggSceneDataset(SceneDataset):
    """EGG-convention view: (sender_input, labels, receiver_input, aux_input).

    * sender_input   — the target crop (the only thing the Sender sees)
    * labels         — the target's grid-cell index
    * receiver_input — the full scene
    * aux_input      — dict with the ground-truth bbox for IoU evaluation
    """

    def __getitem__(self, index: int):
        s = super().__getitem__(index)
        return (
            s["target_crop"],
            s["target_cell"],
            s["scene"],
            {"target_bbox": s["target_bbox"]},
        )


class _ClassificationDataset(SceneDataset):
    """Shared machinery for the optional vision-pretraining views.

    Builds per-attribute class vocabularies from the dataset config and the
    string->index maps used to produce integer labels. Class index i maps to
    ``dataset.classes[attr][i]``.
    """

    ATTRIBUTES = ("shape", "color", "size_bin")

    def __init__(self, root, split: str = "train", attributes=("shape", "color"),
                 extra_classes: tuple = ()):
        super().__init__(root, split)
        unknown = set(attributes) - set(self.ATTRIBUTES)
        if unknown:
            raise ValueError(f"Unknown attributes {sorted(unknown)}; pick from {self.ATTRIBUTES}")
        self.attributes = tuple(attributes)
        self.classes = {
            "shape": list(self.config.shapes) + list(extra_classes),
            "color": list(self.config.colors) + list(extra_classes),
            "size_bin": list(range(self.config.num_size_bins)) + list(extra_classes),
        }
        self._index = {
            attr: {v: i for i, v in enumerate(vals)} for attr, vals in self.classes.items()
        }

    def num_classes(self, attr: str) -> int:
        return len(self.classes[attr])


class CropClassificationDataset(_ClassificationDataset):
    """(target_crop, labels) view for pretraining the Sender's vision encoder.

    Yields ``(crop, labels)`` where ``labels`` is a dict of integer class
    indices, one per requested attribute — e.g. ``{"shape": 2, "color": 4}``.
    Ints collate into tensors under the default DataLoader collate, so this
    plugs straight into a multi-head classification loop.

    With ``stratify_targets=True`` (the generator default) the crops are
    exactly balanced over attribute combos.
    """

    def __getitem__(self, index: int):
        s = super().__getitem__(index)
        target = next(o for o in s["objects"] if o["is_target"])
        labels = {a: self._index[a][target[a]] for a in self.attributes}
        return s["target_crop"], labels


class CellClassificationDataset(_ClassificationDataset):
    """(scene, labels) view for pretraining the Receiver's scene encoder.

    Yields ``(scene, labels)`` where ``labels[attr]`` is a LongTensor of shape
    ``(rows * cols,)`` over the label grid (row-major, same indexing as
    ``target_cell``). Cells with no object get the extra ``"empty"`` class,
    which is always the LAST index — i.e. ``num_classes(attr) - 1``.

    The task: classify every object in the scene by location — dense
    perception pretraining that never sees the message-matching problem.
    A typical head is a conv encoder ending in a (rows, cols) map with one
    ``num_classes(attr)``-way output per attribute; flatten to (n_cells, C)
    and cross-entropy against these labels.

    Only well-posed for grid datasets (``fix_to_grid=True`` with the label
    grid equal to the placement grid), where "one object per cell" holds.
    """

    EMPTY = "empty"

    def __init__(self, root, split: str = "train", attributes=("shape", "color")):
        super().__init__(root, split, attributes, extra_classes=(self.EMPTY,))
        if not self.config.fix_to_grid:
            raise ValueError(
                "CellClassificationDataset needs a fix_to_grid=True dataset: with free "
                "placement a cell can contain several objects, making labels ambiguous."
            )
        if self.config.effective_label_grid != tuple(self.config.placement_grid):
            raise ValueError(
                "CellClassificationDataset needs label_grid == placement_grid, got "
                f"{self.config.effective_label_grid} vs {self.config.placement_grid}"
            )
        self.grid = self.config.effective_label_grid
        self.n_cells = self.grid[0] * self.grid[1]

    def __getitem__(self, index: int):
        s = super().__getitem__(index)
        empty = {a: self._index[a][self.EMPTY] for a in self.attributes}
        labels = {
            a: torch.full((self.n_cells,), empty[a], dtype=torch.long)
            for a in self.attributes
        }
        for o in s["objects"]:
            for a in self.attributes:
                labels[a][o["cell"]] = self._index[a][o[a]]
        return s["scene"], labels


DATASET_VIEWS = {
    "scene": SceneDataset,                          # spec-schema dicts
    "egg": EggSceneDataset,                         # the communication game
    "crop_classification": CropClassificationDataset,  # Sender vision pretraining
    "cell_classification": CellClassificationDataset,  # Receiver vision pretraining
}


def load_dataset(root, split: str = "train", view: str = "scene", **kwargs):
    """Open a saved dataset under the requested view (flag-style entry point).

    The pretraining views are optional — turn them on by name::

        train = load_dataset(root, "train", view="egg")                  # the game
        sender = load_dataset(root, "train", view="crop_classification") # + attributes=...
        receiver = load_dataset(root, "train", view="cell_classification")
    """
    if view not in DATASET_VIEWS:
        raise ValueError(f"Unknown view {view!r}; pick from {sorted(DATASET_VIEWS)}")
    return DATASET_VIEWS[view](root, split, **kwargs)


def collate_scenes(samples: list[dict]) -> dict:
    """Collate SceneDataset dicts, stacking tensors and listing object metadata."""
    return {
        "scene": torch.stack([s["scene"] for s in samples]),
        "target_crop": torch.stack([s["target_crop"] for s in samples]),
        "target_bbox": torch.stack([s["target_bbox"] for s in samples]),
        "target_cell": torch.stack([s["target_cell"] for s in samples]),
        "objects": [s["objects"] for s in samples],
        "meta": samples[0]["meta"],
    }
