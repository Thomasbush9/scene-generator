"""High-level API: sample -> render -> crop -> (optionally) persist.

Typical usage::

    from scene_generator import SceneConfig, SceneGenerator

    cfg = SceneConfig(n_objects=(4, 4), distractor_policy="hard_negative")
    gen = SceneGenerator(cfg, device="auto")   # cuda > mps > cpu
    batch = gen.generate_batch(256)            # in-memory SceneBatch
    gen.save("data/scenes_v1")                 # full dataset with splits on disk
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .config import SceneConfig
from .render import Renderer, resolve_device
from .sampler import AttributeSpace, ObjectSpec, SceneSampler, split_combos

SPLITS = ("train", "val", "test")


@dataclass
class SceneBatch:
    """A batch of generated samples; indexing yields the spec's per-sample dict."""

    scene: Tensor        # (B, 3, H, W) float32 in [0, 1]
    target_crop: Tensor  # (B, 3, crop, crop) float32 — the Sender's input
    target_bbox: Tensor  # (B, 4) float32, pixel xyxy — Receiver regression target
    target_cell: Tensor  # (B,) int64 — Receiver cross-entropy label
    objects: list[list[ObjectSpec]]
    meta: dict

    def __len__(self) -> int:
        return self.scene.shape[0]

    def __getitem__(self, i: int) -> dict:
        return {
            "scene": self.scene[i],
            "target_crop": self.target_crop[i],
            "target_bbox": self.target_bbox[i],
            "target_cell": self.target_cell[i],
            "objects": [o.to_dict() for o in self.objects[i]],
            "meta": self.meta,
        }

    def to(self, device) -> "SceneBatch":
        return SceneBatch(
            scene=self.scene.to(device),
            target_crop=self.target_crop.to(device),
            target_bbox=self.target_bbox.to(device),
            target_cell=self.target_cell.to(device),
            objects=self.objects,
            meta=self.meta,
        )


class SceneGenerator:
    """Ties the sampler and renderer together and manages splits/persistence."""

    def __init__(self, config: SceneConfig, device: str | torch.device = "auto"):
        config.validate()
        self.config = config
        self.device = resolve_device(device) if isinstance(device, str) else device
        self.renderer = Renderer(config, self.device)
        self.space = AttributeSpace(config)

        holdout_rng = np.random.default_rng(config.seed)
        combo_sets = split_combos(self.space, config, holdout_rng)
        self.samplers = {
            split: SceneSampler(
                config,
                self.space,
                rng=np.random.default_rng([config.seed, i]),
                **combo_sets[split],
            )
            for i, split in enumerate(SPLITS)
        }

    # --- in-memory generation --------------------------------------------- #

    def generate_batch(self, batch_size: int, split: str = "train") -> SceneBatch:
        """Sample and render ``batch_size`` scenes from a split's distribution."""
        sampler = self.samplers[split]
        scenes = [sampler.sample_scene() for _ in range(batch_size)]
        return self._materialize(scenes)

    def iter_batches(self, n_samples: int, batch_size: int = 256, split: str = "train"):
        """Yield SceneBatches totalling ``n_samples`` samples."""
        remaining = n_samples
        while remaining > 0:
            b = min(batch_size, remaining)
            yield self.generate_batch(b, split)
            remaining -= b

    def _materialize(self, scenes: list[list[ObjectSpec]]) -> SceneBatch:
        images = self.renderer.render(scenes)
        targets = [next(o for o in scene if o.is_target) for scene in scenes]
        bboxes = torch.tensor([t.bbox for t in targets], dtype=torch.float32, device=images.device)
        cells = torch.tensor([t.cell for t in targets], dtype=torch.int64, device=images.device)
        crops = self.renderer.crop_targets(images, bboxes)
        cfg = self.config
        meta = {
            "seed": cfg.seed,
            "canvas_size": list(cfg.canvas_hw),
            "n_objects": list(cfg.n_objects),
            "distractor_policy": cfg.distractor_policy,
            "overlap": cfg.overlap,
            "fix_to_grid": cfg.fix_to_grid,
            "placement_grid": list(cfg.placement_grid),
            "label_grid": list(cfg.effective_label_grid),
            "crop_size": cfg.crop_size,
        }
        return SceneBatch(images, crops, bboxes, cells, scenes, meta)

    # --- persistence -------------------------------------------------------- #

    def save(
        self,
        out_dir: str | Path,
        n_samples: int | None = None,
        batch_size: int = 512,
        shard_size: int = 4096,
        progress: bool = True,
    ) -> Path:
        """Generate the full dataset and write it under ``out_dir``.

        Layout::

            out_dir/config.json
            out_dir/{train,val,test}/shard_00000.pt   # uint8 images + labels
            out_dir/{train,val,test}/metadata.jsonl   # per-sample object lists

        Images are stored as uint8 to save disk; ``SceneDataset`` converts
        back to float32 in [0, 1] on load.
        """
        from tqdm import tqdm

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.config.to_json(out / "config.json")

        n_total = n_samples if n_samples is not None else self.config.n_samples
        counts = self._split_counts(n_total)
        for split, n in counts.items():
            if n == 0:
                continue
            split_dir = out / split
            split_dir.mkdir(exist_ok=True)
            pbar = tqdm(total=n, desc=f"{split:5s}", disable=not progress, unit="scene")
            buffer: list[SceneBatch] = []
            buffered = 0
            shard_idx = 0
            sample_idx = 0
            with open(split_dir / "metadata.jsonl", "w") as meta_f:
                for batch in self.iter_batches(n, batch_size, split):
                    buffer.append(batch)
                    buffered += len(batch)
                    for i in range(len(batch)):
                        record = {
                            "index": sample_idx,
                            "shard": sample_idx // shard_size,
                            "target_bbox": [round(v, 3) for v in batch.target_bbox[i].tolist()],
                            "target_cell": int(batch.target_cell[i]),
                            "objects": [o.to_dict() for o in batch.objects[i]],
                        }
                        meta_f.write(json.dumps(record) + "\n")
                        sample_idx += 1
                    pbar.update(len(batch))
                    while buffered >= shard_size:
                        buffer, buffered = self._flush_shard(
                            split_dir, shard_idx, buffer, shard_size
                        )
                        shard_idx += 1
                if buffered:
                    self._flush_shard(split_dir, shard_idx, buffer, buffered)
            pbar.close()
        return out

    def _split_counts(self, n_total: int) -> dict[str, int]:
        fr = self.config.splits
        counts = {s: int(round(f * n_total)) for s, f in zip(SPLITS, fr)}
        counts["train"] += n_total - sum(counts.values())  # rounding remainder
        return counts

    @staticmethod
    def _flush_shard(
        split_dir: Path, shard_idx: int, buffer: list[SceneBatch], shard_size: int
    ) -> tuple[list[SceneBatch], int]:
        """Write ``shard_size`` samples from the batch buffer to one .pt shard."""
        scene = torch.cat([b.scene.cpu() for b in buffer])[:shard_size]
        crop = torch.cat([b.target_crop.cpu() for b in buffer])[:shard_size]
        bbox = torch.cat([b.target_bbox.cpu() for b in buffer])[:shard_size]
        cell = torch.cat([b.target_cell.cpu() for b in buffer])[:shard_size]
        objects = [o for b in buffer for o in b.objects][:shard_size]
        torch.save(
            {
                "scene": (scene * 255).round().to(torch.uint8),
                "target_crop": (crop * 255).round().to(torch.uint8),
                "target_bbox": bbox,
                "target_cell": cell,
                "objects": [[obj.to_dict() for obj in scene_objs] for scene_objs in objects],
                "meta": buffer[0].meta,
            },
            split_dir / f"shard_{shard_idx:05d}.pt",
        )
        # Re-buffer the overflow so no sample is dropped between shards.
        total = sum(len(b) for b in buffer)
        leftover = total - shard_size
        if leftover <= 0:
            return [], 0
        rest = SceneBatch(
            scene=torch.cat([b.scene.cpu() for b in buffer])[shard_size:],
            target_crop=torch.cat([b.target_crop.cpu() for b in buffer])[shard_size:],
            target_bbox=torch.cat([b.target_bbox.cpu() for b in buffer])[shard_size:],
            target_cell=torch.cat([b.target_cell.cpu() for b in buffer])[shard_size:],
            objects=[o for b in buffer for o in b.objects][shard_size:],
            meta=buffer[0].meta,
        )
        return [rest], leftover
