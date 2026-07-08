# scene_generator

Synthetic-scene generator for the referential-localization emergent-communication
game (see `data/scene_generator.md` for the full spec). Each sample is a full
scene of colored shapes with exactly one attribute-unique **target**, plus:

- `target_crop` — tight crop of the target (the **Sender**'s input),
- `target_bbox` / `target_cell` — pixel box and grid-cell label (the **Receiver**'s targets),
- `objects` — shape/color/size/bbox/cell for **every** object (for the MI readout).

## Setup

```bash
module load python && mamba activate envs/lang   # torch 2.11+cu128 already installed
```

## Quick start

```python
from scene_generator import SceneConfig, SceneGenerator, SceneDataset, EggSceneDataset

cfg = SceneConfig(
    n_objects=(4, 6),                  # K — difficulty / channel-capacity dial
    distractor_policy="hard_negative", # forces a compositional protocol
    size="variable", size_bins=(14, 20, 26),
    holdout_combo_frac=0.1,            # compositional-generalization test split
    n_samples=50_000,
)
gen = SceneGenerator(cfg, device="auto")   # cuda > mps > cpu

batch = gen.generate_batch(256)            # on-the-fly SceneBatch
gen.save("data/scenes_v1")                 # persisted dataset with train/val/test

train = SceneDataset("data/scenes_v1", "train")        # spec-schema dicts
egg   = EggSceneDataset("data/scenes_v1", "train")     # (sender_input, labels,
                                                       #  receiver_input, aux_input)
```

### Optional vision-pretraining views

Any saved dataset can be opened under a *view*, selected by flag via
`load_dataset(root, split, view=...)` — no regeneration needed. Besides the
game views (`"scene"`, `"egg"`) there are two opt-in pretraining views that
give each agent "perception" before the communication game starts:

```python
from scene_generator import load_dataset

# Sender: classify the target crop by attribute -> (crop, {"shape": i, "color": j})
sender = load_dataset("data/scenes_v1", "train", view="crop_classification",
                      attributes=("shape", "color"))

# Receiver: classify every object in the full scene by grid cell
# -> (scene, {"shape": LongTensor(16,), "color": LongTensor(16,)})
# Empty cells get the extra "empty" class (always the last index).
receiver = load_dataset("data/scenes_v1", "train", view="cell_classification")
receiver.num_classes("shape")   # |shapes| + 1
```

`cell_classification` labels are row-major over the label grid (same indexing
as `target_cell`) and require a `fix_to_grid=True` dataset (one object per
cell, so labels are unambiguous). Both views collate under the default
DataLoader collate. With `stratify_targets=True` the crop view is exactly
class-balanced.

CLI:

```bash
python -m scene_generator --config cfg.yaml --out data/scenes_v1 --n 50000 --device cuda
python -m scene_generator --config cfg.yaml --preview preview.png   # eyeball scenes/crops
```

## Guarantees (enforced, tested in `tests/`)

- **Target uniqueness** — no distractor matches the target's (shape, color,
  size-bin) tuple; distractors may duplicate each other.
- **Feasibility validation** — `SceneConfig.validate()` raises `FeasibilityError`
  when |shapes|·|colors|·|size-bins| < max K, when the grid can't hold K objects,
  or when a color is invisible against the background.
- **Balanced marginals** — attributes uniform & independent, positions uniform;
  `stratify_targets=True` (default) makes target marginals *exactly* balanced.
- **Reproducibility** — all sampling is CPU/numpy driven by `seed`, so datasets
  are identical regardless of the rendering device.

## Architecture

| module | role |
|---|---|
| `shapes.py` | shape vocabulary as SDFs on the unit box; subclass `Shape` to add one |
| `config.py` | `SceneConfig` dataclass + `validate()` (all spec knobs) |
| `sampler.py` | attributes, hard negatives, placement, holdout splits (all RNG) |
| `render.py` | batched anti-aliased rasterizer, device-agnostic (`cuda`/`mps`/`cpu`) |
| `generator.py` | `SceneGenerator` / `SceneBatch`, dataset persistence |
| `dataset.py` | dataset views: `SceneDataset`, `EggSceneDataset`, pretraining views, `load_dataset` |

On-disk layout: `config.json` + per-split `shard_*.pt` (uint8 images + labels)
and `metadata.jsonl` (per-sample object lists — load with pandas for the MI
analysis without touching images).

Throughput (A100 MIG 3g.20gb, 128×128, K=5): ~3.2k scenes/s render, ~550
scenes/s end-to-end to disk.
