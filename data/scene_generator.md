# Data-Generation Library Spec — Referential Localization Game

## Goal

Build a library that generates synthetic scenes for an emergent-communication localization game. Each sample is a **full scene image** containing multiple objects ("subjects"), one designated **target**, plus ground-truth annotations. A Sender will see only the target's crop; a Receiver will see the full scene and must localize the target. The generator must therefore produce everything both agents and the downstream MI analysis need.

## Core mechanic (why the constraints below exist)

- **Sender sees only the target crop**, so the message can encode *what* the target is, not *where* it is.
- **Receiver sees the full raw scene** and predicts the target's location (grid cell / heatmap over the scene).
- Consequence: the target's attribute signature **must be unique within its scene**, or the trial is unwinnable. This is a hard correctness constraint on the generator, not a preference.

## Parameters

**Content**
- `shapes`: allowed shape vocabulary (default: all).
- `colors`: allowed colors (default: all).
- `size`: `"same"` or `"variable"`; if variable, `size_range: [min, max]` (or discrete `size_bins`).

**Scene composition**
- `n_objects`: `[min, max]` — number of objects per scene. **This is the distractor count and the primary difficulty dial; it sets the chance floor (1/K) and the minimum channel bits (~log₂K). Required.**
- `distractor_policy`: `"iid"` | `"hard_negative"`.
  - `iid`: objects sampled independently.
  - `hard_negative`: force ≥1 distractor to share all-but-one attribute with the target (target = small red circle → inject small blue circle, large red circle, small red square). This is what turns discrimination into a compositionality probe.
- `overlap`: `"none"` (default) | `"allow"`. Default non-overlapping via rejection sampling with a min-distance (or one-per-cell when `fix_to_grid=True`). Occlusion corrupts crops, so keep it off by default.

**Layout**
- `fix_to_grid`: `True` → objects placed exactly inside grid cells (makes cell labels exact); `False` → free placement.
- `placement_grid`: how the canvas is divided **for placement**. Keep this conceptually separate from the Receiver's prediction resolution — they are different objects and may differ.

**Bookkeeping**
- `n_samples`: N.
- `seed`: for reproducible datasets.
- `canvas_size` + `background`: explicit; guard against invisible objects (no white shape on white bg).
- `splits`: train/val/test fractions, with an **option** to hold out attribute combinations for a compositional-generalization split later (need not be used day one).

## Correctness / well-posedness constraints

1. **Target uniqueness**: no distractor may equal the target's full attribute tuple (shape + color + size-bin). Distractors may duplicate *each other*, just not the target. With continuous size, exact match is measure-zero (uniqueness is free) — but that is precisely when *near*-match becomes the interesting hard-negative case.
2. **Feasibility validation**: fail loudly when the attribute space is too small for the requested K under all-unique — i.e. require `|shapes| × |colors| × |size_bins| ≥ K` and raise a clear error otherwise (e.g. 2 shapes × 2 colors × fixed size, K=6 → error, not silent duplication).
3. **Balanced marginals**: sample attributes (and position, if free) with balanced/independent marginals. If "red" is 5× "blue," or red correlates with top-left, the MI readout is confounded *and* the agents can cheat on the correlation. This one matters specifically for the analysis being clean.

## Output schema (per sample)

Emit **rich per-object metadata**, not just the target box — the readout is MI between message symbols and {category, color, size, position}, which requires structured attributes for every object.

```
{
  "scene": <full scene image>,
  "target_crop": <tight crop of the target — the Sender's input>,
  "target_bbox": <GT box in pixel coords>,          # Receiver regression target / IoU eval
  "target_cell": <cell index>,                       # Receiver cross-entropy label
  "objects": [
    { "shape", "color", "size", "bbox", "cell", "is_target": bool },
    ...                                              # target + all distractors
  ],
  "meta": { "seed", "canvas_size", "n_objects", "distractor_policy", ... }
}
```

Build the metadata rich now; retrofitting it after generating a large dataset is painful.

## Deliverable

Generate N samples per the config, each a full scene + target crop + GT box + target cell + per-object attribute list, with target-uniqueness guaranteed and marginals balanced. Provide a config dataclass exposing all params above, and an annotation schema from which both the Receiver's cell labels and the MI readout fall out directly.

## Two knobs that carry the science

- `n_objects` (K) — difficulty / channel-capacity floor.
- `distractor_policy` — `hard_negative` is what forces a descriptive, compositional protocol and is the substrate for later negation experiments ("the one that is *not* red").
