"""scene_generator — synthetic scenes for a referential-localization game.

Generates full scenes of colored shapes with one attribute-unique target,
plus the target crop (Sender input), ground-truth bbox and grid-cell label
(Receiver targets), and rich per-object metadata (for the MI readout).

Quick start::

    from scene_generator import SceneConfig, SceneGenerator, SceneDataset

    cfg = SceneConfig(n_objects=(4, 4), distractor_policy="hard_negative")
    gen = SceneGenerator(cfg, device="auto")       # cuda > mps > cpu
    batch = gen.generate_batch(256)                # on-the-fly batch
    gen.save("data/scenes_v1", n_samples=50_000)   # persisted dataset w/ splits
    train = SceneDataset("data/scenes_v1", "train")
"""

from .config import PALETTE, FeasibilityError, SceneConfig
from .dataset import (
    DATASET_VIEWS,
    CellClassificationDataset,
    CentroidHeatmapDataset,
    CropClassificationDataset,
    EggSceneDataset,
    SceneDataset,
    collate_scenes,
    load_dataset,
)
from .generator import SceneBatch, SceneGenerator
from .render import Renderer, resolve_device
from .sampler import AttributeSpace, ObjectSpec, PlacementError, SceneSampler
from .shapes import SHAPES, Shape

__all__ = [
    "SceneConfig",
    "SceneGenerator",
    "SceneBatch",
    "SceneDataset",
    "EggSceneDataset",
    "CropClassificationDataset",
    "CellClassificationDataset",
    "CentroidHeatmapDataset",
    "load_dataset",
    "DATASET_VIEWS",
    "collate_scenes",
    "Renderer",
    "resolve_device",
    "SceneSampler",
    "AttributeSpace",
    "ObjectSpec",
    "Shape",
    "SHAPES",
    "PALETTE",
    "FeasibilityError",
    "PlacementError",
]
