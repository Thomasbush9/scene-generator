"""Correctness tests for the well-posedness guarantees of scene_generator."""

import itertools

import numpy as np
import pytest
import torch

from scene_generator import (
    CellClassificationDataset,
    CentroidHeatmapDataset,
    CropClassificationDataset,
    FeasibilityError,
    SceneConfig,
    SceneDataset,
    EggSceneDataset,
    SceneGenerator,
    load_dataset,
)


def combo_tuple(obj: dict) -> tuple:
    return (obj["shape"], obj["color"], obj["size_bin"])


def make_gen(**kwargs) -> SceneGenerator:
    cfg = SceneConfig(**{"n_objects": (3, 5), "seed": 7, **kwargs})
    return SceneGenerator(cfg, device="cpu")


# --- Feasibility validation ---------------------------------------------- #

def test_attribute_space_too_small_raises():
    cfg = SceneConfig(
        n_objects=(6, 6), shapes=("circle", "square"), colors=("red", "blue")
    )
    with pytest.raises(FeasibilityError, match="Attribute space too small"):
        cfg.validate()


def test_invisible_color_raises():
    cfg = SceneConfig(n_objects=(2, 2), colors=("white", "red"), background="white")
    with pytest.raises(FeasibilityError, match="invisible"):
        cfg.validate()


def test_grid_capacity_raises():
    cfg = SceneConfig(n_objects=(10, 10), fix_to_grid=True, placement_grid=(3, 3))
    with pytest.raises(FeasibilityError, match="cells"):
        cfg.validate()


# --- Core game constraints ------------------------------------------------ #

def test_target_unique_in_every_scene():
    gen = make_gen(distractor_policy="iid")
    for scene in (gen.samplers["train"].sample_scene() for _ in range(300)):
        target = next(o for o in scene if o.is_target)
        distractors = [o for o in scene if not o.is_target]
        assert all(o.combo != target.combo for o in distractors)
        assert sum(o.is_target for o in scene) == 1


def test_hard_negative_shares_all_but_one_attribute():
    gen = make_gen(
        distractor_policy="hard_negative",
        size="variable",
        size_bins=(14, 20, 26),
    )
    for scene in (gen.samplers["train"].sample_scene() for _ in range(200)):
        target = next(o for o in scene if o.is_target)
        diffs = [
            sum(a != b for a, b in zip(o.combo, target.combo))
            for o in scene
            if not o.is_target
        ]
        assert min(diffs) == 1  # at least one one-attribute-off distractor


def test_hard_negatives_cycle_attributes():
    # With 2 flippable attributes and 2 hard negatives, EVERY scene must
    # contain a shape-only-off and a color-only-off distractor, so no single
    # attribute ever identifies the target.
    gen = make_gen(distractor_policy="hard_negative", n_hard_negatives=2)
    for scene in (gen.samplers["train"].sample_scene() for _ in range(200)):
        target = next(o for o in scene if o.is_target)
        flipped_dims = set()
        for o in scene:
            if o.is_target:
                continue
            diff = [a != b for a, b in zip(o.combo, target.combo)]
            if sum(diff) == 1:
                flipped_dims.add(diff.index(True))
        assert {0, 1} <= flipped_dims  # both shape (0) and color (1) covered


def test_fixed_window_crop_preserves_size():
    gen = make_gen(
        n_objects=(1, 1),
        size="variable",
        size_bins=(14, 26),
        crop_mode="fixed_window",
    )
    assert gen.config.effective_crop_window == pytest.approx(1.4 * 26)
    batch = gen.generate_batch(64)
    assert batch.target_crop.shape == (64, 3, 64, 64)
    areas = (batch.target_crop.sum(dim=1) > 0.05).float().mean(dim=(1, 2))
    sizes = torch.tensor([next(o for o in s if o.is_target).size for s in batch.objects])
    small = areas[sizes < 20].mean()
    large = areas[sizes > 20].mean()
    # Foreground fraction must scale ~quadratically with object size.
    assert large / small == pytest.approx((26 / 14) ** 2, rel=0.15)


def test_fixed_window_too_small_raises():
    cfg = SceneConfig(
        n_objects=(1, 1), size="variable", size_bins=(14, 26),
        crop_mode="fixed_window", crop_window=20,
    )
    with pytest.raises(FeasibilityError, match="crop_window"):
        cfg.validate()


def test_variable_size_tight_crop_warns():
    cfg = SceneConfig(n_objects=(1, 1), size="variable", size_bins=(14, 26))
    with pytest.warns(UserWarning, match="cannot perceive object size"):
        cfg.validate()


def test_no_overlap_free_placement():
    gen = make_gen(n_objects=(5, 5))
    for scene in (gen.samplers["train"].sample_scene() for _ in range(100)):
        for a, b in itertools.combinations(scene, 2):
            ax0, ay0, ax1, ay1 = a.bbox
            bx0, by0, bx1, by1 = b.bbox
            assert ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0


def test_grid_placement_cells_exact():
    gen = make_gen(fix_to_grid=True, placement_grid=(4, 4))
    for scene in (gen.samplers["train"].sample_scene() for _ in range(50)):
        cells = [o.cell for o in scene]
        assert len(set(cells)) == len(cells)  # one object per cell
        rows, cols = 4, 4
        h, w = gen.config.canvas_hw
        for o in scene:
            r, c = o.cell // cols, o.cell % cols
            assert abs(o.cx - (c + 0.5) * w / cols) < 1e-6
            assert abs(o.cy - (r + 0.5) * h / rows) < 1e-6


def test_stratified_targets_balanced():
    gen = make_gen(
        stratify_targets=True,
        n_objects=(2, 3),
        shapes=("circle", "square"),
        colors=("red", "blue"),
    )
    n_combos = 4
    scenes = [gen.samplers["train"].sample_scene() for _ in range(n_combos * 25)]
    counts = {}
    for scene in scenes:
        t = next(o for o in scene if o.is_target)
        counts[t.combo] = counts.get(t.combo, 0) + 1
    assert set(counts.values()) == {25}  # exactly balanced


def test_compositional_holdout():
    gen = make_gen(holdout_combo_frac=0.2, n_objects=(3, 4))
    held = set(gen.samplers["test"].target_combos)
    seen = set(gen.samplers["train"].object_combos)
    assert held and not (held & seen)
    for scene in (gen.samplers["train"].sample_scene() for _ in range(100)):
        assert all(o.combo not in held for o in scene)
    for scene in (gen.samplers["test"].sample_scene() for _ in range(100)):
        target = next(o for o in scene if o.is_target)
        assert target.combo in held


def test_reproducible_across_instances():
    a = make_gen().samplers["train"].sample_scene()
    b = make_gen().samplers["train"].sample_scene()
    assert [o.to_dict() for o in a] == [o.to_dict() for o in b]


# --- Rendering and IO ------------------------------------------------------ #

def test_batch_shapes_and_ranges():
    gen = make_gen(crop_size=48)
    batch = gen.generate_batch(8)
    h, w = gen.config.canvas_hw
    assert batch.scene.shape == (8, 3, h, w)
    assert batch.target_crop.shape == (8, 3, 48, 48)
    assert batch.target_bbox.shape == (8, 4)
    assert batch.target_cell.shape == (8,)
    assert 0.0 <= batch.scene.min() and batch.scene.max() <= 1.0
    sample = batch[0]
    assert {"scene", "target_crop", "target_bbox", "target_cell", "objects", "meta"} <= set(sample)


def test_rendered_target_matches_bbox():
    # The scene must actually contain non-background pixels inside the bbox.
    gen = make_gen(n_objects=(1, 1))
    batch = gen.generate_batch(4)
    for i in range(4):
        x0, y0, x1, y1 = batch.target_bbox[i].round().long().tolist()
        inside = batch.scene[i, :, y0:y1, x0:x1]
        assert inside.sum() > 0
        assert batch.target_crop[i].sum() > 0


def test_save_and_load_roundtrip(tmp_path):
    gen = make_gen(n_samples=60, splits=(0.5, 0.25, 0.25))
    gen.save(tmp_path / "ds", batch_size=16, shard_size=10, progress=False)
    ds = SceneDataset(tmp_path / "ds", "train")
    assert len(ds) == 30
    s = ds[17]
    assert s["scene"].dtype == torch.float32
    assert any(o["is_target"] for o in s["objects"])
    meta = ds.load_metadata()
    assert len(meta) == 30 and meta[17]["target_cell"] == int(s["target_cell"])

    egg = EggSceneDataset(tmp_path / "ds", "val")
    sender_input, label, receiver_input, aux = egg[0]
    assert sender_input.shape[0] == 3 and receiver_input.shape[0] == 3
    assert aux["target_bbox"].shape == (4,)


def test_crop_classification_dataset(tmp_path):
    gen = make_gen(n_samples=40, splits=(1.0, 0.0, 0.0))
    gen.save(tmp_path / "ds", batch_size=16, shard_size=64, progress=False)
    ds = CropClassificationDataset(tmp_path / "ds", "train", attributes=("shape", "color"))
    assert len(ds) == 40
    assert ds.num_classes("shape") == len(gen.config.shapes)
    crop, labels = ds[3]
    assert crop.shape == (3, gen.config.crop_size, gen.config.crop_size)
    # Labels must agree with the stored ground truth for the target object.
    target = next(o for o in SceneDataset(tmp_path / "ds", "train")[3]["objects"] if o["is_target"])
    assert ds.classes["shape"][labels["shape"]] == target["shape"]
    assert ds.classes["color"][labels["color"]] == target["color"]

    loader = torch.utils.data.DataLoader(ds, batch_size=8)
    crops, labels = next(iter(loader))
    assert crops.shape[0] == 8 and labels["shape"].shape == (8,)


def test_cell_classification_dataset(tmp_path):
    gen = make_gen(
        n_samples=40, splits=(1.0, 0.0, 0.0), fix_to_grid=True, placement_grid=(4, 4)
    )
    gen.save(tmp_path / "ds", batch_size=16, shard_size=64, progress=False)
    ds = CellClassificationDataset(tmp_path / "ds", "train", attributes=("shape", "color"))
    assert len(ds) == 40
    n_shapes = len(gen.config.shapes)
    assert ds.num_classes("shape") == n_shapes + 1  # + "empty"
    empty_shape = ds.num_classes("shape") - 1
    empty_color = ds.num_classes("color") - 1

    scene, labels = ds[5]
    assert scene.shape == (3, *gen.config.canvas_hw)
    assert labels["shape"].shape == (16,) and labels["shape"].dtype == torch.long

    objects = SceneDataset(tmp_path / "ds", "train")[5]["objects"]
    occupied = {o["cell"] for o in objects}
    for o in objects:
        assert ds.classes["shape"][labels["shape"][o["cell"]]] == o["shape"]
        assert ds.classes["color"][labels["color"][o["cell"]]] == o["color"]
    for cell in set(range(16)) - occupied:
        assert labels["shape"][cell] == empty_shape
        assert labels["color"][cell] == empty_color

    loader = torch.utils.data.DataLoader(ds, batch_size=8)
    scenes, labels = next(iter(loader))
    assert scenes.shape[0] == 8 and labels["shape"].shape == (8, 16)


def test_cell_classification_requires_grid(tmp_path):
    gen = make_gen(n_samples=20, splits=(1.0, 0.0, 0.0))  # free placement
    gen.save(tmp_path / "ds", batch_size=16, shard_size=64, progress=False)
    with pytest.raises(ValueError, match="fix_to_grid"):
        CellClassificationDataset(tmp_path / "ds", "train")


def test_centroid_heatmap_dataset(tmp_path):
    gen = make_gen(n_samples=30, splits=(1.0, 0.0, 0.0))  # free placement
    gen.save(tmp_path / "ds", batch_size=16, shard_size=64, progress=False)
    ds = CentroidHeatmapDataset(tmp_path / "ds", "train", stride=4)
    gy, gx = ds.grid_hw
    scene, targets = ds[7]
    assert targets["heatmap"].shape == (gy, gx)
    assert targets["shape"].shape == (gy * gx,)
    # Peak value depends on where the centroid falls within its stride cell;
    # with sigma >= stride the on-grid peak is always > 0.75.
    assert targets["heatmap"].max() > 0.75

    objects = SceneDataset(tmp_path / "ds", "train")[7]["objects"]
    centers = {}
    for o in objects:
        x0, y0, x1, y1 = o["bbox"]
        cell = ds._center_cell((x0 + x1) / 2, (y0 + y1) / 2)
        centers[cell] = o
    for cell, o in centers.items():
        assert ds.classes["shape"][targets["shape"][cell]] == o["shape"]
        assert ds.classes["color"][targets["color"][cell]] == o["color"]
        assert targets["heatmap"].flatten()[cell] > 0.6  # near a Gaussian peak
    off = torch.ones(gy * gx, dtype=torch.bool)
    off[list(centers)] = False
    assert (targets["shape"][off] == -1).all()

    loader = torch.utils.data.DataLoader(ds, batch_size=8)
    scenes, t = next(iter(loader))
    assert scenes.shape[0] == 8 and t["heatmap"].shape == (8, gy, gx)


def test_load_dataset_views(tmp_path):
    gen = make_gen(n_samples=20, splits=(1.0, 0.0, 0.0), fix_to_grid=True)
    gen.save(tmp_path / "ds", batch_size=16, shard_size=64, progress=False)
    assert isinstance(load_dataset(tmp_path / "ds"), SceneDataset)
    assert isinstance(load_dataset(tmp_path / "ds", "train", "egg"), EggSceneDataset)
    ds = load_dataset(tmp_path / "ds", "train", "crop_classification", attributes=("shape",))
    assert isinstance(ds, CropClassificationDataset) and ds.attributes == ("shape",)
    assert isinstance(
        load_dataset(tmp_path / "ds", "train", "cell_classification"),
        CellClassificationDataset,
    )
    with pytest.raises(ValueError, match="Unknown view"):
        load_dataset(tmp_path / "ds", "train", "nope")


def test_position_marginals_roughly_uniform():
    gen = make_gen(n_objects=(4, 4))
    xs = []
    for scene in (gen.samplers["train"].sample_scene() for _ in range(500)):
        t = next(o for o in scene if o.is_target)
        xs.append(t.cx)
    w = gen.config.canvas_hw[1]
    # Target x-position should span the canvas, not cluster.
    hist, _ = np.histogram(xs, bins=4, range=(0, w))
    assert hist.min() > 0.5 * hist.mean()
