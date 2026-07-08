"""CLI: generate a dataset from a YAML/JSON config, or preview sample scenes.

Examples::

    python -m scene_generator --out data/scenes_v1 --n 50000 --device cuda \
        --config my_config.yaml
    python -m scene_generator --preview preview.png --config my_config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import SceneConfig
from .generator import SceneGenerator


def load_config(path: str | None) -> SceneConfig:
    if path is None:
        return SceneConfig(n_objects=(4, 4))
    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        import yaml

        with open(p) as f:
            return SceneConfig.from_dict(yaml.safe_load(f))
    return SceneConfig.from_json(p)


def save_preview(gen: SceneGenerator, path: str, n: int = 16) -> None:
    import torchvision.utils as vutils

    batch = gen.generate_batch(n, split="train")
    vutils.save_image(batch.scene, path, nrow=4, padding=2, pad_value=0.5)
    crop_path = str(Path(path).with_stem(Path(path).stem + "_crops"))
    vutils.save_image(batch.target_crop, crop_path, nrow=4, padding=2, pad_value=0.5)
    print(f"Wrote {path} (scenes) and {crop_path} (sender crops)")


def main() -> None:
    ap = argparse.ArgumentParser(prog="scene_generator", description=__doc__)
    ap.add_argument("--config", help="YAML or JSON file with SceneConfig fields")
    ap.add_argument("--out", help="Output dataset directory")
    ap.add_argument("--n", type=int, help="Total samples (overrides config.n_samples)")
    ap.add_argument("--seed", type=int, help="Override config.seed")
    ap.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--preview", metavar="PNG", help="Render a preview grid instead of a dataset")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    gen = SceneGenerator(cfg, device=args.device)
    print(f"Device: {gen.device}")

    if args.preview:
        save_preview(gen, args.preview)
        return
    if not args.out:
        ap.error("either --out or --preview is required")
    out = gen.save(args.out, n_samples=args.n, batch_size=args.batch_size)
    print(f"Dataset written to {out}")


if __name__ == "__main__":
    main()
