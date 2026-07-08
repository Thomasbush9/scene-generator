# scene-generator

Synthetic-scene generator for a **referential-localization emergent-communication
game** built on [EGG](https://github.com/facebookresearch/EGG): a **Sender** sees
only a crop of the target object and sends a message; a **Receiver** sees the full
scene and must localize the target (grid-cell prediction).

Each generated sample is a scene of colored shapes with exactly one
attribute-unique target, plus the target crop (Sender input), the ground-truth
bbox / grid-cell label (Receiver targets), and per-object metadata for
mutual-information analyses of the emergent protocol. Rendering is batched,
anti-aliased (SDF-based), and device-agnostic (`cuda` / `mps` / `cpu`), while all
sampling is CPU-seeded so datasets are bit-identical across devices.

- **Full documentation:** [`scene_generator/README.md`](scene_generator/README.md)
- **Design spec:** [`data/scene_generator.md`](data/scene_generator.md)

## Install

```bash
pip install "git+https://github.com/Thomasbush9/scene-generator.git"
# or, for development:
git clone git@github.com:Thomasbush9/scene-generator.git && cd scene-generator
pip install -e ".[dev]" && pytest scene_generator/tests
```

## Quick start

```python
from scene_generator import SceneConfig, SceneGenerator, load_dataset

cfg = SceneConfig(n_objects=(3, 4), fix_to_grid=True, n_samples=10_000)
SceneGenerator(cfg, device="auto").save("data/scenes_v1")

game     = load_dataset("data/scenes_v1", "train", view="egg")                 # (sender_input, labels, receiver_input, aux)
sender   = load_dataset("data/scenes_v1", "train", view="crop_classification") # Sender vision pretraining
receiver = load_dataset("data/scenes_v1", "train", view="cell_classification") # Receiver vision pretraining
```

CLI: `python -m scene_generator --config cfg.yaml --out data/scenes_v1 --n 50000 --device cuda`
