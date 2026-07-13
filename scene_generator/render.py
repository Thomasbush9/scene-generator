"""Batched, device-agnostic scene rasterizer.

Scenes are rendered analytically from the shapes' signed-distance functions:
for every pixel, coverage alpha = clamp(0.5 - distance_px, 0, 1), giving
~1px anti-aliasing with no supersampling. Everything is vectorized over the
batch, so rendering runs efficiently on CUDA / MPS / CPU alike.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .config import SceneConfig
from .sampler import ObjectSpec
from .shapes import SHAPES


def resolve_device(device: str = "auto") -> torch.device:
    """Map "auto" to the best available accelerator (cuda > mps > cpu)."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Renderer:
    """Rasterizes batches of symbolic scenes into (B, 3, H, W) float images."""

    def __init__(self, config: SceneConfig, device: str | torch.device = "auto"):
        self.config = config
        self.device = resolve_device(device) if isinstance(device, str) else device
        h, w = config.canvas_hw
        # Pixel-center coordinates, shared by every SDF evaluation.
        ys = torch.arange(h, dtype=torch.float32, device=self.device) + 0.5
        xs = torch.arange(w, dtype=torch.float32, device=self.device) + 0.5
        self._Y, self._X = torch.meshgrid(ys, xs, indexing="ij")
        self._bg = (
            torch.tensor(config.background_rgb, dtype=torch.float32, device=self.device) / 255.0
        )
        self._shape_names = list(config.shapes)
        self._shape_index = {name: i for i, name in enumerate(self._shape_names)}
        palette = config.full_palette
        self._color_rgb = {
            name: torch.tensor(palette[name], dtype=torch.float32, device=self.device) / 255.0
            for name in config.colors
        }

    @torch.no_grad()
    def render(self, scenes: list[list[ObjectSpec]]) -> Tensor:
        """Render a batch of scenes; returns (B, 3, H, W) float32 in [0, 1]."""
        b = len(scenes)
        k_max = max(len(s) for s in scenes)
        h, w = self.config.canvas_hw
        dev = self.device

        # Pack ragged scenes into padded (B, K) tensors; padded slots are invalid.
        shape_ids = torch.full((b, k_max), -1, dtype=torch.long)
        centers = torch.zeros(b, k_max, 2)
        halves = torch.ones(b, k_max)
        colors = torch.zeros(b, k_max, 3)
        for i, scene in enumerate(scenes):
            for j, obj in enumerate(scene):
                shape_ids[i, j] = self._shape_index[obj.shape]
                centers[i, j, 0] = obj.cx
                centers[i, j, 1] = obj.cy
                halves[i, j] = obj.size / 2.0
                colors[i, j] = self._color_rgb[obj.color].cpu()
        shape_ids = shape_ids.to(dev)
        centers = centers.to(dev)
        halves = halves.to(dev)
        colors = colors.to(dev)

        img = self._bg.view(1, 3, 1, 1).expand(b, 3, h, w).clone()
        # Composite slot by slot (objects don't overlap by default, so order
        # is irrelevant; with overlap="allow" later slots paint on top).
        for j in range(k_max):
            alpha = torch.zeros(b, h, w, device=dev)
            for name, s_id in self._shape_index.items():
                sel = (shape_ids[:, j] == s_id).nonzero(as_tuple=True)[0]
                if sel.numel() == 0:
                    continue
                cx = centers[sel, j, 0].view(-1, 1, 1)
                cy = centers[sel, j, 1].view(-1, 1, 1)
                hf = halves[sel, j].view(-1, 1, 1)
                # Unit-box, y-up coordinates expected by the shape SDFs.
                p = torch.stack(((self._X - cx) / hf, (cy - self._Y) / hf), dim=-1)
                d_px = SHAPES[name].sdf(p) * hf
                alpha[sel] = (0.5 - d_px).clamp(0.0, 1.0)
            a = alpha.unsqueeze(1)
            img = img * (1.0 - a) + colors[:, j].view(b, 3, 1, 1) * a
        return img

    @torch.no_grad()
    def crop_targets(self, images: Tensor, bboxes: Tensor) -> Tensor:
        """Extract fixed-size target crops with roi_align (batched, on-device).

        images: (B, 3, H, W); bboxes: (B, 4) pixel xyxy. Returns
        (B, 3, crop_size, crop_size).

        crop_mode="tight": bbox + crop_pad, clamped to the canvas, resized —
        absolute size is normalized away. crop_mode="fixed_window": a
        crop_window-sized box centered on the target, NOT clamped (clamping
        would change the scale for near-edge targets); the out-of-canvas
        region samples as zeros, i.e. black background.
        """
        from torchvision.ops import roi_align

        cfg = self.config
        h, w = cfg.canvas_hw
        boxes = bboxes.to(images.device, torch.float32).clone()
        if cfg.crop_mode == "fixed_window":
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            half = cfg.effective_crop_window / 2.0
            boxes = torch.cat([centers - half, centers + half], dim=1)
        else:
            boxes[:, :2] -= cfg.crop_pad
            boxes[:, 2:] += cfg.crop_pad
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
        idx = torch.arange(len(boxes), dtype=torch.float32, device=images.device).unsqueeze(1)
        rois = torch.cat([idx, boxes], dim=1)
        return roi_align(images, rois, output_size=cfg.crop_size, aligned=True)
