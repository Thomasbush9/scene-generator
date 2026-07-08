"""Shape vocabulary as signed-distance functions (SDFs).

Each shape is defined on the unit box: centered at the origin with half-extent
1.0, in y-up coordinates. The renderer rescales pixel coordinates into this
frame, so shapes are resolution- and size-agnostic and can be rasterized with
analytic anti-aliasing on any torch device.

New shapes: subclass ``Shape`` with a unique ``name`` — registration into
``SHAPES`` is automatic.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor

SHAPES: dict[str, "Shape"] = {}


class Shape(ABC):
    """A 2D shape defined by a signed distance function on the unit box."""

    name: str = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name:
            SHAPES[cls.name] = cls()

    @abstractmethod
    def sdf(self, p: Tensor) -> Tensor:
        """Signed distance for points ``p`` of shape (..., 2) (negative inside)."""


class Circle(Shape):
    name = "circle"

    def sdf(self, p: Tensor) -> Tensor:
        return torch.linalg.vector_norm(p, dim=-1) - 1.0


class Square(Shape):
    name = "square"

    def sdf(self, p: Tensor) -> Tensor:
        return p.abs().amax(dim=-1) - 1.0


class Diamond(Shape):
    name = "diamond"

    def sdf(self, p: Tensor) -> Tensor:
        return (p.abs().sum(dim=-1) - 1.0) / math.sqrt(2.0)


class Triangle(Shape):
    """Equilateral triangle, apex up, vertically centered in the unit box."""

    name = "triangle"

    def sdf(self, p: Tensor) -> Tensor:
        k = math.sqrt(3.0)
        x = p[..., 0].abs() - 1.0
        y = p[..., 1] + k / 2.0
        flip = x + k * y > 0.0
        x2 = (x - k * y) / 2.0
        y2 = (-k * x - y) / 2.0
        x = torch.where(flip, x2, x)
        y = torch.where(flip, y2, y)
        x = x - x.clamp(-2.0, 0.0)
        return -torch.sqrt(x * x + y * y) * torch.sign(y)


class Plus(Shape):
    """Axis-aligned cross: the union of a horizontal and a vertical bar."""

    name = "plus"
    _bar = 0.34  # bar half-thickness

    @staticmethod
    def _box(p: Tensor, bx: float, by: float) -> Tensor:
        qx = p[..., 0].abs() - bx
        qy = p[..., 1].abs() - by
        outside = torch.sqrt(qx.clamp(min=0.0) ** 2 + qy.clamp(min=0.0) ** 2)
        inside = torch.maximum(qx, qy).clamp(max=0.0)
        return outside + inside

    def sdf(self, p: Tensor) -> Tensor:
        return torch.minimum(self._box(p, 1.0, self._bar), self._box(p, self._bar, 1.0))


class Star(Shape):
    """Five-pointed star, one tip up, circumscribed by the unit circle."""

    name = "star"
    _rf = 0.5  # inner/outer radius ratio

    def sdf(self, p: Tensor) -> Tensor:
        k1x, k1y = 0.809016994, -0.587785252
        k2x, k2y = -k1x, k1y
        px = p[..., 0].abs()
        py = p[..., 1]
        f = 2.0 * (px * k1x + py * k1y).clamp(min=0.0)
        px, py = px - f * k1x, py - f * k1y
        f = 2.0 * (px * k2x + py * k2y).clamp(min=0.0)
        px, py = px - f * k2x, py - f * k2y
        px = px.abs()
        py = py - 1.0
        bax = self._rf * (-k1y)
        bay = self._rf * k1x - 1.0
        h = ((px * bax + py * bay) / (bax * bax + bay * bay)).clamp(0.0, 1.0)
        dx, dy = px - bax * h, py - bay * h
        return torch.sqrt(dx * dx + dy * dy) * torch.sign(py * bax - px * bay)
