# -*- coding: utf-8 -*-
"""
floclib.segment — composable image-segmentation stage.

Public API
----------
* :class:`Compose` — ordered list of ops; validates the
  ``preprocess* → segment → post*`` type contract.
* :func:`register`, :func:`get_op`, :func:`list_ops` — the ops registry.
* The curated ops (re-exported when :mod:`floclib.segment.ops` imports).

Importing this subpackage does **not** import cv2 / scikit-image — those
are lazy-loaded inside each op's ``__call__``.  See the ``[seg]`` extra.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np

from .registry import Op, FuncOp, register, get_op, list_ops, OP_REGISTRY
from . import ops as _ops  # noqa: F401 — triggers registration of built-in ops
from .ops import (  # re-export for convenience
    GaussianBlur, MedianBlur, CLAHE, BilateralFilter, Grayscale,
    MorphOpenGray, MorphCloseGray,
    ThresholdManual, ThresholdOtsu, ThresholdAdaptive, ThresholdTriangle,
    Watershed,
    RemoveSmallObjects, FillHoles, ClearBorder, MorphOpen, MorphClose,
)

__all__ = [
    "Compose",
    "register", "get_op", "list_ops",
    "Op", "FuncOp",
    # ops
    "GaussianBlur", "MedianBlur", "CLAHE", "BilateralFilter", "Grayscale",
    "MorphOpenGray", "MorphCloseGray",
    "ThresholdManual", "ThresholdOtsu", "ThresholdAdaptive",
    "ThresholdTriangle", "Watershed",
    "RemoveSmallObjects", "FillHoles", "ClearBorder", "MorphOpen", "MorphClose",
]

#: order in which op kinds must appear in a forward pipeline
_KIND_ORDER = {"preprocess": 0, "segment": 1, "post": 2}


class Compose:
    """An ordered list of :class:`Op` steps.

    Acts like the Albumentations/scikit-image-pipeline idiom: a pipeline is
    just a list.  ``Compose`` enforces that op kinds are monotonic in
    :data:`_KIND_ORDER` (``preprocess`` → ``segment`` → ``post``); a
    ``preprocess`` op after a ``segment`` op raises a clear :class:`TypeError`.

    Any step may also be a bare Python callable, which is wrapped in a
    :class:`FuncOp` of the expected ``default_kind`` (or inferred to the next
    permissible kind).
    """

    def __init__(
        self,
        steps: Optional[Sequence[Any]] = None,
        *,
        default_kind: str = "preprocess",
    ) -> None:
        self.steps: list[Op] = []
        self.default_kind = default_kind
        if steps:
            for s in steps:
                self.append(s)

    def append(self, op: Any, kind: Optional[str] = None) -> "Compose":
        coerced = _coerce_op(op, kind or self._next_kind())
        self._validate_kind(coerced.kind)
        self.steps.append(coerced)
        return self

    def _next_kind(self) -> str:
        if not self.steps:
            return self.default_kind
        return self.steps[-1].kind

    def _validate_kind(self, kind: str) -> None:
        if kind not in _KIND_ORDER:
            raise ValueError(f"Unknown op kind {kind!r}; expected one of {list(_KIND_ORDER)}")
        if self.steps and _KIND_ORDER[kind] < _KIND_ORDER[self.steps[-1].kind]:
            raise TypeError(
                f"Op ordering violated: a {kind!r} op cannot follow a "
                f"{self.steps[-1].kind!r} op. Forward order is "
                f"preprocess -> segment -> post."
            )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        out = image
        for op in self.steps:
            out = op(out)
        return out

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, i) -> Op:
        return self.steps[i]

    def __repr__(self) -> str:
        return f"Compose({self.steps!r})"

    def to_config(self) -> list[dict[str, Any]]:
        return [op.to_config() for op in self.steps]


def _coerce_op(op: Any, expected_kind: Optional[str]) -> Op:
    """Normalize a step to an :class:`Op`, inferring kind when possible."""
    if isinstance(op, Op):
        return op
    if callable(op):
        # bare callable -> FuncOp; infer kind from expected if unset
        kind = expected_kind or "preprocess"
        return FuncOp(op, kind=kind)
    raise TypeError(
        f"Each step must be an Op or a callable, got {type(op).__name__}"
    )
