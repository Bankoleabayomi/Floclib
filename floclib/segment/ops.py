# -*- coding: utf-8 -*-
"""
floclib/segment/ops.py
======================

Curated catalog of image-processing ops for the floclib segmentation stage.

The catalog is deliberately finite (~16 ops) — enough to cover the
overwhelming majority of floc images — and extended infinitely via the
:class:`~floclib.segment.registry.register` decorator and the callable
escape hatch (see :mod:`floclib.segment.registry`).

Every op **lazy-imports** its imaging dependencies (cv2 / scikit-image /
scipy.ndimage) inside ``__call__``.  This means ``import floclib.segment``
succeeds even without the ``[seg]`` extra installed; the op only raises a
clear :class:`ImportError` (with install hint) when it is actually invoked.

Op kinds
--------
* ``preprocess``  — grayscale image  ->  grayscale image
* ``segment``     — grayscale image  ->  integer label array (0 background)
* ``post``        — label array      ->  label array

A forward pipeline must be ordered ``preprocess*`` then ``segment`` then
``post*``; :class:`~floclib.segment.Compose` enforces this ordering.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .registry import Op, register

__all__ = [
    # preprocess
    "GaussianBlur", "MedianBlur", "CLAHE", "BilateralFilter",
    "Grayscale", "MorphOpenGray", "MorphCloseGray",
    # segment
    "ThresholdManual", "ThresholdOtsu", "ThresholdAdaptive",
    "ThresholdTriangle", "Watershed",
    # post
    "RemoveSmallObjects", "FillHoles", "ClearBorder",
    "MorphOpen", "MorphClose",
]

# 8-connectivity structure for scipy.ndimage.label
_EIGHT_CONN = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

_INSTALL_HINT = (
    "Image-segmentation ops require the optional image dependencies. "
    "Install them with:  pip install floclib[seg]"
)


def _require_cv2():
    try:
        import cv2
        return cv2
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(_INSTALL_HINT) from e


def _require_skimage():
    try:
        import skimage
        return skimage
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(_INSTALL_HINT) from e


def _require_ndimage():
    try:
        from scipy import ndimage
        return ndimage
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(_INSTALL_HINT) from e


def _label(mask: np.ndarray) -> np.ndarray:
    """Label a boolean/binary mask with 8-connectivity, returning int labels."""
    ndimage = _require_ndimage()
    labeled, _ = ndimage.label(np.asarray(mask).astype(bool), structure=_EIGHT_CONN)
    return labeled


# ============================================================================
#  PREPROCESS  (gray -> gray)
# ============================================================================

@register("gaussian_blur", kind="preprocess")
class GaussianBlur(Op):
    """Gaussian blur (OpenCV)."""

    _defaults = {"ksize": 5}

    def __call__(self, image: np.ndarray) -> np.ndarray:
        cv2 = _require_cv2()
        k = int(self.params["ksize"])
        if k <= 0:
            return image
        if k % 2 == 0:
            k += 1  # cv2 requires odd ksize
        return cv2.GaussianBlur(image, (k, k), 0)


@register("median_blur", kind="preprocess")
class MedianBlur(Op):
    """Median blur (OpenCV)."""

    _defaults = {"ksize": 5}

    def __call__(self, image: np.ndarray) -> np.ndarray:
        cv2 = _require_cv2()
        k = int(self.params["ksize"])
        if k <= 0:
            return image
        if k % 2 == 0:
            k += 1
        return cv2.medianBlur(image, k)


@register("clahe", kind="preprocess")
class CLAHE(Op):
    """Contrast-Limited Adaptive Histogram Equalization (OpenCV)."""

    _defaults = {"clip_limit": 2.0, "tile_size": 8}

    def __call__(self, image: np.ndarray) -> np.ndarray:
        cv2 = _require_cv2()
        img = image
        if img.ndim == 3:  # CLAHE on luminance
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=float(self.params["clip_limit"]),
            tileGridSize=(int(self.params["tile_size"]),) * 2,
        )
        return clahe.apply(img)


@register("bilateral_filter", kind="preprocess")
class BilateralFilter(Op):
    """Edge-preserving bilateral filter (OpenCV)."""

    _defaults = {"d": 9, "sigma_color": 75.0, "sigma_space": 75.0}

    def __call__(self, image: np.ndarray) -> np.ndarray:
        cv2 = _require_cv2()
        return cv2.bilateralFilter(
            image,
            int(self.params["d"]),
            float(self.params["sigma_color"]),
            float(self.params["sigma_space"]),
        )


@register("grayscale", kind="preprocess")
class Grayscale(Op):
    """Force a single-channel grayscale image (passthrough if already 2D)."""

    _defaults: dict[str, Any] = {}

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        cv2 = _require_cv2()
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image


@register("morph_open_gray", kind="preprocess")
class MorphOpenGray(Op):
    """Grayscale morphological opening (erosion then dilation)."""

    _defaults = {"kernel_size": 3}

    def __call__(self, image: np.ndarray) -> np.ndarray:
        cv2 = _require_cv2()
        k = int(self.params["kernel_size"])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


@register("morph_close_gray", kind="preprocess")
class MorphCloseGray(Op):
    """Grayscale morphological closing (dilation then erosion)."""

    _defaults = {"kernel_size": 3}

    def __call__(self, image: np.ndarray) -> np.ndarray:
        cv2 = _require_cv2()
        k = int(self.params["kernel_size"])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


# ============================================================================
#  SEGMENT  (gray -> labels)
# ============================================================================

def _mask_from_above(image: np.ndarray, threshold: float, invert: bool) -> np.ndarray:
    return image > threshold if not invert else image < threshold


@register("threshold_manual", kind="segment")
class ThresholdManual(Op):
    """Fixed-threshold binarization then 8-connectivity labeling.

    ``invert=False`` (default) treats pixels *above* ``value`` as flocs;
    ``invert=True`` treats pixels *below* ``value`` as flocs (light flocs
    on a dark background).
    """

    _defaults = {"value": 128, "invert": False}

    def __call__(self, image: np.ndarray) -> np.ndarray:
        cv2 = _require_cv2()
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flag = cv2.THRESH_BINARY_INV if self.params["invert"] else cv2.THRESH_BINARY
        _, t = cv2.threshold(image, float(self.params["value"]), 255, flag)
        return _label(t == 255)


@register("threshold_otsu", kind="segment")
class ThresholdOtsu(Op):
    """Otsu's auto threshold then labeling."""

    _defaults = {"invert": False}

    def __call__(self, image: np.ndarray) -> np.ndarray:
        skimage = _require_skimage()
        cv2 = _require_cv2()
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = skimage.filters.threshold_otsu(image)
        return _label(_mask_from_above(image, float(thresh), bool(self.params["invert"])))


@register("threshold_adaptive", kind="segment")
class ThresholdAdaptive(Op):
    """Adaptive (local) Gaussian threshold (OpenCV) then labeling."""

    _defaults = {"block_size": 15, "C": 5, "invert": False}

    def __call__(self, image: np.ndarray) -> np.ndarray:
        cv2 = _require_cv2()
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bs = int(self.params["block_size"])
        if bs % 2 == 0:
            bs += 1  # must be odd and >1
        flag = cv2.THRESH_BINARY_INV if self.params["invert"] else cv2.THRESH_BINARY
        t = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, flag, bs, float(self.params["C"])
        )
        return _label(t == 255)


@register("threshold_triangle", kind="segment")
class ThresholdTriangle(Op):
    """Triangle auto threshold (skimage) then labeling — good for skewed histograms."""

    _defaults = {"invert": False}

    def __call__(self, image: np.ndarray) -> np.ndarray:
        skimage = _require_skimage()
        cv2 = _require_cv2()
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = skimage.filters.threshold_triangle(image)
        return _label(_mask_from_above(image, float(thresh), bool(self.params["invert"])))


@register("watershed", kind="segment")
class Watershed(Op):
    """Marker-controlled watershed to separate touching flocs.

    Builds markers from local maxima of the Euclidean distance transform
    and floods from them using :func:`skimage.segmentation.watershed`.
    """

    _defaults = {"min_distance": 20, "peak_footprint": 20}

    def __call__(self, image: np.ndarray) -> np.ndarray:
        skimage = _require_skimage()
        ndimage = _require_ndimage()
        cv2 = _require_cv2()
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Otsu mask as the flooding foreground
        thresh = skimage.filters.threshold_otsu(image)
        mask = image > thresh
        distance = ndimage.distance_transform_edt(mask)
        coords = skimage.feature.peak_local_max(
            distance,
            min_distance=int(self.params["min_distance"]),
            footprint=np.ones((int(self.params["peak_footprint"]),) * 2),
            labels=mask.astype(int),
        )
        if len(coords) == 0:
            return _label(mask)
        markers = np.zeros(distance.shape, dtype=int)
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
        labels = skimage.segmentation.watershed(-distance, markers, mask=mask)
        return labels.astype(int)


# ============================================================================
#  POST  (labels -> labels)
# ============================================================================

@register("remove_small_objects", kind="post")
class RemoveSmallObjects(Op):
    """Drop objects smaller than ``min_size`` pixels (skimage)."""

    _defaults = {"min_size": 50}

    def __call__(self, labels: np.ndarray) -> np.ndarray:
        import warnings
        skimage = _require_skimage()
        labels = labels.astype(int)
        # skimage emits a spurious "Only one label was provided ... did you mean
        # to use a boolean array?" UserWarning when a label image legitimately
        # contains a single object.  Suppress that false alarm only.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Only one label was provided",
                category=UserWarning,
            )
            return skimage.morphology.remove_small_objects(
                labels, min_size=int(self.params["min_size"])
            )


@register("fill_holes", kind="post")
class FillHoles(Op):
    """Fill interior holes within each object, then re-label."""

    _defaults: dict[str, Any] = {}

    def __call__(self, labels: np.ndarray) -> np.ndarray:
        ndimage = _require_ndimage()
        mask = np.asarray(labels) > 0
        filled = ndimage.binary_fill_holes(mask)
        return _label(filled)


@register("clear_border", kind="post")
class ClearBorder(Op):
    """Remove objects touching the image border (skimage)."""

    _defaults: dict[str, Any] = {}

    def __call__(self, labels: np.ndarray) -> np.ndarray:
        skimage = _require_skimage()
        labels = labels.astype(int)
        return skimage.segmentation.clear_border(labels)


@register("morph_open", kind="post")
class MorphOpen(Op):
    """Binary morphological opening on the mask, then re-label. Cleans specks."""

    _defaults = {"kernel_size": 3}

    def __call__(self, labels: np.ndarray) -> np.ndarray:
        cv2 = _require_cv2()
        k = int(self.params["kernel_size"])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = (np.asarray(labels) > 0).astype(np.uint8) * 255
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return _label(opened == 255)


@register("morph_close", kind="post")
class MorphClose(Op):
    """Binary morphological closing on the mask, then re-label. Fills small gaps."""

    _defaults = {"kernel_size": 3}

    def __call__(self, labels: np.ndarray) -> np.ndarray:
        cv2 = _require_cv2()
        k = int(self.params["kernel_size"])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = (np.asarray(labels) > 0).astype(np.uint8) * 255
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return _label(closed == 255)
