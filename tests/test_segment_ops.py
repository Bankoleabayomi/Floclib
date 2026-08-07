# -*- coding: utf-8 -*-
"""Tests for segment ops + measures on a synthetic image with known circles."""
import numpy as np
import pytest

# skip the whole module gracefully if seg deps aren't installed
cv2 = pytest.importorskip("cv2")
skimage = pytest.importorskip("skimage")


def _make_circle_image(n=8, size=(256, 256), radius=20, gap=64, value=200, seed=0):
    """Grayscale image with `n` bright circles on a dark background.

    ``gap`` is the spacing between circle centers; it must exceed the
    diameter (``2*radius``) so circles don't touch and merge into one
    connected component under 8-connectivity labelling.
    """
    img = np.zeros(size, dtype=np.uint8)
    for i in range(n):
        cx = 30 + (i % 4) * gap
        cy = 30 + (i // 4) * gap
        cv2.circle(img, (cx, cy), radius, value, -1)
    return img


def test_threshold_otsu_finds_circles():
    from floclib.segment import get_op
    img = _make_circle_image(n=8, radius=28)
    labels = get_op("threshold_otsu")(img)
    n_labels = labels.max()
    assert n_labels == 8, f"expected 8 objects, got {n_labels}"


def test_remove_small_objects_after_otsu():
    from floclib.segment import Compose, get_op
    img = _make_circle_image(n=6, radius=28)
    # add speckles
    rng = np.random.default_rng(1)
    noise = (rng.random(img.shape) < 0.005).astype(np.uint8) * 255
    img = np.clip(img.astype(int) + noise.astype(int), 0, 255).astype(np.uint8)
    labels_raw = get_op("threshold_otsu")(img)
    n_raw = len(np.unique(labels_raw)) - 1
    c = Compose([get_op("threshold_otsu"), get_op("remove_small_objects", min_size=200)])
    labels = c(img)
    # skimage.remove_small_objects preserves label numbers (does not relabel),
    # so count non-zero unique labels rather than using labels.max().
    n_clean = len(np.unique(labels)) - 1
    assert n_clean <= 6
    assert n_clean < n_raw  # speckles were actually removed


def test_measure_particles_columns_and_scaling():
    from floclib.segment import get_op
    from floclib.segment.measures import measure_particles
    img = _make_circle_image(n=4, radius=25, value=255)
    labels = get_op("threshold_otsu")(img)
    df = measure_particles(labels, pixels_to_um=0.5,
                           image="img.png", condition="C1", tf=2, gf=18.0)
    expected = {"Gf", "Condition", "Tf", "Image", "Particle_num",
                "area", "equivalent_diameter_area", "longest_length",
                "perimeter", "aspect_ratio", "eccentricity"}
    assert expected.issubset(df.columns)
    assert len(df) == 4
    # a circle of radius 25px * 0.5 um/px -> diameter ~25 um
    diam = df["equivalent_diameter_area"].mean()
    assert 22 < diam < 28, f"diameter {diam} not ≈ 25 um"


def test_fill_holes():
    from floclib.segment import Compose, get_op
    # draw a ring (filled circle minus inner circle)
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img, (50, 50), 25, 255, -1)
    cv2.circle(img, (50, 50), 12, 0, -1)
    labels_no_fill = get_op("threshold_otsu")(img)
    labels_fill = Compose([get_op("threshold_otsu"), get_op("fill_holes")])(img)
    from floclib.segment.measures import measure_particles
    a_no = measure_particles(labels_no_fill, 1.0)["area"].sum()
    a_fill = measure_particles(labels_fill, 1.0)["area"].sum()
    assert a_fill > a_no


def test_fractal_dimension_unit():
    from floclib.segment.measures import fractal_dimension
    # area ∝ L^2  -> Df ≈ 2 for circles
    L = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    area = np.pi * (L / 2) ** 2
    df = fractal_dimension(area, L)
    assert 1.8 < df < 2.2


def test_clear_border():
    from floclib.segment import Compose, get_op
    img = np.zeros((100, 100), dtype=np.uint8)
    # one object in center, one touching the corner
    cv2.circle(img, (50, 50), 15, 255, -1)
    cv2.circle(img, (0, 0), 20, 255, -1)  # unmistakably border-touching
    labels = Compose([get_op("threshold_otsu"), get_op("clear_border")])(img)
    nonzero = len(np.unique(labels)) - 1  # subtract background (0)
    assert nonzero == 1  # only the center object remains


def test_watershed_separates_touching():
    from floclib.segment import get_op
    # two circles close together
    img = np.zeros((128, 256), dtype=np.uint8)
    cv2.circle(img, (90, 64), 40, 200, -1)
    cv2.circle(img, (160, 64), 40, 200, -1)
    labels_otsu = get_op("threshold_otsu")(img)
    labels_ws = get_op("watershed", min_distance=50, peak_footprint=60)(img)
    assert labels_ws.max() >= labels_otsu.max()
