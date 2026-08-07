# -*- coding: utf-8 -*-
"""Tests for the segment ops registry + FuncOp escape hatch."""
import numpy as np
import pytest

from floclib.segment import register, get_op, list_ops


def test_builtin_ops_registered():
    names = list_ops()
    for expected in ["threshold_otsu", "median_blur", "remove_small_objects", "watershed"]:
        assert expected in names, f"missing built-in op {expected}"


def test_list_ops_filtered_by_kind():
    seg = list_ops(kind="segment")
    pre = list_ops(kind="preprocess")
    post = list_ops(kind="post")
    assert "threshold_otsu" in seg and "threshold_otsu" not in pre
    assert "median_blur" in pre and "median_blur" not in post
    assert "remove_small_objects" in post


def test_get_op_instantiates_with_params():
    op = get_op("median_blur", ksize=7)
    assert op.name == "median_blur"
    assert op.params["ksize"] == 7


def test_get_op_unknown_raises_with_catalog():
    with pytest.raises(KeyError) as exc:
        get_op("nope_op")
    assert "threshold_otsu" in str(exc.value)  # catalog in the message


def test_register_function_decorator():
    @register("double_brightness", kind="preprocess")
    def double_brightness(image, factor=2):
        return image * factor

    op = get_op("double_brightness", factor=3)
    img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    out = op(img)
    assert out[0, 0] == 3 and out[1, 1] == 12


def test_register_op_subclass():
    from floclib.segment.registry import Op

    @register("add_constant", kind="preprocess")
    class AddConstant(Op):
        _defaults = {"value": 1}
        def __call__(self, image):
            return image + self.params["value"]

    op = get_op("add_constant", value=5)
    img = np.zeros((2, 2), dtype=np.uint8)
    assert op(img).sum() == 20


def test_config_round_trip():
    from floclib.segment import Compose
    op = get_op("median_blur", ksize=5)
    cfg = op.to_config()
    restored = get_op(cfg["op"], **cfg["params"])
    assert restored == op


def test_compose_config_round_trip():
    from floclib.segment import Compose
    c = Compose([get_op("median_blur", ksize=3), get_op("threshold_otsu")])
    cfg = c.to_config()
    assert cfg[0]["op"] == "median_blur"
    assert cfg[1]["op"] == "threshold_otsu"


def test_compose_enforces_order():
    from floclib.segment import Compose
    # preprocess after segment must raise
    with pytest.raises(TypeError):
        Compose([get_op("threshold_otsu"), get_op("median_blur")])


def test_funcop_callable_escape_hatch():
    from floclib.segment import Compose
    # bare callable as a step
    c = Compose([lambda img: img * 0])  # default kind preprocess
    assert isinstance(c.steps[0].kind, str)


def test_duplicate_name_raises():
    with pytest.raises(ValueError):
        @register("median_blur", kind="preprocess")  # already registered
        def dup(image):
            return image
