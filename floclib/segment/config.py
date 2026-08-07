# -*- coding: utf-8 -*-
"""
floclib/segment/config.py
=========================

Serialize a :class:`~floclib.pipeline.Pipeline` to a config dict / YAML
and rebuild it.

A config is a plain dict so an experiment is reproducible from a file —
the CellProfiler / Galaxy model.  Example YAML::

    pixels_to_um: 0.27
    size_col: longest_length
    image_ext: [".jpg", ".png", ".tif", ".tiff"]
    bins: [0.3, 2.375, 0.125]
    preprocess:
      - {op: median_blur, params: {ksize: 3}}
    segment:
      {op: threshold_otsu, params: {invert: false}}
    post:
      - {op: remove_small_objects, params: {min_size: 50}}

Bare callables (the escape hatch) can be described by name only if they
have been re-registered via ``@register`` in the running session; their
``_callable`` marker is recorded so :func:`config_to_compose` raises a
clear error if the callable is missing on load.
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = [
    "compose_to_config",
    "config_to_compose",
    "pipeline_to_config",
    "to_yaml",
    "from_yaml",
]


def compose_to_config(compose) -> list[dict[str, Any]]:
    """Serialize a :class:`~floclib.segment.Compose` (or list of ops) to a list of dicts."""
    if compose is None:
        return []
    ops = getattr(compose, "steps", compose)
    return [op.to_config() for op in ops]


def config_to_compose(cfg_list, *, default_kind: Optional[str] = None):
    """Rebuild a :class:`~floclib.segment.Compose` from a list of op config dicts."""
    from .registry import get_op, OP_REGISTRY
    from . import Compose

    if not cfg_list:
        return None
    steps = []
    for cfg in cfg_list:
        name = cfg["op"]
        params = cfg.get("params", {})
        is_callable = cfg.get("_callable", False)
        if is_callable and name not in OP_REGISTRY:
            raise KeyError(
                f"Config references callable op {name!r} which is not registered "
                f"in this session. Register it with @register({name!r}, ...) "
                f"before loading this config."
            )
        steps.append(get_op(name, **params))
    return Compose(steps, default_kind=default_kind)


def pipeline_to_config(pipe) -> dict[str, Any]:
    """Serialize a :class:`~floclib.pipeline.Pipeline` to a plain dict."""
    cfg: dict[str, Any] = {
        "pixels_to_um": pipe.pixels_to_um,
        "size_col": pipe.size_col,
        "image_ext": list(pipe.image_ext) if not isinstance(pipe.image_ext, str) else [pipe.image_ext],
        "bins": _bins_to_config(pipe.bins) if pipe.bins is not None else None,
        "min_size": pipe.min_size,
        "max_size": pipe.max_size,
        "interval": pipe.interval,
        "preprocess": compose_to_config(pipe.preprocess),
        "segment": _seg_to_config(pipe.segment),
        "post": compose_to_config(pipe.post),
    }
    if pipe.condition_pattern is not None:
        cfg["condition_pattern"] = pipe.condition_pattern
    return cfg


def _bins_to_config(bins) -> list[float]:
    try:
        return [float(b) for b in bins]
    except TypeError:
        # numpy array
        import numpy as np
        return np.asarray(bins, dtype=float).tolist()


def _seg_to_config(seg):
    """segment may be a single Op or a Compose."""
    if seg is None:
        return None
    if isinstance(seg, list):
        return compose_to_config(seg)
    # single Op
    return [seg.to_config()]


def to_yaml(path: str, cfg: dict[str, Any]) -> str:
    """Write a config dict to ``path`` as YAML if PyYAML is available."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "YAML serialization requires pyyaml. "
            "Install it with:  pip install floclib[yaml]"
        ) from e
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def from_yaml(path: str) -> dict[str, Any]:
    """Read a YAML config file into a plain dict."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "YAML serialization requires pyyaml. "
            "Install it with:  pip install floclib[yaml]"
        ) from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
