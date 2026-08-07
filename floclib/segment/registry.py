# -*- coding: utf-8 -*-
"""
floclib/segment/registry.py
===========================

Operation registry for the image-segmentation stage.

The registry is the seam that lets users bring their own preprocessing /
segmentation / post-processing steps without floclib baking every OpenCV /
scikit-image call into the library:

* Built-in ops live in :mod:`floclib.segment.ops` and decorate themselves
  with ``@register`` at import time.
* Users add their own with::

      from floclib.segment import register

      @register("my_denoise", kind="preprocess")
      def my_denoise(image, radius=3):
          ...

* The callable escape hatch: any place that accepts an op also accepts a
  bare Python callable, which is wrapped transparently in a :class:`FuncOp`.

Ops carry their parameters so a whole pipeline can be serialized to a
config dict / YAML and reproduced later (the CellProfiler/Galaxy model).
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

import numpy as np

__all__ = [
    "Op",
    "FuncOp",
    "register",
    "get_op",
    "list_ops",
    "OP_REGISTRY",
]

# name -> registered class (Op subclass) or bare callable (wrapped on demand)
OP_REGISTRY: dict[str, Any] = {}
# name -> kind ("preprocess" | "segment" | "post")
OP_KINDS: dict[str, str] = {}

#: the three valid op stages, in pipeline order
VALID_KINDS = ("preprocess", "segment", "post")


class Op:
    """Base class for a segmentation-pipeline operation.

    Subclasses define:
      * ``_defaults`` — mapping of parameter name -> default value.
      * ``__call__(self, image)`` — apply the op, returning either an image
        (preprocess / post on a mask) or an integer label array (segment / post).

    The class attributes ``name`` and ``kind`` are set by :func:`register`.
    """

    name: str = "op"
    kind: str = "preprocess"
    _defaults: dict[str, Any] = {}

    def __init__(self, **params: Any) -> None:
        unknown = [k for k in params if k not in self._defaults]
        if unknown:
            raise TypeError(
                f"{type(self).__name__} got unexpected parameter(s): {unknown}. "
                f"Known: {list(self._defaults)}"
            )
        self.params: dict[str, Any] = {**self._defaults, **params}

    # --- to be overridden -------------------------------------------------
    def __call__(self, image: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    # --- config (de)serialization ----------------------------------------
    def to_config(self) -> dict[str, Any]:
        return {"op": self.name, "kind": self.kind, "params": dict(self.params)}

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Op":
        return get_op(cfg["op"], **cfg.get("params", {}))

    def __repr__(self) -> str:
        ps = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({ps})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Op)
            and type(self) is type(other)
            and self.name == other.name
            and self.params == other.params
        )


class FuncOp(Op):
    """Wrap a user-supplied callable as an :class:`Op`.

    Used both as the explicit escape hatch and as the implementation of
    ``@register`` applied to a bare function.  User callables reference
    arbitrary code, so their ``to_config()`` records the name + params but
    they can only be restored from config if the callable is re-registered
    under the same name in the running session.
    """

    def __init__(
        self,
        func: Callable[..., np.ndarray],
        *,
        kind: str = "preprocess",
        name: Optional[str] = None,
        **params: Any,
    ) -> None:
        self.func = func
        self.kind = kind
        self.name = name or getattr(func, "__name__", "func")
        self._defaults = {}
        self.params = dict(params)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.func(image, **self.params)

    def to_config(self) -> dict[str, Any]:
        return {
            "op": self.name,
            "kind": self.kind,
            "params": dict(self.params),
            "_callable": True,
        }

    def __repr__(self) -> str:
        ps = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"FuncOp({self.name!r}, kind={self.kind!r}{', ' + ps if ps else ''})"


def _func_defaults(func: Callable) -> dict[str, Any]:
    """Extract callable parameter defaults (excluding the leading ``image``)."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return {}
    defaults: dict[str, Any] = {}
    params = list(sig.parameters.values())
    # skip the first positional (image) if present
    for p in params[1:]:
        if p.default is not inspect.Parameter.empty:
            defaults[p.name] = p.default
    return defaults


def register(name: Optional[str] = None, kind: Optional[str] = None):
    """Register an :class:`Op` subclass or bare callable under ``name``.

    Examples
    --------
    As a class decorator::

        @register("clahe", kind="preprocess")
        class CLAHE(Op):
            _defaults = {"clip_limit": 2.0, "tile_size": 8}
            def __call__(self, image): ...

    As a function decorator (becomes a :class:`FuncOp`)::

        @register("my_denoise", kind="preprocess")
        def my_denoise(image, radius=3): ...

    The decorated object is returned unchanged so it keeps working as a
    plain class/function; it is simply also reachable via :func:`get_op`.
    """

    def decorator(obj):
        actual_name = name or getattr(obj, "__name__", None)
        if not actual_name:
            raise ValueError("register() requires a name for this object")

        if kind is not None and kind not in VALID_KINDS:
            raise ValueError(
                f"kind must be one of {VALID_KINDS}, got {kind!r}"
            )

        if inspect.isclass(obj) and issubclass(obj, Op):
            obj.name = actual_name
            if kind is not None:
                obj.kind = kind
            _store(actual_name, obj, obj.kind)
            return obj

        if callable(obj) and not inspect.isclass(obj):
            # bare callable -> register the function; get_op wraps it in FuncOp
            k = kind or "preprocess"
            _store(actual_name, obj, k)
            # attach defaults so get_op can validate/serialize
            if k not in (None,) and not hasattr(obj, "_defaults"):
                obj._defaults = _func_defaults(obj)  # type: ignore[attr-defined]
            return obj

        raise TypeError("register() can decorate an Op subclass or a callable")

    return decorator


def _store(name: str, obj: Any, kind: str) -> None:
    if name in OP_REGISTRY and OP_REGISTRY[name] is not obj:
        raise ValueError(
            f"An op named {name!r} is already registered "
            f"({OP_REGISTRY[name]!r}). Choose a different name."
        )
    OP_REGISTRY[name] = obj
    OP_KINDS[name] = kind


def get_op(name: str, **params: Any) -> Op:
    """Instantiate a registered op by name with the given parameters.

    Raises a :class:`KeyError` listing the available ops when ``name`` is
    unknown.  A bare-registered callable is wrapped in :class:`FuncOp`.
    """
    if name not in OP_REGISTRY:
        raise KeyError(
            f"No op registered as {name!r}. Available: {list_ops()}"
        )
    entry = OP_REGISTRY[name]
    kind = OP_KINDS[name]
    if inspect.isclass(entry) and issubclass(entry, Op):
        return entry(**params)
    # bare callable registered via @register on a function
    return FuncOp(entry, kind=kind, name=name, **params)


def list_ops(kind: Optional[str] = None) -> list[str]:
    """List registered op names, optionally filtered by ``kind``."""
    if kind is None:
        return sorted(OP_REGISTRY)
    return sorted(n for n in OP_REGISTRY if OP_KINDS[n] == kind)
