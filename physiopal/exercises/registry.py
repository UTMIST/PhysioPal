"""Name -> Exercise class lookup, so the runner can do --exercise squat."""

from __future__ import annotations

from .base import Exercise

_REGISTRY: dict[str, type[Exercise]] = {}


def register(cls: type[Exercise]) -> type[Exercise]:
    """Put ``@register`` above your Exercise class."""
    if cls.name in _REGISTRY:
        raise ValueError(f"Exercise {cls.name!r} registered twice")
    _REGISTRY[cls.name] = cls
    return cls


def build_exercise(name: str) -> Exercise:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise KeyError(f"Unknown exercise {name!r}. Available: {available()}") from None


def available() -> list[str]:
    return sorted(_REGISTRY)
