from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

FeatureParams = dict[str, Any]
FeatureSpec = tuple[str, FeatureParams]


class Feature(ABC):
    """Minimal OOP contract each feature implements."""

    def __init__(self, params: FeatureParams | None = None) -> None:
        self.params: FeatureParams = params or {}

    @abstractmethod
    def compute(self, views: "FeatureViews") -> pd.DataFrame:
        ...


class FeatureRegistry:
    """Registry that owns the lifecycle of feature implementations."""

    def __init__(self) -> None:
        self._items: dict[str, type[Feature]] = {}

    @staticmethod
    def _normalize(raw: str) -> str:
        key = raw.strip()
        if key.endswith("Feature"):
            key = key[:-7]
        key = key.lower()
        if not key:
            raise ValueError("Feature name cannot be empty.")
        return key

    def register(self, cls: type[Feature], *, alias: str | None = None) -> type[Feature]:
        key = self._normalize(alias or cls.__name__)
        if key in self._items:
            raise ValueError(f"Feature '{key}' is already registered.")
        self._items[key] = cls
        return cls

    def decorator(
        self, cls: type[Feature] | None = None, /, *, name: str | None = None
    ):
        def _register(target: type[Feature]) -> type[Feature]:
            return self.register(target, alias=name)

        if cls is None:
            return _register
        return _register(cls)

    def create(self, name: str, params: FeatureParams | None = None) -> Feature:
        key = self._normalize(name)
        try:
            feature_cls = self._items[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"Unknown feature '{key}'. Available: [{available}]") from exc
        return feature_cls(params or {})

    def __contains__(self, name: str) -> bool:
        try:
            key = self._normalize(name)
        except ValueError:
            return False
        return key in self._items


FEATURE_REGISTRY = FeatureRegistry()
# Legacy alias that mirrors previous module-level mapping.
FEATURES: dict[str, type[Feature]] = FEATURE_REGISTRY._items


def register_feature(
    cls: type[Feature] | None = None, /, *, name: str | None = None
):  # noqa: D401 - documented by FeatureRegistry
    return FEATURE_REGISTRY.decorator(cls, name=name)


@dataclass(frozen=True)
class FeatureRequest:
    name: str
    params: FeatureParams = field(default_factory=dict)

    @classmethod
    def _from_entry(cls, entry: Any) -> "FeatureRequest | None":
        if isinstance(entry, str):
            name = entry.strip().lower()
            if not name:
                return None
            return cls(name=name, params={})
        if isinstance(entry, Mapping):
            raw_name = str(entry.get("name", "")).strip().lower()
            if not raw_name:
                return None
            params = {k: v for k, v in entry.items() if k != "name"}
            return cls(name=raw_name, params=dict(params))
        if isinstance(entry, tuple) and entry:
            name = str(entry[0]).strip().lower()
            if not name:
                return None
            mapping = entry[1] if len(entry) > 1 and isinstance(entry[1], Mapping) else {}
            return cls(name=name, params=dict(mapping))
        if entry is None:
            return None
        raise TypeError("feature_defs must contain strings, mappings, or (name, params) tuples.")

    @classmethod
    def parse_many(cls, items: Iterable[Any]) -> tuple["FeatureRequest", ...]:
        parsed = [req for entry in items if (req := cls._from_entry(entry)) is not None]
        return tuple(parsed)


@dataclass(frozen=True)
class FeaturePanel:
    values: np.ndarray  # (time, assets, features)
    mask: np.ndarray  # (time, assets) bool mask

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError("values must be a 3D tensor shaped as (time, assets, features).")
        mask = np.asarray(self.mask, dtype=bool)
        if mask.shape != self.values.shape[:2]:
            raise ValueError("mask must match the first two dimensions of values.")
        object.__setattr__(self, "mask", mask)

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray]:
        return self.values, self.mask


@dataclass(frozen=True)
class FeatureBuilder:
    requests: tuple[FeatureRequest, ...]
    normalization: str = "none"
    volume_window: int = 20
    zero_as_invalid: bool = False
    registry: FeatureRegistry = field(default=FEATURE_REGISTRY)

    def __post_init__(self) -> None:
        if not self.requests:
            raise ValueError("feature_defs must provide at least one feature name.")
        if self.volume_window <= 0:
            raise ValueError("volume_window must be positive.")

    @classmethod
    def from_defs(
        cls,
        feature_defs: Iterable[Any],
        *,
        normalization: str = "none",
        volume_window: int = 20,
        zero_as_invalid: bool = False,
        registry: FeatureRegistry = FEATURE_REGISTRY,
    ) -> "FeatureBuilder":
        requests = FeatureRequest.parse_many(feature_defs)
        return cls(
            requests=requests,
            normalization=normalization,
            volume_window=volume_window,
            zero_as_invalid=zero_as_invalid,
            registry=registry,
        )

    def build(self, frames: Mapping[str, pd.DataFrame]) -> FeaturePanel:
        views = FeatureViews.from_frames(frames, self.volume_window)
        tensors: list[np.ndarray] = []
        for req in self.requests:
            feature = self.registry.create(req.name, req.params)
            tensors.append(feature.compute(views).to_numpy(copy=True))

        values = np.stack(tensors, axis=-1)
        values = normalize(values, self.normalization)

        mask = np.isfinite(values).all(axis=2)
        if self.zero_as_invalid:
            mask &= values.any(axis=2)

        values = np.nan_to_num(values, nan=0.0).astype(np.float32, copy=False)
        return FeaturePanel(values=values, mask=mask)


def build_feature_panel(
    frames: Mapping[str, pd.DataFrame],
    *,
    feature_defs: Iterable[Any],
    normalization: str = "none",
    volume_window: int = 20,
    zero_as_invalid: bool = False,
) -> FeaturePanel:
    builder = FeatureBuilder.from_defs(
        feature_defs,
        normalization=normalization,
        volume_window=volume_window,
        zero_as_invalid=zero_as_invalid,
    )
    return builder.build(frames)


def build_features(
    frames: Mapping[str, pd.DataFrame],
    *,
    feature_defs: Iterable[Any],
    normalization: str = "none",
    volume_window: int = 20,
    zero_as_invalid: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    panel = build_feature_panel(
        frames,
        feature_defs=feature_defs,
        normalization=normalization,
        volume_window=volume_window,
        zero_as_invalid=zero_as_invalid,
    )
    return panel.as_tuple()


@dataclass(frozen=True)
class FeatureViews:
    close: pd.DataFrame
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    volume_window: int
    log_ret: pd.DataFrame = field(init=False, repr=False)
    hl_spread: pd.DataFrame = field(init=False, repr=False)
    oc_gap: pd.DataFrame = field(init=False, repr=False)
    volume_z: pd.DataFrame = field(init=False, repr=False)

    @classmethod
    def from_frames(cls, frames: Mapping[str, pd.DataFrame], volume_window: int) -> "FeatureViews":
        required = ("close", "open", "high", "low", "volume")
        missing = [name for name in required if name not in frames]
        if missing:
            raise KeyError(f"Missing price frames: {missing}")

        return cls(
            close=frames["close"],
            open=frames["open"],
            high=frames["high"],
            low=frames["low"],
            volume=frames["volume"],
            volume_window=volume_window,
        )

    def __post_init__(self) -> None:
        if self.volume_window <= 0:
            raise ValueError("volume_window must be positive.")

        close = self.close.replace(0.0, np.nan)
        prev_close = close.shift(1)

        object.__setattr__(self, "log_ret", np.log(close / prev_close))
        object.__setattr__(self, "hl_spread", (self.high - self.low) / close)
        object.__setattr__(self, "oc_gap", (self.open - prev_close) / prev_close)

        mean_volume = self.volume.rolling(self.volume_window, min_periods=1).mean().replace(0.0, np.nan)
        volume_z = (self.volume / mean_volume) - 1.0
        object.__setattr__(self, "volume_z", volume_z)


@register_feature
class LogReturnFeature(Feature):
    def compute(self, views: FeatureViews) -> pd.DataFrame:
        return views.log_ret


@register_feature
class HlSpreadFeature(Feature):
    def compute(self, views: FeatureViews) -> pd.DataFrame:
        return views.hl_spread


@register_feature
class OcGapFeature(Feature):
    def compute(self, views: FeatureViews) -> pd.DataFrame:
        return views.oc_gap


@register_feature
class VolumeZFeature(Feature):
    def compute(self, views: FeatureViews) -> pd.DataFrame:
        return views.volume_z


@register_feature
class MaGapFeature(Feature):
    def compute(self, views: FeatureViews) -> pd.DataFrame:
        window = max(1, int(self.params.get("window", 20)))
        ma = views.close.rolling(window, min_periods=1).mean()
        return (views.close / ma.replace(0.0, np.nan)) - 1.0


@register_feature
class RollingVolFeature(Feature):
    def compute(self, views: FeatureViews) -> pd.DataFrame:
        window = max(2, int(self.params.get("window", 20)))
        return views.log_ret.rolling(window, min_periods=1).std()


def normalize(panel: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or "none").lower()
    if mode == "none":
        return panel

    data = np.array(panel, copy=True)
    if mode == "asset":
        mean = np.nanmean(data, axis=0, keepdims=True)
        std = np.nanstd(data, axis=0, keepdims=True)
    elif mode == "cross":
        mean = np.nanmean(data, axis=1, keepdims=True)
        std = np.nanstd(data, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown normalization '{mode}'.")

    std = np.where(std == 0.0, 1.0, std)
    return (data - mean) / std


# Example:
# panel, mask = build_features(
#     frames=data_frames,
#     feature_defs=[
#         "log_ret",
#         {"name": "ma_gap", "window": 10},
#         ("rolling_vol", {"window": 63}),
#     ],
#     normalization="asset",
#     volume_window=20,
#     zero_as_invalid=True,
# )
