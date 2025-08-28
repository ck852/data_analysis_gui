# params.py
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple, Iterable

@dataclass(frozen=True)
class AxisConfig:
    measure: str
    channel: Optional[str]

@dataclass(frozen=True)
class AnalysisParameters:
    range1_start: float
    range1_end: float
    use_dual_range: bool
    range2_start: Optional[float]
    range2_end: Optional[float]
    stimulus_period: float
    x_axis: AxisConfig
    y_axis: AxisConfig
    channel_config: Dict[str, Any] = field(default_factory=dict)

    def cache_key(self) -> Tuple:
        """Immutable, order-invariant cache key derived from fields."""
        def r(x):
            return round(x, 9) if isinstance(x, (float, int)) else x

        # If channel_config ever contains lists/tuples, freeze them:
        def freeze(v):
            if isinstance(v, dict):
                return tuple(sorted((k, freeze(vv)) for k, vv in v.items()))
            if isinstance(v, (list, tuple)):
                return tuple(freeze(x) for x in v)
            return v

        return (
            r(self.range1_start),
            r(self.range1_end),
            self.use_dual_range,
            r(self.range2_start) if self.range2_start is not None else None,
            r(self.range2_end) if self.range2_end is not None else None,
            r(self.stimulus_period),
            (self.x_axis.measure, self.x_axis.channel),
            (self.y_axis.measure, self.y_axis.channel),
            # order-invariant, hashable representation of channel_config
            freeze(self.channel_config),
        )

    # ---------- universal compat shim (dict-like) ----------
    _legacy_aliases = {
        "dual_range": "use_dual_range",
        "stim_period": "stimulus_period",
        # add more aliases if you encounter them
    }

    def _flatten(self) -> Dict[str, Any]:
        """Flatten to a legacy-friendly dict once; cheap and predictable."""
        base = {
            "range1_start": self.range1_start,
            "range1_end": self.range1_end,
            "use_dual_range": self.use_dual_range,
            "range2_start": self.range2_start,
            "range2_end": self.range2_end,
            "stimulus_period": self.stimulus_period,
            "x_axis": self.x_axis,   # keep nested objects available
            "y_axis": self.y_axis,
            "channel_config": self.channel_config,
            # legacy flattened keys expected by older code:
            "x_measure": self.x_axis.measure,
            "x_channel": self.x_axis.channel,
            "y_measure": self.y_axis.measure,
            "y_channel": self.y_axis.channel,
        }
        # add alias keys mirroring the canonical ones
        for alias, canon in self._legacy_aliases.items():
            base[alias] = base[canon]
        return base

    def to_legacy_dict(self) -> Dict[str, Any]:
        """If something truly needs a plain dict (e.g., JSON or templating)."""
        d = self._flatten().copy()
        # Convert dataclasses to dicts for JSON-friendliness
        d["x_axis"] = asdict(self.x_axis)
        d["y_axis"] = asdict(self.y_axis)
        return d

    # Mapping-like methods
    def __getitem__(self, key: str) -> Any:
        flat = self._flatten()
        if key in flat:
            return flat[key]
        # Allow direct attribute names as fallback (e.g., "x_axis", "y_axis")
        try:
            return getattr(self, key)
        except AttributeError as e:
            raise KeyError(key) from e

    def get(self, key: str, default=None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        flat = self._flatten()
        return key in flat or hasattr(self, key)

    def keys(self) -> Iterable[str]:
        flat = self._flatten()
        # Combine flattened keys with dataclass field names to be generous
        return set(flat.keys()) | {
            "range1_start","range1_end","use_dual_range","range2_start","range2_end",
            "stimulus_period","x_axis","y_axis","channel_config"
        }

    def items(self) -> Iterable[Tuple[str, Any]]:
        flat = self._flatten()
        for k in self.keys():
            yield (k, flat[k] if k in flat else getattr(self, k))