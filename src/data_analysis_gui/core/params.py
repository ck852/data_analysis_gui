# params.py
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple, Iterable, ClassVar

@dataclass(frozen=True)
class AxisConfig:
    measure: str
    channel: Optional[str]
    peak_type: Optional[str] = "Absolute"  # Options: "Absolute", "Positive", "Negative", "Peak-Peak"

@dataclass  # Removed frozen=True to allow mutation via dict-style access
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

    # Legacy aliases for backward compatibility
    _legacy_aliases: ClassVar[Dict[str, str]] = {
        "start1": "range1_start",
        "end1": "range1_end",
        "start2": "range2_start",
        "end2": "range2_end",
        # Removed broken "capacitance": "capacitance_pf" - field doesn't exist
    }

    def __post_init__(self):
        """Validate parameters on creation."""
        if self.range1_end <= self.range1_start:
            raise ValueError(f"Range 1 end ({self.range1_end}) must be after start ({self.range1_start})")
        
        if self.use_dual_range:
            if self.range2_start is None or self.range2_end is None:
                raise ValueError("Dual range enabled but range 2 values not provided")
            if self.range2_end <= self.range2_start:
                raise ValueError(f"Range 2 end ({self.range2_end}) must be after start ({self.range2_start})")

    def cache_key(self) -> Tuple:
        """Immutable, order-invariant cache key for memoization."""
        def r(x):
            return round(x, 9) if isinstance(x, (float, int)) else x

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
            freeze(asdict(self.x_axis)),
            freeze(asdict(self.y_axis)),
            freeze(self.channel_config)
        )

    def _flatten(self) -> Dict[str, Any]:
        """Flattens the dataclass into a simple dictionary for easy access."""
        flat_dict = {
            "range1_start": self.range1_start,
            "range1_end": self.range1_end,
            "use_dual_range": self.use_dual_range,
            "range2_start": self.range2_start,
            "range2_end": self.range2_end,
            "stimulus_period": self.stimulus_period,
            "x_measure": self.x_axis.measure,
            "x_channel": self.x_axis.channel,
            "x_peak_type": self.x_axis.peak_type,
            "y_measure": self.y_axis.measure,
            "y_channel": self.y_axis.channel,
            "y_peak_type": self.y_axis.peak_type,
            "channel_config": self.channel_config,
        }
        # Add legacy aliases for backward compatibility
        for alias, canon in self._legacy_aliases.items():
            if canon in flat_dict:
                flat_dict[alias] = flat_dict[canon]
        return flat_dict

    def to_dict(self) -> Dict[str, Any]:
        """Returns a plain dict for JSON serialization or templating."""
        return self._flatten()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnalysisParameters":
        """Create from dictionary (e.g., from GUI or JSON)."""
        # Handle legacy aliases in input
        for alias, canon in cls._legacy_aliases.items():
            if alias in d and canon not in d:
                d[canon] = d[alias]
        
        return cls(
            range1_start=d.get("range1_start", 0.0),
            range1_end=d.get("range1_end", 100.0),
            use_dual_range=d.get("use_dual_range", False),
            range2_start=d.get("range2_start"),
            range2_end=d.get("range2_end"),
            stimulus_period=d.get("stimulus_period", 1000.0),
            x_axis=AxisConfig(
                measure=d.get("x_measure", "Average"),
                channel=d.get("x_channel"),
                peak_type=d.get("x_peak_type", "Absolute")
            ),
            y_axis=AxisConfig(
                measure=d.get("y_measure", "Average"),
                channel=d.get("y_channel"),
                peak_type=d.get("y_peak_type", "Absolute")
            ),
            channel_config=d.get("channel_config", {})
        )

    # Dict-like methods for backward compatibility
    def __getitem__(self, key: str) -> Any:
        """Support dict-style read access."""
        flat = self._flatten()
        if key in flat:
            return flat[key]
        # Allow direct attribute names as fallback
        try:
            return getattr(self, key)
        except AttributeError as e:
            raise KeyError(key) from e

    def __setitem__(self, key: str, value: Any):
        """Support dict-style write access."""
        # Handle legacy aliases
        if key in self._legacy_aliases:
            key = self._legacy_aliases[key]
        
        # Handle flattened keys for nested AxisConfig
        if key == "x_measure":
            self.x_axis = AxisConfig(value, self.x_axis.channel, self.x_axis.peak_type)
        elif key == "x_channel":
            self.x_axis = AxisConfig(self.x_axis.measure, value, self.x_axis.peak_type)
        elif key == "x_peak_type":
            self.x_axis = AxisConfig(self.x_axis.measure, self.x_axis.channel, value)
        elif key == "y_measure":
            self.y_axis = AxisConfig(value, self.y_axis.channel, self.y_axis.peak_type)
        elif key == "y_channel":
            self.y_axis = AxisConfig(self.y_axis.measure, value, self.y_axis.peak_type)
        elif key == "y_peak_type":
            self.y_axis = AxisConfig(self.y_axis.measure, self.y_axis.channel, value)
        elif hasattr(self, key):
            # Direct attribute
            setattr(self, key, value)
        else:
            raise KeyError(f"Unknown parameter: {key}")

    def get(self, key: str, default=None) -> Any:
        """Dict-like get with default."""
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        """Check if key exists."""
        if not isinstance(key, str):
            return False
        flat = self._flatten()
        return key in flat or hasattr(self, key)

    def keys(self) -> Iterable[str]:
        """Return all accessible keys."""
        return self._flatten().keys()

    def values(self) -> Iterable[Any]:
        """Return all values."""
        flat = self._flatten()
        return flat.values()

    def items(self) -> Iterable[Tuple[str, Any]]:
        """Return key-value pairs."""
        flat = self._flatten()
        return flat.items()

    def copy(self) -> "AnalysisParameters":
        """Create a shallow copy."""
        return AnalysisParameters(
            range1_start=self.range1_start,
            range1_end=self.range1_end,
            use_dual_range=self.use_dual_range,
            range2_start=self.range2_start,
            range2_end=self.range2_end,
            stimulus_period=self.stimulus_period,
            x_axis=AxisConfig(
                self.x_axis.measure,
                self.x_axis.channel,
                self.x_axis.peak_type
            ),
            y_axis=AxisConfig(
                self.y_axis.measure,
                self.y_axis.channel,
                self.y_axis.peak_type
            ),
            channel_config=self.channel_config.copy()
        )