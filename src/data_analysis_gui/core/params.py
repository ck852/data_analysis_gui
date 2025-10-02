"""
PatchBatch Electrophysiology Data Analysis Tool.

Analysis parameters data model.

This module defines immutable data classes for configuring analysis operations
and plot axes in PatchBatch electrophysiology data analysis.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple


@dataclass(frozen=True)
class AxisConfig:
    """
    Immutable configuration for a plot axis (X or Y).

    Specifies how data should be extracted and displayed for a particular axis.

    Args:
        measure: Type of measurement ("Time", "Average", or "Peak").
        channel: Channel type ("Voltage" or "Current"), or None for Time.
        peak_type: Peak type for "Peak" measure ("Absolute", "Positive", "Negative", "Peak-Peak").
    """

    measure: str  # "Time", "Average", or "Peak"
    channel: Optional[str]  # "Voltage" or "Current" (None for Time)
    peak_type: Optional[str] = (
        "Absolute"  # "Absolute", "Positive", "Negative", "Peak-Peak"
    )


@dataclass(frozen=True)
class AnalysisParameters:
    """
    Immutable parameters for analysis operations.

    Encapsulates all parameters needed for an analysis operation, with validation
    at construction time to ensure data integrity.

    Args:
        range1_start: Start time in ms for Range 1.
        range1_end: End time in ms for Range 1.
        use_dual_range: Whether to use dual range analysis.
        range2_start: Start time in ms for Range 2 (if dual range).
        range2_end: End time in ms for Range 2 (if dual range).
        x_axis: AxisConfig for X-axis.
        y_axis: AxisConfig for Y-axis.
        channel_config: Dictionary mapping channel configuration.

    Example:
        >>> params = AnalysisParameters(
        ...     range1_start=150.0,
        ...     range1_end=500.0,
        ...     use_dual_range=False,
        ...     range2_start=None,
        ...     range2_end=None,
        ...     x_axis=AxisConfig(measure="Average", channel="Voltage"),
        ...     y_axis=AxisConfig(measure="Peak", channel="Current", peak_type="Absolute"),
        ...     channel_config={'voltage': 0, 'current': 1}
        ... )
    """

    # Range configuration
    range1_start: float  # Start time in ms for Range 1
    range1_end: float  # End time in ms for Range 1
    use_dual_range: bool  # Whether to use dual range analysis
    range2_start: Optional[float]  # Start time in ms for Range 2 (if dual range)
    range2_end: Optional[float]  # End time in ms for Range 2 (if dual range)

    # Axis configurations
    x_axis: AxisConfig  # Configuration for X-axis
    y_axis: AxisConfig  # Configuration for Y-axis

    # Channel mapping
    channel_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Validate parameters on creation.

        Ensures that:
            - Range end times are after start times.
            - Dual range parameters are provided when dual range is enabled.

        Raises:
            ValueError: If validation fails.
        """
        # Validate Range 1
        if self.range1_end <= self.range1_start:
            raise ValueError(
                f"Range 1 end ({self.range1_end}) must be after start ({self.range1_start})"
            )

        # Validate Range 2 if dual range is enabled
        if self.use_dual_range:
            if self.range2_start is None or self.range2_end is None:
                raise ValueError("Dual range enabled but range 2 values not provided")
            if self.range2_end <= self.range2_start:
                raise ValueError(
                    f"Range 2 end ({self.range2_end}) must be after start ({self.range2_start})"
                )

    def cache_key(self) -> Tuple:
        """
        Generate an immutable, hashable cache key for this parameter set.

        Used for caching analysis results. Includes all parameters that affect
        the analysis output, rounded to avoid floating point precision issues.

        Returns:
            Tuple suitable for use as a dictionary key.
        """

        def round_value(x):
            """Round numeric values to avoid floating point issues."""
            return round(x, 9) if isinstance(x, (float, int)) else x

        def freeze_value(v):
            """Convert mutable structures to immutable tuples."""
            if isinstance(v, dict):
                return tuple(sorted((k, freeze_value(vv)) for k, vv in v.items()))
            if isinstance(v, (list, tuple)):
                return tuple(freeze_value(x) for x in v)
            return v

        # Create cache key with all relevant parameters
        return (
            round_value(self.range1_start),
            round_value(self.range1_end),
            self.use_dual_range,
            round_value(self.range2_start) if self.range2_start is not None else None,
            round_value(self.range2_end) if self.range2_end is not None else None,
            freeze_value(asdict(self.x_axis)),
            freeze_value(asdict(self.y_axis)),
            freeze_value(self.channel_config),
        )

    def with_updates(self, **kwargs) -> "AnalysisParameters":
        """
        Create a new AnalysisParameters instance with updated values.

        Since this class is immutable (frozen), this method provides a way to
        create modified copies.

        Args:
            **kwargs: Keyword arguments for fields to update.

        Returns:
            New AnalysisParameters instance with updated values.

        Example:
            >>> new_params = params.with_updates(range1_start=200.0)
        """
        # Get current values as dict
        current = asdict(self)

        # Update with provided values
        current.update(kwargs)

        # Handle nested AxisConfig objects
        if "x_axis" in current and isinstance(current["x_axis"], dict):
            current["x_axis"] = AxisConfig(**current["x_axis"])
        if "y_axis" in current and isinstance(current["y_axis"], dict):
            current["y_axis"] = AxisConfig(**current["y_axis"])

        # Create new instance
        return AnalysisParameters(**current)

    def to_export_dict(self) -> Dict[str, Any]:
        """
        Export parameters as a dictionary for serialization.

        Intended for export purposes (e.g., saving to JSON), not for general dict-like access.

        Returns:
            Dictionary representation of parameters.
        """
        return {
            "range1_start": self.range1_start,
            "range1_end": self.range1_end,
            "use_dual_range": self.use_dual_range,
            "range2_start": self.range2_start,
            "range2_end": self.range2_end,
            "x_axis": asdict(self.x_axis),
            "y_axis": asdict(self.y_axis),
            "channel_config": self.channel_config,
        }

    def describe(self) -> str:
        """
        Generate a human-readable description of the parameters.

        Useful for logging or display purposes.

        Returns:
            String description of parameters.
        """
        desc = [f"Range 1: {self.range1_start:.1f}-{self.range1_end:.1f} ms"]

        if self.use_dual_range:
            desc.append(f"Range 2: {self.range2_start:.1f}-{self.range2_end:.1f} ms")

        desc.extend(
            [
                f"X-Axis: {self.x_axis.measure} {self.x_axis.channel or ''}".strip(),
                f"Y-Axis: {self.y_axis.measure} {self.y_axis.channel or ''}".strip(),
            ]
        )

        if self.x_axis.measure == "Peak" and self.x_axis.peak_type:
            desc.append(f"X Peak Type: {self.x_axis.peak_type}")
        if self.y_axis.measure == "Peak" and self.y_axis.peak_type:
            desc.append(f"Y Peak Type: {self.y_axis.peak_type}")

        return " | ".join(desc)
