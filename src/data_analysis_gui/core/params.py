"""
PatchBatch Electrophysiology Data Analysis Tool

Container for analysis parameters passed from ControlPanel through the analysis pipeline.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AxisConfig:
    """Configuration for a plot axis (X or Y)."""

    measure: str  # "Time", "Average", "Peak", or "Conductance"
    channel: Optional[str]  # "Voltage" or "Current" (None for Time/Conductance)
    peak_type: Optional[str] = "Absolute"  # "Absolute", "Positive", "Negative", "Peak-Peak"

    def __post_init__(self):
        logger.debug(
            f"Created AxisConfig: measure={self.measure}, "
            f"channel={self.channel}, peak_type={self.peak_type}"
        )


@dataclass(frozen=True)
class ConductanceConfig:
    """Configuration for conductance calculation: G = I / (V - Vrev)."""
    
    i_measure: str  # "Average" or "Peak"
    v_measure: str  # "Average" or "Peak"
    vrev: float  # Reversal potential in mV
    units: str  # "nS", "μS", or "pS"
    tolerance: float = 0.1  # mV
    
    def __post_init__(self):
        logger.debug(
            f"Created ConductanceConfig: I={self.i_measure}, V={self.v_measure}, "
            f"Vrev={self.vrev}mV, units={self.units}, tolerance={self.tolerance}mV"
        )
        
        valid_measures = ["Average", "Peak"]
        if self.i_measure not in valid_measures:
            raise ValueError(f"Invalid i_measure: {self.i_measure}. Must be {valid_measures}")
        if self.v_measure not in valid_measures:
            raise ValueError(f"Invalid v_measure: {self.v_measure}. Must be {valid_measures}")
        
        valid_units = ["pS", "nS", "μS", "mS", "S"]
        if self.units not in valid_units:
            raise ValueError(f"Invalid units: {self.units}. Must be {valid_units}")
        
        if self.tolerance <= 0:
            raise ValueError(f"Tolerance must be positive, got {self.tolerance}")


@dataclass(frozen=True)
class AnalysisParameters:
    """
    Immutable parameters for analysis operations.
    
    Validates range bounds and dual-range consistency at construction.
    """

    # Range configuration
    range1_start: float
    range1_end: float
    use_dual_range: bool
    range2_start: Optional[float]
    range2_end: Optional[float]

    # Axis configurations
    x_axis: AxisConfig
    y_axis: AxisConfig

    # Channel mapping
    channel_config: Dict[str, Any] = field(default_factory=dict)
    
    # Conductance configuration (required when y_axis.measure == "Conductance")
    conductance_config: Optional[ConductanceConfig] = None

    def __post_init__(self):
        logger.debug(
            f"Validating AnalysisParameters: "
            f"R1=[{self.range1_start}, {self.range1_end}], "
            f"dual={self.use_dual_range}, "
            f"R2=[{self.range2_start}, {self.range2_end}]"
        )
        
        if self.range1_end <= self.range1_start:
            error_msg = f"Range 1 end ({self.range1_end}) must be after start ({self.range1_start})"
            logger.error(f"Validation failed: {error_msg}")
            raise ValueError(error_msg)

        if self.use_dual_range:
            if self.range2_start is None or self.range2_end is None:
                error_msg = "Dual range enabled but range 2 values not provided"
                logger.error(f"Validation failed: {error_msg}")
                raise ValueError(error_msg)
            if self.range2_end <= self.range2_start:
                error_msg = f"Range 2 end ({self.range2_end}) must be after start ({self.range2_start})"
                logger.error(f"Validation failed: {error_msg}")
                raise ValueError(error_msg)
        
        if self.y_axis.measure == "Conductance" and self.conductance_config is None:
            error_msg = "Y-axis is Conductance but conductance_config not provided"
            logger.error(f"Validation failed: {error_msg}")
            raise ValueError(error_msg)
        
        logger.debug("AnalysisParameters validation passed")

    def with_updates(self, **kwargs) -> "AnalysisParameters":
        """Create a new instance with updated values (since this class is frozen)."""
        logger.debug(f"Creating updated parameters with changes: {list(kwargs.keys())}")
        
        current = asdict(self)
        current.update(kwargs)

        # Reconstruct nested dataclasses from dicts
        if "x_axis" in current and isinstance(current["x_axis"], dict):
            current["x_axis"] = AxisConfig(**current["x_axis"])
        if "y_axis" in current and isinstance(current["y_axis"], dict):
            current["y_axis"] = AxisConfig(**current["y_axis"])
        if "conductance_config" in current and current["conductance_config"] is not None:
            if isinstance(current["conductance_config"], dict):
                current["conductance_config"] = ConductanceConfig(**current["conductance_config"])

        new_params = AnalysisParameters(**current)
        logger.debug(f"Created updated parameters: {new_params.describe()}")
        
        return new_params

    def to_export_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (e.g., JSON export)."""
        logger.debug("Exporting parameters to dictionary")
        
        export_dict = {
            "range1_start": self.range1_start,
            "range1_end": self.range1_end,
            "use_dual_range": self.use_dual_range,
            "range2_start": self.range2_start,
            "range2_end": self.range2_end,
            "x_axis": asdict(self.x_axis),
            "y_axis": asdict(self.y_axis),
            "channel_config": self.channel_config,
            "conductance_config": asdict(self.conductance_config) if self.conductance_config else None,
        }
        
        logger.debug(f"Exported {len(export_dict)} parameter fields")
        return export_dict

    def describe(self) -> str:
        """Generate human-readable description for logging/display."""
        desc = [f"Range 1: {self.range1_start:.1f}-{self.range1_end:.1f} ms"]

        if self.use_dual_range:
            desc.append(f"Range 2: {self.range2_start:.1f}-{self.range2_end:.1f} ms")

        desc.extend([
            f"X-Axis: {self.x_axis.measure} {self.x_axis.channel or ''}".strip(),
            f"Y-Axis: {self.y_axis.measure} {self.y_axis.channel or ''}".strip(),
        ])

        if self.x_axis.measure == "Peak" and self.x_axis.peak_type:
            desc.append(f"X Peak Type: {self.x_axis.peak_type}")
        if self.y_axis.measure == "Peak" and self.y_axis.peak_type:
            desc.append(f"Y Peak Type: {self.y_axis.peak_type}")
        
        if self.conductance_config:
            desc.append(
                f"Conductance: I={self.conductance_config.i_measure}, "
                f"V={self.conductance_config.v_measure}, "
                f"Vrev={self.conductance_config.vrev}mV, "
                f"units={self.conductance_config.units}"
            )

        description = " | ".join(desc)
        logger.debug(f"Generated parameter description: {description}")
        
        return description