import numpy as np


def process_sweep_data(time, data, start_ms, end_ms, channel=0):
    """Process sweep data within time range.
    Extracted from process_all_sweeps() logic"""
    mask = (time >= start_ms) & (time <= end_ms)
    return data[:, channel][mask]


def calculate_peak(data, peak_type="Max"):
    """Calculate peak value based on type.
    Extracted from run_analysis()"""
    if len(data) == 0:
        return np.nan
    
    if peak_type == "Max":
        return np.max(data)
    elif peak_type == "Min":
        return np.min(data)
    else:  # Absolute Max
        return data[np.abs(data).argmax()]


def calculate_average(data):
    """Calculate average of data.
    Used throughout the code for averaging"""
    return np.mean(data) if len(data) > 0 else np.nan


def apply_analysis_mode(data, mode="Average", peak_type=None):
    """Apply selected analysis mode to data.
    From ModernMatSweepAnalyzer.apply_analysis_mode()"""
    if mode == "Average":
        return calculate_average(data)
    elif mode == "Peak":
        return calculate_peak(data, peak_type)
    else:
        return 0


def calculate_current_density(current, cslow):
    """Calculate current density by dividing current by Cslow.
    From CurrentDensityIVDialog.update_cd_plot()"""
    if cslow > 0 and not np.isnan(current):
        return current / cslow
    return np.nan


def calculate_sem(values):
    """Calculate standard error of mean.
    From CurrentDensityIVDialog"""
    if len(values) > 1:
        return np.std(values, ddof=1) / np.sqrt(len(values))
    return 0


def calculate_average_voltage(voltage_data):
    """Calculate average voltage for range.
    Used for creating descriptive headers"""
    if len(voltage_data) > 0:
        mean_v = np.nanmean(voltage_data)
        rounded_v = int(round(mean_v))
        formatted_v = f"+{rounded_v}" if rounded_v >= 0 else str(rounded_v)
        return formatted_v
    return ""


def format_voltage_label(voltage):
    """Format voltage value for display.
    Used in multiple places for consistent voltage formatting"""
    rounded = int(round(voltage))
    return f"+{rounded}" if rounded >= 0 else str(rounded)