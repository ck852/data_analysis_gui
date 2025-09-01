import numpy as np


def process_sweep_data(time, data, start_ms, end_ms, channel_id):
    """Process sweep data within time range.
    
    Args:
        time: Time array in milliseconds
        data: 2D data array with shape (samples, channels)
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        channel_id: The channel index to extract data from
    
    Returns:
        Filtered data from the specified channel within the time range
    """
    mask = (time >= start_ms) & (time <= end_ms)
    return data[:, channel_id][mask]


def calculate_peak(data, peak_type="Absolute"):
    """Calculate peak value based on type.
    
    Args:
        data: Array of data values
        peak_type: Type of peak - "Absolute", "Positive", "Negative", or "Peak-Peak"
    
    Returns:
        Peak value according to the specified type
    """
    if len(data) == 0:
        return np.nan
    
    if peak_type == "Positive" or peak_type == "Max":
        return np.max(data)
    elif peak_type == "Negative" or peak_type == "Min":
        return np.min(data)
    elif peak_type == "Peak-Peak":
        return np.max(data) - np.min(data)
    else:  # "Absolute" or "Absolute Max"
        return data[np.abs(data).argmax()]


def calculate_average(data):
    """Calculate average of data.
    
    Args:
        data: Array of data values
    
    Returns:
        Mean of the data, or NaN if empty
    """
    return np.mean(data) if len(data) > 0 else np.nan


def apply_analysis_mode(data, mode="Average", peak_type=None):
    """Apply selected analysis mode to data.
    
    Args:
        data: Array of data values
        mode: Analysis mode - "Average" or "Peak"
        peak_type: If mode is "Peak", the type of peak to find 
                  ("Absolute", "Positive", "Negative", "Peak-Peak")
    
    Returns:
        Analyzed value according to the specified mode
    """
    if mode == "Average":
        return calculate_average(data)
    elif mode == "Peak":
        return calculate_peak(data, peak_type or "Absolute")
    else:
        return 0


def calculate_current_density(current, cslow):
    """Calculate current density by dividing current by Cslow.
    
    Args:
        current: Current value in pA
        cslow: Cell capacitance in pF
    
    Returns:
        Current density in pA/pF, or NaN if invalid
    """
    if cslow > 0 and not np.isnan(current):
        return current / cslow
    return np.nan


def calculate_sem(values):
    """Calculate standard error of mean.
    
    Args:
        values: Array of values
    
    Returns:
        Standard error of the mean
    """
    if len(values) > 1:
        return np.std(values, ddof=1) / np.sqrt(len(values))
    return 0


def calculate_average_voltage(voltage_data):
    """Calculate average voltage for range.
    
    Args:
        voltage_data: Array of voltage values
    
    Returns:
        Formatted string representation of average voltage
    """
    if len(voltage_data) > 0:
        mean_v = np.nanmean(voltage_data)
        rounded_v = int(round(mean_v))
        formatted_v = f"+{rounded_v}" if rounded_v >= 0 else str(rounded_v)
        return formatted_v
    return ""


def format_voltage_label(voltage):
    """Format voltage value for display.
    
    Args:
        voltage: Voltage value to format
    
    Returns:
        Formatted string with appropriate sign
    """
    rounded = int(round(voltage))
    return f"+{rounded}" if rounded >= 0 else str(rounded)