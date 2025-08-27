import os
import re
import csv
import pandas as pd
import numpy as np

# Import the new dataset loader
from data_analysis_gui.core.dataset import DatasetLoader


def load_mat_file(filepath, channel_config=None):
    """Load and parse MAT file using the Dataset API.
    
    Args:
        filepath: Path to the MAT file
        channel_config: Optional ChannelConfiguration instance for channel mapping
    
    Returns:
        Dictionary of sweeps in legacy format {index: (time_ms, data_matrix)}
    
    Raises:
        IOError: If file cannot be loaded
        ValueError: If file structure is invalid
    """
    from data_analysis_gui.core.dataset import DatasetLoader
    
    # Load file using the Dataset API
    dataset = DatasetLoader.load(filepath, channel_config)
    
    # Convert to legacy format for compatibility with existing code
    sweeps = {}
    for sweep_idx in dataset.sweeps():
        time_ms, data_matrix = dataset.get_sweep(sweep_idx)
        sweeps[sweep_idx] = (time_ms, data_matrix)
    
    return sweeps


def load_csv_file(filepath):
    """Load CSV file and validate structure.
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        pandas DataFrame containing the CSV data
    
    Raises:
        ValueError: If CSV doesn't have at least 2 columns
    """
    df = pd.read_csv(filepath)
    
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least 2 columns (time and data)")
    
    return df


def export_to_csv(filepath, data, header, format_spec='%.6f'):
    """Export numpy array to CSV with header.
    
    Args:
        filepath: Output file path
        data: Numpy array to export
        header: Header string for the CSV
        format_spec: Format specification for numbers (default: '%.6f')
    """
    np.savetxt(filepath, data, delimiter=',', fmt=format_spec,
               header=header, comments='')


def get_next_available_filename(path):
    """Find available filename by appending _1, _2, etc.
    
    Args:
        path: Initial file path to check
    
    Returns:
        Available file path (original or with suffix)
    """
    if not os.path.exists(path):
        return path
    
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        new_path = f"{base}_{i}{ext}"
        if not os.path.exists(new_path):
            return new_path
        i += 1


def sanitize_filename(name):
    """Sanitize string for use as filename.
    
    Args:
        name: String to sanitize
    
    Returns:
        Safe filename string with problematic characters replaced
    """
    def replacer(match):
        content = match.group(1)
        if '+' in content or '-' in content:
            return '_' + content
        return ''
    
    name_after_parens = re.sub(r'\s*\((.*?)\)', replacer, name).strip()
    safe_name = re.sub(r'[^\w+-]', '_', name_after_parens).replace('__', '_')
    return safe_name


def extract_file_number(filepath):
    """Extract number from filename for sorting.
    
    Args:
        filepath: File path to extract number from
    
    Returns:
        Integer extracted from filename, or 0 if not found
    """
    filename = os.path.basename(filepath)
    try:
        number_part = filename.split('_')[-1].split('.')[0]
        return int(number_part)
    except (IndexError, ValueError):
        return 0


def extract_channel_data(data, channel_id):
    """Extract data for a specific channel from multi-channel data.
    
    Args:
        data: Multi-dimensional data array where last dimension is channels
        channel_id: The channel index to extract
    
    Returns:
        Data for the specified channel
    
    Raises:
        IndexError: If channel_id is out of bounds
    """
    if data.ndim == 1:
        # Single channel data
        if channel_id != 0:
            raise IndexError(f"Channel {channel_id} not available in single-channel data")
        return data
    elif data.ndim == 2:
        # Multi-channel data (samples x channels)
        if channel_id >= data.shape[1]:
            raise IndexError(f"Channel {channel_id} out of range. Data has {data.shape[1]} channels")
        return data[:, channel_id]
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")


def get_channel_count(data):
    """Get the number of channels in the data.
    
    Args:
        data: Data array
    
    Returns:
        Number of channels (1 for 1D array, shape[1] for 2D array)
    """
    if data.ndim == 1:
        return 1
    elif data.ndim == 2:
        return data.shape[1]
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")


def validate_channel_id(data, channel_id):
    """Validate that a channel ID is valid for the given data.
    
    Args:
        data: Data array to check against
        channel_id: Channel ID to validate
    
    Returns:
        True if channel_id is valid
    
    Raises:
        ValueError: If channel_id is invalid for the data
    """
    num_channels = get_channel_count(data)
    
    if channel_id < 0:
        raise ValueError(f"Channel ID must be non-negative, got {channel_id}")
    
    if channel_id >= num_channels:
        raise ValueError(f"Channel {channel_id} out of range. Data has {num_channels} channel(s)")
    
    return True