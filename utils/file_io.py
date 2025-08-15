import os
import re
import csv
import scipy.io
import pandas as pd
import numpy as np


def load_mat_file(filepath):
    """Load and parse MAT file, return sweeps dictionary.
    Extracted from ModernMatSweepAnalyzer.load_mat_file()"""
    mat_data = scipy.io.loadmat(filepath)
    sweeps = {}
    
    for key in mat_data:
        if key.startswith("T"):
            index = key[1:]
            if f"Y{index}" in mat_data:
                t = mat_data[key].squeeze() * 1000  # Convert to ms
                y = mat_data[f"Y{index}"]
                sweeps[index] = (t, y)
    
    return sweeps


def load_csv_file(filepath):
    """Load CSV file and validate structure.
    Extracted from ConcentrationResponseDialog.process_and_plot_file()"""
    df = pd.read_csv(filepath)
    
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least 2 columns (time and data)")
    
    return df


def export_to_csv(filepath, data, header, format_spec='%.6f'):
    """Export numpy array to CSV with header.
    Generalized from multiple export methods throughout the code"""
    np.savetxt(filepath, data, delimiter=',', fmt=format_spec,
               header=header, comments='')


def get_next_available_filename(path):
    """Find available filename by appending _1, _2, etc.
    From ConcentrationResponseDialog._get_next_available_filename()"""
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
    Extracted from ConcentrationResponseDialog.export_results()"""
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
    From batch_analyze()"""
    filename = os.path.basename(filepath)
    try:
        number_part = filename.split('_')[-1].split('.')[0]
        return int(number_part)
    except (IndexError, ValueError):
        return 0