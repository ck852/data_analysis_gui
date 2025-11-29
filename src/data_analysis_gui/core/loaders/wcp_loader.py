"""
PatchBatch WCP File Loader

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

This module provides functionality to load .wcp (WinWCP) files into the
standardized ElectrophysiologyDataset format used throughout the application.

Parsing logic based on analysis of WCPFileUnit.pas from WinWCP V5.7.9 source code
(https://github.com/johndempster/WinWCPXE/blob/master/WCPFIleUnit.pas)
"""

import struct
import logging
from pathlib import Path
from typing import Optional, Any, Union, Dict, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)

from data_analysis_gui.core.dataset import ElectrophysiologyDataset


def _detect_channel_configuration_wcp(channels: List[Any]) -> Dict[str, Any]:
    """
    Identify which channel is voltage and which is current based on units.
    
    Raises ValueError if the file doesn't have exactly one of each - there's no
    unambiguous way to map channels in that case.
    """
    voltage_channels = []
    current_channels = []
    
    for i, ch in enumerate(channels):
        units_lower = ch.units.lower()
        
        if 'mv' in units_lower or units_lower == 'v':
            voltage_channels.append({
                'index': i,
                'name': ch.name,
                'units': ch.units
            })
        elif any(u in units_lower for u in ['pa', 'na', 'µa', 'ua', 'ma', 'a']):
            current_channels.append({
                'index': i,
                'name': ch.name,
                'units': ch.units
            })
    
    if len(voltage_channels) != 1 or len(current_channels) != 1:
        raise ValueError(
            f"Cannot identify channels: found {len(voltage_channels)} voltage, "
            f"{len(current_channels)} current (need exactly 1 of each)"
        )
    
    v_ch = voltage_channels[0]
    i_ch = current_channels[0]
    
    logger.info(
        f"Mapped channels: Ch.{v_ch['index']} ({v_ch['units']}) → voltage, "
        f"Ch.{i_ch['index']} ({i_ch['units']}) → current"
    )
    
    return {
        'voltage_channel': v_ch['index'],
        'current_channel': i_ch['index'],
        'voltage_units': v_ch['units'],
        'current_units': i_ch['units'].replace('uA', 'μA').replace('ua', 'μA')
    }


def load_wcp(
    file_path: Union[str, Path],
    validate_data: bool = True,
) -> "ElectrophysiologyDataset":
    """
    Load a WCP file into the standard dataset format.
    
    The file must have exactly one voltage and one current channel. The loader
    identifies which is which automatically based on units, so channel ordering
    doesn't matter. Also extracts sweep timing and classification metadata (RecType,
    Group Number, Status) used for leak subtraction.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"WCP file not found: {file_path}")

    logger.info(f"Loading WCP file: {file_path.name}")
    
    try:
        with WCPParser(str(file_path)) as wcp:
            # Identify voltage and current channels
            channel_config = _detect_channel_configuration_wcp(wcp.file_header.channels)
            
            # Create dataset
            dataset = ElectrophysiologyDataset()
            
            # Store metadata
            dataset.metadata["format"] = "wcp"
            dataset.metadata["source_file"] = str(file_path)
            dataset.metadata["sampling_rate_hz"] = 1000.0 / wcp.file_header.dt if wcp.file_header.dt > 0 else None
            dataset.metadata["wcp_version"] = wcp.file_header.version
            dataset.metadata["channel_count"] = wcp.file_header.num_channels
            dataset.metadata["sweep_count"] = wcp.file_header.num_records
            dataset.metadata["channel_labels"] = [ch.name for ch in wcp.file_header.channels]
            dataset.metadata["channel_units"] = [ch.units for ch in wcp.file_header.channels]
            dataset.metadata["channel_config"] = channel_config
            dataset.metadata["sweep_times"] = {}
            dataset.metadata["sweep_info"] = {}
            
            # Load sweeps
            logger.debug(f"Loading {wcp.file_header.num_records} sweeps")
            
            for record_num in range(1, wcp.file_header.num_records + 1):
                try:
                    header, data = wcp.read_record(record_num, calibrated=True)
                    time_ms = wcp.get_time_axis() * 1000.0
                    
                    sweep_index = str(record_num)
                    dataset.metadata["sweep_times"][sweep_index] = float(header.time)
                    dataset.metadata["sweep_info"][sweep_index] = {
                        "time": float(header.time),
                        "rec_type": header.rec_type,
                        "group": int(header.number),
                        "status": header.status
                    }
                    
                    if validate_data:
                        if np.any(np.isnan(time_ms)):
                            raise ValueError(f"Sweep {record_num} contains NaN time values")
                        if np.any(np.isnan(data)):
                            logger.warning(f"Sweep {record_num} contains NaN data values")
                        if np.any(np.isinf(data)):
                            logger.warning(f"Sweep {record_num} contains infinite data values")
                    
                    dataset.add_sweep(sweep_index, time_ms, data)
                    
                except Exception as e:
                    logger.error(f"Failed to load sweep {record_num}: {e}")
                    if validate_data:
                        raise
                    else:
                        logger.warning(f"Skipped corrupted sweep {record_num}")
                        continue
            
            if dataset.is_empty():
                raise ValueError("No valid sweeps could be loaded from WCP file")
            
            # Log sweep classification summary
            rec_types = {}
            for sweep_idx, info in dataset.metadata["sweep_info"].items():
                rec_type = info["rec_type"]
                rec_types[rec_type] = rec_types.get(rec_type, 0) + 1
            
            if rec_types:
                logger.info(f"Sweep classification: {rec_types}")
            
            logger.info(f"Successfully loaded {dataset.sweep_count()} sweeps")
            return dataset
            
    except Exception as e:
        logger.error(f"Failed to load WCP file: {e}")
        raise


# =============================================================================
# WCP Parser Classes
# =============================================================================

from dataclasses import dataclass


@dataclass
class WCPChannel:
    """Channel metadata"""
    name: str
    units: str
    calibration_factor: float
    amplifier_gain: float
    adc_zero: int
    adc_zero_at: int
    channel_offset: int


@dataclass
class WCPRecordHeader:
    """Record (sweep) metadata"""
    status: str
    rec_type: str
    number: float
    time: float
    dt: float
    adc_voltage_range: List[float]
    ident: str


@dataclass
class WCPFileHeader:
    """WCP file metadata"""
    version: float
    num_channels: int
    num_samples: int
    num_records: int
    dt: float
    adc_voltage_range: float
    max_adc_value: int
    min_adc_value: int
    num_bytes_in_header: int
    num_analysis_bytes_per_record: int
    num_data_bytes_per_record: int
    num_bytes_per_record: int
    num_zero_avg: int
    channels: List[WCPChannel]


class WCPParser:
    """Internal parser used by load_wcp() to read WCP files."""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.file_header: Optional[WCPFileHeader] = None
        self._file = None
        
    def __enter__(self):
        self._file = open(self.filepath, 'rb')
        self.file_header = self._parse_file_header()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
            
    def _parse_key_value_header(self, header_bytes: bytes) -> Dict[str, str]:
        """Parse text-based key=value header"""
        header_text = header_bytes.decode('ascii', errors='ignore').rstrip('\x00')
        
        params = {}
        for line in header_text.split('\n'):
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                params[key.strip()] = value.strip()
        
        return params
    
    def _get_param_float(self, params: Dict[str, str], key: str, default: float = 0.0) -> float:
        try:
            return float(params.get(key, default))
        except (ValueError, TypeError):
            return default
    
    def _get_param_int(self, params: Dict[str, str], key: str, default: int = 0) -> int:
        try:
            return int(params.get(key, default))
        except (ValueError, TypeError):
            return default
    
    def _parse_file_header(self) -> WCPFileHeader:
        self._file.seek(0)
        initial_header = self._file.read(1024)
        params = self._parse_key_value_header(initial_header)
        
        num_bytes_in_header = self._get_param_int(params, 'NBH', 1024)
        
        if num_bytes_in_header > 1024:
            self._file.seek(0)
            initial_header = self._file.read(num_bytes_in_header)
            params = self._parse_key_value_header(initial_header)
        
        version = self._get_param_float(params, 'VER', 9.0)
        num_channels = self._get_param_int(params, 'NC', 1)
        max_adc_value = self._get_param_int(params, 'ADCMAX', 2047)
        min_adc_value = -max_adc_value - 1
        
        nba_sectors = self._get_param_int(params, 'NBA', 2)
        num_analysis_bytes_per_record = nba_sectors * 512
        
        nbd_sectors = self._get_param_int(params, 'NBD', 0)
        num_data_bytes_per_record = nbd_sectors * 512
        
        num_bytes_per_record = num_analysis_bytes_per_record + num_data_bytes_per_record
        num_samples = num_data_bytes_per_record // (2 * num_channels)
        
        num_records = self._get_param_int(params, 'NR', 0)
        dt = self._get_param_float(params, 'DT', 0.001)
        adc_voltage_range = self._get_param_float(params, 'AD', 5.0)
        num_zero_avg = max(self._get_param_int(params, 'NZ', 20), 1)
        
        channels = []
        for ch in range(num_channels):
            name = params.get(f'YN{ch}', f'Ch.{ch}')
            units = params.get(f'YU{ch}', 'mV')
            calibration_factor = self._get_param_float(params, f'YG{ch}', 0.001)
            amplifier_gain = 1.0
            adc_zero = self._get_param_int(params, f'YZ{ch}', 0)
            adc_zero_at = self._get_param_int(params, f'YR{ch}', -1)
            channel_offset = self._get_param_int(params, f'YO{ch}', ch)
            
            channels.append(WCPChannel(
                name=name,
                units=units,
                calibration_factor=calibration_factor,
                amplifier_gain=amplifier_gain,
                adc_zero=adc_zero,
                adc_zero_at=adc_zero_at,
                channel_offset=channel_offset
            ))
        
        return WCPFileHeader(
            version=version,
            num_channels=num_channels,
            num_samples=num_samples,
            num_records=num_records,
            dt=dt,
            adc_voltage_range=adc_voltage_range,
            max_adc_value=max_adc_value,
            min_adc_value=min_adc_value,
            num_bytes_in_header=num_bytes_in_header,
            num_analysis_bytes_per_record=num_analysis_bytes_per_record,
            num_data_bytes_per_record=num_data_bytes_per_record,
            num_bytes_per_record=num_bytes_per_record,
            num_zero_avg=num_zero_avg,
            channels=channels
        )
    
    def _parse_record_header(self, record_num: int) -> WCPRecordHeader:
        fh = self.file_header
        
        record_offset = fh.num_bytes_in_header + (record_num - 1) * fh.num_bytes_per_record
        self._file.seek(record_offset)
        
        status = self._file.read(8).decode('ascii', errors='ignore').strip('\x00').strip()
        rec_type = self._file.read(4).decode('ascii', errors='ignore').strip('\x00').strip()
        number = struct.unpack('<f', self._file.read(4))[0]
        time = struct.unpack('<f', self._file.read(4))[0]
        dt = struct.unpack('<f', self._file.read(4))[0]
        
        adc_voltage_range = []
        for _ in range(fh.num_channels):
            voltage_range = struct.unpack('<f', self._file.read(4))[0]
            adc_voltage_range.append(voltage_range)
        
        ident = self._file.read(16).decode('ascii', errors='ignore').strip('\x00').strip()
        
        return WCPRecordHeader(
            status=status,
            rec_type=rec_type,
            number=number,
            time=time,
            dt=dt,
            adc_voltage_range=adc_voltage_range,
            ident=ident
        )
    
    def read_record(self, record_num: int, calibrated: bool = True) -> Tuple[WCPRecordHeader, np.ndarray]:
        """
        Read a single sweep with header metadata.
        
        Returns calibrated physical units by default. Set calibrated=False for raw ADC values.
        """
        if not (1 <= record_num <= self.file_header.num_records):
            raise ValueError(f"Record number must be between 1 and {self.file_header.num_records}")
        
        fh = self.file_header
        header = self._parse_record_header(record_num)
        
        # Read raw data
        data_offset = (fh.num_bytes_in_header + 
                    (record_num - 1) * fh.num_bytes_per_record + 
                    fh.num_analysis_bytes_per_record)
        self._file.seek(data_offset)
        
        num_values = fh.num_samples * fh.num_channels
        raw_data = np.frombuffer(
            self._file.read(num_values * 2),
            dtype=np.int16
        )
        
        data = raw_data.reshape((fh.num_samples, fh.num_channels)).copy()
        
        if calibrated:
            data = data.astype(np.float64)
            
            for ch_idx, channel in enumerate(fh.channels):
                # Dynamic zero calculation if baseline region specified
                if channel.adc_zero_at >= 0:
                    zero_level = self._calculate_dynamic_zero(
                        data[:, ch_idx], 
                        channel.adc_zero_at, 
                        fh.num_zero_avg, 
                        fh.num_samples
                    )
                else:
                    zero_level = channel.adc_zero
                
                adc_scale = (abs(header.adc_voltage_range[ch_idx]) / 
                        (channel.calibration_factor * (fh.max_adc_value + 1)))
                
                data[:, ch_idx] = (data[:, ch_idx] - zero_level) * adc_scale
        
        return header, data
    
    def _calculate_dynamic_zero(
        self, 
        raw_channel_data: np.ndarray, 
        adc_zero_at: int, 
        num_zero_avg: int,
        num_samples: int
    ) -> float:
        """
        Calculate baseline from a specified region in the sweep.
        
        Matches WinWCP's baseline calculation - averages num_zero_avg samples
        starting at adc_zero_at. This handles per-sweep baseline drift.
        """
        i0 = max(0, min(adc_zero_at, num_samples - 1))
        i1 = max(0, min(i0 + num_zero_avg - 1, num_samples - 1))
        return np.mean(raw_channel_data[i0:i1+1])
    
    def get_time_axis(self) -> np.ndarray:
        """Get time axis for a record in seconds"""
        return np.arange(self.file_header.num_samples) * self.file_header.dt