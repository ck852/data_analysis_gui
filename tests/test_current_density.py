"""
PatchBatch Electrophysiology Data Analysis Tool

Test script for current density analysis workflow with golden file validation.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

This module tests the complete workflow from batch analysis through current density
calculation and export, validating all outputs against golden reference files.
It ensures that intermediate and final results match expected values and formats.
"""

import os
import csv
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple
from dataclasses import replace

import numpy as np
import pytest

from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
from data_analysis_gui.services.batch_processor import BatchProcessor
from data_analysis_gui.services.data_manager import DataManager
from data_analysis_gui.services.current_density_service import CurrentDensityService


# Expected Cslow values for each file
CSLOW_VALUES = {
    "250514_001": 34.4,
    "250514_002": 14.5,
    "250514_003": 20.5,
    "250514_004": 16.3,
    "250514_005": 18.4,
    "250514_006": 17.3,
    "250514_007": 14.4,
    "250514_008": 14.1,
    "250514_009": 18.4,
    "250514_010": 21.0,
    "250514_011": 22.2,
    "250514_012": 23.2,
}


def load_csv_data(filepath: Path) -> Tuple[List[str], np.ndarray]:
    """
    Load CSV headers and data array from a file.

    Args:
        filepath (Path): Path to the CSV file.

    Returns:
        Tuple[List[str], np.ndarray]: A tuple containing the list of headers and the data array.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    with open(filepath, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = []
        for row in reader:
            data.append([float(val) if val and val != "nan" else np.nan for val in row])

    return headers, np.array(data)


def compare_csv_files(
    generated: Path, golden: Path, rtol: float = 1e-5, atol: float = 1e-6
) -> None:
    """
    Compare two CSV files for header and data consistency, with detailed error reporting.

    Args:
        generated (Path): Path to the generated CSV file.
        golden (Path): Path to the golden reference CSV file.
        rtol (float, optional): Relative tolerance for numerical comparison. Defaults to 1e-5.
        atol (float, optional): Absolute tolerance for numerical comparison. Defaults to 1e-6.

    Raises:
        AssertionError: If headers, shapes, NaN positions, or numerical values do not match within tolerances.
    """
    # Load both files
    gen_headers, gen_data = load_csv_data(generated)
    gold_headers, gold_data = load_csv_data(golden)

    # FIX 1: Exact header comparison for individual files
    try:
        assert (
            gen_headers == gold_headers
        ), f"Headers mismatch:\nGenerated: {gen_headers}\nGolden: {gold_headers}"
    except AssertionError as e:
        # FIX 6: Better error messages
        raise AssertionError(
            f"Header validation failed for {generated.name}\n"
            f"Generated file: {generated}\n"
            f"Golden file: {golden}\n"
            f"{str(e)}"
        )

    # Compare data shape
    try:
        assert (
            gen_data.shape == gold_data.shape
        ), f"Data shape mismatch:\nGenerated: {gen_data.shape}\nGolden: {gold_data.shape}"
    except AssertionError as e:
        # FIX 6: Better error messages
        raise AssertionError(
            f"Shape validation failed for {generated.name}\n"
            f"Generated file: {generated}\n"
            f"Golden file: {golden}\n"
            f"{str(e)}"
        )

    # Compare data values
    if gen_data.size > 0:
        # Check for NaN positions matching
        gen_nan_mask = np.isnan(gen_data)
        gold_nan_mask = np.isnan(gold_data)

        # NaNs are a red flag in this test
        if np.any(gen_nan_mask):
            nan_count = np.sum(gen_nan_mask)
            nan_positions = np.where(gen_nan_mask)
            raise AssertionError(
                f"WARNING: Found {nan_count} NaN values in generated file {generated.name}\n"
                f"NaN positions (row, col): {list(zip(*nan_positions))[:5]}..."  # Show first 5
            )

        try:
            assert np.array_equal(
                gen_nan_mask, gold_nan_mask
            ), f"NaN positions don't match in {generated.name}"
        except AssertionError as e:
            # FIX 6: Better error messages
            raise AssertionError(
                f"NaN position validation failed for {generated.name}\n"
                f"Generated file: {generated}\n"
                f"Golden file: {golden}\n"
                f"{str(e)}"
            )

        # Compare non-NaN values
        if not np.all(gen_nan_mask):
            valid_mask = ~gen_nan_mask
            try:
                np.testing.assert_allclose(
                    gen_data[valid_mask],
                    gold_data[valid_mask],
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"Data mismatch in {generated.name}",
                )
            except AssertionError as e:
                # FIX 6: Provide more detailed error info
                diff = np.abs(gen_data[valid_mask] - gold_data[valid_mask])
                max_diff_idx = np.argmax(diff)
                max_diff = diff[max_diff_idx]
                gen_val = gen_data[valid_mask][max_diff_idx]
                gold_val = gold_data[valid_mask][max_diff_idx]
                raise AssertionError(
                    f"Numerical validation failed for {generated.name}\n"
                    f"Generated file: {generated}\n"
                    f"Golden file: {golden}\n"
                    f"Max difference: {max_diff:.6e}\n"
                    f"Generated value: {gen_val:.6f}\n"
                    f"Golden value: {gold_val:.6f}\n"
                    f"Tolerance: rtol={rtol}, atol={atol}\n"
                    f"{str(e)}"
                )


def compare_summary_csv(generated: Path, golden: Path) -> None:
    """
    Compare summary CSV files for structure and data accuracy, using appropriate tolerances.

    Summary files have the format:
        Voltage (mV) | File1 (Cslow1 pF) | File2 (Cslow2 pF) | ...

    Args:
        generated (Path): Path to the generated summary CSV.
        golden (Path): Path to the golden reference summary CSV.

    Raises:
        AssertionError: If header structure, shapes, NaN positions, or numerical values do not match within tolerances.
    """
    gen_headers, gen_data = load_csv_data(generated)
    gold_headers, gold_data = load_csv_data(golden)

    # For summary, headers might have formatting differences due to Cslow values
    # So check structure rather than exact match
    try:
        assert len(gen_headers) == len(
            gold_headers
        ), f"Header count mismatch: {len(gen_headers)} vs {len(gold_headers)}"
    except AssertionError as e:
        # FIX 6: Better error messages
        raise AssertionError(
            f"Summary header validation failed\n"
            f"Generated file: {generated}\n"
            f"Golden file: {golden}\n"
            f"{str(e)}"
        )

    # First header should be voltage
    assert (
        "Voltage" in gen_headers[0]
    ), f"First column should be Voltage, got: {gen_headers[0]}"

    # Compare data
    try:
        assert (
            gen_data.shape == gold_data.shape
        ), f"Data shape mismatch:\nGenerated: {gen_data.shape}\nGolden: {gold_data.shape}"
    except AssertionError as e:
        # FIX 6: Better error messages
        raise AssertionError(
            f"Summary shape validation failed\n"
            f"Generated file: {generated}\n"
            f"Golden file: {golden}\n"
            f"{str(e)}"
        )

    if gen_data.size > 0:
        # Check for NaNs - they're a red flag
        gen_nan_mask = np.isnan(gen_data)
        if np.any(gen_nan_mask):
            nan_count = np.sum(gen_nan_mask)
            raise AssertionError(
                f"WARNING: Found {nan_count} NaN values in summary file {generated.name}\n"
                f"This indicates missing data or calculation errors"
            )

        # Voltage column (column 0) - use tight tolerance
        try:
            np.testing.assert_allclose(
                gen_data[:, 0],
                gold_data[:, 0],
                rtol=1e-4,
                atol=0.1,  # 0.1 mV tolerance for voltages
                err_msg="Voltage column mismatch",
            )
        except AssertionError as e:
            # FIX 6: Better error messages
            raise AssertionError(
                f"Summary voltage column validation failed\n"
                f"Generated file: {generated}\n"
                f"Golden file: {golden}\n"
                f"{str(e)}"
            )

        # Current density columns - use appropriate tolerance
        for col_idx in range(1, gen_data.shape[1]):
            col_gen = gen_data[:, col_idx]
            col_gold = gold_data[:, col_idx]

            # Handle NaN values
            gen_nan_mask = np.isnan(col_gen)
            gold_nan_mask = np.isnan(col_gold)

            try:
                assert np.array_equal(
                    gen_nan_mask, gold_nan_mask
                ), f"NaN positions don't match in column {col_idx}"
            except AssertionError as e:
                # FIX 6: Better error messages
                raise AssertionError(
                    f"Summary column {col_idx} NaN validation failed\n"
                    f"Column header: {gen_headers[col_idx]}\n"
                    f"Generated file: {generated}\n"
                    f"Golden file: {golden}\n"
                    f"{str(e)}"
                )

            # Compare non-NaN values
            valid_mask = ~gen_nan_mask
            if np.any(valid_mask):
                try:
                    np.testing.assert_allclose(
                        col_gen[valid_mask],
                        col_gold[valid_mask],
                        rtol=1e-4,
                        atol=1e-3,  # 0.001 pA/pF tolerance
                        err_msg=f"Current density mismatch in column {col_idx} ({gen_headers[col_idx]})",
                    )
                except AssertionError as e:
                    # FIX 6: Better error messages
                    diff = np.abs(col_gen[valid_mask] - col_gold[valid_mask])
                    max_diff = np.max(diff)
                    raise AssertionError(
                        f"Summary column {col_idx} validation failed\n"
                        f"Column header: {gen_headers[col_idx]}\n"
                        f"Max difference: {max_diff:.6e} pA/pF\n"
                        f"Generated file: {generated}\n"
                        f"Golden file: {golden}\n"
                        f"{str(e)}"
                    )


class CurrentDensityTestBase:
    """
    Base class for current density workflow tests.

    Subclasses should specify the file type and extension to test.
    Provides fixtures and utility methods for running the workflow and validating results.
    """

    # Subclasses should define these
    FILE_TYPE = None  # 'abf' or 'mat'
    FILE_EXTENSION = None  # '*.abf' or '*.mat'

    @property
    def sample_data_dir(self) -> Path:
        """
        Get the sample data directory for the specified file type.

        Returns:
            Path: Path to the sample data directory.
        """
        return Path(f"tests/fixtures/sample_data/IV+CD/{self.FILE_TYPE}")

    @property
    def golden_data_dir(self) -> Path:
        """
        Get the golden data directory for the specified file type.

        Returns:
            Path: Path to the golden data directory.
        """
        return Path(f"tests/fixtures/golden_data/golden_CD/{self.FILE_TYPE}")

    @pytest.fixture
    def analysis_params(self):
        """
        Create analysis parameters matching the GUI state.

        Returns:
            AnalysisParameters: Configured analysis parameters for the test.
        """
        return AnalysisParameters(
            range1_start=150.1,
            range1_end=649.2,
            use_dual_range=False,
            range2_start=None,
            range2_end=None,
            x_axis=AxisConfig(measure="Average", channel="Voltage"),
            y_axis=AxisConfig(measure="Average", channel="Current"),
            channel_config={"voltage": 0, "current": 1, "current_units": "pA"},
        )

    @pytest.fixture
    def temp_output_dir(self):
        """
        Create a temporary directory for test outputs.

        Yields:
            str: Path to the temporary output directory.

        Cleans up the directory after the test completes.
        """
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir)

    def get_test_files(self) -> List[str]:
        """
        Retrieve all test files from the sample data directory.

        Returns:
            List[str]: Sorted list of test file paths.

        Skips the test if the sample data directory or files are missing.
        """
        if not self.sample_data_dir.exists():
            pytest.skip(f"Sample data directory not found: {self.sample_data_dir}")

        test_files = list(self.sample_data_dir.glob(self.FILE_EXTENSION))
        if not test_files:
            pytest.skip(
                f"No {self.FILE_TYPE.upper()} files found in {self.sample_data_dir}"
            )

        return [str(f) for f in sorted(test_files)]

    def test_current_density_workflow(self, analysis_params, temp_output_dir):
        """
        Test the complete current density analysis workflow, including golden file validation.

        Steps:
            1. Initialize services as the GUI does.
            2. Perform batch analysis.
            3. Validate intermediate results (pre- and post-current density calculation).
            4. Apply current density calculations.
            5. Export individual current density CSVs.
            6. Generate and export summary CSV.
            7. Validate all outputs against golden reference files.

        Args:
            analysis_params (AnalysisParameters): Analysis parameters for the test.
            temp_output_dir (str): Temporary directory for output files.

        Raises:
            AssertionError: If any validation step fails.
        """

        # ==================================================================
        # PHASE 1: Initialize Services (exactly as GUI does)
        # ==================================================================
        batch_processor = BatchProcessor()
        data_manager = DataManager()
        cd_service = CurrentDensityService()

        # ==================================================================
        # PHASE 2: Batch Analysis (following GUI workflow)
        # ==================================================================
        test_files = self.get_test_files()
        assert (
            len(test_files) == 12
        ), f"Expected 12 {self.FILE_TYPE.upper()} files, found {len(test_files)}"

        print(f"\n{'='*60}")
        print(f"Testing {self.FILE_TYPE.upper()} Current Density Workflow")
        print(f"{'='*60}")
        print(f"Processing {len(test_files)} files...")

        # BatchProcessor.process_files() - exactly as GUI does
        batch_result = batch_processor.process_files(
            file_paths=test_files, params=analysis_params
        )

        assert (
            len(batch_result.successful_results) == 12
        ), f"Expected 12 successful results, got {len(batch_result.successful_results)}"
        assert (
            len(batch_result.failed_results) == 0
        ), f"Unexpected failures: {[r.file_path for r in batch_result.failed_results]}"

        # FIX 4: Intermediate validation - verify units are still pA after batch analysis
        print("Validating intermediate state (pre-CD)...")
        for result in batch_result.successful_results:
            if result.export_table and "headers" in result.export_table:
                headers_str = str(result.export_table["headers"])
                # Should have pA but not pF in headers before CD calculation
                assert (
                    "(pA)" in headers_str or "Current" in headers_str
                ), f"Expected current units in headers for {result.base_name}, got: {headers_str}"
                assert (
                    "(pA/pF)" not in headers_str
                ), f"Found pA/pF in headers before CD calculation for {result.base_name}: {headers_str}"

        # ==================================================================
        # PHASE 3: Current Density Calculation (using service method)
        # ==================================================================
        print("Applying current density calculations...")

        # Create copies for CD calculation (as GUI does)
        original_batch_result = batch_result
        active_batch_result = replace(
            batch_result,
            successful_results=list(batch_result.successful_results),  # Create new list
        )

        # Apply CD calculations using the service (matching GUI's _apply_initial_current_density)
        for i, result in enumerate(active_batch_result.successful_results):
            file_name = result.base_name
            cslow = CSLOW_VALUES.get(file_name)

            assert (
                cslow is not None and cslow > 0
            ), f"Invalid Cslow value for {file_name}"

            # Use the service method - NOT manual calculation
            updated_result = cd_service.recalculate_cd_for_file(
                file_name, cslow, active_batch_result, original_batch_result
            )

            # FIX 2: Verify export table header changed to pA/pF
            if updated_result.export_table and "headers" in updated_result.export_table:
                headers_str = str(updated_result.export_table["headers"])
                assert (
                    "(pA/pF)" in headers_str
                ), f"Expected (pA/pF) in headers after CD calculation for {file_name}, got: {headers_str}"
                # Should no longer have plain (pA) - it should be (pA/pF)
                # Note: Voltage should still be (mV) not affected
                for header in updated_result.export_table["headers"]:
                    if (
                        "Current" in header
                        or "Average" in header
                        and "Voltage" not in header
                    ):
                        assert (
                            "(pA/pF)" in header or "(mV)" in header
                        ), f"Invalid unit in header after CD: {header}"

            # Update the result in place
            active_batch_result.successful_results[i] = updated_result

        # FIX 4: Validate all results have been converted to current density
        print("Validating intermediate state (post-CD)...")
        for result in active_batch_result.successful_results:
            if result.export_table and "headers" in result.export_table:
                headers_str = str(result.export_table["headers"])
                assert (
                    "(pA/pF)" in headers_str
                ), f"Expected pA/pF units after CD for {result.base_name}, got: {headers_str}"

        # ==================================================================
        # PHASE 4: Export Individual CSVs (with _CD suffix)
        # ==================================================================
        print("Exporting individual current density CSVs...")

        # Add "_CD" suffix as GUI does
        cd_results = []
        for result in active_batch_result.successful_results:
            cd_result = replace(result, base_name=f"{result.base_name}_CD")
            cd_results.append(cd_result)

        cd_batch_result = replace(
            active_batch_result,
            successful_results=cd_results,
            selected_files={r.base_name for r in cd_results},
        )

        # Export using BatchProcessor
        cd_output_dir = os.path.join(temp_output_dir, "current_density")
        os.makedirs(cd_output_dir, exist_ok=True)

        export_result = batch_processor.export_results(cd_batch_result, cd_output_dir)
        assert (
            export_result.success_count == 12
        ), f"Expected 12 successful exports, got {export_result.success_count}"

        # ==================================================================
        # PHASE 5: Generate and Export Summary
        # ==================================================================
        print("Generating current density summary...")

        # Prepare data structure as GUI does
        voltage_data = {}
        file_mapping = {}
        sorted_results = sorted(
            active_batch_result.successful_results,
            key=lambda r: int(r.base_name.split("_")[-1]),
        )

        for idx, result in enumerate(sorted_results):
            recording_id = f"Recording {idx + 1}"
            file_mapping[recording_id] = result.base_name

            for i, voltage in enumerate(result.x_data):
                voltage_rounded = round(float(voltage), 1)
                if voltage_rounded not in voltage_data:
                    voltage_data[voltage_rounded] = [np.nan] * len(sorted_results)
                if i < len(result.y_data):
                    voltage_data[voltage_rounded][idx] = result.y_data[i]

        # Use service to prepare summary
        selected_files = {r.base_name for r in sorted_results}
        summary_data = cd_service.prepare_summary_export(
            voltage_data, file_mapping, CSLOW_VALUES, selected_files, "pA/pF"
        )

        summary_path = os.path.join(temp_output_dir, "Current_Density_Summary.csv")
        summary_result = data_manager.export_to_csv(summary_data, summary_path)
        assert (
            summary_result.success
        ), f"Summary export failed: {summary_result.error_message}"

        # ==================================================================
        # PHASE 6: Validate Against Golden Files (CRITICAL)
        # ==================================================================
        print("\nValidating against golden reference files...")

        # Validate individual CSV files
        for file_name in CSLOW_VALUES.keys():
            generated_csv = Path(cd_output_dir) / f"{file_name}_CD.csv"
            golden_csv = self.golden_data_dir / f"{file_name}_CD.csv"

            print(f"  Comparing {file_name}_CD.csv...", end=" ")

            # FIX 6: Better error handling with context
            try:
                compare_csv_files(
                    generated_csv,
                    golden_csv,
                    rtol=1e-4,  # 0.01% relative tolerance
                    atol=1e-3,  # 0.001 absolute tolerance for small values
                )
                print("✔")
            except AssertionError as e:
                print("✗")
                raise AssertionError(
                    f"\nValidation failed for individual file: {file_name}_CD.csv\n"
                    f"Cslow value used: {CSLOW_VALUES[file_name]} pF\n"
                    f"{str(e)}"
                )

        # Validate summary CSV
        print("  Comparing Current_Density_Summary.csv...", end=" ")
        golden_summary = self.golden_data_dir / "Current_Density_Summary.csv"

        # FIX 6: Better error handling for summary
        try:
            compare_summary_csv(Path(summary_path), golden_summary)
            print("✔")
        except AssertionError as e:
            print("✗")
            raise AssertionError(
                f"\nValidation failed for summary file\n"
                f"Number of files: {len(CSLOW_VALUES)}\n"
                f"{str(e)}"
            )

        print(f"\n{'='*60}")
        print(f"✔ All {self.FILE_TYPE.upper()} current density tests passed!")
        print(f"{'='*60}\n")


class TestCurrentDensityABF(CurrentDensityTestBase):
    """
    Test current density workflow using ABF files.

    Inherits from CurrentDensityTestBase and sets file type and extension for ABF.
    """

    FILE_TYPE = "abf"
    FILE_EXTENSION = "*.abf"


class TestBatchIVAnalysisWCP(CurrentDensityTestBase):
    FILE_TYPE = "wcp" 
    FILE_EXTENSION = "*.wcp"
    # Uses auto-detected channel config per file


if __name__ == "__main__":
    """
    Run the test directly if this script is executed as the main module.
    """
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
