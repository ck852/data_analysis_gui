"""
PatchBatch Electrophysiology Data Analysis Tool

Test script for batch IV analysis workflow with golden file validation.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

"""
This test suite validates the complete batch IV analysis workflow, including:
- Batch analysis of electrophysiology files
- IV summary export
- Individual CSV export
- Validation of outputs against golden reference files

The workflow mirrors the GUI process and ensures correctness of all exported data.
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
from data_analysis_gui.core.iv_analysis import IVAnalysisService, IVSummaryExporter


def load_csv_data(filepath: Path) -> Tuple[List[str], np.ndarray]:
    """
    Load CSV headers and data array from a file.

    Args:
        filepath (Path): Path to the CSV file.

    Returns:
        Tuple[List[str], np.ndarray]: Headers and data array.
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
    Compare two CSV files for header and numerical data equality.

    Args:
        generated (Path): Path to generated CSV file.
        golden (Path): Path to golden reference CSV file.
        rtol (float): Relative tolerance for numerical comparison.
        atol (float): Absolute tolerance for numerical comparison.

    Raises:
        AssertionError: If any mismatch is found.
    """
    # Load both files
    gen_headers, gen_data = load_csv_data(generated)
    gold_headers, gold_data = load_csv_data(golden)

    # Exact header comparison for individual files
    try:
        assert (
            gen_headers == gold_headers
        ), f"Headers mismatch:\nGenerated: {gen_headers}\nGolden: {gold_headers}"
    except AssertionError as e:
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

        # Check if NaN positions match
        try:
            assert np.array_equal(
                gen_nan_mask, gold_nan_mask
            ), f"NaN positions don't match in {generated.name}"
        except AssertionError as e:
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


def compare_iv_summary_csv(generated: Path, golden: Path) -> None:
    """
    Compare IV summary CSV files with appropriate tolerances.

    IV summary files format:
        Voltage (mV) | File1 (pA) | File2 (pA) | ...

    Args:
        generated (Path): Path to generated summary CSV.
        golden (Path): Path to golden reference summary CSV.

    Raises:
        AssertionError: If any mismatch is found.
    """
    gen_headers, gen_data = load_csv_data(generated)
    gold_headers, gold_data = load_csv_data(golden)

    # Check header count matches
    try:
        assert len(gen_headers) == len(
            gold_headers
        ), f"Header count mismatch: {len(gen_headers)} vs {len(gold_headers)}"
    except AssertionError as e:
        raise AssertionError(
            f"IV summary header validation failed\n"
            f"Generated file: {generated}\n"
            f"Golden file: {golden}\n"
            f"{str(e)}"
        )

    # First header should be Voltage
    assert (
        "Voltage" in gen_headers[0]
    ), f"First column should be Voltage, got: {gen_headers[0]}"

    # Compare data shape
    try:
        assert (
            gen_data.shape == gold_data.shape
        ), f"Data shape mismatch:\nGenerated: {gen_data.shape}\nGolden: {gold_data.shape}"
    except AssertionError as e:
        raise AssertionError(
            f"IV summary shape validation failed\n"
            f"Generated file: {generated}\n"
            f"Golden file: {golden}\n"
            f"{str(e)}"
        )

    if gen_data.size > 0:
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
            raise AssertionError(
                f"IV summary voltage column validation failed\n"
                f"Generated file: {generated}\n"
                f"Golden file: {golden}\n"
                f"{str(e)}"
            )

        # Current columns - use appropriate tolerance
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
                raise AssertionError(
                    f"IV summary column {col_idx} NaN validation failed\n"
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
                        atol=1e-2,  # 0.01 pA tolerance
                        err_msg=f"Current mismatch in column {col_idx} ({gen_headers[col_idx]})",
                    )
                except AssertionError as e:
                    diff = np.abs(col_gen[valid_mask] - col_gold[valid_mask])
                    max_diff = np.max(diff)
                    raise AssertionError(
                        f"IV summary column {col_idx} validation failed\n"
                        f"Column header: {gen_headers[col_idx]}\n"
                        f"Max difference: {max_diff:.6e} pA\n"
                        f"Generated file: {generated}\n"
                        f"Golden file: {golden}\n"
                        f"{str(e)}"
                    )


class BatchIVAnalysisTestBase:
    """
    Base class for batch IV analysis workflow tests.

    Subclasses should specify FILE_TYPE and FILE_EXTENSION.
    Provides fixtures and utility methods for batch analysis validation.
    """

    # Subclasses should define these
    FILE_TYPE = None  # 'abf' or 'mat'
    FILE_EXTENSION = None  # '*.abf' or '*.mat'

    @property
    def sample_data_dir(self) -> Path:
        """
        Get the sample data directory for the current file type.

        Returns:
            Path: Directory containing sample data files.
        """
        # Using uppercase directory names as shown in filetree
        if self.FILE_TYPE == "abf":
            return Path("tests/fixtures/sample_data/IV+CD/abf")
        else:
            return Path(f"tests/fixtures/sample_data/IV+CD/{self.FILE_TYPE}")

    @property
    def golden_data_dir(self) -> Path:
        """
        Get the golden data directory for the current file type.

        Returns:
            Path: Directory containing golden reference files.
        """
        return Path(f"tests/fixtures/golden_data/golden_IV/{self.FILE_TYPE}")

    @pytest.fixture
    def analysis_params(self):
        """
        Create analysis parameters matching the GUI state.

        Returns:
            AnalysisParameters: Configured analysis parameters.
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
        """
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir)

    def get_test_files(self) -> List[str]:
        """
        Get all test files from the sample data directory.

        Returns:
            List[str]: List of file paths.

        Skips test if no files are found.
        """
        if not self.sample_data_dir.exists():
            pytest.skip(f"Sample data directory not found: {self.sample_data_dir}")

        test_files = list(self.sample_data_dir.glob(self.FILE_EXTENSION))
        if not test_files:
            pytest.skip(
                f"No {self.FILE_TYPE.upper()} files found in {self.sample_data_dir}"
            )

        return [str(f) for f in sorted(test_files)]

    def test_batch_iv_analysis_workflow(self, analysis_params, temp_output_dir):
        """
        Test the complete batch IV analysis workflow with golden file validation.

        Steps:
            1. Set analysis parameters
            2. Initialize services
            3. Load files
            4. Run batch analysis
            5. Validate results
            6. Export IV summary
            7. Export individual CSVs
            8. Validate outputs against golden files

        Args:
            analysis_params (AnalysisParameters): Analysis parameters fixture.
            temp_output_dir (str): Temporary output directory fixture.

        Raises:
            AssertionError: If any validation fails.
        """

        # ==================================================================
        # STEP 1: SET ANALYSIS PARAMETERS (already done via fixture)
        # ==================================================================
        # analysis_params created by fixture with specified values

        # ==================================================================
        # STEP 2: INITIALIZE SERVICES (as MainWindow does)
        # ==================================================================
        batch_processor = BatchProcessor()
        data_manager = DataManager()

        # ==================================================================
        # STEP 3: LOAD FILES (BatchAnalysisDialog.add_files)
        # ==================================================================
        test_files = self.get_test_files()
        assert (
            len(test_files) == 12
        ), f"Expected 12 {self.FILE_TYPE.upper()} files, found {len(test_files)}"

        print(f"\n{'='*60}")
        print(f"Testing {self.FILE_TYPE.upper()} Batch IV Analysis Workflow")
        print(f"{'='*60}")
        print(f"Processing {len(test_files)} files...")
        print(
            f"Parameters: Range [{analysis_params.range1_start}, {analysis_params.range1_end}] ms"
        )
        print(
            f"X-axis: {analysis_params.x_axis.measure} {analysis_params.x_axis.channel}"
        )
        print(
            f"Y-axis: {analysis_params.y_axis.measure} {analysis_params.y_axis.channel}"
        )

        # ==================================================================
        # STEP 4: START ANALYSIS (BatchAnalysisWorker.run)
        # ==================================================================
        # BatchProcessor.process_files() - exactly as GUI does
        batch_result = batch_processor.process_files(
            file_paths=test_files, params=analysis_params
        )

        # Validate batch processing results
        assert (
            len(batch_result.successful_results) == 12
        ), f"Expected 12 successful results, got {len(batch_result.successful_results)}"
        assert (
            len(batch_result.failed_results) == 0
        ), f"Unexpected failures: {[r.file_path for r in batch_result.failed_results]}"

        print(
            f"✓ Batch analysis complete: {batch_result.success_rate:.1f}% success rate"
        )

        # ==================================================================
        # STEP 5: VIEW RESULTS (BatchResultsWindow opens)
        # ==================================================================
        # In the GUI, this would open BatchResultsWindow
        # We'll verify the data structure is correct

        # Ensure selected_files is initialized (as BatchResultsWindow does)
        if (
            not hasattr(batch_result, "selected_files")
            or batch_result.selected_files is None
        ):
            batch_result = replace(
                batch_result,
                selected_files={r.base_name for r in batch_result.successful_results},
            )

        # Sort results as the GUI does
        sorted_results = sorted(
            batch_result.successful_results,
            key=lambda r: (
                int(r.base_name.split("_")[-1])
                if r.base_name.split("_")[-1].isdigit()
                else 0
            ),
        )

        # ==================================================================
        # STEP 6: EXPORT IV SUMMARY (BatchResultsWindow._export_iv_summary)
        # ==================================================================
        print("\nExporting IV Summary...")

        # Prepare batch data structure as GUI does
        batch_data = {
            r.base_name: {
                "x_values": r.x_data.tolist(),
                "y_values": r.y_data.tolist(),
                "x_values2": r.x_data2.tolist() if r.x_data2 is not None else None,
                "y_values2": r.y_data2.tolist() if r.y_data2 is not None else None,
            }
            for r in sorted_results
        }

        # IVAnalysisService.prepare_iv_data()
        iv_data_r1, iv_file_mapping, iv_data_r2 = IVAnalysisService.prepare_iv_data(
            batch_data, batch_result.parameters
        )

        # Verify IV data structure
        assert (
            len(iv_data_r1) == 11
        ), f"Expected 11 voltage points, got {len(iv_data_r1)}"
        assert all(
            len(currents) == 12 for currents in iv_data_r1.values()
        ), "Each voltage should have 12 current measurements"

        # Get current units from parameters
        current_units = analysis_params.channel_config.get("current_units", "pA")

        # IVSummaryExporter.prepare_summary_table()
        selected_files = batch_result.selected_files
        iv_summary_table = IVSummaryExporter.prepare_summary_table(
            iv_data_r1, iv_file_mapping, selected_files, current_units
        )

        # DataManager.export_to_csv() for summary
        iv_summary_path = os.path.join(temp_output_dir, "IV_Summary.csv")
        summary_result = data_manager.export_to_csv(iv_summary_table, iv_summary_path)
        assert (
            summary_result.success
        ), f"IV summary export failed: {summary_result.error_message}"
        print(f"✓ Exported IV summary with {summary_result.records_exported} records")

        # ==================================================================
        # STEP 7: EXPORT INDIVIDUAL CSVs (BatchResultsWindow._export_individual_csvs)
        # ==================================================================
        print("\nExporting individual CSVs...")

        # Create filtered batch result (in GUI, this uses selected files)
        filtered_batch = replace(
            batch_result,
            successful_results=sorted_results,
            selected_files=selected_files,
        )

        # BatchProcessor.export_results()
        individual_output_dir = os.path.join(temp_output_dir, "individual_csvs")
        os.makedirs(individual_output_dir, exist_ok=True)

        export_result = batch_processor.export_results(
            filtered_batch, individual_output_dir
        )
        assert (
            export_result.success_count == 12
        ), f"Expected 12 successful exports, got {export_result.success_count}"
        print(f"✓ Exported {export_result.success_count} individual CSV files")

        # ==================================================================
        # STEP 8: VALIDATE AGAINST GOLDEN FILES
        # ==================================================================
        print("\nValidating against golden reference files...")

        # Get expected file names (without bracketed parts)
        expected_files = [
            "250514_001",
            "250514_002",
            "250514_003",
            "250514_004",
            "250514_005",
            "250514_006",
            "250514_007",
            "250514_008",
            "250514_009",
            "250514_010",
            "250514_011",
            "250514_012",
        ]

        # Validate individual CSV files
        print("  Individual CSVs:")
        for file_name in expected_files:
            generated_csv = Path(individual_output_dir) / f"{file_name}.csv"
            golden_csv = self.golden_data_dir / f"{file_name}.csv"

            print(f"    Comparing {file_name}.csv...", end=" ")

            try:
                compare_csv_files(
                    generated_csv,
                    golden_csv,
                    rtol=1e-4,  # 0.01% relative tolerance
                    atol=1e-2,  # 0.01 absolute tolerance for small values
                )
                print("✓")
            except AssertionError as e:
                print("✗")
                raise AssertionError(
                    f"\nValidation failed for individual file: {file_name}.csv\n"
                    f"{str(e)}"
                )

        # Validate IV summary CSV
        print("  IV Summary:")
        print("    Comparing IV_Summary.csv...", end=" ")

        # Determine golden summary name (different for ABF vs MAT)
        if self.FILE_TYPE == "mat":
            golden_summary_name = "Summary IV.csv"
        else:
            golden_summary_name = "IV_Summary.csv"

        golden_summary = self.golden_data_dir / golden_summary_name

        try:
            compare_iv_summary_csv(Path(iv_summary_path), golden_summary)
            print("✓")
        except AssertionError as e:
            print("✗")
            raise AssertionError(
                f"\nValidation failed for IV summary file\n"
                f"Number of files: {len(expected_files)}\n"
                f"{str(e)}"
            )

        print(f"\n{'='*60}")
        print(f"✓ All {self.FILE_TYPE.upper()} batch IV analysis tests passed!")
        print(f"{'='*60}\n")


class TestBatchIVAnalysisABF(BatchIVAnalysisTestBase):
    """
    Test batch IV analysis workflow with ABF files.

    Inherits from BatchIVAnalysisTestBase.
    """

    FILE_TYPE = "abf"
    FILE_EXTENSION = "*.abf"


class TestBatchIVAnalysisWCP(BatchIVAnalysisTestBase):
    FILE_TYPE = "wcp" 
    FILE_EXTENSION = "*.wcp"
    # Uses auto-detected channel config per file


if __name__ == "__main__":
    # Run the test directly
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
