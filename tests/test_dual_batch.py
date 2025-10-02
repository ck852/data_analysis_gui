"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

"""
Test dual range batch analysis workflow following the exact GUI workflow.

This module tests the dual range batch analysis workflow using both ABF and WCP file formats.
It follows the same function calls as the GUI and compares results against golden reference files.
"""

import pytest
from pathlib import Path
import numpy as np
import csv

# Import the exact same components used in the GUI workflow
from data_analysis_gui.core.app_controller import ApplicationController
from data_analysis_gui.core.params import AnalysisParameters, AxisConfig


class TestDualRangeBatchWorkflow:
    """
    Base class for testing dual range batch analysis workflow.

    This class tests both ABF and MAT file formats using the exact workflow as the GUI.
    It provides fixtures and utility methods for running the workflow and validating results.
    """

    @pytest.fixture
    def analysis_parameters(self):
        """
        Create AnalysisParameters as ControlPanel.get_parameters() would.

        Returns:
            AnalysisParameters: Configured parameters for dual range analysis.
        """
        return AnalysisParameters(
            # Range 1 settings (50.45 - 249.8 ms)
            range1_start=50.45,
            range1_end=249.8,
            # Enable dual range and set Range 2 (250.45 - 449.5 ms)
            use_dual_range=True,
            range2_start=250.45,
            range2_end=449.5,
            # X-Axis: Time (no channel needed for Time)
            x_axis=AxisConfig(measure="Time", channel=None, peak_type=None),
            # Y-Axis: Average Current
            y_axis=AxisConfig(measure="Average", channel="Current", peak_type=None),
            # Channel configuration (default)
            channel_config={"voltage": 0, "current": 1},
        )

    @pytest.fixture
    def controller(self):
        """
        Create ApplicationController as MainWindow would.

        Returns:
            ApplicationController: Initialized controller with all services.
        """
        controller = ApplicationController()
        
        return controller

    @pytest.fixture
    def batch_processor(self, controller):
        """
        Get the BatchProcessor from the controller.

        Returns:
            BatchProcessor: Batch processor instance from the controller.
        """
        return controller.batch_processor

    def get_sample_files(self, file_format):
        """
        Retrieve sample files for testing from the filetree structure.

        Args:
            file_format (str): 'abf' or 'wcp'

        Returns:
            List[str]: List of file paths in the sample_data/dual_range/{format} directory.
        """
        # Define the path to the format-specific subdirectory
        search_path = (
            Path(__file__).parent
            / "fixtures"
            / "sample_data"
            / "dual_range"
            / file_format
        )

        # Get all files with the specified extension
        pattern = f"*.{file_format}"
        files = sorted(search_path.glob(pattern))

        # Convert to string paths as the GUI would pass them
        return [str(f) for f in files]

    def get_golden_files(self, file_format):
        """
        Retrieve golden CSV files for comparison.

        Args:
            file_format (str): 'abf' or 'mat'

        Returns:
            Dict[str, Path]: Mapping of base_name to golden CSV path.
        """
        golden_path = (
            Path(__file__).parent
            / "fixtures"
            / "golden_data"
            / "golden_dual_range"
            / file_format
        )

        golden_files = {}
        for csv_file in sorted(golden_path.glob("*.csv")):
            # Extract base name (e.g., "250202_007" from "250202_007.csv")
            base_name = csv_file.stem
            golden_files[base_name] = csv_file

        return golden_files

    def compare_csv_files(self, generated_path, golden_path, rtol=1e-6, atol=1e-9):
        """
        Compare a generated CSV file against a golden CSV file.

        Args:
            generated_path (Path): Path to the generated CSV.
            golden_path (Path): Path to the golden CSV.
            rtol (float): Relative tolerance for numeric comparison.
            atol (float): Absolute tolerance for numeric comparison.

        Raises:
            AssertionError: If headers, shapes, NaN positions, or numeric values do not match.
        """

        # Helper function to load CSV data
        def load_csv_data(filepath):
            """Load CSV headers and data array from file."""
            with open(filepath, "r") as f:
                reader = csv.reader(f)
                headers = next(reader)
                # Remove '#' prefix if present
                if headers[0].startswith("#"):
                    headers[0] = headers[0][1:].strip()
                data = []
                for row in reader:
                    data.append(
                        [float(val) if val and val != "nan" else np.nan for val in row]
                    )
            return headers, np.array(data)

        # Load both files
        gen_headers, gen_data = load_csv_data(generated_path)
        gold_headers, gold_data = load_csv_data(golden_path)

        # Check headers
        assert (
            gen_headers == gold_headers
        ), f"Headers mismatch:\nGenerated: {gen_headers}\nGolden: {gold_headers}"

        # Check shape
        assert (
            gen_data.shape == gold_data.shape
        ), f"Shape mismatch: generated {gen_data.shape} vs golden {gold_data.shape}"

        # Compare numeric data
        if gen_data.size > 0:
            # Check for NaN positions matching
            gen_nan_mask = np.isnan(gen_data)
            gold_nan_mask = np.isnan(gold_data)

            assert np.array_equal(
                gen_nan_mask, gold_nan_mask
            ), "NaN positions don't match"

            # Compare non-NaN values
            if not np.all(gen_nan_mask):
                valid_mask = ~gen_nan_mask
                np.testing.assert_allclose(
                    gen_data[valid_mask],
                    gold_data[valid_mask],
                    rtol=rtol,
                    atol=atol,
                    err_msg="Data mismatch",
                )

    def _run_dual_range_batch_workflow(
        self, controller, batch_processor, analysis_parameters, file_format, tmp_path
    ):
        """
        Run the complete dual range batch workflow following the GUI workflow.

        Steps:
            1. Load files (add_files)
            2. Start analysis (BatchProcessor.process_files)
            3. Export individual CSVs (BatchProcessor.export_results)
            4. Compare against golden files

        Args:
            controller (ApplicationController): Controller fixture.
            batch_processor (BatchProcessor): Batch processor instance.
            analysis_parameters (AnalysisParameters): Dual range analysis parameters.
            file_format (str): 'abf' or 'mat'.
            tmp_path (Path): Pytest temporary directory fixture.

        Raises:
            AssertionError: If any validation step fails.
        """
        # Step 1: Get the files to analyze (equivalent to BatchAnalysisDialog.add_files)
        file_paths = self.get_sample_files(file_format)
        assert (
            len(file_paths) > 0
        ), f"No {file_format} files found in sample_data/dual_range"

        # Step 2: Run batch analysis (equivalent to BatchAnalysisWorker.run)
        # This calls BatchProcessor.process_files exactly as the GUI would
        batch_result = batch_processor.process_files(
            file_paths=file_paths, params=analysis_parameters
        )

        # Verify all files were processed successfully
        assert (
            len(batch_result.failed_results) == 0
        ), f"Some files failed: {[r.error_message for r in batch_result.failed_results]}"
        assert len(batch_result.successful_results) == len(
            file_paths
        ), f"Expected {len(file_paths)} results, got {len(batch_result.successful_results)}"

        # Step 3: Export individual CSVs (equivalent to BatchResultsWindow._export_individual_csvs)
        # This calls BatchProcessor.export_results exactly as the GUI would
        output_dir_path = tmp_path / "exports"
        output_dir_path.mkdir(exist_ok=True)

        export_result = batch_processor.export_results(
            batch_result=batch_result, output_dir=str(output_dir_path)
        )

        # Verify exports were successful
        assert export_result.success_count == len(
            batch_result.successful_results
        ), f"Not all exports succeeded: {export_result.success_count}/{len(batch_result.successful_results)}"

        # Step 4: Compare generated CSVs against golden files
        golden_files = self.get_golden_files(file_format)

        for file_result in batch_result.successful_results:
            base_name = file_result.base_name

            # Check if we have a golden file for this result
            if base_name in golden_files:
                generated_csv = output_dir_path / f"{base_name}.csv"
                assert (
                    generated_csv.exists()
                ), f"Expected CSV not found: {generated_csv}"

                golden_csv = golden_files[base_name]

                # Compare the CSVs
                try:
                    self.compare_csv_files(generated_csv, golden_csv)
                    print(f"✓ {base_name}.csv matches golden file")
                except AssertionError as e:
                    pytest.fail(f"CSV comparison failed for {base_name}: {e}")
            else:
                pytest.skip(f"No golden file for {base_name}")

        # Verify we tested at least one file
        tested_count = sum(
            1 for r in batch_result.successful_results if r.base_name in golden_files
        )
        assert tested_count > 0, "No files were compared against golden files"

        print(f"\n✓ All {tested_count} {file_format.upper()} files match golden data")


class TestDualRangeABF(TestDualRangeBatchWorkflow):
    """
    Test dual range batch workflow with ABF files.

    Inherits from TestDualRangeBatchWorkflow and sets file format to ABF.
    """

    @pytest.fixture
    def file_format(self):
        """
        Specify the file format for this test.

        Returns:
            str: 'abf'
        """
        return "abf"

    def test_abf_dual_range_workflow(
        self, controller, batch_processor, analysis_parameters, file_format, tmp_path
    ):
        """
        Test the complete dual range batch workflow with ABF files.

        Args:
            controller (ApplicationController): Controller fixture.
            batch_processor (BatchProcessor): Batch processor instance.
            analysis_parameters (AnalysisParameters): Dual range analysis parameters.
            file_format (str): 'abf'.
            tmp_path (Path): Pytest temporary directory fixture.
        """
        self._run_dual_range_batch_workflow(
            controller, batch_processor, analysis_parameters, file_format, tmp_path
        )

class TestDualRangeWCP(TestDualRangeBatchWorkflow):
    @pytest.fixture
    def file_format(self):
        """
        Specify the file format for this test.

        Returns:
            str: 'wcp'
        """
        return "wcp"

    def test_wcp_dual_range_workflow(
        self, controller, batch_processor, analysis_parameters, file_format, tmp_path
    ):
        """
        Test the complete dual range batch workflow with WCP files.

        Args:
            controller (ApplicationController): Controller fixture.
            batch_processor (BatchProcessor): Batch processor instance.
            analysis_parameters (AnalysisParameters): Dual range analysis parameters.
            file_format (str): 'wcp'.
            tmp_path (Path): Pytest temporary directory fixture.
        """
        self._run_dual_range_batch_workflow(
            controller, batch_processor, analysis_parameters, file_format, tmp_path
        )