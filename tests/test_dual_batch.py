"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

"""Test dual range batch analysis workflow with WCP files."""

import pytest
from pathlib import Path
import numpy as np
import csv

from data_analysis_gui.core.app_controller import ApplicationController
from data_analysis_gui.core.params import AnalysisParameters, AxisConfig


class TestDualRangeBatchWorkflow:
    """Test dual range batch analysis workflow with WCP files."""

    @pytest.fixture
    def analysis_parameters(self):
        """Create dual range analysis parameters."""
        return AnalysisParameters(
            range1_start=50.45,
            range1_end=249.8,
            use_dual_range=True,
            range2_start=250.45,
            range2_end=449.5,
            x_axis=AxisConfig(measure="Time", channel=None, peak_type=None),
            y_axis=AxisConfig(measure="Average", channel="Current", peak_type=None),
            channel_config={"voltage": 0, "current": 1},
        )

    @pytest.fixture
    def controller(self):
        """Create ApplicationController."""
        return ApplicationController()

    @pytest.fixture
    def batch_processor(self, controller):
        """Get BatchProcessor from controller."""
        return controller.batch_processor

    def get_sample_files(self, file_format):
        """Get sample files for testing."""
        search_path = (
            Path(__file__).parent
            / "fixtures"
            / "sample_data"
            / "dual_range"
            / file_format
        )
        pattern = f"*.{file_format}"
        files = sorted(search_path.glob(pattern))
        return [str(f) for f in files]

    def get_golden_files(self, file_format):
        """Get golden CSV files for comparison."""
        golden_path = (
            Path(__file__).parent
            / "fixtures"
            / "golden_data"
            / "golden_dual_range"
            / file_format
        )

        golden_files = {}
        for csv_file in sorted(golden_path.glob("*.csv")):
            base_name = csv_file.stem
            golden_files[base_name] = csv_file

        return golden_files

    def compare_csv_files(self, generated_path, golden_path, rtol=1e-6, atol=1e-9):
        """Compare generated CSV against golden CSV."""

        def load_csv_data(filepath):
            """Load CSV headers and data array."""
            with open(filepath, "r") as f:
                reader = csv.reader(f)
                headers = next(reader)
                if headers[0].startswith("#"):
                    headers[0] = headers[0][1:].strip()
                data = []
                for row in reader:
                    data.append(
                        [float(val) if val and val != "nan" else np.nan for val in row]
                    )
            return headers, np.array(data)

        gen_headers, gen_data = load_csv_data(generated_path)
        gold_headers, gold_data = load_csv_data(golden_path)

        assert (
            gen_headers == gold_headers
        ), f"Headers mismatch:\nGenerated: {gen_headers}\nGolden: {gold_headers}"

        assert (
            gen_data.shape == gold_data.shape
        ), f"Shape mismatch: generated {gen_data.shape} vs golden {gold_data.shape}"

        if gen_data.size > 0:
            gen_nan_mask = np.isnan(gen_data)
            gold_nan_mask = np.isnan(gold_data)

            assert np.array_equal(
                gen_nan_mask, gold_nan_mask
            ), "NaN positions don't match"

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
        """Run complete dual range batch workflow."""
        # Get files to analyze
        file_paths = self.get_sample_files(file_format)
        assert (
            len(file_paths) > 0
        ), f"No {file_format} files found in sample_data/dual_range"

        # Run batch analysis
        batch_result = batch_processor.process_files(
            file_paths=file_paths, params=analysis_parameters
        )

        # Verify all files processed successfully
        assert (
            len(batch_result.failed_results) == 0
        ), f"Some files failed: {[r.error_message for r in batch_result.failed_results]}"
        assert len(batch_result.successful_results) == len(
            file_paths
        ), f"Expected {len(file_paths)} results, got {len(batch_result.successful_results)}"

        # Export individual CSVs
        output_dir_path = tmp_path / "exports"
        output_dir_path.mkdir(exist_ok=True)

        export_result = batch_processor.export_results(
            batch_result=batch_result, output_dir=str(output_dir_path)
        )

        assert export_result.success_count == len(
            batch_result.successful_results
        ), f"Not all exports succeeded: {export_result.success_count}/{len(batch_result.successful_results)}"

        # Compare generated CSVs against golden files
        golden_files = self.get_golden_files(file_format)

        for file_result in batch_result.successful_results:
            base_name = file_result.base_name

            if base_name in golden_files:
                generated_csv = output_dir_path / f"{base_name}.csv"
                assert (
                    generated_csv.exists()
                ), f"Expected CSV not found: {generated_csv}"

                golden_csv = golden_files[base_name]

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


class TestDualRangeWCP(TestDualRangeBatchWorkflow):
    """Test dual range batch workflow with WCP files."""

    @pytest.fixture
    def file_format(self):
        """File format for this test."""
        return "wcp"

    def test_wcp_dual_range_workflow(
        self, controller, batch_processor, analysis_parameters, file_format, tmp_path
    ):
        """Test complete dual range batch workflow with WCP files."""
        self._run_dual_range_batch_workflow(
            controller, batch_processor, analysis_parameters, file_format, tmp_path
        )