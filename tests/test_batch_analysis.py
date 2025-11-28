"""
PatchBatch Electrophysiology Data Analysis Tool

Test script for batch IV analysis workflow with golden file validation.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
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
    """Load CSV headers and numeric data from file."""
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
    """Compare generated CSV against golden reference within tolerance."""
    gen_headers, gen_data = load_csv_data(generated)
    gold_headers, gold_data = load_csv_data(golden)

    # Header comparison
    try:
        assert gen_headers == gold_headers, (
            f"Headers mismatch:\nGenerated: {gen_headers}\nGolden: {gold_headers}"
        )
    except AssertionError as e:
        raise AssertionError(
            f"Header validation failed for {generated.name}\n"
            f"Generated file: {generated}\n"
            f"Golden file: {golden}\n"
            f"{str(e)}"
        )

    # Shape comparison
    try:
        assert gen_data.shape == gold_data.shape, (
            f"Data shape mismatch:\nGenerated: {gen_data.shape}\nGolden: {gold_data.shape}"
        )
    except AssertionError as e:
        raise AssertionError(
            f"Shape validation failed for {generated.name}\n"
            f"Generated file: {generated}\n"
            f"Golden file: {golden}\n"
            f"{str(e)}"
        )

    # Data comparison
    if gen_data.size > 0:
        gen_nan_mask = np.isnan(gen_data)
        gold_nan_mask = np.isnan(gold_data)

        try:
            assert np.array_equal(gen_nan_mask, gold_nan_mask), (
                f"NaN positions don't match in {generated.name}"
            )
        except AssertionError as e:
            raise AssertionError(
                f"NaN position validation failed for {generated.name}\n"
                f"Generated file: {generated}\n"
                f"Golden file: {golden}\n"
                f"{str(e)}"
            )

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
                raise AssertionError(
                    f"Numerical validation failed for {generated.name}\n"
                    f"Generated file: {generated}\n"
                    f"Golden file: {golden}\n"
                    f"Max difference: {diff[max_diff_idx]:.6e}\n"
                    f"Generated value: {gen_data[valid_mask][max_diff_idx]:.6f}\n"
                    f"Golden value: {gold_data[valid_mask][max_diff_idx]:.6f}\n"
                    f"Tolerance: rtol={rtol}, atol={atol}\n"
                    f"{str(e)}"
                )


def compare_iv_summary_csv(generated: Path, golden: Path) -> None:
    """Compare IV summary CSVs (Voltage column + per-file current columns)."""
    gen_headers, gen_data = load_csv_data(generated)
    gold_headers, gold_data = load_csv_data(golden)

    try:
        assert len(gen_headers) == len(gold_headers), (
            f"Header count mismatch: {len(gen_headers)} vs {len(gold_headers)}"
        )
    except AssertionError as e:
        raise AssertionError(
            f"IV summary header validation failed\n"
            f"Generated file: {generated}\n"
            f"Golden file: {golden}\n"
            f"{str(e)}"
        )

    assert "Voltage" in gen_headers[0], f"First column should be Voltage, got: {gen_headers[0]}"

    try:
        assert gen_data.shape == gold_data.shape, (
            f"Data shape mismatch:\nGenerated: {gen_data.shape}\nGolden: {gold_data.shape}"
        )
    except AssertionError as e:
        raise AssertionError(
            f"IV summary shape validation failed\n"
            f"Generated file: {generated}\n"
            f"Golden file: {golden}\n"
            f"{str(e)}"
        )

    if gen_data.size > 0:
        # Voltage column - tight tolerance
        try:
            np.testing.assert_allclose(
                gen_data[:, 0], gold_data[:, 0],
                rtol=1e-4, atol=0.1,
                err_msg="Voltage column mismatch",
            )
        except AssertionError as e:
            raise AssertionError(
                f"IV summary voltage column validation failed\n"
                f"Generated file: {generated}\n"
                f"Golden file: {golden}\n"
                f"{str(e)}"
            )

        # Current columns
        for col_idx in range(1, gen_data.shape[1]):
            col_gen = gen_data[:, col_idx]
            col_gold = gold_data[:, col_idx]

            gen_nan_mask = np.isnan(col_gen)
            gold_nan_mask = np.isnan(col_gold)

            try:
                assert np.array_equal(gen_nan_mask, gold_nan_mask), (
                    f"NaN positions don't match in column {col_idx}"
                )
            except AssertionError as e:
                raise AssertionError(
                    f"IV summary column {col_idx} NaN validation failed\n"
                    f"Column header: {gen_headers[col_idx]}\n"
                    f"Generated file: {generated}\n"
                    f"Golden file: {golden}\n"
                    f"{str(e)}"
                )

            valid_mask = ~gen_nan_mask
            if np.any(valid_mask):
                try:
                    np.testing.assert_allclose(
                        col_gen[valid_mask], col_gold[valid_mask],
                        rtol=1e-4, atol=1e-2,
                        err_msg=f"Current mismatch in column {col_idx} ({gen_headers[col_idx]})",
                    )
                except AssertionError as e:
                    diff = np.abs(col_gen[valid_mask] - col_gold[valid_mask])
                    raise AssertionError(
                        f"IV summary column {col_idx} validation failed\n"
                        f"Column header: {gen_headers[col_idx]}\n"
                        f"Max difference: {np.max(diff):.6e} pA\n"
                        f"Generated file: {generated}\n"
                        f"Golden file: {golden}\n"
                        f"{str(e)}"
                    )


class BatchIVAnalysisTestBase:
    """Base class for batch IV analysis tests. Subclasses set FILE_TYPE and FILE_EXTENSION."""

    FILE_TYPE = None
    FILE_EXTENSION = None

    @property
    def sample_data_dir(self) -> Path:
        if self.FILE_TYPE == "abf":
            return Path("tests/fixtures/sample_data/IV+CD/abf")
        else:
            return Path(f"tests/fixtures/sample_data/IV+CD/{self.FILE_TYPE}")

    @property
    def golden_data_dir(self) -> Path:
        return Path(f"tests/fixtures/golden_data/golden_IV/{self.FILE_TYPE}")

    @pytest.fixture
    def analysis_params(self):
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
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def get_test_files(self) -> List[str]:
        if not self.sample_data_dir.exists():
            pytest.skip(f"Sample data directory not found: {self.sample_data_dir}")

        test_files = list(self.sample_data_dir.glob(self.FILE_EXTENSION))
        if not test_files:
            pytest.skip(f"No {self.FILE_TYPE.upper()} files found in {self.sample_data_dir}")

        return [str(f) for f in sorted(test_files)]

    def test_batch_iv_analysis_workflow(self, analysis_params, temp_output_dir):
        """Full batch IV workflow: analyze files, export summary and individual CSVs, validate against golden files."""
        
        # Initialize services
        batch_processor = BatchProcessor()
        data_manager = DataManager()

        # Load and validate test files
        test_files = self.get_test_files()
        assert len(test_files) == 12, f"Expected 12 {self.FILE_TYPE.upper()} files, found {len(test_files)}"

        print(f"\n{'='*60}")
        print(f"Testing {self.FILE_TYPE.upper()} Batch IV Analysis Workflow")
        print(f"{'='*60}")
        print(f"Processing {len(test_files)} files...")
        print(f"Parameters: Range [{analysis_params.range1_start}, {analysis_params.range1_end}] ms")
        print(f"X-axis: {analysis_params.x_axis.measure} {analysis_params.x_axis.channel}")
        print(f"Y-axis: {analysis_params.y_axis.measure} {analysis_params.y_axis.channel}")

        # Run batch analysis
        batch_result = batch_processor.process_files(file_paths=test_files, params=analysis_params)

        assert len(batch_result.successful_results) == 12, (
            f"Expected 12 successful results, got {len(batch_result.successful_results)}"
        )
        assert len(batch_result.failed_results) == 0, (
            f"Unexpected failures: {[r.file_path for r in batch_result.failed_results]}"
        )

        print(f"✓ Batch analysis complete: {batch_result.success_rate:.1f}% success rate")

        # Initialize selection state
        if not hasattr(batch_result, "selected_files") or batch_result.selected_files is None:
            batch_result = replace(
                batch_result,
                selected_files={r.base_name for r in batch_result.successful_results},
            )

        sorted_results = sorted(
            batch_result.successful_results,
            key=lambda r: int(r.base_name.split("_")[-1]) if r.base_name.split("_")[-1].isdigit() else 0,
        )

        # Export IV summary
        print("\nExporting IV Summary...")

        batch_data = {
            r.base_name: {
                "x_values": r.x_data.tolist(),
                "y_values": r.y_data.tolist(),
                "x_values2": r.x_data2.tolist() if r.x_data2 is not None else None,
                "y_values2": r.y_data2.tolist() if r.y_data2 is not None else None,
            }
            for r in sorted_results
        }

        iv_data_r1, iv_file_mapping, iv_data_r2 = IVAnalysisService.prepare_iv_data(
            batch_data, batch_result.parameters
        )

        assert len(iv_data_r1) == 11, f"Expected 11 voltage points, got {len(iv_data_r1)}"
        assert all(len(currents) == 12 for currents in iv_data_r1.values())

        current_units = analysis_params.channel_config.get("current_units", "pA")
        selected_files = batch_result.selected_files
        iv_summary_table = IVSummaryExporter.prepare_summary_table(
            iv_data_r1, iv_file_mapping, selected_files, current_units
        )

        iv_summary_path = os.path.join(temp_output_dir, "IV_Summary.csv")
        summary_result = data_manager.export_to_csv(iv_summary_table, iv_summary_path)
        assert summary_result.success, f"IV summary export failed: {summary_result.error_message}"
        print(f"✓ Exported IV summary with {summary_result.records_exported} records")

        # Export individual CSVs
        print("\nExporting individual CSVs...")

        filtered_batch = replace(
            batch_result,
            successful_results=sorted_results,
            selected_files=selected_files,
        )

        individual_output_dir = os.path.join(temp_output_dir, "individual_csvs")
        os.makedirs(individual_output_dir, exist_ok=True)

        export_result = batch_processor.export_results(filtered_batch, individual_output_dir)
        assert export_result.success_count == 12, (
            f"Expected 12 successful exports, got {export_result.success_count}"
        )
        print(f"✓ Exported {export_result.success_count} individual CSV files")

        # Validate against golden files
        print("\nValidating against golden reference files...")

        expected_files = [f"250514_{i:03d}" for i in range(1, 13)]

        print("  Individual CSVs:")
        for file_name in expected_files:
            generated_csv = Path(individual_output_dir) / f"{file_name}.csv"
            golden_csv = self.golden_data_dir / f"{file_name}.csv"

            print(f"    Comparing {file_name}.csv...", end=" ")
            try:
                compare_csv_files(generated_csv, golden_csv, rtol=1e-4, atol=1e-2)
                print("✓")
            except AssertionError as e:
                print("✗")
                raise AssertionError(f"\nValidation failed for: {file_name}.csv\n{str(e)}")

        # Validate IV summary
        print("  IV Summary:")
        print("    Comparing IV_Summary.csv...", end=" ")

        golden_summary_name = "Summary IV.csv" if self.FILE_TYPE == "mat" else "IV_Summary.csv"
        golden_summary = self.golden_data_dir / golden_summary_name

        try:
            compare_iv_summary_csv(Path(iv_summary_path), golden_summary)
            print("✓")
        except AssertionError as e:
            print("✗")
            raise AssertionError(f"\nValidation failed for IV summary file\n{str(e)}")

        print(f"\n{'='*60}")
        print(f"✓ All {self.FILE_TYPE.upper()} batch IV analysis tests passed!")
        print(f"{'='*60}\n")


class TestBatchIVAnalysisABF(BatchIVAnalysisTestBase):
    FILE_TYPE = "abf"
    FILE_EXTENSION = "*.abf"


class TestBatchIVAnalysisWCP(BatchIVAnalysisTestBase):
    FILE_TYPE = "wcp"
    FILE_EXTENSION = "*.wcp"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))