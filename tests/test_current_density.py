"""
PatchBatch Electrophysiology Data Analysis Tool

Test script for current density analysis workflow with golden file validation.

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
from data_analysis_gui.services.current_density_service import CurrentDensityService


# Cslow values (pF) for each recording
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

    if gen_data.size > 0:
        gen_nan_mask = np.isnan(gen_data)
        gold_nan_mask = np.isnan(gold_data)

        # NaNs indicate missing data - flag them
        if np.any(gen_nan_mask):
            nan_count = np.sum(gen_nan_mask)
            nan_positions = np.where(gen_nan_mask)
            raise AssertionError(
                f"WARNING: Found {nan_count} NaN values in generated file {generated.name}\n"
                f"NaN positions (row, col): {list(zip(*nan_positions))[:5]}..."
            )

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


def compare_summary_csv(generated: Path, golden: Path) -> None:
    """Compare current density summary CSVs with appropriate tolerances."""
    gen_headers, gen_data = load_csv_data(generated)
    gold_headers, gold_data = load_csv_data(golden)

    try:
        assert len(gen_headers) == len(gold_headers), (
            f"Header count mismatch: {len(gen_headers)} vs {len(gold_headers)}"
        )
    except AssertionError as e:
        raise AssertionError(
            f"Summary header validation failed\n"
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
            f"Summary shape validation failed\n"
            f"Generated file: {generated}\n"
            f"Golden file: {golden}\n"
            f"{str(e)}"
        )

    if gen_data.size > 0:
        gen_nan_mask = np.isnan(gen_data)
        if np.any(gen_nan_mask):
            raise AssertionError(
                f"WARNING: Found {np.sum(gen_nan_mask)} NaN values in summary file {generated.name}\n"
                f"This indicates missing data or calculation errors"
            )

        # Voltage column - tight tolerance
        try:
            np.testing.assert_allclose(
                gen_data[:, 0], gold_data[:, 0],
                rtol=1e-4, atol=0.1,
                err_msg="Voltage column mismatch",
            )
        except AssertionError as e:
            raise AssertionError(
                f"Summary voltage column validation failed\n"
                f"Generated file: {generated}\n"
                f"Golden file: {golden}\n"
                f"{str(e)}"
            )

        # Current density columns
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
                    f"Summary column {col_idx} NaN validation failed\n"
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
                        rtol=1e-4, atol=1e-3,
                        err_msg=f"Current density mismatch in column {col_idx} ({gen_headers[col_idx]})",
                    )
                except AssertionError as e:
                    diff = np.abs(col_gen[valid_mask] - col_gold[valid_mask])
                    raise AssertionError(
                        f"Summary column {col_idx} validation failed\n"
                        f"Column header: {gen_headers[col_idx]}\n"
                        f"Max difference: {np.max(diff):.6e} pA/pF\n"
                        f"Generated file: {generated}\n"
                        f"Golden file: {golden}\n"
                        f"{str(e)}"
                    )


class CurrentDensityTestBase:
    """Base class for current density tests. Subclasses set FILE_TYPE and FILE_EXTENSION."""

    FILE_TYPE = None
    FILE_EXTENSION = None

    @property
    def sample_data_dir(self) -> Path:
        return Path(f"tests/fixtures/sample_data/IV+CD/{self.FILE_TYPE}")

    @property
    def golden_data_dir(self) -> Path:
        return Path(f"tests/fixtures/golden_data/golden_CD/{self.FILE_TYPE}")

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

    def test_current_density_workflow(self, analysis_params, temp_output_dir):
        """Full current density workflow: batch analysis, CD calculation, export, and golden file validation."""

        # Initialize services
        batch_processor = BatchProcessor()
        data_manager = DataManager()
        cd_service = CurrentDensityService()

        # Load test files
        test_files = self.get_test_files()
        assert len(test_files) == 12, f"Expected 12 {self.FILE_TYPE.upper()} files, found {len(test_files)}"

        print(f"\n{'='*60}")
        print(f"Testing {self.FILE_TYPE.upper()} Current Density Workflow")
        print(f"{'='*60}")
        print(f"Processing {len(test_files)} files...")

        # Run batch analysis
        batch_result = batch_processor.process_files(file_paths=test_files, params=analysis_params)

        assert len(batch_result.successful_results) == 12, (
            f"Expected 12 successful results, got {len(batch_result.successful_results)}"
        )
        assert len(batch_result.failed_results) == 0, (
            f"Unexpected failures: {[r.file_path for r in batch_result.failed_results]}"
        )

        # Verify units are pA before CD calculation
        print("Validating intermediate state (pre-CD)...")
        for result in batch_result.successful_results:
            if result.export_table and "headers" in result.export_table:
                headers_str = str(result.export_table["headers"])
                assert "(pA)" in headers_str or "Current" in headers_str, (
                    f"Expected current units in headers for {result.base_name}"
                )
                assert "(pA/pF)" not in headers_str, (
                    f"Found pA/pF before CD calculation for {result.base_name}"
                )

        # Apply current density calculations
        print("Applying current density calculations...")

        original_batch_result = batch_result
        active_batch_result = replace(
            batch_result,
            successful_results=list(batch_result.successful_results),
        )

        for i, result in enumerate(active_batch_result.successful_results):
            file_name = result.base_name
            cslow = CSLOW_VALUES.get(file_name)
            assert cslow is not None and cslow > 0, f"Invalid Cslow value for {file_name}"

            updated_result = cd_service.recalculate_cd_for_file(
                file_name, cslow, active_batch_result, original_batch_result
            )

            # Verify header changed to pA/pF
            if updated_result.export_table and "headers" in updated_result.export_table:
                headers_str = str(updated_result.export_table["headers"])
                assert "(pA/pF)" in headers_str, (
                    f"Expected (pA/pF) in headers after CD calculation for {file_name}"
                )

            active_batch_result.successful_results[i] = updated_result

        # Verify all results converted
        print("Validating intermediate state (post-CD)...")
        for result in active_batch_result.successful_results:
            if result.export_table and "headers" in result.export_table:
                headers_str = str(result.export_table["headers"])
                assert "(pA/pF)" in headers_str, (
                    f"Expected pA/pF units after CD for {result.base_name}"
                )

        # Export individual CSVs with _CD suffix
        print("Exporting individual current density CSVs...")

        cd_results = []
        for result in active_batch_result.successful_results:
            cd_result = replace(result, base_name=f"{result.base_name}_CD")
            cd_results.append(cd_result)

        cd_batch_result = replace(
            active_batch_result,
            successful_results=cd_results,
            selected_files={r.base_name for r in cd_results},
        )

        cd_output_dir = os.path.join(temp_output_dir, "current_density")
        os.makedirs(cd_output_dir, exist_ok=True)

        export_result = batch_processor.export_results(cd_batch_result, cd_output_dir)
        assert export_result.success_count == 12, (
            f"Expected 12 successful exports, got {export_result.success_count}"
        )

        # Generate and export summary
        print("Generating current density summary...")

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

        selected_files = {r.base_name for r in sorted_results}
        summary_data = cd_service.prepare_summary_export(
            voltage_data, file_mapping, CSLOW_VALUES, selected_files, "pA/pF"
        )

        summary_path = os.path.join(temp_output_dir, "Current_Density_Summary.csv")
        summary_result = data_manager.export_to_csv(summary_data, summary_path)
        assert summary_result.success, f"Summary export failed: {summary_result.error_message}"

        # Validate against golden files
        print("\nValidating against golden reference files...")

        for file_name in CSLOW_VALUES.keys():
            generated_csv = Path(cd_output_dir) / f"{file_name}_CD.csv"
            golden_csv = self.golden_data_dir / f"{file_name}_CD.csv"

            print(f"  Comparing {file_name}_CD.csv...", end=" ")
            try:
                compare_csv_files(generated_csv, golden_csv, rtol=1e-4, atol=1e-3)
                print("✓")
            except AssertionError as e:
                print("✗")
                raise AssertionError(
                    f"\nValidation failed for: {file_name}_CD.csv\n"
                    f"Cslow value used: {CSLOW_VALUES[file_name]} pF\n"
                    f"{str(e)}"
                )

        # Validate summary
        print("  Comparing Current_Density_Summary.csv...", end=" ")
        golden_summary = self.golden_data_dir / "Current_Density_Summary.csv"

        try:
            compare_summary_csv(Path(summary_path), golden_summary)
            print("✓")
        except AssertionError as e:
            print("✗")
            raise AssertionError(f"\nValidation failed for summary file\n{str(e)}")

        print(f"\n{'='*60}")
        print(f"✓ All {self.FILE_TYPE.upper()} current density tests passed!")
        print(f"{'='*60}\n")


class TestCurrentDensityABF(CurrentDensityTestBase):
    FILE_TYPE = "abf"
    FILE_EXTENSION = "*.abf"


class TestBatchIVAnalysisWCP(CurrentDensityTestBase):
    FILE_TYPE = "wcp"
    FILE_EXTENSION = "*.wcp"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))