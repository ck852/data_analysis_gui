#!/usr/bin/env python
"""
CR_pytest - Concentration Response Dialog Test Runner

This script runs the pytest tests for the Concentration Response Dialog.

Usage:
    python CR_pytest.py              # Run tests with default settings
    python CR_pytest.py -v           # Run tests with verbose output
    python CR_pytest.py -v -s        # Run tests with verbose output and print statements
    python CR_pytest.py --tb=short   # Run tests with shorter traceback
    
Or if made executable (chmod +x CR_pytest.py on Unix/Mac):
    ./CR_pytest.py
    ./CR_pytest.py -v
    
Windows users can create a batch file CR_pytest.bat with:
    @echo off
    python "%~dp0CR_pytest.py" %*
"""

import sys
import os
from pathlib import Path
import subprocess

def main():
    """Run the concentration response dialog tests."""
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    
    # Find the test file
    test_file = script_dir / "test_concentration_response.py"
    
    # Check if test file exists
    if not test_file.exists():
        print(f"Error: Test file not found at {test_file}")
        print("\nSearching for test file in common locations...")
        
        # Try alternative locations
        possible_locations = [
            script_dir / "Data-Analysis-GUI-CR_testing" / "test_concentration_response.py",
            script_dir / "tests" / "test_concentration_response.py",
            script_dir.parent / "test_concentration_response.py",
        ]
        
        for location in possible_locations:
            if location.exists():
                test_file = location
                print(f"Found test file at: {test_file}")
                break
        else:
            print("Could not find test_concentration_response.py")
            print("\nPlease ensure the test file is in one of these locations:")
            print("  - Same directory as CR_pytest.py")
            print("  - Data-Analysis-GUI-CR_testing/test_concentration_response.py")
            print("  - tests/test_concentration_response.py")
            return 1
    
    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("Error: pytest is not installed.")
        print("\nTo install pytest, run:")
        print("  pip install pytest pytest-qt")
        return 1
    
    # Check if pytest-qt is installed
    try:
        import pytestqt
    except ImportError:
        print("Error: pytest-qt is not installed.")
        print("\nTo install pytest-qt, run:")
        print("  pip install pytest-qt")
        return 1
    
    # Prepare pytest arguments
    pytest_args = [str(test_file)]
    
    # Add any command-line arguments passed to this script
    if len(sys.argv) > 1:
        pytest_args.extend(sys.argv[1:])
    else:
        # Default to showing test names
        pytest_args.append("-v")
    
    # Add some helpful defaults if not specified
    if "--tb" not in " ".join(pytest_args):
        pytest_args.append("--tb=short")
    
    # Change to the test directory to ensure proper imports
    original_dir = os.getcwd()
    os.chdir(str(test_file.parent))
    
    print(f"Running Concentration Response Dialog Tests")
    print(f"Test file: {test_file}")
    print(f"Arguments: {' '.join(pytest_args)}")
    print("-" * 60)
    
    try:
        # Run pytest with the specified arguments
        exit_code = pytest.main(pytest_args)
    finally:
        # Restore original directory
        os.chdir(original_dir)
    
    # Print summary
    print("-" * 60)
    if exit_code == 0:
        print("✓ All tests passed successfully!")
    else:
        print(f"✗ Tests completed with exit code: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())