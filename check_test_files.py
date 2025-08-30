# check_test_files.py
from pathlib import Path
import glob

# Navigate to the test data directory
test_dir = Path(__file__).parent
fixtures_dir = test_dir / "tests" / "fixtures"
sample_data_dir = fixtures_dir / "sample_data"
iv_cd_dir = sample_data_dir / "IV+CD"

print(f"Looking in: {iv_cd_dir}")
print(f"Directory exists: {iv_cd_dir.exists()}")

if iv_cd_dir.exists():
    # List all .mat files
    mat_files = list(iv_cd_dir.glob("*.mat"))
    print(f"\nFound {len(mat_files)} .mat files:")
    
    # Group by base name
    from collections import defaultdict
    file_groups = defaultdict(list)
    
    for f in sorted(mat_files):
        name = f.name
        # Extract base name (before bracket or extension)
        if '[' in name:
            base = name.split('[')[0]
        else:
            base = name.split('.')[0]
        file_groups[base].append(name)
    
    for base, files in sorted(file_groups.items()):
        print(f"\n{base}: {len(files)} files")
        for f in files[:3]:  # Show first 3
            print(f"  - {f}")
        if len(files) > 3:
            print(f"  ... and {len(files)-3} more")
else:
    print("Directory doesn't exist!")