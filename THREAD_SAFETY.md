# Thread Safety Documentation
## Data Analysis GUI - Phase 3 Refactoring

### Overview
This document provides comprehensive thread safety guarantees for all core components after Phase 3 refactoring. Components are classified into three categories:

1. **THREAD-SAFE**: Can be used concurrently without any synchronization
2. **CONDITIONALLY SAFE**: Safe under specific conditions (documented per component)  
3. **NOT THREAD-SAFE**: Requires external synchronization for concurrent access

---

## Core Components

### AnalysisPlotter (`core/analysis_plot.py`)
**Classification: THREAD-SAFE**

All methods are static pure functions with no shared state.

**Guarantees:**
- All methods can be called concurrently from multiple threads
- No instance variables or shared mutable state
- Thread-local Figure creation when using 'Agg' backend
- File I/O operations may require directory-level synchronization

**Methods:**
- `create_figure()` - Safe, creates new Figure each call
- `save_figure()` - Safe for different files, may need sync for same directory
- `create_and_save_plot()` - Combines above, same guarantees

**Usage Example:**
```python
from concurrent.futures import ThreadPoolExecutor
import threading

# Safe concurrent plotting
def process_file(data, filename):
    fig, ax = AnalysisPlotter.create_figure(data, "X", "Y", "Title")
    AnalysisPlotter.save_figure(fig, filename)

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(process_file, data, f"plot_{i}.png") 
               for i, data in enumerate(datasets)]
```

---

### AnalysisEngine (`core/analysis_engine.py`)
**Classification: CONDITIONALLY SAFE**

Safe for concurrent reads, requires synchronization for writes.

**Guarantees:**
- Read operations (`get_*` methods) are thread-safe
- Cache modifications protected by internal RLock
- Content-based cache keys prevent memory address reuse bugs
- Dataset should not be modified during analysis

**Thread-Safe Methods:**
- `get_sweep_series()` - Safe for concurrent reads
- `get_all_metrics()` - Safe for concurrent reads
- `get_plot_data()` - Safe for concurrent reads
- `get_export_table()` - Safe for concurrent reads
- `get_peak_analysis_data()` - Safe for concurrent reads

**Methods Requiring Synchronization:**
- `set_channel_definitions()` - Modifies state, needs external lock
- `clear_caches()` - Modifies state, needs external lock

**Cache Implementation:**
```python
# Internal implementation uses content-based keys
def _get_dataset_key(self, dataset):
    # Uses file path + modification time instead of id(dataset)
    # Prevents cache corruption from memory address reuse
    file_path = Path(dataset.source_path)
    mtime = os.path.getmtime(file_path)
    return hashlib.md5(f"{file_path}:{mtime}".encode()).hexdigest()
```

---

### Data Processing Utilities (`core/analysis_engine.py`)
**Classification: THREAD-SAFE**

All utility functions are pure with no side effects.

**Functions:**
- `process_sweep_data()` - Pure function, thread-safe
- `calculate_peak()` - Pure function, thread-safe
- `calculate_average()` - Pure function, thread-safe
- `apply_analysis_mode()` - Pure function, thread-safe
- `calculate_current_density()` - Pure function, thread-safe
- `calculate_sem()` - Pure function, thread-safe
- `format_voltage_label()` - Pure function, thread-safe

---

### Matplotlib Backend Configuration
**Classification: CONDITIONALLY SAFE**

Thread safety depends on backend selection.

**Backend Guidelines:**
```python
import matplotlib
matplotlib.use('Agg')  # Thread-safe, no GUI

# Alternative backends:
# 'Agg' - SAFE: No GUI, pure image generation
# 'PDF' - SAFE: File output only
# 'SVG' - SAFE: File output only  
# 'Qt5Agg' - NOT SAFE: GUI event loop required
# 'TkAgg' - NOT SAFE: GUI event loop required
```

**Best Practices:**
1. Use 'Agg' backend for batch processing
2. Set backend before any imports of pyplot
3. GUI backends must run on main thread only

---

## Parallel Processing Patterns

### Safe Pattern: Parallel Analysis
```python
from concurrent.futures import ThreadPoolExecutor
from data_analysis_gui.core.analysis_engine import AnalysisEngine
from data_analysis_gui.core.analysis_plot import AnalysisPlotter

def analyze_and_plot(dataset, params, output_file):
    """Thread-safe analysis and plotting."""
    # Engine can be shared (read-only operations)
    plot_data = engine.get_plot_data(dataset, params)
    
    # Create plot using static methods
    plot_obj = AnalysisPlotData.from_dict(plot_data)
    fig = AnalysisPlotter.create_and_save_plot(
        plot_obj, 
        plot_data['x_label'],
        plot_data['y_label'], 
        "Analysis",
        output_file
    )
    return output_file

# Safe to run in parallel
engine = AnalysisEngine()  # Shared instance OK for reads
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = []
    for dataset, params in work_items:
        future = executor.submit(
            analyze_and_plot, 
            dataset, 
            params, 
            f"output_{dataset.name}.png"
        )
        futures.append(future)
```

### Unsafe Pattern: Shared State Modification
```python
# DON'T DO THIS - Race condition!
def unsafe_analysis(dataset, params):
    engine.clear_caches()  # NOT THREAD-SAFE!
    engine.set_channel_definitions(new_defs)  # NOT THREAD-SAFE!
    return engine.get_plot_data(dataset, params)
```

### Safe Pattern: Synchronized Modifications
```python
import threading

engine_lock = threading.Lock()

def safe_reconfigure(new_channel_defs):
    """Thread-safe reconfiguration."""
    with engine_lock:
        engine.set_channel_definitions(new_channel_defs)
        engine.clear_caches()
```

---

## File I/O Considerations

### Directory-Level Synchronization
When multiple threads write to the same directory:

```python
import threading
from pathlib import Path

# Directory locks for concurrent file writes
dir_locks = {}
lock_creation_lock = threading.Lock()

def get_dir_lock(directory):
    """Get or create a lock for a directory."""
    with lock_creation_lock:
        if directory not in dir_locks:
            dir_locks[directory] = threading.Lock()
        return dir_locks[directory]

def safe_save_to_directory(figure, filepath):
    """Thread-safe save with directory locking."""
    directory = Path(filepath).parent
    dir_lock = get_dir_lock(directory)
    
    with dir_lock:
        AnalysisPlotter.save_figure(figure, filepath)
```

---

## Testing Thread Safety

### Recommended Test Pattern
```python
import unittest
import threading
import time
from concurrent.futures import ThreadPoolExecutor

class TestThreadSafety(unittest.TestCase):
    def test_concurrent_analysis(self):
        """Test concurrent analysis operations."""
        engine = AnalysisEngine()
        results = []
        errors = []
        
        def analyze(dataset, params):
            try:
                result = engine.get_plot_data(dataset, params)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent analyses
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(100):
                future = executor.submit(analyze, test_dataset, test_params)
                futures.append(future)
            
            # Wait for completion
            for future in futures:
                future.result()
        
        # Verify no errors and consistent results
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 100)
        
        # All results should be identical (same input = same output)
        first_result = results[0]
        for result in results[1:]:
            np.testing.assert_array_equal(result['x_data'], first_result['x_data'])
            np.testing.assert_array_equal(result['y_data'], first_result['y_data'])
```

---

## Migration Guide for Batch Processing

When reimplementing batch processing in a future phase:

1. **Use ThreadPoolExecutor for I/O-bound operations**
   - File loading
   - Plot saving
   - CSV export

2. **Use ProcessPoolExecutor for CPU-bound operations**
   - Heavy numerical computations
   - Large dataset processing

3. **Ensure proper backend configuration**
   ```python
   import matplotlib
   matplotlib.use('Agg')  # Set before any plotting imports
   ```

4. **Use content-based cache keys**
   - Already implemented in AnalysisEngine
   - Safe for concurrent access

5. **Avoid shared mutable state**
   - Use immutable data structures where possible
   - Return new objects instead of modifying in-place

---

## Summary

Phase 3 refactoring has established a solid foundation for parallel processing:

1. **AnalysisPlotter** is now completely stateless and thread-safe
2. **AnalysisEngine** uses content-based cache keys, preventing memory address bugs
3. All data processing utilities are pure functions
4. Thread safety is explicitly documented and testable
5. Clear patterns established for safe concurrent operations

The architecture now supports trivial parallelization - batch processing can be implemented by simply changing sequential loops to parallel map operations, with no modifications to core components required.