## PatchBatch Analysis Architecture

The architecture follows a layered design where each component has a single responsibility. Data flows down through the layers during analysis and back up for display.

### Layer 1: GUI (MainWindow, Dialogs)

MainWindow is the entry point for user interaction. It owns a ControlPanel where users set analysis parameters (time ranges, measure types, peak modes) and a PlotManager that displays raw sweep data. When the user clicks "Generate Analysis Plot" or triggers batch analysis, MainWindow collects the current parameter state and passes it to the ApplicationController.

MainWindow doesn't perform any calculations. It assembles an `AnalysisParameters` object from the UI state and delegates everything else. This keeps the GUI code testable and prevents analysis logic from leaking into widget code.

### Layer 2: ApplicationController

ApplicationController is the coordination layer between GUI and backend. It maintains the application state—specifically `current_dataset` (the loaded file's data) and `loaded_file_path`. When MainWindow requests an analysis, ApplicationController validates that data is loaded, then forwards the request to AnalysisManager.

ApplicationController also holds references to the three main services: DataManager (file I/O), AnalysisManager (analysis orchestration), and BatchProcessor (multi-file operations). It doesn't do analysis itself; it routes requests to the appropriate service and wraps responses in result objects that include success/failure status and error messages.

For batch operations, ApplicationController delegates to BatchProcessor, which internally creates its own DataManager and AnalysisManager instances to process each file in isolation.

### Layer 3: AnalysisManager

AnalysisManager translates high-level analysis requests into AnalysisEngine operations. It owns an AnalysisEngine instance (created via `create_analysis_engine()`) and a DataManager for any file operations needed during analysis.

When `analyze()` is called, AnalysisManager passes the dataset and parameters to AnalysisEngine, receives back a list of `SweepMetrics`, and wraps them in an `AnalysisResult` for return to the controller. It also provides convenience methods like `get_export_table()` and `get_sweep_plot_data()` that call through to the engine with appropriate formatting.

### Layer 4: AnalysisEngine

AnalysisEngine is the orchestrator of the actual computation. It wires together three components: DataExtractor (gets raw arrays from the dataset), MetricsCalculator (computes numerical metrics), and PlotFormatter (transforms metrics into plot-ready or export-ready structures).

The core method is `_compute_all_metrics()`. It iterates through each sweep in the dataset, calls DataExtractor to get the time/voltage/current arrays, retrieves the sweep's timestamp from metadata, and passes everything to MetricsCalculator. Failed sweeps are logged but don't halt processing—the method collects as many valid results as possible.

AnalysisEngine also handles rejected sweeps. If the user has marked certain sweeps for exclusion, those indices are passed down and skipped during iteration.

### Layer 5: DataExtractor

DataExtractor bridges the gap between the generic `ElectrophysiologyDataset` structure (which stores numbered channels) and the semantic meaning needed for analysis (which channel is voltage, which is current).

When `extract_sweep_data()` is called, DataExtractor reads the `channel_config` from dataset metadata (set by the file loader), maps channel indices to roles (voltage_channel, current_channel), and returns a dictionary with `time_ms`, `voltage`, and `current` arrays. This abstraction lets the rest of the pipeline work with named channels rather than numeric indices.

DataExtractor validates that the requested sweep exists and that the data arrays don't contain NaN in critical fields (time). It warns but doesn't fail on NaN in voltage/current, since some experimental data has legitimate gaps.

### Layer 6: MetricsCalculator

MetricsCalculator is a stateless class containing the actual numerical calculations. The main method is `compute_sweep_metrics()`, which takes time, voltage, and current arrays plus the analysis ranges, and returns a `SweepMetrics` dataclass.

For each range, it applies a time mask to extract the relevant portion of the data, then computes: mean (average), absolute peak (largest magnitude regardless of sign), positive peak (maximum), negative peak (minimum), and peak-to-peak (max minus min). These are computed for both voltage and current channels.

If dual range is enabled, the same calculations run on range2, populating the `_r2` fields of SweepMetrics. The calculator doesn't decide which metrics to use—it computes everything and lets PlotFormatter select the appropriate values later.

### Layer 7: PlotFormatter

PlotFormatter transforms raw `SweepMetrics` lists into structures suitable for plotting or export. It's where user selections (Average vs Peak, Voltage vs Current, which peak type) get applied.

When `format_for_plot()` is called, PlotFormatter reads the `AxisConfig` from parameters to determine what the user wants on each axis. If Y-axis is "Average Current", it extracts `current_mean_r1` from each SweepMetrics. If it's "Peak Current" with peak_type "Negative", it extracts `current_negative_r1`. The formatter builds parallel arrays of x and y values for plotting.

PlotFormatter also handles derived quantities. If the user selects Conductance, it calls the conductance calculation service to compute G = I / (V - Vrev) with appropriate unit conversions. For dual range, it produces separate y_data and y_data2 arrays.

For export, `format_for_export()` adds headers with appropriate units and structures the data for CSV output.

### Layer 8: ElectrophysiologyDataset

At the bottom is the data container itself. `ElectrophysiologyDataset` stores sweeps as a dictionary mapping sweep indices to (time_ms, data_matrix) tuples. The data_matrix is 2D: samples × channels.

The dataset is populated by file loaders (WCP loader, ABF loader via pyABF). Loaders parse the file format, extract calibrated data, and populate both the sweep data and metadata. Critical metadata includes `channel_config` (which channel is voltage/current), `sweep_times` (timestamp for each sweep), and format-specific info.

The dataset provides low-level access methods: `get_sweep()` returns raw arrays, `get_channel_vector()` extracts a single channel. Higher layers use DataExtractor rather than calling these directly.

---

### Data Flow Summary

For a single-file analysis:

```
User clicks "Generate Analysis Plot"
    ↓
MainWindow.collect_parameters() → AnalysisParameters
    ↓
ApplicationController.perform_analysis(params)
    ↓
AnalysisManager.analyze(dataset, params)
    ↓
AnalysisEngine._compute_all_metrics(dataset, params)
    ↓ (for each sweep)
    DataExtractor.extract_sweep_data() → {time_ms, voltage, current}
    MetricsCalculator.compute_sweep_metrics() → SweepMetrics
    ↓
PlotFormatter.format_for_plot(metrics, params) → {x_data, y_data, labels}
    ↓
Returns up through each layer
    ↓
MainWindow displays in AnalysisPlotDialog
```

For batch analysis, BatchProcessor wraps this flow, iterating over files and collecting results into a `BatchAnalysisResult`.