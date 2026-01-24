## Design Philosophy

The program is designed to be intuitive to electrophysiologists. In the program's core data analysis pipeline, a raw data file is loaded, an analysis time range per-sweep is set, analysis parameters are set (average, peak, or conductance measurement, Current or Voltage or Time channel, axes assignments), and the desired calculations are performed and returned. The main feature is the ability to apply those same parameters to a larger set (batch) of raw data files and perform analyses on all of them in one action, with the ultimate goal of expediting the conversion of raw data into presentation-ready, interpretable data. Some quality of life features are included, such as the "Dual Analysis" mode which enables users to define two time ranges to analyze from raw data files in one analysis.

Early in development, it became clear that a stateless data processing architecture was the clear logical choice to enable batch file analysis while minimizing codebase complexity and maintaining data integrity and thus the reliability of the software's outputs. Contrarily, the graphical user interface (GUI) logic uses stateful frameworks as the software is explicitly intended to apply the exact same analysis parameters, as set in the GUI, to all files selected for batch analysis. These stateless and stateful characteristics are essential for the program's core functionality. A stateless GUI parameter-gathering framework could be used for more advanced applications (if users wanted to define different parameters for different files for analysis in a batch), but need such applications are not presently needed and would require greater architectural complexity. Likewise, parallel processing of files was considered but ultimately deemed unnecessary due to the relatively small file size and quantity in most/all expected use cases and the higher degree of coding sophistication required for a feature that is simply not needed for an efficient final product. Analysis of very large datasets (i.e. 100+ raw data files that each contain several minutes of recordings) would likely benefit from parallel processing, but such a large application is not anticipated for most users.

Most apparent during development was the strength of a well-designed layered service architecture. It became clear that regardless of stateless or stateful logic, parallel processing, etc., the foundation on which the features are built had the largest influence on the adaptability and functionality of the codebase and the ease of modifications and enhancements. The layered service architecture emerged as a result of successive refinements to this foundation. For example, separation of data processing concerns from GUI functions was fundamental for expanding and enhancing user interface (UI) and user experience (UX) without needing to rebuild core analysis functions. For this reason, the codebase is organized into specific subdirectories to separate basic configuration logic (visual presentation as well as debug tools), core numerical processing, dialog windows, GUI-specific services, dialog-specific data processing services, and GUI widgets. Likewise, the architecture is designed such that numerical processing scripts, which are organized into /core and /services, are disallowed from importing any PySide6 or other GUI dependencies as a design principle. This separation facilitates maintenance of the GUI without touching the underlying data processing logic, and vice versa. Some mixing of concerns is permitted in ancillary features (such as sweep extraction or leak subtraction) as long as they function independently of the core data analysis pipeline. 

Most importantly as a foundational structure of this program, I decided that each data file, no matter its format nor any experimental factors, should be converted into a single standard format in terms that the rest of the codebase would be built to accept. This is the ElectrophysiologyDataset object. My work, and that of most other electrophysiologists in my discipline, exclusively involves repeating data "sweeps" or "traces" that record two electrophysiology channels (one Voltage and one Current) and two time components: the sampled data points in each sweep which occur on a millisecond (ms) scale, and the time relative to the first sweep at which each sweep initiates (typically on a second (s) scale). All of these data, as well as Voltage and Current units of measurement, are contained in all valid input files regardless of file format. Converting these values to a single format canonical for this program greatly simplifies all downstream data operations and must be incorporated into any new file formats loaders added in the future. The ElectrophysiologyDataset object is a key component of the core analysis pipeline architecture, which is separated into multiple service layers. 

## PatchBatch Analysis Architecture

The architecture follows a layered design where each component has a single responsibility. Data flows down through the layers during analysis and back up for display.

### Layer 1: GUI (MainWindow, Dialogs)

MainWindow (main_window.py) is the entry point for user interaction. It owns a ControlPanel (widgets/control_panel.py) where users set analysis parameters (time ranges, measure types, peak modes) and a PlotManager (plot_manager.py) that displays raw sweep data. When the user clicks "Generate Analysis Plot" or triggers a batch analysis, MainWindow assembles an `AnalysisParameters` (core/params.py) object from the parameters set in the present UI state and passes it to the ApplicationController (core/app_controller.py).

### Layer 2: ApplicationController

ApplicationController is the coordination layer between GUI and backend. It maintains the application state; specifically `current_dataset` (the loaded file's data) and `loaded_file_path`. When MainWindow requests an analysis, ApplicationController validates that data is loaded, then forwards the request to AnalysisManager (services/analysis_manager.py).

ApplicationController also holds references to the three main services: DataManager (services/data_manager.py) for file I/O, AnalysisManager for analysis orchestration, and BatchProcessor (services/batch_processor.py) for multi-file operations. It doesn't do analysis itself; it routes requests to the appropriate service and wraps responses in result objects that include success/failure status and error messages.

For batch operations, ApplicationController delegates to BatchProcessor, which internally creates its own DataManager instances to process each file in isolation.

### Layer 3: AnalysisManager

AnalysisManager translates high-level analysis requests into AnalysisEngine (core/analysis_engine.py) operations. It owns an AnalysisEngine instance (created via `create_analysis_engine()`) and a DataManager for any file operations needed during analysis.

When `analyze()` is called, AnalysisManager passes the dataset and parameters to AnalysisEngine, receives back a list of `SweepMetrics`, and wraps them in an `AnalysisResult` for return to the controller. It also provides convenience methods like `get_export_table()` and `get_sweep_plot_data()` that call through to the engine with appropriate formatting.

### Layer 4: AnalysisEngine

AnalysisEngine is the orchestrator of the actual computation. It wires together three components: DataExtractor (core/data_extractor.py), which gets raw arrays from the dataset, MetricsCalculator (core/metrics_calculator.py), which performs the numerical data processing operations on the raw data in the presently loaded file, and PlotFormatter (core/plot_formatter.py), which transforms metrics into plot-ready and/or export-ready data structures.

The central method is `_compute_all_metrics()`. It iterates through each sweep in the dataset, calls DataExtractor to get the time/voltage/current arrays, retrieves the sweep's timestamp from metadata, and passes everything to MetricsCalculator. Failed sweeps are logged but don't halt processing to collect as many valid results as possible, as incomplete sweeps can be typical, especially if a recording is stopped mid-sweep.

AnalysisEngine also handles rejected sweeps. If the user has marked certain sweeps for exclusion, those indices are passed down and skipped during iteration.

### Layer 5: DataExtractor

DataExtractor bridges the gap between the generic `ElectrophysiologyDataset` structure (which stores numbered channels) and the semantic meaning needed for analysis (which channel is voltage, which is current).

When `extract_sweep_data()` is called, DataExtractor reads the `channel_config` from dataset metadata as set by the file loader (core/loaders/wcp_loader.py|abf_loader.py), maps channel indices to canonical roles (voltage_channel, current_channel), and returns a dictionary with `time_ms`, `voltage`, and `current` arrays. This abstraction lets the rest of the pipeline work with canonically named channels rather than arbitrary numeric indices that can vary by experimental setup.

DataExtractor validates that the requested sweep exists and that the data arrays don't contain NaN in critical fields (time). It warns but doesn't fail on NaN in voltage/current, in case some experimental data has legitimate gaps.

### Layer 6: MetricsCalculator

MetricsCalculator is a stateless class containing the actual numerical calculations. The main method is `compute_sweep_metrics()`, which takes time, voltage, and current arrays plus the analysis ranges, and returns a `SweepMetrics` dataclass.

For each range, it applies a time mask to extract the relevant portion of the data, then computes: mean (average), absolute peak (largest magnitude regardless of sign), positive peak (maximum), negative peak (minimum), and peak-to-peak (max minus min). These are computed for both voltage and current channels.

If dual range is enabled, the same calculations run on range2, populating the `_r2` fields of SweepMetrics. The calculator doesn't decide which metrics to use—it computes everything and lets PlotFormatter select the appropriate values later.

### Layer 7: PlotFormatter

PlotFormatter transforms raw `SweepMetrics` lists into structures suitable for plotting or export. It's where user selections (Average vs Peak, Voltage vs Current, which peak type) get applied.

When `format_for_plot()` is called, PlotFormatter reads the `AxisConfig` from parameters to determine what the user wants on each axis. If Y-axis is "Average Current", it extracts `current_mean_r1` from each SweepMetrics. If it's "Peak Current" with peak_type "Negative", it extracts `current_negative_r1`. The formatter builds parallel arrays of x and y values for plotting.

PlotFormatter also handles derived quantities. For example, if the user selects Conductance as an analysis parameter, PlotFormatter calls the conductance calculation service (services/conductance_calculator.py) to compute G = I / (V - Vrev) with appropriate unit conversions. For dual range, it produces separate y_data and y_data2 arrays.

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