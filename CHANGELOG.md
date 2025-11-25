# Changelog
All notable changes to this project will be documented in this file.  
This project follows [Semantic Versioning](https://semver.org/).

## [1.0.0]
### Added
- Batch background subtraction

### Changed
- Concentration/Dose Response tool removed for initial release. Still needs styling optimizations. Not needed for majority of the program's utility (bare but functional in previous versions).

### Known Issues
- Dual Analysis range recognized as invalid if load new file with smaller time range which constrains saved cursor positions. Fixed by unchecking and re-checking the box

## [0.9.2-b.6] 2025-11-09
### Added
- Conductance measurement option
- Batch analysis summaries for non-IV analyses
- Batch analysis and current density analysis for ramp IV protocols
- Copy sweeps directly from MainWindow
- Batch analysis for conductance
- Reject sweeps dialog
- Leak Subtraction (beta)

### Changed
- Fixed mu character in plots
- Auto-open file dialog on batch analysis open
- Per-file unit retrieval in batch analysis
- Voltage units always extracted from wcp files


## [0.9.2-b.5] 2025-11-03
### Added
- "Reject Sweep" button
- Batch Sweep extraction
- Current and Voltage sweep views now independent
- Slider to adjust which sweep is displayed

### Changed
- Dual analysis cursors remove when box unchecked
- MainWindow Toolbar streamlined
- Extract Sweeps dialog saves settings, better UI
- Cursor labels sticking to MainWindow plot fixed
- MainWindow plot has thinner, more prominent lines

## [0.9.2-b.4] 2025-10-22
### Added
- "Copy Data" button for remaining dialogs except dose response
- Axis-specific zoom buttons
- "Copy File Names" button for all result dialogs except dose response
- Batch background subtraction

### Changed
- Extract Sweeps dialog receives initial time range from MainWindow
- Splitter position in main window saves between sessions
- File names in batch and current density dialogs easier to check/uncheck
- Cursor/spinboxes in MainWindow snap to nearest availalbe time point
- Maintain zoom state across sweeps
- One-click zoom
- Remove unused MainWindow toolbar buttons
- IV works for peak or average
- Remove "View Results" button from batch dialog
- Home/Reset button in MainWindow plot autoscales to present sweep

### Known Issues
- Dual Analysis cursors don't remove after unchecking box

## [0.9.2-b.3] 2025-10-09
### Added
- "Copy Data" button to analysis plot and ramp iv

### Changed
- Removed unnecessary app_controller use in analysis plot

## [0.9.2-b.2] 2025-10-08
### Changed
- Data points changed back to idealized time points
- actual time points yielded larger discrepancy versus winwcp

## [0.9.2-b.1] 2025-10-08
### Changed

- Updated golden files according to improved wcp sweep data extraction
- Data points within sweeps now correspond to actual (not idealized) time points

## [0.9.1-b.9] 2025-10-08
### Added

- Sweep Extraction module (Beta)

## [0.9.1-b.4] through [0.9.1-b.8] 2025-10-06

- re-attempt windows and mac build

## [0.9.1-b.3] 2025-10-06
### Added
- Dose-response analysis module

### Changed
- Fixed file order in batch analyses with multiple dates (requires xxxxxx_xxx file name format)
- Added x=0 and y=0 gridlines
- Centralized cursor-spinbox coupling widget
- better directory location perstistence across sessions

## [0.9.1-beta.2] - 2025-10-03
### Added
- Enhanced sweep selection UI for ramp_iv

### Changed
- overactive error detection in ramp_iv voltage tolerances corrected
- quick tab change between Cslow boxes

### Known Issues
- error messages from ramp_iv suggest wrong voltage set, but data results seem unaffected
- file directory location not saving consistently between sessions

## [0.9.1-beta.1] - 2025-10-03
### Added
- WinWCP File support
- Channel ID, units, and sweep times auto-extracted from WCP and ABF Files
- Ramp IV
- Background subtraction

### Changed
- Removed MAT file support (buggy and not needed)
- Removed user inputs for channel ID, current units, and stimulus time

### Known Issues
- Ramp IV sweep selection is tedious
- Font size not consistent across platforms
- error messages from ramp_iv suggest wrong voltage set, but data results seem unaffected
- Compatibilty issues with Sequoia 15.6.1 (Mac)

## [0.9.0-beta.3] - 2025-09-21
### Added
- Mac Support

## [0.9.0-beta.2] - 2025-09-21
### Added
- More comprehensive documentation
- Peak mode and swapped channel validation

### Changed
- Export headers updated for consistency

### Known Issues
- Plot styling has display issues
- Current unit adjustment has not been tested

## [0.9.0-beta.1] - 2025-09-19
### Added
- First beta release intended for internal testing by lab members.
- Core functionality for patch-clamp electrophysiology data analysis:
  - Single-file analysis workflow.
  - Batch analysis workflow with Current Density analysis available for IV curves.
- See README for full description

### Changed
- Updated dependency versions for PySide6, numpy, scipy, matplotlib, and pyabf.
- Polished GUI layout and parameter handling.

### Known Issues
- Some presentation elements (GUI styling, export headers) are not finalized.
- Minor tweaks to overall layout and usability expected in upcoming betas.

---
