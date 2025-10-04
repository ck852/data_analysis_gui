# Changelog
All notable changes to this project will be documented in this file.  
This project follows [Semantic Versioning](https://semver.org/).

## [0.9.1-b.3]

### Changed
- Fixed file order in batch analyses with multiple dates (requires xxxxxx_xxx file name format)
- Added x=0 and y=0 gridlines
- Centralized cursor-spinbox coupling widget

## [0.9.1-beta.2] - 2025-10-03
### Added
- Enhanced sweep selection UI for ramp_iv

### Changed
- overactive error detection in ramp_iv voltage tolerances corrected
- quick tab change between Cslow boxes

### Known Issues
- error messages from ramp_iv suggest wrong voltage set, but data results seem unaffected

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
