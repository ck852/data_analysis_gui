# Changelog
All notable changes to this project will be documented in this file.  
This project follows [Semantic Versioning](https://semver.org/).

## [0.0.0-beta.4] - 2025-09-28
### Added
- Background subtraction
- Ramp IV

### Changed
- Removed unhelpful current units error messages

### Known Issues
- parameter settings don't all save between sessions
- MAT file support is not confirmed due to mat version issues. May deprecate

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
