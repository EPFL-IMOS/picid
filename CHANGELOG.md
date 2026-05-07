# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Initial public release of PICID (piCID) framework
- Unified config-driven pipeline for PHM and forecasting
- Support for multiple datasources, transforms, and model types
- Documentation and getting-started guides

### Changed

- **Breaking:** Renamed `EntryInterface` to `PicidExperiment`. Replace
  `from picid.interface import EntryInterface` with
  `from picid.interface import PicidExperiment`.

## [0.1.0] - 2025-02-23

### Added
- First release of PICID benchmark arena
