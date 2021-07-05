# Changelog

## [Unreleased]
- Local graph drawing;
- QASM Convertor.

## [2.0.1] - July-05-2021
### Added
- Examples update;
  + QAPP, quantum computing toolbox.

## [2.0.0] - Jun-07-2021
### Added
- New QPU:
  + IoPCAS. 10 qubits superconducting QPU from Institute of Physics (IOP), Chinese Academy of Sciences
- New cloud simulators:
  + cloud_baidu_sim2_lake. The simulator based on GPU
- New module:
  + InverseCircuit;
- New functions for local simulator:
  + output_probability;
  + output_state;
- New debug sector for all backends:
  + Ancilla;
  + Auto workflow for modules.

## [1.1.0] - Apr-19-2021
### Added
- Type hints
- RegPool createList/toListPair;
- New cloud simulators:
  + cloud_baidu_sim2_wind
  + cloud_qpu2
### Changed
- Time format;
- Matrix must be C-contiguous;
### Break Changed
- New protobuf structs;

## [1.0.3] - Jan-15-2021
### Added
- New cloud simulators:
  + cloud_baidu_sim2_water
  + cloud_baidu_sim2_earth
  + cloud_baidu_sim2_thunder
  + cloud_baidu_sim2_heaven
- Examples update;
- Related docs update;
- Requirements update.
### Changed
- Local file/dir name;
### Deprecated
- cloud_baidu_sim2;
  + New name is cloud_baidu_sim2_water

## [1.0.2] - Sep-14-2020
### Added
- Update doc links;
- Examples update;
- Related docs update;
- Requirements update.

## [0.0.1] - Sep-02-2020
### Added
- First commit;
- QCompute SDK contains quantum toolkits, simulator, examples and docs.