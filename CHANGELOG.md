# Changelog

## [3.0.0] - June-31-2022

### Added

- New QPUs:
    + CloudBaiduQPUQian;
    + CloudIonAPM.

## [2.0.6] - June-11-2022

### Changed

- Repair some range-restriction issues;
- Update requirements.

## [2.0.5] - June-01-2022

### Added

- Support for the UBQC(Universal blind quantum computing) plugin;
- More convertors
    + Qasm Convertor;
    + XanaduSF Convertor;
    + IonQ Convertor;
- Calibration functions;
- InteractiveModule.

### Changed

- CompositeGate:
    + CK Gate;
    + MS Gate.

### Removed

- RZZ Gate.

## [2.0.4] - May-05-2022

### Added

- Support for the QEP plugin.

## [2.0.3] - Dec-20-2021

### Changed

- Repair IoPCAS backend.

## [2.0.2] - Nov-04-2021

### Added

- Local graph drawing;
- New error code assembly.

### Changed

- Remove RZZ example.

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