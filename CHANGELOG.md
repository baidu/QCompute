# Changelog

## [3.3.5] - Oct-01-2023

### Added

- Commit QASM Circuit directly for execution with an option parameter.

### Changed

- Modify the order of objects within the returned tuple of the controlProcedure function.

## [3.3.4] - Aug-08-2023

### Removed

- Simulator;
  - The quantum simulators CloudBaiduSim2Thunder (single instance C++ version) and CloudBaiduSim2Lake (single instance GPU version) are deprecated from SDK this version [3.3.4]. Any task to these two cloud simulators would be redirected to CloudBaiduSim2Water.

## [3.3.3] - Apr-19-2023

### Added

- New Simulator;
    - LocalBaiduSimPhotonic, used for simulating photonic quantum circuit based on gaussian or fock state.

## [3.3.0] - Feb-16-2023

### Added

- New Simulator;
    - LocalCuQuantum, the brand-new local quantum simulator is constructed with the NVIDIA cuQuantum SDK (https://developer.nvidia.com/cuquantum-sdk).

## [3.2.1] - Dec-30-2022

### Added

- Noise;
    - Add parallelization method for noisy simulator.

### Changed

- Combine local ideal simulator and noisy simulator.

## [3.2.0] - Dec-05-2022

### Added

- Noise;
    - Support for noisy simulation, including simulator, noise models and examples.

## [3.1.0] - Nov-18-2022

### Added

- BatchID;
    - A BatchID would be generated for every QComputeSDK-Python process. All related tasks belonging to the process are
      precisely the same. The BatchID can be used for variational algorithms and task group sets.

### Changed

- Slack the version requirements of matplotlib, tqdm.

## [3.0.2] - Oct-30-2022

### Added

- API for QPU and simulator status monitor.

### Changed

- Modify python/numpy/scipy version requirements.

## [3.0.1] - Sept-02-2022

### Changed

- Disable mapping and enable unrolling when calling quantum hardware in the QuantumErrorProcessing extension.

## [3.0.0] - June-31-2022

### Added

- New QPUs:
    - CloudBaiduQPUQian;
    - CloudIonAPM.

## [2.0.6] - June-11-2022

### Changed

- Repair some range-restriction issues;
- Update requirements.

## [2.0.5] - June-01-2022

### Added

- Support for the UBQC(Universal blind quantum computing) plugin;
- More convertors
    - Qasm Convertor;
    - XanaduSF Convertor;
    - IonQ Convertor;
- Calibration functions;
- InteractiveModule.

### Changed

- CompositeGate:
    - CK Gate;
    - MS Gate.

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
    - QAPP, quantum computing toolbox.

## [2.0.0] - Jun-07-2021

### Added

- New QPU:
    - IoPCAS. 10 qubits superconducting QPU from Institute of Physics (IOP), Chinese Academy of Sciences
- New cloud simulators:
    - cloud_baidu_sim2_lake. The simulator based on GPU
- New module:
    - InverseCircuit;
- New functions for local simulator:
    - output_probability;
    - output_state;
- New debug sector for all backends:
    - Ancilla;
    - Auto workflow for modules.

## [1.1.0] - Apr-19-2021

### Added

- Type hints
- RegPool createList/toListPair;
- New cloud simulators:
    - cloud_baidu_sim2_wind
    - cloud_qpu2

### Changed

- Time format;
- Matrix must be C-contiguous;

### Break Changed

- New protobuf structs;

## [1.0.3] - Jan-15-2021

### Added

- New cloud simulators:
    - cloud_baidu_sim2_water
    - cloud_baidu_sim2_earth
    - cloud_baidu_sim2_thunder
    - cloud_baidu_sim2_heaven
- Examples update;
- Related docs update;
- Requirements update.

### Changed

- Local file/dir name;

### Deprecated

- cloud_baidu_sim2;
    - New name is cloud_baidu_sim2_water

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
