# QCompute
![](https://release-data.cdn.bcebos.com/github-qleaf%2F%E9%87%8F%E6%98%93%E4%BC%8F%E5%9B%BE%E6%A0%87.png)

[![](https://img.shields.io/badge/license-Apache%202.0-green)](./LICENSE) [![](https://img.shields.io/badge/build-passing-green)]() ![](https://img.shields.io/badge/Python-3.6--3.9-blue) ![](https://img.shields.io/badge/release-v2.0.3-blue)

Quantum Leaf (量易伏) is a Cloud-Native quantum computing platform developed by the Institute for Quantum Computing, Baidu. It is used for programming, simulating and executing quantum computers, aiming at providing the quantum programming environment for Quantum infrastructure as a Service (QaaS). 

QCompute is a Python-based open-source SDK. It provides a full-stack programming experience for advanced users via the features of hybrid quantum programming language  and a high-performance simulator. Users can use the already-built objects and modules of quantum programming environment, pass parameters to build and execute the quantum circuits on the local simulator or the cloud simulator/hardware.

QCompute provides services for creating and analyzing quantum circuits, and calling quantum backend. The architecture of Quantum Leaf including QCompute is shown in the figure below.

**In particular, cloud service requires login at [Quantum-hub](https://quantum-hub.baidu.com). The token, large-scale results and more information can be found from the website.**
![](https://release-data.cdn.bcebos.com/architecture_edit_en.png)


## Getting Started
### Use one-step live setup

    pip install qcompute

### Or use local setup 

    pip install -e .

Then, config the python interpreter to execute examples.


Please prepare Python environment and Pip tool. Be careful about different path separators on operating systems. At present, Python 3.6-3.8 versions are compatible.

## Running the tests

    python -m QCompute.Test.PostInstall.PostInstall

Please test on a local simulator first, and then fill in your token of Quantum-hub to test on a cloud simulator.

## Development
1. QCompute SDK contains quantum toolkits, simulators, examples and docs. If concerned with quantum toolkits, e.g., the QCompute subfolder, you are highly suggested using 'local setup' process to ensure that any development could be reflected in the executing process. 
2. Most researchers who only work on the quantum applications (examples) are suggested to use one-step live setup. In this case, the local modification of QCompute would **NOT** be reflected in the executing process. However, the modification on the subfolder, Example, would be still valid.

## Contributing
Coding requirements:

1. Be familiar with quantum circuit model. Any pull should be tested first and then submitted. Be careful about the order of qubits.
2. Please comply with development specifications of relevant programming languages.

## Discussion
1. If any questions, advices, suggestions, please contact us via Email: quantum@baidu.com ;
2. Or, you can use internal feedback table in Quantum-hub to provide any feedbacks;
3. Or, you are also welcomed to join our discussion QQ group:
QQ Group Number：1147781135

![](https://release-data.cdn.bcebos.com/github-qleaf%2Fqrcode.png)

## Maintainers & Authors
Institute for Quantum Computing, Baidu.

## Changelog
The changelog of this project can be found in [CHANGELOG.md](https://github.com/baidu/QCompute/blob/master/CHANGELOG.md).

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/baidu/QCompute/blob/master/LICENSE) file for details


