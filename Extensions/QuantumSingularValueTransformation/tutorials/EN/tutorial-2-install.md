# Installation

*Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.*

## Create Python Environment

We recommend using Anaconda as the development environment management tool for Python3. Anaconda supports multiple mainstream operating systems (Windows, macOS, and Linux). It provides Scipy, Numpy, Matplotlib, and many other scientific computing and drawing packages. Conda, a virtual environment manager, can be used to install and update the latest Python packages. Here we give simple instructions on how to use conda to create and manage virtual environments.

1. First, open the command line (Terminal) interface: Windows users can enter ``Anaconda Prompt``; macOS users can use the key combination ``command + space`` and then enter ``Terminal``.

2. After opening the Terminal window, enter

    ```bash
    conda create --name qsvt_env python=3.9
    ```

    to create a Python 3.9 environment named ``qsvt_env``. With the following command, we can activate the virtual environment created,

    ```bash
    conda activate qsvt_env
    ```

For more detailed instructions on conda, please refer to the [Official Tutorial](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

## Install QSVT Toolkit

QSVT toolkit is compatible with 64-bit Python 3.9, on Linux, macOS (10.14 or later) and Windows. We recommend installing it with ``pip``. After Activating the conda environment, then enter

```bash
pip install qcompute-qsvt
```

This will install the QSVT toolkit binaries as well as the QSVT toolkit package. 

For those using an older version of QSVT toolkit, update by installing with the `--upgrade` flag. The new version includes additional features and bug fixes.

## Run Examples

Now, you can try to write a simple program to check whether QSVT toolkit has been installed successfully. For example, run the following program,

```python
import numpy as np
from Extensions.QuantumSingularValueTransformation.qcompute_qsvt.Application.HamiltonianSimulation.HamiltonianSimulation import func_HS_QSVT

print(func_HS_QSVT(list_str_Pauli_rep=[(1, 'X0X1'), (1, 'X0Z1'), (1, 'Z0X1'), (1, 'Z0Z1')], 
                   num_qubit_sys=2, float_tau=-np.pi / 8, float_epsilon=1e-6, circ_output=False)['counts'])
```

we will operate a time evolution operator on initial state $|00\rangle$ for Hamiltonian  $X\otimes X + X\otimes Z + Z\otimes X + Z\otimes Z$ and time $-\pi/8$ with precision `1e-6`, and then measure such final state.

Note that more examples are provided in the [source files](https://github.com/baidu/QCompute/tree/master/Extensions/QuantumSingularValueTransformation). You can get started from there.
