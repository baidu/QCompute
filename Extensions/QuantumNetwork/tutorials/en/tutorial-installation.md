# Installation

*Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.*

## Create Python environment

We recommend using [Anaconda](https://www.anaconda.com/products/distribution) as the development environment management tool for Python3. Anaconda supports multiple mainstream operating systems (Windows, macOS, and Linux). It provides Scipy, Numpy, Matplotlib, and many other scientific computing and drawing packages. Conda, a virtual environment manager, can be used to install and update the latest Python packages. Here we give simple instructions on how to use conda to create and manage virtual environments.

1. First, open the command line (Terminal) interface: Windows users can enter ``Anaconda Prompt``; MacOS users can use the key combination ``command + space`` and then enter ``Terminal``.

2. After opening the Terminal window, enter
```bash
conda create --name qnet_env python=3.8
```
to create a Python 3.8 environment named qnet_env. With the following command, we can activate the virtual environment created,
```bash
conda activate qnet_env
```

For more detailed instructions on conda, please refer to the [Official Tutorial](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).


## Install QNET

QNET is compatible with 64-bit Python 3.8+, on Linux, MacOS (10.14 or later) and Windows. We recommend installing it with ``pip``. Activate the conda environment and enter
```bash
pip install qcompute-qnet
```

This will install the QNET binaries as well as the QNET package. For those using an older version of QNET, keep up to date by installing with the ``--upgrade`` flag for additional features and bug fixes.


## Run examples

Now, you can try to write a simple program to check whether QNET has been successfully installed. For example,


```python
from qcompute_qnet.core.des import DESEnv
from qcompute_qnet.topology import Network, Node, Link

# Create a simulation environment
env = DESEnv("Simulation Environment", default=True)
# Create a network
network = Network("First Network")  
# Create a node named Alice
alice = Node("Alice")  
# Create another node named Bob
bob = Node("Bob")  
# Create a link between Alice and Bob
link = Link("Alice_Bob", ends=(alice, bob)) 
# Build up the network from nodes and links 
network.install([alice, bob, link])  
# Initialize the simulation environment
env.init()
# Run the network simulation
env.run()
```

Note that more examples are provided in the API documentation. You can get started from there.
