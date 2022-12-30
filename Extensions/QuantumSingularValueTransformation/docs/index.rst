QSVT Toolkit
=========================================

**QSVT** toolkit is a **Q**\uantum **S**\ingular **V**\alue **T**\ransformation toolkit based on `QCompute <https://quantum-hub.baidu.com/opensource>`_ and developed by the `Institute for Quantum Computing <https://quantum.baidu.com>`_ at `Baidu Research <http://research.baidu.com>`_. It aims to implement quantum simulation algorithms on quantum devices or simulators. Currently, it includes three main modules:

+ **Quantum Singular Value Transformation** is used for implementing functions of operators, which may be input as block-encodings of quantum operations or quantum circuits.

+ **Symmetric Quantum Signal Processing** is used for completing quantum circuits in quantum singular value transformation, since such functions are encoded by processing parameters in quantum signal processing. Symmetric quantum signal processing is proposed since those parameters could be found more effectively.

+ **Hamiltonian Simulation** is one of the most significant applications for QSVT, and even quantum computing. It provides functions to generate quantum circuits for time evolution operators of Hamiltonians.

QSVT toolkit is based on `QCompute <https://quantum-hub.baidu.com/opensource>`_, a Python-based open-source quantum computing platform SDK also developed by `Institute for Quantum Computing <https://quantum.baidu.com>`_. It provides a full-stack programming experience for senior users via hybrid quantum programming language features and high-performance simulators. You can install QCompute via `pypi <https://pypi.org/project/qcompute/>`_. When you install QSVT toolkit, the dependency QCompute will be automatically installed. Please refer to QCompute's official `Open Source <https://quantum-hub.baidu.com/opensource>`_ page for more details.

Installation
============

The package QSVT toolkit is compatible with 64-bit Python 3.9, on Linux, MacOS (10.14 or later) and Windows. We highly recommend the users to install QSVT toolkit via ``pip``. Open the Terminal and run

::

   pip install qcompute-qsvt

This will install the QSVT toolkit binaries as well as the QSVT toolkit package. For those using an older version of QSVT toolkit, update by installing with the ``--upgrade`` flag. The new version includes additional features and bug fixes.

After successfully installing QSVT toolkit, you can try to write a simple program to check whether QSVT toolkit has been successfully installed. For example, run the following program,

.. code-block:: python
   :linenos:

   import numpy as np
   from qcompute_qsvt.Applications.HamiltonianSimulation import func_HS_QSVT

   func_HS_QSVT(list_str_Pauli_rep=[(1, 'X0X1'), (1, 'X0Z1'), (1, 'Z0X1'), (1, 'Z0Z1')],
      num_qubit_sys=2, float_tau=-np.pi / 8, float_epsilon=1e-6, circ_output=False)

we will operate a time evolution operator on the initial state :math:`|00\rangle` for Hamiltonian :math:`X\otimes X + X\otimes Z + Z\otimes X + Z\otimes Z` and time :math:`-\pi/8` with precision :math:`10^{-6}`, and then measure such final state.


Tutorials
=========

QSVT toolkit provides `introduction tutorials <https://quantum-hub.baidu.com/docs/qsvt>`_ for Symmetric Quantum Signal Processing (SQSP), Quantum Singular Value Transformation (QSVT) and their application in Hamiltonian Simulation. Besides, Block-encoding and the decomposition for multictrl gates are also included as elementary infrastructure. We will supply more detailed and comprehensive tutorials in the next version.

QSVT toolkit provides `quick start <https://quantum-hub.baidu.com/qsvt/tutorial-quickstart>`_ for using the Hamiltonian Simulation module, as well as a `brief introduction <https://quantum-hub.baidu.com/qsvt/tutorial-introduction>`_ to the theories for users to learn and get started. The current content of the following tutorial is organized as follows, and it is recommended that beginners read and study in order:

- `Brief Introduction <https://quantum-hub.baidu.com/qsvt/tutorial-introduction>`_
- `Quantum Signal Processing <https://quantum-hub.baidu.com/qsvt/tutorial-qsp>`_
- `Block-Encoding and Linear Combination of Unitary Operations <https://quantum-hub.baidu.com/qsvt/tutorial-be>`_
- `Quantum Eigenvalue and Singular Value Transformation <https://quantum-hub.baidu.com/qsvt/tutorial-qet>`_
- `Hamiltonian Simulation <https://quantum-hub.baidu.com/qsvt/tutorial-hs>`_

We will supply more detailed and comprehensive tutorials in the future.

Contents
========

The content of API documentation for QSVT toolkit is organized as follows:

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Makefile

   Makefile/

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: make.bat

   make.bat/

.. toctree::
   :maxdepth: 2
   :glob:
   :caption: Application

   Application/HamiltonianSimulation

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Gate

   Gate/MultiCtrlGates

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Oracle

   Oracle/BlockEncoding
   Oracle/StatePreparation

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: QSVT

   QSVT/QSVT

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: SymmetricQSP

   SymmetricQSP/Settings
   SymmetricQSP/SymmetricQSPExternal
   SymmetricQSP/SymmetricQSPInternalPy

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
