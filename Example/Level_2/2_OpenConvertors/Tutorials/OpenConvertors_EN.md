# OpenConvertors

OpenConvertors are used to implement the conversion of the data model of the circuit. It should be noted that:
- When using the converter, the modules won't be automatically called.
- The converter supports some quantum gate operations, please refer to the following instructions for details.
- The converter does not support subroutines. When there are subroutines in the circuit, you need to call the UnrollProcedureModule module to decompose.

## Circuit

Run the following code to get `Circuit`, namely `PBProgram circuit model`, also known as `QCompute circuit model`.

```
from QCompute import *
from pprint import pprint
from QCompute.Define import Settings
Settings.outputInfo = False

env = QEnv()

q = env.Q.createList(2)

X(q[0])
CX(q[0], q[1])
MeasureZ(*env.Q.toListPair())

# Generate PBProgram and output
env.publish()
pprint(env.program)
```

## DrawConsole

The QCompute SDK provides a one-way converter for implementing `Circuit` to `DrawConsole`.

```
from QCompute import *
from pprint import pprint
from QCompute.Define import Settings
Settings.outputInfo = False

env = QEnv()

q = env.Q.createList(2)

X(q[0])
CX(q[0], q[1])
MeasureZ(*env.Q.toListPair())

# Generate PBProgram
env.publish()

# Convert PBProgram to DrawConsole and output
DrawConsole = CircuitToDrawConsole().convert(env.program)
pprint(DrawConsole)
```

## InternalStruct

The QCompute SDK provides bidirectional converters for implementing `Circuit` and `InternalStruct`.

```
from QCompute import *
from pprint import pprint
from QCompute.Define import Settings
Settings.outputInfo = False

env = QEnv()

q = env.Q.createList(2)

X(q[0])
CX(q[0], q[1])
MeasureZ(*env.Q.toListPair())

# Generate PBProgram
env.publish()
pprint(env.program)

# Convert PBProgram to InternalStruct and output
circuitLineList = CircuitToInternalStruct().convert(env.program.body.circuit)
pprint(circuitLineList)

# Convert InternalStruct to PBProgram and output
circuit = InternalStructToCircuit().convert(circuitLineList)
pprint(circuit)
```

## JSON

The QCompute SDK provides bidirectional converters for implementing Circuit and JSON circuit models.

```
from QCompute import *
from pprint import pprint
from QCompute.Define import Settings
Settings.outputInfo = False

env = QEnv()

q = env.Q.createList(2)

X(q[0])
CX(q[0], q[1])
MeasureZ(*env.Q.toListPair())

# Generate PBProgram
env.publish()

# Convert PBProgram to JSON and output
jsonStr = CircuitToJson().convert(env.program)
pprint(jsonStr)

# Convert JSON to PBProgram and output
circuit = JsonToCircuit().convert(jsonStr)
pprint(circuit)
```

## QASM

The QCompute SDK provides bidirectional converters for implementing Circuit and QASM circuit models.

```
from QCompute import *
from pprint import pprint
from QCompute.Define import Settings
Settings.outputInfo = False

env = QEnv()

q = env.Q.createList(2)

X(q[0])
CX(q[0], q[1])
MeasureZ(*env.Q.toListPair())

# Generate PBProgram
env.publish()

# Convert PBProgram to QASM and output
qasmStr = CircuitToQasm().convert(env.program)
pprint(qasmStr)

# Convert QASM to PBProgram and output
circuit = QasmToCircuit().convert(qasmStr)
pprint(circuit)
```

## Qobj

The QCompute SDK provides a one-way converter for implementing Circuit to the Qobj circuit model used by the third-party open source simulator Aer.

```
from QCompute import *
from pprint import pprint
from QCompute.Define import Settings
Settings.outputInfo = False

env = QEnv()

q = env.Q.createList(2)

X(q[0])
CX(q[0], q[1])
MeasureZ(*env.Q.toListPair())

# Generate PBProgram
env.publish()

# Convert PBProgram to Qobj and output
qobj = CircuitToQobj().convert(env.program, 1024)
pprint(qobj)
```

## IonQ

The ConvertorIonQ is used to realize bidirectional conversion of QCompute circuit model to IonQ circuit model.

```
from QCompute import *
from pprint import pprint
from QCompute.Define import Settings
Settings.outputInfo = False

env = QEnv()

q = env.Q.createList(2)

X(q[0])
CX(q[0], q[1])
MeasureZ(*env.Q.toListPair())

# Generate PBProgram
env.publish()

# Convert PBProgram to IonQ-JSON and output
ionq_program = CircuitToIonQ().convert(env.program)
pprint(ionq_program)

# Convert IonQ-JSON to PBProgram and output
circuit = IonQToCircuit().convert(ionq_program)
pprint(circuit)
```

### Method description

**CircuitToIonQ().convert()**

Used to convert PBProgram circuit model to JSON circuit model received by IonQ.

- Quantum gates supported in PBProgram circuits
  > Fixed Gates: X - Y - Z - H - CX - S - SDG - T - TDG - SWAP - CCX
  > 
  > Rotation Gates: RX - RY - RZ
- Input : PBProgram
- Output : Str


**IonQToCircuit().convert()**

Used to convert IonQ-JSON circuit model to PBProgram circuit model.

- Quantum gates supported in IonQ-JSON circuits
  > x - y - z - rx - ry - rz - h - not - cnot - s - si - t - ti - v - vi - xx - yy - zz - swap
- Input : Str
- Output : PBProgram

## XanaduSF

The ConvertorXanaduSF is used to realize conversion of QCompute circuit model to Xanadu-StrawberryFields(SF) circuit model.The converter generates Xanadu-SF circuit objects to the Blackbird format.

### Method description

**CircuitToXanaduSF().convert()**

The method is used to convert PBProgram circuit model to Blackbird circuit model.

> Quantum gates supported in PBProgram circuits
>
> Fixed Gate : X - H - CX - CZ
>
> Rotation Gate : RX - RY

**Input : PBProgram, TwoQubitsGate**

TwoQubitsGate is used to select the conversion scheme of the two-qubit gates in the circuit.

The converter offers 3 options:

- TwoQubitsGate.quarter : 1/4 probabilistic two-qubit gates (default)
- TwoQubitsGate.third : 1/3 probabilistic two-qubit gates
- TwoQubitsGate.kerr : cross-Kerr two-qubit gates

**Output : Blackbird**
