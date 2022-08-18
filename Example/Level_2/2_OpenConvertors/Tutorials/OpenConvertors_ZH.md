# OpenConvertors

OpenConvertors 用于实现电路的数据模型的转换。需注意的是：
- 使用转换器时，不会自动调用模块。
- 转换器支持部分量子门操作，具体参考以下说明。
- 转换器不支持子程序，当电路中存在子程序时，需要先调用 UnrollProcedureModule 模块进行分解。

## Circuit

运行如下代码即可得到 Circuit，即 PBProgram 电路模型，也称 QCompute 电路模型。

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

# 生成 PBProgram 并输出
env.publish()
pprint(env.program)
```

## DrawConsole

QCompute SDK 提供用于实现 Circuit 到 DrawConsole 的单向转换器。

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

# 生成 PBProgram
env.publish()

# 将 PBProgram 转换为 DrawConsole 并输出
DrawConsole = CircuitToDrawConsole().convert(env.program)
pprint(DrawConsole)
```

## InternalStruct

QCompute SDK 提供用于实现 Circuit 和 InternalStruct 的双向转换器。

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

# 生成 PBProgram
env.publish()
pprint(env.program)

# 将 PBProgram 转换为 InternalStruct 并输出
circuitLineList = CircuitToInternalStruct().convert(env.program.body.circuit)
pprint(circuitLineList)

# 将 InternalStruct 转换为 PBProgram 并输出
circuit = InternalStructToCircuit().convert(circuitLineList)
pprint(circuit)
```

## JSON

QCompute SDK 提供用于实现 Circuit 和 JSON 电路模型的双向转换器。

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

# 生成 PBProgram
env.publish()

# 将 PBProgram 转换为 JSON 并输出
jsonStr = CircuitToJson().convert(env.program)
pprint(jsonStr)

# 将 JSON 转换为 PBProgram 并输出
circuit = JsonToCircuit().convert(jsonStr)
pprint(circuit)
```

## QASM

QCompute SDK 提供用于实现 Circuit 和 QASM 电路模型的双向转换器。

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

# 生成 PBProgram
env.publish()

# 将 PBProgram 转换为 QASM 并输出
qasmStr = CircuitToQasm().convert(env.program)
pprint(qasmStr)

# 将 QASM 转换为 PBProgram 并输出
circuit = QasmToCircuit().convert(qasmStr)
pprint(circuit)
```

## Qobj

QCompute SDK 提供用于实现 Circuit 到第三方开源模拟器 Aer 使用的 Qobj 电路模型的单向转换器。

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

# 生成 PBProgram
env.publish()

# 将 PBProgram 转换为 Qobj 并输出
qobj = CircuitToQobj().convert(env.program, 1024)
pprint(qobj)
```

## IonQ

QCompute SDK 提供用于实现 Circuit 和 IonQ 使用的 JSON 电路模型的双向转换器。

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

# 生成 PBProgram
env.publish()

# 将 PBProgram 转换为 IonQ 接收的 JSON 并输出
ionq_program = CircuitToIonQ().convert(env.program)
pprint(ionq_program)

# 将 IonQ 接收的 JSON 转换为 PBProgram 并输出
circuit = IonQToCircuit().convert(ionq_program)
pprint(circuit)
```

### 方法说明

**CircuitToIonQ().convert()**

用于将 PBProgram 电路模型转换为 IonQ 接收的 JSON 电路模型。

- PBProgram 电路中支持量子门
  > 固定门 : X - Y - Z - H - CX - S - SDG - T - TDG - SWAP - CCX
  > 
  > 旋转门 : RX - RY - RZ
- Input : PBProgram
- Output : Str

**IonQToCircuit().convert()**

用于将 IonQ 的 JSON 电路模型转换为 IonQ 接收的 JSON 电路模型。

- IonQ-JSON 电路中支持量子门 
  > x - y - z - rx - ry - rz - h - not - cnot - s - si - t - ti - v - vi - xx - yy - zz - swap
- Input : Str
- Output : PBProgram

## XanaduSF

ConvertorXanaduSF 转换器用于实现 QCompute 电路模型到 Xanadu-StrawberryFields(SF) 电路模型的转换。转换器生成 Xanadu-SF 电路对象到 i/o 层所使用的 Blackbird 格式。

### 方法说明

**CircuitToXanaduSF().convert()**

用于将 PBProgram 电路模型转换为 Blackbird 电路模型。

> PBProgram 电路中支持量子门
>
> 固定门 : X - H - CX - CZ
>
> 旋转门 : RX - RY

**Input : PBProgram, TwoQubitsGate**

TwoQubitsGate 用于选择电路中双比特门的转换方案。

转换器提供 3 种方案：
- TwoQubitsGate.quarter : 1/4 概率性双比特门 （默认）
- TwoQubitsGate.third : 1/3 概率性双比特门
- TwoQubitsGate.kerr : cross-Kerr 双比特门

**Output : Blackbird**
