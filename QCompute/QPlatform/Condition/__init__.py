from typing import Tuple, Set

from QCompute.QProtobuf import PBProgram, PBCircuitLine


def checkRealCondition(program: 'PBProgram') -> Tuple[int, int, int]:
    qRegSet = set()  # type: Set[int]
    cRegSet = set()  # type: Set[int]
    for circuitLine in program.body.circuit:  # type: PBCircuitLine
        qRegSet.update(circuitLine.qRegList)
        if circuitLine.WhichOneof('op') == 'measure':
            cRegSet.update(circuitLine.measure.cRegList)
    return len(qRegSet), len(cRegSet), len(program.body.circuit)
