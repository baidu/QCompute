"""
//The rules in this file heavily rely on the qelib1.inc which is the Quantum Experience (QE) Standard Header of IBM Qiskit.
//qelib1.inc can be obtained from the link:
//https://github.com/Qiskit/qiskit-terra/blob/master/qiskit/qasm/libs/qelib1.inc
"""
from copy import deepcopy
from typing import List, Dict, Optional, Union

import numpy as np

from QCompute.OpenModule import ModuleImplement, ModuleErrorCode
from QCompute.QPlatform import Error
from QCompute.QPlatform.CircuitTools import gateToProtobuf
from QCompute.QPlatform.QOperation.FixedGate import X, H, S, SDG, T, TDG, CX, CCX
from QCompute.QPlatform.QOperation.RotationGate import U
from QCompute.QProtobuf import PBProgram, PBCircuitLine, PBFixedGate, PBRotationGate

FileErrorCode = 3


class UnrollCircuitModule(ModuleImplement):
    """
    UnrollCircuit supported gates are: CX, U3, barrier, measure

    Supported fixed gates: ID, X, Y, Z, H, S, SDG, T, TDG, CX, CY, CZ, CH, SWAP, CCX, CSWAP

    Supported rotation gates: U, RX, RY, RZ, CU, CRX, CRY, CRZ

    Composite gates are supported since they can be processed by the CompositeGateModule module in advance.

    Must unrollProcedure before, because rotation gate must have all rotation arguments.

    Example:

    env.module(UnrollCircuitModule())

    env.module(UnrollCircuitModule({'disable': True}))  # Disable

    env.module(UnrollCircuitModule({'errorOnUnsupported': True, 'targetGates': ['CX', 'U']}))

    env.module(UnrollCircuitModule({'errorOnUnsupported': False, 'targetGates': ['CX', 'U'], 'sourceGates': ['CH', 'CSWAP']}))
    """

    arguments = None  # type: Optional[Dict[str, Union[List[str], bool]]]
    targetGatesNames = ['CX', 'U']  # type: List[str]
    sourceGatesNames = []  # type: List[str]
    errorOnUnsupported = True

    def __init__(self, arguments: Optional[Dict[str, Union[List[str], bool]]] = None):
        """
        Initialize the Module.

        Json serialization is allowed by the requested parameter.
        """

        self.arguments = arguments
        if arguments is not None and type(arguments) is dict:
            if 'disable' in arguments:
                self.disable = arguments['disable']

            if 'targetGates' in arguments:
                self.targetGatesNames = arguments['targetGates']

            if 'sourceGates' in arguments:
                self.sourceGatesNames = arguments['sourceGates']

            if 'errorOnUnsupported' in arguments:
                self.errorOnUnsupported = arguments['errorOnUnsupported']

    def __call__(self, program: 'PBProgram') -> 'PBProgram':
        """
        Process the Module

        :param program: the program
        :return: unrolled circuit
        """

        ret = deepcopy(program)

        for name, procedure in program.body.procedureMap.items():
            targetProcedure = ret.body.procedureMap[name]
            del targetProcedure.circuit[:]
            self._decompose(targetProcedure.circuit, procedure.circuit)
        del ret.body.circuit[:]
        self._decompose(ret.body.circuit, program.body.circuit)
        return ret

    def _unrollRecursively(self, circuitLine: 'PBCircuitLine', circuitOut: List['PBCircuitLine']):
        """
        Only CX,U,barrier, measure are processed directly by circuitOut.append()

        Other gates are processed recursively

        _expandAnglesInUGate() is used only once in process of U
        """

        op = circuitLine.WhichOneof('op')
        if op == 'procedureName' or op == 'measure' or op == 'barrier':
            ret = deepcopy(circuitLine)  # type: 'PBCircuitLine'
            circuitOut.append(ret)
            return
        elif op == 'fixedGate' or op == 'rotationGate':
            # Name the regs
            nRegs = len(circuitLine.qRegList)
            if nRegs == 1:
                [a] = circuitLine.qRegList
            elif nRegs == 2:
                [a, b] = circuitLine.qRegList
            elif nRegs == 3:
                [a, b, c] = circuitLine.qRegList
            else:
                raise Error.ArgumentError(f'Wrong regs count. regs: {circuitLine.qRegList}!', ModuleErrorCode,
                                          FileErrorCode, 1)

            if op == 'fixedGate':
                fixedGate = circuitLine.fixedGate  # type: PBFixedGate
                # Recognize known gates
                gateName = PBFixedGate.Name(fixedGate)
                if len(self.sourceGatesNames) > 0 and gateName not in self.sourceGatesNames:
                    # Don't need unroll: copy
                    ret = deepcopy(circuitLine)  # type: 'PBCircuitLine'
                    circuitOut.append(ret)
                    return
                elif gateName in self.targetGatesNames:
                    # Already supported by target machine: copy
                    ret = deepcopy(circuitLine)  # type: 'PBCircuitLine'
                    circuitOut.append(ret)
                    return
                elif fixedGate == PBFixedGate.ID:
                    # ID: gate id a { U(0,0,0) a; }
                    self._unrollRecursively(gateToProtobuf(U(0.0, 0.0, 0.0), [a]), circuitOut)
                    return
                elif fixedGate == PBFixedGate.X:
                    # X: gate x a { u3(pi,0,pi) a; }
                    self._unrollRecursively(gateToProtobuf(U(np.pi, 0.0, np.pi), [a]), circuitOut)
                    return
                elif fixedGate == PBFixedGate.Y:
                    # Y: gate y a { u3(pi,pi/2,pi/2) a; }
                    self._unrollRecursively(gateToProtobuf(U(np.pi, np.pi / 2.0, np.pi / 2.0), [a]), circuitOut)
                    return
                elif fixedGate == PBFixedGate.Z:
                    # Z: gate z a { u1(pi) a; }
                    self._unrollRecursively(gateToProtobuf(U(np.pi), [a]), circuitOut)
                    return
                elif fixedGate == PBFixedGate.H:
                    # H: gate h a { u2(0,pi) a; }
                    self._unrollRecursively(gateToProtobuf(U(0.0, np.pi), [a]), circuitOut)
                    return
                elif fixedGate == PBFixedGate.S:
                    # S: gate s a { u1(pi/2) a; }
                    self._unrollRecursively(gateToProtobuf(U(np.pi / 2.0), [a]), circuitOut)
                    return
                elif fixedGate == PBFixedGate.SDG:
                    # SDG: gate sdg a { u1(-pi/2) a; }
                    self._unrollRecursively(gateToProtobuf(U(-np.pi / 2.0), [a]), circuitOut)
                    return
                elif fixedGate == PBFixedGate.T:
                    # T: gate t a { u1(pi/4) a; }
                    self._unrollRecursively(gateToProtobuf(U(np.pi / 4.0), [a]), circuitOut)
                    return
                elif fixedGate == PBFixedGate.TDG:
                    # TDG: gate tdg a { u1(-pi/4) a; }
                    self._unrollRecursively(gateToProtobuf(U(-np.pi / 4.0), [a]), circuitOut)
                    return
                elif fixedGate == PBFixedGate.CY:
                    # CY: gate cy a,b {
                    # sdg b;
                    self._unrollRecursively(gateToProtobuf(SDG, [b]), circuitOut)
                    # cx a,b;
                    self._unrollRecursively(gateToProtobuf(CX, [a, b]), circuitOut)
                    # s b;
                    self._unrollRecursively(gateToProtobuf(S, [b]), circuitOut)
                    # }
                    return
                elif fixedGate == PBFixedGate.CZ:
                    # CZ: gate cz a,b {
                    # h b;
                    self._unrollRecursively(gateToProtobuf(H, [b]), circuitOut)
                    # cx a,b;
                    self._unrollRecursively(gateToProtobuf(CX, [a, b]), circuitOut)
                    # h b;
                    self._unrollRecursively(gateToProtobuf(H, [b]), circuitOut)
                    # }
                    return
                elif fixedGate == PBFixedGate.CH:
                    # CH:
                    # gate ch a,b {
                    # h b;
                    self._unrollRecursively(gateToProtobuf(H, [b]), circuitOut)
                    # sdg b;
                    self._unrollRecursively(gateToProtobuf(SDG, [b]), circuitOut)
                    # cx a,b;
                    self._unrollRecursively(gateToProtobuf(CX, [a, b]), circuitOut)
                    # h b;
                    self._unrollRecursively(gateToProtobuf(H, [b]), circuitOut)
                    # t b;
                    self._unrollRecursively(gateToProtobuf(T, [b]), circuitOut)
                    # cx a,b;
                    self._unrollRecursively(gateToProtobuf(CX, [a, b]), circuitOut)
                    # t b;
                    self._unrollRecursively(gateToProtobuf(T, [b]), circuitOut)
                    # h b;
                    self._unrollRecursively(gateToProtobuf(H, [b]), circuitOut)

                    # our modification: s b;x b -> t b;x b;tdg b;
                    # s b;
                    # self._unrollRecursively(gateToProtobuf(S, [b]), circuitOut)
                    # x b;
                    # self._unrollRecursively(gateToProtobuf(X, [b]), circuitOut)
                    # t b
                    self._unrollRecursively(gateToProtobuf(T, [b]), circuitOut)
                    # x b
                    self._unrollRecursively(gateToProtobuf(X, [b]), circuitOut)
                    # tdg b;
                    self._unrollRecursively(gateToProtobuf(TDG, [b]), circuitOut)

                    # s a;
                    self._unrollRecursively(gateToProtobuf(S, [a]), circuitOut)
                    # }
                    return
                elif fixedGate == PBFixedGate.SWAP:
                    # SWAP: gate swap a,b {
                    # cx a,b;
                    self._unrollRecursively(gateToProtobuf(CX, [a, b]), circuitOut)
                    # cx b,a;
                    self._unrollRecursively(gateToProtobuf(CX, [b, a]), circuitOut)
                    # cx a,b;
                    self._unrollRecursively(gateToProtobuf(CX, [a, b]), circuitOut)
                    # }
                    return
                elif fixedGate == PBFixedGate.CCX:
                    # CCX:
                    # gate ccx a,b,c
                    # {
                    #   h c;
                    self._unrollRecursively(gateToProtobuf(H, [c]), circuitOut)
                    #   cx b,c;
                    self._unrollRecursively(gateToProtobuf(CX, [b, c]), circuitOut)
                    #   tdg c;
                    self._unrollRecursively(gateToProtobuf(TDG, [c]), circuitOut)
                    #   cx a,c;
                    self._unrollRecursively(gateToProtobuf(CX, [a, c]), circuitOut)
                    #   t c;
                    self._unrollRecursively(gateToProtobuf(T, [c]), circuitOut)
                    #   cx b,c;
                    self._unrollRecursively(gateToProtobuf(CX, [b, c]), circuitOut)
                    #   tdg c;
                    self._unrollRecursively(gateToProtobuf(TDG, [c]), circuitOut)
                    #   cx a,c;
                    self._unrollRecursively(gateToProtobuf(CX, [a, c]), circuitOut)
                    #   t b;
                    self._unrollRecursively(gateToProtobuf(T, [b]), circuitOut)
                    #   t c;
                    self._unrollRecursively(gateToProtobuf(T, [c]), circuitOut)
                    #   h c;
                    self._unrollRecursively(gateToProtobuf(H, [c]), circuitOut)
                    #   cx a,b;
                    self._unrollRecursively(gateToProtobuf(CX, [a, b]), circuitOut)
                    #   t a;
                    self._unrollRecursively(gateToProtobuf(T, [a]), circuitOut)
                    #   tdg b;
                    self._unrollRecursively(gateToProtobuf(TDG, [b]), circuitOut)
                    #   cx a,b;
                    self._unrollRecursively(gateToProtobuf(CX, [a, b]), circuitOut)
                    # }
                    return
                elif fixedGate == PBFixedGate.CSWAP:
                    # CSWAP: gate cswap a,b,c {
                    # cx c,b;
                    self._unrollRecursively(gateToProtobuf(CX, [c, b]), circuitOut)
                    # ccx a,b,c;
                    self._unrollRecursively(gateToProtobuf(CCX, [a, b, c]), circuitOut)
                    # cx c,b;
                    self._unrollRecursively(gateToProtobuf(CX, [c, b]), circuitOut)
                    # }
                    return
            elif op == 'rotationGate':
                rotationGate = circuitLine.rotationGate  # type: PBRotationGate
                if len(circuitLine.argumentIdList) > 0:
                    raise Error.ArgumentError(f'Can not unroll argument id. angles id: {circuitLine.argumentIdList}!',
                                              ModuleErrorCode, FileErrorCode, 2)
                nAngles = len(circuitLine.argumentValueList)
                if nAngles == 1:
                    [theta] = circuitLine.argumentValueList
                elif nAngles == 2:
                    [theta, phi] = circuitLine.argumentValueList
                elif nAngles == 3:
                    [theta, phi, lamda] = circuitLine.argumentValueList
                else:
                    raise Error.ArgumentError(f'Wrong angles count. angles value: {circuitLine.argumentValueList}!',
                                              ModuleErrorCode, FileErrorCode, 3)

                gateName = PBRotationGate.Name(rotationGate)
                if len(self.sourceGatesNames) > 0 and gateName not in self.sourceGatesNames:
                    # Don't need unroll: copy
                    ret = deepcopy(circuitLine)  # type: 'PBCircuitLine'
                    circuitOut.append(ret)
                    return
                elif gateName in self.targetGatesNames:
                    # Already supported by target machine: copy or expand+copy
                    if rotationGate == PBRotationGate.U:
                        # U3: gate u3(theta,phi,lamda) a { U(theta,phi,lamda) a; }
                        # U2: gate u2(theta,phi) a { U(pi/2,theta,phi) a; }
                        # U1: gate u1(theta) a { U(0,0,theta) a; }
                        circuitOut.append(
                            gateToProtobuf(U(*_expandAnglesInUGate(list(circuitLine.argumentValueList))), [a]))
                        return
                    else:
                        ret = deepcopy(circuitLine)  # type: 'PBCircuitLine'
                        circuitOut.append(ret)
                        return
                elif rotationGate == PBRotationGate.RX:
                    # RX: gate rx(theta) a { u3(theta, -pi/2,pi/2) a; }
                    self._unrollRecursively(gateToProtobuf(U(theta, -np.pi / 2.0, np.pi / 2.0), [a]), circuitOut)
                    return
                elif rotationGate == PBRotationGate.RY:
                    # RY: gate ry(theta) a { u3(theta,0,0) a; }
                    self._unrollRecursively(gateToProtobuf(U(theta, 0.0, 0.0), [a]), circuitOut)
                    return
                elif rotationGate == PBRotationGate.RZ:
                    # RZ: gate rz(theta) a { u1(theta) a; }
                    self._unrollRecursively(gateToProtobuf(U(theta), [a]), circuitOut)
                    return
                elif rotationGate == PBRotationGate.CU:
                    # CU: gate cu3(theta,phi,lamda) a, b
                    # {
                    #   u1((lamda+phi)/2) a;
                    self._unrollRecursively(gateToProtobuf(U((lamda + phi) / 2.0), [a]), circuitOut)
                    #   u1((lamda-phi)/2) b;
                    self._unrollRecursively(gateToProtobuf(U((lamda - phi) / 2.0), [b]), circuitOut)
                    #   cx a, b;
                    self._unrollRecursively(gateToProtobuf(CX, [a, b]), circuitOut)
                    #   u3(-theta/2,0,-(phi+lamda)/2) b;
                    self._unrollRecursively(gateToProtobuf(U(-theta / 2.0, 0.0, -(phi + lamda) / 2.0), [b]), circuitOut)
                    #   cx a, b;
                    self._unrollRecursively(gateToProtobuf(CX, [a, b]), circuitOut)
                    #   u3(theta/2,phi,0) b;
                    self._unrollRecursively(gateToProtobuf(U(theta / 2.0, phi, 0.0), [b]), circuitOut)
                    # }
                    return
                elif rotationGate == PBRotationGate.CRX:
                    # CRX:
                    # gate crx(theta) a,b
                    # {
                    #   u1(pi/2) b;
                    self._unrollRecursively(gateToProtobuf(U(np.pi / 2.0), [b]), circuitOut)
                    #   cx a,b;
                    self._unrollRecursively(gateToProtobuf(CX, [a, b]), circuitOut)
                    #   u3(-theta/2,0,0) b;
                    self._unrollRecursively(gateToProtobuf(U(-theta / 2.0, 0.0, 0.0), [b]), circuitOut)
                    #   cx a,b;
                    self._unrollRecursively(gateToProtobuf(CX, [a, b]), circuitOut)
                    #   u3(theta/2,-pi/2,0) b;
                    self._unrollRecursively(gateToProtobuf(U(theta / 2.0, -np.pi / 2.0, 0.0), [b]), circuitOut)
                    # }
                    return
                elif rotationGate == PBRotationGate.CRY:
                    # CRY:
                    # gate cry(theta) a,b
                    # {
                    #   u3(theta/2,0,0) b;
                    self._unrollRecursively(gateToProtobuf(U(theta / 2.0, 0.0, 0.0), [b]), circuitOut)
                    #   cx a,b;
                    self._unrollRecursively(gateToProtobuf(CX, [a, b]), circuitOut)
                    #   u3(-theta/2,0,0) b;
                    self._unrollRecursively(gateToProtobuf(U(-theta / 2.0, 0.0, 0.0), [b]), circuitOut)
                    #   cx a,b;
                    self._unrollRecursively(gateToProtobuf(CX, [a, b]), circuitOut)
                    # }
                    return
                elif rotationGate == PBRotationGate.CRZ:
                    # CRZ:
                    # gate crz(theta) a,b
                    # {
                    #   u1(theta/2) b;
                    self._unrollRecursively(gateToProtobuf(U(theta / 2.0), [b]), circuitOut)
                    #   cx a,b;
                    self._unrollRecursively(gateToProtobuf(CX, [a, b]), circuitOut)
                    #   u1(-theta/2) b;
                    self._unrollRecursively(gateToProtobuf(U(-theta / 2.0), [b]), circuitOut)
                    #   cx a,b;
                    self._unrollRecursively(gateToProtobuf(CX, [a, b]), circuitOut)
                    # }
                    return

        # Unsupported gate
        if self.errorOnUnsupported:
            # error
            raise Error.ArgumentError(f'Unsupported operation {circuitLine}!', ModuleErrorCode, FileErrorCode, 4)
        else:
            # ignore
            ret = deepcopy(circuitLine)  # type: 'PBCircuitLine'
            circuitOut.append(ret)

    def _decompose(self, circuitOut: List['PBCircuitLine'], circuitIn: List['PBCircuitLine']):
        """
        Unroll the gates

        :param circuitOut: input circuit
        :param circuitIn: output circuit
        """

        for circuitLine in circuitIn:
            self._unrollRecursively(circuitLine, circuitOut)


def _expandAnglesInUGate(angles: List[float]):
    """
    Expand the angles list by following the relation among u1, u2, and u3
    """

    # Expand
    nAngles = len(angles)
    if nAngles == 3:
        pass
    elif nAngles == 2:
        angles = [np.pi / 2.0] + angles
    elif nAngles == 1:
        angles = [0.0, 0.0] + angles
    else:
        raise Error.ArgumentError(f'Wrong angles count. angles: {angles}!', ModuleErrorCode, FileErrorCode, 5)

    return angles
