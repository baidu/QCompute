"""
//The rules in this file heavily rely on the qelib1.inc which is the Quantum Experience (QE) Standard Header of IBM Qiskit.
//qelib1.inc can be obtained from the link:
//https://github.com/Qiskit/qiskit-terra/blob/master/qiskit/qasm/libs/qelib1.inc
"""

import copy

import numpy as np

from QCompute.QuantumPlatform.QuantumOperation.FixedGate import X, H, S, SDG, T, TDG, CX, CCX
from QCompute.QuantumPlatform.QuantumOperation.RotationGate import U
from QCompute.QuantumProtobuf.Library.QuantumOperation_pb2 import FixedGate as FixedGateEnum
from QCompute.QuantumProtobuf.Library.QuantumOperation_pb2 import RotationGate as RotationGateEnum


class UnrollCircuit:
    """
    Unroll supported gates to CX, U3, barrier, measure

    Supported fixed gates: ID, X, Y, Z, H, S, SDG, T, TDG, CX, CY, CZ, CH, SWAP, CCX, CSWAP

    Supported rotation gates: U, R, RX, RY, RZ, CU, CRX, CRY, CRZ

    Composite gates are supported since they can be processed by the CompositeGateModule module in advance.

    Must unrollProcedure before, because of paramIds to paramValues.

    Example:

    env.module(UnrollCircuit())

    env.module(UnrollCircuit({'errorOnUnsupported': True, 'targetGates': ['CX', 'U']}))

    env.module(UnrollCircuit({'errorOnUnsupported': False, 'targetGates': ['CX', 'U'], 'sourceGates': ['CH', 'CSWAP']}))
    """

    targetGatesNames = ['CX', 'U']
    sourceGatesNames = []
    errorOnUnsupported = True

    def __init__(self, params=None):
        """
        Initialize the Module.

        Json serialization is allowed by the requested parameter.

        :param params: {'errorOnUnsupported', True, 'targetGates': [CX, U]}.
        """

        if params is not None:
            if 'targetGates' in params:
                self.targetGatesNames = params['targetGates']

            if 'sourceGates' in params:
                self.sourceGatesNames = params['sourceGates']

            if 'errorOnUnsupported' in params:
                self.errorOnUnsupported = params['errorOnUnsupported']

    def __call__(self, program):
        """
        Process the Module

        :param program: the program
        :return: unrolled circuit
        """

        ret = copy.deepcopy(program)
        ret.body.ClearField('circuit')
        for id, procedure in program.body.procedureMap.items():
            targetProcedure = ret.body.procedureMap[id]
            targetProcedure.ClearField('circuit')
            self._decompose(targetProcedure.circuit, procedure.circuit)
        self._decompose(ret.body.circuit, program.body.circuit)
        return ret

    def _expandAnglesInUGate(self, angles):
        """
        Expand the angles list by following the relation among u1, u2, and u3
        """

        # normalize RepeatedScalarContainer
        angles = list(angles)

        # expand
        nAngles = len(angles)
        if nAngles == 3:
            pass
        elif nAngles == 2:
            angles = [np.pi / 2.0] + angles
        elif nAngles == 1:
            angles = [0.0, 0.0] + angles
        else:
            raise Exception('wrong angles length. angles: ' + str(angles))

        return angles

    def _unrollRecursively(self, circuitLine, circuitOut):
        """
        Only CX U barrier measure are processed directly by circuitOut.append()

        Other gates are processed recursively

        _expandAnglesInUGate() is used only once in process of U
        """

        if circuitLine.HasField('fixedGate') or circuitLine.HasField('rotationGate'):
            # name the regs
            nRegs = len(circuitLine.qRegs)
            if nRegs == 1:
                [a] = circuitLine.qRegs
            elif nRegs == 2:
                [a, b] = circuitLine.qRegs
            elif nRegs == 3:
                [a, b, c] = circuitLine.qRegs
            else:
                raise Exception('wrong regs length. regs: ' + str(circuitLine.qRegs))

        # name the angles
        if circuitLine.HasField('rotationGate'):
            nAngles = len(circuitLine.paramValues)
            if nAngles == 1:
                [theta] = circuitLine.paramValues
            elif nAngles == 2:
                [theta, phi] = circuitLine.paramValues
            elif nAngles == 3:
                [theta, phi, lamda] = circuitLine.paramValues
            else:
                raise Exception('wrong angles length. angles: ' + str(circuitLine.paramValues))

        # recognize known gates
        if circuitLine.HasField('fixedGate'):
            gateName = FixedGateEnum.Name(circuitLine.fixedGate)
            if len(self.sourceGatesNames) > 0 and gateName not in self.sourceGatesNames:
                # don't need unroll: copy
                circuitOut.append(circuitLine)
                return
            elif gateName in self.targetGatesNames:
                # already supported by target machine: copy
                circuitOut.append(circuitLine)
                return
            elif circuitLine.fixedGate == FixedGateEnum.ID:
                # ID: gate id a { U(0,0,0) a; }
                self._unrollRecursively(U(0.0, 0.0, 0.0)._toPB(a), circuitOut)
                return
            elif circuitLine.fixedGate == FixedGateEnum.X:
                # X: gate x a { u3(pi,0,pi) a; }
                self._unrollRecursively(U(np.pi, 0.0, np.pi)._toPB(a), circuitOut)
                return
            elif circuitLine.fixedGate == FixedGateEnum.Y:
                # Y: gate y a { u3(pi,pi/2,pi/2) a; }
                self._unrollRecursively(U(np.pi, np.pi / 2.0, np.pi / 2.0)._toPB(a), circuitOut)
                return
            elif circuitLine.fixedGate == FixedGateEnum.Z:
                # Z: gate z a { u1(pi) a; }
                self._unrollRecursively(U(np.pi)._toPB(a), circuitOut)
                return
            elif circuitLine.fixedGate == FixedGateEnum.H:
                # H: gate h a { u2(0,pi) a; }
                self._unrollRecursively(U(0.0, np.pi)._toPB(a), circuitOut)
                return
            elif circuitLine.fixedGate == FixedGateEnum.S:
                # S: gate s a { u1(pi/2) a; }
                self._unrollRecursively(U(np.pi / 2.0)._toPB(a), circuitOut)
                return
            elif circuitLine.fixedGate == FixedGateEnum.SDG:
                # SDG: gate sdg a { u1(-pi/2) a; }
                self._unrollRecursively(U(-np.pi / 2.0)._toPB(a), circuitOut)
                return
            elif circuitLine.fixedGate == FixedGateEnum.T:
                # T: gate t a { u1(pi/4) a; }
                self._unrollRecursively(U(np.pi / 4.0)._toPB(a), circuitOut)
                return
            elif circuitLine.fixedGate == FixedGateEnum.TDG:
                # TDG: gate tdg a { u1(-pi/4) a; }
                self._unrollRecursively(U(-np.pi / 4.0)._toPB(a), circuitOut)
                return
            elif circuitLine.fixedGate == FixedGateEnum.CY:
                # CY: gate cy a,b {
                # sdg b;
                self._unrollRecursively(SDG._toPB(b), circuitOut)
                # cx a,b;
                self._unrollRecursively(CX._toPB(a, b), circuitOut)
                # s b;
                self._unrollRecursively(S._toPB(b), circuitOut)
                # }
                return
            elif circuitLine.fixedGate == FixedGateEnum.CZ:
                # CZ: gate cz a,b {
                # h b;
                self._unrollRecursively(H._toPB(b), circuitOut)
                # cx a,b;
                self._unrollRecursively(CX._toPB(a, b), circuitOut)
                # h b;
                self._unrollRecursively(H._toPB(b), circuitOut)
                # }
                return
            elif circuitLine.fixedGate == FixedGateEnum.CH:
                # CH:
                # gate ch a,b {
                # h b;
                self._unrollRecursively(H._toPB(b), circuitOut)
                # sdg b;
                self._unrollRecursively(SDG._toPB(b), circuitOut)
                # cx a,b;
                self._unrollRecursively(CX._toPB(a, b), circuitOut)
                # h b;
                self._unrollRecursively(H._toPB(b), circuitOut)
                # t b;
                self._unrollRecursively(T._toPB(b), circuitOut)
                # cx a,b;
                self._unrollRecursively(CX._toPB(a, b), circuitOut)
                # t b;
                self._unrollRecursively(T._toPB(b), circuitOut)
                # h b;
                self._unrollRecursively(H._toPB(b), circuitOut)
                # s b;
                self._unrollRecursively(S._toPB(b), circuitOut)
                # x b;
                self._unrollRecursively(X._toPB(b), circuitOut)
                # s a;
                self._unrollRecursively(S._toPB(a), circuitOut)
                # }
                return
            elif circuitLine.fixedGate == FixedGateEnum.SWAP:
                # SWAP: gate swap a,b {
                # cx a,b;
                self._unrollRecursively(CX._toPB(a, b), circuitOut)
                # cx b,a;
                self._unrollRecursively(CX._toPB(b, a), circuitOut)
                # cx a,b;
                self._unrollRecursively(CX._toPB(a, b), circuitOut)
                # }
                return
            elif circuitLine.fixedGate == FixedGateEnum.CCX:
                # CCX:
                # gate ccx a,b,c
                # {
                #   h c;
                self._unrollRecursively(H._toPB(c), circuitOut)
                #   cx b,c;
                self._unrollRecursively(CX._toPB(b, c), circuitOut)
                #   tdg c;
                self._unrollRecursively(TDG._toPB(c), circuitOut)
                #   cx a,c;
                self._unrollRecursively(CX._toPB(a, c), circuitOut)
                #   t c;
                self._unrollRecursively(T._toPB(c), circuitOut)
                #   cx b,c;
                self._unrollRecursively(CX._toPB(b, c), circuitOut)
                #   tdg c;
                self._unrollRecursively(TDG._toPB(c), circuitOut)
                #   cx a,c;
                self._unrollRecursively(CX._toPB(a, c), circuitOut)
                #   t b;
                self._unrollRecursively(T._toPB(b), circuitOut)
                #   t c;
                self._unrollRecursively(T._toPB(c), circuitOut)
                #   h c;
                self._unrollRecursively(H._toPB(c), circuitOut)
                #   cx a,b;
                self._unrollRecursively(CX._toPB(a, b), circuitOut)
                #   t a;
                self._unrollRecursively(T._toPB(a), circuitOut)
                #   tdg b;
                self._unrollRecursively(TDG._toPB(b), circuitOut)
                #   cx a,b;
                self._unrollRecursively(CX._toPB(a, b), circuitOut)
                # }
                return
            elif circuitLine.fixedGate == FixedGateEnum.CSWAP:
                # CSWAP: gate cswap a,b,c {
                # cx c,b;
                self._unrollRecursively(CX._toPB(c, b), circuitOut)
                # ccx a,b,c;
                self._unrollRecursively(CCX._toPB(a, b, c), circuitOut)
                # cx c,b;
                self._unrollRecursively(CX._toPB(c, b), circuitOut)
                # }
                return
        elif circuitLine.HasField('rotationGate'):
            gateName = RotationGateEnum.Name(circuitLine.rotationGate)
            if len(self.sourceGatesNames) > 0 and gateName not in self.sourceGatesNames:
                # don't need unroll: copy
                circuitOut.append(circuitLine)
                return
            elif gateName in self.targetGatesNames:
                # already supported by target machine: copy or expand+copy
                if circuitLine.rotationGate == RotationGateEnum.U:
                    # U3: gate u3(theta,phi,lamda) a { U(theta,phi,lamda) a; }
                    # U2: gate u2(theta,phi) a { U(pi/2,theta,phi) a; }
                    # U1: gate u1(theta) a { U(0,0,theta) a; }
                    circuitOut.append(
                        U(*self._expandAnglesInUGate(circuitLine.paramValues))._toPB(a))
                    return
                else:
                    circuitOut.append(circuitLine)
                    return
            elif circuitLine.rotationGate == RotationGateEnum.RX:
                # RX: gate rx(theta) a { u3(theta, -pi/2,pi/2) a; }
                self._unrollRecursively(U(theta, -np.pi / 2.0, np.pi / 2.0)._toPB(a), circuitOut)
                return
            elif circuitLine.rotationGate == RotationGateEnum.RY:
                # RY: gate ry(theta) a { u3(theta,0,0) a; }
                self._unrollRecursively(U(theta, 0.0, 0.0)._toPB(a), circuitOut)
                return
            elif circuitLine.rotationGate == RotationGateEnum.RZ:
                # RZ: gate rz(theta) a { u1(theta) a; }
                self._unrollRecursively(U(theta)._toPB(a), circuitOut)
                return
            elif circuitLine.rotationGate == RotationGateEnum.CU:
                # CU: gate cu3(theta,phi,lamda) a, b
                # {
                #   u1((lamda+phi)/2) a;
                self._unrollRecursively(U((lamda + phi) / 2.0)._toPB(a), circuitOut)
                #   u1((lamda-phi)/2) b;
                self._unrollRecursively(U((lamda - phi) / 2.0)._toPB(b), circuitOut)
                #   cx a, b;
                self._unrollRecursively(CX._toPB(a, b), circuitOut)
                #   u3(-theta/2,0,-(phi+lamda)/2) b;
                self._unrollRecursively(U(-theta / 2.0, 0.0, -(phi + lamda) / 2.0)._toPB(b), circuitOut)
                #   cx a, b;
                self._unrollRecursively(CX._toPB(a, b), circuitOut)
                #   u3(theta/2,phi,0) b;
                self._unrollRecursively(U(theta / 2.0, phi, 0.0)._toPB(b), circuitOut)
                # }
                return
            elif circuitLine.rotationGate == RotationGateEnum.CRX:
                # CRX:
                # gate crx(theta) a,b
                # {
                #   u1(pi/2) b;
                self._unrollRecursively(U(np.pi / 2.0)._toPB(b), circuitOut)
                #   cx a,b;
                self._unrollRecursively(CX._toPB(a, b), circuitOut)
                #   u3(-theta/2,0,0) b;
                self._unrollRecursively(U(-theta / 2.0, 0.0, 0.0)._toPB(b), circuitOut)
                #   cx a,b;
                self._unrollRecursively(CX._toPB(a, b), circuitOut)
                #   u3(theta/2,-pi/2,0) b;
                self._unrollRecursively(U(theta / 2.0, -np.pi / 2.0, 0.0)._toPB(b), circuitOut)
                # }
                return
            elif circuitLine.rotationGate == RotationGateEnum.CRY:
                # CRY:
                # gate cry(theta) a,b
                # {
                #   u3(theta/2,0,0) b;
                self._unrollRecursively(U(theta / 2.0, 0.0, 0.0)._toPB(b), circuitOut)
                #   cx a,b;
                self._unrollRecursively(CX._toPB(a, b), circuitOut)
                #   u3(-theta/2,0,0) b;
                self._unrollRecursively(U(-theta / 2.0, 0.0, 0.0)._toPB(b), circuitOut)
                #   cx a,b;
                self._unrollRecursively(CX._toPB(a, b), circuitOut)
                # }
                return
            elif circuitLine.rotationGate == RotationGateEnum.CRZ:
                # CRZ:
                # gate crz(theta) a,b
                # {
                #   u1(theta/2) b;
                self._unrollRecursively(U(theta / 2.0)._toPB(b), circuitOut)
                #   cx a,b;
                self._unrollRecursively(CX._toPB(a, b), circuitOut)
                #   u1(-theta/2) b;
                self._unrollRecursively(U(-theta / 2.0)._toPB(b), circuitOut)
                #   cx a,b;
                self._unrollRecursively(CX._toPB(a, b), circuitOut)
                # }
                return
        elif circuitLine.HasField('procedureName'):
            # procedureName: copy
            circuitOut.append(circuitLine)
            return
        elif circuitLine.HasField('barrier'):
            # barrier: copy
            circuitOut.append(circuitLine)
            return
        elif circuitLine.HasField('measure'):
            # measure: copy
            circuitOut.append(circuitLine)
            return

        # unsupported gate
        if self.errorOnUnsupported:
            # error
            raise Exception('unsupported gate:\n' + str(circuitLine))
        else:
            # ignore
            circuitOut.append(circuitLine)

    def _decompose(self, circuitOut, circuitIn):
        """
        Unroll the gates

        :param circuitOut: input circuit
        :param circuitIn: output circuit
        """

        for circuitLine in circuitIn:
            self._unrollRecursively(circuitLine, circuitOut)
