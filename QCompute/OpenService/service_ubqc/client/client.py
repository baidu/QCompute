# Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# !/usr/bin/env python3

"""
client
"""

FileErrorCode = 3

from typing import List, Tuple, Dict
from numpy import random, pi

import json
import traceback
import threading
import websocket
from queue import Queue
from base64 import b64decode, b64encode
from websocket._app import WebSocketApp
from tqdm import tqdm

from QCompute.Define import blindCompAddr
from QCompute.QPlatform import Error
from QCompute.QPlatform.QTask import QTask
from QCompute.QPlatform.Utilities import numpyMatrixToProtobufMatrix
from QCompute.QProtobuf import PBCircuitLine, PBFixedGate, PBRotationGate, PBMeasure, PBProgram
from QCompute.QProtobuf import PBUbpcInitState, PBEncryptedMeasureReq, PBEncryptedMeasureRes

from QCompute.OpenService import ModuleErrorCode
from QCompute.OpenService.service_ubqc.client.qobject import Circuit
from QCompute.OpenService.service_ubqc.client.transpiler import transpile_to_brickwork
from QCompute.OpenService.service_ubqc.client.utils import plus_state, rotation_gate, decompose, u3_gate, matmul, eps

__all__ = [
    "BrickworkVertex",
    "UbqcClient",
    "PlatformClient",
]


class BrickworkVertex:
    r"""Define the ``BrickworkVertex`` class.

    This class describes the vertices on the brickwork graph in UBQC.
    It stores parameters including positions, rotation encryption angles, flipping encryption angles, commands,
    and measurement outcomes of every vertices on the graph. These parameters are crucial to the calculations.

    Attributes:
        position (Tuple): vertex position
    """

    def __init__(self, position):
        r"""``BrickworkVertex`` constructor, used to instantiate a ``BrickworkVertex`` object.

        This class describes the vertices on the brickwork graph in UBQC.
        It stores parameters including positions, rotation encryption angles, flipping encryption angles, commands,
        and measurement outcomes of every vertices on the graph. These parameters are crucial to the calculations.

        Args:
            position (Tuple): vertex position
        """
        if not isinstance(position, Tuple):
            raise Error.ArgumentError(f'Invalid position ({position}) with the type: `{type(position)}`!\nOnly `Tuple` is supported as the type of vertex position.', ModuleErrorCode, FileErrorCode, 1)

        self.__position = position  # vertex position
        self.__rotation_encryption_angle: float = None  # angle for rotation encryption
        self.__flipping_encryption_angletype: float = None  # angle for flipping encryption
        self.__commands: List[str] = []  # commands to be executed
        self.__outcome = None  # measurement outcome

    def set_rotation_encryption_angle(self, angle: float = 0.0):
        r"""Set the rotation encryption angle.

        Args:
            angle (float, optional): rotation encryption angle with the range of
                                     [0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4]
        """
        if not isinstance(angle, float):
            raise Error.ArgumentError(f'Invalid rotation encryption angle ({angle}) with the type: `{type(angle)}`!\nOnly `float` is supported as the type of rotation encryption angle.', ModuleErrorCode, FileErrorCode, 2)

        if all(abs(angle - quarter_pi) >= eps for quarter_pi in
               [0, pi / 4, pi / 2, 3 * pi / 4, pi, 5 * pi / 4, 3 * pi / 2, 7 * pi / 4]):
            raise Error.ArgumentError(f"Invalid rotation encryption angle: ({angle})!\nOnly '0', 'pi/4', 'pi/2', '3pi/4', 'pi', '5pi/4', '3pi/2' and '7pi/4' are supported as the rotation encryption angle.", ModuleErrorCode, FileErrorCode, 3)

        self.__rotation_encryption_angle = angle

    def get_rotation_encryption_angle(self):
        r"""Get the rotation encryption angle.

        Returns:
            float: rotation encryption angle with the range of
                   [0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4]
        """
        return self.__rotation_encryption_angle

    def set_flipping_encryption_angle(self, angle: float = 0.0):
        r"""Set the flipping encryption angle.

        Args:
            angle (float, optional): flipping encryption angle with the range of [0, pi]
        """
        if not isinstance(angle, float):
            raise Error.ArgumentError(f'Invalid flipping encryption angle ({angle}) with the type: `{type(angle)}`!\nOnly `float` is supported as the type of flipping encryption angle.', ModuleErrorCode, FileErrorCode, 4)

        if all(abs(angle - zero_or_pi) >= eps for zero_or_pi in [0, pi]):
            raise Error.ArgumentError(f"Invalid flipping encryption angle: ({angle})!\nOnly '0' and 'pi' are supported as the flipping encryption angle.", ModuleErrorCode, FileErrorCode, 5)

        self.__flipping_encryption_angle = angle

    def get_flipping_encryption_angle(self):
        r"""Get the flipping encryption angle.

        Returns:
            float: flipping encryption angle with the range of [0, pi]
        """
        return self.__flipping_encryption_angle

    def record_commands(self, which_command):
        r"""Record the commands executed on this vertex.

        Args:
            which_command (Pattern.CommandM / Pattern.CommandX / Pattern.CommandZ / Pattern.CommandS): commands
        """
        name: str = which_command.name
        if not {name}.issubset(['M', 'X', 'Z', 'S']):
            raise Error.ArgumentError(f"Invalid command name: ({name})!\nOnly 'M', 'X', 'Z' and 'S' are supported as the names of operation commands on vertices, where 'M' is the measurement command, 'X' is the X byproduct correction command, 'Z' is the Z byproduct correction command and 'S' is the signal shifting command.", ModuleErrorCode, FileErrorCode, 6)

        self.__commands.append(which_command)

    def get_commands(self):
        r"""Get the commands executed on this vertex.

        Returns:
            List: command list
        """
        return self.__commands

    def record_outcome(self, outcome: int):
        r"""Record the measurement outcome of this vertex.

        Note:
            This method obtains the primitive measurement outcome.
            Then it decrypts the outcome according to the flipping encryption angle.

        Args:
            outcome (int): primitive measurement outcome of this vertex
        """
        self.__outcome = outcome if abs(self.__flipping_encryption_angle - 0) < eps else (outcome + 1) % 2

    def get_outcome(self):
        r"""Get the decrypted measurement outcome of this vertex.

        Returns:
            int: decrypted measurement outcome of this vertex
        """
        return self.__outcome


class PlatformClient:
    r"""Define the ``PlatformClient`` class.

    This class contains methods for the client to communicate with server according to the UBQC protocol.
    In theory, quantum states are transmitted through the quantum channel.
    And classical messages are transmitted through the classical channel.
    However here, we utilized this UBQC backend to simulate the protocol totally from a classical perspective.
    Therefore, all messages in UBQC are transmitted through the platform and channels defined in this class.

    Attributes:
        shots (int): number of the sample times
        program (PBProgram): program of quantum circuit to be executed
    """

    def __init__(self, shots: int, program: PBProgram):
        r"""``PlatformClient`` constructor, used to instantiate a ``PlatformClient`` object.

        This class contains methods for the client to communicate with server according to the UBQC protocol.
        In theory, quantum states are transmitted through the quantum channel.
        And classical messages are transmitted through the classical channel.
        However here, we utilized this UBQC backend to simulate the protocol totally from a classical perspective.
        Therefore, all messages in UBQC are transmitted through the platform and channels defined in this class.

        Args:
            shots (int): number of the sample times
            program (PBProgram): program of quantum circuit to be executed
        """
        self.ws_opened = threading.Event()
        self.__shots = shots
        self.__shot_finished = [False for _ in range(self.__shots)]
        self.__shot_inboxes = [Queue() for _ in range(self.__shots)]

        self.__bar = None
        self.__ws_conn = None

        # Queue is thread-safe so that it can be shared
        self.shot_outbox = Queue()
        self.ret: Dict[str, int] = {}
        self.shot_threadings: Dict[int, threading.Thread] = {}
        self.program = program
        self.phase_auth = True  # whether or not in the testing process
        self.entry = None
        self.secret = None
        self.failed = True

        width, depth = self.__extract_graph_shape()

        ok = self.wait_for_server_ready({
            "width": width,
            "depth": depth,
            "shots": shots,
        })
        if not ok:
            raise Error.ArgumentError('Failed to acquire computing resource!', ModuleErrorCode, FileErrorCode, 7)

        self.wsapp = websocket.WebSocketApp(
            self.entry,  # just for local tests
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open,
        )
        self.dup = threading.Thread(target=self.on_queue)
        self.dup.start()
        self.wsapp.run_forever()

    def __extract_graph_shape(self):
        r"""Extract the shape (width and depth) of the brickwork graph.

        Note:
            This is an intrinsic method. No need to call it externally.

        Returns:
            int (width): brickwork graph width
            int (depth): brickwork graph depth
        """
        circuit = UbqcClient.pb_to_circuit(self.program)
        bw_pattern = transpile_to_brickwork(circuit)

        # Get shape
        input_ = bw_pattern.input_
        c_out, q_out = bw_pattern.output_
        output_ = c_out + q_out
        width = len(input_)
        depth = output_[0][1] - input_[0][1] + 1

        return width, depth

    def on_open(self, ws):
        r"""Open the 'websocket' connection.

        Args:
            ws (WebSocketApp): 'websocket' connection
        """
        self.__bar = tqdm(range(0, self.__shots), desc='shots computing...', unit=' shot')
        self.__ws_conn = ws
        self.ws_opened.set()

    def on_phase_auth_message(self, buf):
        r"""Prepare to authenticate the message.

        Args:
            buf (bytes): message to be authenticated
        """
        msg = self.__nanojsonrpc_unpack(buf)
        method = msg['method']
        if method == 'auth':
            self.__ws_conn.send(self.__nanojsonrpc_pack('auth', [self.secret]))
        elif method == 'ready':
            self.phase_auth = False
        else:
            self.wsapp.keep_running = False
            raise Error.LogicError(f'Unexpected message: ({msg})!', ModuleErrorCode, FileErrorCode, 8)

    def on_message(self, ws, buf):
        r"""Prepare to transmit the message.

        Args:
            ws (WebSocketApp): websocket connection
            buf (bytes): message to be transmitted
        """
        if self.phase_auth:
            self.on_phase_auth_message(buf)
            return

        msg = self.__nanojsonrpc_unpack(buf)
        shot = msg['shot']

        if shot < 0:
            # Authenticate
            self.__ws_conn.send(self.__nanojsonrpc_pack('auth'))

        elif self.__shot_finished[shot]:
            # End
            pass

        elif shot in self.shot_threadings:
            # Forward
            self.__shot_inboxes[shot].put(msg)

        else:
            # If the shot does not exist, a 'worker' will be established.
            worker = UbqcClient(
                shot,
                self.__shot_inboxes[shot],
                self.shot_outbox,
                self.program,
            )
            worker.daemon = True

            self.shot_threadings[shot] = worker
            worker.start()
            self.__shot_inboxes[shot].put(msg)

    def on_queue(self):
        r"""Forward the message in a queue to the websocket connection.
        """
        self.ws_opened.wait()

        while self.wsapp.keep_running:
            try:
                msg = self.shot_outbox.get(timeout=0.001)
            except:
                continue
            action = msg['action']
            payload = msg['payload']

            if action == 'remote':
                # Choose the remote server
                buf = json.dumps(payload)
                self.__ws_conn.send(f"{buf}\n")
            elif action == 'local':
                # Choose the local server
                result = payload['params']['result']
                shot = payload['shot']
                prev = self.ret.get(result, 0)
                self.ret[result] = prev + 1

                del self.shot_threadings[shot]
                self.__shot_finished[shot] = True
                self.__bar.update(1)
                if all(self.__shot_finished):
                    # All shots are completed
                    self.failed = False
                    self.wsapp.keep_running = False
                    break

    def on_error(self, ws, error):
        r"""Error.

        Args:
            error (str): error message
        """
        self.wsapp.keep_running = False
        if not self.ws_opened.is_set():
            self.ws_opened.set()
        raise Error.Error(f'Unexpected Error: ({error})!', ModuleErrorCode, FileErrorCode, 9)

    def on_close(self, ws, code, reason):
        r"""Close the websocket connection.

        Args:
            code (str): the last message to be transmitted (if it exists)
            reason (str): reason to close the connection
        """
        self.wsapp.keep_running = False
        if not self.ws_opened.is_set():
            self.ws_opened.set()

    def __nanojsonrpc_pack(self, method, params=None):
        r"""Pack the 'jsonrpc'.

        Note:
            This is an intrinsic method. No need to call it externally.

        Args:
            method (str): packing method name
            params (List / Dict): other parameters
        """
        pack = {'method': method}

        if params is not None:
            pack['params'] = params

        return json.dumps(pack)

    def __nanojsonrpc_unpack(self, msg):
        r"""Unpack the 'jsonrpc'.

        If it returns 'None', it means that this package is not a valid 'nanojsonrpc'.

        Note:
            This is an intrinsic method. No need to call it externally.

        Args:
            msg (str): the message of this package
        """
        try:
            pack = json.loads(msg)
            if 'method' not in pack:
                return None
            else:
                return pack
        except:
            traceback.print_exc()
            return None

    def wait_for_server_ready(self, params):
        r"""Build the computational task and wait until the server is ready.

        Note:
            A computational task is built on the client end.
            The server will return an ID number and an authentication number.
            They are crucial to establish a websocket connection between the client and the server.
            The ports are polling in turns until a computational resource is available for the task or the task is error.

        Args:
            params (Dict): a dictionary that can be serialized to 'json' strings

        Returns:
            bool: True if it succeeded to acquire a computational resource;
                  False if it failed to acquire a computational resource
        """
        hub_client = QTask()
        hub_client.createBlindTask('ubqc', params)
        ret = hub_client.waitBlindTask(1)
        if ret is None:
            return False

        path, secret = ret
        self.entry = f"{blindCompAddr}/{path}"
        self.secret = secret

        return True


class UbqcClient(threading.Thread):
    r"""Define the ``UbqcClient`` class.

    Note:
        Each shot of the task is executed in a single queue.
        The queue is established by the ``PlatformClient`` class.
        It is also used to transmit messages to the server.

    Attributes:
        shots (int): number of the sample times
        inbox (Queue): inbox queue
        outbox (Queue): outbox queue
        program (PBProgram): program of quantum circuit to be executed
    """

    def __init__(self, shots, inbox, outbox, program):
        r"""``UbqcClient`` constructor, used to instantiate a ``UbqcClient`` object.

        Note:
            Each shot of the task is executed in a single queue.
            The queue is established by the ``PlatformClient`` class.
            It is also used to transmit messages to the server.

        Args:
            shots (int): number of the sample times
            inbox (Queue): inbox queue
            outbox (Queue): outbox queue
            program (PBProgram): program of quantum circuit to be executed
        """
        self.__circuit = None  # circuit
        self.__client_knowledge = {}  # client knowledge
        self.__bw_pattern = None  # brickwork pattern
        self.__width: int = None  # width
        self.__depth: int = None  # depth
        self.__shots: int = shots  # shot
        self.__states = None
        self.inbox: Queue = inbox
        self.outbox: Queue = outbox
        self.program = program
        super().__init__()

    def send_back(self, action, payload):
        r"""Send back the action and payload messages.

        Args:
            action (str): action name
            payload (Dict): queue payload
        """
        self.outbox.put({
            'action': action,
            'payload': payload
        })

    def wait_for_messages(self):
        r"""Wait for the incoming messages.

        Returns:
            bytes: incoming messages
        """
        msg = self.inbox.get()
        return msg

    def __wrap_shot_message(self, method, params=None):
        r"""Wrap the 'jsonrpc'.

        Note:
            This is an intrinsic method. No need to call it externally.

        Args:
            method (str): packing method name
            params (List / Dict): other parameters

        Returns:
            Dict: packed messages
        """
        pack = {'method': method, 'shot': self.__shots}

        if params is not None:
            pack['params'] = params

        return pack

    def send_shape_and_states(self):
        r"""Send the shape and states to the server.

        Send the brickwork graph shape (width and depth) and the prepared single qubit states
        to the server as initializations.

        Note:
            This method corresponds to the first round of the UBQC protocol.
            Please see the reference [arXiv:0807.4154] for more details.
            It is confirmed that the current codes are executed following the UBQC protocol.
        """
        self.commit()

        _init_ = PBUbpcInitState()
        _init_.width = self.__width
        _init_.depth = self.__depth

        for vec in self.__states:
            vec_buf = numpyMatrixToProtobufMatrix(vec)
            _init_.vector.append(vec_buf)

        init_stateBuf: bytes = _init_.SerializeToString()

        self.send_back(
            'remote',
            self.__wrap_shot_message(
                'setInitState',
                {'state': b64encode(init_stateBuf).decode()}
            )
        )

    def send_angle_bulks(self, col):
        r"""Send the angles in bulks during each sample.

        These bulks are set according to the circuit width.
        After sending the angles, this method is waiting to receive the measurement outcomes from the server.

        Note:
            This method corresponds to the second round of the UBQC protocol.
            Please see the reference [arXiv:0807.4154] for more details.
            It is confirmed that the current codes are executed following the UBQC protocol.

        Warning:
            In this method, a trick in code execution is implemented to improve the running efficiency.
            The trick is that the vertices tagged with the same column number are measured at the same time.
            This trick takes the advantage of the parallel operations in MBQC model.
            It does not lose the accuracy of the computational result.
            But the message transmission frequency is reduced.

        Args:
            col (int): column tag of the vertices to be measured
        """
        # The client calculates the encrypted angles and sends them in columns to the server
        positions = [(row, col) for row in range(self.__width)]
        # To obtain the angles, the client firstly calculates the adaptive angles according to the measurement pattern
        # Then he encrypts the angles with the rotation and flipping methods to protect the privacy
        angles = self.calculate_and_encrypt_angles(positions)

        # Establish a request to send the messages to the server
        req = PBEncryptedMeasureReq()
        for angle in angles:
            req.angles.append(angle)

        # Encode the message by serialization
        reqBuf: bytes = req.SerializeToString()

        self.send_back(
            'remote',
            self.__wrap_shot_message(
                'setAngleBulk',
                {'angle': b64encode(reqBuf).decode()},
            )
        )

        # Receive the measurement outcomes
        measure = self.wait_for_messages()
        if measure is None:
            raise Error.ArgumentError('Failed to receive the measurement outcome!', ModuleErrorCode, FileErrorCode, 10)

        serverResBuf = b64decode(measure['params']['outcome'])

        # Decode the serialized message
        res = PBEncryptedMeasureRes()
        res.ParseFromString(serverResBuf)
        self.decrypt_outcomes(positions, res.results)

    def run(self):
        r"""Run the quantum circuit following UBQC protocol.

        Note:
            This method corresponds to the whole UBQC protocol.
            Please see the reference [arXiv:0807.4154] for more details.
            It is confirmed that the current codes are executed following the UBQC protocol.
        """
        # Wait for the 'shot' message ready
        self.wait_for_messages()
        # Send the initial states to the server
        self.send_shape_and_states()
        # Wait for the 'method' message ready
        self.wait_for_messages()

        # Send the measurement angles to the server
        for y in range(self.__depth):
            self.send_angle_bulks(y)

        # Obtain the measurement outcomes
        result = self.get_classical_output()[::-1]
        self.send_back(
            'local',
            self.__wrap_shot_message(
                'setResult',
                {'result': result, 'shot': self.__shots},
            )
        )

    def commit(self):
        r"""Commit the quantum circuit program.

        The client commits the program and send shape and states to the server as initializations.

        Note:
            This is an integrated method called by the user.
            It can publish and submit the quantum circuit task.
            To run the program, choose the UBQC backend before calling this method.

        Returns:
            int (width): brickwork graph width
            int (depth): brickwork graph depth
            List: a list containing all single qubit quantum states of vertices on the graph.
                  It has the form of ``List[ndarray]``
        """
        # Translate ``PBProgram`` into ``Circuit``
        self.__circuit = self.pb_to_circuit(self.program)

        # Extract the graph shape and measurement pattern
        self.__map_to_graph()

        # Randomly generate a quantum state for each vertex on the graph
        self.__states = self.prepare_states()

        return self.__width, self.__depth, self.__states

    @staticmethod
    def pb_to_circuit(program: PBProgram):
        r"""Translate ``PBProgram`` into the ``Circuit`` class.

        Args:
            program (PBProgram): the published quantum program

        Returns:
            Circuit: a quantum circuit which supports the translation to its equivalent MBQC model
        """
        pbCircuit = program.body.circuit

        # Obtain the circuit width
        bit_idxes = set()
        for gate in pbCircuit:
            bit_idxes.update(gate.qRegList)
        width = max(bit_idxes) + 1
        if width <= 0:
            raise Error.ArgumentError(f'Invalid circuit ({program}) in the program!\nThis circuit is empty and has no qubit.', ModuleErrorCode, FileErrorCode, 11)

        # Instantiate ``Circuit`` and map gates and measurements to methods in ``Circuit``
        circuit = Circuit(width)

        # Warning: In the circuit model, the quantum states are initialized with |0>.
        # While in MBQC model, the quantum states are initialized with |+>.
        # Therefore, each qubit in the MBQC circuit should be operated by a ``Hadamard`` gate in the front.
        for i in range(width):
            circuit.h(i)

        for PBGate in pbCircuit:
            op = PBGate.WhichOneof('op')
            # Map ``fixedGate`` (including 'H', 'CX', 'X', 'Y', 'Z', 'S', 'T', 'CZ') to the methods in ``Circuit``
            if op == 'fixedGate':
                fixedGate: PBFixedGate = PBGate.fixedGate
                gateName = PBFixedGate.Name(fixedGate)
                bit_idx = PBGate.qRegList
                if gateName == 'H':
                    circuit.h(bit_idx[0])
                elif gateName == 'CX':
                    circuit.cnot(bit_idx)
                elif gateName == 'X':
                    circuit.x(bit_idx[0])
                elif gateName == 'Y':
                    circuit.y(bit_idx[0])
                elif gateName == 'Z':
                    circuit.z(bit_idx[0])
                elif gateName == 'S':
                    circuit.s(bit_idx[0])
                elif gateName == 'T':
                    circuit.t(bit_idx[0])
                elif gateName == 'CZ':
                    # CZ [q1, q2] = H [q2] + CNOT [q1, q2] + H [q2]
                    circuit.h(bit_idx[1])
                    circuit.cnot(bit_idx)
                    circuit.h(bit_idx[1])
                else:
                    raise Error.ArgumentError(f"Invalid gate: ({gateName})!\nOnly 'H', 'CX', 'X', 'Y', 'Z', 'S', 'T', 'CZ' are supported as the fixed gates in UBQC in this version.", ModuleErrorCode, FileErrorCode, 12)

            # Map ``rotationGate`` (including 'RX', 'RY', 'RZ', 'U') to the methods in ``Circuit``
            elif op == 'rotationGate':
                rotationGate: PBRotationGate = PBGate.rotationGate
                gateName = PBRotationGate.Name(rotationGate)
                bit_idx = PBGate.qRegList

                if gateName == 'RX':
                    circuit.rx(PBGate.argumentValueList[0], bit_idx[0])
                elif gateName == 'RY':
                    circuit.ry(PBGate.argumentValueList[0], bit_idx[0])
                elif gateName == 'RZ':
                    circuit.rz(PBGate.argumentValueList[0], bit_idx[0])

                # Warning: unitary gate in MBQC has a decomposition form different from the commonly used ``U3`` gate!
                elif gateName == 'U':
                    # In circuit model, the ``U3`` gate has a decomposition form of "Rz Ry Rz",
                    # with angles of "theta, phi, lamda", that is:
                    # U3(theta, phi, lamda) = Rz(phi) Ry(theta) Rz(lamda)
                    angles = PBGate.argumentValueList

                    # Warning: Sometimes, The angles have only one or two valid parameters!
                    # In these cases, set the other parameters to be zeros
                    if len(angles) == 1:
                        theta1 = angles[0]
                        phi1 = 0
                        lamda1 = 0
                    elif len(angles) == 2:
                        theta1 = angles[0]
                        phi1 = angles[1]
                        lamda1 = 0
                    else:
                        theta1 = angles[0]
                        phi1 = angles[1]
                        lamda1 = angles[2]
                    u3 = u3_gate(theta1, phi1, lamda1)

                    # In MBQC model, the unitary gate has a decomposition form of "Rz Rx Rz",
                    # with angles of "theta, phi, lamda", that is:
                    # U(theta, phi, lamda) = Rz(phi) Rx(theta) Rz(lamda)
                    theta2, phi2, lamda2 = decompose(u3)

                    circuit.u(theta2, phi2, lamda2, bit_idx[0])

            elif op == 'customizedGate':
                raise Error.ArgumentError(f'Invalid gate type: ({op})!\nCustomized gates are not supported in UBQC in this version.', ModuleErrorCode, FileErrorCode, 13)

            elif op == 'measure':
                measurement_qubits = set(PBGate.qRegList)
                if measurement_qubits != set(range(width)):
                    raise Error.ArgumentError(f'Invalid measurement qubits: ({measurement_qubits})!\nAll qubits must be measured in UBQC in this version.', ModuleErrorCode, FileErrorCode, 14)

                for qReg in PBGate.qRegList:
                    typeName: PBMeasure = PBMeasure.Type.Name(PBGate.measure.type)
                    if typeName == 'Z':
                        circuit.measure(qReg)
                    else:
                        raise Error.ArgumentError(f"Invalid measurement type: ({typeName})!\nOnly 'Z measurement' is supported as the measurement type in UBQC in this version.", ModuleErrorCode, FileErrorCode, 15)

            else:
                raise Error.ArgumentError(f'Invalid operation: ({op})!\nThis operation is not supported in UBQC in this version.', ModuleErrorCode, FileErrorCode, 16)

        return circuit

    def __map_to_graph(self):
        r"""Map the measurement pattern to the brickwork graph.

        Note:
            This is an intrinsic method. No need to call it externally.
        """
        self.__bw_pattern = transpile_to_brickwork(self.__circuit)

        # Get the shape
        input_ = self.__bw_pattern.input_
        c_out, q_out = self.__bw_pattern.output_
        output_ = c_out + q_out
        self.__width = len(input_)
        self.__depth = output_[0][1] - input_[0][1] + 1

        # Initialize brickwork graph in a dictionary whose keys are positions and values are brickwork vertices
        self.__client_knowledge = {(row, col): BrickworkVertex((row, col))
                                   for row in range(self.__width) for col in range(self.__depth)}

        # Record commands information on the graph
        for cmd in self.__bw_pattern.commands:
            # Store measurement, correction and signal shifting commands information on vertices
            if {cmd.name}.issubset(['M', 'X', 'Z', 'S']):
                pos = cmd.which_qubit
                vertex = self.__client_knowledge[pos]
                vertex.record_commands(cmd)
            else:
                continue

        # There should be no vertex left with no command on it
        for vertex in self.__client_knowledge.values():
            cmds = vertex.get_commands()
            if len(cmds) == 0:
                raise Error.ArgumentError('Invalid brickwork pattern! This brickwork pattern has not been standardized yet.', ModuleErrorCode, FileErrorCode, 17)

    def prepare_states(self):
        r"""Prepare the single qubit quantum states.

        Randomly generate a single qubit quantum state on each vertex.
        The rotation angle encryption method is implemented in this step.
        Record the corresponding rotation encryption angles for later calculations.

        .. math::

            \text{This method is used to generate the single qubit quantum states in UBQC. }
            \text{The rotation encryption method is implemented with angles } \alpha \text{ randomly chosen from }
            \{0, \pi / 4, \pi / 2, .... 7 \pi / 4\}
            \text{. The corresponding single qubit states are:}
            |\psi\rangle = Rz(\alpha) |+\rangle = 1 / \sqrt{2} (|0\rangle + e^{i \alpha} |1\rangle)

        Warning:
            In the UBQC protocol, the information on each vertex is protected from being revealed to the server.
            In other words, the server is incapable to obtain the rotation encryption angles nor the quantum states.
            However, in the realm of classical simulations, the server requires the exact data for calculations.
            So the client's privacy is inevitably known to the server.
            Therefore, the information-theoretic security (i.e. 'blindness') can not be demonstrated here.

        Returns:
            List: a list containing all single qubit quantum states of vertices on the graph.
                  It has the form of ``List[ndarray]``
        """
        # Prepare random states
        multiple_quarter_pi = [(i * pi) / 4 for i in range(8)]
        zero_or_pi = [0, pi]
        states = []
        for pos in self.__client_knowledge.keys():
            rand_ang = random.choice(multiple_quarter_pi).item()
            rand_flip = random.choice(zero_or_pi).item()
            self.__client_knowledge[pos].set_rotation_encryption_angle(rand_ang)
            self.__client_knowledge[pos].set_flipping_encryption_angle(rand_flip)
            states.append(matmul(rotation_gate('z', rand_ang), plus_state()))

        return states

    def calculate_and_encrypt_angles(self, positions: List):
        r"""Calculate and encrypt the angles.

        Calculate measurement angles of the tagged vertices.
        Encode the angles with rotation and flipping encryption method to protect the information.

        Note:
            This is a method called by the client for multiple times.
            The whole process can be manipulated manually by controlling the vertices and
            calculating and encrypting the measurement angles.

        Args:
            positions (List): a list containing a column of the positions of vertices on the graph

        Returns:
            List: a list containing a column of the encrypted measurement angels of vertices on the graph
        """
        angles = []
        for pos in positions:
            vertex = self.__client_knowledge[pos]
            cmds = vertex.get_commands()
            for cmd in cmds:
                if cmd.name == 'M':
                    # Calculate the adaptive angle
                    signal_s = sum([self.__client_knowledge[pos].get_outcome() for pos in cmd.domain_s]) % 2
                    signal_t = sum([self.__client_knowledge[pos].get_outcome() for pos in cmd.domain_t]) % 2
                    adaptive_angle = (-1) ** signal_s * cmd.angle + signal_t * pi

                    # Encrypt each angle with rotation and flipping method
                    encrypted_angle = adaptive_angle + \
                                      vertex.get_rotation_encryption_angle() + \
                                      vertex.get_flipping_encryption_angle()
                    angles.append(encrypted_angle)

                else:
                    continue
        return angles

    def decrypt_outcomes(self, positions: List[Tuple], outcomes: List[int]):
        r"""Decrypt the measurement outcomes and record them on the corresponding vertices.

        Note:
            This is a method called by the client for multiple times.
            It decrypts the measurement outcomes according to the flipping encryption angles.

        Args:
            positions (List): a list containing a column of the positions of vertices on the graph
            outcomes (List): a list containing a column of the measurement outcomes of vertices on the graph
        """
        for idx in range(len(positions)):
            # Decrypt and record the measurement outcome
            pos = positions[idx]
            self.__client_knowledge[pos].record_outcome(outcomes[idx])

    def get_classical_output(self):
        r"""Obtain the result as classical output equivalent to the circuit model.

        Note:
            This is a method called by the client in the end. It returns the result of this task.
            If the input is a quantum circuit, it returns the result as classical output equivalent to the circuit model.
            If the input is a graph, it returns the measurement outcomes of all vertices on the graph.

        Returns:
            str: classical output
        """
        # If the input is a pattern, return the bit string equivalent to the result in circuit model
        if self.__circuit is not None:
            c_out, q_out = self.__bw_pattern.output_

            # Obtain the string
            # Mark the classical outputs with their measurement outcomes and mark quantum outputs with '?'
            vertex_list = [str(self.__client_knowledge[(i, self.__depth - 1)].get_outcome())
                           if (i, self.__depth - 1) in c_out else '?'
                           for i in range(self.__width)]

            bit_str = ''.join(vertex_list)
            return bit_str

        # If the input is a graph, return the whole dictionary as the output
        else:
            bit_dict = {pos: self.__client_knowledge[pos].get_outcome()
                        for pos in self.__client_knowledge.keys()}
            return bit_dict