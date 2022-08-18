#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Baidu, Inc. All Rights Reserved.
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

r"""
An example of satellite-to-ground QKD simulation.

See reference:
Liao, Sheng-Kai, et al.
"Satellite-to-ground quantum key distribution."
Nature 549.7670 (2017): 43-47.
"""

import sys
sys.path.append('..')

import pandas as pd
from qcompute_qnet.functionalities.mobility import Track
from qcompute_qnet.core.des import DESEnv
from qcompute_qnet.topology.network import Network
from qcompute_qnet.models.qkd.node import QKDSatellite, QKDNode
from qcompute_qnet.models.qkd.key_generation import PrepareAndMeasure
from qcompute_qnet.topology.link import Link
from qcompute_qnet.devices.channel import ClassicalFreeSpaceChannel, QuantumFreeSpaceChannel


# Load data of the Micius satellite
micius_track = pd.read_csv("data/micius2017_track.csv")
micius_link_efficiency = pd.read_csv("data/micius2017_link_efficiency.csv")


# Set orbit track model of the Micius satellite
class MiciusOrbit(Track):
    def __init__(self, ref_node, ref_time=0):
        super().__init__(ref_node, ref_time)

    def time2distance(self, current_time: int) -> float:
        micius_time = round((current_time - self.ref_time) * 1e-12, 1)

        global micius_track
        index = micius_track.loc[micius_track['time'] == micius_time].index[0]

        return micius_track.loc[index].distance

    def distance2loss(self, distance: float) -> float:
        distance = round(distance, 1)

        global micius_link_efficiency
        index = micius_link_efficiency.loc[micius_link_efficiency['distance'] == distance].index[0]

        return micius_link_efficiency.loc[index].loss


def micius(start_time):
    # Create a simulation environment
    env = DESEnv("Micius Satellite Experiment Simulation", default=True)

    # Create the satellite-to-ground QKD network
    network = Network("Satellite-to-ground QKD Network")

    # Create QKD nodes
    micius = QKDSatellite("Micius")
    xinglong = QKDNode("Xinglong")

    # Set orbit track of the Micius satellite
    micius.mobility.set_track(MiciusOrbit(ref_node=xinglong, ref_time=-round(start_time)))

    # Set parameters of photon source and polar detector
    source_options = {"frequency": 100e6, "wavelength": 848.62, "bandwidth": 0.1}
    detector_options = {"efficiency": 0.5}
    micius.photon_source.set(**source_options)
    xinglong.polar_detector.set_detectors(**detector_options)

    # Set parameters for the key generation protocol
    intensities = {"prob": [0.5, 0.25, 0.25], "mean_photon_num": [0.8, 0.1, 0]}
    transmitter_options = {"protocol": "DecoyBB84", "tx_bases_ratio": [0.5, 0.5], "intensities": intensities}
    receiver_options = {"protocol": "DecoyBB84", "rx_bases_ratio": [0.5, 0.5]}

    # Set up decoy-state BB84 protocol
    decoy_bb84_micius = micius.set_key_generation(xinglong, **transmitter_options)
    micius.protocol_stack.build(decoy_bb84_micius)
    decoy_bb84_xinglong = xinglong.set_key_generation(micius, **receiver_options)
    xinglong.protocol_stack.build(decoy_bb84_xinglong)

    # Create the link between the Micius satellite and the ground station, connect both nodes
    link_micius_xinglong = Link("Micius_Xinglong", ends=(micius, xinglong))

    # Create communication channels and connect the nodes
    cchannel1 = ClassicalFreeSpaceChannel("c_Micius2Xinglong", sender=micius, receiver=xinglong, is_mobile=True)
    cchannel2 = ClassicalFreeSpaceChannel("c_Xinglong2Micius", sender=xinglong, receiver=micius, is_mobile=True)
    qchannel = QuantumFreeSpaceChannel("q_Micius2Xinglong", sender=micius, receiver=xinglong, is_mobile=True)

    # Install the channels to the link
    link_micius_xinglong.install([cchannel1, cchannel2, qchannel])

    # Install the nodes and the link to the network
    network.install([micius, xinglong, link_micius_xinglong])

    # Start the protocol stack
    micius.protocol_stack.start(role=PrepareAndMeasure.Role.TRANSMITTER, key_num=float("inf"), key_length=256)
    xinglong.protocol_stack.start(role=PrepareAndMeasure.Role.RECEIVER, key_num=float("inf"), key_length=256)

    # Initialize the simulation environment
    env.init()

    # Run the simulation environment and save the log records
    env.set_log(level="INFO")
    env.run(end_time=1e11, logging=True)

    # Estimate the sifted key rate
    return decoy_bb84_micius.key_rate_estimation()


# Start the simulation after 100s since the satellite starts moving
key_rate = micius(100e12)
print(f"Sifted key rate: {key_rate:.4f} kbit/s")
