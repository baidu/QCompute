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
An example of discrete-event simulation.
"""

import sys
sys.path.append('..')

from qcompute_qnet.core.des import DESEnv, EventHandler, Entity


class Customer:

    def __init__(self, service_no, arrival_time):
        self.service_no = service_no  # service number of the customer
        self.arrival_time = arrival_time  # arrival time of the customer


class ServiceCounter(Entity):

    def __init__(self, name, env=None):
        super().__init__(name, env)
        self.status = 0  # status of the clerk, 0 for idle and 1 for busy
        self.service_no = 1  # current service number
        self.queue = []  # queue of the customers
        self.waiting_time = []  # records of the waiting time

    def init(self):
        # The first customer arrives
        self.scheduler.schedule_now(EventHandler(self, "customer_arrive"))

    def customer_arrive(self):
        # Create a customer and append him to the queue, record the service number and arrival time
        self.queue.append(Customer(self.service_no, self.env.now))
        self.env.logger.debug(f"Customer {self.service_no} arrives at {self.env.now}")
        # If the current customer is the first one in the queue and the clerk is idle, begin service
        if len(self.queue) == 1 and self.status == 0:
            self.scheduler.schedule_now(EventHandler(self, "begin_service"))
        self.service_no += 1
        # Schedule the arrival of next customer after a time delay 20
        self.scheduler.schedule_after(20, EventHandler(self, "customer_arrive"))

    def begin_service(self):
        # If the queue is not empty, the clerk begins service for the customers
        if len(self.queue) >= 1:
            # Set the working status as 1 (busy)
            self.status = 1
            # Call the first customer in the queue
            customer = self.queue.pop(0)
            self.env.logger.debug(f"Customer {customer.service_no} served at {self.env.now}")
            # Calculate the waiting time of the customer
            self.waiting_time.append(self.env.now - customer.arrival_time)
            # Schedule the end of the service after a time delay 30
            self.scheduler.schedule_after(30, EventHandler(self, "end_service", [customer]))
        # Else, hold the current status
        else:
            pass

    def end_service(self, customer):
        # End the service and set the status as 0 (idle)
        self.status = 0
        self.env.logger.debug(f"Customer {customer.service_no} leaves at {self.env.now}")
        # Schedule to serve the next customer after a time delay 5
        self.scheduler.schedule_after(5, EventHandler(self, "begin_service"))


# Create a simulation environment and set it to the default
env = DESEnv("Queuing Model Simulation", default=True)

# Create an instance of service counter
counter = ServiceCounter("Counter")

# Initialize the simulation environment
env.init()
# Run the simulation with an end time and turn on the logging
env.run(end_time=3600, logging=True)

# Calculate and print the average waiting time
print("The number of served customers is", len(counter.waiting_time))
print("Average waiting time is:", sum(counter.waiting_time) / len(counter.waiting_time))
