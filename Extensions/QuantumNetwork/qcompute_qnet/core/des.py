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
Module for discrete-event simulation (DES).
"""

import time
from io import StringIO
from abc import ABC, abstractmethod
from typing import Any, List, Union
from enum import Enum, unique
from heapq import heappush, heappop
import numpy as np
import pandas as pd
from qcompute_qnet.core.log import log_init

__all__ = [
    "Entity",
    "Event",
    "EventHandler",
    "Scheduler",
    "DESEnv"
]


class Entity(ABC):
    r"""Class for creating an entity in the discrete-event simulation.

    Attributes:
        name (str): name of the entity
        env (DESEnv): related discrete-event simulation environment
        owner (Entity): owner of the entity
        components (List[Entity]): components that belong to the entity
        scheduler (Scheduler): an event scheduler
        agenda (List[Event]): events scheduled for the entity
        signed_events (List[Event]): events scheduled by the entity
    """

    def __init__(self, name: str, env=None):
        r"""Constructor for Entity class.

        Args:
            name (str): name of the entity
            env (DESEnv): related discrete-event simulation environment
        """
        self.name = name
        if env is None:
            if DESEnv.get_default_env() is None:
                raise ValueError(f"Default DES environment is None. "
                                 f"Should set a default DES environment first"
                                 f"or assign an explicit DES environment to {self.name}.")
            else:
                self.env = DESEnv.get_default_env()
        else:
            self.env = env
        self.env.entities.append(self)
        self.owner = self
        self.components = []
        self.scheduler = Scheduler(self, self.env)
        self.agenda = []
        self.signed_events = []

    @abstractmethod
    def init(self):
        r"""Abstract method that should be implemented by a specific entity.
        """
        pass

    def attach(self, env: "DESEnv") -> None:
        r"""Attach the entity to a discrete-event simulation environment.

        Args:
            env (DESEnv): discrete-event simulation environment to attach
        """
        assert self.env is None, f"The entity is already attached to '{self.env.name}'."
        self.env = env
        self.env.entities.append(self)

    def detach(self) -> None:
        r"""Detach the entity from its discrete-event simulation environment.

        Note:
            Once an entity is detached from a discrete-event simulation environment, all events scheduled for
            the entity will be cancelled.
        """
        self.env.entities.remove(self)
        # Reset the environment and cancel all scheduled events for the entity
        self.env = None
        for event in self.agenda:
            self.scheduler.cancel(event)

    def install(self, entities: Union["Entity", List["Entity"]]) -> None:
        r"""Install a component or a list of components.

        The components installed will be saved in the ``components`` list of the entity.

        Args:
            entities (Union[Entity, List[Entity]]): component entities to install
        """
        if isinstance(entities, Entity):
            self.__install_entity(entities)
        elif isinstance(entities, List) and all(isinstance(entity, Entity) for entity in entities):
            for entity in entities:
                self.__install_entity(entity)
        else:
            raise ValueError("Should input an entity or a list of entities!")

    def __install_entity(self, other: "Entity") -> None:
        r"""Install a component to the entity.

        Args:
            other (Entity): component entity to install
        """
        assert self.env == other.env, "Cannot install an entity with different environment!"
        assert other not in self.components, f"'{other.name}' has already been installed to '{self.name}'."
        other.owner = self
        self.components.append(other)

    def print_components(self) -> None:
        r"""Print the components of the entity.
        """
        df = pd.DataFrame(columns=["type", "name", "env", "owner"])

        for i, component in enumerate(self.components):
            component_info = pd.DataFrame({"type": component.__class__.__name__,
                                           "name": component.name,
                                           "env": component.env.name,
                                           "owner": component.owner.name},
                                          index=[f"Component {i + 1}"])
            df = pd.concat([df, component_info])

        print(f"\nComponent list of the {self.__class__.__name__}, '{self.name}':\n{df.to_string()}")

    def print_agenda(self) -> None:
        r"""Print the events scheduled for the entity.
        """
        df = Event.events_to_dataframe(self.agenda)
        print(f"\nAgenda of {self.name} (unsorted):\n{df.to_string()}")

    def print_signed_events(self) -> None:
        r"""Print the events scheduled by the entity.
        """
        df = Event.events_to_dataframe(self.signed_events)
        print(f"\nSigned events by {self.name} (unsorted):\n{df.to_string()}")


class Event:
    r"""Class for creating an event in the discrete-event simulation.

    An event is scheduled by a scheduler and will be triggered at a given time.

    Attributes:
        time (int): time of the event to be triggered
        handler (EventHandler): handler for processing the event
        signature (Union[Entity, Protocol]): entity or protocol that creates the event
        priority (int): execution priority of the event, smaller value for higher priority
        status (Status): the status of the event

    Note:
        Once two or more events are scheduled at the same time,
        the order of execution is determined by their priorities.
    """

    def __init__(self, time: int, handler: "EventHandler", signature: Union["Entity", "Protocol"], priority=None):
        r"""Constructor for Event class.

        Args:
            time (int): time of the event to be triggered
            handler (EventHandler): handler for processing the event
            signature (Union[Entity, Protocol]): entity or protocol that creates the event
            priority (int): execution priority of the event, smaller value for higher priority
        """
        self.time = time
        self.handler = handler
        self.signature = signature
        self.priority = float("inf") if priority is None else priority
        self.status = Event.Status.CREATED

    @unique
    class Status(Enum):
        r"""Event status.
        """

        CREATED = "Created"
        SCHEDULED = "Scheduled"
        PROCESSED = "Processed"
        CANCELLED = "Cancelled"

    def cancel(self) -> None:
        r"""Set the event status as cancelled.
        """
        self.status = Event.Status.CANCELLED

    @classmethod
    def is_time_valid(cls, time: Any) -> bool:
        r"""Check if the time input is valid.

        Args:
            time (Any): input time

        Returns:
            bool: a bool value to indicate the validity
        """
        return (isinstance(time, int) and time >= 0) or time == float("inf")

    @classmethod
    def is_priority_valid(cls, priority: Any) -> bool:
        r"""Check if a priority input is valid.

        Args:
            priority (Any): input priority

        Returns:
            bool: a bool value to indicate the validity
        """
        return (isinstance(priority, int) and priority >= 0) or priority == float("inf")

    @classmethod
    def events_to_dataframe(cls, events: List["Event"]) -> "pandas.DataFrame":
        r"""Convert a list of events to dataframe.

        Args:
            events (List[Event]): a list of events to convert

        Returns:
            pandas.DataFrame: dataframe structure of the events list
        """
        df = pd.DataFrame(columns=["time", "priority", "object", "method", "status", "signature"])
        for i, event in enumerate(events):
            event_info = pd.DataFrame({"time": event.time,
                                       "priority": event.priority,
                                       "object": event.handler.owner.name,
                                       "method": event.handler.method,
                                       "status": event.status.value,
                                       "signature": event.signature.name},
                                      index=[f"Event {i + 1}"])
            df = pd.concat([df, event_info])
        return df

    def print(self) -> None:
        r"""Print the event details.
        """
        df = Event.events_to_dataframe([self])
        print(f"\nEvent details:\n{df.to_string()}")

    def __eq__(self, other: "Event") -> bool:
        return (self.time == other.time) and (self.priority == other.priority)

    def __ne__(self, other: "Event") -> bool:
        return (self.time != other.time) or (self.priority != other.priority)

    def __gt__(self, other: "Event") -> bool:
        return (self.time > other.time) or (self.time == other.time and self.priority > other.priority)

    def __lt__(self, other: "Event") -> bool:
        return (self.time < other.time) or (self.time == other.time and self.priority < other.priority)


class EventHandler:
    r"""Class for creating an event handler in the discrete-event simulation.

    Once an event is triggered at a certain time, a pre-specified event handler will be activated to handle the event.

    Attributes:
        owner (Union[Entity, Protocol]): an entity or a protocol to process the event
        method (str): method for a specific process
        params (List[Any]): parameters for processing the method
        **kwargs: keyword arguments for processing the method
    """

    def __init__(self, owner: Union["Entity", "Protocol"], method: str, params=None, **kwargs):
        r"""Constructor for EventHandler class.

        Args:
            owner (Union[Entity, Protocol]): an entity or a protocol to process the event
            method (str): method for specific execution
            params (List[Any]): parameters for processing the method
            **kwargs: keyword arguments for processing the method
        """
        assert params is None or isinstance(params, List), "Should input parameters as a list."

        self.owner = owner
        self.method = method
        self.params = [] if params is None else params
        self.kwargs = kwargs

    def handle(self) -> None:
        r"""Method to execute the event.
        """
        return getattr(self.owner, self.method)(*self.params, **self.kwargs)


class Scheduler:
    r"""Class for creating an event scheduler in the discrete-event simulation.

    Note:
        Only objects of ``Entity`` and ``Protocol`` class contain a scheduler which can schedule events
        to be triggered at a specified time on the timeline.

    Attributes:
        owner (Union[Entity, Protocol]): owner of the scheduler
        env (DESEnv): related discrete-event simulation environment
    """

    def __init__(self, owner: Union["Entity", "Protocol"], env=None):
        r"""Constructor for Scheduler class.

        Args:
            owner (Union[Entity, Protocol]): owner of the scheduler
            env (DESEnv): related discrete-event simulation environment
        """
        self.owner = owner
        self.env = env

    def schedule_at(self, time: int, handler: "EventHandler", priority=None) -> None:
        r"""Schedule an event at a given time.

        Args:
            time (int): time of the event
            handler (EventHandler): handler to process the event
            priority (int): priority of the event
        """
        assert self.env is not None, "Cannot find a DES environment for the scheduler."
        event = Event(time, handler, self.owner, priority)
        self.owner.signed_events.append(event)
        event.handler.owner.agenda.append(event)
        self.env.future_events.push(event)
        self.env.counter_scheduled += 1
        event.status = Event.Status.SCHEDULED

    def schedule_after(self, delay: int, handler: "EventHandler", priority=None) -> None:
        r"""Schedule an event after a time delay.

        Args:
            delay (int): time to delay
            handler (EventHandler): handler to process the event
            priority (int): priority of the event
        """
        self.schedule_at(self.env.now + delay, handler, priority)

    def schedule_now(self, handler: "EventHandler", priority=None) -> None:
        r"""Schedule an event at the current time.

        Args:
            handler (EventHandler): handler to process the event
            priority (int): priority of the event
        """
        self.schedule_at(self.env.now, handler, priority)

    def reschedule(self, event: "Event", time=None, priority=None) -> None:
        r"""Reschedule the time or priority of an event.

        This is implemented by cancelling the previous event and scheduling a new event.

        Args:
            event (Event): event to reschedule
            time (int): new time of the event
            priority (int): new priority of the event
        """
        assert time is not None or priority is not None, "Should either input a new time or a new priority."
        if time is not None:
            assert Event.is_time_valid(time), f"Invalid event time!"
        if priority is not None:
            assert Event.is_priority_valid(priority), f"Invalid event priority!"
        new_time = event.time if time is None else time
        new_priority = event.priority if priority is None else priority
        self.cancel(event)
        self.schedule_at(new_time, event.handler, new_priority)

    @staticmethod
    def cancel(event: "Event") -> None:
        r"""Cancel a scheduled event.

        Args:
            event (Event): event to cancel
        """
        event.cancel()


class DESEnv:
    r"""Class for discrete-event simulation environment.

    Important:
        This is the simulation environment for network simulations.
        We should always instantiate a ``DESEnv`` class to create an environment first before any network setup.

    Examples:
        Create a simulation environment and set it as a default.
        Then any entities created will be automatically attached to this default environment
        unless an explict environment parameter is given.

        >>> env = DESEnv(name="BB84", default=True)

    Attributes:
        name (str): name of the discrete-event simulation environment
        end_time (float): end time of the simulation in picoseconds
        entities (List[Entity]): a list of entities attached to this environment
        future_events (FutureEvents): a list of scheduled events to the timeline
        counter_scheduled (int): counter of scheduled events
        counter_handled (int): counter of handled events
        status (str): status of the simulation environment
        logging (bool): whether to output the simulation log file
        logger (logging.Logger): logger of the simulation environment
        log_config (dict): configuration of the simulation log
    """

    __DEFAULT_ENV = None

    def __init__(self, name: str, default=False):
        r"""Constructor for DESEnv class.

        Args:
            name (str): name of the discrete-event simulation environment
            default (bool): whether to set the current environment as default
        """
        self.name = name
        self.__current_time = 0  # use self.now to access this time
        self.end_time = float("inf")
        self.entities = []
        self.future_events = self.FutureEvents()
        self.counter_scheduled = 0
        self.counter_handled = 0
        self.status = DESEnv.Status.STOPPED
        self.logging = False
        self.logger = None
        self.log_config = {'path': None, 'level': "DEBUG"}
        if default:
            self.set_default_env()

    @unique
    class Status(Enum):
        r"""Simulation status.
        """

        READY = "Ready"
        RUNNING = "Running"
        STOPPED = "Stopped"

    def init(self) -> None:
        r"""Manually initialize the simulation environment.

        Initialize all the entities in the simulation environment.
        """
        for entity in self.entities:
            entity.init()
        self.status = DESEnv.Status.READY  # switch the environment status to 'Ready'

    def set_default_env(self) -> None:
        r"""Set the current environment as default.
        """
        DESEnv.__DEFAULT_ENV = self

    @classmethod
    def get_default_env(cls) -> "DESEnv":
        r"""Get the default discrete-event simulation environment.

        Returns:
            DESEnv: the default discrete-event simulation environment
        """
        return DESEnv.__DEFAULT_ENV

    def get_node(self, name: str) -> "Node":
        r"""Get a node entity by its name.

        Args:
            name (str): name of the entity

        Returns:
            Node: node entity to return
        """
        from qcompute_qnet.topology.node import Node
        for entity in self.entities:
            if isinstance(entity, Node) and entity.name == name:
                return entity

    @property
    def network(self) -> "Network":
        r"""Get the quantum network in this environment.

        Returns:
            Network: network in the discrete-event simulation environment
        """
        from qcompute_qnet.topology.network import Network
        for entity in self.entities:
            if isinstance(entity, Network):
                return entity

    def set_log(self, path=None, level=None) -> None:
        r"""Log settings for the discrete-event simulation.

        Args:
            path (str): name and storing path of the log file
            level (str): logging level of the logger
        """
        if path is not None:
            self.log_config['path'] = path
        if level is not None:
            self.log_config['level'] = level

    @property
    def now(self) -> int:
        r"""Return the current time.

        Returns:
            int: current time of the simulation
        """
        return self.__current_time

    def run(self, end_time=None, logging=False, summary=True) -> None:
        r"""Run the simulation and collect necessary statistics.

        The simulation keeps popping events in the ``future_events`` list and executing them by the assigned handlers.
        The simulation time leaps at discrete timestamps where the events take place.
        The simulation will keep running until the future event list is empty or the end time is reached.

        Args:
            end_time (float): end time of the simulation in picoseconds
            logging (bool): whether to output the simulation log file
            summary (bool): whether to print the simulation report on the terminal
        """
        self.end_time = float("inf") if end_time is None else round(end_time)

        self.logging = logging
        self.logger = log_init(self, self.log_config['path'], self.log_config['level'])

        # Use time.time_ns() instead of time.time() to avoid float type accuracy loss
        sim_start = time.time_ns()
        self.logger.info("-" * 20 + "Simulation starts" + "-" * 20)

        self.status = DESEnv.Status.RUNNING  # switch the environment status to 'Running'

        while not self.future_events.is_empty():
            event = self.future_events.pop()
            # Terminate if the event is scheduled after the end time of the simulation
            # The events scheduled at the end time will still be processed
            if event.time > self.end_time:
                self.__current_time = self.end_time
                self.logger.info(f"Reach end time! No more events will be processed")
                break

            assert self.now <= event.time, f"Inappropriate event time scheduled by {event.handler.owner}"

            # Continue if the current event is cancelled
            if event.status == Event.Status.CANCELLED:
                continue

            self.__current_time = event.time  # time leaps
            event.handler.handle()  # event execution
            event.status = Event.Status.PROCESSED
            self.counter_handled += 1

        self.status = DESEnv.Status.STOPPED  # switch the environment status to 'Stopped'

        sim_end = time.time_ns()
        elapsed_time = sim_end - sim_start

        self.logger.info("-" * 20 + "Simulation ends" + "-" * 20)
        self.logger.info(f"Elapsed wall time: {elapsed_time * 1e-9:.4f}s \t"
                         f" Elapsed virtual time: {self.now * 1e-12:.4f}s")
        self.logger.debug(f"Scheduled events: {self.counter_scheduled} \t Handled events: {self.counter_handled}")
        if summary:
            self.print_summary(elapsed_time)

    def stop(self) -> None:
        r"""Stop the simulation.

        Set the ``end_time`` as ``now`` to end the simulation.
        """
        self.end_time = self.now
        self.logger.info("Simulation is stopped")

    def reset(self) -> None:
        r"""Reset the discrete-event simulation environment.
        """
        raise NotImplementedError("Not available in this version!")

    @staticmethod
    def seed(random_seed: int) -> None:
        r"""Set a random seed.

        Args:
            random_seed (int): random seed for the simulation
        """
        np.random.seed = random_seed

    def print_summary(self, elapsed_time: int) -> None:
        r"""Print the summary information of the simulation.
        """
        string = StringIO()
        string.write("-" * 50 + "\n")
        string.write(f"Simulation status:\t'{self.status.value}'\n\n")
        string.write(f"- Elapsed wall time: {elapsed_time * 1e-9:.4f}s\n")
        string.write(f"- Elapsed virtual time: {self.now * 1e-12:.4f}s\n\n")
        string.write(f"- Scheduled events: {self.counter_scheduled}\n")
        string.write(f"- Handled events: {self.counter_handled}\n")
        if self.logging:
            string.write(f"\nLog records saved in: '{self.logger.handlers[0].baseFilename}'\n")
        string.write("-" * 50)
        print(string.getvalue())

    class FutureEvents:
        r"""Class for creating a future event list in the discrete-event simulation.

        ``FutureEvents`` manages a list of scheduled events during the simulation.
        The events are maintained in a heap structure to fasten the event scheduling.

        Attributes:
            events (List[Event]): a list of the scheduled events
        """

        def __init__(self):
            r"""Constructor for FutureEvents class.
            """
            self.events = []

        def push(self, event: "Event") -> None:
            r"""Push the event into the event heap.

            Args:
                event (Event): new scheduled event
            """
            heappush(self.events, event)

        def pop(self) -> "Event":
            r"""Pop the event with the highest executing priority from the event list.

            Returns:
                Event: event with the highest executing priority
            """
            return heappop(self.events)

        def is_empty(self) -> bool:
            r"""Check if the event list is empty.

            Returns:
                bool: whether the event list is empty
            """
            return True if not self.events else False

        def print(self) -> None:
            r"""Print the future event list.
            """
            df = Event.events_to_dataframe(self.events)
            print(f"\nFuture event list (unsorted):\n{df.to_string()}")
