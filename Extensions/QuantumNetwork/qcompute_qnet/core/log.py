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
Module for log records.
"""

import logging

__all__ = [
    "LoggingFilter",
    "log_init",
]


class LoggingFilter(logging.Filter):
    r"""Class for the filter of log file.

    Attributes:
        env (DESEnv): discrete-event simulation environment to log
    """
    def __init__(self, env: "DESEnv"):
        r"""Constructor for LoggingFilter class.

        Args:
            env (DESEnv): discrete-event simulation environment to log
        """
        super().__init__()
        self.env = env

    def filter(self, record: logging.LogRecord) -> bool:
        r"""Filter the log records.

        Args:
            record (logging.LogRecord): an event to be logged

        Returns:
            bool: whether the specified record is to be logged
        """
        record.virtual_time = self.env.now

        return True


def log_init(env: "DESEnv", path=None, level="DEBUG") -> logging.Logger:
    r"""Initialization of a log file.

    Args:
        env (DESEnv): discrete-event simulation environment to log
        path (str): name and storing path of the log
        level (str): logging level of the logger

    Returns:
        logging.Logger: initialized logger
    """
    _log_path = f"./{env.name}.log" if path is None else path
    _logger = logging.getLogger(_log_path)
    _logger.setLevel(level)

    if env.logging:
        open(_log_path, "w").close()
        file_handler = logging.FileHandler(_log_path)
        log_format = "%(asctime)s:\t %(levelname)s\t %(virtual_time)s\t %(message)s"
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        file_filter = LoggingFilter(env)

        _logger.addHandler(file_handler)
        _logger.addFilter(file_filter)

    return _logger
