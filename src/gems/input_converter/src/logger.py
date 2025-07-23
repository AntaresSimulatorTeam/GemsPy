# Copyright (c) 2024, RTE (https://www.rte-france.com)
#
# See AUTHORS.txt
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0
#
# This file is part of the Antares project.
import logging
import logging.config
import sys
from typing import Optional


def Logger(name: str, file_name: Optional[str]) -> logging.Logger:
    formatter = logging.Formatter(
        fmt="%(asctime)s %(module)s,line: %(lineno)d %(levelname)8s | %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )  # %I:%M:%S %p AM|PM format

    if file_name and not str(file_name).endswith(".log"):
        file_name = f"{file_name}.log"
    logging.basicConfig(
        filename=file_name,
        format="%(asctime)s %(module)s,line: %(lineno)d %(levelname)8s | %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        filemode="a",
        level=logging.INFO,
    )

    log_obj = logging.getLogger(name)
    log_obj.setLevel(logging.DEBUG)

    # console printer
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logging.getLogger().addHandler(screen_handler)

    log_obj.info("Logger object created successfully.. ")
    return log_obj
