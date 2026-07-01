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

"""
Module for technical utilities.
"""

from pydantic import BaseModel, ConfigDict

"""
Only _to_kebab adn ModifiedBaseModel are used in the project.
"""


# Design note: actual parsing and validation is delegated to pydantic models
def _to_kebab(snake: str) -> str:
    return snake.replace("_", "-")


class ModifiedBaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_kebab, extra="forbid", populate_by_name=True
    )
