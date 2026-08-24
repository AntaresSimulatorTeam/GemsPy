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

"""Cross-validation of a resolved system against the models it refers to.

Kept apart from `resolve_components.py`, which only resolves the parsed system
into its runtime objects — mirroring `optim_config/parsing.py` and
`optim_config/validation.py`.
"""

from typing import Dict

from gems_craft.model import Model
from gems_craft.study.system import System


def consistency_check(system: System, input_models: Dict[str, Model]) -> bool:
    """
    Checks if all components in the System have a valid model from the library.
    Returns True if all components are consistent, raises ValueError otherwise.
    """
    # TODO: Update this consistency check to check if each component have a valid model from the lib it refers to (and not all libs)
    model_ids_set = input_models.keys()
    for component in system.all_components:
        if component.model.id not in model_ids_set:
            raise ValueError(
                f"Error: Component {component.id} has invalid model ID: {component.model.id}"
            )
    return True
