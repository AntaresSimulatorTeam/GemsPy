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

from gems.expression import ParameterValueProvider, param, resolve_parameters, var


def test_parameters_resolution() -> None:
    class TestParamProvider(ParameterValueProvider):
        def get_component_parameter_value(self, component_id: str, name: str) -> float:
            raise NotImplementedError()

        def get_parameter_value(self, name: str) -> float:
            return 2

    x = var("x")
    p = param("p")
    expr = (5 * x + 3) / p
    assert resolve_parameters(expr, TestParamProvider()) == (5 * x + 3) / 2
