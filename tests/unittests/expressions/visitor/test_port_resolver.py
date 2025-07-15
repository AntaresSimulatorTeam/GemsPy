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

from typing import Dict, List

from gems.expression import ExpressionNode, var
from gems.expression.equality import expressions_equal
from gems.expression.expression import port_field
from gems.expression.port_resolver import PortFieldKey, resolve_port
from gems.model.port import PortFieldId


def test_port_field_resolution() -> None:
    ports_expressions: Dict[PortFieldKey, List[ExpressionNode]] = {}

    key = PortFieldKey("com_id", PortFieldId(field_name="field", port_name="port"))
    expression = var("flow")

    ports_expressions[key] = [expression]

    expression_2 = port_field("port", "field") + 2

    assert expressions_equal(
        resolve_port(expression_2, "com_id", ports_expressions), var("flow") + 2
    )


def test_port_field_resolution_sum() -> None:
    ports_expressions: Dict[PortFieldKey, List[ExpressionNode]] = {}

    key = PortFieldKey("com_id", PortFieldId(field_name="field", port_name="port"))

    ports_expressions[key] = [var("flow1"), var("flow2")]

    expression_2 = port_field("port", "field").sum_connections()
    assert expressions_equal(
        resolve_port(expression_2, "com_id", ports_expressions),
        var("flow1") + var("flow2"),
    )
