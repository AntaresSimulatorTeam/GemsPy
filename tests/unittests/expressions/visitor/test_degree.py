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

import pytest

from gems.expression import ExpressionDegreeVisitor, LiteralNode, param, var, visit


def test_degree() -> None:
    x = var("x")
    p = param("p")
    expr = (5 * x + 3) / p

    assert visit(expr, ExpressionDegreeVisitor()) == 1

    expr = x * expr
    assert visit(expr, ExpressionDegreeVisitor()) == 2


@pytest.mark.xfail(reason="Degree simplification not implemented")
def test_degree_computation_should_take_into_account_simplifications() -> None:
    x = var("x")
    expr = x - x
    assert visit(expr, ExpressionDegreeVisitor()) == 0

    expr = LiteralNode(0) * x
    assert visit(expr, ExpressionDegreeVisitor()) == 0
