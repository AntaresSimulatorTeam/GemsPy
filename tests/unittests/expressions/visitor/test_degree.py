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

import math

import pytest

from gems.expression import (
    ExpressionDegreeVisitor,
    LiteralNode,
    maximum,
    minimum,
    param,
    var,
    visit,
)
from gems.expression.expression import CeilNode, FloorNode


def test_degree() -> None:
    x = var("x")
    p = param("p")
    expr = (5 * x + 3) / p

    assert visit(expr, ExpressionDegreeVisitor()) == 1

    expr = x * expr
    assert visit(expr, ExpressionDegreeVisitor()) == 2


def test_floor_ceil_degree() -> None:
    x = var("x")
    p = param("p")

    assert visit(FloorNode(p), ExpressionDegreeVisitor()) == 0
    assert visit(CeilNode(p), ExpressionDegreeVisitor()) == 0
    assert visit(FloorNode(x), ExpressionDegreeVisitor()) == math.inf
    assert visit(CeilNode(x), ExpressionDegreeVisitor()) == math.inf


def test_max_min_degree() -> None:
    x = var("x")
    p = param("p")
    q = param("q")

    assert visit(maximum(p, q), ExpressionDegreeVisitor()) == 0
    assert visit(minimum(p, q), ExpressionDegreeVisitor()) == 0
    assert visit(maximum(x, p), ExpressionDegreeVisitor()) == math.inf
    assert visit(minimum(p, x), ExpressionDegreeVisitor()) == math.inf
    assert visit(maximum(x, x), ExpressionDegreeVisitor()) == math.inf
    # variadic (3+ operands)
    assert visit(maximum(p, q, param("r")), ExpressionDegreeVisitor()) == 0
    assert visit(minimum(p, q, param("r")), ExpressionDegreeVisitor()) == 0
    assert visit(maximum(p, q, x), ExpressionDegreeVisitor()) == math.inf
    assert visit(minimum(p, x, q), ExpressionDegreeVisitor()) == math.inf


@pytest.mark.xfail(reason="Degree simplification not implemented")
def test_degree_computation_should_take_into_account_simplifications() -> None:
    x = var("x")
    expr = x - x
    assert visit(expr, ExpressionDegreeVisitor()) == 0

    expr = LiteralNode(0) * x
    assert visit(expr, ExpressionDegreeVisitor()) == 0
