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

from gems_craft.expression import (
    ExpressionDegreeVisitor,
    LiteralNode,
    maximum,
    minimum,
    param,
    var,
    visit,
)
from gems_craft.expression.expression import (
    AbsNode,
    CeilNode,
    DualNode,
    FloorNode,
    ReducedCostNode,
    RoundNode,
)


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


def test_abs_round_degree() -> None:
    x = var("x")
    p = param("p")

    assert visit(AbsNode(p), ExpressionDegreeVisitor()) == 0
    assert visit(RoundNode(p), ExpressionDegreeVisitor()) == 0
    assert visit(AbsNode(x), ExpressionDegreeVisitor()) == math.inf
    assert visit(RoundNode(x), ExpressionDegreeVisitor()) == math.inf
    assert visit(AbsNode(p - param("q")), ExpressionDegreeVisitor()) == 0


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


def test_dual_reduced_cost_degree() -> None:
    assert visit(DualNode("balance"), ExpressionDegreeVisitor()) == math.inf
    assert visit(ReducedCostNode("p"), ExpressionDegreeVisitor()) == math.inf


@pytest.mark.xfail(reason="Degree simplification not implemented")
def test_degree_computation_should_take_into_account_simplifications() -> None:
    x = var("x")
    expr = x - x
    assert visit(expr, ExpressionDegreeVisitor()) == 0

    expr = LiteralNode(0) * x
    assert visit(expr, ExpressionDegreeVisitor()) == 0


def test_power_degree() -> None:
    x = var("x")
    p = param("p")

    # Literals and parameters stay constant, whatever the exponent.
    assert visit(param("p") ** 2, ExpressionDegreeVisitor()) == 0
    assert visit(LiteralNode(2) ** p, ExpressionDegreeVisitor()) == 0
    assert visit(p ** (1 + param("q")), ExpressionDegreeVisitor()) == 0

    # A variable base raised to a non-negative integer literal stays polynomial.
    assert visit(x**2, ExpressionDegreeVisitor()) == 2
    assert visit(x**1, ExpressionDegreeVisitor()) == 1
    assert visit(x**0, ExpressionDegreeVisitor()) == 0
    assert visit((x * x) ** 2, ExpressionDegreeVisitor()) == 4

    # Any other exponent on a variable base is not polynomial.
    assert visit(x**p, ExpressionDegreeVisitor()) == math.inf
    assert visit(x**0.5, ExpressionDegreeVisitor()) == math.inf
    assert visit(x ** (-2), ExpressionDegreeVisitor()) == math.inf

    # An inf-degree base stays inf, including for the exponent 0 (inf * 0 is nan).
    assert visit(FloorNode(x) ** 2, ExpressionDegreeVisitor()) == math.inf
    assert visit(FloorNode(x) ** 0, ExpressionDegreeVisitor()) == 0


def test_variable_exponent_raises() -> None:
    with pytest.raises(
        ValueError, match="Exponent of a power expression must not depend on variables"
    ):
        visit(param("p") ** var("x"), ExpressionDegreeVisitor())
