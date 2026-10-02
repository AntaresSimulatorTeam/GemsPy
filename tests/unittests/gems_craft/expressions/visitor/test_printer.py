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

from gems_craft.expression import ExpressionNode, PrinterVisitor, param, var, visit
from gems_craft.expression.expression import (
    DualNode,
    LowerBoundNode,
    ReducedCostNode,
    UpperBoundNode,
)


def test_comparison() -> None:
    x = var("x")
    p = param("p")
    expr: ExpressionNode = (5 * x + 3) >= p - 2

    assert visit(expr, PrinterVisitor()) == "((5.0 * x) + 3.0) >= (p - 2.0)"


def test_floor_ceil_max_min_printer() -> None:
    from gems_craft.expression.expression import maximum, minimum

    p = param("p")
    q = param("q")

    assert visit(p.floor(), PrinterVisitor()) == "floor(p)"
    assert visit(p.ceil(), PrinterVisitor()) == "ceil(p)"
    assert visit(maximum(p, q), PrinterVisitor()) == "max(p, q)"
    assert visit(minimum(p, q), PrinterVisitor()) == "min(p, q)"
    assert visit((p / q).ceil(), PrinterVisitor()) == "ceil((p / q))"
    assert (
        visit(maximum(param("a"), (p / q).ceil()), PrinterVisitor())
        == "max(a, ceil((p / q)))"
    )
    # variadic (3+ operands)
    assert visit(maximum(p, q, param("r")), PrinterVisitor()) == "max(p, q, r)"
    assert visit(minimum(p, q, param("r")), PrinterVisitor()) == "min(p, q, r)"


def test_abs_round_printer() -> None:
    p = param("p")
    q = param("q")

    assert visit(p.abs(), PrinterVisitor()) == "abs(p)"
    assert visit(p.round(), PrinterVisitor()) == "round(p)"
    assert visit((p - q).abs(), PrinterVisitor()) == "abs((p - q))"
    assert visit((p / q).round(), PrinterVisitor()) == "round((p / q))"


def test_dual_reduced_cost_printer() -> None:
    assert visit(DualNode("balance"), PrinterVisitor()) == "dual(balance)"
    assert visit(ReducedCostNode("p"), PrinterVisitor()) == "reduced_cost(p)"


def test_lower_upper_bound_printer() -> None:
    assert visit(LowerBoundNode("x"), PrinterVisitor()) == "lower_bound(x)"
    assert visit(UpperBoundNode("x"), PrinterVisitor()) == "upper_bound(x)"


def test_set_index_sum_over_printer() -> None:
    x = var("x")

    assert visit(x.set_index("fuel"), PrinterVisitor()) == "(x[fuel])"
    assert visit(x.set_index("fuel", position=2), PrinterVisitor()) == "(x[fuel=2.0])"
    assert (
        visit(x.set_index("fuel", relative_shift=1), PrinterVisitor())
        == "(x[fuel+1.0])"
    )
    assert visit(x.sum_over("fuel"), PrinterVisitor()) == "sum_over(fuel, x)"


def test_multiple_set_index_printer() -> None:
    """Multi-set indexing is represented as nested single-set SetIndexNodes
    in the AST, but the printer collapses a directly-nested chain into one
    bracket: `x[fuel=1.0, segment=2.0]`."""
    x = var("x")
    expr = x.set_index("fuel", position=1).set_index("segment", position=2)

    assert visit(expr, PrinterVisitor()) == "(x[fuel=1.0, segment=2.0])"

    mixed = x.set_index("fuel").set_index("segment", relative_shift=1)
    assert visit(mixed, PrinterVisitor()) == "(x[fuel, segment+1.0])"

    # x[t+1, fuel]: the time term (a TimeShiftNode, not a SetIndexNode) prints
    # its own bracket, so it must not be merged into the set's bracket
    shifted_then_indexed = x.shift(1).set_index("fuel")
    assert visit(shifted_then_indexed, PrinterVisitor()) == "((x[t+1.0])[fuel])"
