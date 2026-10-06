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


@pytest.mark.parametrize(
    "expr, printed",
    [
        (var("x").set_index("fuel"), "(x[fuel])"),
        (var("x").set_index("fuel", position=2), "(x[fuel=2.0])"),
        (var("x").set_index("fuel", relative_shift=1), "(x[fuel+1.0])"),
        (var("x").set_index("fuel", relative_shift=-1), "(x[fuel-1.0])"),
        (var("x").sum_over("fuel"), "sum_over(fuel, x)"),
        # multi-set indexing: nested SetIndexNodes collapse into one bracket
        (
            var("x").set_index("fuel", position=1).set_index("segment", position=2),
            "(x[fuel=1.0, segment=2.0])",
        ),
        (
            var("x").set_index("fuel").set_index("segment", relative_shift=1),
            "(x[fuel, segment+1.0])",
        ),
        # a time term is a TimeShiftNode: it keeps its own bracket
        (var("x").shift(1).set_index("fuel"), "((x[t+1.0])[fuel])"),
        # negative time shifts print with a minus sign, never "+-"
        (var("x").shift(-1), "(x[t-1.0])"),
        (var("x").shift(-param("p")), "(x[t-p])"),
    ],
)
def test_set_index_sum_over_printer(expr: ExpressionNode, printed: str) -> None:
    assert visit(expr, PrinterVisitor()) == printed
