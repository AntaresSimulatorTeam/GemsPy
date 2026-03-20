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

from gems.expression import ExpressionNode, PrinterVisitor, param, var, visit


def test_comparison() -> None:
    x = var("x")
    p = param("p")
    expr: ExpressionNode = (5 * x + 3) >= p - 2

    assert visit(expr, PrinterVisitor()) == "((5.0 * x) + 3.0) >= (p - 2.0)"


def test_floor_ceil_max_min_printer() -> None:
    from gems.expression.expression import maximum, minimum

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
