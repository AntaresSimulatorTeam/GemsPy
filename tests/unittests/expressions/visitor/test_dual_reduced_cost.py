# Copyright (c) 2024, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for DualNode and ReducedCostNode across all visitors."""

import math

import pytest

from gems.expression import (
    ExpressionDegreeVisitor,
    PrinterVisitor,
    copy_expression,
    param,
    var,
    visit,
)
from gems.expression.equality import expressions_equal
from gems.expression.evaluate import EvaluationVisitor
from gems.expression.expression import DualNode, ReducedCostNode
from gems.expression.indexing import (
    IndexingStructure,
    IndexingStructureProvider,
    compute_indexation,
)
from gems.expression.predicates import contains_dual_or_reduced_cost

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FixedStructureProvider(IndexingStructureProvider):
    def __init__(self, time: bool = True, scenario: bool = True) -> None:
        self._structure = IndexingStructure(time, scenario)

    def get_parameter_structure(self, name: str) -> IndexingStructure:
        return self._structure

    def get_variable_structure(self, name: str) -> IndexingStructure:
        return self._structure

    def get_constraint_structure(self, name: str) -> IndexingStructure:
        return self._structure


# ---------------------------------------------------------------------------
# Printer visitor
# ---------------------------------------------------------------------------


def test_printer_dual() -> None:
    assert visit(DualNode("balance"), PrinterVisitor()) == "dual(balance)"


def test_printer_reduced_cost() -> None:
    assert visit(ReducedCostNode("p"), PrinterVisitor()) == "reduced_cost(p)"


# ---------------------------------------------------------------------------
# Degree visitor
# ---------------------------------------------------------------------------


def test_degree_dual_is_inf() -> None:
    assert visit(DualNode("balance"), ExpressionDegreeVisitor()) == math.inf


def test_degree_reduced_cost_is_inf() -> None:
    assert visit(ReducedCostNode("p"), ExpressionDegreeVisitor()) == math.inf


# ---------------------------------------------------------------------------
# Equality / copy
# ---------------------------------------------------------------------------


def test_equality_dual_same_constraint() -> None:
    assert expressions_equal(DualNode("balance"), copy_expression(DualNode("balance")))


def test_equality_dual_different_constraints() -> None:
    assert not expressions_equal(DualNode("balance"), DualNode("other"))


def test_equality_reduced_cost_same_variable() -> None:
    assert expressions_equal(
        ReducedCostNode("p"), copy_expression(ReducedCostNode("p"))
    )


def test_equality_reduced_cost_different_variables() -> None:
    assert not expressions_equal(ReducedCostNode("p"), ReducedCostNode("q"))


def test_equality_dual_vs_reduced_cost() -> None:
    assert not expressions_equal(DualNode("p"), ReducedCostNode("p"))


# ---------------------------------------------------------------------------
# Evaluation visitor — must raise NotImplementedError
# ---------------------------------------------------------------------------


def test_evaluate_dual_raises() -> None:
    from gems.expression import EvaluationContext

    ctx = EvaluationContext()
    with pytest.raises(NotImplementedError, match="dual"):
        visit(DualNode("balance"), EvaluationVisitor(ctx))


def test_evaluate_reduced_cost_raises() -> None:
    from gems.expression import EvaluationContext

    ctx = EvaluationContext()
    with pytest.raises(NotImplementedError, match="reduced_cost"):
        visit(ReducedCostNode("p"), EvaluationVisitor(ctx))


# ---------------------------------------------------------------------------
# Indexing visitor
# ---------------------------------------------------------------------------


def test_indexing_dual_uses_constraint_structure() -> None:
    provider = _FixedStructureProvider(time=True, scenario=False)
    result = compute_indexation(DualNode("balance"), provider)
    assert result == IndexingStructure(True, False)


def test_indexing_reduced_cost_uses_variable_structure() -> None:
    provider = _FixedStructureProvider(time=True, scenario=True)
    result = compute_indexation(ReducedCostNode("p"), provider)
    assert result == IndexingStructure(True, True)


# ---------------------------------------------------------------------------
# contains_dual_or_reduced_cost predicate
# ---------------------------------------------------------------------------


def test_predicate_dual_node_is_detected() -> None:
    assert contains_dual_or_reduced_cost(DualNode("balance"))


def test_predicate_reduced_cost_node_is_detected() -> None:
    assert contains_dual_or_reduced_cost(ReducedCostNode("p"))


def test_predicate_plain_expression_returns_false() -> None:
    assert not contains_dual_or_reduced_cost(var("x") + param("p"))


def test_predicate_nested_dual_is_detected() -> None:
    expr = var("x") + DualNode("balance")
    assert contains_dual_or_reduced_cost(expr)


def test_predicate_nested_reduced_cost_is_detected() -> None:
    expr = param("cost") * ReducedCostNode("p")
    assert contains_dual_or_reduced_cost(expr)
