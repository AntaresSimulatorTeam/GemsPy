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

from gems_craft.expression import literal, param, relative_time, var
from gems_craft.expression.expression import (
    DualNode,
    ExpressionNode,
    LowerBoundNode,
    ReducedCostNode,
    UpperBoundNode,
    port_field,
)
from gems_craft.expression.indexing import (
    IndexingStructureProvider,
    IndexingUsageError,
    UnresolvedPortFieldError,
    compute_indexation,
)
from gems_craft.expression.indexing_structure import IndexingStructure


class StructureProvider(IndexingStructureProvider):
    def get_component_variable_structure(
        self, component_id: str, name: str
    ) -> IndexingStructure:
        return IndexingStructure(True, True)

    def get_component_parameter_structure(
        self, component_id: str, name: str
    ) -> IndexingStructure:
        return IndexingStructure(True, True)

    def get_parameter_structure(self, name: str) -> IndexingStructure:
        return IndexingStructure(True, True)

    def get_variable_structure(self, name: str) -> IndexingStructure:
        return IndexingStructure(True, True)

    def get_constraint_structure(self, name: str) -> IndexingStructure:
        return IndexingStructure(True, True)


def test_shift() -> None:
    x = var("x")
    expr = x.shift(1)

    provider = StructureProvider()
    assert compute_indexation(expr, provider) == IndexingStructure(True, True)


def test_time_eval() -> None:
    x = var("x")
    expr = x.eval(1)

    provider = StructureProvider()
    assert compute_indexation(expr, provider) == IndexingStructure(False, True)


def test_time_sum() -> None:
    x = var("x")
    expr = x.time_sum(1, 4)
    provider = StructureProvider()

    assert compute_indexation(expr, provider) == IndexingStructure(True, True)


def test_sum_over_whole_block() -> None:
    x = var("x")
    expr = x.time_sum()
    provider = StructureProvider()

    assert compute_indexation(expr, provider) == IndexingStructure(False, True)


def test_expectation() -> None:
    x = var("x")
    expr = x.expec()
    provider = StructureProvider()

    assert compute_indexation(expr, provider) == IndexingStructure(True, False)


def test_indexing_structure_comparison() -> None:
    free = IndexingStructure(True, True)
    constant = IndexingStructure(False, False)
    assert free | constant == IndexingStructure(True, True)


def test_multiplication_of_differently_indexed_terms() -> None:
    x = var("x")
    p = param("p")
    expr = p * x

    class CustomStructureProvider(IndexingStructureProvider):
        def get_component_variable_structure(
            self, component_id: str, name: str
        ) -> IndexingStructure:
            raise NotImplementedError()

        def get_component_parameter_structure(
            self, component_id: str, name: str
        ) -> IndexingStructure:
            raise NotImplementedError()

        def get_parameter_structure(self, name: str) -> IndexingStructure:
            return IndexingStructure(False, False)

        def get_variable_structure(self, name: str) -> IndexingStructure:
            return IndexingStructure(True, True)

        def get_constraint_structure(self, name: str) -> IndexingStructure:
            raise NotImplementedError()

    provider = CustomStructureProvider()

    assert compute_indexation(expr, provider) == IndexingStructure(True, True)


def test_dual_reduced_cost_indexing() -> None:
    provider = StructureProvider()
    assert compute_indexation(DualNode("balance"), provider) == IndexingStructure(
        True, True
    )
    assert compute_indexation(ReducedCostNode("p"), provider) == IndexingStructure(
        True, True
    )


class _SetStructureProvider(IndexingStructureProvider):
    """Variable 'x' is time/scenario-varying and indexed by 'fuel' and 'segment'."""

    def get_component_variable_structure(
        self, component_id: str, name: str
    ) -> IndexingStructure:
        raise NotImplementedError()

    def get_component_parameter_structure(
        self, component_id: str, name: str
    ) -> IndexingStructure:
        raise NotImplementedError()

    def get_parameter_structure(self, name: str) -> IndexingStructure:
        raise NotImplementedError()

    def get_variable_structure(self, name: str) -> IndexingStructure:
        return IndexingStructure(True, True, frozenset({"fuel", "segment"}))

    def get_constraint_structure(self, name: str) -> IndexingStructure:
        raise NotImplementedError()


@pytest.mark.parametrize(
    "expr, sets",
    [
        # an explicit position or a sum_over removes the set dimension...
        (var("x").set_index("fuel", position=literal(2)), {"segment"}),
        (var("x").sum_over("fuel"), {"segment"}),
        # ...a bare or relative-shift index keeps it
        (var("x").set_index("fuel"), {"fuel", "segment"}),
        (var("x").set_index("fuel", relative_shift=literal(1)), {"fuel", "segment"}),
    ],
)
def test_set_index_and_sum_over_structure(expr: ExpressionNode, sets: set) -> None:
    assert compute_indexation(expr, _SetStructureProvider()) == IndexingStructure(
        True, True, frozenset(sets)
    )


def test_combine_does_not_drop_sets_after_time_scenario_settled() -> None:
    """Regression test for _combine's former short-circuit: once time and
    scenario are both known to vary, further operands must still contribute
    their `sets` to the union instead of being skipped."""
    x = var("x")  # fully time/scenario-varying, no sets
    y = var("y")

    class MixedProvider(IndexingStructureProvider):
        def get_component_variable_structure(self, component_id, name):  # type: ignore[no-untyped-def]
            raise NotImplementedError()

        def get_component_parameter_structure(self, component_id, name):  # type: ignore[no-untyped-def]
            raise NotImplementedError()

        def get_parameter_structure(self, name: str) -> IndexingStructure:
            raise NotImplementedError()

        def get_variable_structure(self, name: str) -> IndexingStructure:
            if name == "x":
                return IndexingStructure(True, True)
            return IndexingStructure(True, True, frozenset({"fuel"}))

        def get_constraint_structure(self, name: str) -> IndexingStructure:
            raise NotImplementedError()

    provider = MixedProvider()
    assert compute_indexation(x + y, provider) == IndexingStructure(
        True, True, frozenset({"fuel"})
    )


class _ScenarioIndexProvider(_SetStructureProvider):
    """'x' as in _SetStructureProvider; parameters vary by scenario only."""

    def get_parameter_structure(self, name: str) -> IndexingStructure:
        return IndexingStructure(False, True)


@pytest.mark.parametrize(
    "expr",
    [
        var("x").set_index("fuel", position=param("p")),
        var("x").set_index("fuel", relative_shift=param("p")),
    ],
)
def test_set_index_brings_index_dependency(expr: ExpressionNode) -> None:
    # The position or shift brings its own scenario dependency.
    assert compute_indexation(expr, _ScenarioIndexProvider()).scenario


def test_lower_upper_bound_indexing() -> None:
    provider = StructureProvider()
    assert compute_indexation(LowerBoundNode("x"), provider) == IndexingStructure(
        True, True
    )
    assert compute_indexation(UpperBoundNode("x"), provider) == IndexingStructure(
        True, True
    )


class ConstantParameterProvider(StructureProvider):
    def get_parameter_structure(self, name: str) -> IndexingStructure:
        return IndexingStructure(False, False)


def test_unresolved_port_field_raises() -> None:
    with pytest.raises(UnresolvedPortFieldError):
        compute_indexation(
            port_field("p", "f").sum_connections(), ConstantParameterProvider()
        )


def test_index_error_reported_alongside_unresolved_port_field() -> None:
    """An invalid set index is still reported when a sibling operand (before or
    after it) holds an unresolved port field."""
    provider = ConstantParameterProvider()
    port = port_field("p", "f").sum_connections()
    bad = param("p").set_index("fuel")
    with pytest.raises(IndexingUsageError):
        compute_indexation(port + bad, provider)
    with pytest.raises(IndexingUsageError):
        compute_indexation(bad + port, provider)


def test_time_shift_on_unresolved_port_field_is_deferred() -> None:
    with pytest.raises(UnresolvedPortFieldError):
        compute_indexation(
            port_field("p", "f").sum_connections().shift(1), ConstantParameterProvider()
        )


def test_sum_over_unindexed_operand_warns() -> None:
    x = var("x")
    provider = _SetStructureProvider()
    with pytest.warns(UserWarning, match="not indexed by .other."):
        compute_indexation(x.sum_over("other"), provider)


def test_time_sum_with_mixed_bounds_of_a_constant_depends_on_time() -> None:
    # sum(0 .. t, k) = (t + 1) * k: the number of terms changes with t.
    expr = param("k").time_sum_between(0, relative_time(0))

    assert compute_indexation(expr, ConstantParameterProvider()) == IndexingStructure(
        True, False
    )


class NamedParameterProvider(StructureProvider):
    """d varies in time only, p in scenario only."""

    def get_parameter_structure(self, name: str) -> IndexingStructure:
        return {
            "d": IndexingStructure(True, False),
            "p": IndexingStructure(False, True),
        }[name]


@pytest.mark.parametrize(
    "start, end, expected",
    [
        pytest.param(0, param("p"), IndexingStructure(False, True), id="absolute"),
        pytest.param(
            0, relative_time(param("p")), IndexingStructure(True, True), id="mixed"
        ),
        pytest.param(
            relative_time(-param("p")),
            relative_time(0),
            IndexingStructure(True, True),
            id="relative",
        ),
    ],
)
def test_time_sum_with_scenario_dependent_bound_depends_on_scenario(
    start, end, expected
) -> None:
    expr = param("d").time_sum_between(start, end)

    assert compute_indexation(expr, NamedParameterProvider()) == expected


@pytest.mark.parametrize(
    "expr, expected",
    [
        # d read at t + p: a time step per scenario
        pytest.param(
            param("d").shift(param("p")), IndexingStructure(True, True), id="shift"
        ),
        # d read at time p: one value per scenario
        pytest.param(
            param("d").eval(param("p")), IndexingStructure(False, True), id="index"
        ),
        # p does not depend on time: reading it elsewhere changes nothing
        pytest.param(
            param("p").eval(param("d")),
            IndexingStructure(False, True),
            id="index-of-constant",
        ),
    ],
)
def test_time_operator_given_by_a_parameter(expr, expected) -> None:
    assert compute_indexation(expr, NamedParameterProvider()) == expected
