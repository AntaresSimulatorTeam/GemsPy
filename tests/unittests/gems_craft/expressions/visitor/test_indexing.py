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

from gems_craft.expression import literal, param, var
from gems_craft.expression.expression import (
    DualNode,
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


def test_set_index_removes_set_dimension_when_explicit_position() -> None:
    x = var("x")
    provider = _SetStructureProvider()

    assert compute_indexation(
        x.set_index("fuel", position=literal(2)), provider
    ) == IndexingStructure(True, True, frozenset({"segment"}))


def test_set_index_keeps_set_dimension_when_bare_or_shift() -> None:
    x = var("x")
    provider = _SetStructureProvider()

    assert compute_indexation(x.set_index("fuel"), provider) == IndexingStructure(
        True, True, frozenset({"fuel", "segment"})
    )
    assert compute_indexation(
        x.set_index("fuel", relative_shift=literal(1)), provider
    ) == IndexingStructure(True, True, frozenset({"fuel", "segment"}))


def test_sum_over_removes_set_dimension() -> None:
    x = var("x")
    provider = _SetStructureProvider()

    assert compute_indexation(x.sum_over("fuel"), provider) == IndexingStructure(
        True, True, frozenset({"segment"})
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


def test_lower_upper_bound_indexing() -> None:
    provider = StructureProvider()
    assert compute_indexation(LowerBoundNode("x"), provider) == IndexingStructure(
        True, True
    )
    assert compute_indexation(UpperBoundNode("x"), provider) == IndexingStructure(
        True, True
    )


class _ConstantParamProvider(StructureProvider):
    def get_parameter_structure(self, name: str) -> IndexingStructure:
        return IndexingStructure(False, False)


def test_time_shift_on_non_time_dependent_raises() -> None:
    with pytest.raises(IndexingUsageError, match="not time-dependent"):
        compute_indexation(param("p").shift(1), _ConstantParamProvider())


def test_time_eval_on_non_time_dependent_raises() -> None:
    with pytest.raises(IndexingUsageError, match="not time-dependent"):
        compute_indexation(param("p").eval(1), _ConstantParamProvider())


def test_unresolved_port_field_raises() -> None:
    with pytest.raises(UnresolvedPortFieldError):
        compute_indexation(
            port_field("p", "f").sum_connections(), _ConstantParamProvider()
        )


def test_index_error_reported_alongside_unresolved_port_field() -> None:
    """An invalid time shift is still reported when a sibling operand (before or
    after it) holds an unresolved port field."""
    provider = _ConstantParamProvider()
    port = port_field("p", "f").sum_connections()
    bad = param("p").shift(1)
    with pytest.raises(IndexingUsageError):
        compute_indexation(port + bad, provider)
    with pytest.raises(IndexingUsageError):
        compute_indexation(bad + port, provider)


def test_time_shift_on_unresolved_port_field_is_deferred() -> None:
    with pytest.raises(UnresolvedPortFieldError):
        compute_indexation(
            port_field("p", "f").sum_connections().shift(1), _ConstantParamProvider()
        )


def test_sum_over_unindexed_operand_warns() -> None:
    x = var("x")
    provider = _SetStructureProvider()
    with pytest.warns(UserWarning, match="not indexed by .other."):
        compute_indexation(x.sum_over("other"), provider)
