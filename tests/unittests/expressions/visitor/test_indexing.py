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


from gems.expression import param, var
from gems.expression.indexing import IndexingStructureProvider, compute_indexation
from gems.expression.indexing_structure import IndexingStructure


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

    provider = CustomStructureProvider()

    assert compute_indexation(expr, provider) == IndexingStructure(True, True)
