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

from gems_craft.expression.indexing_structure import IndexingStructure


def test_or_unions_sets() -> None:
    a = IndexingStructure(True, False, frozenset({"fuel"}))
    b = IndexingStructure(False, True, frozenset({"segment"}))
    assert a | b == IndexingStructure(True, True, frozenset({"fuel", "segment"}))


def test_or_with_default_empty_sets() -> None:
    assert IndexingStructure(True, True) == IndexingStructure(True, True, frozenset())


def test_is_set_varying() -> None:
    assert IndexingStructure(False, False, frozenset({"fuel"})).is_set_varying()
    assert not IndexingStructure(False, False).is_set_varying()


def test_is_constant_accounts_for_sets() -> None:
    assert IndexingStructure(False, False).is_constant()
    assert not IndexingStructure(False, False, frozenset({"fuel"})).is_constant()
