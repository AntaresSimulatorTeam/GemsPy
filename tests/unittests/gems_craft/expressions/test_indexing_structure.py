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


def test_set_varying_and_constant() -> None:
    assert IndexingStructure(True, True) == IndexingStructure(True, True, frozenset())
    with_set = IndexingStructure(False, False, frozenset({"fuel"}))
    assert with_set.is_set_varying() and not with_set.is_constant()
    assert not IndexingStructure(False, False).is_set_varying()
    assert IndexingStructure(False, False).is_constant()
