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

from gems.utils import get_or_add


def test_get_or_add_should_evaluate_lazily() -> None:
    d = {"key1": "value1"}

    def raise_factory() -> str:
        raise AssertionError("No value should be created")

    assert get_or_add(d, "key1", raise_factory) == "value1"
    with pytest.raises(AssertionError, match="No value should be created"):
        get_or_add(d, "key2", raise_factory)

    def value_factory() -> str:
        return "value2"

    assert get_or_add(d, "key2", value_factory) == "value2"
