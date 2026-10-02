# Copyright (c) 2026, RTE (https://www.rte-france.com)
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

import io

import pytest

from gems_craft.model.parsing import parse_yaml_library
from gems_craft.model.resolve_library import resolve_library
from gems_craft.study.parsing import parse_yaml_system
from gems_craft.study.validation import check_custom_sets

_LIB_WITH_GLOBAL_AND_LOCAL_SET = """\
library:
  id: setlib
  sets:
    - id: fuel
  models:
    - id: m
      sets:
        - id: segment
      parameters:
        - id: p
          indexed-by: [fuel, segment]
"""


def _resolve(lib_yaml: str):
    lib = parse_yaml_library(io.StringIO(lib_yaml))
    lib_dict = resolve_library([lib])
    model_dict = {}
    for library in lib_dict.values():
        model_dict |= library.models
    return model_dict, lib_dict


def _parse_system(system_yaml: str):
    return parse_yaml_system(io.StringIO(system_yaml))


_SYSTEM_OK = """\
system:
  sets:
    - id: fuel
      elements: [gas, coal]
  components:
    - id: A
      model: setlib.m
      sets:
        - id: segment
          elements: 0..2
      parameters:
        - id: p
          value: 1.0
"""


def test_valid_global_and_local_set_instantiation_ok() -> None:
    model_dict, lib_dict = _resolve(_LIB_WITH_GLOBAL_AND_LOCAL_SET)
    system = _parse_system(_SYSTEM_OK)
    check_custom_sets(system, model_dict, lib_dict)  # must not raise


_SYSTEM_MISSING_GLOBAL = """\
system:
  components:
    - id: A
      model: setlib.m
      sets:
        - id: segment
          elements: 0..2
      parameters:
        - id: p
          value: 1.0
"""


def test_missing_global_set_instantiation_raises() -> None:
    model_dict, lib_dict = _resolve(_LIB_WITH_GLOBAL_AND_LOCAL_SET)
    system = _parse_system(_SYSTEM_MISSING_GLOBAL)
    with pytest.raises(ValueError, match="not instantiated"):
        check_custom_sets(system, model_dict, lib_dict)


_SYSTEM_EXTRA_GLOBAL = """\
system:
  sets:
    - id: fuel
      elements: [gas, coal]
    - id: unknown_set
      elements: [a, b]
  components:
    - id: A
      model: setlib.m
      sets:
        - id: segment
          elements: 0..2
      parameters:
        - id: p
          value: 1.0
"""


def test_extra_unused_global_set_instantiation_is_allowed() -> None:
    """Instantiating a set nothing actually uses is harmless (mirrors extra,
    undeclared component properties/parameters also being allowed)."""
    model_dict, lib_dict = _resolve(_LIB_WITH_GLOBAL_AND_LOCAL_SET)
    system = _parse_system(_SYSTEM_EXTRA_GLOBAL)
    check_custom_sets(system, model_dict, lib_dict)  # must not raise


def test_unused_global_set_not_required_to_be_instantiated() -> None:
    """A library-declared global set that no model actually references via
    indexed_by doesn't need to be instantiated (mirrors how declaring a
    parameter time-dependent never forces the system to actually vary it)."""
    lib = parse_yaml_library(io.StringIO("""
library:
  id: unused_set_lib
  sets:
    - id: fuel
  models:
    - id: m
      parameters:
        - id: p
"""))
    lib_dict = resolve_library([lib])
    model_dict = {}
    for library in lib_dict.values():
        model_dict |= library.models
    system = _parse_system("""
system:
  components:
    - id: A
      model: unused_set_lib.m
      parameters:
        - id: p
          value: 1.0
""")
    check_custom_sets(system, model_dict, lib_dict)  # must not raise


_SYSTEM_MISSING_LOCAL = """\
system:
  sets:
    - id: fuel
      elements: [gas, coal]
  components:
    - id: A
      model: setlib.m
      parameters:
        - id: p
          value: 1.0
"""


def test_missing_local_set_instantiation_raises() -> None:
    model_dict, lib_dict = _resolve(_LIB_WITH_GLOBAL_AND_LOCAL_SET)
    system = _parse_system(_SYSTEM_MISSING_LOCAL)
    with pytest.raises(ValueError, match="missing instantiation"):
        check_custom_sets(system, model_dict, lib_dict)


_SYSTEM_DUPLICATE_ELEMENTS = """\
system:
  sets:
    - id: fuel
      elements: [gas, gas]
  components:
    - id: A
      model: setlib.m
      sets:
        - id: segment
          elements: 0..2
      parameters:
        - id: p
          value: 1.0
"""


def test_duplicate_set_elements_raises() -> None:
    model_dict, lib_dict = _resolve(_LIB_WITH_GLOBAL_AND_LOCAL_SET)
    system = _parse_system(_SYSTEM_DUPLICATE_ELEMENTS)
    with pytest.raises(ValueError, match="duplicate elements"):
        check_custom_sets(system, model_dict, lib_dict)
