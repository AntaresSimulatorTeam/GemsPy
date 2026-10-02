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


def _parse_lib(yaml_content: str):
    return parse_yaml_library(io.StringIO(yaml_content))


# --- local set / global set naming collision ---


def test_local_set_colliding_with_global_set_raises() -> None:
    lib = _parse_lib("""
library:
  id: collision_lib
  sets:
    - id: fuel
  models:
    - id: m
      sets:
        - id: fuel
      parameters:
        - id: p
""")
    with pytest.raises(ValueError, match="collide with"):
        resolve_library([lib])


def test_local_set_not_colliding_with_unrelated_global_set_ok() -> None:
    lib = _parse_lib("""
library:
  id: ok_lib
  sets:
    - id: fuel
  models:
    - id: m
      sets:
        - id: segment
      parameters:
        - id: p
""")
    resolve_library([lib])  # must not raise


# --- duplicate global set across a dependency chain ---


def test_duplicate_global_set_across_dependency_raises() -> None:
    lib_a = _parse_lib("""
library:
  id: libA
  sets:
    - id: fuel
  models:
    - id: m
""")
    lib_b = _parse_lib("""
library:
  id: libB
  dependencies: [libA]
  sets:
    - id: fuel
  models:
    - id: n
""")
    with pytest.raises(Exception, match="defined twice"):
        resolve_library([lib_a, lib_b])


# --- indexed-by referencing an undeclared set ---


def test_indexed_by_undeclared_set_raises() -> None:
    lib = _parse_lib("""
library:
  id: undecl_lib
  models:
    - id: m
      parameters:
        - id: p
          indexed-by: [fuel]
""")
    with pytest.raises(ValueError, match="undeclared set"):
        resolve_library([lib])


# --- dimension-existence: set index on a non-indexed-by identifier ---


def test_set_index_on_non_indexed_by_variable_raises() -> None:
    lib = _parse_lib("""
library:
  id: dim_lib
  sets:
    - id: fuel
  models:
    - id: m
      parameters:
        - id: p
      variables:
        - id: x
      constraints:
        - id: c
          expression: x[fuel] <= p
""")
    with pytest.raises(ValueError, match="not indexed by 'fuel'"):
        resolve_library([lib])


def test_set_index_on_indexed_by_variable_ok() -> None:
    lib = _parse_lib("""
library:
  id: dim_ok_lib
  sets:
    - id: fuel
  models:
    - id: m
      parameters:
        - id: p
      variables:
        - id: x
          indexed-by: [fuel]
      constraints:
        - id: c
          expression: x[fuel] <= p
""")
    resolve_library([lib])  # must not raise


# --- dimension-existence: absolute time index on a non-time-dependent identifier ---


def test_time_eval_on_non_time_dependent_parameter_raises() -> None:
    lib = _parse_lib("""
library:
  id: time_dim_lib
  models:
    - id: m
      parameters:
        - id: p
          time-dependent: false
      variables:
        - id: x
      constraints:
        - id: c
          expression: p[5] <= x
""")
    with pytest.raises(ValueError, match="not time-dependent"):
        resolve_library([lib])


def test_time_eval_on_time_dependent_parameter_ok() -> None:
    lib = _parse_lib("""
library:
  id: time_dim_ok_lib
  models:
    - id: m
      parameters:
        - id: p
          time-dependent: true
      variables:
        - id: x
      constraints:
        - id: c
          expression: p[5] <= x
""")
    resolve_library([lib])  # must not raise


def test_relative_time_shift_on_non_time_dependent_parameter_raises() -> None:
    """A relative shift (X[t+1]/X.shift(...)) on a non-time-dependent identifier
    is rejected the same way as the absolute/explicit-position form (X[5])."""
    lib = _parse_lib("""
library:
  id: time_shift_lib
  models:
    - id: m
      parameters:
        - id: p
          time-dependent: false
      variables:
        - id: x
      constraints:
        - id: c
          expression: p[t+1] <= x
""")
    with pytest.raises(ValueError, match="not time-dependent"):
        resolve_library([lib])


def test_relative_time_shift_on_time_dependent_parameter_ok() -> None:
    lib = _parse_lib("""
library:
  id: time_shift_ok_lib
  models:
    - id: m
      parameters:
        - id: p
          time-dependent: true
      variables:
        - id: x
      constraints:
        - id: c
          expression: p[t+1] <= x
""")
    resolve_library([lib])  # must not raise


# --- bound set-consistency ---


def test_bound_indexed_by_set_variable_lacks_raises() -> None:
    lib = _parse_lib("""
library:
  id: bound_lib
  sets:
    - id: fuel
  models:
    - id: m
      parameters:
        - id: cap
          indexed-by: [fuel]
      variables:
        - id: x
          upper-bound: cap[fuel]
""")
    with pytest.raises(ValueError, match="not declared in 'x'"):
        resolve_library([lib])


def test_bound_summed_over_set_variable_lacks_is_ok() -> None:
    lib = _parse_lib("""
library:
  id: bound_ok_lib
  sets:
    - id: fuel
  models:
    - id: m
      parameters:
        - id: cap
          indexed-by: [fuel]
      variables:
        - id: x
          upper-bound: sum_over(fuel, cap[fuel])
""")
    resolve_library([lib])  # must not raise


def test_bound_indexed_by_set_variable_also_indexed_by_is_ok() -> None:
    lib = _parse_lib("""
library:
  id: bound_ok_lib
  sets:
    - id: fuel
  models:
    - id: m
      parameters:
        - id: cap
          indexed-by: [fuel]
      variables:
        - id: x
          indexed-by: [fuel]
          upper-bound: cap[fuel]
""")
    resolve_library([lib])  # must not raise


# --- local sets cannot cross a port unless summed out first ---


def test_port_field_definition_indexed_by_local_set_raises() -> None:
    lib = _parse_lib("""
library:
  id: port_lib
  port-types:
    - id: flow
      fields:
        - id: flow
  models:
    - id: m
      sets:
        - id: segment
      parameters:
        - id: gen
          indexed-by: [segment]
      ports:
        - id: injection_port
          type: flow
      port-field-definitions:
        - port: injection_port
          field: flow
          definition: gen[segment]
""")
    with pytest.raises(ValueError, match="wrap it in sum_over"):
        resolve_library([lib])


def test_port_field_definition_summed_over_local_set_ok() -> None:
    lib = _parse_lib("""
library:
  id: port_ok_lib
  port-types:
    - id: flow
      fields:
        - id: flow
  models:
    - id: m
      sets:
        - id: segment
      parameters:
        - id: gen
          indexed-by: [segment]
      ports:
        - id: injection_port
          type: flow
      port-field-definitions:
        - port: injection_port
          field: flow
          definition: sum_over(segment, gen[segment])
""")
    resolve_library([lib])  # must not raise


def test_port_field_definition_indexed_by_global_set_ok() -> None:
    lib = _parse_lib("""
library:
  id: port_global_lib
  sets:
    - id: fuel
  port-types:
    - id: flow
      fields:
        - id: flow
  models:
    - id: m
      parameters:
        - id: gen
          indexed-by: [fuel]
      ports:
        - id: injection_port
          type: flow
      port-field-definitions:
        - port: injection_port
          field: flow
          definition: gen[fuel]
""")
    resolve_library([lib])  # global sets are exempt, must not raise
