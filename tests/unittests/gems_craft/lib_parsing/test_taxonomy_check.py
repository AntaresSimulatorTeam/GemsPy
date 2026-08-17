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
from pathlib import Path
from typing import Optional

import pytest

from gems_craft.model.parsing import parse_yaml_library
from gems_craft.model.taxonomy import (
    Taxonomy,
    TaxonomyCategory,
    TaxonomyItem,
    check_library_against_taxonomy,
    load_taxonomy,
)


def _make_taxonomy(*categories: TaxonomyCategory) -> Taxonomy:
    return Taxonomy(id="test_taxonomy", categories=list(categories))


def _make_category(cat_id: str, port_ids: list[str]) -> TaxonomyCategory:
    return TaxonomyCategory(id=cat_id, ports=[TaxonomyItem(id=p) for p in port_ids])


def _parse_lib(yaml_content: str, taxonomy: Optional[Taxonomy] = None):
    return parse_yaml_library(io.StringIO(yaml_content), taxonomy=taxonomy)


# --- valid cases ---


def test_model_with_valid_taxonomy_category_and_matching_port() -> None:
    taxonomy = _make_taxonomy(_make_category("production", ["injection_port"]))
    lib = _parse_lib("""
library:
  id: mylib
  port-types:
    - id: flow
      fields:
        - id: flow
  models:
    - id: generator
      taxonomy-category: production
      ports:
        - id: injection_port
          type: flow
""")
    check_library_against_taxonomy(lib, taxonomy)  # must not raise


def test_model_with_extra_ports_beyond_taxonomy_is_valid() -> None:
    taxonomy = _make_taxonomy(_make_category("production", ["injection_port"]))
    lib = _parse_lib("""
library:
  id: mylib
  port-types:
    - id: flow
      fields:
        - id: flow
  models:
    - id: generator
      taxonomy-category: production
      ports:
        - id: injection_port
          type: flow
        - id: emission_port
          type: flow
""")
    check_library_against_taxonomy(lib, taxonomy)  # extra port is allowed


def test_model_without_taxonomy_category_is_skipped() -> None:
    taxonomy = _make_taxonomy()
    lib = _parse_lib("""
library:
  id: mylib
  port-types:
    - id: flow
      fields:
        - id: flow
  models:
    - id: node
      ports:
        - id: injection_port
          type: flow
""")
    check_library_against_taxonomy(lib, taxonomy)  # must not raise


def test_category_with_no_required_ports_always_passes() -> None:
    taxonomy = _make_taxonomy(_make_category("storage", []))
    lib = _parse_lib("""
library:
  id: mylib
  models:
    - id: battery
      taxonomy-category: storage
""")
    check_library_against_taxonomy(lib, taxonomy)  # no ports required


# --- error cases ---


def test_unknown_taxonomy_category_raises() -> None:
    taxonomy = _make_taxonomy(_make_category("production", []))
    lib = _parse_lib("""
library:
  id: mylib
  models:
    - id: bus
      taxonomy-category: balance
""")
    with pytest.raises(ValueError, match="balance"):
        check_library_against_taxonomy(lib, taxonomy)


def test_model_missing_required_taxonomy_port_raises() -> None:
    taxonomy = _make_taxonomy(
        _make_category("production", ["injection_port", "emission_port"])
    )
    lib = _parse_lib("""
library:
  id: mylib
  port-types:
    - id: flow
      fields:
        - id: flow
  models:
    - id: generator
      taxonomy-category: production
      ports:
        - id: injection_port
          type: flow
""")
    with pytest.raises(ValueError, match="emission_port"):
        check_library_against_taxonomy(lib, taxonomy)


def test_error_message_includes_model_id_and_category() -> None:
    taxonomy = _make_taxonomy(_make_category("balance", ["balance_port"]))
    lib = _parse_lib("""
library:
  id: mylib
  models:
    - id: my_bus
      taxonomy-category: balance
""")
    with pytest.raises(ValueError, match="my_bus") as exc_info:
        check_library_against_taxonomy(lib, taxonomy)
    assert "balance" in str(exc_info.value)
    assert "balance_port" in str(exc_info.value)


# --- load_taxonomy ---


def test_load_taxonomy_from_yaml_file(tmp_path: Path) -> None:
    taxonomy_file = tmp_path / "taxonomy.yml"
    taxonomy_file.write_text("""
taxonomy:
  id: my_taxonomy
  description: "Test taxonomy"
  categories:
    - id: production
      ports:
        - id: injection_port
      parameters:
        - id: p_max
    - id: balance
      ports:
        - id: balance_port
    - id: storage
""")
    taxonomy = load_taxonomy(taxonomy_file)

    assert taxonomy.id == "my_taxonomy"
    assert taxonomy.description == "Test taxonomy"
    assert len(taxonomy.categories) == 3

    production = next(c for c in taxonomy.categories if c.id == "production")
    assert [p.id for p in production.ports] == ["injection_port"]

    balance = next(c for c in taxonomy.categories if c.id == "balance")
    assert [p.id for p in balance.ports] == ["balance_port"]

    storage = next(c for c in taxonomy.categories if c.id == "storage")
    assert storage.ports == []


def test_load_taxonomy_missing_root_key_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.yml"
    bad_file.write_text("categories:\n  - id: foo\n")
    with pytest.raises(ValueError, match="taxonomy"):
        load_taxonomy(bad_file)


def test_load_and_check_roundtrip(tmp_path: Path) -> None:
    taxonomy_file = tmp_path / "taxonomy.yml"
    taxonomy_file.write_text("""
taxonomy:
  id: antares_taxonomy
  categories:
    - id: production
      ports:
        - id: balance_port
    - id: consumption
      ports:
        - id: balance_port
""")
    taxonomy = load_taxonomy(taxonomy_file)
    lib = _parse_lib("""
library:
  id: antares
  port-types:
    - id: flow
      fields:
        - id: flow
  models:
    - id: generator
      taxonomy-category: production
      ports:
        - id: balance_port
          type: flow
    - id: load
      taxonomy-category: consumption
      ports:
        - id: balance_port
          type: flow
""")
    check_library_against_taxonomy(lib, taxonomy)  # must not raise


# --- per-field-group checks (variables, parameters, constraints, ... ) ---


def test_model_missing_required_taxonomy_variable_raises() -> None:
    taxonomy = _make_taxonomy(
        TaxonomyCategory(id="production", variables=[TaxonomyItem(id="generation")])
    )
    lib = _parse_lib("""
library:
  id: mylib
  models:
    - id: generator
      taxonomy-category: production
""")
    with pytest.raises(ValueError, match="generation") as exc_info:
        check_library_against_taxonomy(lib, taxonomy)
    assert "variable" in str(exc_info.value)


def test_model_missing_required_taxonomy_parameter_raises() -> None:
    taxonomy = _make_taxonomy(
        TaxonomyCategory(id="production", parameters=[TaxonomyItem(id="p_max")])
    )
    lib = _parse_lib("""
library:
  id: mylib
  models:
    - id: generator
      taxonomy-category: production
""")
    with pytest.raises(ValueError, match="p_max") as exc_info:
        check_library_against_taxonomy(lib, taxonomy)
    assert "parameter" in str(exc_info.value)


def test_model_missing_required_taxonomy_constraint_raises() -> None:
    taxonomy = _make_taxonomy(
        TaxonomyCategory(id="production", constraints=[TaxonomyItem(id="max_output")])
    )
    lib = _parse_lib("""
library:
  id: mylib
  models:
    - id: generator
      taxonomy-category: production
""")
    with pytest.raises(ValueError, match="max_output") as exc_info:
        check_library_against_taxonomy(lib, taxonomy)
    assert "constraint" in str(exc_info.value)


def test_model_missing_required_taxonomy_binding_constraint_raises() -> None:
    taxonomy = _make_taxonomy(
        TaxonomyCategory(
            id="production",
            binding_constraints=[TaxonomyItem(id="balance")],
        )
    )
    lib = _parse_lib("""
library:
  id: mylib
  models:
    - id: generator
      taxonomy-category: production
""")
    with pytest.raises(ValueError, match="balance") as exc_info:
        check_library_against_taxonomy(lib, taxonomy)
    assert "binding-constraint" in str(exc_info.value)


def test_model_missing_required_taxonomy_extra_output_raises() -> None:
    taxonomy = _make_taxonomy(
        TaxonomyCategory(id="production", extra_outputs=[TaxonomyItem(id="co2")])
    )
    lib = _parse_lib("""
library:
  id: mylib
  models:
    - id: generator
      taxonomy-category: production
""")
    with pytest.raises(ValueError, match="co2") as exc_info:
        check_library_against_taxonomy(lib, taxonomy)
    assert "extra-output" in str(exc_info.value)


def test_model_missing_required_taxonomy_property_raises() -> None:
    taxonomy = _make_taxonomy(
        TaxonomyCategory(id="production", properties=[TaxonomyItem(id="technology")])
    )
    lib = _parse_lib("""
library:
  id: mylib
  models:
    - id: generator
      taxonomy-category: production
""")
    with pytest.raises(ValueError, match="technology") as exc_info:
        check_library_against_taxonomy(lib, taxonomy)
    assert "property" in str(exc_info.value)


def test_model_missing_required_taxonomy_port_field_definition_raises() -> None:
    taxonomy = _make_taxonomy(
        TaxonomyCategory(
            id="production",
            port_field_definitions=[TaxonomyItem(id="injection_port.flow")],
        )
    )
    lib = _parse_lib("""
library:
  id: mylib
  port-types:
    - id: flow
      fields:
        - id: flow
  models:
    - id: generator
      taxonomy-category: production
      ports:
        - id: injection_port
          type: flow
""")
    with pytest.raises(ValueError, match="injection_port.flow") as exc_info:
        check_library_against_taxonomy(lib, taxonomy)
    assert "port-field-definition" in str(exc_info.value)


def test_model_exposing_all_required_fields_is_valid() -> None:
    taxonomy = _make_taxonomy(
        TaxonomyCategory(
            id="production",
            variables=[TaxonomyItem(id="generation")],
            parameters=[TaxonomyItem(id="p_max")],
            ports=[TaxonomyItem(id="injection_port")],
            port_field_definitions=[TaxonomyItem(id="injection_port.flow")],
            constraints=[TaxonomyItem(id="max_output")],
            binding_constraints=[TaxonomyItem(id="balance")],
            extra_outputs=[TaxonomyItem(id="co2")],
            properties=[TaxonomyItem(id="technology")],
        )
    )
    lib = _parse_lib("""
library:
  id: mylib
  port-types:
    - id: flow
      fields:
        - id: flow
  models:
    - id: generator
      taxonomy-category: production
      parameters:
        - id: p_max
      variables:
        - id: generation
      ports:
        - id: injection_port
          type: flow
      port-field-definitions:
        - port: injection_port
          field: flow
          definition: generation
      constraints:
        - id: max_output
          expression: generation <= p_max
      binding-constraints:
        - id: balance
          expression: sum_connections(injection_port.flow) = 0
      extra-outputs:
        - id: co2
          expression: generation
      properties:
        - id: technology
""")
    check_library_against_taxonomy(lib, taxonomy)  # must not raise


# --- parse_yaml_library wiring ---

_CONFORMING_LIB = """
library:
  id: mylib
  taxonomy: test_taxonomy
  port-types:
    - id: flow
      fields:
        - id: flow
  models:
    - id: generator
      taxonomy-category: production
      ports:
        - id: injection_port
          type: flow
"""

_VIOLATING_LIB = """
library:
  id: mylib
  taxonomy: test_taxonomy
  models:
    - id: generator
      taxonomy-category: production
"""


def test_parse_library_declaring_taxonomy_is_checked() -> None:
    taxonomy = _make_taxonomy(_make_category("production", ["injection_port"]))
    lib = _parse_lib(_CONFORMING_LIB, taxonomy)
    assert lib.taxonomy == "test_taxonomy"


def test_parse_library_declaring_taxonomy_raises_on_violation() -> None:
    taxonomy = _make_taxonomy(_make_category("production", ["injection_port"]))
    with pytest.raises(ValueError, match="injection_port"):
        _parse_lib(_VIOLATING_LIB, taxonomy)


def test_parse_library_declaring_taxonomy_without_argument_raises() -> None:
    with pytest.raises(ValueError, match="no taxonomy was provided"):
        _parse_lib(_CONFORMING_LIB)


def test_parse_library_with_mismatched_taxonomy_id_raises() -> None:
    taxonomy = Taxonomy(
        id="other_taxonomy",
        categories=[_make_category("production", ["injection_port"])],
    )
    with pytest.raises(ValueError, match="other_taxonomy"):
        _parse_lib(_CONFORMING_LIB, taxonomy)


def test_parse_library_without_declared_taxonomy_is_not_checked() -> None:
    """A model may carry a taxonomy-category without the library opting in."""
    taxonomy = _make_taxonomy(_make_category("production", ["injection_port"]))
    lib = _parse_lib(
        """
library:
  id: mylib
  models:
    - id: generator
      taxonomy-category: production
""",
        taxonomy,
    )
    assert lib.taxonomy is None
