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

import pytest

from gems.model.parsing import parse_yaml_library
from gems.model.taxonomy import (
    Taxonomy,
    TaxonomyCategory,
    TaxonomyItem,
    check_library_against_taxonomy,
    load_taxonomy,
)


def _make_taxonomy(*categories: TaxonomyCategory) -> Taxonomy:
    return Taxonomy(id="test_taxonomy", categories=list(categories))


def _make_category(cat_id: str, port_ids: list[str]) -> TaxonomyCategory:
    return TaxonomyCategory(
        id=cat_id, ports=[TaxonomyItem(id=p) for p in port_ids]
    )


def _parse_lib(yaml_content: str):
    return parse_yaml_library(io.StringIO(yaml_content))


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
