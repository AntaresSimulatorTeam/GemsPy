import io
from pathlib import Path

import pytest
from pydantic import ValidationError
from yaml import dump, safe_load

from gems.model.parsing import LibrarySchema, parse_yaml_library
from gems.model.resolve_library import resolve_library
from gems.study.parsing import SystemSchema, load_input_system, parse_yaml_components
from gems.study.resolve_components import consistency_check, resolve_system

COMPO_FILE = Path(__file__).parent / "systems/system.yml"


@pytest.fixture
def input_system() -> SystemSchema:
    with COMPO_FILE.open() as c:
        return parse_yaml_components(c)


@pytest.fixture
def input_library() -> LibrarySchema:
    library = Path(__file__).parent / "libs/lib_unittest.yml"

    with library.open() as lib:
        return parse_yaml_library(lib)


def test_parsing_components_ok(
    input_system: SystemSchema, input_library: LibrarySchema
) -> None:
    assert len(input_system.components) == 3
    assert input_system.connections is not None
    assert len(input_system.connections) == 2
    lib_dict = resolve_library([input_library])
    result = resolve_system(input_system, lib_dict)

    assert len(result.components) == 3
    assert len(result.connections) == 2


def test_consistency_check_ok(
    input_system: SystemSchema, input_library: LibrarySchema
) -> None:
    result_lib = resolve_library([input_library])
    result_system = resolve_system(input_system, result_lib)
    consistency_check(result_system, result_lib["basic"].models)


def test_load_input_system_ok(tmp_path: Path) -> None:
    data = safe_load(COMPO_FILE.read_text())
    system_only = data["system"]
    file_for_load = tmp_path / "system.yml"
    file_for_load.write_text(dump(system_only))

    result = load_input_system(file_for_load)

    assert isinstance(result, SystemSchema)
    assert len(result.components) == 3
    assert result.components[0].id == "N"
    assert result.components[1].id == "G"
    assert result.components[2].id == "D"
    assert result.connections is not None
    assert len(result.connections) == 2


def test_load_input_system_invalid_yaml_raises_value_error(tmp_path: Path) -> None:
    data = safe_load(COMPO_FILE.read_text())
    system_only = data["system"].copy()
    system_only["unknown_field"] = "not_allowed"
    bad_file = tmp_path / "system.yml"
    bad_file.write_text(dump(system_only))

    with pytest.raises(ValueError, match="An error occurred during parsing"):
        load_input_system(bad_file)


def test_load_input_system_missing_file_raises_error() -> None:
    missing = Path(__file__).parent / "systems/does_not_exist.yml"

    with pytest.raises(FileNotFoundError):
        load_input_system(missing)


def test_consistency_check_ko(
    input_system: SystemSchema, input_library: LibrarySchema
) -> None:
    result_lib = resolve_library([input_library])
    result_comp = resolve_system(input_system, result_lib)
    result_lib["basic"].models.pop("basic.generator")
    with pytest.raises(
        ValueError,
        match=r"Error: Component G has invalid model ID: basic.generator",
    ):
        consistency_check(result_comp, result_lib["basic"].models)


_SYSTEM_WITH_COMPONENT_PROPERTIES = """\
system:
  id: basic_example
  components:
    - id: load
      model: basic.demand
      parameters:
        - id: demand
          time-dependent: true
          scenario-dependent: true
          value: load_data
    - id: nuclear_1
      model: basic.generator
      properties:
        - id: technology
          value: nuclear
        - id: company
          value: rhonepower
"""


def test_parse_yaml_components_properties_optional_and_normalized() -> None:
    system = parse_yaml_components(io.StringIO(_SYSTEM_WITH_COMPONENT_PROPERTIES))
    props_by_id = {c.id: c.properties for c in system.components}
    assert props_by_id["load"] is None
    raw = props_by_id["nuclear_1"]
    assert isinstance(raw, list) and len(raw) == 2
    assert [p.model_dump() for p in raw] == [
        {"key": "technology", "value": "nuclear"},
        {"key": "company", "value": "rhonepower"},
    ]


def test_resolve_system_normalizes_list_properties_to_dict(
    input_library: LibrarySchema,
) -> None:
    system = parse_yaml_components(io.StringIO(_SYSTEM_WITH_COMPONENT_PROPERTIES))
    lib_dict = resolve_library([input_library])
    resolved = resolve_system(system, lib_dict)
    assert resolved.get_component("nuclear_1").properties == {
        "technology": "nuclear",
        "company": "rhonepower",
    }


_SYSTEM_WITH_PROPERTIES_MISSING_KEY = """\
system:
  components:
    - id: A
      model: basic.area
      properties:
        - value: nuclear
"""


def test_parse_yaml_components_properties_missing_key_raises() -> None:
    with pytest.raises(ValidationError):
        parse_yaml_components(io.StringIO(_SYSTEM_WITH_PROPERTIES_MISSING_KEY))


_SYSTEM_WITH_PROPERTIES_DUPLICATE_KEYS = """\
system:
  components:
    - id: A
      model: basic.node
      properties:
        - id: technology
          value: nuclear
        - id: technology
          value: gas
"""


def test_resolve_component_properties_duplicate_keys_raises(
    input_library: LibrarySchema,
) -> None:
    system = parse_yaml_components(io.StringIO(_SYSTEM_WITH_PROPERTIES_DUPLICATE_KEYS))
    lib_dict = resolve_library([input_library])
    with pytest.raises(ValueError, match="duplicate properties key"):
        resolve_system(system, lib_dict)


_SYSTEM_WITH_SYSTEM_LEVEL_PROPERTIES = """\
system:
  properties:
    technology: nuclear
  components:
    - id: A
      model: basic.area
"""


def test_parse_yaml_components_system_level_properties_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_yaml_components(io.StringIO(_SYSTEM_WITH_SYSTEM_LEVEL_PROPERTIES))
