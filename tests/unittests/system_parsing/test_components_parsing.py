from pathlib import Path

import pytest
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
