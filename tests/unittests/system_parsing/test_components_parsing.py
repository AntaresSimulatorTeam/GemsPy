from pathlib import Path

import pytest

from gems.model.parsing import InputLibrary, parse_yaml_library
from gems.model.resolve_library import resolve_library
from gems.study.parsing import InputSystem, parse_yaml_components
from gems.study.resolve_components import consistency_check, resolve_system


@pytest.fixture
def input_system() -> InputSystem:
    compo_file = Path(__file__).parent / "systems/system.yml"

    with compo_file.open() as c:
        return parse_yaml_components(c)


@pytest.fixture
def input_library() -> InputLibrary:
    library = Path(__file__).parent / "libs/lib_unittest.yml"

    with library.open() as lib:
        return parse_yaml_library(lib)


def test_parsing_components_ok(
    input_system: InputSystem, input_library: InputLibrary
) -> None:
    assert len(input_system.components) == 2
    assert len(input_system.nodes) == 1
    assert len(input_system.connections) == 2
    lib_dict = resolve_library([input_library])
    result = resolve_system(input_system, lib_dict)

    assert len(result.components) == 2
    assert len(result.nodes) == 1
    assert len(result.connections) == 2


def test_consistency_check_ok(
    input_system: InputSystem, input_library: InputLibrary
) -> None:
    result_lib = resolve_library([input_library])
    result_system = resolve_system(input_system, result_lib)
    consistency_check(result_system.components, result_lib["basic"].models)


def test_consistency_check_ko(
    input_system: InputSystem, input_library: InputLibrary
) -> None:
    result_lib = resolve_library([input_library])
    result_comp = resolve_system(input_system, result_lib)
    result_lib["basic"].models.pop("generator")
    with pytest.raises(
        ValueError,
        match=r"Error: Component G has invalid model ID: generator",
    ):
        consistency_check(result_comp.components, result_lib["basic"].models)
