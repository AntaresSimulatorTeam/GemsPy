from pathlib import Path

import pytest
from yaml import dump, safe_load

from gems.expression import literal, maximum, var
from gems.expression.expression import port_field
from gems.model import Constraint, ModelPort, PortType, model
from gems.model.parsing import LibrarySchema, parse_yaml_library
from gems.model.port import PortField, PortFieldDefinition, PortFieldId
from gems.model.resolve_library import resolve_library
from gems.model.variable import float_variable
from gems.study import Component, PortRef, PortsConnection
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


# ---------------------------------------------------------------------------
# sum_connections linearity checks
# ---------------------------------------------------------------------------

_MY_PORT_TYPE = PortType(id="my_port_type", fields=[PortField("flow")])

_NONLINEAR_GENERATOR = model(
    id="NONLINEAR_GEN",
    variables=[float_variable("x"), float_variable("y")],
    ports=[ModelPort(port_type=_MY_PORT_TYPE, port_name="my_port")],
    port_fields_definitions=[
        PortFieldDefinition(
            port_field=PortFieldId("my_port", "flow"),
            definition=maximum(var("x"), var("y")),
        )
    ],
)

_LINEAR_GENERATOR = model(
    id="LINEAR_GEN",
    variables=[float_variable("x")],
    ports=[ModelPort(port_type=_MY_PORT_TYPE, port_name="my_port")],
    port_fields_definitions=[
        PortFieldDefinition(
            port_field=PortFieldId("my_port", "flow"),
            definition=var("x"),
        )
    ],
)

_NODE_WITH_SUM_CONNECTIONS = model(
    id="NODE",
    ports=[ModelPort(port_type=_MY_PORT_TYPE, port_name="my_port")],
    binding_constraints=[
        Constraint(
            name="Balance",
            expression=port_field("my_port", "flow").sum_connections() == literal(0),
        )
    ],
)


def test_sum_connections_with_nonlinear_port_field_raises() -> None:
    gen = Component(id="G", model=_NONLINEAR_GENERATOR)
    node = Component(id="N", model=_NODE_WITH_SUM_CONNECTIONS)
    with pytest.raises(ValueError, match="non-linear"):
        PortsConnection(PortRef(gen, "my_port"), PortRef(node, "my_port"))


def test_sum_connections_with_linear_port_field_ok() -> None:
    gen = Component(id="G", model=_LINEAR_GENERATOR)
    node = Component(id="N", model=_NODE_WITH_SUM_CONNECTIONS)
    PortsConnection(PortRef(gen, "my_port"), PortRef(node, "my_port"))
