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
import io
from pathlib import Path

import pytest

from gems.expression import literal, param, var
from gems.expression.equality import expressions_equal, expressions_equal_if_present
from gems.expression.expression import (
    DualNode,
    ReducedCostNode,
    maximum,
    minimum,
    port_field,
)
from gems.expression.indexing_structure import IndexingStructure
from gems.expression.parsing.parse_expression import ParsingException
from gems.model import (
    Constraint,
    ModelPort,
    PortField,
    PortType,
    ValueType,
    float_parameter,
    float_variable,
    model,
)
from gems.model.model import PortFieldDefinition, PortFieldId
from gems.model.parsing import parse_yaml_library
from gems.model.resolve_library import resolve_library

CONSTANT = IndexingStructure(False, False)


def test_library_parsing(libs_dir: Path) -> None:
    lib_file = libs_dir / "lib_unittest.yml"

    with lib_file.open() as f:
        input_lib = parse_yaml_library(f)
    assert input_lib.id == "basic"
    assert len(input_lib.models) == 7
    assert len(input_lib.port_types) == 1

    lib = resolve_library([input_lib])
    assert len(lib) == 1
    assert len(lib[input_lib.id].models) == 7
    assert len(lib[input_lib.id].port_types) == 1
    port_type = lib[input_lib.id].port_types["flow"]
    assert port_type == PortType(id="flow", fields=[PortField(name="flow")])
    gen_model = lib[input_lib.id].models["basic.generator"]
    assert gen_model == model(
        id="basic.generator",
        parameters=[
            float_parameter("cost", structure=CONSTANT),
            float_parameter("p_max", structure=CONSTANT),
        ],
        variables=[
            float_variable(
                "generation", lower_bound=literal(0), upper_bound=param("p_max")
            )
        ],
        ports=[ModelPort(port_type=port_type, port_name="injection_port")],
        port_fields_definitions=[
            PortFieldDefinition(
                port_field=PortFieldId(port_name="injection_port", field_name="flow"),
                definition=var("generation"),
            )
        ],
        objective_contributions={
            "operational": (param("cost") * var("generation")).time_sum().expec()
        },
    )
    short_term_storage = lib[input_lib.id].models["basic.short-term-storage"]
    assert short_term_storage == model(
        id="basic.short-term-storage",
        parameters=[
            float_parameter("efficiency", structure=CONSTANT),
            float_parameter("level_min", structure=CONSTANT),
            float_parameter("level_max", structure=CONSTANT),
            float_parameter("p_max_withdrawal", structure=CONSTANT),
            float_parameter("p_max_injection", structure=CONSTANT),
            float_parameter("inflows", structure=CONSTANT),
        ],
        variables=[
            float_variable(
                "injection",
                lower_bound=literal(0),
                upper_bound=param("p_max_injection"),
            ),
            float_variable(
                "withdrawal",
                lower_bound=literal(0),
                upper_bound=param("p_max_withdrawal"),
            ),
            float_variable(
                "level",
                lower_bound=param("level_min"),
                upper_bound=param("level_max"),
            ),
        ],
        ports=[ModelPort(port_type=port_type, port_name="injection_port")],
        port_fields_definitions=[
            PortFieldDefinition(
                port_field=PortFieldId(port_name="injection_port", field_name="flow"),
                definition=var("injection") - var("withdrawal"),
            )
        ],
        constraints=[
            Constraint(
                name="Level equation",
                expression=var("level")
                - var("level").shift(-literal(1))
                - param("efficiency") * var("injection")
                + var("withdrawal")
                == param("inflows"),
            )
        ],
    )


def test_binary_variable_parsing() -> None:
    yaml_content = """
library:
  id: test
  models:
    - id: binary_model
      variables:
        - id: on_off
          variable-type: binary
"""
    input_lib = parse_yaml_library(io.StringIO(yaml_content))
    lib = resolve_library([input_lib])
    on_off = lib["test"].models["test.binary_model"].variables["on_off"]
    assert on_off.data_type == ValueType.BINARY


def test_library_error_parsing(libs_dir: Path) -> None:
    lib_file = libs_dir / "model_port_definition_ko.yml"

    with lib_file.open() as f:
        input_lib = parse_yaml_library(f)
    assert input_lib.id == "basic"
    with pytest.raises(
        ParsingException,
        match=r"An error occurred during parsing: ParseCancellationException",
    ):
        resolve_library([input_lib])


def test_library_port_model_ok_parsing(libs_dir: Path) -> None:
    lib_file = libs_dir / "model_port_definition_ok.yml"

    with lib_file.open() as f:
        input_lib = parse_yaml_library(f)
    assert input_lib.id == "basic"

    lib = resolve_library([input_lib])
    port_type = lib[input_lib.id].port_types["flow"]
    assert port_type == PortType(id="flow", fields=[PortField(name="flow")])
    short_term_storage = lib[input_lib.id].models["basic.short-term-storage-2"]
    assert short_term_storage == model(
        id="basic.short-term-storage-2",
        parameters=[
            float_parameter("p_max_withdrawal", structure=CONSTANT),
            float_parameter("p_max_injection", structure=CONSTANT),
        ],
        variables=[
            float_variable(
                "injection",
                lower_bound=literal(0),
                upper_bound=param("p_max_injection"),
            ),
            float_variable(
                "withdrawal",
                lower_bound=literal(0),
                upper_bound=param("p_max_withdrawal"),
            ),
        ],
        ports=[ModelPort(port_type=port_type, port_name="injection_port")],
        constraints=[
            Constraint(
                name="Level equation",
                expression=port_field("injection_port", "flow").sum_connections()
                == var("withdrawal"),
            )
        ],
    )


def test_dual_in_constraint_is_rejected() -> None:
    lib_yaml = io.StringIO("""
library:
  id: basic
  port-types: []
  models:
    - id: bad-model
      variables:
        - id: x
          variable-type: continuous
      parameters: []
      ports: []
      binding-constraints:
        - id: balance
          expression: x >= 0
      constraints:
        - id: bad
          expression: dual(balance) + x = 0
""")
    input_lib = parse_yaml_library(lib_yaml)
    with pytest.raises(ValueError, match="Non-linear expression is not allowed"):
        resolve_library([input_lib])


def test_reduced_cost_in_objective_is_rejected() -> None:
    lib_yaml = io.StringIO("""
library:
  id: basic
  port-types: []
  models:
    - id: bad-model
      variables:
        - id: x
          variable-type: continuous
      parameters: []
      ports: []
      objective-contributions:
        - id: bad-obj
          expression: reduced_cost(x)
""")
    input_lib = parse_yaml_library(lib_yaml)
    with pytest.raises(ValueError, match="Non-linear expression is not allowed"):
        resolve_library([input_lib])


# ---------------------------------------------------------------------------
# Parametrized expression sets
# ---------------------------------------------------------------------------

_PFIELD_ID = PortFieldId("port1", "flow")
_SUM_CONNECTIONS_EXPR = port_field("port1", "flow").sum_connections()

# Nonlinear/post-solve expressions accepted in port-field definitions and
# extra-outputs only.  Rejected in: variable bounds (not constant), objectives
# (nonlinear or dual/rc forbidden), and binding-constraint expressions
# (dual/reduced_cost via _forbid_dual_or_rc).
_UNRESTRICTED_EXPRS = [
    pytest.param("dual(balance)", DualNode("balance"), id="dual"),
    pytest.param("reduced_cost(x)", ReducedCostNode("x"), id="reduced_cost"),
    pytest.param("max(x, y)", maximum(var("x"), var("y")), id="max"),
    pytest.param("min(x, y)", minimum(var("x"), var("y")), id="min"),
    pytest.param("abs(x)", var("x").abs(), id="abs"),
    pytest.param("floor(x)", var("x").floor(), id="floor"),
    pytest.param("ceil(x)", var("x").ceil(), id="ceil"),
]

# Parameter-only expressions (degree 0).  Accepted in every context:
# port-field defs, extra-outputs, variable bounds, objectives, and constraints.
_CONSTANT_EXPRS = [
    pytest.param("max(a, b)", maximum(param("a"), param("b")), id="max"),
    pytest.param("min(a, b)", minimum(param("a"), param("b")), id="min"),
    pytest.param("abs(a)", param("a").abs(), id="abs"),
    pytest.param("floor(a)", param("a").floor(), id="floor"),
    pytest.param("ceil(a)", param("a").ceil(), id="ceil"),
]

# Degree-1 variable expressions.  Rejected in variable bounds (not constant);
# accepted in port-field defs, extra-outputs, objectives, and constraints.
_LINEAR_EXPRS = [
    pytest.param("a * x", param("a") * var("x"), id="param_times_var"),
    pytest.param("x + y", var("x") + var("y"), id="sum_vars"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_lib_yaml(pfd_expr: str) -> str:
    """Model with variables x,y, params a,b, one port, binding constraint 'balance'."""
    return f"""
library:
  id: test
  port-types:
    - id: flow
      fields:
        - id: flow
  models:
    - id: test_model
      variables:
        - id: x
          variable-type: continuous
        - id: y
          variable-type: continuous
      parameters:
        - id: a
        - id: b
      ports:
        - id: port1
          type: flow
      binding-constraints:
        - id: balance
          expression: x >= 0
      port-field-definitions:
        - port: port1
          field: flow
          definition: {pfd_expr}
"""


def _no_port_lib_yaml(
    *, var_bound: str = "", constraint_expr: str = "", objective_expr: str = ""
) -> str:
    """Minimal model without ports, for bound/constraint/objective tests.

    Always includes a 'balance' binding constraint so that dual(balance) and
    reduced_cost(x) are syntactically valid in any expression position.
    An optional second binding constraint 'bc1' can be added via constraint_expr.
    """
    bound_section = f"\n          upper-bound: {var_bound}" if var_bound else ""
    extra_bc = (
        f"\n        - id: bc1\n          expression: {constraint_expr}"
        if constraint_expr
        else ""
    )
    objective_section = (
        f"\n      objective-contributions:\n        - id: cost\n          expression: {objective_expr}"
        if objective_expr
        else ""
    )
    return f"""
library:
  id: test
  port-types: []
  models:
    - id: test_model
      variables:
        - id: x
          variable-type: continuous{bound_section}
        - id: y
          variable-type: continuous
      parameters:
        - id: a
        - id: b
      ports: []
      binding-constraints:
        - id: balance
          expression: x >= 0{extra_bc}{objective_section}
"""


# ---------------------------------------------------------------------------
# Acceptance tests: port-field definitions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "yaml_expr,expected_expr",
    _UNRESTRICTED_EXPRS + _CONSTANT_EXPRS + _LINEAR_EXPRS,
)
def test_expr_in_port_field_definition(yaml_expr: str, expected_expr: object) -> None:
    input_lib = parse_yaml_library(io.StringIO(_base_lib_yaml(yaml_expr)))
    lib = resolve_library([input_lib])
    pfd = lib["test"].models["test.test_model"].port_fields_definitions[_PFIELD_ID]
    assert expressions_equal(pfd.definition, expected_expr)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Acceptance tests: variable bounds (_CONSTANT_EXPRS only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("yaml_expr,expected_expr", _CONSTANT_EXPRS)
def test_constant_expr_in_variable_upper_bound(
    yaml_expr: str, expected_expr: object
) -> None:
    input_lib = parse_yaml_library(io.StringIO(_no_port_lib_yaml(var_bound=yaml_expr)))
    lib = resolve_library([input_lib])
    variable = lib["test"].models["test.test_model"].variables["x"]
    assert expressions_equal_if_present(variable.upper_bound, expected_expr)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Acceptance tests: objectives (_CONSTANT_EXPRS direct, _LINEAR_EXPRS wrapped)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("yaml_expr,expected_expr", _CONSTANT_EXPRS)
def test_constant_expr_in_objective(yaml_expr: str, expected_expr: object) -> None:
    input_lib = parse_yaml_library(
        io.StringIO(_no_port_lib_yaml(objective_expr=yaml_expr))
    )
    lib = resolve_library([input_lib])
    obj = lib["test"].models["test.test_model"].objective_contributions
    assert obj is not None
    assert expressions_equal(obj["cost"], expected_expr)  # type: ignore[arg-type]


@pytest.mark.parametrize("yaml_expr,expected_expr", _LINEAR_EXPRS)
def test_linear_expr_in_objective(yaml_expr: str, expected_expr: object) -> None:
    input_lib = parse_yaml_library(
        io.StringIO(_no_port_lib_yaml(objective_expr=f"expec(sum({yaml_expr}))"))
    )
    lib = resolve_library([input_lib])
    obj = lib["test"].models["test.test_model"].objective_contributions
    assert obj is not None
    assert expressions_equal(obj["cost"], expected_expr.time_sum().expec())  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Acceptance tests: binding-constraint expressions (_CONSTANT_EXPRS, _LINEAR_EXPRS)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("yaml_expr,expected_expr", _CONSTANT_EXPRS)
def test_constant_expr_in_binding_constraint(
    yaml_expr: str, expected_expr: object
) -> None:
    input_lib = parse_yaml_library(
        io.StringIO(_no_port_lib_yaml(constraint_expr=f"{yaml_expr} = 0"))
    )
    lib = resolve_library([input_lib])
    bc = lib["test"].models["test.test_model"].binding_constraints["bc1"]
    assert bc == Constraint(name="bc1", expression=expected_expr == literal(0))  # type: ignore[arg-type]


@pytest.mark.parametrize("yaml_expr,expected_expr", _LINEAR_EXPRS)
def test_linear_expr_in_binding_constraint(
    yaml_expr: str, expected_expr: object
) -> None:
    input_lib = parse_yaml_library(
        io.StringIO(_no_port_lib_yaml(constraint_expr=f"{yaml_expr} >= 0"))
    )
    lib = resolve_library([input_lib])
    bc = lib["test"].models["test.test_model"].binding_constraints["bc1"]
    assert bc == Constraint(name="bc1", expression=expected_expr >= literal(0))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Rejection tests: binding-constraint expressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("yaml_expr,expected_expr", _UNRESTRICTED_EXPRS)
def test_unrestricted_expr_rejected_in_binding_constraint(
    yaml_expr: str, expected_expr: object
) -> None:
    input_lib = parse_yaml_library(
        io.StringIO(_no_port_lib_yaml(constraint_expr=f"{yaml_expr} = 0"))
    )
    with pytest.raises(ValueError, match="Non-linear expression is not allowed"):
        resolve_library([input_lib])


# ---------------------------------------------------------------------------
# Rejection tests: variable bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("yaml_expr,expected_expr", _UNRESTRICTED_EXPRS + _LINEAR_EXPRS)
def test_expr_rejected_in_variable_upper_bound(
    yaml_expr: str, expected_expr: object
) -> None:
    input_lib = parse_yaml_library(io.StringIO(_no_port_lib_yaml(var_bound=yaml_expr)))
    with pytest.raises(ValueError, match="bounds of variables must be constant"):
        resolve_library([input_lib])


# ---------------------------------------------------------------------------
# Rejection tests: objectives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("yaml_expr,expected_expr", _UNRESTRICTED_EXPRS)
def test_unrestricted_expr_rejected_in_objective(
    yaml_expr: str, expected_expr: object
) -> None:
    input_lib = parse_yaml_library(
        io.StringIO(_no_port_lib_yaml(objective_expr=yaml_expr))
    )
    with pytest.raises(ValueError):
        resolve_library([input_lib])


# ---------------------------------------------------------------------------
# Helper: model with a port-field-definition for balance_port.flow
# ---------------------------------------------------------------------------


def _port_model_yaml(*, constraint_expr: str = "", extra_output_expr: str = "") -> str:
    """Minimal library with a model that defines balance_port.flow = generation.

    The port type exposes two fields (flow, price) so tests can reference
    balance_port.price as an undefined-in-this-model field.
    """
    bc_section = (
        f"\n      binding-constraints:\n        - id: bc\n          expression: {constraint_expr}"
        if constraint_expr
        else ""
    )
    eo_section = (
        f"\n      extra-outputs:\n        - id: eo\n          expression: {extra_output_expr}"
        if extra_output_expr
        else ""
    )
    return f"""
library:
  id: test
  port-types:
    - id: flow
      fields:
        - id: flow
        - id: price
  models:
    - id: gen_model
      variables:
        - id: generation
          variable-type: continuous
      parameters:
        - id: cost
      ports:
        - id: balance_port
          type: flow
      port-field-definitions:
        - port: balance_port
          field: flow
          definition: generation{bc_section}{eo_section}
"""


# ---------------------------------------------------------------------------
# Rule 1: sum_connections cannot refer to a port field defined in this model
# ---------------------------------------------------------------------------


def test_sum_connections_on_own_port_in_binding_constraint_raises() -> None:
    """sum_connections(balance_port.flow) in a BC is invalid: flow is defined here."""
    input_lib = parse_yaml_library(
        io.StringIO(
            _port_model_yaml(constraint_expr="sum_connections(balance_port.flow) >= 0")
        )
    )
    with pytest.raises(ValueError, match="sum_connections"):
        resolve_library([input_lib])


def test_sum_connections_on_own_port_in_extra_output_raises() -> None:
    """sum_connections(balance_port.flow) in an extra-output is invalid: flow is defined here."""
    input_lib = parse_yaml_library(
        io.StringIO(
            _port_model_yaml(
                extra_output_expr="sum_connections(balance_port.flow) * generation"
            )
        )
    )
    with pytest.raises(ValueError, match="sum_connections"):
        resolve_library([input_lib])


# ---------------------------------------------------------------------------
# Rule 2: bare port.field cannot appear outside sum_connections
# ---------------------------------------------------------------------------


def test_bare_defined_port_field_in_binding_constraint_raises() -> None:
    """balance_port.flow bare (no sum_connections) in a BC is invalid."""
    input_lib = parse_yaml_library(
        io.StringIO(_port_model_yaml(constraint_expr="balance_port.flow >= 0"))
    )
    with pytest.raises(ValueError, match="Bare port field"):
        resolve_library([input_lib])


def test_bare_undefined_port_field_in_binding_constraint_raises() -> None:
    """balance_port.price bare (not defined in this model either) in a BC is invalid."""
    input_lib = parse_yaml_library(
        io.StringIO(_port_model_yaml(constraint_expr="balance_port.price >= 0"))
    )
    with pytest.raises(ValueError, match="Bare port field"):
        resolve_library([input_lib])


# ---------------------------------------------------------------------------
# Acceptance: sum_connections on a port field NOT defined in this model is valid
# ---------------------------------------------------------------------------


def test_sum_connections_on_non_own_port_accepted() -> None:
    """sum_connections(balance_port.price) is valid: price is not defined in this model."""
    input_lib = parse_yaml_library(
        io.StringIO(
            _port_model_yaml(constraint_expr="sum_connections(balance_port.price) >= 0")
        )
    )
    resolve_library([input_lib])  # must not raise
