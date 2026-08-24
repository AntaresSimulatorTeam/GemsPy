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
from typing import Dict, List, Optional, Set

from gems_craft.expression import ExpressionNode, literal
from gems_craft.expression.degree import is_linear
from gems_craft.expression.expression import (
    AbsNode,
    AdditionNode,
    AllTimeSumNode,
    CeilNode,
    ComparisonNode,
    DivisionNode,
    DualNode,
    FloorNode,
    LiteralNode,
    LowerBoundNode,
    MaxNode,
    MinNode,
    MultiplicationNode,
    NegationNode,
    ParameterNode,
    PortFieldAggregatorNode,
    PortFieldNode,
    ReducedCostNode,
    RoundNode,
    ScenarioOperatorNode,
    TimeEvalNode,
    TimeShiftNode,
    TimeSumNode,
    UpperBoundNode,
    VariableNode,
)
from gems_craft.expression.indexing_structure import IndexingStructure
from gems_craft.expression.parsing.parse_expression import (
    ModelIdentifiers,
    parse_expression,
)
from gems_craft.expression.uses_sum_connections_on import uses_sum_connections_on
from gems_craft.expression.visitor import ExpressionVisitor, visit
from gems_craft.model import (
    Constraint,
    Model,
    ModelPort,
    Parameter,
    PortField,
    PortType,
    ValueType,
    Variable,
    model,
)
from gems_craft.model.library import Library
from gems_craft.model.model import model
from gems_craft.model.parsing import (
    ConstraintSchema,
    FieldSchema,
    LibrarySchema,
    ModelPortSchema,
    ModelSchema,
    ParameterSchema,
    PortFieldDefinitionSchema,
    PortTypeSchema,
    VariableSchema,
)
from gems_craft.model.port import PortFieldDefinition, port_field_def


def resolve_library(
    input_libs: List[LibrarySchema], preloaded_libs: Optional[List[Library]] = None
) -> Dict[str, Library]:
    """
    Converts parsed data into an actually usable library of models.

     - resolves references between models and ports
     - parses expressions and resolves references to variables/params
    """
    yaml_lib_dict = dict((l.id, l) for l in input_libs)

    preloaded_port_types = {}
    if preloaded_libs:
        for preloaded_lib in preloaded_libs:
            preloaded_port_types.update(preloaded_lib.port_types)

    output_lib_dict: Dict[str, Library] = (
        dict((l.id, l) for l in preloaded_libs) if preloaded_libs else {}
    )

    remaining_lib_ids: List[str] = list(yaml_lib_dict)
    treated_lib_ids: Set[str] = set()
    import_stack: List[str] = []

    while remaining_lib_ids:
        next_lib_id = remaining_lib_ids.pop()

        if next_lib_id in treated_lib_ids:
            continue
        else:
            import_stack.append(next_lib_id)

        while import_stack:
            cur_yaml_lib = yaml_lib_dict[import_stack[-1]]
            current_lib = Library(
                id=cur_yaml_lib.id,
                port_types={},
                models={},
                taxonomy=cur_yaml_lib.taxonomy,
            )

            # Add already parsed port types from dependencies in current lib
            _add_preloaded_port_types_to_current_lib(preloaded_port_types, current_lib)
            _add_resolved_dependent_port_types_to_current_lib(
                output_lib_dict, treated_lib_ids, cur_yaml_lib, current_lib
            )

            remaining_dependencies = set(cur_yaml_lib.dependencies) - treated_lib_ids

            if remaining_dependencies:
                _add_dependencies_to_stack(import_stack, remaining_dependencies)

            else:
                _resolve_lib(current_lib, cur_yaml_lib, output_lib_dict)
                _update_treated_libs_and_import_stack(treated_lib_ids, import_stack)

    return output_lib_dict


def _add_preloaded_port_types_to_current_lib(
    preloaded_port_types: dict[str, PortType], current_lib: Library
) -> None:
    current_lib.port_types.update(preloaded_port_types)


def _add_resolved_dependent_port_types_to_current_lib(
    output_lib_dict: Dict[str, Library],
    treated_lib_ids: Set[str],
    cur_yaml_lib: LibrarySchema,
    current_lib: Library,
) -> None:
    done_dependencies = set(cur_yaml_lib.dependencies) & treated_lib_ids
    for done_lib in done_dependencies:
        current_lib.port_types.update(output_lib_dict[done_lib].port_types)


def _update_treated_libs_and_import_stack(
    treated_lib_ids: Set[str], import_stack: List[str]
) -> None:
    treated_lib_ids.add(import_stack.pop())


def _resolve_lib(
    current_lib: Library, cur_yaml_lib: LibrarySchema, output_lib: Dict[str, Library]
) -> None:
    port_types = [_convert_port_type(p) for p in cur_yaml_lib.port_types]
    port_types_dict = dict((p.id, p) for p in port_types)

    if current_lib.port_types.keys() & port_types_dict.keys():
        raise Exception(
            f"Port(s) : {str(current_lib.port_types.keys() & port_types_dict.keys())} is(are) defined twice."
        )
    current_lib.port_types.update(port_types_dict)

    cur_yaml_lib_model_ids = [model.id for model in cur_yaml_lib.models]
    for id in cur_yaml_lib_model_ids:
        if cur_yaml_lib_model_ids.count(id) > 1:
            raise Exception(f"Model {id} is defined twice")

    models = [
        _resolve_model(m, current_lib.port_types, current_lib.id)
        for m in cur_yaml_lib.models
    ]

    models_dict = dict((m.id, m) for m in models)

    current_lib.models.update(models_dict)
    output_lib[current_lib.id] = current_lib


def _add_dependencies_to_stack(
    import_stack: List[str], remaining_dependencies: Set[str]
) -> None:
    first_dependency = remaining_dependencies.pop()

    if first_dependency in import_stack:
        raise Exception("Circular import in yaml libraries")
    import_stack.append(first_dependency)


def _convert_field(field: FieldSchema) -> PortField:
    return PortField(name=field.id)


def _convert_port_type(port_type: PortTypeSchema) -> PortType:
    return PortType(
        id=port_type.id, fields=[_convert_field(f) for f in port_type.fields]
    )


# TODO: these _forbid_* checks are expression validation embedded in the
# resolution pass — consider isolating them from the build logic below.
def _forbid_nonlinear(expr: ExpressionNode, context: str) -> None:
    if not is_linear(expr):
        raise ValueError(f"Non-linear expression is not allowed in {context}.")


class _ForbidBarePortFieldVisitor(ExpressionVisitor[None]):
    """Raises if a bare PortFieldNode appears outside of sum_connections."""

    def __init__(self, context: str) -> None:
        self._context = context

    def literal(self, node: LiteralNode) -> None:
        pass

    def negation(self, node: NegationNode) -> None:
        visit(node.operand, self)

    def addition(self, node: AdditionNode) -> None:
        for o in node.operands:
            visit(o, self)

    def multiplication(self, node: MultiplicationNode) -> None:
        visit(node.left, self)
        visit(node.right, self)

    def division(self, node: DivisionNode) -> None:
        visit(node.left, self)
        visit(node.right, self)

    def comparison(self, node: ComparisonNode) -> None:
        visit(node.left, self)
        visit(node.right, self)

    def variable(self, node: VariableNode) -> None:
        pass

    def parameter(self, node: ParameterNode) -> None:
        pass

    def time_shift(self, node: TimeShiftNode) -> None:
        visit(node.operand, self)

    def time_eval(self, node: TimeEvalNode) -> None:
        visit(node.operand, self)

    def time_sum(self, node: TimeSumNode) -> None:
        visit(node.operand, self)

    def all_time_sum(self, node: AllTimeSumNode) -> None:
        visit(node.operand, self)

    def scenario_operator(self, node: ScenarioOperatorNode) -> None:
        visit(node.operand, self)

    def port_field(self, node: PortFieldNode) -> None:
        raise ValueError(
            f"Bare port field '{node.port_name}.{node.field_name}' is not allowed "
            f"outside sum_connections in {self._context}."
        )

    def port_field_aggregator(self, node: PortFieldAggregatorNode) -> None:
        pass  # sum_connections wrapping a port field is valid; do not recurse

    def floor(self, node: FloorNode) -> None:
        visit(node.operand, self)

    def ceil(self, node: CeilNode) -> None:
        visit(node.operand, self)

    def abs(self, node: AbsNode) -> None:
        visit(node.operand, self)

    def round(self, node: RoundNode) -> None:
        visit(node.operand, self)

    def maximum(self, node: MaxNode) -> None:
        for o in node.operands:
            visit(o, self)

    def minimum(self, node: MinNode) -> None:
        for o in node.operands:
            visit(o, self)

    def dual(self, node: DualNode) -> None:
        pass

    def reduced_cost(self, node: ReducedCostNode) -> None:
        pass

    def lower_bound(self, node: LowerBoundNode) -> None:
        pass

    def upper_bound(self, node: UpperBoundNode) -> None:
        pass


def _forbid_bare_port_field(expr: ExpressionNode, context: str) -> None:
    visit(expr, _ForbidBarePortFieldVisitor(context))


def _forbid_sum_connections_on_own_port(
    expr: ExpressionNode,
    own_port_fields: Set[tuple],
    context: str,
) -> None:
    for port_name, field_name in own_port_fields:
        if uses_sum_connections_on(expr, port_name, field_name):
            raise ValueError(
                f"sum_connections({port_name}.{field_name}) is not allowed in {context}: "
                f"this port field is defined in the current model."
            )


def _resolve_model(
    input_model: ModelSchema, port_types: Dict[str, PortType], library_id: str
) -> Model:
    identifiers = ModelIdentifiers(
        variables={v.id for v in input_model.variables},
        parameters={p.id for p in input_model.parameters},
        constraints={c.id for c in input_model.binding_constraints}
        | {c.id for c in input_model.constraints},
    )

    own_port_fields: Set[tuple] = {
        (d.port, d.field) for d in input_model.port_field_definitions
    }

    binding_constraints = [
        _to_constraint(c, identifiers) for c in input_model.binding_constraints
    ]
    constraints = [_to_constraint(c, identifiers) for c in input_model.constraints]

    for c in binding_constraints + constraints:
        _forbid_nonlinear(c.expression, f"constraint '{c.name}'")
        _forbid_bare_port_field(c.expression, f"constraint '{c.name}'")
        _forbid_sum_connections_on_own_port(
            c.expression, own_port_fields, f"constraint '{c.name}'"
        )

    objective_contributions = None
    if input_model.objective_contributions:
        objective_contributions = {
            contrib.id: parse_expression(contrib.expression, identifiers)
            for contrib in input_model.objective_contributions
        }
        for oid, expr in objective_contributions.items():
            _forbid_nonlinear(expr, f"objective contribution '{oid}'")
            _forbid_bare_port_field(expr, f"objective contribution '{oid}'")
            _forbid_sum_connections_on_own_port(
                expr, own_port_fields, f"objective contribution '{oid}'"
            )

    extra_outputs = (
        {
            eo.id: parse_expression(eo.expression, identifiers)
            for eo in input_model.extra_outputs
        }
        if input_model.extra_outputs
        else None
    )

    if extra_outputs:
        for eo_id, eo_expr in extra_outputs.items():
            _forbid_bare_port_field(eo_expr, f"extra-output '{eo_id}'")
            _forbid_sum_connections_on_own_port(
                eo_expr, own_port_fields, f"extra-output '{eo_id}'"
            )

    return model(
        id=f"{library_id}.{input_model.id}",
        parameters=[_to_parameter(p) for p in input_model.parameters],
        variables=[_to_variable(v, identifiers) for v in input_model.variables],
        ports=[_resolve_model_port(p, port_types) for p in input_model.ports],
        port_fields_definitions=[
            _resolve_field_definition(d, identifiers)
            for d in input_model.port_field_definitions
        ],
        binding_constraints=binding_constraints,
        constraints=constraints,
        objective_contributions=objective_contributions,
        extra_outputs=extra_outputs,
        properties=[p.id for p in input_model.properties],
    )


def _resolve_model_port(
    port: ModelPortSchema, port_types: Dict[str, PortType]
) -> ModelPort:
    return ModelPort(port_name=port.id, port_type=port_types[port.type])


def _resolve_field_definition(
    definition: PortFieldDefinitionSchema, ids: ModelIdentifiers
) -> PortFieldDefinition:
    return port_field_def(
        port_name=definition.port,
        field_name=definition.field,
        definition=parse_expression(definition.definition, ids),
    )


def _to_parameter(param: ParameterSchema) -> Parameter:
    return Parameter(
        name=param.id,
        type=ValueType.CONTINUOUS,
        structure=IndexingStructure(param.time_dependent, param.scenario_dependent),
    )


def _to_expression_if_present(
    expr: Optional[str], identifiers: ModelIdentifiers
) -> Optional[ExpressionNode]:
    if not expr:
        return None
    return parse_expression(expr, identifiers)


def _to_variable(var: VariableSchema, identifiers: ModelIdentifiers) -> Variable:
    return Variable(
        name=var.id,
        data_type={
            "continuous": ValueType.CONTINUOUS,
            "integer": ValueType.INTEGER,
            "binary": ValueType.BINARY,
        }[var.variable_type],
        structure=IndexingStructure(var.time_dependent, var.scenario_dependent),
        lower_bound=_to_expression_if_present(var.lower_bound, identifiers),
        upper_bound=_to_expression_if_present(var.upper_bound, identifiers),
    )


def _to_constraint(
    constraint: ConstraintSchema, identifiers: ModelIdentifiers
) -> Constraint:
    lb = _to_expression_if_present(constraint.lower_bound, identifiers)
    ub = _to_expression_if_present(constraint.upper_bound, identifiers)
    return Constraint(
        name=constraint.id,
        expression=parse_expression(constraint.expression, identifiers),
        lower_bound=(lb if lb is not None else literal(-float("inf"))),
        upper_bound=(ub if ub is not None else literal(float("inf"))),
    )
