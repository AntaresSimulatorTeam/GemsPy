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

from gems.expression import ExpressionNode, literal
from gems.expression.indexing_structure import IndexingStructure
from gems.expression.parsing.parse_expression import ModelIdentifiers, parse_expression
from gems.model import (
    Constraint,
    Model,
    ModelPort,
    Parameter,
    PortField,
    PortType,
    ProblemContext,
    ValueType,
    Variable,
    model,
)
from gems.model.library import Library
from gems.model.parsing import (
    InputConstraint,
    InputField,
    InputLibrary,
    InputModel,
    InputModelPort,
    InputParameter,
    InputPortFieldDefinition,
    InputPortType,
    InputVariable,
)
from gems.model.port import PortFieldDefinition, port_field_def


def resolve_library(
    input_libs: List[InputLibrary], preloaded_libs: Optional[List[Library]] = None
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
            current_lib = Library(id=cur_yaml_lib.id, port_types={}, models={})

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
    cur_yaml_lib: InputLibrary,
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
    current_lib: Library, cur_yaml_lib: InputLibrary, output_lib: Dict[str, Library]
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

    models = [_resolve_model(m, current_lib.port_types) for m in cur_yaml_lib.models]

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


def _convert_field(field: InputField) -> PortField:
    return PortField(name=field.id)


def _convert_port_type(port_type: InputPortType) -> PortType:
    return PortType(
        id=port_type.id, fields=[_convert_field(f) for f in port_type.fields]
    )


def _resolve_model(input_model: InputModel, port_types: Dict[str, PortType]) -> Model:
    identifiers = ModelIdentifiers(
        variables={v.id for v in input_model.variables},
        parameters={p.id for p in input_model.parameters},
    )
    return model(
        id=input_model.id,
        parameters=[_to_parameter(p) for p in input_model.parameters],
        variables=[_to_variable(v, identifiers) for v in input_model.variables],
        ports=[_resolve_model_port(p, port_types) for p in input_model.ports],
        port_fields_definitions=[
            _resolve_field_definition(d, identifiers)
            for d in input_model.port_field_definitions
        ],
        binding_constraints=[
            _to_constraint(c, identifiers) for c in input_model.binding_constraints
        ],
        constraints=[_to_constraint(c, identifiers) for c in input_model.constraints],
        objective_operational_contribution=_to_expression_if_present(
            input_model.objective, identifiers
        ),
    )


def _resolve_model_port(
    port: InputModelPort, port_types: Dict[str, PortType]
) -> ModelPort:
    return ModelPort(port_name=port.id, port_type=port_types[port.type])


def _resolve_field_definition(
    definition: InputPortFieldDefinition, ids: ModelIdentifiers
) -> PortFieldDefinition:
    return port_field_def(
        port_name=definition.port,
        field_name=definition.field,
        definition=parse_expression(definition.definition, ids),
    )


def _to_parameter(param: InputParameter) -> Parameter:
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


def _to_variable(var: InputVariable, identifiers: ModelIdentifiers) -> Variable:
    return Variable(
        name=var.id,
        data_type={"continuous": ValueType.CONTINUOUS, "integer": ValueType.INTEGER}[
            var.variable_type
        ],
        structure=IndexingStructure(var.time_dependent, var.scenario_dependent),
        lower_bound=_to_expression_if_present(var.lower_bound, identifiers),
        upper_bound=_to_expression_if_present(var.upper_bound, identifiers),
        context=ProblemContext.OPERATIONAL,
    )


def _to_constraint(
    constraint: InputConstraint, identifiers: ModelIdentifiers
) -> Constraint:
    lb = _to_expression_if_present(constraint.lower_bound, identifiers)
    ub = _to_expression_if_present(constraint.upper_bound, identifiers)
    return Constraint(
        name=constraint.id,
        expression=parse_expression(constraint.expression, identifiers),
        lower_bound=(lb if lb is not None else literal(-float("inf"))),
        upper_bound=(ub if ub is not None else literal(float("inf"))),
    )
