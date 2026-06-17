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

from typing import List

from andromede.expression import (
    ExpressionNode,
    PrinterVisitor,
    literal,
    param,
    var,
    visit,
)
from andromede.expression.indexing_structure import IndexingStructure
from andromede.model import Model, Variable, float_parameter, float_variable, model
from andromede.model.constraint import Constraint
from andromede.model.parameter import Parameter, float_parameter, int_parameter
from andromede.model.variable import ValueType, float_variable, int_variable

CONSTANT = IndexingStructure(False, False)
TIME_AND_SCENARIO_FREE = IndexingStructure(True, True)
ANTICIPATIVE_TIME_VARYING = IndexingStructure(True, True)
NON_ANTICIPATIVE_TIME_VARYING = IndexingStructure(True, False)
CONSTANT_PER_SCENARIO = IndexingStructure(False, True)


class ModelEditor:
    def __init__(self, initial_model: Model) -> None:
        self.initial_model = initial_model

    def linearize_variables(self) -> List[Variable]:
        return [
            float_variable(
                v.name,
                lower_bound=v.lower_bound,
                upper_bound=v.upper_bound,
                structure=v.structure,
            )
            for v in self.initial_model.variables.values()
        ]

    def variable_in_constraint(self, c: Constraint, variables: List[str]) -> bool:
        res = False
        if self.variable_in_expression(c.lower_bound, variables):
            res = True
        elif self.variable_in_expression(c.expression, variables):
            res = True
        elif self.variable_in_expression(c.upper_bound, variables):
            res = True
        return res

    def variable_in_expression(
        self, expr: ExpressionNode, variables: List[str]
    ) -> bool:
        res = False
        str_expr = visit(expr, PrinterVisitor())
        for v in variables:
            if v in str_expr:
                res = True
        return res

    def get_name_integer_variables(self) -> List[str]:
        return [
            v.name
            for v in self.initial_model.variables.values()
            if v.data_type == ValueType.INTEGER
        ]

    def fix_to_zero_integer_variables_and_keep_others(self) -> List[Variable]:
        return [
            float_variable(
                v.name,
                lower_bound=(
                    (v.lower_bound) if v.data_type == ValueType.FLOAT else literal(0)
                ),
                upper_bound=(
                    (v.upper_bound) if v.data_type == ValueType.FLOAT else literal(0)
                ),
                structure=v.structure,
            )
            for v in self.initial_model.variables.values()
        ]

    def keep_constraints_with_no_variables_from_list(
        self, variables: List[str]
    ) -> List[Constraint]:
        return [
            c
            for c in self.initial_model.constraints.values()
            if not (self.variable_in_constraint(c, variables))
        ]

    def keep_variables_with_no_variables_from_list_and_linearize(
        self, variables: List[str]
    ) -> List[Variable]:
        return [
            float_variable(
                v.name,
                lower_bound=v.lower_bound,
                upper_bound=v.upper_bound,
                structure=v.structure,
            )
            for v in self.initial_model.variables.values()
            if v.name not in variables
        ]


class AccurateModelBuilder(ModelEditor):
    def __init__(self, initial_model: Model) -> None:
        super().__init__(initial_model)
        THERMAL_CLUSTER_MODEL_LP = model(
            id=self.initial_model.id,
            parameters=self.initial_model.parameters.values(),
            variables=self.linearize_variables(),
            ports=self.initial_model.ports.values(),
            port_fields_definitions=self.initial_model.port_fields_definitions.values(),
            constraints=self.initial_model.constraints.values(),
            objective_operational_contribution=self.initial_model.objective_operational_contribution,
        )
        self.model = THERMAL_CLUSTER_MODEL_LP


class FastModelBuilder(ModelEditor):
    def __init__(self, initial_model: Model) -> None:
        super().__init__(initial_model)
        integer_variables = self.get_name_integer_variables()

        THERMAL_CLUSTER_MODEL_FAST = model(
            id=self.initial_model.id,
            parameters=self.initial_model.parameters.values(),
            variables=self.fix_to_zero_integer_variables_and_keep_others(),
            ports=self.initial_model.ports.values(),
            port_fields_definitions=self.initial_model.port_fields_definitions.values(),
            constraints=self.keep_constraints_with_no_variables_from_list(
                integer_variables
            ),
            objective_operational_contribution=self.initial_model.objective_operational_contribution,
        )

        self.model = THERMAL_CLUSTER_MODEL_FAST


class HeuristicAccurateModelBuilder(ModelEditor):
    def __init__(self, initial_model: Model) -> None:
        super().__init__(initial_model)
        generation_variable = ["generation"]

        THERMAL_CLUSTER_MODEL_ACCURATE_HEURISTIC = model(
            id=self.initial_model.id,
            parameters=self.initial_model.parameters.values(),
            variables=self.keep_variables_with_no_variables_from_list_and_linearize(
                generation_variable
            ),
            constraints=self.keep_constraints_with_no_variables_from_list(
                generation_variable
            ),
            objective_operational_contribution=(var("nb_on")).sum().expec(),
        )
        self.model = THERMAL_CLUSTER_MODEL_ACCURATE_HEURISTIC


class HeuristicFastModelBuilder:
    def __init__(self, number_hours: int, slot_length: int):
        BLOCK_MODEL_FAST_HEURISTIC = model(
            id="BLOCK_FAST",
            parameters=self.get_parameters(number_hours // slot_length, slot_length),
            variables=self.get_variables(number_hours // slot_length, slot_length),
            constraints=self.get_constraints(number_hours // slot_length, slot_length),
            objective_operational_contribution=self.get_objective_operational_contribution(
                slot_length
            ),
        )
        self.model = BLOCK_MODEL_FAST_HEURISTIC

    def get_objective_operational_contribution(
        self, slot_length: int
    ) -> ExpressionNode:
        return (var("n")).sum().expec() + sum(
            [
                var(f"t_ajust_{h}") * (h + 1) / 10 / slot_length
                for h in range(slot_length)
            ]
        ).expec()  # type:ignore

    def get_constraints(self, slot: int, slot_length: int) -> List[Constraint]:
        return (
            [
                Constraint(
                    f"Definition of n block {k} for {h}",
                    var(f"n_block_{k}")
                    >= param("n_guide") * param(f"alpha_{k}_{h}")
                    - param("n_max") * (literal(1) - var(f"t_ajust_{h}")),
                )
                for k in range(slot)
                for h in range(slot_length)
            ]
            + [
                Constraint(
                    f"Definition of n ajust for {h}",
                    var(f"n_ajust")
                    >= param("n_guide") * param(f"alpha_ajust_{h}")
                    - param("n_max") * (literal(1) - var(f"t_ajust_{h}")),
                )
                for h in range(slot_length)
            ]
            + [
                Constraint(
                    f"Definition of n with relation to block {k} for {h}",
                    var(f"n")
                    >= param(f"alpha_{k}_{h}") * var(f"n_block_{k}")
                    - param("n_max") * (literal(1) - var(f"t_ajust_{h}")),
                )
                for k in range(slot)
                for h in range(slot_length)
            ]
            + [
                Constraint(
                    f"Definition of n with relation to ajust for {h}",
                    var(f"n")
                    >= param(f"alpha_ajust_{h}") * var(f"n_ajust")
                    - param("n_max") * (literal(1) - var(f"t_ajust_{h}")),
                )
                for h in range(slot_length)
            ]
            + [
                Constraint(
                    "Choose one t ajust",
                    literal(0) + sum([var(f"t_ajust_{h}") for h in range(slot_length)])
                    == literal(1),
                )
            ]
        )

    def get_variables(self, slot: int, slot_length: int) -> List[Variable]:
        return (
            [
                float_variable(
                    f"n_block_{k}",
                    lower_bound=literal(0),
                    upper_bound=param("n_max"),
                    structure=CONSTANT_PER_SCENARIO,
                )
                for k in range(slot)
            ]
            + [
                float_variable(
                    "n_ajust",
                    lower_bound=literal(0),
                    upper_bound=param("n_max"),
                    structure=CONSTANT_PER_SCENARIO,
                )
            ]
            + [
                int_variable(
                    f"t_ajust_{h}",
                    lower_bound=literal(0),
                    upper_bound=literal(1),
                    structure=CONSTANT_PER_SCENARIO,
                )
                for h in range(slot_length)
            ]
            + [
                float_variable(
                    "n",
                    lower_bound=literal(0),
                    upper_bound=param("n_max"),
                    structure=TIME_AND_SCENARIO_FREE,
                )
            ]
        )

    def get_parameters(self, slot: int, slot_length: int) -> List[Parameter]:
        return (
            [
                float_parameter("n_guide", TIME_AND_SCENARIO_FREE),
                float_parameter("slot_length", CONSTANT),
                float_parameter("n_max", CONSTANT),
            ]
            + [
                int_parameter(f"alpha_{k}_{h}", NON_ANTICIPATIVE_TIME_VARYING)
                for k in range(slot)
                for h in range(slot_length)
            ]
            + [
                int_parameter(f"alpha_ajust_{h}", NON_ANTICIPATIVE_TIME_VARYING)
                for h in range(slot_length)
            ]
        )
