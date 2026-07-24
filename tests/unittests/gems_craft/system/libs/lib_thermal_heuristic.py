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

from andromede.expression import literal, param, var
from andromede.expression.expression import ExpressionRange, port_field
from andromede.expression.indexing_structure import IndexingStructure
from andromede.libs.standard import BALANCE_PORT_TYPE
from andromede.model import ModelPort, float_parameter, float_variable, model
from andromede.model.constraint import Constraint
from andromede.model.model import PortFieldDefinition, PortFieldId
from andromede.model.parameter import float_parameter, int_parameter
from andromede.model.port import PortField, PortType
from andromede.model.variable import float_variable, int_variable

CONSTANT = IndexingStructure(False, False)
TIME_AND_SCENARIO_FREE = IndexingStructure(True, True)
ANTICIPATIVE_TIME_VARYING = IndexingStructure(True, True)
NON_ANTICIPATIVE_TIME_VARYING = IndexingStructure(True, False)
CONSTANT_PER_SCENARIO = IndexingStructure(False, True)

THERMAL_CLUSTER_MODEL_MILP = model(
    id="GEN",
    parameters=[
        float_parameter("p_max", CONSTANT),  # p_max of a single unit
        float_parameter("p_min", CONSTANT),
        float_parameter("d_min_up", CONSTANT),
        float_parameter("d_min_down", CONSTANT),
        float_parameter("cost", CONSTANT),
        float_parameter("startup_cost", CONSTANT),
        float_parameter("fixed_cost", CONSTANT),
        int_parameter("nb_units_min", TIME_AND_SCENARIO_FREE),
        int_parameter("nb_units_max", TIME_AND_SCENARIO_FREE),
        float_parameter("min_generating", TIME_AND_SCENARIO_FREE),
        float_parameter("max_generating", TIME_AND_SCENARIO_FREE),
        int_parameter("max_failure", TIME_AND_SCENARIO_FREE),
        int_parameter("nb_units_max_min_down_time", TIME_AND_SCENARIO_FREE),
    ],
    variables=[
        float_variable(
            "generation",
            lower_bound=param("min_generating"),
            upper_bound=param("max_generating"),
            structure=ANTICIPATIVE_TIME_VARYING,
        ),
        int_variable(
            "nb_on",
            lower_bound=param("nb_units_min"),
            upper_bound=param("nb_units_max"),
            structure=ANTICIPATIVE_TIME_VARYING,
        ),
        int_variable(
            "nb_stop",
            lower_bound=literal(0),
            structure=ANTICIPATIVE_TIME_VARYING,
        ),
        int_variable(
            "nb_failure",
            lower_bound=literal(0),
            upper_bound=param("max_failure"),
            structure=ANTICIPATIVE_TIME_VARYING,
        ),
        int_variable(
            "nb_start",
            lower_bound=literal(0),
            structure=ANTICIPATIVE_TIME_VARYING,
        ),
    ],
    ports=[ModelPort(port_type=BALANCE_PORT_TYPE, port_name="balance_port")],
    port_fields_definitions=[
        PortFieldDefinition(
            port_field=PortFieldId("balance_port", "flow"),
            definition=var("generation"),
        )
    ],
    constraints=[
        Constraint(
            "Max generation",
            var("generation") <= param("p_max") * var("nb_on"),
        ),
        Constraint(
            "Min generation",
            var("generation") >= param("p_min") * var("nb_on"),
        ),
        Constraint(
            "NODU balance",
            var("nb_on") == var("nb_on").shift(-1) + var("nb_start") - var("nb_stop"),
        ),
        Constraint(
            "Max failures",
            var("nb_failure") <= var("nb_stop"),
        ),
        Constraint(
            "Min up time",
            var("nb_start")
            .shift(ExpressionRange(-param("d_min_up") + 1, literal(0)))
            .sum()
            - var("nb_failure")
            .shift(ExpressionRange(-param("d_min_up") + 1, literal(0)))
            .sum()
            <= var("nb_on"),
        ),
        Constraint(
            "Min down time",
            var("nb_stop")
            .shift(ExpressionRange(-param("d_min_down") + 1, literal(0)))
            .sum()
            <= param("nb_units_max_min_down_time") - var("nb_on"),
        ),
        # It also works by writing ExpressionRange(-param("d_min_down") + 1, 0) as ExpressionRange's __post_init__ wraps integers to literal nodes. However, MyPy does not seem to infer that ExpressionRange's attributes are necessarily of ExpressionNode type and raises an error if the arguments in the constructor are integer (whereas it runs correctly), this why we specify it here with literal(0) instead of 0.
    ],
    objective_operational_contribution=(
        param("cost") * var("generation")
        + param("startup_cost") * var("nb_start")
        + param("fixed_cost") * var("nb_on")
    )
    .sum()
    .expec(),
)

UPPER_BOUND_ON_SUM_OF_GENERATION = model(
    id="BC",
    parameters=[float_parameter("upper_bound", structure=CONSTANT)],
    ports=[ModelPort(port_type=BALANCE_PORT_TYPE, port_name="balance_port")],
    binding_constraints=[
        Constraint(
            name="Binding constraint",
            expression=port_field("balance_port", "flow").sum_connections()
            <= param("upper_bound"),
        )
    ],
)

THERMAL_CLUSTER_MODEL_MILP_WITH_RAMP = model(
    id="GEN",
    parameters=[
        float_parameter("p_max", CONSTANT),  # p_max of a single unit
        float_parameter("p_min", CONSTANT),
        float_parameter("d_min_up", CONSTANT),
        float_parameter("d_min_down", CONSTANT),
        float_parameter("cost", CONSTANT),
        float_parameter("startup_cost", CONSTANT),
        float_parameter("fixed_cost", CONSTANT),
        int_parameter("nb_units_min", TIME_AND_SCENARIO_FREE),
        int_parameter("nb_units_max", TIME_AND_SCENARIO_FREE),
        float_parameter("min_generating", TIME_AND_SCENARIO_FREE),
        float_parameter("max_generating", TIME_AND_SCENARIO_FREE),
        int_parameter("max_failure", TIME_AND_SCENARIO_FREE),
        int_parameter("nb_units_max_min_down_time", TIME_AND_SCENARIO_FREE),
        float_parameter("max_upward_ramping_rate", CONSTANT),
        float_parameter("max_downward_ramping_rate", CONSTANT),
        int_parameter("nb_start_max", TIME_AND_SCENARIO_FREE),
        int_parameter("nb_stop_max", TIME_AND_SCENARIO_FREE),
    ],
    variables=[
        float_variable(
            "generation",
            lower_bound=param("min_generating"),
            upper_bound=param("max_generating"),
            structure=ANTICIPATIVE_TIME_VARYING,
        ),
        int_variable(
            "nb_on",
            lower_bound=param("nb_units_min"),
            upper_bound=param("nb_units_max"),
            structure=ANTICIPATIVE_TIME_VARYING,
        ),
        int_variable(
            "nb_stop",
            lower_bound=literal(0),
            upper_bound=param("nb_stop_max"),
            structure=ANTICIPATIVE_TIME_VARYING,
        ),
        int_variable(
            "nb_failure",
            lower_bound=literal(0),
            upper_bound=param("max_failure"),
            structure=ANTICIPATIVE_TIME_VARYING,
        ),
        int_variable(
            "nb_start",
            lower_bound=literal(0),
            upper_bound=param("nb_start_max"),
            structure=ANTICIPATIVE_TIME_VARYING,
        ),
    ],
    ports=[ModelPort(port_type=BALANCE_PORT_TYPE, port_name="balance_port")],
    port_fields_definitions=[
        PortFieldDefinition(
            port_field=PortFieldId("balance_port", "flow"),
            definition=var("generation"),
        )
    ],
    constraints=[
        Constraint(
            "Max generation",
            var("generation") <= param("p_max") * var("nb_on"),
        ),
        Constraint(
            "Min generation",
            var("generation") >= param("p_min") * var("nb_on"),
        ),
        Constraint(
            "NODU balance",
            var("nb_on") == var("nb_on").shift(-1) + var("nb_start") - var("nb_stop"),
        ),
        Constraint(
            "Max failures",
            var("nb_failure") <= var("nb_stop"),
        ),
        Constraint(
            "Min up time",
            var("nb_start")
            .shift(ExpressionRange(-param("d_min_up") + 1, literal(0)))
            .sum()
            - var("nb_failure")
            .shift(ExpressionRange(-param("d_min_up") + 1, literal(0)))
            .sum()
            <= var("nb_on"),
        ),
        Constraint(
            "Min down time",
            var("nb_stop")
            .shift(ExpressionRange(-param("d_min_down") + 1, literal(0)))
            .sum()
            <= param("nb_units_max_min_down_time") - var("nb_on"),
        ),
        # It also works by writing ExpressionRange(-param("d_min_down") + 1, 0) as ExpressionRange's __post_init__ wraps integers to literal nodes. However, MyPy does not seem to infer that ExpressionRange's attributes are necessarily of ExpressionNode type and raises an error if the arguments in the constructor are integer (whereas it runs correctly), this why we specify it here with literal(0) instead of 0.
        Constraint(
            "Max_upward",
            var("generation")
            <= var("generation").shift(literal(-1))
            + param("max_upward_ramping_rate") * var("nb_on")
            + param("p_max") * var("nb_start"),
        ),
        Constraint(
            "Max_downward",
            var("generation")
            >= var("generation").shift(literal(-1))
            - param("max_downward_ramping_rate") * var("nb_on")
            - param("p_max") * var("nb_stop"),
        ),
    ],
    objective_operational_contribution=(
        param("cost") * var("generation")
        + param("startup_cost") * var("nb_start")
        + param("fixed_cost") * var("nb_on")
    )
    .sum()
    .expec(),
)

RESERVE_PORT_TYPE = PortType(
    id="reserve_balance",
    fields=[PortField("energy"), PortField("primary_reserve")],
)

NODE_WITH_RESERVE_MODEL = model(
    id="NODE_MODEL",
    parameters=[
        float_parameter("spillage_cost"),
        float_parameter("ens_cost"),
        float_parameter("reserve_not_supplied_cost"),
    ],
    variables=[
        float_variable("spillage_energy", lower_bound=literal(0)),
        float_variable("unsupplied_energy", lower_bound=literal(0)),
        float_variable("unsupplied_reserve", lower_bound=literal(0)),
    ],
    ports=[
        ModelPort(port_type=RESERVE_PORT_TYPE, port_name="balance_port"),
    ],
    binding_constraints=[
        Constraint(
            name="Balance energy",
            expression=port_field("balance_port", "energy").sum_connections()
            == var("spillage_energy") - var("unsupplied_energy"),
        ),
        Constraint(
            name="Balance reserve",
            expression=port_field("balance_port", "primary_reserve").sum_connections()
            == -var("unsupplied_reserve"),
        ),
    ],
    objective_operational_contribution=(
        param("spillage_cost") * var("spillage_energy")
        + param("ens_cost") * var("unsupplied_energy")
        + param("reserve_not_supplied_cost") * var("unsupplied_reserve")
    )
    .sum()
    .expec(),
)

DEMAND_WITH_RESERVE_MODEL = model(
    id="DEMAND_MODEL",
    parameters=[
        float_parameter("demand_energy", TIME_AND_SCENARIO_FREE),
        float_parameter("demand_primary_reserve", TIME_AND_SCENARIO_FREE),
    ],
    ports=[ModelPort(port_type=RESERVE_PORT_TYPE, port_name="balance_port")],
    port_fields_definitions=[
        PortFieldDefinition(
            port_field=PortFieldId("balance_port", "energy"),
            definition=-param("demand_energy"),
        ),
        PortFieldDefinition(
            port_field=PortFieldId("balance_port", "primary_reserve"),
            definition=-param("demand_primary_reserve"),
        ),
    ],
)

THERMAL_CLUSTER_WITH_RESERVE_MODEL_MILP = model(
    id="GEN",
    parameters=[
        float_parameter("p_max", CONSTANT),  # p_max of a single unit
        float_parameter("p_min", CONSTANT),
        float_parameter("d_min_up", CONSTANT),
        float_parameter("d_min_down", CONSTANT),
        float_parameter("cost", CONSTANT),
        float_parameter("startup_cost", CONSTANT),
        float_parameter("fixed_cost", CONSTANT),
        int_parameter("nb_units_min", TIME_AND_SCENARIO_FREE),
        int_parameter("nb_units_max", TIME_AND_SCENARIO_FREE),
        float_parameter("min_generating", TIME_AND_SCENARIO_FREE),
        float_parameter("max_generating", TIME_AND_SCENARIO_FREE),
        int_parameter("max_failure", TIME_AND_SCENARIO_FREE),
        int_parameter("nb_units_max_min_down_time", TIME_AND_SCENARIO_FREE),
    ],
    variables=[
        float_variable(
            "total_generation",
            lower_bound=literal(0),
            structure=ANTICIPATIVE_TIME_VARYING,
        ),
        float_variable(
            "generation_reserve",
            lower_bound=literal(0),
            structure=ANTICIPATIVE_TIME_VARYING,
        ),
        int_variable(
            "nb_on",
            lower_bound=param("nb_units_min"),
            upper_bound=param("nb_units_max"),
            structure=ANTICIPATIVE_TIME_VARYING,
        ),
        int_variable(
            "nb_stop",
            lower_bound=literal(0),
            structure=ANTICIPATIVE_TIME_VARYING,
        ),
        int_variable(
            "nb_failure",
            lower_bound=literal(0),
            upper_bound=param("max_failure"),
            structure=ANTICIPATIVE_TIME_VARYING,
        ),
        int_variable(
            "nb_start",
            lower_bound=literal(0),
            structure=ANTICIPATIVE_TIME_VARYING,
        ),
    ],
    ports=[ModelPort(port_type=RESERVE_PORT_TYPE, port_name="balance_port")],
    port_fields_definitions=[
        PortFieldDefinition(
            port_field=PortFieldId("balance_port", "energy"),
            definition=var("total_generation") - var("generation_reserve"),
        ),
        PortFieldDefinition(
            port_field=PortFieldId("balance_port", "primary_reserve"),
            definition=var("generation_reserve"),
        ),
    ],
    constraints=[
        Constraint(
            "Max generation",
            var("total_generation") <= param("max_generating"),
        ),
        Constraint(
            "Min generation",
            var("total_generation") >= param("min_generating"),
        ),
        Constraint(
            "Max generation with NODU",
            var("total_generation") <= param("p_max") * var("nb_on"),
        ),
        Constraint(
            "Min generation with NODU",
            var("total_generation") >= param("p_min") * var("nb_on"),
        ),
        Constraint(
            "Primary reserve generation above p_min",
            var("generation_reserve")
            <= var("total_generation") - param("p_min") * var("nb_on"),
        ),
        Constraint(
            "NODU balance",
            var("nb_on") == var("nb_on").shift(-1) + var("nb_start") - var("nb_stop"),
        ),
        Constraint(
            "Max failures",
            var("nb_failure") <= var("nb_stop"),
        ),
        Constraint(
            "Min up time",
            var("nb_start")
            .shift(ExpressionRange(-param("d_min_up") + 1, literal(0)))
            .sum()
            - var("nb_failure")
            .shift(ExpressionRange(-param("d_min_up") + 1, literal(0)))
            .sum()
            <= var("nb_on"),
        ),
        Constraint(
            "Min down time",
            var("nb_stop")
            .shift(ExpressionRange(-param("d_min_down") + 1, literal(0)))
            .sum()
            <= param("nb_units_max_min_down_time") - var("nb_on"),
        ),
        # It also works by writing ExpressionRange(-param("d_min_down") + 1, 0) as ExpressionRange's __post_init__ wraps integers to literal nodes. However, MyPy does not seem to infer that ExpressionRange's attributes are necessarily of ExpressionNode type and raises an error if the arguments in the constructor are integer (whereas it runs correctly), this why we specify it here with literal(0) instead of 0.
    ],
    objective_operational_contribution=(
        param("cost") * var("total_generation")
        + param("startup_cost") * var("nb_start")
        + param("fixed_cost") * var("nb_on")
    )
    .sum()
    .expec(),
)
