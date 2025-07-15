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

"""
The standard module contains the definition of standard models.
"""
from gems.expression import literal, param, var
from gems.expression.expression import port_field
from gems.expression.indexing_structure import IndexingStructure
from gems.model.common import ProblemContext
from gems.model.constraint import Constraint
from gems.model.model import ModelPort, model
from gems.model.parameter import float_parameter, int_parameter
from gems.model.port import PortField, PortFieldDefinition, PortFieldId, PortType
from gems.model.variable import float_variable, int_variable

CONSTANT = IndexingStructure(False, False)
TIME_AND_SCENARIO_FREE = IndexingStructure(True, True)
ANTICIPATIVE_TIME_VARYING = IndexingStructure(True, True)
NON_ANTICIPATIVE_TIME_VARYING = IndexingStructure(True, False)

BALANCE_PORT_TYPE = PortType(id="balance", fields=[PortField("flow")])

NODE_BALANCE_MODEL = model(
    id="NODE_BALANCE_MODEL",
    ports=[ModelPort(port_type=BALANCE_PORT_TYPE, port_name="balance_port")],
    binding_constraints=[
        Constraint(
            name="Balance",
            expression=port_field("balance_port", "flow").sum_connections()
            == literal(0),
        )
    ],
)

NODE_WITH_SPILL_AND_ENS = model(
    id="NODE_WITH_SPILL_AND_ENS_MODEL",
    parameters=[float_parameter("spillage_cost"), float_parameter("ens_cost")],
    variables=[
        float_variable("spillage", lower_bound=literal(0)),
        float_variable("unsupplied_energy", lower_bound=literal(0)),
    ],
    ports=[ModelPort(port_type=BALANCE_PORT_TYPE, port_name="balance_port")],
    binding_constraints=[
        Constraint(
            name="Balance",
            expression=port_field("balance_port", "flow").sum_connections()
            == var("spillage") - var("unsupplied_energy"),
        )
    ],
    objective_operational_contribution=(
        param("spillage_cost") * var("spillage")
        + param("ens_cost") * var("unsupplied_energy")
    )
    .time_sum()
    .expec(),
)

"""
Basic link model using ports
"""
LINK_MODEL = model(
    id="LINK",
    parameters=[float_parameter("f_max", TIME_AND_SCENARIO_FREE)],
    variables=[
        float_variable("flow", lower_bound=-param("f_max"), upper_bound=param("f_max"))
    ],
    ports=[
        ModelPort(port_type=BALANCE_PORT_TYPE, port_name="balance_port_from"),
        ModelPort(port_type=BALANCE_PORT_TYPE, port_name="balance_port_to"),
    ],
    port_fields_definitions=[
        PortFieldDefinition(
            port_field=PortFieldId("balance_port_from", "flow"),
            definition=-var("flow"),
        ),
        PortFieldDefinition(
            port_field=PortFieldId("balance_port_to", "flow"),
            definition=var("flow"),
        ),
    ],
)

"""
A standard model for a fixed demand of energy.
"""
DEMAND_MODEL = model(
    id="FIXED_DEMAND",
    parameters=[
        float_parameter("demand", TIME_AND_SCENARIO_FREE),
    ],
    ports=[ModelPort(port_type=BALANCE_PORT_TYPE, port_name="balance_port")],
    port_fields_definitions=[
        PortFieldDefinition(
            port_field=PortFieldId("balance_port", "flow"),
            definition=-param("demand"),
        )
    ],
)

"""
A standard model for a linear cost generation, limited by a maximum generation.
"""
GENERATOR_MODEL = model(
    id="GEN",
    parameters=[
        float_parameter("p_max", CONSTANT),
        float_parameter("cost", CONSTANT),
    ],
    variables=[float_variable("generation", lower_bound=literal(0))],
    ports=[ModelPort(port_type=BALANCE_PORT_TYPE, port_name="balance_port")],
    port_fields_definitions=[
        PortFieldDefinition(
            port_field=PortFieldId("balance_port", "flow"),
            definition=var("generation"),
        )
    ],
    constraints=[
        Constraint(
            name="Max generation", expression=var("generation") <= param("p_max")
        ),
    ],
    objective_operational_contribution=(param("cost") * var("generation"))
    .time_sum()
    .expec(),
)

GENERATOR_MODEL_WITH_PMIN = model(
    id="GEN",
    parameters=[
        float_parameter("p_max", CONSTANT),
        float_parameter("p_min", CONSTANT),
        float_parameter("cost", CONSTANT),
    ],
    variables=[float_variable("generation")],
    ports=[ModelPort(port_type=BALANCE_PORT_TYPE, port_name="balance_port")],
    port_fields_definitions=[
        PortFieldDefinition(
            port_field=PortFieldId("balance_port", "flow"),
            definition=var("generation"),
        )
    ],
    constraints=[
        Constraint(
            name="Max generation", expression=var("generation") <= param("p_max")
        ),
        Constraint(
            name="Min generation",
            expression=var("generation") - param("p_min"),
            lower_bound=literal(0),
        ),  # To test both ways of setting constraints
    ],
    objective_operational_contribution=(param("cost") * var("generation"))
    .time_sum()
    .expec(),
)

"""
A model for a linear cost generation limited by a maximum generation per time-step
and total generation in whole period. It considers a full storage with no replenishing
"""
GENERATOR_MODEL_WITH_STORAGE = model(
    id="GEN",
    parameters=[
        float_parameter("p_max", CONSTANT),
        float_parameter("cost", CONSTANT),
        float_parameter("full_storage", CONSTANT),
    ],
    variables=[float_variable("generation", lower_bound=literal(0))],
    ports=[ModelPort(port_type=BALANCE_PORT_TYPE, port_name="balance_port")],
    port_fields_definitions=[
        PortFieldDefinition(
            port_field=PortFieldId("balance_port", "flow"),
            definition=var("generation"),
        )
    ],
    constraints=[
        Constraint(
            name="Max generation", expression=var("generation") <= param("p_max")
        ),
        Constraint(
            name="Total storage",
            expression=var("generation").time_sum() <= param("full_storage"),
        ),
    ],
    objective_operational_contribution=(param("cost") * var("generation"))
    .time_sum()
    .expec(),
)

# For now, no starting cost
THERMAL_CLUSTER_MODEL_HD = model(
    id="GEN",
    parameters=[
        float_parameter("p_max", CONSTANT),  # p_max of a single unit
        float_parameter("p_min", CONSTANT),
        float_parameter("d_min_up", CONSTANT),
        float_parameter("d_min_down", CONSTANT),
        float_parameter("cost", CONSTANT),
        int_parameter("nb_units_max", CONSTANT),
        int_parameter("nb_failures", TIME_AND_SCENARIO_FREE),
    ],
    variables=[
        float_variable(
            "generation",
            lower_bound=literal(0),
            upper_bound=param("nb_units_max") * param("p_max"),
            structure=ANTICIPATIVE_TIME_VARYING,
        ),
        int_variable(
            "nb_on",
            lower_bound=literal(0),
            upper_bound=param("nb_units_max"),
            structure=ANTICIPATIVE_TIME_VARYING,
        ),
        int_variable(
            "nb_stop",
            lower_bound=literal(0),
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
            "Min up time",
            var("nb_start").time_sum(-param("d_min_up") + 1, literal(0))
            <= var("nb_on"),
        ),
        Constraint(
            "Min down time",
            var("nb_stop").time_sum(-param("d_min_down") + 1, literal(0))
            <= param("nb_units_max").shift(-param("d_min_down")) - var("nb_on"),
        ),
    ],
    objective_operational_contribution=(param("cost") * var("generation"))
    .time_sum()
    .expec(),
)

# Same model as previous one, except that starting/stopping variables are now non anticipative
THERMAL_CLUSTER_MODEL_DHD = model(
    id="GEN",
    parameters=[
        float_parameter("p_max", CONSTANT),  # p_max of a single unit
        float_parameter("p_min", CONSTANT),
        float_parameter("d_min_up", CONSTANT),
        float_parameter("d_min_down", CONSTANT),
        float_parameter("cost", CONSTANT),
        int_parameter("nb_units_max", CONSTANT),
        int_parameter("nb_failures", TIME_AND_SCENARIO_FREE),
    ],
    variables=[
        float_variable(
            "generation",
            lower_bound=literal(0),
            upper_bound=param("nb_units_max") * param("p_max"),
            structure=ANTICIPATIVE_TIME_VARYING,
        ),
        int_variable(
            "nb_on",
            lower_bound=literal(0),
            upper_bound=param("nb_units_max"),
            structure=NON_ANTICIPATIVE_TIME_VARYING,
        ),
        int_variable(
            "nb_stop",
            lower_bound=literal(0),
            structure=NON_ANTICIPATIVE_TIME_VARYING,
        ),
        int_variable(
            "nb_start",
            lower_bound=literal(0),
            structure=NON_ANTICIPATIVE_TIME_VARYING,
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
            "Min up time",
            var("nb_start").time_sum(-param("d_min_up") + 1, literal(0))
            <= var("nb_on"),
        ),
        Constraint(
            "Min down time",
            var("nb_stop").time_sum(-param("d_min_down") + 1, literal(0))
            <= param("nb_units_max").shift(-param("d_min_down")) - var("nb_on"),
        ),
    ],
    objective_operational_contribution=(param("cost") * var("generation"))
    .time_sum()
    .expec(),
)

SPILLAGE_MODEL = model(
    id="SPI",
    parameters=[float_parameter("cost", CONSTANT)],
    variables=[float_variable("spillage", lower_bound=literal(0))],
    ports=[ModelPort(port_type=BALANCE_PORT_TYPE, port_name="balance_port")],
    port_fields_definitions=[
        PortFieldDefinition(
            port_field=PortFieldId("balance_port", "flow"),
            definition=-var("spillage"),
        )
    ],
    objective_operational_contribution=(param("cost") * var("spillage"))
    .time_sum()
    .expec(),
)

UNSUPPLIED_ENERGY_MODEL = model(
    id="UNSP",
    parameters=[float_parameter("cost", CONSTANT)],
    variables=[float_variable("unsupplied_energy", lower_bound=literal(0))],
    ports=[ModelPort(port_type=BALANCE_PORT_TYPE, port_name="balance_port")],
    port_fields_definitions=[
        PortFieldDefinition(
            port_field=PortFieldId("balance_port", "flow"),
            definition=var("unsupplied_energy"),
        )
    ],
    objective_operational_contribution=(param("cost") * var("unsupplied_energy"))
    .time_sum()
    .expec(),
)

# Simplified model
# - In antares-solver, some constraints have modulation we decide not to include it here for the sake of simplicity.
# - No capacity (level max)
# - The initial level is not fixed (it is optimized)
SHORT_TERM_STORAGE_SIMPLE = model(
    id="STS_SIMPLE",
    parameters=[
        float_parameter("p_max_injection"),
        float_parameter("p_max_withdrawal"),
        float_parameter("level_min"),
        float_parameter("level_max"),
        float_parameter("inflows"),
        float_parameter(
            "efficiency"
        ),  # Should be constant, but time-dependent values should work as well
    ],
    variables=[
        float_variable(
            "injection", lower_bound=literal(0), upper_bound=param("p_max_injection")
        ),
        float_variable(
            "withdrawal", lower_bound=literal(0), upper_bound=param("p_max_withdrawal")
        ),
        float_variable(
            "level", lower_bound=param("level_min"), upper_bound=param("level_max")
        ),
    ],
    ports=[ModelPort(port_type=BALANCE_PORT_TYPE, port_name="balance_port")],
    port_fields_definitions=[
        PortFieldDefinition(
            port_field=PortFieldId("balance_port", "flow"),
            definition=var("withdrawal") - var("injection"),
        )
    ],
    constraints=[
        Constraint(
            name="Level",
            expression=var("level")
            - var("level").shift(-1)
            - param("efficiency") * var("injection")
            + var("withdrawal")
            == param("inflows"),
        ),
    ],
    objective_operational_contribution=literal(0),  # Implcitement nul ?
)

""" Simple thermal unit that can be invested on"""
THERMAL_CANDIDATE = model(
    id="GEN",
    parameters=[
        float_parameter("op_cost", CONSTANT),
        float_parameter("invest_cost", CONSTANT),
        float_parameter("max_invest", CONSTANT),
    ],
    variables=[
        float_variable("generation", lower_bound=literal(0)),
        float_variable(
            "p_max",
            lower_bound=literal(0),
            upper_bound=param("max_invest"),
            structure=CONSTANT,
            context=ProblemContext.COUPLING,
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
        Constraint(name="Max generation", expression=var("generation") <= var("p_max"))
    ],
    objective_operational_contribution=(param("op_cost") * var("generation"))
    .time_sum()
    .expec(),
    objective_investment_contribution=param("invest_cost") * var("p_max"),
)

""" Simple thermal unit that can be invested on and with already installed capacity"""
THERMAL_CANDIDATE_WITH_ALREADY_INSTALLED_CAPA = model(
    id="GEN",
    parameters=[
        float_parameter("op_cost", CONSTANT),
        float_parameter("invest_cost", CONSTANT),
        float_parameter("max_invest", CONSTANT),
        float_parameter("already_installed_capa", CONSTANT),
    ],
    variables=[
        float_variable("generation", lower_bound=literal(0)),
        float_variable(
            "invested_capa",
            lower_bound=literal(0),
            upper_bound=param("max_invest"),
            structure=CONSTANT,
            context=ProblemContext.COUPLING,
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
            name="Max generation",
            expression=var("generation")
            <= param("already_installed_capa") + var("invested_capa"),
        )
    ],
    objective_operational_contribution=(param("op_cost") * var("generation"))
    .time_sum()
    .expec(),
    objective_investment_contribution=param("invest_cost") * var("invested_capa"),
)
