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

from pathlib import Path
from typing import Callable, List, Optional

from gems.model import Model, PortType
from gems.model.library import library
from gems.simulation import (
    BlockBorderManagement,
    OptimizationProblem,
    OutputValues,
    TimeBlock,
    build_problem,
)
from gems.study import ConstantData, DataBase, Network, create_component
from gems.study.data import ComponentParameterIndex
from gems.study.parsing import InputComponents, parse_yaml_components
from gems.study.resolve_components import (
    build_data_base,
    build_network,
    resolve_components_and_cnx,
)
from gems.thermal_heuristic.cluster_parameter import (
    complete_database_for_fast_heuristic,
    complete_database_with_cluster_parameters,
)
from gems.thermal_heuristic.time_scenario_parameter import (
    BlockScenarioIndex,
    TimeScenarioHourParameter,
    timesteps,
)


class ThermalProblemBuilder:
    def __init__(
        self,
        time_scenario_hour_parameter: TimeScenarioHourParameter,
        network: Network,
        database: DataBase,
    ) -> None:
        self.time_scenario_hour_parameter = time_scenario_hour_parameter
        self.database = database
        self.network = network

    def main_resolution_step(
        self,
        index: BlockScenarioIndex,
    ) -> OptimizationProblem:
        problem = build_problem(
            self.network,
            self.database,
            TimeBlock(1, timesteps(index, self.time_scenario_hour_parameter)),
            [index.scenario],
            border_management=BlockBorderManagement.CYCLE,
        )

        return problem

    def update_database_heuristic(
        self,
        output: OutputValues,
        index: BlockScenarioIndex,
        list_cluster_id: List[str],
        param_to_update: str,
        var_to_read: str,
        fn_to_apply: Callable,
        param_needed_to_compute: Optional[List[str]] = None,
    ) -> None:
        for cluster in list_cluster_id:
            if (
                ComponentParameterIndex(cluster, param_to_update)
                not in self.database.__dict__.keys()
            ):
                self.database.add_data(cluster, param_to_update, ConstantData(0))
            self.database.convert_to_time_scenario_series_data(
                ComponentParameterIndex(cluster, param_to_update),
                self.time_scenario_hour_parameter.hour
                * self.time_scenario_hour_parameter.week,
                self.time_scenario_hour_parameter.scenario,
            )

            sol = output.component(cluster).var(var_to_read).value[0]  # type:ignore

            param = {}
            if param_needed_to_compute is not None:
                for p in param_needed_to_compute:
                    param[p] = [
                        self.database.get_value(
                            ComponentParameterIndex(cluster, p),
                            t,
                            index.scenario,
                        )
                        for t in timesteps(index, self.time_scenario_hour_parameter)
                    ]

            for i, t in enumerate(timesteps(index, self.time_scenario_hour_parameter)):
                self.database.set_value(
                    ComponentParameterIndex(cluster, param_to_update),
                    fn_to_apply(sol[i], *[p[i] for p in param.values()]),  # type:ignore
                    t,
                    index.scenario,
                )

    def heuristic_resolution_step(
        self,
        index: BlockScenarioIndex,
        id_component: str,
        model: Model,
    ) -> OptimizationProblem:
        cluster = create_component(model=model, id=id_component)

        network = Network("test")
        network.add_component(cluster)

        problem = build_problem(
            network,
            self.database,
            TimeBlock(1, timesteps(index, self.time_scenario_hour_parameter)),
            [index.scenario],
            border_management=BlockBorderManagement.CYCLE,
        )

        return problem


def get_database(
    components_file: InputComponents,
    data_dir: Path,
    fast: bool,
    cluster: List[str],
    time_scenario_hour_parameter: TimeScenarioHourParameter,
) -> DataBase:
    database = build_data_base(components_file, data_dir)

    complete_database_with_cluster_parameters(
        database, cluster, time_scenario_hour_parameter
    )

    if fast:
        complete_database_for_fast_heuristic(
            database, cluster, time_scenario_hour_parameter
        )

    return database


def get_heuristic_components(
    components_file: InputComponents, id_heuristic_model: str
) -> List[str]:
    return [c.id for c in components_file.components if c.model == id_heuristic_model]


def get_network(
    components_file: InputComponents, port_types: List[PortType], models: List[Model]
) -> Network:
    lib = library(
        port_types=port_types,
        models=models,
    )
    components_input = resolve_components_and_cnx(components_file, lib)
    network = build_network(components_input)
    return network


def get_input_components(compo_file: Path) -> InputComponents:
    with compo_file.open() as c:
        components_file = parse_yaml_components(c)
    return components_file
