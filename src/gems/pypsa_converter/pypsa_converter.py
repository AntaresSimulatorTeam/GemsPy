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
import logging
from dataclasses import dataclass
from math import inf
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from pypsa import Network

from gems.pypsa_converter.utils import any_to_float
from gems.study.parsing import (
    InputComponent,
    InputComponentParameter,
    InputPortConnections,
    InputSystem,
)


@dataclass
class PyPSAComponentData:
    pypsa_model_id: str
    constant_data: pd.DataFrame
    time_dependent_data: dict[str, pd.DataFrame]
    gems_model_id: str
    pypsa_params_to_gems_params: dict[str, str]
    pypsa_params_to_gems_connections: dict[str, tuple[str, str]]

    def check_params_consistency(self) -> None:
        for key in self.pypsa_params_to_gems_params:
            self._check_key_in_constant_data(key)
        for key in self.pypsa_params_to_gems_connections:
            self._check_key_in_constant_data(key)

    def _check_key_in_constant_data(self, key: str) -> None:
        if key not in self.constant_data.columns:
            raise ValueError(
                f"Parameter {key} not available in constant data, defining all available parameters for model {self.pypsa_model_id}"
            )


@dataclass
class PyPSAGlobalConstraintData:
    pypsa_name: str
    # pypsa_investment_period
    pypsa_carrier_attribute: str
    pypsa_sense: str
    pypsa_constant: float
    gems_model_id: str  # gems model for this GlobalConstraint
    gems_port_id: str  # gems port for this GlobalConstraint
    gems_components_and_ports: list[tuple[str, str]]


class PyPSAStudyConverter:
    def __init__(
        self,
        pypsa_network: Network,
        logger: logging.Logger,
        system_dir: Path,
        series_dir: Path,
        series_file_format: str,
    ):
        """
        Initialize processor
        """
        self.logger = logger
        self.system_dir = system_dir
        self.series_dir = series_dir
        self.pypsa_network = pypsa_network
        self.pypsalib_id = "pypsa_models"
        self.null_carrier_id = "null"
        self.system_name = pypsa_network.name
        self.series_file_format = series_file_format

        self.pypsa_components = [
            "buses",
            "loads",
            "generators",
            "stores",
            "storage_units",
            "links",
            "lines",
            "transformers",
        ]
        self._pypsa_network_assertion()
        self._pypsa_network_preprocessing()
        self._preprocess_pypsa_components("loads", False, "/")
        self._preprocess_pypsa_components("generators", True, "p_nom")
        self._preprocess_pypsa_components("stores", True, "e_nom")
        self._preprocess_pypsa_components("storage_units", True, "p_nom")
        self._preprocess_pypsa_components("links", True, "p_nom")

        self.pypsa_components_data: dict[str, PyPSAComponentData] = {}
        self._register_pypsa_components()
        self.pypsa_globalconstraints_data: dict[str, PyPSAGlobalConstraintData] = {}
        self._register_pypsa_globalconstraints()

    def _pypsa_network_assertion(self) -> None:
        """Assertion function to keep trace of the limitations of the converter"""
        assert len(self.pypsa_network.investment_periods) == 0
        assert (self.pypsa_network.snapshot_weightings.values == 1.0).all()
        ### PyPSA components : Generators
        if not (all((self.pypsa_network.generators["marginal_cost_quadratic"] == 0))):
            raise ValueError(f"Converter supports only Generators with linear cost")
        if not (all((self.pypsa_network.generators["active"] == 1))):
            raise ValueError(f"Converter supports only Generators with active = 1")
        if not (all((self.pypsa_network.generators["committable"] == False))):
            raise ValueError(
                f"Converter supports only Generators with commitable = False"
            )
        ### PyPSA components : Loads
        if not (all((self.pypsa_network.loads["active"] == 1))):
            raise ValueError(f"Converter supports only Loads with active = 1")
        ### PyPSA components : Links
        if not (all((self.pypsa_network.links["active"] == 1))):
            raise ValueError(f"Converter supports only Links with active = 1")
        ### PyPSA components : Lines
        if not len(self.pypsa_network.lines) == 0:
            raise ValueError(f"Converter does not support Lines yet")
        ### PyPSA components : Storage Units
        if not (all((self.pypsa_network.links["active"] == 1))):
            raise ValueError(f"Converter supports only Storage Units with active = 1")
        if not (all((self.pypsa_network.storage_units["sign"] == 1))):
            raise ValueError(f"Converter supports only Storage Units with sign = 1")
        if not (all((self.pypsa_network.storage_units["cyclic_state_of_charge"] == 1))):
            raise ValueError(
                f"Converter supports only Storage Units with cyclic_state_of_charge"
            )
        if not (
            all((self.pypsa_network.storage_units["marginal_cost_quadratic"] == 0))
        ):
            raise ValueError(f"Converter supports only Storage Units with linear cost")
        ### PyPSA components : Stores
        if not (all((self.pypsa_network.links["active"] == 1))):
            raise ValueError(f"Converter supports only Stores with active = 1")
        if not (all((self.pypsa_network.stores["sign"] == 1))):
            raise ValueError(f"Converter supports only Stores with sign = 1")
        if not (all((self.pypsa_network.stores["e_cyclic"] == 1))):
            raise ValueError(f"Converter supports only Stores with e_cyclic = True")
        if not (all((self.pypsa_network.stores["marginal_cost_quadratic"] == 0))):
            raise ValueError(f"Converter supports only Stores with linear cost")
        ### PyPSA components : GlobalConstraint
        for pypsa_model_id in self.pypsa_network.global_constraints.index:
            assert (
                self.pypsa_network.global_constraints.loc[pypsa_model_id, "type"]
                == "primary_energy"
            )
            assert (
                self.pypsa_network.global_constraints.loc[
                    pypsa_model_id, "carrier_attribute"
                ]
                == "co2_emissions"
            )

    def _rename_buses(self) -> None:
        ### Rename PyPSA buses, to delete spaces
        if len(self.pypsa_network.buses) > 0:
            self.pypsa_network.buses.index = self.pypsa_network.buses.index.str.replace(
                " ", "_"
            )
            for _, val in self.pypsa_network.buses_t.items():
                val.columns = val.columns.str.replace(" ", "_")
        ### Update the 'bus' columns for the different types of PyPSA components
        for component_type in self.pypsa_components:
            df = getattr(self.pypsa_network, component_type)
            if len(df) > 0:
                for col in ["bus", "bus0", "bus1"]:
                    if col in df.columns:
                        df[col] = df[col].str.replace(" ", "_")

    def _pypsa_network_preprocessing(self) -> None:
        ###Add fictitious carrier
        self.pypsa_network.add(
            "Carrier",
            self.null_carrier_id,
            co2_emissions=0,
            max_growth=any_to_float(inf),
        )
        self.pypsa_network.carriers[
            "carrier"
        ] = self.pypsa_network.carriers.index.values
        self._rename_buses()

    def _rename_pypsa_components(self, component_type: str) -> None:
        df = getattr(self.pypsa_network, component_type)
        if len(df) == 0:
            return
        ### Rename PyPSA components, to make sure that the names are uniques (used as id in the Gems model)
        prefix = component_type[:-1]
        df.index = prefix + "_" + df.index.str.replace(" ", "_")
        dictionnary = getattr(self.pypsa_network, component_type + "_t")
        for _, val in dictionnary.items():
            val.columns = prefix + "_" + val.columns.str.replace(" ", "_")

    def _fix_capacities(self, component_type: str, capa_str: str) -> None:
        df = getattr(self.pypsa_network, component_type)
        if len(df) == 0:
            return
        ### Adding min and max capacities to non-extendable objects
        for field in [capa_str + "_min", capa_str + "_max"]:
            df.loc[df[capa_str + "_extendable"] == False, field] = df[capa_str]
            df.loc[df[capa_str + "_extendable"] == False, "capital_cost"] = 0.0

    def _preprocess_pypsa_components(
        self, component_type: str, extendable: bool, capa_str: str
    ) -> None:
        ### Handling PyPSA objects without carriers
        df = getattr(self.pypsa_network, component_type)
        for comp in df.index:
            if len(df.loc[comp, "carrier"]) == 0:
                df.loc[comp, "carrier"] = self.null_carrier_id
        setattr(
            self.pypsa_network,
            component_type,
            df.join(
                self.pypsa_network.carriers,
                on="carrier",
                how="left",
                rsuffix="_carrier",
            ),
        )
        self._rename_pypsa_components(component_type)
        if extendable:
            self._fix_capacities(component_type, capa_str)

    def _register_pypsa_components(self) -> None:
        ### PyPSA components : Generators
        self._register_pypsa_components_of_given_model(
            "generators",
            self.pypsa_network.generators,
            self.pypsa_network.generators_t,
            "generator",
            {
                "p_nom_min": "p_nom_min",
                "p_nom_max": "p_nom_max",
                "p_min_pu": "p_min_pu",
                "p_max_pu": "p_max_pu",
                "marginal_cost": "marginal_cost",
                "capital_cost": "capital_cost",
                "e_sum_min": "e_sum_min",
                "e_sum_max": "e_sum_max",
                "sign": "sign",
                "efficiency": "efficiency",
                "co2_emissions": "emission_factor",
            },
            {"bus": ("p_balance_port", "p_balance_port")},
        )
        ### PyPSA components : Loads
        self._register_pypsa_components_of_given_model(
            "loads",
            self.pypsa_network.loads,
            self.pypsa_network.loads_t,
            "load",
            {
                "p_set": "p_set",
                "q_set": "q_set",
                "sign": "sign",
            },
            {"bus": ("p_balance_port", "p_balance_port")},
        )
        ### PyPSA components : Buses
        self._register_pypsa_components_of_given_model(
            "buses",
            self.pypsa_network.buses,
            self.pypsa_network.buses_t,
            "bus",
            {
                "v_nom": "v_nom",
                "x": "x",
                "y": "y",
                "v_mag_pu_set": "v_mag_pu_set",
                "v_mag_pu_min": "v_mag_pu_min",
                "v_mag_pu_max": "v_mag_pu_max",
            },
            {},
        )
        ### PyPSA components : Links
        self._register_pypsa_components_of_given_model(
            "links",
            self.pypsa_network.links,
            self.pypsa_network.links_t,
            "link",
            {
                "efficiency": "efficiency",
                "p_nom_min": "p_nom_min",
                "p_nom_max": "p_nom_max",
                "p_min_pu": "p_min_pu",
                "p_max_pu": "p_max_pu",
                "marginal_cost": "marginal_cost",
                "capital_cost": "capital_cost",
            },
            {
                "bus0": ("p0_port", "p_balance_port"),
                "bus1": ("p1_port", "p_balance_port"),
            },
        )
        ### PyPSA components : Storage Units
        self._register_pypsa_components_of_given_model(
            "storage_units",
            self.pypsa_network.storage_units,
            self.pypsa_network.storage_units_t,
            "storage_unit",
            {
                "p_nom_min": "p_nom_min",
                "p_nom_max": "p_nom_max",
                "p_min_pu": "p_min_pu",
                "p_max_pu": "p_max_pu",
                "sign": "sign",
                "efficiency_store": "efficiency_store",
                "efficiency_dispatch": "efficiency_dispatch",
                "standing_loss": "standing_loss",
                "max_hours": "max_hours",
                "marginal_cost": "marginal_cost",
                "capital_cost": "capital_cost",
                "marginal_cost_storage": "marginal_cost_storage",
                "spill_cost": "spill_cost",
                "inflow": "inflow",
                "co2_emissions": "emission_factor",
            },
            {"bus": ("p_balance_port", "p_balance_port")},
        )
        ### PyPSA components : Stores
        self._register_pypsa_components_of_given_model(
            "stores",
            self.pypsa_network.stores,
            self.pypsa_network.stores_t,
            "store",
            {
                "sign": "sign",
                "e_nom_min": "e_nom_min",
                "e_nom_max": "e_nom_max",
                "e_min_pu": "e_min_pu",
                "e_max_pu": "e_max_pu",
                "standing_loss": "standing_loss",
                "marginal_cost": "marginal_cost",
                "capital_cost": "capital_cost",
                "marginal_cost_storage": "marginal_cost_storage",
                "co2_emissions": "emission_factor",
            },
            {"bus": ("p_balance_port", "p_balance_port")},
        )

    def _add_contributors_to_globalconstraints(
        self, gems_components_and_ports: list[tuple[str, str]], component_type: str
    ) -> list[tuple[str, str]]:
        df = getattr(self.pypsa_network, component_type)
        gems_components_and_ports += [
            (comp, "emission_port")
            for comp in df[df["carrier"] != self.null_carrier_id].index
        ]
        return gems_components_and_ports

    def _register_pypsa_globalconstraints(self) -> None:
        gems_components_and_ports: list[tuple[str, str]] = []
        for component_type in ["generators", "stores", "storage_units"]:
            gems_components_and_ports = self._add_contributors_to_globalconstraints(
                gems_components_and_ports, component_type
            )

        for pypsa_model_id in self.pypsa_network.global_constraints.index:
            name, sense, carrier_attribute = (
                pypsa_model_id,
                self.pypsa_network.global_constraints.loc[pypsa_model_id, "sense"],
                self.pypsa_network.global_constraints.loc[
                    pypsa_model_id, "carrier_attribute"
                ],
            )
            if carrier_attribute == "co2_emissions" and sense == "<=":
                self.pypsa_globalconstraints_data[
                    pypsa_model_id
                ] = PyPSAGlobalConstraintData(
                    name,
                    carrier_attribute,
                    sense,
                    self.pypsa_network.global_constraints.loc[
                        pypsa_model_id, "constant"
                    ],
                    "global_constraint_co2_max",
                    "emission_port",
                    gems_components_and_ports,
                )
            elif carrier_attribute == "co2_emissions" and sense == "==":
                self.pypsa_globalconstraints_data[
                    pypsa_model_id
                ] = PyPSAGlobalConstraintData(
                    name,
                    carrier_attribute,
                    sense,
                    self.pypsa_network.global_constraints.loc[
                        pypsa_model_id, "constant"
                    ],
                    "global_constraint_co2_eq",
                    "emission_port",
                    gems_components_and_ports,
                )
            else:
                raise ValueError("Type of GlobalConstraint not supported.")

    def _register_pypsa_components_of_given_model(
        self,
        pypsa_model_id: str,
        constant_data: pd.DataFrame,
        time_dependent_data: dict[str, pd.DataFrame],
        gems_model_id: str,
        pypsa_params_to_gems_params: dict[str, str],
        pypsa_params_to_gems_connections: dict[str, tuple[str, str]],
    ) -> None:
        if pypsa_model_id in self.pypsa_components_data:
            raise ValueError(f"{pypsa_model_id} already registered !")

        self.pypsa_components_data[pypsa_model_id] = PyPSAComponentData(
            pypsa_model_id,
            constant_data,
            time_dependent_data,
            gems_model_id,
            pypsa_params_to_gems_params,
            pypsa_params_to_gems_connections,
        )

    def to_gems_study(self) -> InputSystem:
        """Main function, to export PyPSA as Gems system"""

        self.logger.info("Study conversion started")
        list_components, list_connections = [], []

        for pypsa_components_data in self.pypsa_components_data.values():
            components, connections = self._convert_pypsa_components_of_given_model(
                pypsa_components_data
            )
            list_components.extend(components)
            list_connections.extend(connections)

        for pypsa_global_constraint_data in self.pypsa_globalconstraints_data.values():
            (
                components,
                connections,
            ) = self._convert_pypsa_globalconstraint_of_given_model(
                pypsa_global_constraint_data
            )
            list_components.extend(components)
            list_connections.extend(connections)

        return InputSystem(
            nodes=[], components=list_components, connections=list_connections
        )

    def _convert_pypsa_components_of_given_model(
        self, pypsa_components_data: PyPSAComponentData
    ) -> tuple[list[InputComponent], list[InputPortConnections]]:
        """
        Generic function to handle the different PyPSA classes

        """

        self.logger.info(
            f"Creating objects of type: {pypsa_components_data.gems_model_id}. "
        )

        # We test whether the keys of the conversion dictionary are allowed in the PyPSA model : all authorized parameters are columns in the constant data frame (even though they are specified as time-varying values in the time-varying data frame)
        pypsa_components_data.check_params_consistency()

        # List of params that may be time-dependent in the pypsa model, among those we want to keep
        time_dependent_params = set(
            pypsa_components_data.pypsa_params_to_gems_params
        ).intersection(set(pypsa_components_data.time_dependent_data.keys()))
        # Save time series and memorize the time-dependent parameters
        comp_param_to_timeseries_name = self._write_and_register_timeseries(
            pypsa_components_data.time_dependent_data, time_dependent_params
        )

        connections = self._create_gems_connections(
            pypsa_components_data.constant_data,
            pypsa_components_data.pypsa_params_to_gems_connections,
        )

        components = self._create_gems_components(
            pypsa_components_data.constant_data,
            pypsa_components_data.gems_model_id,
            pypsa_components_data.pypsa_params_to_gems_params,
            comp_param_to_timeseries_name,
        )
        return components, connections

    def _convert_pypsa_globalconstraint_of_given_model(
        self, pypsa_gc_data: PyPSAGlobalConstraintData
    ) -> tuple[list[InputComponent], list[InputPortConnections]]:
        self.logger.info(
            f"Creating PyPSA GlobalConstraint of type: {pypsa_gc_data.gems_model_id}. "
        )
        components = [
            InputComponent(
                id=pypsa_gc_data.pypsa_name,
                model=f"{self.pypsalib_id}.{pypsa_gc_data.gems_model_id}",
                parameters=[
                    InputComponentParameter(
                        id="quota",
                        time_dependent=False,
                        scenario_dependent=False,
                        value=pypsa_gc_data.pypsa_constant,
                    )
                ],
            )
        ]
        connections = []
        for component_id, port_id in pypsa_gc_data.gems_components_and_ports:
            connections.append(
                InputPortConnections(
                    component1=pypsa_gc_data.pypsa_name,
                    port1=pypsa_gc_data.gems_port_id,
                    component2=component_id,
                    port2=port_id,
                )
            )

        return components, connections

    def _write_and_register_timeseries(
        self,
        time_dependent_data: dict[str, pd.DataFrame],
        time_dependent_params: set[str],
    ) -> dict[tuple[str, str], str]:
        comp_param_to_timeseries_name = dict()
        for param in time_dependent_params:
            param_df = time_dependent_data[param]
            for component in param_df.columns:
                timeseries_name = self.system_name + "_" + component + "_" + param
                comp_param_to_timeseries_name[(component, param)] = timeseries_name
                param_df[[component]].to_csv(
                    self.series_dir / Path(timeseries_name + self.series_file_format),
                    index=False,
                    header=False,
                )

        return comp_param_to_timeseries_name

    def _create_gems_components(
        self,
        constant_data: pd.DataFrame,
        gems_model_id: str,
        pypsa_params_to_gems_params: dict[str, str],
        comp_param_to_timeseries_name: dict[tuple[str, str], str],
    ) -> list[InputComponent]:
        components = []
        for component in constant_data.index:
            components.append(
                InputComponent(
                    id=component,
                    model=f"{self.pypsalib_id}.{gems_model_id}",
                    parameters=[
                        InputComponentParameter(
                            id=pypsa_params_to_gems_params[param],
                            time_dependent=(component, param)
                            in comp_param_to_timeseries_name,
                            scenario_dependent=False,
                            value=(
                                comp_param_to_timeseries_name[(component, param)]
                                if (component, param) in comp_param_to_timeseries_name
                                else any_to_float(constant_data.loc[component, param])
                            ),
                        )
                        for param in pypsa_params_to_gems_params
                    ],
                )
            )
        return components

    def _create_gems_connections(
        self,
        constant_data: pd.DataFrame,
        pypsa_params_to_gems_connections: dict[str, tuple[str, str]],
    ) -> list[InputPortConnections]:
        connections = []
        for bus_id, (
            model_port,
            bus_port,
        ) in pypsa_params_to_gems_connections.items():
            buses = constant_data[bus_id].values
            for component_id, component in enumerate(constant_data.index):
                connections.append(
                    InputPortConnections(
                        component1=buses[component_id],
                        port1=bus_port,
                        component2=component,
                        port2=model_port,
                    )
                )
        return connections
