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
import pytest
from antares.craft.model.area import AreaProperties
from antares.craft.model.st_storage import STStorageProperties
from antares.craft.model.study import Study, create_study_local
from antares.craft.model.thermal import (
    LawOption,
    LocalTSGenerationBehavior,
    ThermalClusterGroup,
    ThermalClusterProperties,
    ThermalCostGeneration,
)


@pytest.fixture
def local_study(tmp_path) -> Study:
    """
    Create an empty study
    """
    study_name = "studyTest"
    study_version = "880"
    return create_study_local(study_name, study_version, tmp_path.absolute())


@pytest.fixture
def default_thermal_cluster_properties() -> ThermalClusterProperties:
    return ThermalClusterProperties(
        group=ThermalClusterGroup.OTHER1,
        enabled=True,
        unit_count=1,
        nominal_capacity=150,
        gen_ts=LocalTSGenerationBehavior.USE_GLOBAL,
        min_stable_power=0,
        min_up_time=1,
        min_down_time=1,
        must_run=False,
        spinning=0,
        volatility_forced=0,
        volatility_planned=0,
        law_forced=LawOption.UNIFORM,
        law_planned=LawOption.UNIFORM,
        marginal_cost=1.1,
        spread_cost=0,
        fixed_cost=0,
        startup_cost=0,
        market_bid_cost=0,
        co2=0,
        nh3=0,
        so2=0,
        nox=0,
        pm2_5=0,
        pm5=0,
        pm10=0,
        nmvoc=0,
        op1=0,
        op2=0,
        op3=0,
        op4=0,
        op5=0,
        cost_generation=ThermalCostGeneration.SET_MANUALLY,
        efficiency=100,
        variable_o_m_cost=0,
    )


@pytest.fixture
def local_study_end_to_end_simple(local_study):
    """
    Create an empty study
    Create an area with custom area properties
    """
    areas_to_create = ["fr"]
    for area in areas_to_create:
        area_properties = AreaProperties(
            energy_cost_spilled="0", energy_cost_unsupplied="1"
        )
        local_study.create_area(area, properties=area_properties)
    return local_study


@pytest.fixture
def local_study_end_to_end_w_thermal(local_study, default_thermal_cluster_properties):
    """
    Create an empty study
    Create an area with custom area properties
    Create a thermal cluster with custom thermal properties
    """
    areas_to_create = ["fr"]
    for area in areas_to_create:
        area_properties = AreaProperties(
            energy_cost_spilled="0", energy_cost_unsupplied="10"
        )
        local_study.create_area(area, properties=area_properties)
    thermal_name = "gaz"
    local_study.get_areas()["fr"].create_thermal_cluster(
        thermal_name, properties=default_thermal_cluster_properties
    )
    return local_study


@pytest.fixture
def default_st_storage_cluster_properties() -> STStorageProperties:
    return STStorageProperties(
        injection_nominal_capacity=10,
        withdrawal_nominal_capacity=10,
        reservoir_capacity=150,
        efficiency=1,
        initial_level=0.5,
        initial_level_optim=False,
        enabled=True,
    )


@pytest.fixture
def local_study_with_st_storage(
    local_study_end_to_end_w_thermal, default_st_storage_cluster_properties
) -> Study:
    """
    Create an empty study
    Create an area with custom area properties
    Create a thermal cluster with custom thermal properties
    Create a short term storage
    """
    storage_name = "battery"
    local_study_end_to_end_w_thermal.get_areas()["fr"].create_st_storage(
        storage_name, properties=default_st_storage_cluster_properties
    )
    return local_study_end_to_end_w_thermal
