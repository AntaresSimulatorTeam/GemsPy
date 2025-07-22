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
import pandas as pd
import pytest
from antares.craft.model.area import Area, AreaProperties
from antares.craft.model.hydro import HydroProperties
from antares.craft.model.link import Link
from antares.craft.model.renewable import RenewableClusterProperties
from antares.craft.model.st_storage import STStorageProperties
from antares.craft.model.study import Study, create_study_local
from antares.craft.model.thermal import ThermalClusterProperties


@pytest.fixture
def lib_id() -> str:
    return "antares-historic"


@pytest.fixture
def local_study(tmp_path) -> Study:
    """
    Create an empty study
    """
    study_name = "studyTest"
    study_version = "880"
    return create_study_local(study_name, study_version, tmp_path.absolute())


def create_dataframe_from_constant(
    lines: int,
    columns: int = 1,
    value: int = 1,
) -> pd.DataFrame:
    """
    Creates a DataFrame filled with a constant value for testing.

    Args:
        lines (int): Number of rows in the DataFrame.
        columns (int, optional): Number of columns. Defaults to 1.
        value (int, optional): Constant value to fill the DataFrame. Defaults to 1.
    Returns:
        pd.DataFrame: A DataFrame with the specified dimensions, filled with the constant value.
    """
    data = {f"col_{i+1}": [value] * lines for i in range(columns)}
    return pd.DataFrame(data)


@pytest.fixture
def local_study_w_areas(local_study) -> Study:
    """
    Create an empty study
    Create 2 areas with custom area properties
    """
    areas_to_create = ["fr", "it"]
    for area in areas_to_create:
        area_properties = AreaProperties(
            energy_cost_spilled="1", energy_cost_unsupplied="0.5"
        )
        local_study.create_area(area, properties=area_properties)
    return local_study


@pytest.fixture
def local_study_w_links(local_study_w_areas) -> Study:
    """
    Create an empty study
    Create 2 areas with custom area properties
    Create another area and 3 links
    """
    local_study_w_areas.create_area("at")
    links_to_create = ["fr_at", "at_it", "fr_it"]
    for link in links_to_create:
        area_from, area_to = link.split("_")
        local_study_w_areas.create_link(area_from=area_from, area_to=area_to)

    return local_study_w_areas


@pytest.fixture
def local_study_w_thermal(
    local_study_w_links: Study, request: pytest.FixtureRequest
) -> Study:
    """
    Create an empty study
    Create 2 areas with custom area properties
    Create another area and 3 links
    Create a thermal cluster
    """
    thermal_name = "gaz"
    if hasattr(request, "param"):
        modulation_df, series_df = request.param
        local_study_w_links.get_areas()["fr"].create_thermal_cluster(
            thermal_name,
            ThermalClusterProperties(unit_count=1, nominal_capacity=2.0),
            prepro=None,
            modulation=modulation_df,
            series=series_df,
            co2_cost=None,
            fuel_cost=None,
        )
    else:
        local_study_w_links.get_areas()["fr"].create_thermal_cluster(
            thermal_name, ThermalClusterProperties(unit_count=1, nominal_capacity=2.0)
        )
    return local_study_w_links


@pytest.fixture
def local_study_with_renewable(local_study_w_thermal) -> Study:
    """
    Create an empty study
    Create 2 areas with custom area properties
    Create another area and 3 links
    Create a thermal cluster
    Create a renewable cluster
    """
    renewable_cluster_name = "generation"
    timeseries = pd.DataFrame(
        [
            [-9999999980506447872, 0, 9999999980506447872],
            [0, "fr", 0],
        ],
        dtype="object",
    )
    local_study_w_thermal.get_areas()["fr"].create_renewable_cluster(
        renewable_cluster_name, RenewableClusterProperties(), series=timeseries
    )
    return local_study_w_thermal


@pytest.fixture
def default_storage_cluster_properties() -> STStorageProperties:
    return STStorageProperties(
        injection_nominal_capacity=10,
        withdrawal_nominal_capacity=10,
        reservoir_capacity=0,
        efficiency=1,
        initial_level=0.5,
        initial_level_optim=False,
        enabled=True,
    )


@pytest.fixture
def local_study_with_st_storage(
    local_study_with_renewable, default_storage_cluster_properties
) -> Study:
    """
    Create an empty study
    Create 2 areas with custom area properties
    Create another area and 3 links
    Create a thermal cluster
    Create a renewable cluster
    Create a short term storage
    """
    storage_name = "battery"
    local_study_with_renewable.get_areas()["fr"].create_st_storage(
        storage_name, properties=default_storage_cluster_properties
    )
    return local_study_with_renewable


@pytest.fixture
def local_study_with_hydro(local_study_with_st_storage) -> Study:
    """
    Create an empty study
    Create 2 areas with custom area properties
    Create another area and 3 links
    Create a thermal cluster
    Create a renewable cluster
    Create a short term storage
    Create an hydro cluster
    """
    hydro_properties = HydroProperties()
    local_study_with_st_storage.get_areas()["fr"].hydro.update_properties(
        hydro_properties
    )
    return local_study_with_st_storage


@pytest.fixture
def area_fr(local_study_with_hydro) -> Area:
    """
    return area object from the fixture local_study_with_hydro
    """
    return local_study_with_hydro.get_areas()["fr"]


@pytest.fixture
def fr_solar(area_fr) -> None:
    """
    return area object from the fixture local_study_with_hydro
    """
    return area_fr.set_solar(pd.DataFrame([1, 1, 1]))


@pytest.fixture
def fr_wind(area_fr, request) -> None:
    """
    return area object with a wind object that has custom parameters
    """
    command = request.param if hasattr(request, "param") else [1, 1, 1]
    data = pd.DataFrame(command)
    return area_fr.set_wind(data)


@pytest.fixture
def fr_load(area_fr) -> None:
    """
    return area object with a load object that has custom parameters
    """
    return area_fr.set_load(pd.DataFrame([1, 1, 1]))


@pytest.fixture
def local_study_w_areas_for_battery(local_study) -> Study:
    """
    Create an empty study
    Create 2 areas with custom area properties
    """
    areas_to_create = ["fr", "it"]
    for area in areas_to_create:
        area_properties = AreaProperties(
            energy_cost_spilled="1", energy_cost_unsupplied="0.5"
        )
        local_study.create_area(area, properties=area_properties)
    return local_study


@pytest.fixture
def local_study_with_constraint(
    local_study_w_areas_for_battery, request: pytest.FixtureRequest
) -> Study:
    param = getattr(request, "param", None)
    parameters_df, capacities_df = param if param else (None, None)

    # Add area and links
    local_study_w_areas_for_battery.create_area("z_batteries")
    links_to_create = ["fr|z_batteries", "it|z_batteries", "fr|it"]
    for link in links_to_create:
        area_from, area_to = link.split("|")
        object_link: Link = local_study_w_areas_for_battery.create_link(
            area_from=area_from, area_to=area_to
        )
        if parameters_df is not None:
            object_link.set_parameters(parameters_df)

        if capacities_df is not None:
            object_link.set_capacity_direct(capacities_df)
            object_link.set_capacity_indirect(capacities_df)
    # Add thermal clusters
    for area_id in ["fr", "z_batteries"]:
        local_study_w_areas_for_battery.get_areas()[area_id].create_thermal_cluster(
            f"{area_id}_batteries_inj",
            ThermalClusterProperties(unit_count=1, nominal_capacity=2.0),
        )
        local_study_w_areas_for_battery.get_areas()[area_id].create_thermal_cluster(
            f"z_batteries_batteries_{area_id}_1",
            ThermalClusterProperties(unit_count=1, nominal_capacity=3.0),
        )
        local_study_w_areas_for_battery.get_areas()[area_id].create_thermal_cluster(
            f"z_batteries_batteries_{area_id}_2",
            ThermalClusterProperties(unit_count=1, nominal_capacity=6.0),
        )
    local_study_w_areas_for_battery.get_areas()["z_batteries"].create_thermal_cluster(
        "z_batteries_batteries_fr_1",
        ThermalClusterProperties(unit_count=1, nominal_capacity=6.0),
    )
    local_study_w_areas_for_battery.get_areas()["z_batteries"].create_thermal_cluster(
        "z_batteries_batteries_it_1",
        ThermalClusterProperties(unit_count=1, nominal_capacity=6.0),
    )
    local_study_w_areas_for_battery.get_areas()["it"].create_thermal_cluster(
        "it_batteries_inj", ThermalClusterProperties(unit_count=1, nominal_capacity=6.0)
    )

    # Add binding constraint
    local_study_w_areas_for_battery.create_binding_constraint(name="batteries_fr")

    return local_study_w_areas_for_battery
