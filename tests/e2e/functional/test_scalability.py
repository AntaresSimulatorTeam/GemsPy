import time

import numpy as np
import pandas as pd

from gems.simulation import TimeBlock, build_problem
from gems.study import (
    Component,
    ConstantData,
    DataBase,
    PortRef,
    Study,
    System,
    TimeScenarioSeriesData,
    create_component,
)
from tests.e2e.functional.libs.standard import (
    DEMAND_MODEL,
    GENERATOR_MODEL_WITH_STORAGE,
    NODE_BALANCE_MODEL,
)


def generate_scalar_matrix_data(
    value: float, horizon: int, scenarios: int
) -> TimeScenarioSeriesData:
    data = pd.DataFrame(value, index=range(horizon), columns=range(scenarios))

    return TimeScenarioSeriesData(time_scenario_series=data)


def test_basic_balance_on_whole_year_with_large_sum() -> None:
    """
    Balance on one node with one fixed demand and one generation with storage, on 8760 timestep.
    """

    durations = {}
    for horizon in np.logspace(1, 6, num=10):
        durations[int(horizon)] = build_for_horizon(int(horizon), 1)

    duration_df = pd.DataFrame.from_dict(durations, orient="index")
    duration_df.columns = pd.Index(["build time"])
    print(duration_df)
    duration_df.to_csv("build_time_scalability.csv")


def build_for_horizon(horizon_size: int, scenario_count: int) -> float:
    scenarios = scenario_count
    time_block = TimeBlock(1, list(range(horizon_size)))
    database = DataBase()
    database.add_data(
        "D", "demand", generate_scalar_matrix_data(100, horizon_size, scenarios)
    )

    database.add_data("G", "p_max", ConstantData(100))
    database.add_data("G", "cost", ConstantData(30))
    database.add_data("G", "full_storage", ConstantData(100 * horizon_size))

    node = Component(model=NODE_BALANCE_MODEL, id="N")
    demand = create_component(model=DEMAND_MODEL, id="D")
    gen = create_component(
        model=GENERATOR_MODEL_WITH_STORAGE, id="G"
    )  # Limits the total generation inside a TimeBlock

    system = System("test")
    system.add_component(node)
    system.add_component(demand)
    system.add_component(gen)
    system.connect(PortRef(demand, "balance_port"), PortRef(node, "balance_port"))
    system.connect(PortRef(gen, "balance_port"), PortRef(node, "balance_port"))

    start = time.time()
    problem = build_problem(Study(system, database), time_block, list(range(scenarios)))
    end = time.time()
    print(f"Time elapsed for horizon {horizon_size}: {end - start:.4f}")
    return end - start


if __name__ == "__main__":
    test_basic_balance_on_whole_year_with_large_sum()
