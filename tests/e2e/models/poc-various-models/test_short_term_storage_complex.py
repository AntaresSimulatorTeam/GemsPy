import math

import pandas as pd
from libs.standard import (
    DEMAND_MODEL,
    NODE_BALANCE_MODEL,
    SPILLAGE_MODEL,
    UNSUPPLIED_ENERGY_MODEL,
)
from libs.standard_sc import SHORT_TERM_STORAGE_COMPLEX

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


def generate_data(
    efficiency: float, horizon: int, scenarios: int
) -> TimeScenarioSeriesData:
    # Create an empty DataFrame with index being the range of the horizon
    data = pd.DataFrame(index=range(horizon))

    for scenario in range(scenarios):
        # Create a column name based on the scenario number
        column_name = f"scenario_{scenario}"
        data[column_name] = 0.0  # Initialize the column with zeros

        for absolute_timestep in range(horizon):
            if absolute_timestep == 0:
                data.at[absolute_timestep, column_name] = -18
            else:
                data.at[absolute_timestep, column_name] = 2 * efficiency

    # Return as TimeScenarioSeriesData object
    return TimeScenarioSeriesData(time_scenario_series=data)


def short_term_storage_base(efficiency: float, horizon: int, result: int) -> None:
    # 18 produced in the 1st time-step, then consumed 2 * efficiency in the rest
    time_blocks = [TimeBlock(0, list(range(horizon)))]
    scenarios = 1
    database = DataBase()

    database.add_data("D", "demand", generate_data(efficiency, horizon, scenarios))

    database.add_data("U", "cost", ConstantData(10))
    database.add_data("S", "cost", ConstantData(1))

    database.add_data("STS1", "p_max_injection", ConstantData(100))
    database.add_data("STS1", "p_max_withdrawal", ConstantData(50))
    database.add_data("STS1", "level_min", ConstantData(0))
    database.add_data("STS1", "level_max", ConstantData(1000))
    database.add_data("STS1", "inflows", ConstantData(0))
    database.add_data("STS1", "efficiency", ConstantData(efficiency))
    database.add_data("STS1", "withdrawal_penality", ConstantData(5))
    database.add_data("STS1", "level_penality", ConstantData(0))
    database.add_data("STS1", "Pgrad+i_penality", ConstantData(0))
    database.add_data("STS1", "Pgrad-i_penality", ConstantData(0))
    database.add_data("STS1", "Pgrad+s_penality", ConstantData(0))
    database.add_data("STS1", "Pgrad-s_penality", ConstantData(0))

    node = Component(model=NODE_BALANCE_MODEL, id="1")
    spillage = create_component(model=SPILLAGE_MODEL, id="S")

    unsupplied = create_component(model=UNSUPPLIED_ENERGY_MODEL, id="U")

    demand = create_component(model=DEMAND_MODEL, id="D")

    short_term_storage = create_component(
        model=SHORT_TERM_STORAGE_COMPLEX,
        id="STS1",
    )

    system = System("test")
    system.add_component(node)
    for component in [demand, short_term_storage, spillage, unsupplied]:
        system.add_component(component)
    system.connect(PortRef(demand, "balance_port"), PortRef(node, "balance_port"))
    system.connect(
        PortRef(short_term_storage, "balance_port"), PortRef(node, "balance_port")
    )
    system.connect(PortRef(spillage, "balance_port"), PortRef(node, "balance_port"))
    system.connect(PortRef(unsupplied, "balance_port"), PortRef(node, "balance_port"))

    problem = build_problem(
        Study(system, database),
        time_blocks[0],
        list(range(scenarios)),
    )
    problem.solve(solver_name="highs")
    assert problem.termination_condition == "optimal"
    assert math.isclose(problem.objective_value, result)

    # TODO: update variable access

    database.add_data("STS1", "withdrawal_penality", ConstantData(0))
    database.add_data("STS1", "level_penality", ConstantData(5))
    database.add_data("STS1", "Pgrad+i_penality", ConstantData(0))
    database.add_data("STS1", "Pgrad-i_penality", ConstantData(0))
    database.add_data("STS1", "Pgrad+s_penality", ConstantData(0))
    database.add_data("STS1", "Pgrad-s_penality", ConstantData(0))

    problem.solve(solver_name="highs")
    assert problem.termination_condition == "optimal"
    assert math.isclose(problem.objective_value, result)

    # TODO: update variable access

    database.add_data("STS1", "withdrawal_penality", ConstantData(0))
    database.add_data("STS1", "level_penality", ConstantData(0))
    database.add_data("STS1", "Pgrad+i_penality", ConstantData(5))
    database.add_data("STS1", "Pgrad-i_penality", ConstantData(0))
    database.add_data("STS1", "Pgrad+s_penality", ConstantData(0))
    database.add_data("STS1", "Pgrad-s_penality", ConstantData(0))

    problem.solve(solver_name="highs")
    assert problem.termination_condition == "optimal"
    assert math.isclose(problem.objective_value, result)

    # TODO: update variable access


def test_short_test_horizon_10() -> None:
    short_term_storage_base(efficiency=0.8, horizon=10, result=72)


def test_short_test_horizon_5() -> None:
    short_term_storage_base(efficiency=0.2, horizon=5, result=18)
