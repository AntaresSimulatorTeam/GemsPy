import math

import pandas as pd
from libs.standard import (
    DEMAND_MODEL,
    NODE_BALANCE_MODEL,
    SPILLAGE_MODEL,
    UNSUPPLIED_ENERGY_MODEL,
)
from libs.standard_sc import SHORT_TERM_STORAGE_COMPLEX

from gems.simulation import BlockBorderManagement, TimeBlock, build_problem
from gems.study import (
    ConstantData,
    DataBase,
    Network,
    Node,
    PortRef,
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
        data[column_name] = 0  # Initialize the column with zeros

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

    node = Node(model=NODE_BALANCE_MODEL, id="1")
    spillage = create_component(model=SPILLAGE_MODEL, id="S")

    unsupplied = create_component(model=UNSUPPLIED_ENERGY_MODEL, id="U")

    demand = create_component(model=DEMAND_MODEL, id="D")

    short_term_storage = create_component(
        model=SHORT_TERM_STORAGE_COMPLEX,
        id="STS1",
    )

    network = Network("test")
    network.add_node(node)
    for component in [demand, short_term_storage, spillage, unsupplied]:
        network.add_component(component)
    network.connect(PortRef(demand, "balance_port"), PortRef(node, "balance_port"))
    network.connect(
        PortRef(short_term_storage, "balance_port"), PortRef(node, "balance_port")
    )
    network.connect(PortRef(spillage, "balance_port"), PortRef(node, "balance_port"))
    network.connect(PortRef(unsupplied, "balance_port"), PortRef(node, "balance_port"))

    problem = build_problem(
        network,
        database,
        time_blocks[0],
        scenarios,
        border_management=BlockBorderManagement.CYCLE,
    )
    status = problem.solver.Solve()

    assert status == problem.solver.OPTIMAL

    assert math.isclose(problem.solver.Objective().Value(), result)

    count_variables = 0
    for variable in problem.solver.variables():
        if "injection" in variable.name():
            count_variables += 1
            assert 0 <= variable.solution_value() <= 100
            print(variable.name())
            print(variable.solution_value())
        elif "withdrawal" in variable.name():
            count_variables += 1
            assert 0 <= variable.solution_value() <= 50
            print(variable.name())
            print(variable.solution_value())
        elif "level" in variable.name():
            count_variables += 1
            assert 0 <= variable.solution_value() <= 1000
            print(variable.name())
            print(variable.solution_value())

    assert count_variables == 3 * horizon

    database.add_data("STS1", "withdrawal_penality", ConstantData(0))
    database.add_data("STS1", "level_penality", ConstantData(5))
    database.add_data("STS1", "Pgrad+i_penality", ConstantData(0))
    database.add_data("STS1", "Pgrad-i_penality", ConstantData(0))
    database.add_data("STS1", "Pgrad+s_penality", ConstantData(0))
    database.add_data("STS1", "Pgrad-s_penality", ConstantData(0))

    status = problem.solver.Solve()

    assert status == problem.solver.OPTIMAL

    assert math.isclose(problem.solver.Objective().Value(), result)

    count_variables = 0
    for variable in problem.solver.variables():
        if "injection" in variable.name():
            count_variables += 1
            assert 0 <= variable.solution_value() <= 100
            print(variable.name())
            print(variable.solution_value())
        elif "withdrawal" in variable.name():
            count_variables += 1
            assert 0 <= variable.solution_value() <= 50
            print(variable.name())
            print(variable.solution_value())
        elif "level" in variable.name():
            count_variables += 1
            assert 0 <= variable.solution_value() <= 1000
            print(variable.name())
            print(variable.solution_value())

    assert count_variables == 3 * horizon

    database.add_data("STS1", "withdrawal_penality", ConstantData(0))
    database.add_data("STS1", "level_penality", ConstantData(0))
    database.add_data("STS1", "Pgrad+i_penality", ConstantData(5))
    database.add_data("STS1", "Pgrad-i_penality", ConstantData(0))
    database.add_data("STS1", "Pgrad+s_penality", ConstantData(0))
    database.add_data("STS1", "Pgrad-s_penality", ConstantData(0))

    status = problem.solver.Solve()

    assert status == problem.solver.OPTIMAL

    assert math.isclose(problem.solver.Objective().Value(), result)

    count_variables = 0
    for variable in problem.solver.variables():
        if "injection" in variable.name():
            count_variables += 1
            assert 0 <= variable.solution_value() <= 100
            print(variable.name())
            print(variable.solution_value())
        elif "withdrawal" in variable.name():
            count_variables += 1
            assert 0 <= variable.solution_value() <= 50
            print(variable.name())
            print(variable.solution_value())
        elif "level" in variable.name():
            count_variables += 1
            assert 0 <= variable.solution_value() <= 1000
            print(variable.name())
            print(variable.solution_value())

    assert count_variables == 3 * horizon


def test_short_test_horizon_10() -> None:
    short_term_storage_base(efficiency=0.8, horizon=10, result=72)


def test_short_test_horizon_5() -> None:
    short_term_storage_base(efficiency=0.2, horizon=5, result=18)
