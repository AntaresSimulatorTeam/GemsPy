from dataclasses import dataclass


@dataclass(frozen=True)
class ProblemDimensions:
    """
    Dimensions for the simulation window
    """

    timesteps_count: int
    scenarios_count: int