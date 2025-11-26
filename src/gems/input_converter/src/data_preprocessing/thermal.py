import os
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from antares.craft.model.thermal import ThermalCluster

from gems.study.parsing import InputComponentParameter


class Direction(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


class ThermalDataPreprocessing:
    DEFAULT_PERIOD: int = 168

    def __init__(self, thermal: ThermalCluster, study_path: Path, suffix: str = ".tsv"):
        self.thermal = thermal
        self.study_path = study_path
        self.suffix = suffix
        self.output_series_dir = self.study_path / "input" / "data-series"
        self._prepro_parameter_functions: dict[str, Callable[[int], pd.DataFrame]] = {
            "p_min_cluster": lambda _: self._compute_p_min_cluster(),
            "p_max_cluster": lambda _: self._compute_p_max_cluster(),
            "nb_units_min": lambda _: self._compute_nb_units_min(),
            "nb_units_max": lambda _: self._compute_nb_units_max(),
            "nb_units_max_variation_forward": lambda period: self._compute_nb_units_max_variation(
                Direction.FORWARD, period
            ),
            "nb_units_max_variation_backward": lambda period: self._compute_nb_units_max_variation(
                Direction.BACKWARD, period
            ),
        }

    def _compute_p_min_cluster(self) -> pd.DataFrame:
        modulation_data: pd.Series = self.thermal.get_prepro_modulation_matrix().iloc[
            :, 3
        ]
        series_data: pd.DataFrame = self.thermal.get_series_matrix()
        unit_count: int = self.thermal.properties.unit_count
        nominal_capacity: float = self.thermal.properties.nominal_capacity
        scaled_modulation: pd.Series = modulation_data * nominal_capacity * unit_count
        #  min(min_gen_modulation * unit_count * nominal_capacity, p_max_cluster)
        min_values: pd.Series = pd.concat([scaled_modulation, series_data], axis=1).min(
            axis=1
        )
        return min_values.to_frame(name="p_min_cluster")

    def _compute_p_max_cluster(self) -> pd.DataFrame:
        return self.thermal.get_series_matrix()

    def _compute_nb_units_min(self) -> pd.DataFrame:
        p_min_cluster: pd.DataFrame = self._compute_p_min_cluster()
        nominal_capacity: float = self.thermal.properties.nominal_capacity
        return pd.DataFrame(
            np.ceil(p_min_cluster / nominal_capacity),
        )

    def _compute_nb_units_max(self) -> pd.DataFrame:
        series_data: pd.DataFrame = self.thermal.get_series_matrix()
        nominal_capacity: float = self.thermal.properties.nominal_capacity
        return pd.DataFrame(
            np.ceil(series_data / nominal_capacity),
        )

    def _compute_nb_units_max_variation(
        self, direction: Direction, period: int = DEFAULT_PERIOD
    ) -> pd.DataFrame:
        nb_units_max = self._compute_nb_units_max()
        indices = np.arange(len(nb_units_max))
        max_valid_index = len(nb_units_max) - 1  # 8759

        previous_indices: np.ndarray = (indices - 1) % period + (
            indices // period
        ) * period

        previous_indices = np.where(
            indices == 0, 0, np.minimum(previous_indices, max_valid_index)
        )
        variation = pd.DataFrame()
        if direction == Direction.BACKWARD:
            variation = nb_units_max.reset_index(drop=True) - nb_units_max.iloc[
                previous_indices
            ].reset_index(drop=True)
        elif direction == Direction.FORWARD:
            variation = nb_units_max.iloc[previous_indices].reset_index(
                drop=True
            ) - nb_units_max.reset_index(drop=True)

        # Usage of vectorized operation instead of applymap
        # It is the equivalent of max(0, variation(x))
        variation = variation.clip(lower=0)
        return variation.rename(
            columns={variation.columns[0]: f"nb_units_max_variation_{direction.value}"}
        )

    def _build_csv_path_and_name(self, param_id: str) -> tuple[Path, str]:
        name = f"{self.thermal.area_id}_{self.thermal.id}_{param_id}"
        return self.output_series_dir / str(name + self.suffix), name

    def generate_component_parameter(
        self, parameter_id: str, period: int = 0
    ) -> InputComponentParameter:
        if parameter_id not in self._prepro_parameter_functions:
            raise ValueError(f"Unsupported parameter_id: {parameter_id}")

        df = self._prepro_parameter_functions[parameter_id](period)
        csv_path, value_name = self._build_csv_path_and_name(parameter_id)

        output_dir = os.path.dirname(csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        # This separator is chosen to comply with the antares_craft timeseries creation
        df.to_csv(csv_path, sep="\t", index=False, header=False)

        return InputComponentParameter(
            id=parameter_id,
            time_dependent=True,
            scenario_dependent=True,
            value=value_name,
        )
