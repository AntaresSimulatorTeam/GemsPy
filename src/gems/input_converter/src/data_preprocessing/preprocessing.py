from pathlib import Path
from typing import Optional, Union

import pandas as pd
from antares.craft.model.area import Area

# from antares.craft.model.area import BindingConstraint
from antares.craft.model.binding_constraint import BindingConstraint, ConstraintTerm
from antares.craft.model.study import Study
from antares.craft.model.thermal import ThermalCluster
from gems.input_converter.src.utils import check_dataframe_validity, check_file_exists
from gems.input_converter.src.data_preprocessing.dataclasses import (
    BindingConstraintData,
    LinkData,
    Operation,
    ThermalData,
    TimeseriesData
)

from gems.input_converter.src.data_preprocessing.ts_processor import (
    TimeSeriesProcessor
)
FIELD_ALIAS_MAP = {
    "nominalcapacity": "nominal_capacity",
    "min-stable-power": "min_stable_power",
    "min-up-time": "min_up_time",
    "min-down-time": "min_down_time",
}
type_to_data_class = {
    "binding_constraint": BindingConstraintData,
    "thermal": ThermalData,
    "link": LinkData,
    "timeseries": TimeseriesData,
}
DataType = Union[LinkData, ThermalData, BindingConstraintData, TimeseriesData]


class ModelsConfigurationPreprocessing:
    preprocessed_values: dict[str, float] = {}
    param_id: Optional[str] = None

    def __init__(self, study: Study):
        self.study = study
        self.study_path: Path = study.service.config.study_path  # type: ignore


    def calculate_value(self, obj: DataType) -> Union[float, str]:
        if isinstance(obj, ThermalData):
            if obj.timeseries_file_type is not None:
                return TimeSeriesProcessor.process_time_series(obj.area, obj, self.study_path, self.preprocessed_values, self.param_id, "thermal")
            area = self.study.get_areas()[obj.area]

            thermal: ThermalCluster = area.get_thermals()[obj.cluster]
            field_name: str = FIELD_ALIAS_MAP[obj.field]  # type: ignore
            
            parameter_value = getattr(thermal.properties, field_name)
            self.preprocessed_values[self.param_id] = parameter_value  # type: ignore
            return parameter_value
        elif isinstance(obj, BindingConstraintData):
            bindings: BindingConstraint = self.study.get_binding_constraints()[obj.id]
            term: ConstraintTerm = bindings.get_terms()[obj.field]
            if obj.operation:
                parameter_value: float = obj.operation.execute(term.weight)  # type: ignore
            else:
                parameter_value: float = term.weight  # type: ignore
            return parameter_value
        elif isinstance(obj, LinkData):
            return TimeSeriesProcessor.process_time_series(obj.area_from, obj, self.study_path, self.preprocessed_values, self.param_id, "link")
        elif isinstance(obj, TimeseriesData):
            return TimeSeriesProcessor.process_time_series(obj.area, obj, self.study_path, self.preprocessed_values, self.param_id, "standard")

        return ""


    def convert_param_value(self, id: str, parameter: dict) -> Union[str, float]:
        self.param_id = id
        value_type = parameter["type"]
        cls: DataType = type_to_data_class.get(value_type)
        if id == "generation":
            print("hello", id, parameter)

        if value_type == "constant":
            return float(parameter.get("data", ""))
        elif value_type == "path":
            return parameter.get("data", "")

        data: dict = parameter.get("data", {})
        if not cls:
            raise ValueError(f"Unknown value type: {value_type}")

        if "operation" in data:
            data["operation"] = Operation(**data["operation"])
        return self.calculate_value(cls(**data))


    def check_timeseries_validity(self, parameter: dict, processing_type: str = "standard") -> bool:
        if parameter.get("type") != "timeseries":
            return True
        cls: DataType = type_to_data_class.get("timeseries")
        obj = cls(**parameter.get("data", {}))
        _, file_path = TimeSeriesProcessor.get_pathfile_from_object(obj.area, obj, self.study_path, processing_type)
        return check_file_exists(file_path) and check_dataframe_validity(pd.read_csv(file_path, sep="\t", header=None))