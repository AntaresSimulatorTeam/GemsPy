import copy
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from antares.craft.model.binding_constraint import BindingConstraint, ConstraintTerm
from antares.craft.model.link import Link
from antares.craft.model.study import Study
from antares.craft.tools.time_series_tool import TimeSeriesFileType

from gems.input_converter.src.config import (
    MATRIX_TYPES_TO_GET_METHOD,
    TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD,
    TEMPLATE_LINK_TO_TIMESERIES_FILE_TYPE,
    TEMPLATE_TO_TIMESERIES_FILE_TYPE,
    TIMESERIES_NAME_TO_METHOD,
)
from gems.input_converter.src.data_preprocessing.dataclasses import (
    ComplexData,
    ConversionMode,
    MatrixData,
    ObjectProperties,
    Operation,
)
from gems.input_converter.src.utils import check_dataframe_validity, save_to_csv

DataType = Union[ComplexData, MatrixData]

TYPE_TO_DC = {
    "binding_constraint": ComplexData,
    "thermal": ComplexData,
    "link": ComplexData,
    "st_storage": ComplexData,
    "load": MatrixData,
    "solar": MatrixData,
    "wind": MatrixData,
}


class ModelsConfigurationProcessing:
    preprocessed_values: dict[str, float] = {}
    param_id: Optional[str] = None

    def __init__(self, study: Study, mode: ConversionMode):
        self.study = study
        self.mode = mode
        self.study_path: Path = study.service.config.study_path  # type: ignore

    def calculate_value(self, obj: DataType) -> Union[float, str]:
        area: str = obj.object_properties.area
        type: str = obj.object_properties.type

        if type in ["load", "wind", "solar"]:
            time_series: pd.DataFrame = getattr(
                self.study.get_areas()[area], MATRIX_TYPES_TO_GET_METHOD[type]
            )()
            if self.mode == ConversionMode.HYBRID.value:
                output_file = (
                    self.study.path
                    / "input"
                    / "data-series"
                    / f"{self.param_id}_{area}.txt"
                )
            else:
                output_file = (
                    self.study.path
                    / "input"
                    / type
                    / "series"
                    / f"{self.param_id}_{area}.txt"
                )

        elif type == "link":
            link: Link = self.study.get_links()[obj.object_properties.link]
            time_series: pd.DataFrame = getattr(
                link, TIMESERIES_NAME_TO_METHOD[obj.object_properties.field]
            )()
            if self.mode == ConversionMode.HYBRID.value:
                output_file = (
                    self.study.path
                    / "input"
                    / "data-series"
                    / f"{self.param_id}_{link.area_from_id}_{link.area_to_id}.txt"
                )
            else:
                file_path = getattr(
                    TimeSeriesFileType,
                    TEMPLATE_LINK_TO_TIMESERIES_FILE_TYPE[obj.object_properties.field],
                ).value.format(
                    area_id=link.area_from_id, second_area_id=link.area_to_id
                )
                output_file = self.study.path / file_path
        elif type in ["st_storage", "thermal", "renewable"]:
            # TODO Thermal preprocessing not handled for the moment in generic mode
            if area not in self.study.get_areas():
                raise KeyError(f"Area {area} is not found in the study")
            cluster = getattr(
                self.study.get_areas()[area], TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD[type]
            )()[obj.object_properties.cluster]
            if obj.object_properties.field in TIMESERIES_NAME_TO_METHOD:
                time_series: pd.DataFrame = getattr(
                    cluster, TIMESERIES_NAME_TO_METHOD[obj.object_properties.field]
                )()
            else:
                cluster_properties = getattr(cluster, "properties")
                field_name = obj.object_properties.field
                value = getattr(cluster_properties, field_name)
                if type == "thermal":
                    self.preprocessed_values[self.param_id] = value
                return value
            if self.mode == ConversionMode.HYBRID.value:
                output_file = (
                    self.study.path
                    / "input"
                    / "data-series"
                    / f"{self.param_id}_{area}_{obj.object_properties.cluster}.txt"
                )
            else:
                file_path = getattr(
                    TimeSeriesFileType,
                    TEMPLATE_TO_TIMESERIES_FILE_TYPE[obj.object_properties.field],
                ).value.format(area_id=cluster.area_id, cluster_id=cluster.id)

                output_file = self.study.path / file_path

        elif type == "binding_constraint":
            # TODO Add timeseries linked to binding constraints?
            bindings: BindingConstraint = self.study.get_binding_constraints()[
                obj.object_properties.binding_constraint_id
            ]
            term: ConstraintTerm = bindings.get_terms()[obj.object_properties.field]
            if obj.operation:
                return obj.operation.execute(term.weight)  # type: ignore
            else:
                return term.weight  # type: ignore

        if getattr(obj, "column", None) is not None:
            time_series = time_series.iloc[:, obj.column]
        if getattr(obj, "operation", None) is not None:
            parameter_value = obj.operation.execute(
                time_series, self.preprocessed_values
            )
            if isinstance(parameter_value, float):
                self.preprocessed_values[self.param_id] = parameter_value
                return parameter_value
            if isinstance(parameter_value, pd.Series):
                save_to_csv(parameter_value, output_file)
        else:
            # On réécrit par defaut les fichiers, même les wind/solar/load meme si ils ne sont pas modifiés
            save_to_csv(time_series, output_file)
        return str(output_file).removesuffix(".txt")

    def _process_value_content(self, value_content: dict) -> tuple[DataType, dict]:
        local_content = copy.deepcopy(value_content)
        value_type = local_content["object-properties"]["type"]

        cls: DataType = TYPE_TO_DC.get(value_type)

        if not cls:
            raise ValueError(f"Unknown value type: {value_type}")
        if local_content.get("object-properties"):
            local_content["object-properties"] = ObjectProperties.from_yaml(
                local_content["object-properties"]
            )
        else:
            raise KeyError(f"Object properties is not present: {local_content}")

        return cls, local_content

    def convert_param_value(self, id: str, value_content: dict) -> Union[str, float]:
        self.param_id = id
        if value_content.get("constant") is not None:
            return float(value_content.get("constant"))

        cls, local_content = self._process_value_content(value_content)

        if "operation" in local_content:
            local_content["operation"] = Operation(**local_content["operation"])
        return self.calculate_value(cls.from_yaml(local_content))

    def check_timeseries_validity(self, value_content: dict) -> bool:
        if value_content.get("constant"):
            return True
        cls, local_content = self._process_value_content(value_content)
        obj = cls.from_yaml(local_content)
        time_series: pd.DataFrame = getattr(
            self.study.get_areas()[obj.object_properties.area],
            MATRIX_TYPES_TO_GET_METHOD[obj.object_properties.type],
        )()
        return check_dataframe_validity(time_series)
