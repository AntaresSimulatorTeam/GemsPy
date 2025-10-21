from pathlib import Path
from typing import Any, Union

import pandas as pd
from antares.craft.model.binding_constraint import BindingConstraint, ConstraintTerm
from antares.craft.model.link import Link
from antares.craft.model.study import Study

from gems.input_converter.src.config import (
    MATRIX_TYPES_TO_GET_METHOD,
    TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD,
    TIMESERIES_NAME_TO_METHOD,
)
from gems.input_converter.src.data_preprocessing.data_classes import ConversionMode
from gems.input_converter.src.parsing import ConversionValue
from gems.input_converter.src.utils import check_dataframe_validity, save_to_file

ALLOWED_TYPES: list = [
    "binding_constraint",
    "thermal",
    "link",
    "st_storage",
    "load",
    "solar",
    "wind",
]
SERIES_FOLDER = "data-series"


class ModelConversionPreprocessor:
    preprocessed_values: dict[str, float] = {}
    param_id: str

    def __init__(self, study: Study, mode: ConversionMode, output_folder: Path):
        self.study = study
        self.mode = mode
        self.output_folder = output_folder
        self.output_file = Path(".")
        self.file_path = Path(".")

    def calculate_matrix_data_values(
        self, obj: ConversionValue, type_resource: str
    ) -> pd.DataFrame:
        if not obj.object_properties or not obj.object_properties.area:
            raise ValueError(
                f"Object properties and its area from {obj} must not be None"
            )
        area: str = obj.object_properties.area
        self.file_path = Path(f"{self.param_id}_{area}.tsv")
        self.output_file = self.output_folder / "input" / SERIES_FOLDER / self.file_path
        return getattr(
            self.study.get_areas()[area], MATRIX_TYPES_TO_GET_METHOD[type_resource]
        )()

    def calculate_link_data_values(self, obj: ConversionValue) -> pd.DataFrame:
        if (
            not obj.object_properties
            or not obj.object_properties.link
            or not obj.object_properties.field
        ):
            raise ValueError(
                f"Object properties, its link, and field from {obj} must not be None"
            )
        link_id = obj.object_properties.link

        link: Link = self.study.get_links()[link_id]
        self.file_path = Path(
            f"{self.param_id}_{link.area_from_id}_{link.area_to_id}.tsv"
        )
        self.output_file = self.output_folder / "input" / SERIES_FOLDER / self.file_path
        return getattr(link, TIMESERIES_NAME_TO_METHOD[obj.object_properties.field])()

    def calculate_cluster_data_values(
        self, type_resource: str, obj: ConversionValue
    ) -> Union[Any, pd.DataFrame]:
        if (
            not obj.object_properties
            or not obj.object_properties.area
            or not obj.object_properties.field
        ):
            raise ValueError(
                f"Object properties, its area, and field from {obj} must not be None"
            )
        area: str = obj.object_properties.area
        if area not in self.study.get_areas():
            raise KeyError(f"Area {area} is not found in the study")
        cluster = getattr(
            self.study.get_areas()[area],
            TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD[type_resource],
        )()[obj.object_properties.cluster]
        if obj.object_properties.field in TIMESERIES_NAME_TO_METHOD:
            time_series = getattr(
                cluster, TIMESERIES_NAME_TO_METHOD[obj.object_properties.field]
            )()
        else:
            cluster_properties = getattr(cluster, "properties")
            field_name = obj.object_properties.field
            value = getattr(cluster_properties, field_name)
            if type_resource == "thermal":
                self.preprocessed_values[self.param_id] = value
            return value
        self.file_path = Path(
            f"{self.param_id}_{area}_{obj.object_properties.cluster}.tsv"
        )
        self.output_file = self.output_folder / "input" / SERIES_FOLDER / self.file_path
        return time_series

    def calculate_binding_constraint_data_values(
        self, obj: ConversionValue
    ) -> Union[float, pd.Series, pd.DataFrame]:
        if (
            not obj.object_properties
            or not obj.object_properties.field
            or not obj.object_properties.binding_constraint_id
        ):
            raise ValueError(
                f"Object properties, its field, and binding constraint ID from {obj} must not be None"
            )
        binding: BindingConstraint = self.study.get_binding_constraints()[
            obj.object_properties.binding_constraint_id
        ]
        term: ConstraintTerm = binding.get_terms()[obj.object_properties.field]
        if obj.operation:
            return obj.operation.execute(term.weight)
        else:
            return term.weight

    def calculate_value(self, obj: ConversionValue) -> Union[float, str]:
        if obj.object_properties is None or obj.object_properties.type is None:
            raise ValueError(f"Object properties {obj} must not be None")
        type_resource: str = obj.object_properties.type
        time_series: pd.DataFrame = pd.DataFrame()
        if type_resource in ["load", "wind", "solar"]:
            time_series = self.calculate_matrix_data_values(obj, type_resource)
            save_to_file(time_series, self.output_file)
            return str(self.file_path).removesuffix(".tsv")
        elif type_resource == "binding_constraint":
            # TODO No timeseries linked to binding constraints for the moment
            return self.calculate_binding_constraint_data_values(obj)  # type: ignore
        elif type_resource == "link":
            time_series = self.calculate_link_data_values(obj)
        elif type_resource in ["st_storage", "thermal", "renewable"]:
            data = self.calculate_cluster_data_values(type_resource, obj)
            if isinstance(data, pd.DataFrame):
                time_series = data
            else:
                return data

        if getattr(obj, "column", None) is not None:
            time_series: pd.Series = time_series.iloc[:, obj.column]  # type: ignore

            if getattr(obj, "operation") and obj.operation is not None:
                parameter_value: Union[
                    pd.Series, pd.DataFrame, float
                ] = obj.operation.execute(time_series, self.preprocessed_values)
                if isinstance(parameter_value, float):
                    self.preprocessed_values[self.param_id] = parameter_value
                    return parameter_value
                if isinstance(parameter_value, pd.Series):
                    save_to_file(parameter_value, self.output_file)
            else:
                save_to_file(time_series, self.output_file)
        else:
            save_to_file(time_series, self.output_file)

        return str(self.file_path).removesuffix(".tsv")

    def convert_param_value(
        self, id: str, value_content: ConversionValue
    ) -> Union[str, float]:
        self.param_id = id
        if value_content.constant is not None:
            return value_content.constant

        if not value_content.object_properties:
            raise ValueError(f"Object properties from {value_content} must not be None")
        if value_content.object_properties.type not in ALLOWED_TYPES:
            raise ValueError(
                f"Unknown value type: {value_content.object_properties.type}"
            )
        return self.calculate_value(value_content)

    def check_timeseries_validity(self, value_content: ConversionValue) -> bool:
        if value_content.constant is not None:
            return True

        if (
            not value_content.object_properties
            or not value_content.object_properties.area
            or not value_content.object_properties.type
        ):
            raise ValueError(
                f"Object properties, its area, and type from {value_content} must not be None"
            )
        # TODO : Do not duplicate logic with the one in convert_param_value
        if value_content.object_properties.type not in ALLOWED_TYPES:
            raise ValueError(
                f"Unknown value type: {value_content.object_properties.type}"
            )
        time_series: pd.DataFrame = getattr(
            self.study.get_areas()[value_content.object_properties.area],
            MATRIX_TYPES_TO_GET_METHOD[value_content.object_properties.type],
        )()
        return check_dataframe_validity(time_series)
