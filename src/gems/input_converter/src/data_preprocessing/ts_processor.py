from pathlib import Path
from typing import Union, Callable

import pandas as pd

from antares.craft.model.study import Study
from antares.craft.tools.matrix_tool import read_timeseries
from antares.craft.tools.time_series_tool import TimeSeriesFileType


class ProcessingConfig:
    def __init__(
        self,
        read_ts_keys: Callable[[object], dict[str, str]],
        support_operation: bool = True
    ):
        self.read_ts_keys = read_ts_keys
        self.support_operation = support_operation

class TimeSeriesProcessor:
    CONFIGS = {
        "link": ProcessingConfig(
            read_ts_keys=lambda obj: {"second_area_id": obj.area_to},
            support_operation=True
        ),
        "thermal": ProcessingConfig(
            read_ts_keys=lambda obj: {"cluster_id": obj.cluster},
            support_operation=True
        ),
        "standard": ProcessingConfig(
            read_ts_keys=lambda obj: {},
            support_operation=False
        )
    }

    @staticmethod
    def get_pathfile_from_object(area_id: str, obj: object, study_path: Path, processing_type: str) -> Union[str, Path]:
        config = TimeSeriesProcessor.CONFIGS[processing_type]

        ts_file_type = getattr(TimeSeriesFileType, obj.timeseries_file_type.upper())
        format_keys = {"area_id": area_id}
        format_keys.update(config.read_ts_keys(obj))
        return ts_file_type, study_path / ts_file_type.value.format(**format_keys)
        
    @staticmethod
    def process_time_series(
        area_id: str,
        obj: object,
        study_path: Path,
        preprocessed_values: dict[str, float],
        param_id: str,
        processing_type: str
    ) -> Union[float, str]:
        if processing_type not in TimeSeriesProcessor.CONFIGS:
            raise ValueError(f"Unsupported processing type: {processing_type}")
        
        config = TimeSeriesProcessor.CONFIGS[processing_type]
        # if processing_type == "link" and obj.area_from == "area_from_id":
        ####ici on doit avoir les bons first area id  et area_second id
        ts_file_type, input_path = TimeSeriesProcessor.get_pathfile_from_object(area_id, obj, study_path, processing_type)

        time_series = read_timeseries(
            ts_file_type,
            study_path,
            area_id,
            **config.read_ts_keys(obj)
        )
        
        if getattr(obj, "column", None) is not None:
            time_series = time_series.iloc[:, obj.column]
        
        # Ca fait sens pour le mode hybride
        output_file = input_path.parent / f"{param_id}_{area_id}.txt"
        if config.support_operation and getattr(obj, "operation", None):
            parameter_value = obj.operation.execute(time_series, preprocessed_values)
            if isinstance(parameter_value, float):
                preprocessed_values[param_id] = parameter_value
                return parameter_value
            if isinstance(parameter_value, pd.Series):
                parameter_value.to_csv(output_file, sep="\t", index=False, header=False)
        else:
            time_series.to_csv(output_file, sep="\t", index=False, header=False)

        return str(output_file.parent / f"{param_id}_{area_id}")