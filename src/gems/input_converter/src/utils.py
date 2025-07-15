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
from pathlib import Path
from typing import Any

import yaml
from pandas import DataFrame
from pydantic import BaseModel


def resolve_path(path_str: Path) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError

    absolute_path = path.resolve()
    return absolute_path


def check_file_exists(input_path: Path) -> bool:
    if input_path.exists() and input_path.is_file() and input_path.stat().st_size > 0:
        return True
    return False


def check_dataframe_validity(df: DataFrame) -> bool:
    """
    Check and validate the following conditions:
    1. The dataframe from this path is not empty.
    2. The dataframe does not contains only zero values.

    :param df: dataframe to validate.
    """
    if df.empty or (df == 0).all().all():
        return False

    return True


def transform_to_yaml(model: BaseModel, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(
            {"system": model.model_dump(by_alias=True, exclude_unset=True)},
            yaml_file,
            allow_unicode=True,
        )


def read_yaml_file(file_path: Path) -> dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exists")
    with file_path.open("r", encoding="utf-8") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error trying to read yaml file {file_path}: {e}")
