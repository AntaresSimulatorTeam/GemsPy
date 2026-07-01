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
from typing import Optional

import pandas as pd


def load_ts_from_file(
    timeseries_name: Optional[str], path_to_file: Optional[Path]
) -> pd.DataFrame:
    if path_to_file is None or timeseries_name is None:
        raise FileNotFoundError(f"File '{timeseries_name}' does not exist")

    base_path = path_to_file / timeseries_name
    candidates = [base_path.with_suffix(".txt"), base_path.with_suffix(".tsv")]

    last_exc: Optional[Exception] = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            sep = r"\s+" if candidate.suffix == ".txt" else "\t"
            return pd.read_csv(candidate, header=None, sep=sep)
        except Exception as e:
            last_exc = e
            break

    if last_exc is not None:
        raise Exception(
            f"An error has arrived when processing '{candidate}': {last_exc}"
        )

    raise FileNotFoundError(
        f"File '{timeseries_name}.txt' or '{timeseries_name}.tsv' does not exist"
    )
