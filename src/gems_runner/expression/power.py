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

from typing import Any, overload

import numpy as np
import xarray as xr


@overload
def real_power(base: float, exponent: float) -> float: ...


@overload
def real_power(base: xr.DataArray, exponent: xr.DataArray) -> xr.DataArray: ...


def real_power(base: Any, exponent: Any) -> Any:
    """``base ^ exponent``, for scalars or arrays. Like ``/``, it never raises:
    a negative base with a fractional exponent gives ``nan``, zero with a
    negative exponent or an overflow gives ``inf``, cell by cell.
    """
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return np.float_power(base, exponent)
