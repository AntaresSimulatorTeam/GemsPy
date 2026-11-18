# Copyright (c) 2026, RTE (https://www.rte-france.com)
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

"""Rebuilds the problem of one scenario from a parametric export and solves it,
as a consumer of the export would."""

from pathlib import Path

import highspy
import numpy as np
import pandas as pd


def solve_scenario(mps: Path, parameters: pd.DataFrame, scenario: int) -> float:
    """Read *mps*, apply the entries of *scenario*, solve and return the
    objective value, constant included."""
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(str(mps))
    lp = h.getLp()
    rows = {name: i for i, name in enumerate(lp.row_names_)}
    cols = {name: i for i, name in enumerate(lp.col_names_)}
    constant = 0.0
    for entry in parameters[parameters["scenario"] == scenario].itertuples():
        if entry.kind == "coef":
            h.changeCoeff(rows[entry.row], cols[entry.col], entry.value)
        elif entry.kind == "rhs":
            i = rows[entry.row]
            lower, upper = lp.row_lower_[i], lp.row_upper_[i]
            h.changeRowBounds(
                i,
                entry.value if np.isfinite(lower) else lower,
                entry.value if np.isfinite(upper) else upper,
            )
        elif entry.kind == "lower":
            i = cols[entry.col]
            h.changeColBounds(i, entry.value, h.getLp().col_upper_[i])
        elif entry.kind == "upper":
            i = cols[entry.col]
            h.changeColBounds(i, h.getLp().col_lower_[i], entry.value)
        elif entry.kind == "obj":
            h.changeColCost(cols[entry.col], entry.value)
        elif entry.kind == "obj_const":
            constant = entry.value
        else:
            raise ValueError(f"Unknown kind {entry.kind!r}")
    h.run()
    assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
    return float(h.getInfo().objective_function_value) + constant
