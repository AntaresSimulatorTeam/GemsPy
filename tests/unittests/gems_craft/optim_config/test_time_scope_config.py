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

import pytest

from gems_craft.optim_config.parsing import TimeScopeConfig, load_optim_config


def test_equal_bounds_accepted() -> None:
    cfg = TimeScopeConfig.model_validate({"first-time-step": 3, "last-time-step": 3})
    assert (cfg.first_time_step, cfg.last_time_step) == (3, 3)


def test_first_time_step_greater_than_last_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "optim-config.yml"
    config_path.write_text("time-scope:\n  first-time-step: 2\n  last-time-step: 1\n")

    with pytest.raises(ValueError, match="Invalid optim-config") as raised:
        load_optim_config(config_path)
    assert "'first-time-step' (2) must be <= 'last-time-step' (1)" in str(raised.value)
