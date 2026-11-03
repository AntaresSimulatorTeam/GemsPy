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

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# In a group array, a scenario without a line in the file.
_NO_LINE = -1


@dataclass
class ScenarioBuilder:
    """Maps MC scenario indices to data-series column indices (0-based).

    Loaded from a ``modeler-scenariobuilder.dat`` file whose lines have the form::

        scenario_group, mc_scenario = time_serie_number

    where ``time_serie_number`` is 1-based (column 1 = first column of the
    data-series file).  Internally the mapping is stored as one numpy array
    per group so that ``resolve_vectorized`` is a pure array index — no
    Python loop over scenarios.

    A scenario without a line in its group, a gap or after the group's last
    line, uses the default: MC scenario ``i`` → column ``i``, as do parameters
    whose ``scenario_group`` is ``None``. A group that is not in the file is an
    error. When no file is present the builder is empty.
    """

    # What the file says: the column of each scenario, _NO_LINE for a gap.
    _group_arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    # Per group, the column of scenarios 0..n-1, the default filled in; grown
    # on demand, so that resolve_vectorized() stays a single array index.
    _resolved: Dict[str, np.ndarray] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def resolve_vectorized(
        self, scenario_group: Optional[str], mc_scenarios: np.ndarray
    ) -> np.ndarray:
        """Return 0-based col_idx array for a vector of MC scenario indices.

        When ``scenario_group`` is ``None`` the parameter carries no group and
        the identity mapping (col_idx == mc_scenario) is returned, as for the
        scenarios without a line in their group.

        Raises ``ValueError`` when a non-None group is not present in the
        scenario builder — this is always a misconfiguration (e.g. a typo in
        ``system.yml``).

        No Python loop — pure numpy array indexing.
        """
        if scenario_group is None:
            return mc_scenarios
        resolved = self._resolved.get(scenario_group)
        if resolved is not None:
            try:
                return resolved[mc_scenarios]
            except IndexError:  # a scenario after the cached ones
                pass
        if scenario_group not in self._group_arrays:
            raise ValueError(
                f"Scenario group '{scenario_group}' is not defined in the scenario builder. "
                f"Known groups: {list(self._group_arrays.keys())}."
            )
        size = int(mc_scenarios.max()) + 1 if len(mc_scenarios) else 0
        return self._resolve_up_to(scenario_group, size)[mc_scenarios]

    def _resolve_up_to(self, group: str, size: int) -> np.ndarray:
        """The cached columns of *group*, for its first *size* scenarios: only
        those asked for, so that a line with a very large scenario (a typo)
        does not make the cache as large."""
        lines = self._group_arrays[group][:size]
        resolved = np.arange(size)
        has_line = lines != _NO_LINE
        resolved[: len(lines)][has_line] = lines[has_line]
        self._resolved[group] = resolved
        return resolved

    def is_from_file(self, scenario_group: str, mc_scenarios: np.ndarray) -> np.ndarray:
        """``True`` where the scenario has a line in the group, ``False`` where
        it uses the default. Raises ``KeyError`` for a group not in the file."""
        lines = self._group_arrays[scenario_group]
        from_file = np.zeros(len(mc_scenarios), dtype=bool)
        inside = mc_scenarios < len(lines)
        from_file[inside] = lines[mc_scenarios[inside]] != _NO_LINE
        return from_file

    def scenarios_without_line(self, scenario_ids: List[int]) -> Dict[str, List[int]]:
        """For each group, the scenarios of *scenario_ids* that have no line in
        it and use the default; groups where all have a line are left out."""
        mc = np.asarray(sorted(scenario_ids), dtype=int)
        without_line: Dict[str, List[int]] = {}
        for group in self._group_arrays:
            missing = mc[~self.is_from_file(group, mc)]
            if len(missing):
                without_line[group] = missing.tolist()
        return without_line

    def validate_mc_scenarios(self, scenario_ids: List[int]) -> List[str]:
        """Deprecated: a scenario without a line uses the default, so there is
        nothing to report; see ``scenarios_without_line()``."""
        warnings.warn(
            "ScenarioBuilder.validate_mc_scenarios() is deprecated and will be "
            "removed in a following release: scenarios without a line use the "
            "default. Use scenarios_without_line().",
            DeprecationWarning,
            stacklevel=2,
        )
        return []

    @classmethod
    def load(cls, path: Path) -> "ScenarioBuilder":
        """Parse a ``modeler-scenariobuilder.dat`` file.

        Each non-blank, non-comment line must follow::

            group_name, mc_scenario = time_serie_number

        ``time_serie_number`` is 1-based; it is stored internally as a
        0-based column index. Raises ``ValueError`` for a number below 1 or a
        negative MC scenario. Scenarios without a line are allowed.
        """
        raw: Dict[str, Dict[int, int]] = {}
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                left, _, right = line.partition("=")
                parts = [p.strip() for p in left.split(",")]
                group, mc_scenario = parts[0], int(parts[1])
                if mc_scenario < 0:
                    # It would overwrite the line of the last scenario.
                    raise ValueError(
                        f"Scenario group '{group}': MC scenarios start at 0, got "
                        f"{mc_scenario} (line '{line}' of {path})."
                    )
                time_serie_number = int(right.strip())
                if time_serie_number < 1:
                    # 0 would become column -1, which numpy reads as the last.
                    raise ValueError(
                        f"Scenario group '{group}', MC scenario {mc_scenario}: "
                        f"time series numbers start at 1, got {time_serie_number} "
                        f"(line '{line}' of {path})."
                    )
                raw.setdefault(group, {})[mc_scenario] = time_serie_number - 1

        group_arrays: Dict[str, np.ndarray] = {}
        for group, mapping in raw.items():
            arr = np.full(max(mapping) + 1, _NO_LINE, dtype=int)
            for mc, col in mapping.items():
                arr[mc] = col
            group_arrays[group] = arr

        return cls(group_arrays)
