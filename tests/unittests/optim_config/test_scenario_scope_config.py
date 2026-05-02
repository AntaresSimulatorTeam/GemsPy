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

import json
import warnings
from pathlib import Path

import pytest

from gems.optim_config.parsing import ScenarioScopeConfig

# ---------------------------------------------------------------------------
# Inline form — include only
# ---------------------------------------------------------------------------


def test_include_single_int() -> None:
    cfg = ScenarioScopeConfig(include=[1])
    assert cfg.scenario_ids == [0]


def test_include_multiple_ints() -> None:
    cfg = ScenarioScopeConfig(include=[1, 3, 5])
    assert cfg.scenario_ids == [0, 2, 4]


def test_include_range_string() -> None:
    cfg = ScenarioScopeConfig(include=["1-5"])
    assert cfg.scenario_ids == [0, 1, 2, 3, 4]


def test_include_mixed_ints_and_ranges() -> None:
    cfg = ScenarioScopeConfig(include=["1-3", 5, "8-10"])
    assert cfg.scenario_ids == [0, 1, 2, 4, 7, 8, 9]


def test_include_deduplicates_overlapping_entries() -> None:
    cfg = ScenarioScopeConfig(include=["1-5", 3, "4-6"])
    assert cfg.scenario_ids == [0, 1, 2, 3, 4, 5]


def test_include_output_is_sorted_ascending() -> None:
    cfg = ScenarioScopeConfig(include=[5, 2, 8, 1])
    assert cfg.scenario_ids == [0, 1, 4, 7]


# ---------------------------------------------------------------------------
# Inline form — include + exclude
# ---------------------------------------------------------------------------


def test_exclude_ints_from_include() -> None:
    cfg = ScenarioScopeConfig(include=["1-10"], exclude=[3, 7])
    assert cfg.scenario_ids == [0, 1, 3, 4, 5, 7, 8, 9]


def test_exclude_range_from_include() -> None:
    cfg = ScenarioScopeConfig(include=["1-10"], exclude=["4-6"])
    assert cfg.scenario_ids == [0, 1, 2, 6, 7, 8, 9]


def test_exclude_mixed_from_include() -> None:
    cfg = ScenarioScopeConfig(include=["1-10"], exclude=["2-4", 8])
    assert cfg.scenario_ids == [0, 4, 5, 6, 8, 9]


def test_exclude_all_leaves_empty_list() -> None:
    cfg = ScenarioScopeConfig(include=["1-3"], exclude=["1-3"])
    assert cfg.scenario_ids == []


def test_exclude_orphan_raises_warning() -> None:
    cfg = ScenarioScopeConfig(include=[1, 2], exclude=[5])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = cfg.scenario_ids
    assert result == [0, 1]
    assert len(caught) == 1
    assert "5" in str(caught[0].message)


def test_exclude_without_include_raises() -> None:
    with pytest.raises(ValueError, match="'exclude' requires 'include'"):
        ScenarioScopeConfig(exclude=[1])


# ---------------------------------------------------------------------------
# Inline form — validation errors
# ---------------------------------------------------------------------------


def test_include_zero_index_raises() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        ScenarioScopeConfig(include=[0]).scenario_ids


def test_include_negative_index_raises() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        ScenarioScopeConfig(include=[-1]).scenario_ids


def test_include_invalid_range_format_raises() -> None:
    with pytest.raises(ValueError, match="Invalid range"):
        ScenarioScopeConfig(include=["abc"]).scenario_ids


def test_include_reversed_range_raises() -> None:
    with pytest.raises(ValueError, match="start must be <= end"):
        ScenarioScopeConfig(include=["5-3"]).scenario_ids


# ---------------------------------------------------------------------------
# Mutual exclusion
# ---------------------------------------------------------------------------


def test_include_and_playlist_file_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        ScenarioScopeConfig(include=[1], playlist_file=Path("some.json"))


# ---------------------------------------------------------------------------
# Default behaviour (no include, no playlist_file)
# ---------------------------------------------------------------------------


def test_default_returns_single_scenario_zero() -> None:
    cfg = ScenarioScopeConfig()
    assert cfg.scenario_ids == [0]


# ---------------------------------------------------------------------------
# Playlist file form
# ---------------------------------------------------------------------------


def test_playlist_file_flat_integers(tmp_path: Path) -> None:
    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps([1, 3, 5]))
    cfg = ScenarioScopeConfig(playlist_file=playlist)
    assert cfg.scenario_ids == [0, 2, 4]


def test_playlist_file_deduplicates_and_sorts(tmp_path: Path) -> None:
    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps([5, 1, 3, 1, 5]))
    cfg = ScenarioScopeConfig(playlist_file=playlist)
    assert cfg.scenario_ids == [0, 2, 4]


def test_playlist_file_single_scenario(tmp_path: Path) -> None:
    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps([2]))
    cfg = ScenarioScopeConfig(playlist_file=playlist)
    assert cfg.scenario_ids == [1]


def test_playlist_file_large_list(tmp_path: Path) -> None:
    indices = list(range(1, 1001))
    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps(indices))
    cfg = ScenarioScopeConfig(playlist_file=playlist)
    assert cfg.scenario_ids == list(range(1000))


def test_playlist_file_non_list_raises(tmp_path: Path) -> None:
    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps({"scenarios": [1, 2]}))
    cfg = ScenarioScopeConfig(playlist_file=playlist)
    with pytest.raises(ValueError, match="flat JSON array of integers"):
        cfg.scenario_ids


def test_playlist_file_contains_string_raises(tmp_path: Path) -> None:
    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps([1, "2", 3]))
    cfg = ScenarioScopeConfig(playlist_file=playlist)
    with pytest.raises(ValueError, match="flat JSON array of integers"):
        cfg.scenario_ids


def test_playlist_file_zero_index_raises(tmp_path: Path) -> None:
    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps([0, 1, 2]))
    cfg = ScenarioScopeConfig(playlist_file=playlist)
    with pytest.raises(ValueError, match=">= 1"):
        cfg.scenario_ids


# ---------------------------------------------------------------------------
# YAML round-trip via OptimConfig
# ---------------------------------------------------------------------------


def test_yaml_inline_include_only() -> None:
    from gems.optim_config.parsing import OptimConfig

    cfg = OptimConfig.model_validate({"scenario-scope": {"include": ["1-3", 5]}})
    assert cfg.scenario_scope.scenario_ids == [0, 1, 2, 4]


def test_yaml_inline_include_exclude() -> None:
    from gems.optim_config.parsing import OptimConfig

    cfg = OptimConfig.model_validate(
        {"scenario-scope": {"include": ["1-5"], "exclude": [3]}}
    )
    assert cfg.scenario_scope.scenario_ids == [0, 1, 3, 4]


def test_yaml_playlist_file_relative_resolved_by_load_optim_config(
    tmp_path: Path,
) -> None:
    from gems.optim_config.parsing import load_optim_config

    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps([1, 2, 3]))

    config_file = tmp_path / "optim-config.yml"
    config_file.write_text("scenario-scope:\n  playlist-file: playlist.json\n")

    cfg = load_optim_config(config_file)
    assert cfg is not None
    assert cfg.scenario_scope.scenario_ids == [0, 1, 2]


def test_yaml_nb_scenarios_rejected() -> None:
    from gems.optim_config.parsing import OptimConfig

    with pytest.raises(ValueError):
        OptimConfig.model_validate({"scenario-scope": {"nb-scenarios": 1}})
