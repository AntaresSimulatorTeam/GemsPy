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

from gems_craft.optim_config.parsing import ScenarioScopeConfig

# ---------------------------------------------------------------------------
# Inline form — include only
# ---------------------------------------------------------------------------


def test_include_single_int() -> None:
    cfg = ScenarioScopeConfig(include=[0])
    assert cfg.scenario_ids == [0]


def test_include_multiple_ints() -> None:
    cfg = ScenarioScopeConfig(include=[0, 2, 4])
    assert cfg.scenario_ids == [0, 2, 4]


def test_include_range_string() -> None:
    cfg = ScenarioScopeConfig(include=["0-4"])
    assert cfg.scenario_ids == [0, 1, 2, 3, 4]


def test_include_mixed_ints_and_ranges() -> None:
    cfg = ScenarioScopeConfig(include=["0-2", 4, "7-9"])
    assert cfg.scenario_ids == [0, 1, 2, 4, 7, 8, 9]


def test_include_deduplicates_overlapping_entries() -> None:
    cfg = ScenarioScopeConfig(include=["0-4", 2, "3-5"])
    assert cfg.scenario_ids == [0, 1, 2, 3, 4, 5]


def test_include_output_is_sorted_ascending() -> None:
    cfg = ScenarioScopeConfig(include=[4, 1, 7, 0])
    assert cfg.scenario_ids == [0, 1, 4, 7]


# ---------------------------------------------------------------------------
# Inline form — include + exclude
# ---------------------------------------------------------------------------


def test_exclude_ints_from_include() -> None:
    cfg = ScenarioScopeConfig(include=["0-9"], exclude=[2, 6])
    assert cfg.scenario_ids == [0, 1, 3, 4, 5, 7, 8, 9]


def test_exclude_range_from_include() -> None:
    cfg = ScenarioScopeConfig(include=["0-9"], exclude=["3-5"])
    assert cfg.scenario_ids == [0, 1, 2, 6, 7, 8, 9]


def test_exclude_mixed_from_include() -> None:
    cfg = ScenarioScopeConfig(include=["0-9"], exclude=["1-3", 7])
    assert cfg.scenario_ids == [0, 4, 5, 6, 8, 9]


def test_exclude_all_leaves_empty_list() -> None:
    cfg = ScenarioScopeConfig(include=["0-2"], exclude=["0-2"])
    assert cfg.scenario_ids == []


def test_exclude_orphan_raises_warning() -> None:
    cfg = ScenarioScopeConfig(include=[0, 1], exclude=[4])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = cfg.scenario_ids
    assert result == [0, 1]
    assert len(caught) == 1
    assert "4" in str(caught[0].message)


def test_exclude_without_any_base_raises() -> None:
    with pytest.raises(
        ValueError, match="'exclude' requires 'include' or 'playlist-file'"
    ):
        ScenarioScopeConfig(exclude=[0])


# ---------------------------------------------------------------------------
# playlist-file + exclude
# ---------------------------------------------------------------------------


def test_playlist_file_with_exclude(tmp_path: Path) -> None:
    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps([0, 1, 2, 3, 4]))
    cfg = ScenarioScopeConfig(playlist_file=playlist, exclude=[2, 4])
    assert cfg.scenario_ids == [0, 1, 3]


def test_playlist_file_with_exclude_range(tmp_path: Path) -> None:
    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps([0, 1, 2, 3, 4, 5, 6]))
    cfg = ScenarioScopeConfig(playlist_file=playlist, exclude=["2-4"])
    assert cfg.scenario_ids == [0, 1, 5, 6]


def test_playlist_file_with_exclude_all_leaves_empty(tmp_path: Path) -> None:
    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps([0, 1, 2]))
    cfg = ScenarioScopeConfig(playlist_file=playlist, exclude=["0-2"])
    assert cfg.scenario_ids == []


def test_playlist_file_with_exclude_orphan_warns(tmp_path: Path) -> None:
    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps([0, 1, 2]))
    cfg = ScenarioScopeConfig(playlist_file=playlist, exclude=[5])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = cfg.scenario_ids
    assert result == [0, 1, 2]
    assert len(caught) == 1
    assert "5" in str(caught[0].message)


# ---------------------------------------------------------------------------
# Inline form — validation errors
# ---------------------------------------------------------------------------


def test_include_negative_index_raises() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        ScenarioScopeConfig(include=[-1]).scenario_ids


def test_include_zero_is_valid() -> None:
    cfg = ScenarioScopeConfig(include=[0])
    assert cfg.scenario_ids == [0]


def test_include_invalid_range_format_raises() -> None:
    with pytest.raises(ValueError, match="Invalid entry"):
        ScenarioScopeConfig(include=["abc"]).scenario_ids


def test_include_reversed_range_raises() -> None:
    with pytest.raises(ValueError, match="start must be <= end"):
        ScenarioScopeConfig(include=["5-3"]).scenario_ids


# ---------------------------------------------------------------------------
# Mutual exclusion
# ---------------------------------------------------------------------------


def test_include_and_playlist_file_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        ScenarioScopeConfig(include=[0], playlist_file=Path("some.json"))


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
    playlist.write_text(json.dumps([0, 2, 4]))
    cfg = ScenarioScopeConfig(playlist_file=playlist)
    assert cfg.scenario_ids == [0, 2, 4]


def test_playlist_file_deduplicates_and_sorts(tmp_path: Path) -> None:
    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps([4, 0, 2, 0, 4]))
    cfg = ScenarioScopeConfig(playlist_file=playlist)
    assert cfg.scenario_ids == [0, 2, 4]


def test_playlist_file_single_scenario(tmp_path: Path) -> None:
    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps([1]))
    cfg = ScenarioScopeConfig(playlist_file=playlist)
    assert cfg.scenario_ids == [1]


def test_playlist_file_zero_scenario(tmp_path: Path) -> None:
    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps([0]))
    cfg = ScenarioScopeConfig(playlist_file=playlist)
    assert cfg.scenario_ids == [0]


def test_playlist_file_large_list(tmp_path: Path) -> None:
    indices = list(range(1000))
    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps(indices))
    cfg = ScenarioScopeConfig(playlist_file=playlist)
    assert cfg.scenario_ids == list(range(1000))


def test_playlist_file_non_list_raises(tmp_path: Path) -> None:
    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps({"scenarios": [0, 1]}))
    cfg = ScenarioScopeConfig(playlist_file=playlist)
    with pytest.raises(ValueError, match="flat JSON array of integers"):
        cfg.scenario_ids


def test_playlist_file_contains_string_raises(tmp_path: Path) -> None:
    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps([0, "1", 2]))
    cfg = ScenarioScopeConfig(playlist_file=playlist)
    with pytest.raises(ValueError, match="flat JSON array of integers"):
        cfg.scenario_ids


def test_playlist_file_negative_index_raises(tmp_path: Path) -> None:
    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps([-1, 0, 1]))
    cfg = ScenarioScopeConfig(playlist_file=playlist)
    with pytest.raises(ValueError, match=">= 0"):
        cfg.scenario_ids


def test_playlist_file_not_found_raises(tmp_path: Path) -> None:
    cfg = ScenarioScopeConfig(playlist_file=tmp_path / "missing.json")
    with pytest.raises(ValueError, match="not found"):
        cfg.scenario_ids


def test_playlist_file_malformed_json_raises(tmp_path: Path) -> None:
    playlist = tmp_path / "playlist.json"
    playlist.write_text("not valid json {{")
    cfg = ScenarioScopeConfig(playlist_file=playlist)
    with pytest.raises(ValueError, match="Invalid JSON"):
        cfg.scenario_ids


def test_playlist_file_boolean_values_rejected(tmp_path: Path) -> None:
    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps([True, 1, 2]))
    cfg = ScenarioScopeConfig(playlist_file=playlist)
    with pytest.raises(ValueError, match="flat JSON array of integers"):
        cfg.scenario_ids


# ---------------------------------------------------------------------------
# YAML round-trip via OptimConfig
# ---------------------------------------------------------------------------


def test_yaml_inline_include_only() -> None:
    from gems_craft.optim_config.parsing import OptimConfig

    cfg = OptimConfig.model_validate({"scenario-scope": {"include": ["0-2", 4]}})
    assert cfg.scenario_scope.scenario_ids == [0, 1, 2, 4]


def test_yaml_inline_include_exclude() -> None:
    from gems_craft.optim_config.parsing import OptimConfig

    cfg = OptimConfig.model_validate(
        {"scenario-scope": {"include": ["0-4"], "exclude": [2]}}
    )
    assert cfg.scenario_scope.scenario_ids == [0, 1, 3, 4]


def test_yaml_playlist_file_relative_resolved_by_load_optim_config(
    tmp_path: Path,
) -> None:
    from gems_craft.optim_config.parsing import load_optim_config

    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps([0, 1, 2]))

    config_file = tmp_path / "optim-config.yml"
    config_file.write_text("scenario-scope:\n  playlist-file: playlist.json\n")

    cfg = load_optim_config(config_file)
    assert cfg is not None
    assert cfg.scenario_scope.scenario_ids == [0, 1, 2]


def test_yaml_nb_scenarios_rejected() -> None:
    from gems_craft.optim_config.parsing import OptimConfig

    with pytest.raises(ValueError):
        OptimConfig.model_validate({"scenario-scope": {"nb-scenarios": 1}})


# ---------------------------------------------------------------------------
# String integer entries
# ---------------------------------------------------------------------------


def test_include_string_integer_accepted() -> None:
    cfg = ScenarioScopeConfig(include=["4"])
    assert cfg.scenario_ids == [4]


def test_include_string_zero_accepted() -> None:
    cfg = ScenarioScopeConfig(include=["0"])
    assert cfg.scenario_ids == [0]


def test_include_string_integer_mixed_with_range() -> None:
    cfg = ScenarioScopeConfig(include=["0-2", "4", "7"])
    assert cfg.scenario_ids == [0, 1, 2, 4, 7]


def test_exclude_string_integer_accepted() -> None:
    cfg = ScenarioScopeConfig(include=["0-4"], exclude=["2"])
    assert cfg.scenario_ids == [0, 1, 3, 4]


# ---------------------------------------------------------------------------
# Boolean values rejected
# ---------------------------------------------------------------------------


def test_include_boolean_raises() -> None:
    with pytest.raises(ValueError, match="boolean"):
        ScenarioScopeConfig(include=[True]).scenario_ids


def test_exclude_boolean_raises() -> None:
    with pytest.raises(ValueError, match="boolean"):
        ScenarioScopeConfig(include=[0, 1], exclude=[False]).scenario_ids


# ---------------------------------------------------------------------------
# scenario_ids computed and stored only once
# ---------------------------------------------------------------------------


def test_scenario_ids_cached_inline() -> None:
    cfg = ScenarioScopeConfig(include=["0-4"])
    first = cfg.scenario_ids
    second = cfg.scenario_ids
    assert first is second


def test_scenario_ids_cached_playlist_file_via_load_optim_config(
    tmp_path: Path,
) -> None:
    from gems_craft.optim_config.parsing import load_optim_config

    playlist = tmp_path / "playlist.json"
    playlist.write_text(json.dumps([0, 1, 2]))
    config_file = tmp_path / "optim-config.yml"
    config_file.write_text("scenario-scope:\n  playlist-file: playlist.json\n")

    cfg = load_optim_config(config_file)
    assert cfg is not None
    playlist.unlink()  # delete the file — ids already cached at load time
    assert cfg.scenario_scope.scenario_ids == [0, 1, 2]
    assert cfg.scenario_scope.scenario_ids is cfg.scenario_scope.scenario_ids


# ---------------------------------------------------------------------------
# validate_optim_config — ScenarioBuilder cross-check
# ---------------------------------------------------------------------------


def test_validate_optim_config_scenario_builder_rejects_out_of_bounds() -> None:
    import numpy as np

    from gems_craft.optim_config.parsing import OptimConfig, validate_optim_config
    from gems_craft.study.scenario_builder import ScenarioBuilder
    from gems_runner.study.system import System

    config = OptimConfig.model_validate(
        {"scenario-scope": {"include": ["0-4"]}}  # scenarios 0,1,2,3,4
    )
    # ScenarioBuilder defines only 3 scenarios (0,1,2) for group "load"
    sb = ScenarioBuilder(_group_arrays={"load": np.array([0, 1, 0])})
    system = System(id="test")

    with pytest.raises(ValueError, match="not defined for scenario group"):
        validate_optim_config(config, system, sb)


def test_validate_optim_config_scenario_builder_accepts_valid_playlist() -> None:
    import numpy as np

    from gems_craft.optim_config.parsing import OptimConfig, validate_optim_config
    from gems_craft.study.scenario_builder import ScenarioBuilder
    from gems_runner.study.system import System

    config = OptimConfig.model_validate(
        {"scenario-scope": {"include": ["0-2"]}}  # scenarios 0,1,2
    )
    sb = ScenarioBuilder(_group_arrays={"load": np.array([0, 1, 0])})
    system = System(id="test")

    validate_optim_config(config, system, sb)  # must not raise
