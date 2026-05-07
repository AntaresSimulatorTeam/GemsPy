from pathlib import Path

import pytest
from pydantic import ValidationError

from gems.study.parsing import parse_yaml_components


def _parse_system_file(path: Path):
    with path.open(encoding="utf-8") as f:
        return parse_yaml_components(f)


def test_parse_yaml_components_from_file_parses_properties_list(tmp_path: Path) -> None:
    system_yml = """\
system:
  components:
    - id: nuclear_1
      model: basic.generator
      properties:
        - id: technology
          value: nuclear
        - id: company
          value: rhonepower
"""
    p = tmp_path / "system.yml"
    p.write_text(system_yml, encoding="utf-8")

    parsed = _parse_system_file(p)
    comp = parsed.components[0]
    assert comp.id == "nuclear_1"
    assert comp.properties is not None
    assert [x.model_dump() for x in comp.properties] == [
        {"id": "technology", "value": "nuclear"},
        {"id": "company", "value": "rhonepower"},
    ]


def test_parse_yaml_components_from_file_properties_optional(tmp_path: Path) -> None:
    system_yml = """\
system:
  components:
    - id: A
      model: basic.generator
"""
    p = tmp_path / "system.yml"
    p.write_text(system_yml, encoding="utf-8")

    parsed = _parse_system_file(p)
    assert parsed.components[0].properties is None


def test_parse_yaml_components_from_file_invalid_properties_raises(
    tmp_path: Path,
) -> None:
    system_yml = """\
system:
  components:
    - id: A
      model: basic.generator
      properties:
        - value: nuclear
"""
    p = tmp_path / "system.yml"
    p.write_text(system_yml, encoding="utf-8")

    with pytest.raises(ValidationError):
        _parse_system_file(p)
