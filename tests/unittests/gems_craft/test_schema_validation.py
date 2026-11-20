import io
from pathlib import Path

from gems_craft.model.parsing import parse_yaml_library
from gems_craft.optim_config.parsing import OptimConfig, load_optim_config
from gems_craft.study.parsing import parse_yaml_system


def test_library_schema_accepts_gemspy_extensions() -> None:
    library = parse_yaml_library(io.StringIO("""\
library:
  id: example
  dependencies: [base]
  models:
    - id: generator
      constraints:
        - id: bounded
          expression: power
          lower-bound: 0
          upper-bound: 100
"""))

    assert library.dependencies == ["base"]


def test_system_schema_validates_normalized_python_aliases() -> None:
    system = parse_yaml_system(io.StringIO("""\
system:
  model_libraries: example
  components:
    - id: G
      model: example.generator
      parameters:
        - id: demand
          time-dependent: true
          value: 42
"""))

    assert system.model_libraries == "example"
    assert system.components[0].parameters[0].value == 42


def test_empty_optim_config_file_loads_with_schema_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "optim-config.yml"
    config_path.write_text("# intentionally empty\n", encoding="utf-8")

    config = load_optim_config(config_path)

    assert isinstance(config, OptimConfig)
    assert config.models == []


def test_optim_config_schema_accepts_rolling_time_windows(tmp_path: Path) -> None:
    config_path = tmp_path / "optim-config.yml"
    config_path.write_text(
        """\
resolution:
  mode: sequential-subproblems
  block-length: 168
  block-overlap: 24
  carry-over-length: 12
""",
        encoding="utf-8",
    )

    config = load_optim_config(config_path)

    assert config is not None
    assert config.resolution.block_length == 168
    assert config.resolution.block_overlap == 24
    assert config.resolution.carry_over_length == 12
