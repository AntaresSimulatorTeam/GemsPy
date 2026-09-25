from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from gems_craft.optim_config.parsing import OptimConfig, load_optim_config
from gems_craft.study.folder import load_study
from gems_runner.session.session import SimulationSession

OutputFormat = Literal["csv", "parquet"]


def run_study(
    study_dir: Path,
    optim_config_path: Optional[Path] = None,
    output_format: OutputFormat = "csv",
) -> None:
    """
    Runs a simulation study and exports results to CSV or Parquet.

    Run parameters (time scope, solver options, scenario scope) are read from
    ``study_dir/input/optim-config.yml``; defaults apply when the file is absent.
    Results are written to ``study_dir/output/{run_id}/``.

    Args:
        study_dir: The path to the study directory.
        optim_config_path: Optional custom path to an optim-config YAML file.
            If not provided, defaults to ``study_dir/input/optim-config.yml``.
        output_format: Format of the simulation table file, ``"csv"`` (default)
            or ``"parquet"`` (zstd-compressed).
    """
    study = load_study(study_dir)

    resolved_config_path = optim_config_path or (
        study_dir / "input" / "optim-config.yml"
    )
    optim_config = load_optim_config(resolved_config_path) or OptimConfig()

    run_id = datetime.now().strftime("%Y%m%dT%H%M")
    output_dir = study_dir / "output" / run_id
    session = SimulationSession(
        study=study,
        optim_config=optim_config,
        run_id=run_id,
        output_dir=output_dir,
    )
    table = session.run()
    if output_format == "parquet":
        table.to_parquet(output_dir)
    elif output_format == "csv":
        table.to_csv(output_dir)
    else:
        raise ValueError(f"Unsupported output format: {output_format!r}")
