import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from gems_craft.optim_config.parsing import OptimConfig, load_optim_config
from gems_craft.study.folder import load_study
from gems_craft.study.parsing import OutputFormat
from gems_runner.session.session import SimulationSession


def _new_run_id() -> str:
    """Start time to the second plus a short random suffix: seconds alone do
    not separate runs started in quick succession."""
    return f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(3)}"


def _reserve_output_dir(output_root: Path) -> Tuple[str, Path]:
    """Create a new, empty run folder under *output_root* and return its run id
    and path.

    The folder is created without ``exist_ok``, so two runs can never share it.
    It is reserved before solving because Benders mode writes ``structure.txt``
    into it before the simulation table.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    while True:
        run_id = _new_run_id()
        output_dir = output_root / run_id
        try:
            output_dir.mkdir()
        except FileExistsError:
            continue
        return run_id, output_dir


def run_study(
    study_dir: Path,
    optim_config_path: Optional[Path] = None,
    output_format: OutputFormat = OutputFormat.CSV,
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
        output_format: Format of the simulation table file,
            ``OutputFormat.CSV`` (default) or ``OutputFormat.PARQUET``
            (zstd-compressed).
    """
    study = load_study(study_dir)

    resolved_config_path = optim_config_path or (
        study_dir / "input" / "optim-config.yml"
    )
    optim_config = load_optim_config(resolved_config_path) or OptimConfig()

    run_id, output_dir = _reserve_output_dir(study_dir / "output")
    session = SimulationSession(
        study=study,
        optim_config=optim_config,
        run_id=run_id,
        output_dir=output_dir,
    )
    table = session.run()
    if output_format == OutputFormat.PARQUET:
        table.to_parquet(output_dir)
    elif output_format == OutputFormat.CSV:
        table.to_csv(output_dir)
    else:
        raise ValueError(f"Unsupported output format: {output_format!r}")
