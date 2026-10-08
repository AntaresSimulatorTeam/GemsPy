import shutil
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Optional, Tuple

from gems_craft.optim_config.parsing import OptimConfig, load_optim_config
from gems_craft.study.folder import load_study
from gems_craft.study.parsing import OutputFormat
from gems_runner.session.session import SimulationSession
from gems_runner.simulation.simulation_table_writer import SimulationTableWriter

# Run folders are written here, then moved to ``output/<run_id>/`` once the run
# has completed, so that ``output/<run_id>/`` never holds a partial run.
INCOMPLETE_DIR_NAME = "incomplete"


def run_study(
    study_dir: Path,
    optim_config_path: Optional[Path] = None,
    output_format: OutputFormat = OutputFormat.CSV,
) -> None:
    """
    Runs a simulation study and exports results to CSV or Parquet, one
    simulation table file per MC scenario.

    Run parameters (time scope, solver options, scenario scope) are read from
    ``study_dir/input/optim-config.yml``; defaults apply when the file is absent.
    Results are written to ``study_dir/output/{run_id}/``, which only appears
    once the run has completed: files are written to
    ``study_dir/output/incomplete/{run_id}/`` while the run is going, and that
    folder is moved to ``study_dir/output/{run_id}/`` at the end. If the run
    fails, the incomplete folder is removed; if the process is killed, it stays
    in ``output/incomplete/``. ``run_id`` is the start time to the minute, with
    a ``-2``, ``-3``, ... suffix if that run folder already exists.

    Args:
        study_dir: The path to the study directory.
        optim_config_path: Optional custom path to an optim-config YAML file.
            If not provided, defaults to ``study_dir/input/optim-config.yml``.
        output_format: Format of the simulation table files,
            ``OutputFormat.CSV`` (default) or ``OutputFormat.PARQUET``
            (zstd-compressed).
    """
    study = load_study(study_dir)

    resolved_config_path = optim_config_path or (
        study_dir / "input" / "optim-config.yml"
    )
    optim_config = load_optim_config(resolved_config_path) or OptimConfig()

    # Created before solving so that an invalid output format fails fast.
    writer = SimulationTableWriter(output_format)

    output_root = study_dir / "output"
    run_id, incomplete_dir = _reserve_run_folder(
        output_root, datetime.now().strftime("%Y%m%dT%H%M")
    )
    try:
        session = SimulationSession(
            study=study,
            optim_config=optim_config,
            run_id=run_id,
            output_dir=incomplete_dir,
        )
        # Results come one scenario at a time and are written immediately: in
        # sequential/parallel modes as each scenario is solved, in frontal mode
        # one after another after the single solve. No table holding all
        # scenarios is built. Benders mode writes no simulation table.
        for table in session.iter_scenario_tables():
            paths = writer.write(table, incomplete_dir)
            if len(paths) != 1:
                raise RuntimeError(
                    "Expected one simulation table file, got "
                    f"{[p.name for p in paths]}"
                )
    except BaseException:
        # BaseException: also on Ctrl+C. A killed process runs no cleanup; its
        # folder stays in output/incomplete/, never in output/<run_id>/.
        shutil.rmtree(incomplete_dir, ignore_errors=True)
        _remove_empty_dirs(incomplete_dir.parent, output_root)
        raise
    # The run folder is reserved, so output/<run_id>/ does not exist: the rename
    # moves the whole run at once.
    incomplete_dir.rename(output_root / run_id)
    _remove_empty_dirs(incomplete_dir.parent)


def _reserve_run_folder(output_root: Path, base_run_id: str) -> Tuple[str, Path]:
    """Reserve a run id whose folder exists neither in *output_root* nor in its
    incomplete folder, and create the incomplete folder.

    ``mkdir(exist_ok=False)`` is atomic, so two runs started at the same time
    never get the same folder.
    """
    for attempt in count(1):
        run_id = base_run_id if attempt == 1 else f"{base_run_id}-{attempt}"
        if (output_root / run_id).exists():
            continue
        incomplete_dir = output_root / INCOMPLETE_DIR_NAME / run_id
        try:
            incomplete_dir.mkdir(parents=True)
        except (FileExistsError, FileNotFoundError):
            # FileNotFoundError: another run removed the empty incomplete
            # folder while this one was creating it; try the next id.
            continue
        # A run with this id may have completed between the check and mkdir.
        if (output_root / run_id).exists():
            incomplete_dir.rmdir()
            continue
        return run_id, incomplete_dir
    raise AssertionError("unreachable")


def _remove_empty_dirs(*dirs: Path) -> None:
    """Remove each of *dirs* if it exists and is empty, in order."""
    for directory in dirs:
        try:
            directory.rmdir()
        except OSError:
            pass
