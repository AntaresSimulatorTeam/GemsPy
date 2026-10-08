from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, Tuple

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
) -> Path:
    """
    Runs a simulation study and exports results to CSV or Parquet, one
    simulation table file per MC scenario.

    Run parameters (time scope, solver options, scenario scope) are read from
    ``study_dir/input/optim-config.yml``; defaults apply when the file is absent.
    Results are written to ``study_dir/output/{run_id}/``, which only appears
    once the run has completed: files are written to
    ``study_dir/output/incomplete/{run_id}/`` while the run is going, and that
    folder is renamed to ``study_dir/output/{run_id}/`` at the end. If the run
    fails or the process is killed, the folder stays in ``output/incomplete/``
    with the scenarios that finished (it is removed if nothing was written).
    ``run_id`` is the start time to the minute, with a ``-2``, ``-3``, ...
    suffix if that run folder already exists.

    Args:
        study_dir: The path to the study directory.
        optim_config_path: Optional custom path to an optim-config YAML file.
            If not provided, defaults to ``study_dir/input/optim-config.yml``.
        output_format: Format of the simulation table files,
            ``OutputFormat.CSV`` (default) or ``OutputFormat.PARQUET``
            (zstd-compressed).

    Returns:
        The run folder, ``study_dir/output/{run_id}/``.
    """
    study = load_study(study_dir)

    resolved_config_path = optim_config_path or (
        study_dir / "input" / "optim-config.yml"
    )
    optim_config = load_optim_config(resolved_config_path) or OptimConfig()

    # Created before solving so that an invalid output format fails fast.
    writer = SimulationTableWriter(output_format)

    output_root = study_dir / "output"
    start_minute = datetime.now().strftime("%Y%m%dT%H%M")
    with _run_folder(output_root, start_minute) as (run_id, folder):
        session = SimulationSession(
            study=study,
            optim_config=optim_config,
            run_id=run_id,
            output_dir=folder,
        )
        # Results come one scenario at a time and are written immediately: in
        # sequential/parallel modes as each scenario is solved, in frontal mode
        # one after another after the single solve. No table holding all
        # scenarios is built. Benders mode writes no simulation table.
        for table in session.iter_scenario_tables():
            paths = writer.write(table, folder)
            if len(paths) != 1:
                raise RuntimeError(
                    "Expected one simulation table file, got "
                    f"{[p.name for p in paths]}"
                )
    return output_root / run_id


@contextmanager
def _run_folder(output_root: Path, base_run_id: str) -> Iterator[Tuple[str, Path]]:
    """Reserve a run id and its folder in ``output/incomplete/``, and yield
    them. On success, the folder is renamed to ``output/<run_id>/``; the
    reservation guarantees that it does not exist, so the whole run appears at
    once.

    On failure, the folder stays in ``output/incomplete/`` so that the scenarios
    that finished can be inspected, and the error says where it is. A folder
    with nothing in it (e.g. a configuration rejected before solving) is
    removed. A killed process runs no code: its folder stays too.
    """
    output_existed = output_root.exists()
    run_id, incomplete_dir = _reserve_run_folder(output_root, base_run_id)
    try:
        yield run_id, incomplete_dir
    except BaseException as error:  # also on Ctrl+C
        if any(incomplete_dir.iterdir()):
            error.add_note(f"Partial results of this run are in {incomplete_dir}")
        else:
            incomplete_dir.rmdir()
            _remove_empty_dir(incomplete_dir.parent)
            if not output_existed:
                _remove_empty_dir(output_root)
        raise
    incomplete_dir.rename(output_root / run_id)
    _remove_empty_dir(incomplete_dir.parent)


def _reserve_run_folder(output_root: Path, base_run_id: str) -> Tuple[str, Path]:
    """Reserve a run id whose folder exists neither in *output_root* nor in its
    incomplete folder, and create the incomplete folder.

    The first free id among ``base_run_id``, ``base_run_id-2``, ... is taken.
    ``mkdir(exist_ok=False)`` is atomic, so two runs started at the same time
    never get the same folder.
    """
    attempt = 1
    while True:
        run_id = base_run_id if attempt == 1 else f"{base_run_id}-{attempt}"
        attempt += 1
        incomplete_dir = output_root / INCOMPLETE_DIR_NAME / run_id
        try:
            incomplete_dir.mkdir(parents=True)
        except (FileExistsError, FileNotFoundError):
            # FileExistsError: a running or killed run has this id.
            # FileNotFoundError: another run removed the empty incomplete
            # folder while this one was creating it.
            continue
        if (output_root / run_id).exists():
            # A completed run has this id.
            incomplete_dir.rmdir()
            continue
        return run_id, incomplete_dir


def _remove_empty_dir(directory: Path) -> None:
    """Remove *directory* if it exists and is empty."""
    try:
        directory.rmdir()
    except OSError:
        pass
