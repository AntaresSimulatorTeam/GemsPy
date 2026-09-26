from datetime import datetime
from pathlib import Path
from typing import Optional

from gems_craft.optim_config.parsing import OptimConfig, load_optim_config
from gems_craft.study.folder import load_study
from gems_runner.session.session import SimulationSession
from gems_runner.simulation.simulation_table import SimulationTable
from gems_runner.simulation.simulation_table_writer import (
    OutputFormat,
    SimulationTableWriter,
)


def run_study(
    study_dir: Path,
    optim_config_path: Optional[Path] = None,
    output_format: OutputFormat = "csv",
) -> None:
    """
    Runs a simulation study and exports results to CSV or Parquet, one
    simulation table file per MC scenario.

    Run parameters (time scope, solver options, scenario scope) are read from
    ``study_dir/input/optim-config.yml``; defaults apply when the file is absent.
    Results are written to ``study_dir/output/{run_id}/``.

    Args:
        study_dir: The path to the study directory.
        optim_config_path: Optional custom path to an optim-config YAML file.
            If not provided, defaults to ``study_dir/input/optim-config.yml``.
        output_format: Format of the simulation table files, ``"csv"``
            (default) or ``"parquet"`` (zstd-compressed). One file is written per
            MC scenario, plus a ``scenario-common`` file when some rows are shared
            by all scenarios (frontal mode).
    """
    study = load_study(study_dir)

    resolved_config_path = optim_config_path or (
        study_dir / "input" / "optim-config.yml"
    )
    optim_config = load_optim_config(resolved_config_path) or OptimConfig()

    run_id = datetime.now().strftime("%Y%m%dT%H%M")
    output_dir = study_dir / "output" / run_id
    # Created before solving so that an invalid output format fails fast.
    writer = SimulationTableWriter(output_format)

    def write_scenario(scenario_id: Optional[int], table: SimulationTable) -> None:
        writer.write_scenario(table, output_dir, scenario_id)

    # Results are handed over one scenario at a time and written immediately:
    # in sequential/parallel modes as each scenario is solved, in frontal mode
    # concurrently after the single solve. No table holding all scenarios is
    # built. Benders mode writes no simulation table.
    session = SimulationSession(
        study=study,
        optim_config=optim_config,
        run_id=run_id,
        output_dir=output_dir,
        on_scenario_done=write_scenario,
    )
    session.run()
