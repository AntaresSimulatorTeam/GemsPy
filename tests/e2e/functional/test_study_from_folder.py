import shutil
from pathlib import Path

import pandas as pd

from gems.study.folder import load_study
from gems.study.runner import run_study


def test_load_study():
    study_dir = Path(__file__).parent / "studies" / "7_4"

    study = load_study(study_dir)
    assert len(study.system.components) == 12
    assert len(study.system.connections) == 11
    assert len(study.database._data) == 76


def test_run_study(tmp_path: Path) -> None:
    # Copy study to tmp_path so output files don't pollute the source tree.
    study_dir = tmp_path / "7_4"
    shutil.copytree(Path(__file__).parent / "studies" / "7_4", study_dir)

    run_study(study_dir)

    output_files = list((study_dir / "output").glob("**/simulation_table_*.csv"))
    assert len(output_files) == 1
    df = pd.read_csv(output_files[0])
    assert "objective-value" in df["output"].values
