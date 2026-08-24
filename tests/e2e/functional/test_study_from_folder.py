import shutil
from pathlib import Path

import pandas as pd
import pytest

from gems_craft.study.folder import load_study
from gems_runner.study.runner import run_study

TAXONOMY_FILE = Path("input") / "taxonomy.yml"


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


def _study_declaring_taxonomy(tmp_path: Path) -> Path:
    """Copy the 7_4 study, making one of its libraries declare a taxonomy."""
    study_dir = tmp_path / "7_4"
    shutil.copytree(Path(__file__).parent / "studies" / "7_4", study_dir)

    lib_path = study_dir / "input" / "model-libraries" / "antares_historic.yml"
    lib_path.write_text(
        lib_path.read_text().replace(
            "  id: antares-historic-weo",
            "  id: antares-historic-weo\n  taxonomy: study_taxonomy",
            1,
        )
    )
    return study_dir


def test_load_study_checks_libraries_against_study_taxonomy(tmp_path: Path) -> None:
    study_dir = _study_declaring_taxonomy(tmp_path)
    (study_dir / TAXONOMY_FILE).write_text(
        "taxonomy:\n  id: study_taxonomy\n  categories:\n    - id: production\n"
    )

    study = load_study(study_dir)  # must not raise
    assert len(study.system.components) == 12


def test_load_study_raises_when_taxonomy_file_is_missing(tmp_path: Path) -> None:
    study_dir = _study_declaring_taxonomy(tmp_path)

    with pytest.raises(ValueError, match="no taxonomy was provided"):
        load_study(study_dir)


def test_load_study_raises_on_taxonomy_id_mismatch(tmp_path: Path) -> None:
    study_dir = _study_declaring_taxonomy(tmp_path)
    (study_dir / TAXONOMY_FILE).write_text("taxonomy:\n  id: other_taxonomy\n")

    with pytest.raises(ValueError, match="other_taxonomy"):
        load_study(study_dir)
