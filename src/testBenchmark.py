# -*- coding: utf-8 -*-
import shutil
import tempfile
from pathlib import Path

from antares_runner.antares_runner import AntaresHybridStudyBenchmarker
from gems.input_converter.src.converter import AntaresStudyConverter
from gems.input_converter.src.logger import Logger

# === CONFIGURATION === #

# Chemin vers le dossier contenant les binaires Antares (PAS l'exécutable complet)
ANTARES_EXEC_FOLDER = Path(
    "/home/alzoobiali/Desktop/RTE/Antares/antares-8.8.2-Ubuntu-20.04/bin"
)
# /home/alzoobiali/Desktop/Redispatch/Studies/SoLight/BP23_A-Reference_2027
# /home/alzoobiali/Desktop/Redispatch/Studies/BPJM/BP23_A-Reference_2027
# Chemin vers l'étude legacy Antares
# SOURCE_STUDY_PATH = Path("/home/alzoobiali/Desktop/Redispatch/Studies/BPJM/BP23_A-Reference_2027")

SOURCE_STUDY_PATH = Path(
    "/home/alzoobiali/Desktop/modeler/andromede-modeling-prototype/tests/input_converter/resources/mini_test_batterie_BP23"
)

# Fichier scenario builder attendu dans l’étude d’origine
SCENARIO_BUILDER_NAME = "modeler-scenariobuilder.dat"
SCENARIO_BUILDER_FILE = SOURCE_STUDY_PATH / "settings" / SCENARIO_BUILDER_NAME

# Librairies modèles (à adapter si besoin)
LIB_PATH = [SOURCE_STUDY_PATH / "antares_legacy_models.yml"]

TEMPLATE_PATH = "/home/alzoobiali/Desktop/modeler/andromede-modeling-prototype/tests/input_converter/resources/mini_test_batterie_BP23/hybrid_template"
# TEMPLATE_PATH = Path("/home/alzoobiali/Desktop/Redispatch/Studies/SoLight/BP23_A-Reference_2027/hybrid_template")

# def addHybridBehavior(study_path, template_path)->None:
#     shutil.copytree(template_path / "input", study_path / "input", dirs_exist_ok=True)
#     shutil.copy2(template_path / "generaldata.ini", study_path / "settings/generaldata.ini")


def addHybridBehavior(study_path, template_path) -> None:
    study_path = Path(study_path)
    template_path = Path(template_path)

    shutil.copytree(template_path / "input", study_path / "input", dirs_exist_ok=True)
    shutil.copy2(
        template_path / "generaldata.ini", study_path / "settings/generaldata.ini"
    )


def test_converter_preserves_results_with_scenario_group():
    """
    Conversion + Benchmark : Vérifie que le modèle GEMS reconstruit donne
    exactement les mêmes résultats Antares que l'étude legacy.
    """

    # Étapes préliminaires
    assert SOURCE_STUDY_PATH.exists(), f"Study not found: {SOURCE_STUDY_PATH}"

    addHybridBehavior(SOURCE_STUDY_PATH, TEMPLATE_PATH)

    assert (
        ANTARES_EXEC_FOLDER.exists()
    ), f"Antares bin folder not found: {ANTARES_EXEC_FOLDER}"

    # Vérifie ou crée le fichier scenario builder
    if not SCENARIO_BUILDER_FILE.exists():
        SCENARIO_BUILDER_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SCENARIO_BUILDER_FILE, "w") as f:
            f.write("# Placeholder scenario builder\n")
        print(f" Created temporary scenario builder: {SCENARIO_BUILDER_FILE}")

    # === 1. Convert study in a temp directory === #
    with tempfile.TemporaryDirectory() as tmp_dir:
        converted_study_path = Path(tmp_dir) / "converted_study"

        print("\n Converting legacy study via GEMS...")
        converter = AntaresStudyConverter(
            study_input=SOURCE_STUDY_PATH,
            logger=Logger("converter", str(SOURCE_STUDY_PATH)),
            output_folder=converted_study_path,
            lib_paths=LIB_PATH,
            mode="hybrid",
            models_to_convert=["wind"],
            scenario_builder_file=SCENARIO_BUILDER_FILE,
        )

        converter.convert_study_to_input_system()

        # === 2. Benchmark: Compare Legacy vs Converted === #
        print("\n Running Antares Hybrid Benchmark...")
        benchmark = AntaresHybridStudyBenchmarker(
            exec_path=ANTARES_EXEC_FOLDER,  # Correction importante
            study_path_1=SOURCE_STUDY_PATH,  # Legacy original
            study_path_2=converted_study_path,  # Output GEMS converted
        )

        benchmark.run()
        diff = benchmark.weekly_gaps()

        print("\n Benchmark result:", diff)

        assert (
            diff["total_abs_rel_diff"] < 1e-6
        ), f" Converter modified results too much! diff={diff['total_abs_rel_diff']}"
        print(" Results preserved!")

        # === 3. Check scenario builder was copied === #
        expected_path = (
            converted_study_path / "input/data-series" / SCENARIO_BUILDER_NAME
        )
        assert (
            expected_path.exists()
        ), f" Scenario builder file not copied: {expected_path}"
        print(f" Scenario builder file located: {expected_path}")


if __name__ == "__main__":
    try:
        test_converter_preserves_results_with_scenario_group()
        print("\n🎉 SUCCESS: Converter test passed!")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
