# -*- coding: utf-8 -*-
import shutil
import tempfile
from pathlib import Path

from gems import study
from gems.input_converter.src.converter import AntaresStudyConverter
from gems.input_converter.src.logger import Logger

# # Path to study and scenario builder
# study_path = Path(
#     "/home/alzoobiali/Desktop/modeler/andromede-modeling-prototype/tests/input_converter/resources/mini_test_batterie_BP23"
# )
# study_path = Path("/home/alzoobiali/Desktop/Redispatch/Studies/SoLight/BP23_A-Reference_2027")
study_path = Path("tests/input_converter/resources/mini_test_BP_conversion")

# study_path = create_sanitized_study(study_path=study_path)
scenario_file = study_path / "input" / "modeler-scenariobuilder.dat"
libpath = study_path / "antares_legacy_models.yml"
# # Ensure the parent directory exists
scenario_file.parent.mkdir(parents=True, exist_ok=True)


output_folder = study_path / "converted_test"
output_folder.mkdir(exist_ok=True, parents=True)

# 1) CONVERT

converter = AntaresStudyConverter(
    study_input=study_path,
    logger=Logger("converter", str(study_path)),
    output_folder=output_folder,
    lib_paths=[libpath],
    mode="full",
    models_to_convert=["wind_group"],
    scenario_builder_file=scenario_file,
)

input_system = converter.convert_study_to_input_system()

# 2) VALIDATE COMPONENTS

print("\n--- COMPONENTS ---")
scenario_found = False

for comp in input_system.components:
    print(f"* Component: {comp.id}")
    if hasattr(comp, "scenario_group"):
        print(f"  -> scenario_group = {comp.scenario_group}")
        scenario_found = True

assert scenario_found, " No component has scenario-group. Something is wrong."

# 3) VALIDATE SCENARIO FILE COPIED

copied = list(output_folder.glob("**/modeler-scenariobuilder.dat"))
assert copied, " Scenario builder file NOT copied to output!"
print(f"\n✔ Scenario file copied to: {copied[0]}")


print("\n🎉 SUCCESS: scenario-group handled correctly!")
# PosixPath('/home/alzoobiali/Desktop/modeler/andromede-modeling-prototype/converted_test')
