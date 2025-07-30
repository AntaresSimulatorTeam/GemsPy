# PyPSA to Antares end-to-end tests: user guide

This page provides guidelines for conducting end-to-end tests from PyPSA network files to Antares simulations, through the Gems modelling language and data structure.

## 1. Clone the GemsPy repo and switch to the relevant branch

1.a Inside a folder (thereafter called `{parent_folder}`), call the current command line in your Git terminal:

~~~
git clone https://github.com/AntaresSimulatorTeam/GemsPy
~~~

1.b Switch to the branch `pypsa_to_gems/antares_demo`

~~~
git switch pypsa_to_gems/antares_demo
~~~

## 2. Install Python requirements for GemsPy

Install dev requirements with 
~~~
python -m pip install -r requirements-dev.txt
~~~
Depending on your Python installation, you may need to replace `python` with `python3` or `py` or any other reference to your python executable.

## 3.  Get a version of Antares Simulator that includes a Gems interpreter

3.a Download the following .zip archive, from Antares Simulator continuous delivery release: https://github.com/AntaresSimulatorTeam/Antares_Simulator/releases/download/continuous-delivery/rte-antares-cd-installer-64bits.zip.

3.b Unzip the archive inside the folder `parent_folder`. 

3.c The folder that results from the unzipped archived should be renamed `rte-antares-cd-release`.


## 4. Check the paths

The following relative paths should result from the previous steps:
- `{parent_folder}/GemsPy`
- `{parent_folder}/rte-antares-cd-release/bin`, `{parent_folder}/rte-antares-cd-release/lib`... etc


## 5. Run the tests

Being in the GemsPy folder, run the following command in your terminal

~~~
python src/main.py
~~~

This Python script performs, for all PyPSA network files (.nc) stored in the folder [tests\pypsa_converter\pypsa_input_files](tests\pypsa_converter\pypsa_input_files), the following operations:
- From the PyPSA file, create an Antares study (Gems format) in [antares-resources\antares-studies](antares-resources\antares-studies)
- Run Antares Simulator (Gems interpreter),
- Run PyPSA,
- Compare the results (objective function).



