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

The Python script [main.py](https://github.com/AntaresSimulatorTeam/GemsPy/tree/pypsa_to_gems/antares_demo/src/main.py) performs, for all PyPSA network files (.nc) stored in the folder [tests\pypsa_converter\pypsa_input_files](https://github.com/AntaresSimulatorTeam/GemsPy/tree/pypsa_to_gems/antares_demo/tests/pypsa_converter/pypsa_input_files), the following operations:
- From the PyPSA file, create an Antares study (Gems format) in [antares-resources\antares-studies](https://github.com/AntaresSimulatorTeam/GemsPy/tree/pypsa_to_gems/antares_demo/antares-resources/antares-studies)
- Run Antares Simulator (Gems interpreter),
- Run PyPSA,
- Compare the results (objective function).


## Additionnal remarks

- Current limitations of the PyPSA > Gems converter: The following objects are currently supported (July 2025): Bus, Load, Generator, Store, StorageUnit, GlobalConstraint, Link and Carrier. Line and Transformer objects are not yet supported. Other current restrictions of the data converter, such as hourly granularity only, are verified using assertions.

- After executing the `main.py` script, the Antares studies in Gems format representing the PyPSA test cases can be found in `GemsPy/antares-resources/antares-studies`, particularly the inputs (libraries of abstract models, the system file and the timeseries). The outputs are in raw format for now. More structured outputs will be available soon.

- Performance of the Gems interpreter in the Antares simulator: the current implementation is not optimised and is mainly used to demonstrate the concept. Several performance improvements can be made to the implementation, and will be included in the roadmap.
