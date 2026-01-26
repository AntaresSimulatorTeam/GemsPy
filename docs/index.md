# GemsPy, a Python interpreter for the GEMS modelling language


![Gems Logo](/images/gemsV2cropped.png)


## Motivation

As energy systems become more complex and dynamic, we need to improve energy planning tools further, in terms of:

- **Versatility**: easily integrate new models or components without rewriting core code.  
  *Writing and testing new models of energy system components should not require software programming skills!*

- **Transparency**: clearly expose the mathematical logic behind the models.

- **Interoperability**: interact seamlessly with external tools or formats.

- **Code stability and suitability for open-source**: prevent the simulator core from becoming overloaded with hard-coded logic.

## About GEMS

The GEMS framework consists of a **algebraic modelling language**, close to mathematical syntax, and a **data structure** for describing energy systems.

For further information regarding the language, please consult the [GEMS documentation](https://gems-energy.readthedocs.io/en/latest/) website. 

## The GEMS interpreters

Two open-source software packages are capable of reading and simulating the case studies described in the GEMS:

- [GemsPy](https://github.com/AntaresSimulatorTeam/GemsPy)
- [Antares Simulator](https://antares-simulator.org/) *(functionality under development)*

## Getting started

To create a run a study with GemsPy, refer to the [Getting started](getting-started.md) section.

## User guide

To understand in-depth concepts behind the modeler, refer to the [User guide](user-guide.md).
