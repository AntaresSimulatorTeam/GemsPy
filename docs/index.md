# Gems, a framework for generic modelling in energy systems


![Gems Logo](/images/gemsV2cropped.png)


## Motivation

As energy systems become more complex and dynamic, we need to improve energy planning tools further, in terms of:

- **Versatility**: easily integrate new models or components without rewriting core code.  
  *Writing and testing new models of energy system components should not require software programming skills!*

- **Transparency**: clearly expose the mathematical logic behind the models.

- **Interoperability**: interact seamlessly with external tools or formats.

- **Code stability and suitability for open-source**: prevent the simulator core from becoming overloaded with hard-coded logic.

## The Gems framework

The Gems framework consists of a **high-level modelling language**, close to mathematical syntax, and a **data structure** for describing energy systems.

More specifically, three main types of input file can be defined with the Gems framework:

- **Model libraries**: describe abstract component models.  
- **System files**: describe the graph of components that make up a system of interest; refer to model libraries (instantiation of abstract models) and to timeseries files.  
- **Timeseries files**: the data of timeseries.

## The Gems interpreters

Two open-source software packages are capable of reading and simulating the case studies described in the Gems framework:

- [GemsPy](https://github.com/AntaresSimulatorTeam/GemsPy)
- [Antares Simulator](https://antares-simulator.org/) *(functionality under development)*

## Getting started

To create a run a study, refer to the [Getting started](getting-started.md) section.

## User guide

To understand in-depth concepts behind the modeler, refer to the [User guide](user-guide.md).