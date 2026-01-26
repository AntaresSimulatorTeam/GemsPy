# GemsPy, a Python interpreter for GEMS

GemsPy is an open-source tool, developed in Python, for the modelling and the simulation of complex energy systems under uncertainty. This is a generic interpreter for the [**GEMS**](https://gems-energy.readthedocs.io/en/latest/) modelling framework.

<img src="docs/images/gemsV2.png" alt="Description" width="400"/>

**Online GemsPy documentation**: [gemspy.readthedocs.io](https://gemspy.readthedocs.io/en/latest/).

## The [GEMS](https://gems-energy.readthedocs.io/en/latest/) framework

### The rationale behind [GEMS](https://gems-energy.readthedocs.io/en/latest/)

[GEMS](https://gems-energy.readthedocs.io/en/latest/) introduces a novel approach to model and simulate energy systems, centered around a simple principle: getting models out of the code.

To develop and test new models of energy system components, writing software code should not be a prerequisite. This is where the **Gems** framework excels, offering users a "no-code" modelling experience with unparalleled versatility.

### Gems = a high-level modelling language + a data structure

The Gems framework consists of a **high-level modelling language**, close to mathematical syntax, and a **data structure** for describing energy systems.

## The  [GemsPy](https://gemspy.readthedocs.io/en/latest/) package

This Python package features a generic interpreter of **Gems** capable of generating optimisation problems from any study case that adhere to the modelling language syntax. It then employs off-the-shelf optimisation solvers to solve these problems. The Python API facilitates reading case studies stored in YAML format, modifying them, or creating new ones from scratch by scripting.

The [Getting started](https://gemspy.readthedocs.io/en/latest/getting-started/) page of the online documentation introduce you to the **Gems** input file format and the basics of the GemsPy API.


## Link with Antares Simulator software
The GemsPy package forms part of the Antares project, but its implementation is completely independent of that of the AntaresSimulator software. Although it was initially designed to prototype the next features of the Antares software (for more information, see [Antares Simulator documentation](https://antares-simulator.readthedocs.io/en/latest/user-guide/modeler/01-overview-modeler/), its structuring and development practices have resulted in high-quality, self-supporting code. It is currently maintained to offer the flexibility of the designed modelling language and interpreter to Python users and to continue exploring its potential. 
