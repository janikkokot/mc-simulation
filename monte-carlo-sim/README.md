# monte-carlo-sim

[![PyPI - Version](https://img.shields.io/pypi/v/monte-carlo-sim.svg)](https://pypi.org/project/monte-carlo-sim)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/monte-carlo-sim.svg)](https://pypi.org/project/monte-carlo-sim)

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Overview

The package provides a interface to run monte-carlo molecular simulations for Van-der-Waals fluids.
Periodic and non periodic conditions are supported as well as fixed and addaptive step size adjustment.

The in- and output of the coordinates / trajectory is handled via [XYZ-files](https://royallgroup.github.io/TCC/html/xyz_specification.html).
Parameter are stored in [TOML-files](https://toml.io/en/).

## Installation

```console
pip install monte-carlo-sim
```

## User Interface

The program contains a command-line interface (CLI) which provides an explaination for the various flags.
```console
monte-carlo-sim -h
```

Only the input coordinates and the output path are required.
```console
monte-carlo-sim -c data/start.xyz -x data/start_trajectory.xyz
```

## Internals

The package contains three modules:
| module     | purpose | up for evaluation |
|------------|---------|-------------------|
|`handle_xyz`| xyz file reading and writing utilities | :heavy_check_mark: |
|------------|----------------------------------------|--------------------|
|`distance`  | non-periodic and periodic boundary distance calculation | :heavy_check_mark: |
|------------|---------------------------------------------------------|--------------------|
|`simulate`  | energy calculation and monte carlo algorithm | :wavy_dash:|
|------------|----------------------------------------------|------------|

The `run` script is responsible for the CLI. The `toml`-files are parameter-files. 
Currently only parameters for Argon and Methane are defined.

## Tests

The package contains both doctests as well as regular tests intended to be run with `pytest`.

## Dependencies

The package only requires `tomli` in order to be backcompatible.

## License

`monte-carlo-sim` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
