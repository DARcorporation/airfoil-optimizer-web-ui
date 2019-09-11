# Airfoil Optimizer
[![Build Status](https://travis-ci.com/daniel-de-vries/airfoil-optimizer.svg?branch=master)](https://travis-ci.com/daniel-de-vries/airfoil-optimizer)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Tool to optimize an airfoil for minimum drag at a given design lift coefficient using a genetic algorithm.

## Airfoil Specification
An airfoil is specified by a mean chord line, thickness distribution, and trailing edge thickness.
The mean chord line and thickness distribution are parameterized using the Class-Shape Transformation (CST) technique. 
The number of CST coefficients used to parameterize the mean chord line and thickness distributions
can be chosen by the user. 

## Usage
Optimization cases are specified in the `Runfile`. Each line in this file is should to be a comma-separated list of 
positional argument values (`*args`) and followed by /or key-value pairs (**kwargs) and corresponds to a single 
optimization problem to be solved. Acceptable key-value pairs correspond to the inputs taken by the `run` function 
specified in the `runner.py` script:

```python
def run(cl, n_c, n_t, b_c=8, b_t=8, b_te=8, gen=100,
        fix_te=True,
        constrain_thickness=True, constrain_area=True, constrain_moment=True,
        cm_ref=None, seed=None, n_proc=28,
        report=False,
        results_output_folder=None):
    """
    Solve the specified optimization problem and handle reporting of results.

    Parameters
    ----------
    cl : float
        Design lift coefficient
    n_c, n_t : int
        Number of CST coefficients for the chord line and thickness distribution, respectively
    b_c, b_t, b_te : int, optional
        Number of bits to encode each of the CST coefficients of the chord line/thickness distribution, and TE thickness
        8 bits each by default.
    gen : int, optional
        Number of generations to use for the genetic algorithm. 100 by default
    fix_te : bool, optional
        True if the trailing edge thickness should be fixed. True by default
    constrain_thickness, constrain_area, constrain_moment : bool, optional
        True if the thickness, area, and/or moment coefficient should be constrained, respectively. All True by default
    cm_ref : float, optional
        If constrain_moment is True, this will be the maximum (absolute) moment coefficient. If None, initial Cm is used
    seed : int, optional
        Seed to use for the random number generator which creates an initial population for the genetic algorithm
    n_proc : int, optional
        Number of processors to use to evaluate functions in parallel using MPI. 28 by default
    report : bool, optional
        True if the results should be reported via email. False by default
    results_output_folder : str, optional
        Name of the shared folder in which to store results. By default, an ISO formatted UTC timestamp will be used.
    """
    ...
```
For example, a line in the `Runfile` could be:

`1.0, 3, 3, 8, 8, gen=100, fix_te=True, constrain_moment=False, n_proc=16`

This defines an airfoil optimization at a design lift coefficient of `1.0`, using 3 CST coefficients for the chord line 
and thickness distribution, 8 bits to encode each of the CST coefficients, and 100 generations for the genetic algorithm
, while keeping the TE thickness fixed and not constraining the moment coefficient, using 16 processors.

After specifying optimization problems in the `Runfile`, run the command 

```shell script
docker-compose up --build
```

from the root directory of the repository to solve all specified problems in order.

## Results
The results of each optimization run are also written to persistent files, available on the host
machine in the `share` directory relative to the root directory of this repository. By default, each optimization
problem will write to its own sub-folder under the `share` folder with a name based on the ISO formatted UTC date/time 
at the time the optimization finished.

Additionally, if `report=True` is specified, the results of an optimization run will be send to an email address of the 
user's choosing. For this to work, the user should specify an STMP server hostname and port, username, password, and 
receiver address in `stmp_settings.conf`. 
