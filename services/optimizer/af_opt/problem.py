#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains the definition of the main optimization problem.
"""
import numpy as np
import openmdao.api as om
import sys
import time

from datetime import timedelta
from differential_evolution import DifferentialEvolutionDriver

from . import rank, run_parallel
from .components.airfoil_component import cst2coords
from .model import AfOptModel
from .recorders import PopulationReporter


def get_de_driver(
    gen=100,
    tolx=1e-8,
    tolf=1e-8,
    strategy="rand-to-best/1/exp/random",
    f=None,
    cr=None,
    adaptivity=2,
):
    kwargs = dict(
        run_parallel=run_parallel,
        adaptivity=adaptivity,
        max_gen=gen,
        tolx=tolx,
        tolf=tolf,
        strategy=strategy,
        show_progress=True,
    )
    if f is not None:
        kwargs.update({"Pm": f})
    if cr is not None:
        kwargs.update({"Pc": cr})

    driver = DifferentialEvolutionDriver(**kwargs)
    return driver


def get_coords(prob):
    """
    Get the coordinates of the airfoil represented by the current state of the airfoil optimization problem.

    Parameters
    ----------
    prob : openmdao.api.Problem
        Airfoil optimization problem

    Returns
    -------
    np.ndarray
        (n, 2) array of x-, and y-coordinates of the airfoil in counterclockwise direction
    """
    x, y_u, y_l, _, _ = cst2coords(prob["a_c"], prob["a_t"], prob["t_te"])
    x = np.reshape(x, (-1, 1))
    y_u = np.reshape(y_u, (-1, 1))
    y_l = np.reshape(y_l, (-1, 1))

    coords_u = np.concatenate((x, y_u), axis=1)
    coords_l = np.concatenate((x, y_l), axis=1)
    coords = np.concatenate((np.flip(coords_u[1:], axis=0), coords_l))

    return coords


def plot(prob, display=False):
    """
    Plot the airfoil represented by the current state of the airfoil optimization problem.

    Parameters
    ----------
    prob : openmdao.api.Problem
        Airfoil optimization problem
    display : bool, optional
        True if the figure should be displayed. False by default

    Returns
    -------
    figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    fig, ax = plt.subplots()
    x, y_u, y_l, y_c, _ = cst2coords(prob["a_c"], prob["a_t"], prob["t_te"])
    ax.plot(x, y_u, "k", x, y_l, "k", x, y_c, "k--")
    ax.axis("scaled")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.grid(which="both")
    if display:
        fig.show()
    return fig


def write(prob, filename):
    """
    Write airfoil coordinates represented by the current state of the airfoil optimization problem to a file

    Parameters
    ----------
    prob : openmdao.api.Problem
        Airfoil optimization problem
    filename : str
        Filename
    """
    coords = get_coords(prob)
    fmt_str = 2 * ("{: >" + str(6 + 1) + "." + str(6) + "f} ") + "\n"
    with open(filename, "w") as f:
        for i in range(coords.shape[0]):
            f.write(fmt_str.format(coords[i, 0], coords[i, 1]))


def main(
    cl,
    n_c,
    n_t,
    gen=100,
    tolx=1e-8,
    tolf=1e-8,
    fix_te=True,
    t_te_min=0.0,
    t_c_min=0.01,
    A_cs_min=None,
    Cm_max=None,
    strategy="rand-to-best/1/exp/random",
    f=None,
    cr=None,
    adaptivity=2,
    repr_file="repr.yml",
    dat_file="optimized.dat",
    png_file="optimized.png",
):
    """
    Create, analyze, optimize airfoil, and write optimized coordinates to a file. Then clean the problem up and exit.

    Parameters
    ----------
    cl : float
        Design lift coefficient
    n_c, n_t : int
        Number of CST coefficients for the chord line and thickness distribution, respectively
    gen : int, optional
        Number of generations to use for the genetic algorithm. 100 by default
    tolx : float, optional
        Tolerance on the spread of the design vectors.
    tolf: float, optional
        Tolerance on the spread of objective functions.
    fix_te : bool, optional
        True if the trailing edge thickness should be fixed. True by default
    t_te_min : float, optional
        Minimum TE thickness as fraction of chord length. Default is 0.0.
    t_c_min : float or None, optional
        Minimum thickness over chord ratio. None if unconstrained. Defaults is 0.01.
    A_cs_min : float or None, optional
        Minimum cross sectional area. None if unconstrained. Default is None.
    Cm_max : float or None, optional
        Maximum absolute moment coefficient. None if unconstrained. Default is None.
    strategy : string, optional
        Evolution strategy to use. Default is 'rand-to-best/1/exp/random'.
    f : float or None, optional
        Mutation rate
    cr : float or None, optional
        Crossover rate
    adaptivity : 0, 1, or 2
        Which kind of self-adaptivity to ue (0: none, 1: simple, 2: complex)
    repr_file, dat_file, png_file : str, optional
        Paths where the final representation, optimized airfoil coordinates, and output image should be saved.
    """
    # Construct the OpenMDAO Problem
    kwargs = dict(
        n_c=n_c,
        n_t=n_t,
        fix_te=fix_te,
        t_te_min=t_te_min,
        t_c_min=t_c_min,
        A_cs_min=A_cs_min,
        Cm_max=Cm_max,
    )

    prob = om.Problem()
    prob.model = AfOptModel(**kwargs)

    prob.driver = get_de_driver(gen, tolx, tolf, strategy, f, cr, adaptivity)
    prob.driver.add_recorder(PopulationReporter())
    prob.setup()

    # Set reference values
    prob["Cl_des"] = cl

    # Optimize the problem using a genetic algorithm
    t0 = time.time()
    prob.run_driver()
    dt = time.time() - t0

    # Show and write final results
    if rank == 0:
        yaml = prob.model.__repr__()
        print("Optimized airfoil:")
        print("    " + yaml.replace("\n", "\n    "))
        print(f"Time Elapsed: {timedelta(seconds=dt)}")

        with open(repr_file, "w") as f:
            f.write(yaml)
        write(prob, filename=dat_file)
        fig = plot(prob)
        fig.savefig(png_file)

    # Clean up and exit
    prob.cleanup()
    del prob

    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) == 19:
        main(
            cl=float(sys.argv[1]),
            n_c=int(sys.argv[2]),
            n_t=int(sys.argv[3]),
            gen=int(sys.argv[4]),
            tolx=float(sys.argv[5]),
            tolf=float(sys.argv[6]),
            fix_te=(sys.argv[7] == "True"),
            t_te_min=float(sys.argv[8]),
            t_c_min=None if sys.argv[9] == "None" else float(sys.argv[9]),
            A_cs_min=None if sys.argv[10] == "None" else float(sys.argv[10]),
            Cm_max=None if sys.argv[11] == "None" else float(sys.argv[11]),
            strategy=sys.argv[12],
            f=None if sys.argv[13] == "None" else float(sys.argv[13]),
            cr=None if sys.argv[14] == "None" else float(sys.argv[14]),
            adaptivity=int(sys.argv[15]),
            repr_file=sys.argv[16],
            dat_file=sys.argv[17],
            png_file=sys.argv[18],
        )
    else:
        main(1.0, 3, 3, gen=9)
