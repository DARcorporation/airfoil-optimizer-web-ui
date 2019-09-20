import numpy as np

from openmdao.utils.mpi import MPI

from . import *

# Detect whether the script is being run under MPI and grab the rank
if not MPI:
    rank = 0
    n_proc = 1
else:
    rank = MPI.COMM_WORLD.rank
    n_proc = MPI.COMM_WORLD.size


def paraboloid(x):
    import time
    time.sleep(0.01)
    return np.sum(x * x)


def test_differential_evolution():
    import matplotlib.pyplot as plt

    comm = None if not MPI else MPI.COMM_WORLD
    strategy = EvolutionStrategy("rand/1/exp")
    de = DifferentialEvolution(paraboloid, bounds=[(-500, 500)] * 2, comm=comm, strategy=strategy, tolx=0., tolf=0.)
    x_min, f_min = [], []
    for generation in de:
        x_min += [generation.best]
        f_min += [generation.best_fit]
        print(f"{np.array2string(x_min[-1], formatter={'float_kind': '{:15g}'.format})}{f_min[-1]:15g}")

    print()
    print(f"Optimization complete!")
    print(f"x*: {x_min[-1]}")
    print(f"f*: {f_min[-1]}")

    plt.plot(f_min)
    plt.show()
