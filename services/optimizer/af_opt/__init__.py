#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains imports and checks to detect MPI.
"""
# Ensure MPI is defined
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# Detect whether the script is being run under MPI and grab the rank
if not MPI:
    run_parallel = False
    rank = 0
    n_proc = 1
else:
    run_parallel = True
    rank = MPI.COMM_WORLD.rank
    n_proc = MPI.COMM_WORLD.size
