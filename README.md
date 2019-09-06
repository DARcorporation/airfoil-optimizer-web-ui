airfoil-optimizer
=================

Tool to optimize an airfoil for minimum drag at a given design lift coefficient using a genetic algorithm.

Edit/run `problem.py` to optimize the problem locally.

Alternatively, build
```
docker build -t af-opt .
```
and run 
```
docker run -it -e np=16 af-opt
```
the Dockerfile to optimize the problem inside a Docker container. 
This container will use 10 processors by default to evaluate functions in parallel using MPI. T
he number of processors can be changed by specifying the `np` environment variable when running the Docker container.