airfoil-optimizer
=================

Tool to optimize an airfoil for minimum drag at a given design lift coefficient using a genetic algorithm.

The most convenient way to run the optimization is to use Docker. The container can be build by running
```
docker build -t af-opt .
```
from the root of the repo. Then the optimization can be started by running
```
docker run -e np=16 af-opt
```
This will run the optimization using 16 processors to evaluate functions in parallel using MPI.
The number of processors can be changed by specifying the `np` environment variable when running the Docker container.
If `np` is not specified, only a single processor will be used.

The Docker container will not store or plot any results. It will only output those to the console window. To 
post-process the solution, the user may want to use some of the convenience methods in `problem.py` locally.