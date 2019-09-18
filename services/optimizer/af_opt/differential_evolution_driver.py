import copy
import numpy as np

from openmdao.utils.concurrent import concurrent_eval
from openmdao.utils.mpi import MPI
from tqdm import tqdm

# Detect whether the script is being run under MPI and grab the rank
if not MPI:
    rank = 0
    n_proc = 1
else:
    rank = MPI.COMM_WORLD.rank
    n_proc = MPI.COMM_WORLD.size


class DifferentialEvolution:

    def __init__(self,
                 objfun, comm=None, model_mpi=None,
                 gen=1, variant=2, variant_adaptv=1, ftol=1e-6, xtol=1e-6,
                 memory=False, seed=None):
        if 1 > variant > 18:
            raise ValueError("The Differential Evolution mutation variant must be in [1, ... 18].")
        if 1 > variant_adaptv > 2:
            raise ValueError("The variant for self-adaption must be in [1, 2].")

        self.objfun = objfun
        self.comm = comm
        self.model_mpi = model_mpi

        self.gen = gen
        self.variant = variant
        self.variant_adaptv = variant_adaptv
        self.ftol = ftol
        self.xtol = xtol
        self.memory = memory
        self.seed = seed

        self.verbosity = 0

        self._random_state = np.random.RandomState(seed)
        self._F = np.empty((0,))
        self._CR = np.empty((0,))

    def evaluate(self, pop, comm=None):
        n_pop = pop.shape[0]
        fitness = np.full((n_pop,), np.inf)
        success = np.full((n_pop,), False)

        # Evaluate generation
        if comm is not None:
            # Parallel

            # Use population of rank 0 on all processors
            pop = comm.bcast(pop, root=0)

            cases = [((item, ii), None) for ii, item in enumerate(pop)]
            # Pad the cases with some dummy cases to make the cases divisible amongst the procs.
            extra = len(cases) % comm.size
            if extra > 0:
                for j in range(comm.size - extra):
                    cases.append(cases[-1])

            results = concurrent_eval(self.objfun, cases, comm, allgather=True,
                                      model_mpi=self.model_mpi)

            for result in results:
                returns, traceback = result

                if returns:
                    val, _success, ii = returns
                    if _success:
                        fitness[ii] = val
                        success[ii] = True
                else:
                    # Print the traceback if it fails
                    print('A case failed:')
                    print(traceback)
        else:
            # Serial
            for ii in range(n_pop):
                x = pop[ii]

                fitness[ii], success[ii], _ = self.objfun(x, 0)

        return fitness, success

    def evolve(self, pop, vlb, vub):
        if pop.shape[0] < 7:
            raise ValueError("Differential Evolution needs at least 7 individuals in the population.")

        comm = self.comm
        n_pop = pop.shape[0]
        n_dim = pop.shape[1]

        # Initial evaluation of population
        fit, success = self.evaluate(pop, comm)

        # Global bests
        best_idx = np.argmin(fit)
        best_x = pop[best_idx]
        best_f = fit[best_idx]
        r = np.empty((7,))  # Indices of 7 selected individuals

        # Initialize F and CR vectors
        if self._CR.size != n_pop or self._F.size != n_pop or not self.memory:
            if self.variant_adaptv == 1:
                self._CR = self._random_state.rand(n_pop, 1)
                self._F = self._random_state.rand(n_pop, 1) * 0.9 + 0.1
            elif self.variant_adaptv == 2:
                self._CR = self._random_state.normal(0., 1., (n_pop,)) * 0.15 + 0.5
                self._F = self._random_state.normal(0., 1., (n_pop,)) * 0.15 + 0.5

        # Global iteration bests for F and CR
        best_F = self._F[0]
        best_CR = self._CR[0]

        # Main loop
        idxs = np.empty((n_pop,))
        tmp = np.empty_like(pop)

        gen_iter = range(self.gen + 1)
        if rank == 0:
            gen_iter = tqdm(gen_iter, ascii=True)

        nfit = 0
        F, CR = np.empty((n_pop,)), np.empty((n_pop,))
        for generation in gen_iter:
            pop_old = np.copy(pop)
            pop_tmp = np.copy(pop)

            for i in range(n_pop):
                idxs = np.linspace(n_pop)
                for j in range(7):  # Durstenfeld's algorithm to select 7 indexes at random
                    idx = self._random_state.randint(0, n_pop - 1 - j)
                    r[j] = idxs[idx]
                    idxs[[idx, n_pop - 1 - j]] = idxs[[n_pop - 1 - j, idx]]

                # Adapt amplification factor and crossover probability for variant_adpttv= 1
                F[i], CR[i] = 0., 0.
                if self.variant_adaptv == 1:
                    F[i] = self._F[i] if self._random_state.randn() < 0.9 else self._random_state.randn() * 0.9 + 0.1
                    CR[i] = self._CR[i] if self._random_state.randn() < 0.9 else self._random_state.randn()

                n = self._random_state.randint(0, n_dim)
                # -------DE/best/1/exp--------------------------------------------------------------------
                # -------The oldest DE variant but still not bad. However, we have found several----------
                # -------optimization problems where misconvergence occurs.-------------------------------
                if self.variant == 1:
                    if self.variant_adaptv == 2:
                        F[i] = best_F + \
                            self._random_state.normal(0., 1.) * 0.5 * (self._F[r[1]] - self._F[r[2]])
                        CR[i] = best_CR + \
                             self._random_state.normal(0., 1.) * 0.5 * (self._CR[r[1]] - self._CR[r[2]])
                    for L in range(n_dim):
                        pop_tmp[i, n] = best_x[n] + \
                                 F[i] * (pop_old[r[1]][n] - pop_old[r[2]][n])
                        n = (n + 1) % n_dim
                        if self._random_state.randn() < CR[i]:
                            break

                # -------DE/rand/1/exp-------------------------------------------------------------------
                elif self.variant == 2:
                    if self.variant_adaptv == 2:
                        F[i] = self._F[i] + \
                            self._random_state.normal(0., 1.) * 0.5 * (self._F[r[1]] - self._F[r[2]])
                        CR[i] = self._CR[i] + \
                             self._random_state.normal(0., 1.) * 0.5 * (self._CR[r[1]] - self._CR[r[2]])
                    for L in range(n_dim):
                        pop_tmp[i, n] = pop_old[r[0]][n] + \
                                 F[i] * (pop_old[r[1]][n] - pop_old[r[2]][n])
                        n = (n + 1) % n_dim
                        if self._random_state.randn() < CR[i]:
                            break

                # -------DE/rand-to-best/1/exp-----------------------------------------------------------
                elif self.variant == 3:
                    if self.variant_adaptv == 2:
                        F[i] = self._F[i] + \
                            self._random_state.normal(0., 1.) * 0.5 * (best_F - self._F[i]) + \
                            self._random_state.normal(0., 1.) * 0.5 * (self._F[r[0]] - self._F[r[1]])
                        CR[i] = self._CR[i] + \
                             self._random_state.normal(0., 1.) * 0.5 * (best_CR - self._CR[i]) + \
                             self._random_state.normal(0., 1.) * 0.5 * (self._CR[r[0]] - self._CR[r[1]])
                    for L in range(n_dim):
                        pop_tmp[i, n] = tmp[n] + \
                                 F[i] * (best_x[n] - tmp[n]) + \
                                 F[i] * (pop_old[r[0]][n] - pop_old[r[1]][n])
                        n = (n + 1) % n_dim
                        if self._random_state.randn() < CR[i]:
                            break

                # -------DE/best/2/exp is another powerful variant worth trying--------------------------
                elif self.variant == 4:
                    if self.variant_adaptv == 2:
                        F[i] = best_F + \
                            self._random_state.normal(0., 1.) * 0.5 * (self._F[r[0]] - self._F[r[1]]) + \
                            self._random_state.normal(0., 1.) * 0.5 * (self._F[r[2]] - self._F[r[3]])
                        CR[i] = best_CR + \
                             self._random_state.normal(0., 1.) * 0.5 * (self._CR[r[0]] - self._CR[r[1]]) + \
                             self._random_state.normal(0., 1.) * 0.5 * (self._CR[r[2]] - self._CR[r[3]])
                    for L in range(n_dim):
                        pop_tmp[i, n] = best_x[n] + \
                                 F[i] * (pop_old[r[0]][n] - pop_old[r[1]][n]) + \
                                 F[i] * (pop_old[r[2]][n] - pop_old[r[3]][n])
                        n = (n + 1) % n_dim
                        if self._random_state.randn() < CR[i]:
                            break

                # -------DE/rand/2/exp seems to be a robust optimizer for many functions-------------------
                elif self.variant == 5:
                    if self.variant_adaptv == 2:
                        F[i] = self._F[r[4]] + \
                            self._random_state.normal(0., 1.) * 0.5 * (self._F[r[0]] - self._F[r[1]]) + \
                            self._random_state.normal(0., 1.) * 0.5 * (self._F[r[2]] - self._F[r[3]])
                        CR[i] = self._CR[r[4]] + \
                             self._random_state.normal(0., 1.) * 0.5 * (self._CR[r[0]] - self._CR[r[1]]) + \
                             self._random_state.normal(0., 1.) * 0.5 * (self._CR[r[2]] - self._CR[r[3]])
                    for L in range(n_dim):
                        pop_tmp[i, n] = pop_old[r[4]][n] + \
                                 F[i] * (pop_old[r[0]][n] - pop_old[r[1]][n]) + \
                                 F[i] * (pop_old[r[2]][n] - pop_old[r[3]][n])
                        n = (n + 1) % n_dim
                        if self._random_state.randn() < CR[i]:
                            break

                else:
                    raise NotImplementedError

                # Force feasibility
                for j in range(n_dim):
                    if pop_tmp[i, j] < vlb[j] or tmp[j] > vub[j]:
                        pop_tmp[i, j] = vlb[i] + self._random_state.randn() * (vub[j] - vlb[i])

            # Evaluate current population
            fit_old = np.copy(fit)
            fit_tmp, success = self.evaluate(pop_old, comm)
            nfit += np.count_nonzero(success)

            # Replace individuals if they have improved
            idxs = np.argwhere(fit_tmp <= fit_old)
            pop[idxs] = pop_tmp[idxs]
            fit[idxs] = fit_tmp[idxs]
            self._CR[idxs] = CR[idxs]
            self._F[idxs] = F[idxs]

            # Find best performing point in this generation.
            min_idx = np.argmin(fit)
            min_f = np.min(fit)

            # Replace global bests if it is an improvement
            if min_f <= best_f:
                best_f = min_f
                best_x = pop[min_idx]
                best_F = F[min_idx]
                best_CR = CR[min_idx]

            # Check the exit conditions
            best_idx = np.argmin(fit)
            worst_idx = np.argmax(fit)
            dx = np.sum(np.abs(pop[worst_idx] - pop[best_idx]))
            if dx < self.xtol:
                if self.verbosity:
                    print(f"Exit condition -- xtol < {self.xtol}")
                return pop

            df = abs(fit[worst_idx] - fit[best_idx])
            if df < self.ftol:
                if self.verbosity:
                    print(f"Exit condition -- ftol < {self.ftol}")
                return pop

        # End main generation
        if self.verbosity:
            print(f"Exit condition -- generations = {self.gen}")
        return pop
