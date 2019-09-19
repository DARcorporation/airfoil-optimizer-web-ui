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


class EvolutionaryStrategy:

    def __init__(self, string, rng):
        self.name = string
        self.rng = rng

    def mutate(self, parent_idx, population, mutation_rate):
        idxs = [idx for idx in range(population.shape[0]) if idx != parent_idx]
        a, b, c = population[self.rng.choice(idxs, size=3, replace=False)]
        return a + mutation_rate * (b - c)

    def crossover(self, parent, child, probability):
        n = len(parent)
        p = self.rng.uniform(size=n) < probability
        if not np.any(p):
            p[self.rng.integers(n)] = True
        return np.where(p, child, parent)

    def repair(self, child):
        return np.clip(child, 0, 1)

    def __call__(self, target_idx, population, mutation_rate, crossover_probability):
        child = self.mutate(target_idx, population, mutation_rate)
        child = self.crossover(population[target_idx], child, crossover_probability)
        child = self.repair(child)
        return child


class DifferentialEvolution:

    @staticmethod
    def mpi_fobj_wrapper(fobj):
        def wrapped(x, ii):
            return fobj(x), ii
        return wrapped

    def __init__(self, fobj, bounds,
                 mut=0.85, crossp=1., strategy='rand/1/bin',
                 max_gen=100, tolx=1e-6, tolf=1e-6,
                 n_pop=None, seed=None, comm=None, model_mpi=None):
        self.fobj = fobj if comm is None else self.mpi_fobj_wrapper(fobj)

        self._lb, self._ub = np.asarray(bounds).T
        self._range = self._ub - self._lb

        self.f = mut
        self.cr = crossp

        self.max_gen = max_gen
        self.tolx = tolx
        self.tolf = tolf

        self.n_dim = len(bounds)
        self.n_pop = n_pop if n_pop is not None else self.n_dim * 5

        self._rng = np.random.default_rng(seed)

        self.comm = comm
        self.model_mpi = model_mpi

        self.strategy = EvolutionaryStrategy(strategy, self._rng)

        self.pop = self._rng.uniform(size=(self.n_pop, self.n_dim))
        self.fit = self(self.pop)
        self.best_idx, self.worst_idx = 0, 0
        self.best, self.worst = None, None
        self.best_fit, self.worst_fit = 0, 0
        self.update(self.pop, self.fit)

        self.generation = 0

    def __iter__(self):
        while self.generation < self.max_gen:
            pop_new = self.offspring()
            fit_new = self(pop_new)
            self.update(pop_new, fit_new)

            yield self
            self.generation += 1

            dx = np.sum((self._range * (self.worst - self.best)) ** 2) ** 0.5
            df = np.abs(self.worst_fit - self.best_fit)
            if dx < self.tolx:
                break
            if df < self.tolf:
                break

    def __call__(self, pop):
        # Evaluate generation
        if self.comm is not None:
            # Use population of rank 0 on all processors
            pop = self.comm.bcast(pop, root=0)

            cases = [((item, ii), None) for ii, item in enumerate(pop)]
            # Pad the cases with some dummy cases to make the cases divisible amongst the procs.
            extra = len(cases) % self.comm.size
            if extra > 0:
                for j in range(self.comm.size - extra):
                    cases.append(cases[-1])

            results = concurrent_eval(self.fobj, cases, self.comm, allgather=True,
                                      model_mpi=self.model_mpi)

            fit = np.full((self.n_pop,), np.inf)
            for result in results:
                val, ii = result
                fit[ii] = val
        else:
            fit = [self.fobj(ind) for ind in pop]
        return np.asarray(fit)

    def offspring(self):
        pop_old_norm = (np.copy(self.pop) - self._lb) / self._range
        pop_new_norm = [self.strategy(idx, pop_old_norm, self.f, self.cr) for idx in range(self.n_pop)]
        return self._lb + self._range * np.asarray(pop_new_norm)

    def update(self, pop_new, fit_new):
        improved_idxs = np.argwhere(fit_new <= self.fit)
        self.pop[improved_idxs] = pop_new[improved_idxs]
        self.fit[improved_idxs] = fit_new[improved_idxs]

        self.best_idx = np.argmin(self.fit)
        self.best = self.pop[self.best_idx]
        self.best_fit = self.fit[self.best_idx]

        self.worst_idx = np.argmax(self.fit)
        self.worst = self.pop[self.worst_idx]
        self.worst_fit = self.fit[self.worst_idx]


def paraboloid(x):
    import time
    time.sleep(0.01)
    return np.sum(x * x)


def main():
    from yabox.algorithms.de import DE
    comm = None if not MPI else MPI.COMM_WORLD
    de = DifferentialEvolution(paraboloid, bounds=[(-500, 500)] * 2, comm=comm)
    results = [generation for generation in tqdm(de, total=100)]

    print()
    print(f"Optimization complete!")
    print(f"x*: {results[-1].best}")
    print(f"f*: {results[-1].best_fit}")


if __name__ == '__main__':
    main()
