import numpy as np
import openmdao.api as om
import os
import time

from datetime import timedelta
from multiprocessing import TimeoutError
from multiprocessing.pool import ThreadPool
from random import SystemRandom
from xfoil import XFoil
from xfoil.model import Airfoil

from cst import cst, fit
from util import cosspace

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

if not MPI:
    run_parallel = False
    rank = 0
else:
    run_parallel = True
    rank = MPI.COMM_WORLD.rank

formatter = {'float_kind': lambda x: '{: 10.8f}'.format(x)}

coords_file = 'naca0012.dat'
coords_orig = np.loadtxt(coords_file, skiprows=1)

i_0_orig = np.argmin(coords_orig[:, 0])
coords_orig_u = np.flipud(coords_orig[:i_0_orig + 1, :])
coords_orig_l = coords_orig[i_0_orig:, :]
t_te_orig = coords_orig_u[-1, 1] - coords_orig_l[-1, 1]


def fit_coords(n_a_c, n_a_t):
    x = coords_orig_u[:, 0]
    y_u = coords_orig_u[:, 1]
    y_l = np.interp(x, coords_orig_l[:, 0], coords_orig_l[:, 1])

    y_c = (y_u + y_l) / 2
    t = y_u - y_l

    a_c, _ = fit(x, y_c, n_a_c, delta=(0., 0.), n1=1)
    a_t, t_te = fit(x, t, n_a_t)

    return a_c, a_t, t_te[1]


def cst2coords(a_c, a_t, t_te, n_coords, return_intermediate=False):
    x = cosspace(0, 1, n_coords)
    y_c = cst(x, a_c, n1=1)
    t = cst(x, a_t, delta=(0, t_te))

    y_u = y_c + t / 2
    y_l = y_c - t / 2
    if return_intermediate:
        return x, y_c, t, y_u, y_l
    else:
        return x, y_u, y_l


def xfoil_worker(xf, cl_spec, delta, n):
    if n < 1:
        raise ValueError('n needs to be at least 1.')

    _, cd, _, _ = xf.cl(cl_spec)
    if not np.isnan(cd):
        return cd
    elif delta is None or delta < 0.01 or n == 1:
        return cd
    else:
        for i in range(n):
            cl = cl_spec + ((-1.) ** float(i)) * np.ceil(float(i + 1) / 2) * delta
            _, cd, _, _ = xf.cl(cl)
            if not np.isnan(cd):
                _, cd, _, _ = xf.cl(cl_spec)
                return cd
    return cd


def analyze_airfoil(x, y_u, y_l, cl_des, rey, mach=0, xf=None, show_output=False):
    clean = False
    if xf is None:
        xf = XFoil()
        xf.print = show_output
        clean = True

    # If the lower and upper curves swap, this is a bad, self-intersecting airfoil. Return 1e27 immediately.
    if np.any(y_l > y_u):
        return np.nan
    else:
        xf.airfoil = Airfoil(x=np.concatenate((x[-1:0:-1], x)), y=np.concatenate((y_u[-1:0:-1], y_l)))
        # xf.repanel(n_nodes=300, cv_par=2.0, cte_ratio=0.5)
        xf.repanel(n_nodes=240)
        xf.Re = rey
        xf.M = mach
        xf.max_iter = 200

        cd = np.nan
        with ThreadPool(processes=1) as pool:
            future = pool.apply_async(xfoil_worker, args=(xf, cl_des, 0.05, 1))
            try:
                cd = future.get(timeout=10.)
                xf.reset_bls()
            except TimeoutError:
                pass

    if clean:
        del xf

    return cd


class XFoilComp(om.ExplicitComponent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pool = ThreadPool(processes=1)

    def initialize(self):
        self.options.declare('n_c', default=6, types=int)
        self.options.declare('n_t', default=6, types=int)

        self.options.declare('n_coords', default=100, types=int)

        self.options.declare('print', default=False, types=bool)

        xf = XFoil()
        xf.print = False
        self.options.declare('xfoil', default=xf, types=XFoil)

    def setup(self):
        n_c = self.options['n_c']
        n_t = self.options['n_t']

        self.add_input('A_c', shape=n_c)
        self.add_input('A_t', shape=n_t)
        self.add_input('t_te', shape=1)

        self.add_input('Cl_des', val=1.)
        self.add_input('Re', val=1e6)
        self.add_input('M', val=0.)

        self.add_output('Cd', val=1.)

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs, **kwargs):
        t0 = time.time()

        n_coords = self.options['n_coords']
        xf = self.options['xfoil']

        x, y_u, y_l = cst2coords(inputs['A_c'], inputs['A_t'], inputs['t_te'][0], n_coords)
        cd = analyze_airfoil(x, y_u, y_l,
                             inputs['Cl_des'][0], inputs['Re'][0], inputs['M'][0],
                             xf, self.options['print'])
        outputs['Cd'] = cd if not np.isnan(cd) else 1e27

        dt = time.time() - t0
        if self.options['print']:
            print(f'{rank:02d} :: ' +
                  'A_c: {}, '.format(np.array2string(inputs['A_c'], precision=4, suppress_small=True,
                                                     separator=', ', formatter={'float': '{: 7.4f}'.format})) +
                  'A_t: {}, '.format(np.array2string(inputs['A_t'], precision=4, suppress_small=True,
                                                     separator=', ', formatter={'float': '{: 7.4f}'.format})) +
                  f't_te: {inputs["t_te"][0]: 6.4f}, ' +
                  f'C_d: {cd: 7.4f}, dt: {dt:6.3f}'
                  )


class Geom(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n_c', default=6, types=int)
        self.options.declare('n_t', default=6, types=int)

        self.options.declare('n_coords', default=100, types=int)

    def setup(self):
        n_c = self.options['n_c']
        n_t = self.options['n_t']

        self.add_input('A_c', shape=n_c)
        self.add_input('A_t', shape=n_t)
        self.add_input('t_te', shape=1)

        self.add_output('t_c', val=0.)
        self.add_output('A_cs', val=0.)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        n_coords = self.options['n_coords']
        x, _, t, _, _ = cst2coords(inputs['A_c'], inputs['A_t'], inputs['t_te'][0], n_coords, True)
        outputs['t_c'] = np.max(t)
        outputs['A_cs'] = np.trapz(t, x)


class AirfoilOptProblem(om.Problem):

    def __init__(self, n_a_c, n_a_t, seed=None):
        if rank == 0:
            if seed is None:
                seed = int(SystemRandom().random() * (2 ** 31 - 1))
            print(f'SimpleGADriver_seed: {seed}')
            os.environ['SimpleGADriver_seed'] = str(seed)

        a_c_lower = -np.ones(n_a_c)
        a_c_upper = np.ones(n_a_c)
        a_t_lower = 0.01 * np.ones(n_a_t)
        a_t_upper = 0.6 * np.ones(n_a_t)

        a_c, a_t, t_te = fit_coords(n_a_c, n_a_t)

        ivc = om.IndepVarComp()
        ivc.add_output('A_c', val=a_c)
        ivc.add_output('A_t', val=a_t)
        ivc.add_output('t_te', val=t_te)
        ivc.add_output('Re', val=1e6)
        ivc.add_output('M', val=0.)
        ivc.add_output('Cl_des', val=1.)
        ivc.add_output('Cd_0', val=1.)
        ivc.add_output('t_c_0', val=1.)
        ivc.add_output('A_cs_0', val=1.)

        driver = om.SimpleGADriver(bits={'A_c': 10, 'A_t': 10}, run_parallel=run_parallel, max_gen=19)

        # prob.driver = driver = om.ScipyOptimizeDriver()
        # driver.options['optimizer'] = 'SLSQP'
        # driver.options['tol'] = 1e-4
        # driver.options['disp'] = True
        # driver.options['debug_print'] = ['objs']
        # driver.add_recorder(om.SqliteRecorder('dump.sql'))

        model = om.Group()   # model=om.Group(num_par_fd=10))
        model.add_subsystem('ivc', ivc, promotes=['*'])
        model.add_subsystem('XFoil', XFoilComp(n_c=n_a_c, n_t=n_a_t), promotes=['*'])
        model.add_subsystem('Geom', Geom(n_c=n_a_c, n_t=n_a_t), promotes=['*'])
        model.add_subsystem('F', om.ExecComp('obj = Cd / Cd_0',
                                             obj=1, Cd=1., Cd_0=1.), promotes=['*'])
        model.add_subsystem('G1', om.ExecComp('g1 = 1 - t_c / t_c_0', g1=0., t_c=1., t_c_0=1.), promotes=['*'])
        model.add_subsystem('G2', om.ExecComp('g2 = 1 - A_cs / A_cs_0', g2=0, A_cs=1., A_cs_0=1.), promotes=['*'])

        model.add_design_var('A_c', lower=a_c_lower, upper=a_c_upper)
        model.add_design_var('A_t', lower=a_t_lower, upper=a_t_upper)
        model.add_objective('obj')
        model.add_constraint('g1', upper=0.)
        model.add_constraint('g2', upper=0.)

        model.approx_totals(method='fd', step=1e-5)  # method='fd', step=1e-2)

        super().__init__(model=model, driver=driver)

    def __repr__(self):
        s = ''
        s += f'Obj: {self["obj"][0]:6.4f}, C_d: {self["Cd"][0]:6.4f}, \n'
        s += f'A_c: {np.array2string(self["A_c"], formatter=formatter, separator=", ")}, \n'
        s += f'A_t: {np.array2string(self["A_t"], formatter=formatter, separator=", ")}, \n'
        s += f't_te: {self["t_te"][0]: 6.4f}'
        return s


def get_problem(n_a_c, n_a_t, seed=None):
    prob = AirfoilOptProblem(n_a_c, n_a_t, seed)
    prob.setup()

    if rank == 0:
        prob.set_solver_print(2)
    else:
        prob.set_solver_print(-1)

    return prob


def print_problem(prob, dt):
    print(prob.__repr__())
    print(f'Time elapsed: {timedelta(seconds=dt)}')


def analyze(prob, initial=True):
    t0 = time.time()
    prob.run_model()
    dt = time.time() - t0

    if initial:
        prob['Cd_0'] = prob['Cd']
        prob['t_c_0'] = prob['t_c']
        prob['A_cs_0'] = prob['A_cs']

        prob.run_model()

    if rank == 0:
        print_problem(prob, dt)

    return prob


def optimize(prob):
    t0 = time.time()
    prob.run_driver()
    dt = time.time() - t0

    if rank == 0:
        print_problem(prob, dt)

    return prob


def get_coords(prob):
    x, y_u, y_l = cst2coords(prob['A_c'], prob['A_t'], prob['t_te'], 100)
    x = np.reshape(x, (-1, 1))
    y_u = np.reshape(y_u, (-1, 1))
    y_l = np.reshape(y_l, (-1, 1))

    coords_u = np.concatenate((x, y_u), axis=1)
    coords_l = np.concatenate((x, y_l), axis=1)
    coords = np.concatenate((np.flip(coords_u[1:], axis=0), coords_l))

    return coords


def plot(prob_or_coords, show_legend=False, show_title=True):
    import matplotlib.pyplot as plt

    if isinstance(prob_or_coords, om.Problem):
        coords = get_coords(prob_or_coords)
    elif isinstance(prob_or_coords, np.ndarray):
        coords = prob_or_coords
    else:
        raise ValueError('First argument must be either an OpenMDAO Problem or a nunpy.ndarray')

    plt.plot(coords_orig[:, 0], coords_orig[:, 1], 'k', coords[:, 0], coords[:, 1], 'r')
    plt.axis('scaled')
    if show_legend:
        plt.legend(['Original', 'Optimized'])
    if show_title and isinstance(prob_or_coords, om.Problem):
        plt.title(str(prob_or_coords))
    plt.show()


def post_process(prob):
    # Write optimized geometry to dat file
    coords = get_coords(prob)

    fmt_str = 2 * ('{: >' + str(6 + 1) + '.' + str(6) + 'f} ') + '\n'
    with open('optimized.dat', 'w') as f:
        for i in range(coords.shape[0]):
            f.write(fmt_str.format(coords[i, 0], coords[i, 1]))

    if os.environ.get('PLOT_RESULTS'):
        plot(prob)


def main():
    prob = get_problem(3, 3)
    analyze(prob)
    optimize(prob)
    if rank == 0:
        post_process(prob)
    import sys
    sys.exit(0)


if __name__ == '__main__':
    main()
    exit(0)

    x = np.reshape(cosspace(0, 1), (-1, 1))
    y_u = cst(x, [ 0.19530792,  0.31612903,  0.4674486])
    y_l = cst(x, [-0.10244379, -0.04506354,  0.0844086])
    coords_u = np.concatenate((x, y_u), axis=1)
    coords_l = np.concatenate((x, y_l), axis=1)
    coords = np.concatenate((np.flip(coords_u[1:], axis=0), coords_l))

    plt.plot(coords_orig[:, 0], coords_orig[:, 1], 'k', coords[:, 0], coords[:, 1], 'r')
    plt.axis('equal')
    plt.legend(['Original', 'Optimized'])
    plt.show()