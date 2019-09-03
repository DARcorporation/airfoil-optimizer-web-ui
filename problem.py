import numpy as np
import openmdao.api as om
import os
import time

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
with open(coords_file, 'r') as f:
    lines = f.readlines()[2:]

coords_orig = np.zeros((len(lines), 2))
for i in range(len(lines)):
    coords_orig[i, :] = np.fromstring(lines[i], dtype=float, count=2, sep=' ')
# coords_orig = np.loadtxt('naca0012.dat', skiprows=1)


def fit_coords(n_a_u, n_a_l):
    i_0 = np.argmin(coords_orig[:, 0])
    coords_u = coords_orig[:i_0 + 1, :]
    coords_l = coords_orig[i_0:, :]

    A_u, _ = fit(coords_u[:, 0], coords_u[:, 1], n_a_u, delta=(0., 0.))
    A_l, _ = fit(coords_l[:, 0], coords_l[:, 1], n_a_l, delta=(0., 0.))
    return A_u, A_l


class XFoilComp(om.ExplicitComponent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pool = ThreadPool(processes=1)

    def initialize(self):
        self.options.declare('n_u', default=6, types=int)
        self.options.declare('n_l', default=6, types=int)

        self.options.declare('n_coords', default=100, types=int)

        xf = XFoil()
        xf.print = False
        self.options.declare('xfoil', default=xf, types=XFoil)

    def setup(self):
        n_u = self.options['n_u']
        n_l = self.options['n_l']

        self.add_input('A_u', shape=n_u)
        self.add_input('A_l', shape=n_l)

        self.add_input('Cl_des', val=1.)
        self.add_input('Re', val=1e6)
        self.add_input('M', val=0.)

        self.add_output('Cd', val=1.)

        self.declare_partials('*', '*', method='fd')

    @staticmethod
    def worker(xf, cl_spec):
        _, cd, _, _ = xf.cl(cl_spec)
        if np.isnan(cd):
            xf.reset_bls()
            _, cl, cd, _, _ = xf.cseq(cl_spec - 0.05, cl_spec + 0.055, 0.005)
            return np.interp(cl_spec, cl, cd)
        else:
            return cd

    def compute(self, inputs, outputs, **kwargs):
        t0 = time.time()

        n_coords = self.options['n_coords']
        xf = self.options['xfoil']

        x = cosspace(0, 1, n_coords)
        y_u = cst(x, inputs['A_u'])
        y_l = cst(x, inputs['A_l'])

        xf.airfoil = Airfoil(x=np.concatenate((x[-1:0:-1], x)), y=np.concatenate((y_u[-1:0:-1], y_l)))
        # xf.filter()
        xf.repanel(n_nodes=240)
        xf.Re = inputs['Re'][0]
        xf.M = inputs['M'][0]
        xf.max_iter = 200

        future = self._pool.apply_async(XFoilComp.worker, args=(xf, inputs['Cl_des'][0]))
        cd = np.nan
        try:
            cd = future.get(timeout=1)
        except TimeoutError:
            pass
        outputs['Cd'] = 1e27 if np.isnan(cd) else cd

        dt = time.time() - t0
        # print(f'{rank:02d} :: ' +
        #       'A_u: {}, '.format(np.array2string(inputs['A_u'], precision=4, suppress_small=True,
        #                                          separator=' ', formatter={'float': '{: 7.4f}'.format})) +
        #       'A_l: {}, '.format(np.array2string(inputs['A_l'], precision=4, suppress_small=True,
        #                                          separator=' ', formatter={'float': '{: 7.4f}'.format})) +
        #       f'C_d: {cd: 7.4f}, dt: {dt:6.3f}'
        #       )


class Geom(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n_u', default=6, types=int)
        self.options.declare('n_l', default=6, types=int)

        self.options.declare('n_coords', default=100, types=int)

    def setup(self):
        n_u = self.options['n_u']
        n_l = self.options['n_l']

        self.add_input('A_u', shape=n_u)
        self.add_input('A_l', shape=n_l)

        self.add_output('t_c', val=0.)
        self.add_output('A_cs', val=0.)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        n_coords = self.options['n_coords']

        x = np.reshape(cosspace(0, 1, n_coords), (-1, 1))
        y_u = cst(x, inputs['A_u'])
        y_l = cst(x, inputs['A_l'])
        dy = y_u - y_l

        outputs['t_c'] = np.max(dy)
        outputs['A_cs'] = np.trapz(dy.flatten(), x.flatten())


class AirfoilOptProblem(om.Problem):

    def __init__(self, n_a_u, n_a_l, seed=None):
        if rank == 0:
            if seed is None:
                seed = int(SystemRandom().random() * (2 ** 31 - 1))
            print(f'SimpleGADriver_seed: {seed}')
            os.environ['SimpleGADriver_seed'] = str(seed)

        # A_u_lower = np.array([0.15] + (n_a_u - 1) * [0.])
        # A_u_upper = np.array(n_a_u * [0.6])
        # A_l_lower = np.array(n_a_l * [-0.6])
        # A_l_upper = np.array([-0.1] + (n_a_l - 2) * [0.1] + [0.35])

        A_u_lower = np.zeros(n_a_u)
        A_u_upper = np.ones(n_a_u)
        A_l_lower = -np.ones(n_a_l)
        A_l_upper = np.ones(n_a_l)

        A_u, A_l = fit_coords(n_a_u, n_a_l)

        ivc = om.IndepVarComp()
        ivc.add_output('A_u', val=A_u)
        ivc.add_output('A_l', val=A_l)
        ivc.add_output('Re', val=1e6)
        ivc.add_output('M', val=0.)
        ivc.add_output('Cl_des', val=1.0)
        ivc.add_output('Cl_Cd_0', val=1.)
        ivc.add_output('t_c_0', val=0.0)
        ivc.add_output('A_cs_0', val=0.0)

        driver = om.SimpleGADriver(bits={'A_u': 10, 'A_l': 10}, run_parallel=run_parallel, max_gen=20)

        # prob.driver = driver = om.ScipyOptimizeDriver()
        # driver.options['optimizer'] = 'SLSQP'
        # driver.options['tol'] = 1e-4
        # driver.options['disp'] = True
        # driver.options['debug_print'] = ['objs']
        # driver.add_recorder(om.SqliteRecorder('dump.sql'))

        model = om.Group()   # model=om.Group(num_par_fd=10))
        model.add_subsystem('ivc', ivc, promotes=['*'])
        model.add_subsystem('XFoil', XFoilComp(n_u=n_a_u, n_l=n_a_l), promotes=['*'])
        model.add_subsystem('Geom', Geom(n_u=n_a_u, n_l=n_a_l), promotes=['*'])
        model.add_subsystem('F', om.ExecComp('obj = Cl_Cd_0 * Cd / Cl_des',
                                             obj=1, Cl_Cd_0=1, Cd=1., Cl_des=1.), promotes=['*'])
        model.add_subsystem('G1', om.ExecComp('g1 = 1 - t_c / t_c_0', g1=0., t_c=1., t_c_0=1.), promotes=['*'])
        model.add_subsystem('G2', om.ExecComp('g2 = 1 - A_cs / A_cs_0', g2=0, A_cs=1., A_cs_0=1.), promotes=['*'])

        model.add_design_var('A_u', lower=A_u_lower, upper=A_u_upper)
        model.add_design_var('A_l', lower=A_l_lower, upper=A_l_upper)
        model.add_objective('obj')
        model.add_constraint('g1', upper=0.)
        model.add_constraint('g2', upper=0.)

        model.approx_totals(method='fd', step=1e-5)  # method='fd', step=1e-2)

        super().__init__(model=model, driver=driver)

    def __repr__(self):
        s = ''
        s += f'Obj: {self["obj"][0]:6.4f}, L/D: {self["Cl_des"][0] / self["Cd"][0]:6.2f}, \n'
        s += f'A_u: {np.array2string(self["A_u"], formatter=formatter, separator=", ")}, \n'
        s += f'A_l: {np.array2string(self["A_l"], formatter=formatter, separator=", ")}'
        return s


def get_problem(n_a_u, n_a_l, seed=None):
    prob = AirfoilOptProblem(n_a_u, n_a_l, seed)
    prob.setup()

    if rank == 0:
        prob.set_solver_print(2)
    else:
        prob.set_solver_print(-1)

    return prob


def print_problem(prob, dt):
    print(prob.__repr__())
    print(f'Took {dt} seconds.')


def analyze(prob):
    t0 = time.time()
    prob.run_model()
    dt = time.time() - t0

    prob['Cl_Cd_0'] = prob['Cl_des'] / prob['Cd']
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
        print('Optimized:')
        print_problem(prob, dt)

    return prob


def get_coords(prob):
    x = np.reshape(cosspace(0, 1), (-1, 1))
    y_u = cst(x, prob['A_u'])
    y_l = cst(x, prob['A_l'])
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
        plot(coords)


def main():
    prob = get_problem(3, 3)
    analyze(prob)
    optimize(prob)
    if rank == 0:
        post_process(prob)


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