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

# Ensure MPI is defined
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# Detect whether the script is being run under MPI and grab the rank
if not MPI:
    run_parallel = False
    rank = 0
else:
    run_parallel = True
    rank = MPI.COMM_WORLD.rank

# Numpy string formatters
array_formatter = {'float_kind': '{: 7.4f}'.format}

# Reference airfoil coordinates
coords_file = 'naca0012.dat'
coords_ref = np.loadtxt(coords_file, skiprows=1)

# Reference airfoil coordinates split between upper and lower surfaces
i_0_ref = np.argmin(coords_ref[:, 0])
coords_ref_u = np.flipud(coords_ref[:i_0_ref + 1, :])
coords_ref_l = coords_ref[i_0_ref:, :]
t_te_ref = coords_ref_u[-1, 1] - coords_ref_l[-1, 1]

# Upper and lower reference airfoil y-coordinates sampled at identical x-coordinates
x_ref = coords_ref_u[:, 0]
y_u_ref = coords_ref_u[:, 1]
y_l_ref = np.interp(x_ref, coords_ref_l[:, 0], coords_ref_l[:, 1])


def coords2cst(x, y_u, y_l, n_c, n_t):
    """
    Convert airfoil upper/lower curve coordinates to camber line/thickness distribution CST coefficients.

    Parameters
    ----------
    x : array_like
        X-Coordinates
    y_u, y_l : array_like
        Y-Coordinates of the upper and lower curves, respectively
    n_c, n_t : int
        Number of CST coefficients to use for the camber line and thickness distribution of the airfoil

    Returns
    -------
    a_c, a_t : np.ndarray
        CST coefficients describing the camber line and thickness distribution of the airfoil
    t_te : float
        Airfoil trailing edge thickness
    """
    y_c = (y_u + y_l) / 2
    t = y_u - y_l

    a_c, _ = fit(x, y_c, n_c, delta=(0., 0.), n1=1)
    a_t, t_te = fit(x, t, n_t)

    return a_c, a_t, t_te[1]


def cst2coords(a_c, a_t, t_te, n_coords):
    """
    Convert airfoil camber line/thickness distribution CST coefficients to upper/lower curve coordinates.

    Parameters
    ----------
    a_c, a_t : array_like
        CST coefficients describing the camber line and thickness distribution of the airfoil
    t_te : float
        Airfoil trailing edge thickness
    n_coords : int
        Number of x-coordinates to use

    Returns
    -------
    x : np.ndarray
        Airfoil x-coordinates
    y_u, y_l : np.ndarray
        Airfoil upper and lower curves y-coordinates
    y_c, t : np.ndarray
        Airfoil camber line and thickness distribution
    """
    x = cosspace(0, 1, n_coords)
    y_c = cst(x, a_c, n1=1)
    t = cst(x, a_t, delta=(0, t_te))

    y_u = y_c + t / 2
    y_l = y_c - t / 2
    return x, y_u, y_l, y_c, t


def xfoil_worker(xf, cl_spec, delta=None, n=0):
    """
    Try to operate the given XFoil instance at a specified lift coefficient.

    Parameters
    ----------
    xf : XFoil
        Instance of XFoil class with Airfoil already specified
    cl_spec : float
        Lift coefficient
    delta : float, optional
        Increment lift coefficient for retrying evaluation if initial evaluation fails. None by default.
    n : int, optional
        Number of points at which to retry if initial evaluation fails. 0 by default.

    Returns
    -------
    cd : float or np.nan
        Drag coefficient or nan if analysis did not complete successfully
    """
    if n < 0:
        raise ValueError('n needs to be larger than zero.')

    _, cd, _, _ = xf.cl(cl_spec)
    if not np.isnan(cd):
        return cd
    elif delta is None or delta < 0.01 or n == 0:
        return cd
    else:
        for i in range(n):
            cl = cl_spec + ((-1.) ** float(i)) * np.ceil(float(i + 1) / 2) * delta
            _, cd, _, _ = xf.cl(cl)
            if not np.isnan(cd):
                _, cd, _, _ = xf.cl(cl_spec)
                return cd
    return cd


def analyze_airfoil(x, y_u, y_l, cl, rey, mach=0, xf=None, pool=None, show_output=False):
    """
    Analyze an airfoil at a given lift coefficient for given Reynolds and Mach numbers using XFoil.

    Parameters
    ----------
    x : array_like
        Airfoil x-coordinates
    y_u, y_l : array_like
        Airfoil upper and lower curve y-coordinates
    cl : float
        Target lift coefficient
    rey, mach : float
        Reynolds and Mach numbers
    xf : XFoil, optional
        An instance of the XFoil class to use to perform the analysis. Will be created if not given
    pool : multiprocessing.ThreadPool, optional
        An instance of the multiprocessing.Threadpool class used to run the xfoil_worker. Will be created if not given
    show_output : bool, optional
        If True, a debug string will be printed after analyses. False by default.

    Returns
    -------
    cd : float or np.nan
        The drag coefficient of the airfoil at the specified conditions, or nan if XFoil did not run successfully
    """
    # If the lower and upper curves swap, this is a bad, self-intersecting airfoil. Return 1e27 immediately.
    if np.any(y_l > y_u):
        return np.nan
    else:
        clean_xf = False
        if xf is None:
            xf = XFoil()
            xf.print = show_output
            clean_xf = True

        clean_pool = False
        if pool is None:
            pool = ThreadPool(processes=1)
            clean_pool = True

        xf.airfoil = Airfoil(x=np.concatenate((x[-1:0:-1], x)), y=np.concatenate((y_u[-1:0:-1], y_l)))
        # xf.repanel(n_nodes=300, cv_par=2.0, cte_ratio=0.5)
        xf.repanel(n_nodes=240)
        xf.Re = rey
        xf.M = mach
        xf.max_iter = 200

        cd = np.nan
        future = pool.apply_async(xfoil_worker, args=(xf, cl, 0.05, 1))
        try:
            cd = future.get(timeout=2.)
            xf.reset_bls()
        except TimeoutError:
            pass

    if clean_xf:
        del xf
    if clean_pool:
        del pool

    return cd


class AirfoilComponent(om.ExplicitComponent):
    """
    Basic Aifoil specified by CST coefficients for its camber line and thickness distribution and a TE thickness.
    """

    def initialize(self):
        self.options.declare('n_c', default=6, types=int)
        self.options.declare('n_t', default=6, types=int)

        self.options.declare('n_coords', default=100, types=int)

    def setup(self):
        # Number of CST coefficients
        n_c = self.options['n_c']
        n_t = self.options['n_t']

        # Inputs
        self.add_input('a_c', shape=n_c)
        self.add_input('a_t', shape=n_t)
        self.add_input('t_te', shape=1)

    def compute_coords(self, inputs):
        """
        Compute airfoil coordinates from the set of OpenMDAO inputs.
        """
        return cst2coords(inputs['a_c'], inputs['a_t'], inputs['t_te'][0], self.options['n_coords'])


class XFoilComp(AirfoilComponent):
    """
    Computes the drag coefficient of an airfoil at a given lift coefficient, Reynolds nr., and Mach nr.
    """

    def initialize(self):
        super().initialize()
        self.options.declare('print', default=False, types=bool)

        xf = XFoil()
        xf.print = False
        self.options.declare('_xf', default=xf, types=XFoil, allow_none=True)
        self.options.declare('_pool', default=ThreadPool(processes=1), types=ThreadPool, allow_none=True)

    def setup(self):
        super().setup()

        # Inputs
        self.add_input('Cl_des', val=1.)
        self.add_input('Re', val=1e6)
        self.add_input('M', val=0.)

        # Output
        self.add_output('Cd', val=1.)

    def compute(self, inputs, outputs, **kwargs):
        x, y_u, y_l, _, _ = self.compute_coords(inputs)

        t0 = time.time()
        cd = analyze_airfoil(x, y_u, y_l, inputs['Cl_des'][0], inputs['Re'][0], inputs['M'][0],
                             self.options['_xf'], self.options['_pool'], self.options['print'])
        dt = time.time() - t0

        outputs['Cd'] = cd if not np.isnan(cd) else 1e27

        if self.options['print']:
            print(
                f'{rank:02d} :: ' +
                'a_c: {}, '.format(np.array2string(inputs['a_c'], separator=', ', formatter=array_formatter)) +
                'a_t: {}, '.format(np.array2string(inputs['a_t'], separator=', ', formatter=array_formatter)) +
                f't_te: {inputs["t_te"][0]: 6.4f}, ' +
                f'C_d: {cd: 7.4f}, dt: {dt:6.3f}'
            )


class Geom(AirfoilComponent):
    """
    Computes the thickness-over-chord ratio and cross-sectional area of an airfoil.
    """

    def setup(self):
        super().setup()
        # Outputs
        self.add_output('t_c', val=0.)
        self.add_output('A_cs', val=0.)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        x, _, _,  _, t = self.compute_coords(inputs)
        outputs['t_c'] = np.max(t)
        outputs['A_cs'] = np.trapz(t, x)


class AfOptModel(om.Group):
    """
    Airfoil shape optimization using XFoil.
    """

    def initialize(self):
        self.options.declare('n_c', default=6, types=int)
        self.options.declare('n_t', default=6, types=int)
        self.options.declare('fix_te', default=True, types=bool)

        self.options.declare('n_coords', default=100, types=int)

    def setup(self):
        # Number of CST coefficients
        n_c = self.options['n_c']
        n_t = self.options['n_t']

        # Design variable bounds
        a_c_lower = -np.ones(n_c)
        a_c_upper = np.ones(n_c)
        a_t_lower = 0.01 * np.ones(n_t)
        a_t_upper = 0.6 * np.ones(n_t)
        t_te_lower = 0.
        t_te_upper = 0.1

        # Independent variables
        ivc = om.IndepVarComp()
        ivc.add_output('a_c', val=np.zeros(n_c))
        ivc.add_output('a_t', val=np.zeros(n_t))
        ivc.add_output('t_te', val=0.)
        ivc.add_output('Re', val=1e6)
        ivc.add_output('M', val=0.)
        ivc.add_output('Cl_des', val=1.)
        ivc.add_output('Cd_0', val=1.)
        ivc.add_output('t_c_0', val=1.)
        ivc.add_output('A_cs_0', val=1.)

        # Sub-systems
        self.add_subsystem('ivc', ivc, promotes=['*'])
        self.add_subsystem('XFoil', XFoilComp(n_c=n_c, n_t=n_t), promotes=['*'])
        self.add_subsystem('Geom', Geom(n_c=n_c, n_t=n_t), promotes=['*'])
        self.add_subsystem('F', om.ExecComp('obj = Cd / Cd_0', obj=1, Cd=1., Cd_0=1.), promotes=['*'])
        self.add_subsystem('G1', om.ExecComp('g1 = 1 - t_c / t_c_0', g1=0., t_c=1., t_c_0=1.), promotes=['*'])
        self.add_subsystem('G2', om.ExecComp('g2 = 1 - A_cs / A_cs_0', g2=0, A_cs=1., A_cs_0=1.), promotes=['*'])

        # Design variables
        self.add_design_var('a_c', lower=a_c_lower, upper=a_c_upper)
        self.add_design_var('a_t', lower=a_t_lower, upper=a_t_upper)

        if not self.options['fix_te']:
            self.add_design_var('t_te', lower=t_te_lower, upper=t_te_upper)

        # Objective and constraints
        self.add_objective('obj')

        self.add_constraint('g1', upper=0.)
        self.add_constraint('g2', upper=0.)

    def __repr__(self):
        outputs = dict(self.list_outputs(out_stream=None))
        s = ''
        s += f'Obj: {outputs["F.obj"]["value"][0]:6.4f}, ' \
             f'C_l_des: {outputs["ivc.Cl_des"]["value"][0]:6.4f}, ' \
             f'C_d: {outputs["XFoil.Cd"]["value"][0]:6.4f}, \n'
        s += f'a_c: {np.array2string(outputs["ivc.a_c"]["value"], formatter=array_formatter, separator=", ")}, \n'
        s += f'a_t: {np.array2string(outputs["ivc.a_t"]["value"], formatter=array_formatter, separator=", ")}, \n'
        s += f't_te: {outputs["ivc.t_te"]["value"][0]: 7.4f}'
        return s


def get_problem(n_c, n_t, fix_te=True, seed=None):
    """
    Construct an OpenMDAO Problem which minimizes the drag coefficient of an airfoil for a given lift coefficient.

    Parameters
    ----------
    n_c, n_t : int
        Number of CST coefficients for the chord line and thickness distribution, respectively
    fix_te : bool, optional
        True if the trailing edge thickness should be fixed. True by default
    seed : int, optional
        Seed to use for the random number generator which creates an initial population for the genetic algorithm

    Returns
    -------
    openmdao.api.Problem
    """
    # Set a starting seed for the random number generated if given
    if rank == 0:
        if seed is None:
            seed = int(SystemRandom().random() * (2 ** 31 - 1))
        print(f'SimpleGADriver_seed: {seed}')
        os.environ['SimpleGADriver_seed'] = str(seed)

    # Construct the OpenMDAO Problem
    prob = om.Problem()
    prob.model = AfOptModel(n_c=n_c, n_t=n_t, fix_te=fix_te)
    prob.driver = om.SimpleGADriver(bits={'a_c': 31, 'a_t': 31}, run_parallel=run_parallel, max_gen=300)
    prob.setup()

    # Set the reference airfoil as initial conditions
    prob['a_c'], prob['a_t'], prob['t_te'] = coords2cst(x_ref, y_u_ref, y_l_ref, n_c, n_t)

    return prob


def print_problem(prob, dt):
    """
    Print a representation of the state of the optimization problem.

    Parameters
    ----------
    prob : openmdao.api.Problem
        Airfoil optimization problem
    dt : float
        Time in seconds elapsed since last evaluation
    """
    print(prob.model.__repr__())
    print(f'Time elapsed: {timedelta(seconds=dt)}')


def analyze(prob, initial=True):
    """
    Simply analyze the airfoil once.

    Parameters
    ----------
    prob : openmdao.api.Problem
        Airfoil optimization problem
    initial : bool, optional
        True if initial references values should be set based on this analysis. True by default.

    Returns
    -------
    openmdao.api.Problem
    """
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
    """
    Optimize the airfoil optimization problem.

    Parameters
    ----------
    prob : openmdao.api.Problem
        Airfoil optimization problem.

    Returns
    -------
    openmdao.api.Problem
    """
    t0 = time.time()
    prob.run_driver()
    dt = time.time() - t0

    if rank == 0:
        print_problem(prob, dt)

    return prob


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
    x, y_u, y_l, _, _ = cst2coords(prob['a_c'], prob['a_t'], prob['t_te'], 100)
    x = np.reshape(x, (-1, 1))
    y_u = np.reshape(y_u, (-1, 1))
    y_l = np.reshape(y_l, (-1, 1))

    coords_u = np.concatenate((x, y_u), axis=1)
    coords_l = np.concatenate((x, y_l), axis=1)
    coords = np.concatenate((np.flip(coords_u[1:], axis=0), coords_l))

    return coords


def plot(prob, show_legend=False, show_title=True):
    """
    Plot the airfoil represented by the current state of the airfoil optimization problem.

    Parameters
    ----------
    prob : openmdao.api.Problem
        Airfoil optimization problem
    show_legend : bool, optional
        True if a legend should be shown. False by default
    show_title : bool, optional
        True if a title should be shown based on the current state of the problem. True by default

    Returns
    -------

    """
    import matplotlib.pyplot as plt
    x, y_u, y_l, y_c, _ = cst2coords(prob['a_c'], prob['a_t'], prob['t_te'], 100)
    plt.plot(coords_ref[:, 0], coords_ref[:, 1], 'k',
             x, y_u, 'r', x, y_l, 'r', x, y_c, 'r--')
    plt.axis('scaled')
    if show_legend:
        plt.legend(['Original', 'Optimized'])
    if show_title:
        plt.title(prob.model)
    plt.show()


def write(prob, filename='optimized.dat'):
    """
    Write airfoil coordinates represented by the current state of the airfoil optimization problem to a file

    Parameters
    ----------
    prob : openmdao.api.Problem
        Airfoil optimization problem
    filename : str, optional
        Filename. 'optimized.dat' by default
    """
    coords = get_coords(prob)
    fmt_str = 2 * ('{: >' + str(6 + 1) + '.' + str(6) + 'f} ') + '\n'
    with open(filename, 'w') as f:
        for i in range(coords.shape[0]):
            f.write(fmt_str.format(coords[i, 0], coords[i, 1]))


def main():
    """
    Create, analyze, optimize airfoil, and write optimized coordinates to a file. Then clean the problem up and exit.
    """
    prob = get_problem(6, 6)
    analyze(prob)
    optimize(prob)
    if rank == 0:
        write(prob)

    prob.cleanup()
    del prob

    import sys
    sys.exit(0)


if __name__ == '__main__':
    main()
    exit(0)
