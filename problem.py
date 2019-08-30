import numpy as np
import openmdao.api as om
import time

from cst import cst, fit
from util import cosspace

from xfoil import XFoil
from xfoil.model import Airfoil

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

    def compute(self, inputs, outputs, **kwargs):
        n_coords = self.options['n_coords']
        xf = self.options['xfoil']

        x = cosspace(0, 1, n_coords)
        y_u = cst(x, inputs['A_u'])
        y_l = cst(x, inputs['A_l'])

        xf.airfoil = Airfoil(x=np.concatenate((x[-1:0:-1], x)), y=np.concatenate((y_u[-1:0:-1], y_l)))
        xf.filter()
        xf.repanel()
        xf.Re = inputs['Re'][0]
        xf.M = inputs['M'][0]
        xf.max_iter = 200

        _, cd, _, _ = xf.cl(inputs['Cl_des'][0])
        if np.isnan(cd):
            xf.reset_bls()
            _, cl, cd, _, _ = xf.cseq(inputs['Cl_des'][0] - 0.05, inputs['Cl_des'][0] + 0.055, 0.005)
            outputs['Cd'] = np.interp(inputs['Cl_des'][0], cl, cd)
        else:
            outputs['Cd'] = cd


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


if __name__ == '__main__':
    n_a_u = 5
    n_a_l = 5

    # A_u_lower = np.array([0.15] + (n_a_u - 1) * [0.])
    # A_u_upper = np.array(n_a_u * [0.6])
    # A_l_lower = np.array(n_a_l * [-0.6])
    # A_l_upper = np.array([-0.1] + (n_a_l - 2) * [0.1] + [0.35])

    A_u_lower = np.zeros(n_a_u)
    A_u_upper = np.ones(n_a_u)
    A_l_lower = -np.ones(n_a_l)
    A_l_upper = np.ones(n_a_l)

    t0 = time.time()
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

    prob = om.Problem()
    prob.driver = driver = om.ScipyOptimizeDriver()
    driver.options['optimizer'] = 'SLSQP'
    driver.options['tol'] = 1e-4
    driver.options['disp'] = True
    driver.options['debug_print'] = ['objs']
    driver.add_recorder(om.SqliteRecorder('dump.sql'))

    prob.set_solver_print(2)

    prob.model.add_subsystem('ivc', ivc, promotes=['*'])
    prob.model.add_subsystem('XFoil', XFoilComp(n_u=n_a_u, n_l=n_a_l, num_par_fd=n_a_u + n_a_l), promotes=['*'])
    prob.model.add_subsystem('Geom', Geom(n_u=n_a_u, n_l=n_a_l), promotes=['*'])
    prob.model.add_subsystem('F', om.ExecComp('obj = Cl_Cd_0 * Cd / Cl_des', obj=1, Cl_Cd_0=1, Cd=1., Cl_des=1.),
                             promotes=['*'])
    prob.model.add_subsystem('G1', om.ExecComp('g1 = 1 - t_c / t_c_0', g1=0., t_c=1., t_c_0=1.), promotes=['*'])
    prob.model.add_subsystem('G2', om.ExecComp('g2 = 1 - A_cs / A_cs_0', g2=0, A_cs=1., A_cs_0=1.), promotes=['*'])

    prob.model.add_design_var('A_u', lower=A_u_lower, upper=A_u_upper)
    prob.model.add_design_var('A_l', lower=A_l_lower, upper=A_l_upper)
    prob.model.add_objective('obj')
    prob.model.add_constraint('g1', upper=0.)
    prob.model.add_constraint('g2', upper=0.)

    prob.model.approx_totals(method='fd', step=1e-2)
    prob.setup()

    # Run for initial point
    prob.run_model()
    prob['Cl_Cd_0'] = prob['Cl_des'] / prob['Cd']
    prob['t_c_0'] = prob['t_c']
    prob['A_cs_0'] = prob['A_cs']
    print('Initial point:')
    print('A_u: ' + np.array2string(prob['A_u'], formatter=formatter)[1:-2])
    print('A_l: ' + np.array2string(prob['A_l'], formatter=formatter)[1:-2])
    print('t/c: {: 8.3f}, A_cs: {: 8.3f}'.format(prob['t_c'][0], prob['A_cs'][0]))
    print('Cl/Cd: {}'.format(prob['Cl_des'] / prob['Cd']))

    # Optimize
    prob.run_driver()
    print('Optimized:')
    print('A_u: ' + np.array2string(prob['A_u'], formatter=formatter)[1:-2])
    print('A_l: ' + np.array2string(prob['A_l'], formatter=formatter)[1:-2])
    print('Cl/Cd: {}'.format(prob['Cl_des'] / prob['Cd']))

    # Write optimized geometry to dat file
    x = np.reshape(cosspace(0, 1), (-1, 1))
    y_u = cst(x, prob['A_u'])
    y_l = cst(x, prob['A_l'])
    coords_u = np.concatenate((x, y_u), axis=1)
    coords_l = np.concatenate((x, y_l), axis=1)
    coords = np.concatenate((np.flip(coords_u[1:], axis=0), coords_l))

    fmt_str = 2 * ('{: >' + str(6 + 1) + '.' + str(6) + 'f} ') + '\n'
    with open('optimized.dat', 'w') as f:
        for i in range(coords.shape[0]):
            f.write(fmt_str.format(coords[i, 0], coords[i, 1]))

    import matplotlib.pyplot as plt
    plt.plot(coords_orig[:, 0], coords_orig[:, 1], 'k', coords[:, 0], coords[:, 1], 'r')
    plt.axis('equal')
    plt.legend(['Original', 'Optimized'])
    plt.show()

    # Cleanup the problem and exit
    prob.cleanup()

    print('Took {} seconds'.format(time.time() - t0))
    exit(0)
