import numpy as np
import os
import time

from openmdao.api import Problem, IndepVarComp, ScipyOptimizeDriver, ExecComp, SqliteRecorder, ExplicitComponent

from cst import cst, fit
from util import cosspace

from xfoil import XFoil
from xfoil.model import Airfoil

formatter = {'float_kind': lambda x: '{: 10.8f}'.format(x)}

n_a_c = 3
n_a_t = 3

A_c_lower = -np.ones(n_a_c)
A_c_upper = np.ones(n_a_c)
A_t_lower = 0.01 * np.ones(n_a_t)
A_t_upper = 0.50 * np.ones(n_a_t)


class XFoilComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('n_c', default=6, types=int)
        self.options.declare('n_t', default=6, types=int)

        self.options.declare('n_coords', default=100, types=int)

        xf = XFoil()
        xf.print = False
        self.options.declare('xfoil', default=xf, types=XFoil)

    def setup(self):
        n_c = self.options['n_c']
        n_t = self.options['n_t']

        self.add_input('A_c', shape=n_c)
        self.add_input('A_t', shape=n_t)

        self.add_input('Cl_des', val=1.)
        self.add_input('Re', val=1e6)
        self.add_input('M', val=0.)

        self.add_output('Cd', val=1.)

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs, **kwargs):
        n_coords = self.options['n_coords']
        xf = self.options['xfoil']

        x = cosspace(0, 1, n_coords)
        y_c = cst(x, inputs['A_c'], n1=1, n2=1)
        y_t = cst(x, inputs['A_t'], delta=(0, 1e-4), n1=0.5, n2=0.5)

        y_u = y_c + y_t
        y_l = y_c - y_t

        # import matplotlib.pyplot as plt
        # plt.plot(x, y_c + y_t, 'k', x, y_c - y_t, 'k')
        # plt.axis('equal')
        # plt.show()

        xf.airfoil = Airfoil(x=np.concatenate((x[-1:0:-1], x)), y=np.concatenate((y_u[-1:0:-1], y_l)))
        # xf.filter()
        # xf.repanel()
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


class Geom(ExplicitComponent):

    def initialize(self):
        self.options.declare('n_c', default=6, types=int)
        self.options.declare('n_t', default=6, types=int)

        self.options.declare('n_coords', default=100, types=int)

    def setup(self):
        n_c = self.options['n_c']
        n_t = self.options['n_t']

        self.add_input('A_c', shape=n_c)
        self.add_input('A_t', shape=n_t)

        self.add_output('t_c', val=0.)
        self.add_output('A_cs', val=0.)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        n_coords = self.options['n_coords']

        x = np.reshape(cosspace(0, 1, n_coords), (-1, 1))
        y_t = cst(x, inputs['A_t'])

        outputs['t_c'] = np.max(y_t)
        outputs['A_cs'] = np.trapz(y_t.flatten(), x.flatten())


def naca(spec):
    xf = XFoil()
    xf.naca(spec)
    coords = xf.airfoil.coords

    i = np.argwhere(coords[:, 0] == 0)[0, 0]
    x_u = np.flipud(coords[:i + 1, 0])
    y_u = np.flipud(coords[:i + 1, 1])
    x_l = coords[i:, 0]
    y_l = coords[i:, 1]

    x = x_u
    y_c = (np.interp(x, x_u, y_u) + np.interp(x, x_l, y_l)) / 2
    y_t = (np.interp(x, x_u, y_u) - np.interp(x, x_l, y_l)) / 2

    A_c, _ = fit(x, y_c, n_a_c, n1=1, n2=1)
    A_t, _ = fit(x, y_t, n_a_t, delta=(0, 1e-4), n1=0.5, n2=0.5)
    return A_c, A_t, coords


if __name__ == '__main__':
    t0 = time.time()
    A_c, A_t, coords_orig = naca('0012')

    import matplotlib.pyplot as plt
    x = cosspace(0, 1)
    y_c = cst(x, A_c, n1=1, n2=1)
    y_t = cst(x, A_t, delta=(0, 1e-4), n1=0.5, n2=0.5)
    plt.plot(coords_orig[:, 0], coords_orig[:, 1], 'k.', x, y_c + y_t, 'r', x, y_c - y_t, 'r')
    plt.axis('equal')
    plt.show()

    ivc = IndepVarComp()
    ivc.add_output('A_c', val=A_c)
    ivc.add_output('A_t', val=A_t)
    ivc.add_output('Re', val=1e6)
    ivc.add_output('M', val=0.)
    ivc.add_output('Cl_des', val=0.5)
    ivc.add_output('Cl_Cd_0', val=1.)
    ivc.add_output('t_c_0', val=0.0)
    ivc.add_output('A_cs_0', val=0.0)

    prob = Problem()
    prob.driver = ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-4
    prob.driver.options['disp'] = True
    prob.driver.options['debug_print'] = ['objs']
    prob.driver.add_recorder(SqliteRecorder('dump.sql'))

    prob.set_solver_print(2)

    prob.model.add_subsystem('ivc', ivc, promotes=['*'])
    prob.model.add_subsystem('XFoil', XFoilComp(n_c=n_a_c, n_t=n_a_t), promotes=['*'])
    prob.model.add_subsystem('Geom', Geom(n_c=n_a_c, n_t=n_a_t), promotes=['*'])
    prob.model.add_subsystem('F', ExecComp('obj = Cl_Cd_0 * Cd / Cl_des', obj=1, Cl_Cd_0=1, Cd=1., Cl_des=1.),
                             promotes=['*'])
    prob.model.add_subsystem('G1', ExecComp('g1 = 1 - t_c / t_c_0', g1=0., t_c=1., t_c_0=1.), promotes=['*'])
    # prob.model.add_subsystem('G2', ExecComp('g2 = 1 - A_cs / A_cs_0', g2=0, A_cs=1., A_cs_0=1.), promotes=['*'])

    prob.model.add_design_var('A_c', lower=A_c_lower, upper=A_c_upper)
    prob.model.add_design_var('A_t', lower=A_t_lower, upper=A_t_upper)
    prob.model.add_objective('obj')
    prob.model.add_constraint('g1', upper=0.)
    # prob.model.add_constraint('g2', upper=0.)

    prob.model.approx_totals(method='fd', step=1e-2)
    prob.setup()

    # Run for initial point
    prob.run_model()
    prob['Cl_Cd_0'] = prob['Cl_des'] / prob['Cd']
    prob['t_c_0'] = prob['t_c']
    prob['A_cs_0'] = prob['A_cs']
    print('Initial point:')
    print('A_c: ' + np.array2string(prob['A_c'], formatter=formatter)[1:-2])
    print('A_t: ' + np.array2string(prob['A_t'], formatter=formatter)[1:-2])
    print('t/c: {: 8.3f}, A_cs: {: 8.3f}'.format(prob['t_c'][0], prob['A_cs'][0]))
    print('Cl/Cd: {}'.format(prob['Cl_des'] / prob['Cd']))

    # Optimize
    prob.run_driver()
    print('Optimized:')
    print('A_c: ' + np.array2string(prob['A_c'], formatter=formatter)[1:-2])
    print('A_t: ' + np.array2string(prob['A_t'], formatter=formatter)[1:-2])
    print('Cl/Cd: {}'.format(prob['Cl_des'] / prob['Cd']))

    # Write optimized geometry to dat file
    x = np.reshape(cosspace(0, 1), (-1, 1))
    y_c = cst(x, prob['A_c'])
    y_t = cst(x, prob['A_t'])

    y_u = y_c + y_t
    y_l = y_c - y_t
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
