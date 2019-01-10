import numpy as np
import os
import re
import time

from openmdao.api import ExternalCodeComp, Problem, IndepVarComp, ScipyOptimizeDriver, ExecComp, \
    SqliteRecorder, ExplicitComponent
from openmdao.core.analysis_error import AnalysisError

from cst import cst, fit
from util import cosspace, get_random_key

formatter = {'float_kind': lambda x: '{: 10.8f}'.format(x)}

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


class XFoil(ExternalCodeComp):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stdout = open(os.devnull, 'w')

    def initialize(self):
        self.options.declare('n_u', default=6, types=int)
        self.options.declare('n_l', default=6, types=int)

        self.options.declare('n_coords', default=100, types=int)

        self.options.declare('id', types=str)

        self.options.declare('plot', default=False, types=bool)

    @property
    def coordinate_file(self):
        return '0{}.dat'.format(self.options['id'])

    @property
    def command_file(self):
        return '0{}.cmd'.format(self.options['id'])

    @property
    def polar_file(self):
        return '0{}.pol'.format(self.options['id'])

    def setup(self):
        n_u = self.options['n_u']
        n_l = self.options['n_l']

        self.add_input('A_u', shape=n_u)
        self.add_input('A_l', shape=n_l)

        self.add_input('Cl_des', val=1.)
        self.add_input('Re', val=1e6)
        self.add_input('M', val=0.)

        self.add_output('Cd', val=1.)

        self.options['id'] = get_random_key()
        self.options['external_input_files'] = [self.coordinate_file, self.command_file]
        self.options['external_output_files'] = [self.polar_file,]
        self.options['command'] = ['xfoil.exe', '<', self.command_file]
        self.options['allowed_return_codes'] = [1]

        self.declare_partials('*', '*', method='fd')
        # self.declare_partials('Cd', ['Cl_des', 'Re', 'M'], dependent=False)

    def compute(self, inputs, outputs):
        n_coords = self.options['n_coords']

        x = np.reshape(cosspace(0, 1, n_coords), (-1, 1))
        y_u = cst(x, inputs['A_u'])
        y_l = cst(x, inputs['A_l'])

        coords_u = np.concatenate((x, y_u), axis=1)
        coords_l = np.concatenate((x, y_l), axis=1)
        coords = np.concatenate((np.flip(coords_u[1:], axis=0), coords_l))

        Re = inputs['Re'][0]
        M = inputs['M'][0]
        Cl_des = inputs['Cl_des'][0]

        # Write coordinates file
        fmt_str = 2 * ('{: >' + str(6 + 1) + '.' + str(6) + 'f} ') + '\n'
        with open(self.coordinate_file, 'w') as f:
            for i in range(coords.shape[0]):
                f.write(fmt_str.format(coords[i, 0], coords[i, 1]))

        outputs['Cd'] = 10
        dcl = 0.
        ddcl = 0.01
        for i in range(4):
            # Write the Xfoil command file
            with open(self.command_file, 'w') as f:
                if not self.options['plot']:
                    f.write('plop\ng\n\n')

                f.write('load {}\n\n'.format(self.coordinate_file))
                f.write('mdes\nfilt\n\n')
                f.write('pane\n')
                f.write('oper\n')

                if Re > 0.:
                    f.write('visc {}\n'.format(Re))

                if M > 0.:
                    f.write('m {}\n'.format(M))

                f.write('iter 200\n')
                f.write('pacc\n\n\n')

                f.write('cseq {} {} {}\n'.format(Cl_des - dcl/2, Cl_des + dcl/2, dcl))
                f.write('pwrt\n{}\ny\n\n'.format(self.polar_file))
                f.write('quit\n')

            # increment dcl
            dcl += ddcl

            # Run Xfoil
            try:
                super().compute([], [])
            except (AnalysisError, RuntimeError):
                done = False
                while not done:
                    try:
                        os.remove(self.command_file)
                        done = True
                    except Exception:
                        pass
                    time.sleep(0.001)
                continue

            # Read polar
            if os.path.isfile(self.polar_file):
                # Read the polar file
                with open(self.polar_file, 'r') as f:
                    lines = f.readlines()[12:]

                if lines:
                    # Parse the data from the read lines
                    polar = np.zeros((len(lines), len(re.split(r'\s+', lines[0].strip()))))
                    for i, line in enumerate(lines):
                        polar[i, :] = np.fromstring(line, dtype=float, count=polar.shape[1], sep=' ')

                    os.remove(self.polar_file)
                    avgs = np.average(polar, 0)
                    # print('{: 6.3f} {: 5.3f} {: 5.1f} {: 5.2f}'.format(avgs[0], avgs[1], avgs[2] * 1e4, avgs[1]/avgs[2])
                    #       + np.array2string(inputs['A_u'], formatter=formatter)[1:-2] + ' '
                    #       + np.array2string(inputs['A_l'], formatter=formatter)[1:-2])
                    outputs['Cd'] = avgs[2]
                    break

        os.remove(self.coordinate_file)


class Geom(ExplicitComponent):

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


def naca(spec):
    coords_file = 'naca{}.dat'.format(spec)

    # Write the Xfoil command file
    with open('temp', 'w') as f:
        f.write('naca {}\n'.format(spec))
        f.write('save {}\ny\n'.format(coords_file))
        f.write('quit\n')

    os.system('xfoil.exe < temp')
    time.sleep(1)

    if os.path.isfile(coords_file):
        with open(coords_file, 'r') as f:
            lines = f.readlines()[2:]

        coords = np.zeros((len(lines), 2))
        for i in range(len(lines)):
            coords[i, :] = np.fromstring(lines[i], dtype=float, count=2, sep=' ')

        i_0 = np.argmin(coords[:, 0])
        coords_u = coords[:i_0 + 1, :]
        coords_l = coords[i_0:, :]

        A_u, _ = fit(coords_u[:, 0], coords_u[:, 1], n_a_u, delta=(0., 0.))
        A_l, _ = fit(coords_l[:, 0], coords_l[:, 1], n_a_l, delta=(0., 0.))
        return A_u, A_l, coords
    return None


if __name__ == '__main__':
    A_u, A_l, coords_orig = naca('0012')

    ivc = IndepVarComp()
    ivc.add_output('A_u', val=A_u)
    ivc.add_output('A_l', val=A_l)
    ivc.add_output('Re', val=1e6)
    ivc.add_output('M', val=0.)
    ivc.add_output('Cl_des', val=1.0)
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
    prob.model.add_subsystem('XFoil', XFoil(n_u=n_a_u, n_l=n_a_l, plot=False, timeout=10), promotes=['*'])
    prob.model.add_subsystem('Geom', Geom(n_u=n_a_u, n_l=n_a_l), promotes=['*'])
    prob.model.add_subsystem('F', ExecComp('obj = Cl_Cd_0 * Cd / Cl_des', obj=1, Cl_Cd_0=1, Cd=1., Cl_des=1.),
                             promotes=['*'])
    prob.model.add_subsystem('G1', ExecComp('g1 = 1 - t_c / t_c_0', g1=0., t_c=1., t_c_0=1.), promotes=['*'])
    prob.model.add_subsystem('G2', ExecComp('g2 = 1 - A_cs / A_cs_0', g2=0, A_cs=1., A_cs_0=1.), promotes=['*'])

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
    exit(0)
