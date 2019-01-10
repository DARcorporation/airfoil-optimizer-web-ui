from openmdao.api import CaseReader
import matplotlib.pyplot as plt
import numpy as np

from util import cosspace
from cst import cst
from problem import A_u_lower, A_u_upper, A_l_lower, A_l_upper

cr = CaseReader('dump.sql')
driver_cases = cr.list_cases('driver')
case = cr.get_case(driver_cases[-1])
objectives = case.get_objectives()
design_vars = case.get_design_vars()

orig = cr.get_case(driver_cases[0])
opti = cr.get_case(driver_cases[-1])

orig_desvars = orig.get_design_vars()
opti_desvars = opti.get_design_vars()

x = np.reshape(cosspace(0, 1), (-1, 1))

y_u_orig = cst(x, cr.get_case(orig_desvars['A_u']))
y_l_orig = cst(x, cr.get_case(orig_desvars['A_l']))

y_u_opti = cst(x, opti_desvars['A_u'])
y_l_opti = cst(x, opti_desvars['A_l'])

coords_u_orig = np.concatenate((x, y_u_orig), axis=1)
coords_l_orig = np.concatenate((x, y_l_orig), axis=1)
coords_orig = np.concatenate((np.flip(coords_u_orig[1:], axis=0), coords_l_orig))

x = np.reshape(cosspace(0, 1), (-1, 1))

coords_u_opti = np.concatenate((x, y_u_opti), axis=1)
coords_l_opti = np.concatenate((x, y_l_opti), axis=1)
coords_opti = np.concatenate((np.flip(coords_u_opti[1:], axis=0), coords_l_opti))

x_bound = cosspace(0, 1, 20)
y_u_lb = cst(x_bound, A_u_lower)
y_u_ub = cst(x_bound, A_u_upper)
y_l_lb = cst(x_bound, A_l_lower)
y_l_ub = cst(x_bound, A_l_upper)

fig = plt.figure()
ax = plt.subplot(111)
ax.axis('equal')
ax.plot(coords_orig[:, 0], coords_orig[:, 1], 'k:', coords_opti[:, 0], coords_opti[:, 1], 'r',
        x_bound, y_u_lb, 'c--', x_bound, y_u_ub, 'c--', x_bound, y_l_lb, 'm--', x_bound, y_l_ub, 'm--')
plt.legend(['original'.format(), 'optimized'.format()])
plt.show()

