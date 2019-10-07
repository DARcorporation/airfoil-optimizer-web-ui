#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains the OpenMDAO Model of the airfoil optimization problem.
"""
import numpy as np
import openmdao.api as om

from .components import *


class AfOptModel(om.Group):
    """
    Airfoil shape optimization using XFoil.
    """

    def initialize(self):
        self.options.declare("n_c", default=6, types=int)
        self.options.declare("n_t", default=6, types=int)
        self.options.declare("fix_te", default=True, types=bool)

        self.options.declare(
            "t_te_min", default=0.0, lower=0.0, types=float, allow_none=False
        )
        self.options.declare("t_c_min", default=0.1, types=float, allow_none=True)
        self.options.declare("A_cs_min", default=0.1, types=float, allow_none=True)
        self.options.declare("Cm_max", default=None, types=float, allow_none=True)

        self.options.declare("n_coords", default=100, types=int)

    def setup(self):
        # Number of CST coefficients
        n_c = self.options["n_c"]
        n_t = self.options["n_t"]

        # Design variable bounds
        a_c_lower = -0.25 * np.ones(n_c)
        a_c_upper = +0.25 * np.ones(n_c)
        a_t_lower = +0.01 * np.ones(n_t)
        a_t_upper = +0.20 * np.ones(n_t)
        t_te_upper = 0.1

        # Independent variables
        ivc = om.IndepVarComp()
        ivc.add_output("a_c", val=np.zeros(n_c))
        ivc.add_output("a_t", val=np.zeros(n_t))
        ivc.add_output("t_te", val=self.options["t_te_min"])
        ivc.add_output("Re", val=1e6)
        ivc.add_output("M", val=0.0)
        ivc.add_output("Cl_des", val=1.0)

        # Main sub-systems
        self.add_subsystem("ivc", ivc, promotes=["*"])
        self.add_subsystem("XFoil", XFoilComp(n_c=n_c, n_t=n_t), promotes=["*"])

        # Design variables
        self.add_design_var("a_c", lower=a_c_lower, upper=a_c_upper)
        self.add_design_var("a_t", lower=a_t_lower, upper=a_t_upper)

        if not self.options["fix_te"]:
            self.add_design_var(
                "t_te", lower=self.options["t_te_min"], upper=t_te_upper
            )

        # Objective
        self.add_objective("Cd")  # Cd

        # Constraints
        self.add_subsystem("Geom", Geom(n_c=n_c, n_t=n_t), promotes=["*"])

        if self.options["t_c_min"] is not None:
            self.add_subsystem(
                "G1",
                om.ExecComp(
                    f"g1 = 1 - t_c / {self.options['t_c_min']:15g}", g1=0.0, t_c=1.0
                ),
                promotes=["*"],
            )
            self.add_constraint("g1", upper=0.0)  # t_c >= t_c_min

        if self.options["A_cs_min"] is not None:
            self.add_subsystem(
                "G2",
                om.ExecComp(
                    f"g2 = 1 - A_cs / {self.options['A_cs_min']:15g}", g2=0, A_cs=1.0
                ),
                promotes=["*"],
            )
            self.add_constraint("g2", upper=0.0)  # A_cs >= A_cs_min

        if self.options["Cm_max"] is not None:
            self.add_subsystem(
                "G3",
                om.ExecComp(
                    f"g3 = 1 - abs(Cm) / {np.abs(self.options['Cm_max']):15g}",
                    g3=0.0,
                    Cm=1.0,
                ),
                promotes=["*"],
            )
            self.add_constraint("g3", lower=0.0)  # |Cm| <= |Cm_max|

    def __repr__(self):
        outputs = dict(self.list_outputs(out_stream=None))

        s_t_te_des = f"{outputs['ivc.t_te']['value'][0]:.4g}"
        desvar_formatter = {"float_kind": "{: 7.4f}".format}

        yaml = ""
        yaml += f"Cl: {outputs['ivc.Cl_des']['value'][0]:.4g}\n"
        yaml += f"M: {outputs['ivc.M']['value'][0]:.4g}\n"
        yaml += f"Re: {outputs['ivc.Re']['value'][0]:.4g}\n"
        yaml += (
            "" if self.options["fix_te"] else "min "
        ) + f"t_te: {self.options['t_te_min']:.4g}\n"
        if self.options["t_c_min"] is not None:
            yaml += f"t_c_min: {self.options['t_c_min']:.4g}\n"
        if self.options["A_cs_min"] is not None:
            yaml += f"A_cs_min: {self.options['A_cs_min']:.4g}\n"
        if self.options["Cm_max"] is not None:
            yaml += f"Cm_max: {self.options['Cm_max']:.4g}\n"
        yaml += f"Cd: {outputs['XFoil.Cd']['value'][0]:.4g}\n"
        yaml += f"Cm: {outputs['XFoil.Cm']['value'][0]: .4g}\n"
        yaml += f"t_c: {outputs['Geom.t_c']['value'][0]:.4g}\n"
        yaml += f"A_cs: {outputs['Geom.A_cs']['value'][0]:.4g}\n"
        yaml += f"a_c: {np.array2string(outputs['ivc.a_c']['value'], formatter=desvar_formatter, separator=', ')}\n"
        yaml += f"a_t: {np.array2string(outputs['ivc.a_t']['value'], formatter=desvar_formatter, separator=', ')}"
        if not self.options["fix_te"]:
            yaml += f"\nt_te: {s_t_te_des}"

        return yaml
