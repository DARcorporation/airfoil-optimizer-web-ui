#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains the definition of the XFoil OpenMDAO Component and its helper functions.
"""
import numpy as np
import time

from multiprocessing.pool import ThreadPool
from xfoil import XFoil
from xfoil.model import Airfoil

from .. import rank
from .airfoil_component import AirfoilComponent


def xfoil_worker(xf, cl_spec, consistency_check=True):
    """
    Try to operate the given XFoil instance at a specified lift coefficient.

    Parameters
    ----------
    xf : XFoil
        Instance of XFoil class with Airfoil already specified
    cl_spec : float
        Lift coefficient
    consistency_check : bool, optional
        If True, airfoil will be analyzed at least twice to ensure consistent results. True by default.
        This option will run the same airfoil at the same point twice and checks if the results match. If they don't, it
        will be run a third time. It is expected that two out of three results will agree. The third will be considered
        incorrect and discarded. If the first run returns NaN, the airfoil will be assumed unrealistic and it will not
        be run a second time.

    Returns
    -------
    cd, cm : float or np.nan
        Drag and moment coefficients or nan if analysis did not complete successfully

    Notes
    -----
    The consistency check works as follows. Each airfoil is analyzed twice. First with a standard panel distribution,
    then with a panel distribution which is refined around the leading edge. If the two results are within 5%, the
    average result is returned. Otherwise, the larger of the two results is returned to be conservative.
    """
    xf.repanel(n_nodes=240)
    xf.reset_bls()
    _, cd1, cm1, _ = xf.cl(cl_spec)
    if np.isnan(cd1) or not consistency_check:
        return cd1, cm1

    xf.repanel(n_nodes=240, cv_par=2.0, cte_ratio=0.5)
    xf.reset_bls()
    _, cd2, cm2, _ = xf.cl(cl_spec)

    e = np.abs(cd2 - cd1) / cd1
    if e < 0.05:
        return (cd1 + cd2) / 2.0, (cm1 + cm2) / 2.0
    else:
        if cd1 > cd2:
            return cd1, cm1
        else:
            return cd2, cm2


def analyze_airfoil(
    x, y_u, y_l, cl, rey, mach=0, xf=None, pool=None, show_output=False
):
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
    cd, cm : float or np.nan
        Drag and moment coefficients of the airfoil at specified conditions, or nan if XFoil did not run successfully
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

        xf.airfoil = Airfoil(
            x=np.concatenate((x[-1:0:-1], x)), y=np.concatenate((y_u[-1:0:-1], y_l))
        )
        xf.Re = rey
        xf.M = mach
        xf.max_iter = 100
        cd, cm = pool.apply(xfoil_worker, args=(xf, cl))

    if clean_xf:
        del xf
    if clean_pool:
        del pool

    return cd, cm, None if clean_xf else xf


class XFoilComp(AirfoilComponent):
    """
    Computes the drag coefficient of an airfoil at a given lift coefficient, Reynolds nr., and Mach nr.
    """

    # Numpy string formatter
    array_formatter = {"float_kind": "{: 7.4f}".format}

    def initialize(self):
        super().initialize()
        self.options.declare("print", default=False, types=bool)

        xf = XFoil()
        xf.print = False
        self.options.declare("_xf", default=xf, types=XFoil, allow_none=True)
        self.options.declare(
            "_pool", default=ThreadPool(processes=1), types=ThreadPool, allow_none=True
        )

        self.recording_options["options_excludes"] = ["_xf", "_pool"]

    def setup(self):
        super().setup()

        # Inputs
        self.add_input("Cl_des", val=1.0)
        self.add_input("Re", val=1e6)
        self.add_input("M", val=0.0)

        # Output
        self.add_output("Cd", val=1.0)
        self.add_output("Cm", val=1.0)

    def compute(self, inputs, outputs, **kwargs):
        x, y_u, y_l, _, _ = self.compute_coords(inputs)

        t0 = time.time()
        cd, cm, xf = analyze_airfoil(
            x,
            y_u,
            y_l,
            inputs["Cl_des"][0],
            inputs["Re"][0],
            inputs["M"][0],
            self.options["_xf"],
            self.options["_pool"],
            self.options["print"],
        )
        dt = time.time() - t0
        self.options["_xf"] = xf

        outputs["Cd"] = cd if not np.isnan(cd) else 1e27
        outputs["Cm"] = cm if not np.isnan(cm) else 1e27

        if self.options["print"]:
            print(
                f"{rank:02d} :: "
                + "a_c: {}, ".format(
                    np.array2string(
                        inputs["a_c"], separator=", ", formatter=self.array_formatter
                    )
                )
                + "a_t: {}, ".format(
                    np.array2string(
                        inputs["a_t"], separator=", ", formatter=self.array_formatter
                    )
                )
                + f't_te: {inputs["t_te"][0]: 6.4f}, '
                + f"C_d: {cd: 7.4f}, Cm: {cm: 7.4f}, dt: {dt:6.3f}"
            )
