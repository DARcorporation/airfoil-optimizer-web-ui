#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains the definition of the Airfoil Geometry OpenMDAO Component.
"""
import numpy as np

from .airfoil_component import AirfoilComponent


class Geom(AirfoilComponent):
    """
    Computes the thickness-over-chord ratio and cross-sectional area of an airfoil.
    """

    def setup(self):
        super().setup()
        # Outputs
        self.add_output("t_c", val=0.0)
        self.add_output("A_cs", val=0.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        x, _, _, _, t = self.compute_coords(inputs)
        outputs["t_c"] = np.max(t)
        outputs["A_cs"] = np.trapz(t, x)
