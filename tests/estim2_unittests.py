# -*- coding: utf-8 -*-
"""
Copyright (C) 2022 pyprg

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Created on Wed Nov 30 13:04:16 2022

@author: pyprg
"""
import unittest
import numpy as np
import egrid.builder as grid
import src.dssex.util as util  # eval_residual_current
import src.dssex.pfcnum as pfc # get_calc_injected_power_fn
from egrid import make_model
from functools import partial
from numpy.linalg import norm
import src.dssex.estim2 as estim

# square of voltage magnitude, minimum value for load curve, 
#   if value is below _VMINSQR the load curves for P and Q converge
#   towards a linear load curve which is 0 when V=0; P(V=0)=0, Q(V=0)=0
_VMINSQR = 0.8**2
get_injected_power_fn = partial(pfc.get_calc_injected_power_fn, _VMINSQR)


# node: 0               1
#
#       |     line      |
#       +-----=====-----+
#       |               |
#                      \|/ consumer/generator
#                       '

br = (
    grid.Slacknode('n_0', V=1.+0.j),
    grid.Branch('line', 'n_0', 'n_1', y_lo=1e3-1e3j, y_tr=1e-6+1e-6j))

class Power_flow_calculation_basic(unittest.TestCase):
    
    def test_calculate_power_flow_00(self):
        """Power flow calculation with one branch and one 
        pure active power consumer."""
        model = make_model(
            br, grid.Injection('consumer', 'n_1', P10=30.0))
        get_injected_power = get_injected_power_fn(
            model.injections, 
            loadcurve='interpolated')
        # calculate
        success, vnode_ri = estim.calculate_power_flow(
            model, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx = estim.ri_to_complex(vnode_ri)
        Inode = util.eval_residual_current(
            model, get_injected_power, Vnode=vnode_cx)
        # without slack node, slack is at index 0
        max_dev = norm(Inode[1:], np.inf)
        self.assertLess(max_dev, 1e-8, 'residual node current is 0')
        vnode_abs = np.abs(vnode_cx)
        # check voltage at consumer
        self.assertGreater(
            vnode_abs[0], 
            vnode_abs[1], 
            'voltage at slack is greater than voltage at consumer')
    
    def test_calculate_power_flow_01(self):
        """Power flow calculation with one branch and one 
        pure reactive power consumer."""
        model = make_model(
            br, grid.Injection('consumer', 'n_1', Q10=10.0))
        get_injected_power = get_injected_power_fn(
            model.injections, 
            loadcurve='interpolated')
        # calculate
        success, vnode_ri = estim.calculate_power_flow(
            model, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx = estim.ri_to_complex(vnode_ri)
        Inode = util.eval_residual_current(
            model, get_injected_power, Vnode=vnode_cx)
        # without slack node, slack is at index 0
        max_dev = norm(Inode[1:], np.inf)
        self.assertLess(max_dev, 1e-8, 'residual node current is 0')
        vnode_abs = np.abs(vnode_cx)
        # check voltage at consumer
        self.assertGreater(
            vnode_abs[0], 
            vnode_abs[1], 
            'voltage at slack is greater than voltage at consumer')
    
    def test_calculate_power_flow_02(self):
        """Power flow calculation with one branch and one power consumer."""
        model = make_model(
            br, grid.Injection('consumer', 'n_1', P10=30.0, Q10=10.0))
        get_injected_power = get_injected_power_fn(
            model.injections, 
            loadcurve='interpolated')
        # calculate
        success, vnode_ri = estim.calculate_power_flow(
            model, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx = estim.ri_to_complex(vnode_ri)
        Inode = util.eval_residual_current(
            model, get_injected_power, Vnode=vnode_cx)
        # without slack node, slack is at index 0
        max_dev = norm(Inode[1:], np.inf)
        self.assertLess(max_dev, 1e-8, 'residual node current is 0')
        vnode_abs = np.abs(vnode_cx)
        # check voltage at consumer
        self.assertGreater(
            vnode_abs[0], 
            vnode_abs[1], 
            'voltage at slack is greater than voltage at consumer')
    
    def test_calculate_power_flow_03(self):
        """Power flow calculation with one branch and one 
        pure active power generator."""
        model = make_model(
            br, grid.Injection('generator', 'n_1', P10=-30.0))
        get_injected_power = get_injected_power_fn(
            model.injections, 
            loadcurve='interpolated')
        # calculate
        success, vnode_ri = estim.calculate_power_flow(
            model, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx = estim.ri_to_complex(vnode_ri)
        Inode = util.eval_residual_current(
            model, get_injected_power, Vnode=vnode_cx)
        # without slack node, slack is at index 0
        max_dev = norm(Inode[1:], np.inf)
        self.assertLess(max_dev, 1e-8, 'residual node current is 0')
        vnode_abs = np.abs(vnode_cx)
        # check voltage at consumer
        self.assertLess(
            vnode_abs[0], 
            vnode_abs[1], 
            'voltage at slack is smaller than voltage at generator')

if __name__ == '__main__':
    unittest.main()
    






