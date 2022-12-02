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

grid_pfc = (
    grid.Slacknode('n_0', V=1.+0.j),
    grid.Branch('line', 'n_0', 'n_1', y_lo=1e3-1e3j, y_tr=1e-6+1e-6j))

class Power_flow_calculation_basic(unittest.TestCase):
    
    def test_calculate_power_flow_00(self):
        """Power flow calculation with one branch and one 
        pure active power consumer."""
        model = make_model(
            grid_pfc, grid.Injection('consumer', 'n_1', P10=30.0))
        # calculate
        success, vnode_ri = estim.calculate_power_flow(
            model, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx = estim.ri_to_complex(vnode_ri)
        get_injected_power = get_injected_power_fn(
            model.injections, 
            loadcurve='interpolated')
        Inode = util.eval_residual_current(
            model, get_injected_power, Vnode=vnode_cx)
        # without slack node, slack is at index 0
        max_dev = norm(Inode[model.count_of_slacks:], np.inf)
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
            grid_pfc, grid.Injection('consumer', 'n_1', Q10=10.0))
        # calculate
        success, vnode_ri = estim.calculate_power_flow(
            model, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx = estim.ri_to_complex(vnode_ri)
        get_injected_power = get_injected_power_fn(
            model.injections, 
            loadcurve='interpolated')
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
            grid_pfc, grid.Injection('consumer', 'n_1', P10=30.0, Q10=10.0))
        # calculate
        success, vnode_ri = estim.calculate_power_flow(
            model, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx = estim.ri_to_complex(vnode_ri)
        get_injected_power = get_injected_power_fn(
            model.injections, 
            loadcurve='interpolated')
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
            grid_pfc, grid.Injection('generator', 'n_1', P10=-30.0))
        # calculate
        success, vnode_ri = estim.calculate_power_flow(
            model, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx = estim.ri_to_complex(vnode_ri)
        get_injected_power = get_injected_power_fn(
            model.injections, 
            loadcurve='interpolated')
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

# node: 0               1               2
#
#       |    branch_0   |    branch_1   |
#       +-----=====-----+-----=====-----+
#       |               |               |
#                         ^
#                         |  position of taps

grid_pfc2 = (
    grid.Slacknode('n_0', V=1.+0.j),
    grid.Branch('branch_0', 'n_0', 'n_1', y_lo=1e3-1e3j),
    grid.Branch('branch_1', 'n_1', 'n_2', y_lo=1e3-1e3j))

class Power_flow_calculation_taps(unittest.TestCase):
    
    def test_calculate_power_flow_00(self):
        """Power flow calculation with two branches and one consumer.
        Several taps at node_1/branch_1. Neutral tap."""
        model0 = make_model(
            grid_pfc2)
        model1 = make_model(
            grid_pfc2,
            grid.Branchtaps(
                'tap_branch_1',
                id_of_node='n_1',
                id_of_branch='branch_1',
                Vstep=.1/16,
                positionmin=-16,
                positionneutral=0,
                positionmax=16,
                position=0)) # <- neutral
        # calculate
        success0, vnode_ri0 = estim.calculate_power_flow(
            model0, vminsqr=_VMINSQR)
        success1, vnode_ri1 = estim.calculate_power_flow(
            model1, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success0, "calculate_power_flow shall succeed")
        self.assertTrue(success1, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx0 = estim.ri_to_complex(vnode_ri0)
        get_injected_power0 = get_injected_power_fn(
            model0.injections, 
            loadcurve='interpolated')
        Inode0 = util.eval_residual_current(
            model0, get_injected_power0, Vnode=vnode_cx0)
        # without slack node, slack is at index 0
        max_dev0 = norm(Inode0[1:], np.inf)
        self.assertLess(max_dev0, 1e-8, 'residual node current is 0')
        # node voltages are equal
        v_diff = vnode_ri0 - vnode_ri1
        self.assertLess(
            norm(v_diff), 
            1e-12, 
            'neutral tap position does not affect voltage')

    def test_calculate_power_flow_02(self):
        """Power flow calculation with two branches and one consumer.
        10 percent voltage increase by selected tap."""
        model0 = make_model(
            grid_pfc2)
        model1 = make_model(
            grid_pfc2,
            grid.Branchtaps(
                'tap_branch_1',
                id_of_node='n_1',
                id_of_branch='branch_1',
                Vstep=.1/16,
                positionmin=-16,
                positionneutral=0,
                positionmax=16,
                position=-16)) # <- 10 percent increase
        # calculate
        success0, vnode_ri0 = estim.calculate_power_flow(
            model0, vminsqr=_VMINSQR)
        success1, vnode_ri1 = estim.calculate_power_flow(
            model1, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success0, "calculate_power_flow shall succeed")
        self.assertTrue(success1, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx0 = estim.ri_to_complex(vnode_ri0)
        vnode_cx1 = estim.ri_to_complex(vnode_ri1)
        get_injected_power0 = get_injected_power_fn(
            model0.injections, 
            loadcurve='interpolated')
        Inode0 = util.eval_residual_current(
            model0, get_injected_power0, Vnode=vnode_cx0)
        # without slack node, slack is at index 0
        max_dev0 = norm(Inode0[1:], np.inf)
        self.assertLess(max_dev0, 1e-8, 'residual node current is 0')
        # voltage at node 2 is increased by 10 percent == 0.1
        idx_node_2 = model0.nodes.loc['n_2'].index_of_node
        vnode_2_cx0 = vnode_cx0[idx_node_2]
        vnode_2_cx1 = vnode_cx1[idx_node_2]
        vdiff_node_2_abs = np.abs(vnode_2_cx0-vnode_2_cx1)
        self.assertLess(
            np.abs(vnode_2_cx0), 
            np.abs(vnode_2_cx1),
            'voltage is increased')
        self.assertAlmostEqual(
            vdiff_node_2_abs, 
            0.1 * np.abs(vnode_2_cx0),
            delta=1e-12,
            msg='voltage differs by 10 percent')

    def test_calculate_power_flow_03(self):
        """Power flow calculation with two branches and one consumer.
        10 percent voltage decrease by selected tap."""
        model0 = make_model(
            grid_pfc2)
        model1 = make_model(
            grid_pfc2,
            grid.Branchtaps(
                'tap_branch_1',
                id_of_node='n_1',
                id_of_branch='branch_1',
                Vstep=.1/16,
                positionmin=-16,
                positionneutral=0,
                positionmax=16,
                position=16)) # <- 10 percent decrease
        # calculate
        success0, vnode_ri0 = estim.calculate_power_flow(
            model0, vminsqr=_VMINSQR)
        success1, vnode_ri1 = estim.calculate_power_flow(
            model1, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success0, "calculate_power_flow shall succeed")
        self.assertTrue(success1, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx0 = estim.ri_to_complex(vnode_ri0)
        vnode_cx1 = estim.ri_to_complex(vnode_ri1)
        get_injected_power0 = get_injected_power_fn(
            model0.injections, 
            loadcurve='interpolated')
        Inode0 = util.eval_residual_current(
            model0, get_injected_power0, Vnode=vnode_cx0)
        # without slack node, slack is at index 0
        max_dev0 = norm(Inode0[1:], np.inf)
        self.assertLess(max_dev0, 1e-8, 'residual node current is 0')
        # voltage at node 2 is increased by 10 percent == 0.1
        idx_node_2 = model0.nodes.loc['n_2'].index_of_node
        vnode_2_cx0 = vnode_cx0[idx_node_2]
        vnode_2_cx1 = vnode_cx1[idx_node_2]
        vdiff_node_2_abs = np.abs(vnode_2_cx0-vnode_2_cx1)
        self.assertGreater(
            np.abs(vnode_2_cx0), 
            np.abs(vnode_2_cx1),
            'voltage is decreased')
        self.assertAlmostEqual(
            vdiff_node_2_abs, 
            0.1 * np.abs(vnode_2_cx0),
            delta=1e-12,
            msg='voltage differs by 10 percent')

# node: 0               1               2
#
#       |     line_0    |     line_1    |
#       +-----=====-----+-----=====-----+
#       |               |               |
#                                      \|/ consumer
#                                       '

grid0 = (
    grid.Slacknode('n_0', V=1.+0.j),
    grid.Branch('line_0', 'n_0', 'n_1', y_lo=1e3-1e3j),
    grid.Branch('line_1', 'n_1', 'n_2', y_lo=1e3-1e3j),
    grid.Injection('consumer', 'n_2', P10=30.0, Q10=10.0))

class Estimation(unittest.TestCase):
    
    def test_scale_p(self):
        """Scale active power of consumer in order to meet the
        given active power P (measurement or setpoint).
        Given P is at n_0/line_0."""
        model = make_model(
            grid0,
            # give value of active power P at n_0/line_0
            grid.PValue('PQ_line_0', P=40.0),
            grid.Output('PQ_line_0', id_of_device='line_0', id_of_node='n_0'),
            # scaling factor kp for active power P of consumer
            grid.Defk('kp'),
            grid.Link(objid='consumer', part='p', id='kp'))
        ed = estim.get_estimation_data(model, count_of_steps=1)
        scaling_data, Iinj_data = ed['get_scaling_and_injection_data'](step=0)
        Inode = ed['inj_to_node'] @ Iinj_data[:,:2]
        # power flow calculation for initial voltages
        succ, Vnode_ri_ini = estim.calculate_power_flow2(
            model, ed, scaling_data, Inode)
        diff_data = estim.get_diff_expressions(
            model, ed, Iinj_data, selectors='P')
        # estimation
        optimize = estim.get_optimize(model, ed)
        succ, x = optimize(Vnode_ri_ini, scaling_data, Inode, diff_data)
        self.assertTrue(succ, 'estimation succeeds')
        # result processing
        count_of_v_ri = 2 * model.shape_of_Y[0] # real and imaginary voltages
        voltages_ri1 = x[:count_of_v_ri].toarray()
        voltages_cx = estim.ri_to_complex(voltages_ri1)
        get_injected_power = pfc.get_calc_injected_power_fn(
            _VMINSQR, 
            model.injections, 
            pq_factors=estim.get_k(scaling_data, x[count_of_v_ri:]), 
            loadcurve='interpolated')
        Inode_res = util.eval_residual_current(
            model, get_injected_power, Vnode=voltages_cx)
        self.assertAlmostEqual(
            # exclude slacks
            np.max(np.abs(Inode_res[model.count_of_slacks:])), 
            0,
            delta=1e-12,
            msg='Inode is almost 0')
        result_data = pfc.calculate_electric_data(
            model, get_injected_power, model.branchtaps.position, voltages_cx)
        branch_results = result_data['branches'].set_index('id')
        given_values = model.pvalues.set_index('id_of_batch')
        self.assertAlmostEqual(
            branch_results.loc['line_0'].P0_pu,
            given_values.loc['PQ_line_0'].P,
            delta=1e-12,
            msg='estimated active power equals given active power')

if __name__ == '__main__':
    unittest.main()
    






