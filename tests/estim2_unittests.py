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
        """Power flow calculation with two branches.
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

    def test_calculate_power_flow_01(self):
        """Power flow calculation with two branches.
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

    def test_calculate_power_flow_02(self):
        """Power flow calculation with two branches.
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
    
    def test_scale_p_meet_p(self):
        """Scale active power of consumer in order to meet the
        given active power P at a terminal of a branch (measurement or 
        setpoint). Given P is assigned to n_0/line_0."""
        model = make_model(
            grid0,
            # give value of active power P at n_0/line_0
            grid.PValue('PQ_line_0', P=40.0),
            grid.Output('PQ_line_0', id_of_device='line_0', id_of_node='n_0'),
            # scaling factor kp for active power P of consumer
            grid.Defk('kp'),
            grid.Link(objid='consumer', part='p', id='kp'))
        succ, *V_k = estim.estimate(
            *estim.prep_estimate(model, quantities_of_objective='P')) 
        self.assertTrue(succ, 'estimation succeeds')
        ed = pfc.calculate_electric_data2(model, *V_k)
        self.assertAlmostEqual(
            # exclude slacks
            np.max(np.abs(ed.residual_node_current()[model.count_of_slacks:])), 
            0,
            delta=1e-12,
            msg='Inode is almost 0')
        given_values = model.pvalues.set_index('id_of_batch')
        self.assertAlmostEqual(
            ed.branch().loc['line_0'].P0_pu,
            given_values.loc['PQ_line_0'].P,
            delta=1e-12,
            msg='estimated active power equals given active power')
    
    def test_scale_p_meet_p2(self):
        """Scale active power of consumer in order to meet the
        given active power P at a terminal of an injection (measurement or 
        setpoint). Given P is assigned to consumer."""
        model = make_model(
            grid0,
            # give value of active power P at n_0/line_0
            grid.PValue('PQ_consumer', P=40.0),
            grid.Output('PQ_consumer', id_of_device='consumer'),
            # scaling factor kp for active power P of consumer
            grid.Defk('kp'),
            grid.Link(objid='consumer', part='p', id='kp'))
        succ, *V_k = estim.estimate(
            *estim.prep_estimate(model, quantities_of_objective='P')) 
        self.assertTrue(succ, 'estimation succeeds')
        ed = pfc.calculate_electric_data2(model, *V_k)
        self.assertAlmostEqual(
            # exclude slacks
            np.max(np.abs(ed.residual_node_current()[model.count_of_slacks:])), 
            0,
            delta=1e-8,
            msg='Inode is almost 0')
        given_values = model.pvalues.set_index('id_of_batch')
        self.assertAlmostEqual(
            ed.injection().loc['consumer'].P_pu,
            given_values.loc['PQ_consumer'].P,
            delta=1e-8,
            msg='estimated active power equals given active power')
    
    def test_scale_q_meet_q(self):
        """Scale reactive power of consumer in order to meet the
        given reactive power Q at a terminal of a branch (measurement or 
        setpoint). Given Q is assigned to n_0/line_0."""
        model = make_model(
            grid0,
            # give value of active power P at n_0/line_0
            grid.QValue('PQ_line_0', Q=40.0),
            grid.Output('PQ_line_0', id_of_device='line_0', id_of_node='n_0'),
            # scaling factor kq for reactive power Q of consumer
            grid.Defk('kq'),
            grid.Link(objid='consumer', part='q', id='kq'))
        succ, *V_k = estim.estimate(
            *estim.prep_estimate(model, quantities_of_objective='Q')) 
        self.assertTrue(succ, 'estimation succeeds')
        ed = pfc.calculate_electric_data2(model, *V_k)
        self.assertAlmostEqual(
            # exclude slacks
            np.max(np.abs(ed.residual_node_current()[model.count_of_slacks:])), 
            0,
            delta=1e-10,
            msg='Inode is almost 0')
        given_values = model.qvalues.set_index('id_of_batch')
        self.assertAlmostEqual(
            ed.branch().loc['line_0'].Q0_pu,
            given_values.loc['PQ_line_0'].Q,
            delta=1e-10,
            msg='estimated reactive power equals given reactive power')
    
    def test_scale_q_meet_q2(self):
        """Scale reactive power of consumer in order to meet the
        given reactive power Q at a terminal of an injection (measurement or 
        setpoint). Given Q is assigned to consumer."""
        model = make_model(
            grid0,
            # give value of active power P at n_0/line_0
            grid.QValue('PQ_consumer', Q=40.0),
            grid.Output('PQ_consumer', id_of_device='consumer'),
            # scaling factor kq for reactive power Q of consumer
            grid.Defk('kq'),
            grid.Link(objid='consumer', part='q', id='kq'))
        succ, *V_k = estim.estimate(
            *estim.prep_estimate(model, quantities_of_objective='Q')) 
        self.assertTrue(succ, 'estimation succeeds')
        ed = pfc.calculate_electric_data2(model, *V_k)
        self.assertAlmostEqual(
            # exclude slacks
            np.max(np.abs(ed.residual_node_current()[model.count_of_slacks:])), 
            0,
            delta=1e-8,
            msg='Inode is almost 0')
        given_values = model.qvalues.set_index('id_of_batch')
        self.assertAlmostEqual(
            ed.injection().loc['consumer'].Q_pu,
            given_values.loc['PQ_consumer'].Q,
            delta=1e-8,
            msg='estimated reactive power equals given reactive power')
    
    def test_scale_pq_meet_i(self):
        """Scale active and reactive power of consumer in order to meet the
        given current I at a terminal of a branch (measurement or setpoint).
        Given I is assigned to n_0/line_0."""
        model = make_model(
            grid0,
            # give value of electric current I at n_0/line_0
            grid.IValue('I_line_0', I=40.0),
            grid.Output('I_line_0', id_of_device='line_0', id_of_node='n_0'),
            # scaling factor kpq for active/reactive power P/Q of consumer
            grid.Defk('kpq'),
            grid.Link(objid='consumer', part='pq', id='kpq'))
        succ, *V_k = estim.estimate(
            *estim.prep_estimate(model, quantities_of_objective='I')) 
        self.assertTrue(succ, 'estimation succeeds')
        ed = pfc.calculate_electric_data2(model, *V_k)
        self.assertAlmostEqual(
            # exclude slacks
            np.max(np.abs(ed.residual_node_current()[model.count_of_slacks:])), 
            0,
            delta=1e-12,
            msg='Inode is almost 0')
        given_values = model.ivalues.set_index('id_of_batch')
        self.assertAlmostEqual(
            ed.branch().loc['line_0'].I0_pu,
            given_values.loc['I_line_0'].I,
            delta=1e-12,
            msg='estimated electric current equals given electric current')
    
    def test_scale_pq_meet_i2(self):
        """Scale active and reactive power of consumer in order to meet the
        given current I at a terminal of an injection (measurement or setpoint).
        Given I is assigned to consumer."""
        model = make_model(
            grid0,
            # give value of active power P at n_0/line_0
            grid.IValue('I_consumer', I=40.0),
            grid.Output('I_consumer', id_of_device='consumer'),
            # scaling factor kpq for active/reactive power P/Q of consumer
            grid.Defk('kpq'),
            grid.Link(objid='consumer', part='pq', id='kpq'))
        succ, *V_k = estim.estimate(
            *estim.prep_estimate(model, quantities_of_objective='I')) 
        self.assertTrue(succ, 'estimation succeeds')
        ed = pfc.calculate_electric_data2(model, *V_k)
        self.assertAlmostEqual(
            # exclude slacks
            np.max(np.abs(ed.residual_node_current()[model.count_of_slacks:])), 
            0,
            delta=1e-8,
            msg='Inode is almost 0')
        given_values = model.ivalues.set_index('id_of_batch')
        self.assertAlmostEqual(
            ed.injection().loc['consumer'].I_pu,
            given_values.loc['I_consumer'].I,
            delta=1e-8,
            msg='estimated electric current equals given electric current')
    
    def test_scale_q_meet_v(self):
        """Scale reactive power of consumer in order to meet the
        given voltage V (measurement or setpoint).
        Given V is assigned to node 2 ('n_2')."""
        model = make_model(
            grid0,
            # give magnitude of voltage at n_2
            grid.Vvalue('n_2', V=1.02),
            # scaling factor kq for reactive power Q of consumer
            grid.Defk('kq'),
            grid.Link(objid='consumer', part='q', id='kq'))
        succ, *V_k = estim.estimate(
            *estim.prep_estimate(model, quantities_of_objective='V')) 
        self.assertTrue(succ, 'estimation succeeds')
        ed = pfc.calculate_electric_data2(model, *V_k)
        self.assertAlmostEqual(
            # exclude slacks
            np.max(np.abs(ed.residual_node_current()[model.count_of_slacks:])), 
            0,
            delta=1e-8,
            msg='Inode is almost 0')
        given_V_at_node = model.vvalues.set_index('id_of_node').loc['n_2']
        self.assertAlmostEqual(
            np.abs(V_k[0][given_V_at_node.index_of_node]),
            given_V_at_node.V,
            delta=1e-12,
            msg='estimated voltage equals given voltage')

if __name__ == '__main__':
    unittest.main()
    






