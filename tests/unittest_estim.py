# -*- coding: utf-8 -*-
"""
Copyright (C) 2022, 2023 pyprg

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
import context # adds parent folder of dssex to search path
import numpy as np
import egrid.builder as grid
import dssex.pfcnum as pfc
import dssex.estim as estim
from functools import partial
from numpy.linalg import norm
from egrid import make_model
import dssex.factors as ft

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

    def test_calculate_power_flow_slack(self):
        """Power flow calculation with one slack node.
        Minimal configuration."""
        vcx_slack = 0.9+0.2j
        model = make_model(grid.Slacknode('n_0', V=vcx_slack))
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factor_data.index)
        expressions = estim.get_expressions(model, gen_factor_symbols)
        # calculate
        success, vnode_ri = estim.calculate_power_flow(
            model, gen_factor_symbols, expressions, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success, "calculate_power_flow shall succeed")
        vnode_cx = estim.ri_to_complex(vnode_ri)
        self.assertAlmostEqual(
            vnode_cx[0,0],
            vcx_slack,
            'calculated slack voltage is {vcx_slack}')

    def test_calculate_power_flow_slack_consumer(self):
        """Power flow calculation with one consumer."""
        vcx_slack = 0.95+0.2j
        model = make_model(
            grid.Slacknode('n_0', V=vcx_slack),
            grid.Injection('consumer', 'n_0', P10=30.0))
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factor_data.index)
        expressions = estim.get_expressions(model, gen_factor_symbols)
        # calculate
        success, vnode_ri = estim.calculate_power_flow(
            model, gen_factor_symbols, expressions, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success, "calculate_power_flow shall succeed")
        vnode_cx = estim.ri_to_complex(vnode_ri)
        self.assertAlmostEqual(
            vnode_cx[0,0],
            vcx_slack,
            'calculated slack voltage is {vcx_slack}')

    def test_calculate_power_flow_slack_branch(self):
        """Power flow calculation with one branch."""
        vcx_slack = 0.95+0.2j
        model = make_model(
            grid.Slacknode('n_0', V=vcx_slack),
            grid.Branch('line', 'n_0', 'n_1', y_lo=1e3-1e3j, y_tr=1e-6+1e-6j))
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factor_data.index)
        expressions = estim.get_expressions(model, gen_factor_symbols)
        # calculate
        success, vnode_ri = estim.calculate_power_flow(
            model, gen_factor_symbols, expressions, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success, "calculate_power_flow shall succeed")
        vnode_cx = estim.ri_to_complex(vnode_ri)
        self.assertAlmostEqual(
            vnode_cx[0,0],
            vcx_slack,
            'calculated slack voltage is {vcx_slack}')

    def test_calculate_power_flow_00(self):
        """Power flow calculation with one branch and one
        pure active power consumer."""
        model = make_model(
            grid_pfc,
            grid.Injection('consumer', 'n_1', P10=30.0))
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factor_data.index)
        expressions = estim.get_expressions(model, gen_factor_symbols)
        # calculate
        success, vnode_ri = estim.calculate_power_flow(
            model, gen_factor_symbols, expressions, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx = estim.ri_to_complex(vnode_ri)
        get_injected_power = get_injected_power_fn(
            model.injections,
            loadcurve='interpolated')
        Inode = pfc.eval_residual_current(
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
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factor_data.index)
        expressions = estim.get_expressions(model, gen_factor_symbols)
        # calculate
        success, vnode_ri = estim.calculate_power_flow(
            model, gen_factor_symbols, expressions, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx = estim.ri_to_complex(vnode_ri)
        get_injected_power = get_injected_power_fn(
            model.injections,
            loadcurve='interpolated')
        Inode = pfc.eval_residual_current(
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
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factor_data.index)
        expressions = estim.get_expressions(model, gen_factor_symbols)
        # calculate
        success, vnode_ri = estim.calculate_power_flow(
            model, gen_factor_symbols, expressions, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx = estim.ri_to_complex(vnode_ri)
        get_injected_power = get_injected_power_fn(
            model.injections,
            loadcurve='interpolated')
        Inode = pfc.eval_residual_current(
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
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factor_data.index)
        expressions = estim.get_expressions(model, gen_factor_symbols)
        # calculate
        success, vnode_ri = estim.calculate_power_flow(
            model, gen_factor_symbols, expressions, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx = estim.ri_to_complex(vnode_ri)
        get_injected_power = get_injected_power_fn(
            model.injections,
            loadcurve='interpolated')
        Inode = pfc.eval_residual_current(
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
            #  taps
            grid.Deft(
                'tap_branch_1', type='const', min=-16, max=16,
                value=0, m=-.1/16, n=1., is_discrete=True), # <- neutral
            grid.Tlink(
                id_of_node='n_1',
                id_of_branch='branch_1',
                id_of_factor='tap_branch_1'))
        # calculate
        gen_factor_symbols0 = ft._create_symbols_with_ids(
            model0.factors.gen_factor_data.index)
        expr0 = estim.create_v_symbols_gb_expressions(
            model0, gen_factor_symbols0)
        success0, vnode_ri0 = estim.calculate_power_flow(
            model0, gen_factor_symbols0, expr0, vminsqr=_VMINSQR)
        gen_factor_symbols1 = ft._create_symbols_with_ids(
            model1.factors.gen_factor_data.index)
        expr1 = estim.create_v_symbols_gb_expressions(
            model1, gen_factor_symbols1)
        success1, vnode_ri1 = estim.calculate_power_flow(
            model1, gen_factor_symbols1, expr1, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success0, "calculate_power_flow shall succeed")
        self.assertTrue(success1, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx0 = estim.ri_to_complex(vnode_ri0)
        get_injected_power0 = get_injected_power_fn(
            model0.injections,
            loadcurve='interpolated')
        Inode0 = pfc.eval_residual_current(
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
            # taps
            grid.Deft(
                'taps', type='const', min=-16, max=16,
                value=-16, m=-.1/16, n=1., is_discrete=True), # <- 10% increase
            grid.Tlink(
                id_of_node='n_1', id_of_branch='branch_1', id_of_factor='taps'))
        # calculate
        gen_factor_symbols0 = ft._create_symbols_with_ids(
            model0.factors.gen_factor_data.index)
        expr0 = estim.create_v_symbols_gb_expressions(
            model0, gen_factor_symbols0)
        success0, vnode_ri0 = estim.calculate_power_flow(
            model0, gen_factor_symbols0, expr0, vminsqr=_VMINSQR)
        gen_factor_symbols1 = ft._create_symbols_with_ids(
            model1.factors.gen_factor_data.index)
        expr1 = estim.create_v_symbols_gb_expressions(
            model1, gen_factor_symbols1)
        success1, vnode_ri1 = estim.calculate_power_flow(
            model1, gen_factor_symbols1, expr1, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success0, "calculate_power_flow shall succeed")
        self.assertTrue(success1, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx0 = estim.ri_to_complex(vnode_ri0)
        vnode_cx1 = estim.ri_to_complex(vnode_ri1)
        get_injected_power0 = get_injected_power_fn(
            model0.injections,
            loadcurve='interpolated')
        Inode0 = pfc.eval_residual_current(
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
            # taps
            grid.Deft(
                'taps', type='const', min=-16, max=16,
                value=16, m=-.1/16, n=1., is_discrete=True), # <- 10% decrease
            grid.Tlink(
                id_of_node='n_1', id_of_branch='branch_1', id_of_factor='taps'))
        # calculate
        gen_factor_symbols0 = ft._create_symbols_with_ids(
            model0.factors.gen_factor_data.index)
        expr0 = estim.create_v_symbols_gb_expressions(
            model0, gen_factor_symbols0)
        success0, vnode_ri0 = estim.calculate_power_flow(
            model0, gen_factor_symbols0, expr0, vminsqr=_VMINSQR)
        gen_factor_symbols1 = ft._create_symbols_with_ids(
            model1.factors.gen_factor_data.index)
        expr1 = estim.create_v_symbols_gb_expressions(
            model1, gen_factor_symbols1)
        success1, vnode_ri1 = estim.calculate_power_flow(
            model1, gen_factor_symbols1, expr1, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success0, "calculate_power_flow shall succeed")
        self.assertTrue(success1, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx0 = estim.ri_to_complex(vnode_ri0)
        vnode_cx1 = estim.ri_to_complex(vnode_ri1)
        get_injected_power0 = get_injected_power_fn(
            model0.injections,
            loadcurve='interpolated')
        Inode0 = pfc.eval_residual_current(
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
#                                      \|/ consumer_1/2
#                                       '

grid_pfc3 = [
    grid.Slacknode(id_of_node='n_0', V=1.+0.j),
    grid.Branch(
        id='line_0',
        id_of_node_A='n_0',
        id_of_node_B='n_1',
        y_lo=1e3-1e3j,
        y_tr=1e-6+1e-6j),
    grid.Branch(
        id='line_1',
        id_of_node_A='n_1',
        id_of_node_B='n_2',
        y_lo=1e3-1e3j,
        y_tr=1e-6+1e-6j),
    grid.Deft(
        'tap_line1', type='const', min=-16, max=16,
        value=10, m=-.1/16, n=1., is_discrete=True),
    grid.Tlink(
        id_of_node='n_1', id_of_branch='line_1', id_of_factor='tap_line1'),
    grid.Injection(
        id='consumer_0',
        id_of_node='n_2',
        P10=300.0,
        Q10=10.0,
        Exp_v_p=1.0,
        Exp_v_q=0.0),
    grid.Injection(
        id='consumer_1',
        id_of_node='n_2',
        P10=30.0,
        Q10=10.0,
        Exp_v_p=0.0,
        Exp_v_q=2.0)]

class Power_flow_calculation_basic2(unittest.TestCase):

    def test_optimize_step(self):
        """Calculating power flow using function 'optimize_step' creates same
        results like function 'dssex.pfcnum.calculate_power_flow'."""
        model = make_model(grid_pfc3)
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factor_data.index)
        expressions = estim.get_expressions(model, gen_factor_symbols)
        step_data = estim.get_step_data(model, expressions)
        succ_estim, Vnode_ri_estim, _ = estim.optimize_step(*step_data)
        self.assertTrue(succ_estim, 'estimation succeeds')
        Vnode_cx_estim = estim.ri_to_complex(Vnode_ri_estim)
        succ_pfc, Vnode_cx_pfc = pfc.calculate_power_flow(
            model, loadcurve='interpolated')
        self.assertTrue(succ_pfc, 'power flow calculation succeeds')
        self.assertAlmostEqual(
            norm(Vnode_cx_pfc - Vnode_cx_estim, np.inf),
            0.,
            delta=1e-10,
            msg='result of optimize_step equals result of numeric calculation')

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

class Optimize_step(unittest.TestCase):

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
            grid.Defk('kp', step=0),
            grid.Klink(
                id_of_injection='consumer',
                part='p',
                id_of_factor='kp',
                step=0))
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factor_data.index)
        expressions = estim.get_expressions(model, gen_factor_symbols)
        step_data = estim.get_step_data(
            model, expressions, objectives='P')
        succ, x_V, x_scaling = estim.optimize_step(*step_data)
        self.assertTrue(succ, 'estimation succeeds')
        V, k, pos = estim.get_Vcx_kpq(step_data.scaling_data, x_V, x_scaling)
        ed = pfc.calculate_electric_data(model, V, k, pos)
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
            places=7,
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
            grid.Defk('kp', step=0),
            grid.Klink(
                id_of_injection='consumer',
                part='p',
                id_of_factor='kp',
                step=0))
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factor_data.index)
        expressions = estim.get_expressions(model, gen_factor_symbols)
        step_data = estim.get_step_data(
            model, expressions, objectives='P')
        succ, x_V, x_scaling = estim.optimize_step(*step_data)
        self.assertTrue(succ, 'estimation succeeds')
        V, k, pos = estim.get_Vcx_kpq(step_data.scaling_data, x_V, x_scaling)
        ed = pfc.calculate_electric_data(model, V, k, pos)
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
            grid.Defk('kq', step=0),
            grid.Klink(
                id_of_injection='consumer',
                part='q',
                id_of_factor='kq',
                step=0))
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factor_data.index)
        expressions = estim.get_expressions(model, gen_factor_symbols)
        step_data = estim.get_step_data(
            model, expressions, objectives='Q')
        succ, x_V, x_scaling = estim.optimize_step(*step_data)
        self.assertTrue(succ, 'estimation succeeds')
        V, k, pos = estim.get_Vcx_kpq(step_data.scaling_data, x_V, x_scaling)
        ed = pfc.calculate_electric_data(model, V, k, pos)
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
            places=6,
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
            grid.Defk('kq', step=0),
            grid.Klink(
                id_of_injection='consumer',
                part='q',
                id_of_factor='kq',
                step=0))
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factor_data.index)
        expressions = estim.get_expressions(model, gen_factor_symbols)
        step_data = estim.get_step_data(
            model, expressions, objectives='Q')
        succ, x_V, x_scaling = estim.optimize_step(*step_data)
        self.assertTrue(succ, 'estimation succeeds')
        V, k, pos = estim.get_Vcx_kpq(step_data.scaling_data, x_V, x_scaling)
        ed = pfc.calculate_electric_data(model, V, k, pos)
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
            grid.Defk('kpq', step=0),
            grid.Klink(
                id_of_injection='consumer',
                part='pq',
                id_of_factor='kpq',
                step=0))
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factor_data.index)
        expressions = estim.get_expressions(model, gen_factor_symbols)
        step_data = estim.get_step_data(
            model, expressions, objectives='I')
        succ, x_V, x_scaling = estim.optimize_step(*step_data)
        self.assertTrue(succ, 'estimation succeeds')
        V, k, pos = estim.get_Vcx_kpq(step_data.scaling_data, x_V, x_scaling)
        ed = pfc.calculate_electric_data(model, V, k, pos)
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
            places=7,
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
            grid.Defk('kpq', step=0),
            grid.Klink(
                id_of_injection='consumer',
                part='pq',
                id_of_factor=('kpq', 'kpq'),
                step=0))
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factor_data.index)
        expressions = estim.get_expressions(model, gen_factor_symbols)
        step_data = estim.get_step_data(
            model, expressions, objectives='I')
        succ, x_V, x_scaling = estim.optimize_step(*step_data)
        self.assertTrue(succ, 'estimation succeeds')
        V, k, pos = estim.get_Vcx_kpq(step_data.scaling_data, x_V, x_scaling)
        ed = pfc.calculate_electric_data(model, V, k, pos)
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
            grid.Defk('kq', step=0),
            grid.Klink(
                id_of_injection='consumer',
                part='q',
                id_of_factor='kq',
                step=0))
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factor_data.index)
        expressions = estim.get_expressions(model, gen_factor_symbols)
        step_data = estim.get_step_data(
            model, expressions, objectives='V')
        succ, x_V, x_scaling = estim.optimize_step(*step_data)
        self.assertTrue(succ, 'estimation succeeds')
        V, k, ftaps = estim.get_Vcx_kpq(step_data.scaling_data, x_V, x_scaling)
        ed = pfc.calculate_electric_data(model, V, k, ftaps)
        self.assertAlmostEqual(
            # exclude slacks
            np.max(np.abs(ed.residual_node_current()[model.count_of_slacks:])),
            0,
            delta=1e-8,
            msg='Inode is almost 0')
        given_V_at_node = model.vvalues.set_index('id_of_node').loc['n_2']
        self.assertAlmostEqual(
            np.abs(V[given_V_at_node.index_of_node])[0],
            given_V_at_node.V,
            places=10,
            msg='estimated voltage equals given voltage')

if __name__ == '__main__':
    unittest.main()







