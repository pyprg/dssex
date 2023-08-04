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
import pandas as pd
import egrid.builder as grid
import dssex.result as rt
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

    def test_calculate_power_flow_without_elements(self):
        """Power flow calculation with one slack node."""
        model = make_model()
        # calculate
        success, vnode_ri = estim.calculate_power_flow(
            model, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success, "calculate_power_flow shall succeed")
        vnode_cx = estim.ri_to_complex(vnode_ri)
        self.assertEqual(
            len(vnode_cx),
            0,
            'empty voltage vector')

    def test_calculate_power_flow_slack(self):
        """Power flow calculation with one slack node."""
        vcx_slack = 0.9+0.2j
        model = make_model(grid.Slacknode('n_0', V=vcx_slack))
        # calculate
        success, vnode_ri = estim.calculate_power_flow(
            model, vminsqr=_VMINSQR)
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
        # calculate
        success, vnode_ri = estim.calculate_power_flow(
            model, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success, "calculate_power_flow shall succeed")
        vnode_cx = estim.ri_to_complex(vnode_ri)
        self.assertAlmostEqual(
            vnode_cx[0,0],
            vcx_slack,
            'calculated slack voltage is {vcx_slack}')

    def test_calculate_power_flow_slack_bridge(self):
        """Power flow calculation with one branch."""
        vcx_slack = 0.95+0.2j
        model = make_model(
            grid.Slacknode('n_0', V=vcx_slack),
            grid.Branch('switch', 'n_0', 'n_1'))
        # calculate
        success, vnode_ri = estim.calculate_power_flow(
            model, vminsqr=_VMINSQR)
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
        # calculate
        success, vnode_ri = estim.calculate_power_flow(
            model, vminsqr=_VMINSQR)
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
        # calculate
        success, vnode_ri = estim.calculate_power_flow(
            model, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx = estim.ri_to_complex(vnode_ri)
        max_dev = pfc.max_residual_current(
            model, vnode_cx, loadcurve='interpolated')
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
        max_dev = pfc.max_residual_current(
            model, vnode_cx, loadcurve='interpolated')
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
        max_dev = pfc.max_residual_current(
            model, Vnode=vnode_cx, loadcurve='interpolated',
            vminsqr=_VMINSQR)
        # without slack node, slack is at index 0
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
        max_dev = pfc.max_residual_current(
            model, Vnode=vnode_cx, loadcurve='interpolated',
            vminsqr=_VMINSQR)
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
        success0, vnode_ri0 = estim.calculate_power_flow(
            model0, vminsqr=_VMINSQR)
        success1, vnode_ri1 = estim.calculate_power_flow(
            model1, vminsqr=_VMINSQR)
        # test
        self.assertTrue(success0, "calculate_power_flow shall succeed")
        self.assertTrue(success1, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx0 = estim.ri_to_complex(vnode_ri0)
        max_dev0 = pfc.max_residual_current(
            model0, Vnode=vnode_cx0, loadcurve='interpolated',
            vminsqr=_VMINSQR)
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
        max_dev0 = pfc.max_residual_current(
            model0, Vnode=vnode_cx0, loadcurve='interpolated',
            vminsqr=_VMINSQR)
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
                id_of_node='n_1', id_of_branch='branch_1',
                id_of_factor='taps'))
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
        max_dev0 = pfc.max_residual_current(
            model0, Vnode=vnode_cx0, loadcurve='interpolated',
            vminsqr=_VMINSQR)
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
            model.factors.gen_factordata.index)
        expressions = estim.get_expressions(model, gen_factor_symbols)
        step_data = estim.get_step_data(model, expressions)
        _, voltages_ri, __ = estim.calculate_initial_powerflow(step_data)
        succ_estim, Vnode_ri_estim, _ = estim.optimize_step(
            **step_data, Vnode_ri_ini=voltages_ri)
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

class Calculate_power_flow(unittest.TestCase):

    # node: 0               1               2
    #
    #       |     line_0    |     line_1    |
    #       +-----=====-----+-----=====-----+
    #       |               |               |
    #                       |              \|/ consumer_1
    #                       |               '
    #                       |
    # node:                 |               3
    #                       |
    #                       |     line_2    |
    #                       +-----=====-----+---------------+
    #                       |               |               |
    #                                      \|/ consumer_2  \|/ consumer_3
    #                                       '               '

    grid0 = (
        grid.Slacknode('n_0', V=1.+0.j),
        grid.Branch('line_0', 'n_0', 'n_1', y_lo=1e3-1e3j),
        grid.Branch('line_1', 'n_1', 'n_2', y_lo=1e3-1e3j),
        grid.Branch('line_2', 'n_1', 'n_3', y_lo=1e3-1e3j),
        grid.Injection('consumer_1', 'n_2', P10=10.0, Q10=10.0),
        grid.Injection('consumer_2', 'n_3', P10=20.0, Q10=15.0),
        grid.Injection('consumer_3', 'n_3', P10=30.0, Q10=20.0))

    order_of_nodes = pd.Series(
        range(4), dtype=np.int64, index=[f'n_{idx}' for idx in range(4)])

    def test_taps_factor(self):
        kpq = np.ones((3,2), dtype=float)
        # first: tapsposition == 0 -> no impact to voltage
        model0 = make_model(
            self.grid0,
            grid.Deft(
                'taps', type='const', min=-16, max=16, value=0,
                m=-0.00625, n=1., is_discrete=True),
            grid.Tlink(
                id_of_node='n_0', id_of_branch='line_0', id_of_factor='taps'))
        # calculate power flow
        success0, vnode_ri0 = estim.calculate_power_flow(
            model0, vminsqr=_VMINSQR)
        self.assertTrue(success0, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx0 = estim.ri_to_complex(vnode_ri0)
        get_injected_power0 = get_injected_power_fn(
            model0.injections,
            kpq=kpq,
            loadcurve='interpolated')
        Inode0 = pfc.eval_residual_current(
            model0, get_injected_power0, Vnode=vnode_cx0)
        # without slack node, slack is at index 0
        max_dev0 = norm(Inode0[model0.count_of_slacks:], np.inf)
        self.assertLess(max_dev0, 3e-8, 'residual node current is 0')
        res = rt.calculate_electric_data(model0, vnode_cx0, kpq=kpq)
        idx_v = model0.nodes.reindex(self.order_of_nodes.index).index_of_node
        # df_Vcx0 = pd.DataFrame(
        #     {'id_of_node': order_of_nodes.index,
        #       'Vcx': vnode_cx0[idx_v].reshape(-1)})
        # print(df_Vcx0)
        #
        # second: tapsposition == -16 -> voltage increase ~10%
        model1 = make_model(
            self.grid0,
            grid.Deft(
                'taps', type='const', min=-16, max=16, value=-16, # <- V +10%
                m=-0.00625, n=1., is_discrete=True),
            grid.Tlink(
                id_of_node='n_0', id_of_branch='line_0', id_of_factor='taps'))
        # calculate power flow
        success1, vnode_ri1 = estim.calculate_power_flow(
            model1, vminsqr=_VMINSQR)
        self.assertTrue(success1, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx1 = estim.ri_to_complex(vnode_ri1)
        get_injected_power1 = get_injected_power_fn(
            model1.injections,
            kpq=kpq,
            loadcurve='interpolated')
        Inode1 = pfc.eval_residual_current(
            model1, get_injected_power1, Vnode=vnode_cx1)
        # without slack node, slack is at index 0
        max_dev1 = norm(Inode1[model1.count_of_slacks:], np.inf)
        self.assertLess(max_dev1, 3e-8, 'residual node current is 0')
        res = rt.calculate_electric_data(model1, vnode_cx1, kpq=kpq)
        idx_v = model1.nodes.reindex(self.order_of_nodes.index).index_of_node
        # df_Vcx1 = pd.DataFrame(
        #     {'id_of_node': order_of_nodes.index,
        #      'Vcx': vnode_cx1[idx_v].reshape(-1)})
        # print(df_Vcx1)
        diff_max = norm(
            #                                       10% | no slack
            (vnode_cx0[idx_v] - (vnode_cx1[idx_v] / 1.1))[1:],
            np.inf)
        self.assertLess(diff_max, 6e-3, "voltage increase about 10%")

if __name__ == '__main__':
    unittest.main()







