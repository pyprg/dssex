# -*- coding: utf-8 -*-
"""
Copyright (C) 2023 pyprg

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

Created on Tue Feb 28 00:01:32 2023

@author: pyprg
"""
import unittest
import numpy as np
import context # adds parent folder of dssex to search path
import egrid.builder as grid
import dssex.result as rt
import dssex.factors as ft
import dssex.estim as estim
import dssex.pfcnum as pfc
from numpy.testing import assert_array_almost_equal
from egrid import make_model

class Optimize_step(unittest.TestCase):
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


    def test_scale_p_meet_p(self):
        """Scale active power of consumer in order to meet the
        given active power P at a terminal of a branch (measurement or
        setpoint). Given P is assigned to n_0/line_0."""
        model = make_model(
            self.grid0,
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
            model.factors.gen_factordata.index)
        expressions = estim.get_expressions(model, gen_factor_symbols)
        step_data = estim.get_step_data(
            model, expressions, objectives='P')
        _, voltages_ri, __ = estim.calculate_initial_powerflow(step_data)
        succ, x_V, x_scaling = estim.optimize_step(
            **step_data, Vnode_ri_ini=voltages_ri)
        self.assertTrue(succ, 'estimation succeeds')
        V, k, pos = estim.get_Vcx_factors(
            step_data['factordata'], x_V, x_scaling)
        self.assertAlmostEqual(
            # exclude slacks
            pfc.max_residual_current(model, V, positions=pos, kpq=k),
            0,
            delta=1e-12,
            msg='Inode is almost 0')
        given_values = model.pvalues.set_index('id_of_batch')
        res = rt.calculate_electric_data(model, V, kpq=k)
        self.assertAlmostEqual(
            res['branches'].loc['line_0'].P0_pu,
            given_values.loc['PQ_line_0'].P,
            places=7,
            msg='estimated active power equals given active power')

    def test_scale_p_meet_p2(self):
        """Scale active power of consumer in order to meet the
        given active power P at a terminal of an injection (measurement or
        setpoint). Given P is assigned to consumer."""
        model = make_model(
            self.grid0,
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
            model.factors.gen_factordata.index)
        expressions = estim.get_expressions(model, gen_factor_symbols)
        step_data = estim.get_step_data(
            model, expressions, objectives='P')
        _, voltages_ri, __ = estim.calculate_initial_powerflow(step_data)
        succ, x_V, x_scaling = estim.optimize_step(
            **step_data, Vnode_ri_ini=voltages_ri)
        self.assertTrue(succ, 'estimation succeeds')
        V, k, pos = estim.get_Vcx_factors(step_data['factordata'], x_V, x_scaling)
        self.assertAlmostEqual(
            # exclude slacks
            pfc.max_residual_current(model, V, positions=pos, kpq=k),
            0,
            delta=1e-8,
            msg='Inode is almost 0')
        inj_res = rt.calculate_injection_results(model, V, kpq=k)
        given_values = model.pvalues.set_index('id_of_batch')
        self.assertAlmostEqual(
            inj_res.loc['consumer'].P_pu,
            given_values.loc['PQ_consumer'].P,
            delta=1e-8,
            msg='estimated active power equals given active power')

    def test_scale_q_meet_q(self):
        """Scale reactive power of consumer in order to meet the
        given reactive power Q at a terminal of a branch (measurement or
        setpoint). Given Q is assigned to n_0/line_0."""
        model = make_model(
            self.grid0,
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
            model.factors.gen_factordata.index)
        expressions = estim.get_expressions(model, gen_factor_symbols)
        step_data = estim.get_step_data(
            model, expressions, objectives='Q')
        _, voltages_ri, __ = estim.calculate_initial_powerflow(step_data)
        succ, x_V, x_scaling = estim.optimize_step(
            **step_data, Vnode_ri_ini=voltages_ri)
        self.assertTrue(succ, 'estimation succeeds')
        V, k, pos = estim.get_Vcx_factors(step_data['factordata'], x_V, x_scaling)
        self.assertAlmostEqual(
            # exclude slacks
            pfc.max_residual_current(model, V, positions=pos, kpq=k),
            0,
            delta=1e-10,
            msg='Inode is almost 0')
        branch_res = rt.calculate_branch_results(model, V, positions=pos)
        given_values = model.qvalues.set_index('id_of_batch')
        self.assertAlmostEqual(
            branch_res.loc['line_0'].Q0_pu,
            given_values.loc['PQ_line_0'].Q,
            places=6,
            msg='estimated reactive power equals given reactive power')

    def test_scale_q_meet_q2(self):
        """Scale reactive power of consumer in order to meet the
        given reactive power Q at a terminal of an injection (measurement or
        setpoint). Given Q is assigned to consumer."""
        model = make_model(
            self.grid0,
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
            model.factors.gen_factordata.index)
        expressions = estim.get_expressions(model, gen_factor_symbols)
        step_data = estim.get_step_data(
            model, expressions, objectives='Q')
        _, voltages_ri, __ = estim.calculate_initial_powerflow(step_data)
        succ, x_V, x_scaling = estim.optimize_step(
            **step_data, Vnode_ri_ini=voltages_ri)
        self.assertTrue(succ, 'estimation succeeds')
        V, k, pos = estim.get_Vcx_factors(step_data['factordata'], x_V, x_scaling)
        self.assertAlmostEqual(
            # exclude slacks
            pfc.max_residual_current(model, V, positions=pos, kpq=k),
            0,
            delta=1e-8,
            msg='Inode is almost 0')
        inj_res = rt.calculate_injection_results(model, V, kpq=k)
        given_values = model.qvalues.set_index('id_of_batch')
        self.assertAlmostEqual(
            inj_res.loc['consumer'].Q_pu,
            given_values.loc['PQ_consumer'].Q,
            delta=1e-8,
            msg='estimated reactive power equals given reactive power')

    def test_scale_pq_meet_i(self):
        """Scale active and reactive power of consumer in order to meet the
        given current I at a terminal of a branch (measurement or setpoint).
        Given I is assigned to n_0/line_0."""
        model = make_model(
            self.grid0,
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
            model.factors.gen_factordata.index)
        expressions = estim.get_expressions(model, gen_factor_symbols)
        step_data = estim.get_step_data(
            model, expressions, objectives='I')
        _, voltages_ri, __ = estim.calculate_initial_powerflow(step_data)
        succ, x_V, x_scaling = estim.optimize_step(
            **step_data, Vnode_ri_ini=voltages_ri)
        self.assertTrue(succ, 'estimation succeeds')
        V, k, pos = estim.get_Vcx_factors(step_data['factordata'], x_V, x_scaling)
        self.assertAlmostEqual(
            # exclude slacks
            pfc.max_residual_current(model, V, positions=pos, kpq=k),
            0,
            delta=1e-12,
            msg='Inode is almost 0')
        branch_res = rt.calculate_branch_results(model, V, positions=pos)
        given_values = model.ivalues.set_index('id_of_batch')
        self.assertAlmostEqual(
            branch_res.loc['line_0'].I0_pu,
            given_values.loc['I_line_0'].I,
            places=7,
            msg='estimated electric current equals given electric current')

    def test_scale_pq_meet_i2(self):
        """Scale active and reactive power of consumer in order to meet the
        given current I at a terminal of an injection (measurement or setpoint).
        Given I is assigned to consumer."""
        model = make_model(
            self.grid0,
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
            model.factors.gen_factordata.index)
        expressions = estim.get_expressions(model, gen_factor_symbols)
        step_data = estim.get_step_data(
            model, expressions, objectives='I')
        _, voltages_ri, __ = estim.calculate_initial_powerflow(step_data)
        succ, x_V, x_scaling = estim.optimize_step(
            **step_data, Vnode_ri_ini=voltages_ri)
        self.assertTrue(succ, 'estimation succeeds')
        V, k, pos = estim.get_Vcx_factors(step_data['factordata'], x_V, x_scaling)
        inj_res = rt.calculate_injection_results(model, V, kpq=k)
        self.assertAlmostEqual(
            # exclude slacks
            pfc.max_residual_current(model, V, positions=pos, kpq=k),
            0,
            delta=1e-8,
            msg='Inode is almost 0')
        given_values = model.ivalues.set_index('id_of_batch')
        self.assertAlmostEqual(
            inj_res.loc['consumer'].I_pu,
            given_values.loc['I_consumer'].I,
            delta=1e-8,
            msg='estimated electric current equals given electric current')

    def test_scale_q_meet_v(self):
        """Scale reactive power of consumer in order to meet the
        given voltage V (measurement or setpoint).
        Given V is assigned to node 2 ('n_2')."""
        model = make_model(
            self.grid0,
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
            model.factors.gen_factordata.index)
        expressions = estim.get_expressions(model, gen_factor_symbols)
        step_data = estim.get_step_data(
            model, expressions, objectives='V')
        _, voltages_ri, __ = estim.calculate_initial_powerflow(step_data)
        succ, x_V, x_scaling = estim.optimize_step(
            **step_data, Vnode_ri_ini=voltages_ri)
        self.assertTrue(succ, 'estimation succeeds')
        V, k, pos = estim.get_Vcx_factors(
            step_data['factordata'], x_V, x_scaling)
        self.assertAlmostEqual(
            # exclude slacks
            pfc.max_residual_current(model, V, positions=pos, kpq=k),
            0,
            delta=1e-8,
            msg='Inode is almost 0')
        given_V_at_node = model.vvalues.set_index('id_of_node').loc['n_2']
        self.assertAlmostEqual(
            np.abs(V[given_V_at_node.index_of_node])[0],
            given_V_at_node.V,
            places=10,
            msg='estimated voltage equals given voltage')

class Estimate_minimal(unittest.TestCase):

    def test_empty_model(self):
        """estimate empty grid-model"""
        model = make_model()
        res, *tail = estim.estimate(model)
        self.assertIsInstance(
            res, tuple, 'estimate returns tuple for empty model')
        self.assertEqual(
            tail, [], 'estimate returns just initialization values')
        self.assertEqual(
            len(res),
            5,
            'estimate returns 4 values for the initialization step')
        self.assertEqual(
            res[0],
            -1,
            'index of estimation step is -1 for initial '
            'power flow calculation')
        self.assertEqual(
            res[1], True, 'estimate succeeds for empty model')
        self.assertEqual(
            res[2].shape,
            (0, 1),
            'estimate returns zero-length voltage vector for empty model')
        self.assertEqual(
            res[3].shape,
            (0, 2),
            'estimate returns zero-length scaling factor vectors (kp, kq) '
            'for a model without injections')

    def test_slack_only(self):
        """the only element is the slacknode"""
        vcx_slack = 0.95+0.02j
        model = make_model(grid.Slacknode('n_0', V=vcx_slack))
        res, *tail = estim.estimate(model)
        self.assertIsInstance(
            res, tuple, 'estimate returns tuple')
        self.assertEqual(
            tail, [], 'estimate returns just initialization values')
        self.assertEqual(
            len(res),
            5,
            'estimate returns 4 values for the initialization step')
        self.assertEqual(
            res[0],
            -1,
            'index of estimation step is -1 for initial '
            'power flow calculation')
        self.assertEqual(
            res[1],
            True,
            'estimate succeeds for a model having a slacknode only')
        self.assertEqual(
            res[2],
            [vcx_slack],
            'estimate returns a one-element voltage vector with given '
            'slack-voltage for a slacknode-only model')
        self.assertEqual(
            res[3].shape,
            (0, 2),
            'estimate returns zero-length scaling factor vectors (kp, kq) '
            'for a model without injections')

class Estimate_injection(unittest.TestCase):
    """runs basic tests with one (slack-) node and one injection"""

    def test_slack_inj(self):
        """one slacknode, one injection"""
        vcx_slack = 0.95+0.02j
        s = 30.+10.j
        model = make_model(
            grid.Slacknode('n_0', V=vcx_slack),
            grid.Injection('consumer', 'n_0', P10=s.real, Q10=s.imag))
        res, *tail = estim.estimate(model)
        self.assertIsInstance(
            res, tuple, 'estimate returns tuple')
        self.assertEqual(
            tail, [], 'estimate returns just initialization values')
        self.assertEqual(
            len(res),
            5,
            'estimate returns 4 values for the initialization step')
        self.assertEqual(
            res[0],
            -1,
            'index of estimation step is -1 for initial '
            'power flow calculation')
        self.assertEqual(
            res[1], True, 'estimate succeeds')
        self.assertEqual(
            res[2],
            [vcx_slack],
            'estimate returns a one-element voltage vector with given '
            'slack-voltage for a model with just one slacknode')
        self.assertTrue(
            (res[3]==[[1., 1.]]).all(),
            'estimate returns two column scaling factor vectors (kp, kq) '
            'of [[1., 1.]] for initialization step')

    def test_pvalue_objQ(self):
        """one slacknode, one scalabel injection, optimize reactive power
        without Qvalue, yields just power flow result"""
        vcx_slack = 0.95+0.02j
        s = 30.+10.j
        model = make_model(
            grid.Slacknode('n_0', V=vcx_slack),
            grid.Injection('consumer', 'n_0', P10=s.real, Q10=s.imag),
            # scaling
            grid.Defk(id='kp', step=0),
            grid.Klink(
                id_of_injection='consumer',
                part='p',
                id_of_factor='kp',
                step=0),
            # measurement
            grid.PValue(id_of_batch='p_of_consumter', P=9, direction=1.),
            grid.Output(id_of_batch='p_of_consumer', id_of_device='consumer'))
        init, res = estim.estimate(
            model,
            step_params=[dict(objectives='Q')])
        self.assertIsInstance(
            res, tuple, 'estimate returns tuple')
        self.assertEqual(
            len(res),
            5,
            'estimate returns 4 values for the initialization step')
        self.assertEqual(
            res[0],
            0,
            'index of estimation step is 0 for first '
            'optimization')
        self.assertEqual(
            res[1], True, 'estimate succeeds')
        self.assertEqual(
            res[2],
            [vcx_slack],
            'estimate returns a one-element voltage vector with given '
            'slack-voltage for a model with just one slacknode')
        self.assertTrue(
            (res[3]==[[1., 1.]]).all(),
            'estimate returns two column scaling factor vectors (kp, kq) '
            'of [[1., 1.]] equivalent to initialization step')

    def test_pvalue_objP(self):
        """one slacknode, one scalabel injection, optimize active power
        with Pvalue and scalable active power P"""
        vcx_slack = 0.95+0.02j
        s = 30.+10.j
        model = make_model(
            grid.Slacknode('n_0', V=vcx_slack),
            grid.Injection('consumer', 'n_0', P10=s.real, Q10=s.imag),
            # scaling, define scaling factor
            grid.Defk(id='kp', step=0),
            # link scaling factor to active power of consumer
            grid.Klink(
                id_of_injection='consumer',
                part='p',
                id_of_factor='kp',
                step=0),
            # measurement
            grid.PValue(id_of_batch='p_of_consumer', P=20., direction=1.),
            grid.Output(id_of_batch='p_of_consumer', id_of_device='consumer'))
        init, res = estim.estimate(
            model,
            step_params=[dict(objectives='P')])
        self.assertIsInstance(
            res, tuple, 'estimate returns tuple')
        self.assertEqual(
            len(res),
            5,
            'estimate returns 4 values for the otimization step')
        self.assertEqual(
            res[0],
            0,
            'index of estimation step is 0 for first optimization')
        self.assertEqual(
            res[1], True, 'estimate succeeds')
        self.assertEqual(
            res[2],
            [vcx_slack],
            'estimate returns a one-element voltage vector with given '
            'slack-voltage for a model with just one slacknode')
        assert_array_almost_equal(
            res[3],
            [[20./s.real, 1.]],
            decimal=8,
            err_msg="P is scaled, Q is not scaled")

    def test_pvalue_objP2(self):
        """one slacknode, one scalabel injection, optimize active power
        with Pvalue and scalable PQ, PQ share one scaling factor"""
        vcx_slack = 0.95+0.02j
        s = 30.+10.j
        Pval = 20.
        model = make_model(
            grid.Slacknode('n_0', V=vcx_slack),
            grid.Injection('consumer', 'n_0', P10=s.real, Q10=s.imag),
            # scaling, define scaling factor
            grid.Defk(id='kpq', step=0),
            # link scaling factor to active and reactive power of consumer
            grid.Klink(
                id_of_injection='consumer',
                part=('p','q'),
                id_of_factor=('kpq', 'kpq'),
                step=0),
            # measurement
            grid.PValue(id_of_batch='p_of_consumer', P=Pval, direction=1.),
            grid.Output(id_of_batch='p_of_consumer', id_of_device='consumer'))
        init, res = estim.estimate(
            model,
            step_params=[dict(objectives='P')])
        self.assertIsInstance(
            res, tuple, 'estimate returns tuple')
        self.assertEqual(
            len(res),
            5,
            'estimate returns 4 values for the optimization step')
        self.assertEqual(
            res[0],
            0,
            'index of estimation step is 0 for first optimization')
        self.assertEqual(
            res[1], True, 'estimate succeeds')
        self.assertEqual(
            res[2],
            [vcx_slack],
            'estimate returns a one-element voltage vector with given '
            'slack-voltage for a model with just one slacknode')
        kpq = Pval/s.real
        assert_array_almost_equal(
            res[3],
            [[kpq, kpq]],
            decimal=8,
            err_msg="P and Q are scaled with the same factor")

    def test_pvalue_objP_constrQ(self):
        """one slacknode, one scalabel injection, optimize active power
        with Pvalue and scalable PQ, active power of injection is not scaled
        as active and reactive power share one scaling factor but reactive
        power at measured terminal where Qvalue is assigned to,
        is constrained, meaning it is kept equal to previous step
        (in this case the initialization step), the value of Qvalue is not
        taken into consideration, the value of Pvalue is part of the
        objective function"""
        vcx_slack = 0.95+0.02j
        s = 30.+10.j
        Pval = 20.
        Qval = 4.
        model = make_model(
            grid.Slacknode('n_0', V=vcx_slack),
            grid.Injection('consumer', 'n_0', P10=s.real, Q10=s.imag),
            # scaling, define scaling factor
            grid.Defk(id='kpq', step=0),
            # link scaling factor to active and reactive power of consumer
            grid.Klink(
                id_of_injection='consumer',
                part=('p','q'),
                id_of_factor=('kpq', 'kpq'),
                step=0),
            # measurements
            grid.PValue(id_of_batch='pq_of_consumer', P=Pval),
            grid.QValue(id_of_batch='pq_of_consumer', Q=Qval),
            grid.Output(id_of_batch='pq_of_consumer', id_of_device='consumer'))
        init, res = estim.estimate(
            model,
            step_params=[dict(objectives='P', constraints='Q')])
        self.assertIsInstance(
            res, tuple, 'estimate returns tuple')
        self.assertEqual(
            len(res),
            5,
            'estimate returns 4 values for the optimization step')
        self.assertEqual(
            res[0],
            0,
            'index of estimation step is 0 for first optimization')
        self.assertEqual(
            res[1], True, 'estimate succeeds')
        self.assertEqual(
            res[2],
            [vcx_slack],
            'estimate returns a one-element voltage vector with given '
            'slack-voltage for a model with just one slacknode')
        assert_array_almost_equal(
            res[3],
            [[1., 1.]],
            decimal=8,
            err_msg="P and Q shall not be scaled because of 'Q'-constraint")

    def test_pqvalue_objPQ(self):
        """scale active power P and reactive power Q in order to match them
        with measurements P and Q"""
        vcx_slack = 0.95+0.02j
        s = 30.+10.j
        Pval = 20.
        Qval = 4.
        model = make_model(
            grid.Slacknode('n_0', V=vcx_slack),
            grid.Injection('consumer', 'n_0', P10=s.real, Q10=s.imag),
            # scaling, define scaling factors
            grid.Defk(id='kp', step=0),
            grid.Defk(id='kq', step=0),
            # link scaling factors to active and reactive power of consumer
            grid.Klink(
                id_of_injection='consumer',
                part=('p','q'),
                id_of_factor=('kp', 'kq'),
                step=0),
            # measurements
            grid.PValue(id_of_batch='pq_of_consumer', P=Pval),
            grid.QValue(id_of_batch='pq_of_consumer', Q=Qval),
            grid.Output(id_of_batch='pq_of_consumer', id_of_device='consumer'))
        init, res = estim.estimate(
            model,
            step_params=[dict(objectives='PQ')])
        self.assertIsInstance(
            res, tuple, 'estimate returns tuple')
        self.assertEqual(
            len(res),
            5,
            'estimate returns 4 values for the optimization step')
        self.assertEqual(
            res[0],
            0,
            'index of estimation step is 0 for first optimization')
        self.assertEqual(
            res[1], True, 'estimate succeeds')
        self.assertEqual(
            res[2],
            [vcx_slack],
            'estimate returns a one-element voltage vector with given '
            'slack-voltage for a model with just one slacknode')
        kpq = np.array([[Pval, Qval]]) / np.array([[s.real, s.imag]])
        assert_array_almost_equal(
            res[3],
            kpq,
            decimal=8,
            err_msg="separate scaling factors for P and Q shall match")

    def test_pqvalue_objI(self):
        """scale active power P and reactive power Q in order to match them
        with measurements P and Q"""
        vcx_slack = 0.95+0.02j
        pq_abs = 30/(2**.5)
        # apparent power, |s| == 30
        s = complex(pq_abs, pq_abs)  # for three phases
        Ival = 5.                    # for one phase
        model = make_model(
            grid.Slacknode('n_0', V=vcx_slack),
            # power independant of voltage
            grid.Injection('consumer', 'n_0', P10=s.real, Q10=s.imag),
            # scaling, define scaling factors
            grid.Defk(id='kpq', step=0),
            # link scaling factors to active and reactive power of consumer
            grid.Klink(
                id_of_injection='consumer',
                part=('p','q'),
                id_of_factor=('kpq', 'kpq'),
                step=0),
            # measurements
            grid.IValue(id_of_batch='i_of_consumer', I=Ival),
            grid.Output(id_of_batch='i_of_consumer', id_of_device='consumer'))
        init, res = estim.estimate(
            model,
            step_params=[dict(objectives='I')])
        self.assertIsInstance(
            res, tuple, 'estimate returns tuple')
        self.assertEqual(
            len(res),
            5,
            'estimate returns 4 values for the optimization step')
        self.assertEqual(
            res[0],
            0,
            'index of estimation step is 0 for first optimization')
        self.assertEqual(
            res[1], True, 'estimate succeeds')
        self.assertEqual(
            res[2],
            [vcx_slack],
            'estimate returns a one-element voltage vector with given '
            'slack-voltage for a model with just one slacknode')
        kpq = .5 * abs(vcx_slack)
        assert_array_almost_equal(
            res[3],
            [[kpq, kpq]],
            decimal=8,
            err_msg="scaling factor for P and Q shall make Ivalue match")

class Estimate_branch_injection(unittest.TestCase):
    """runs basic tests with two nodes, one branch and one injection,
    flow mesurements are placed at terminal of a branch"""

    def test_no_scaling(self):
        """one slacknode, one branch, one injection"""
        vcx_slack = 0.95+0.02j
        s = 30.+10.j
        model = make_model(
            grid.Slacknode('n_0', V=vcx_slack),
            grid.Branch('line', 'n_0', 'n_1', y_lo=1e3-1e3j, y_tr=1e-6+1e-6j),
            grid.Injection('consumer', 'n_1', P10=s.real, Q10=s.imag))
        res, *tail = estim.estimate(model)
        self.assertIsInstance(
            res, tuple, 'estimate returns tuple')
        self.assertEqual(
            tail, [], 'estimate returns just initialization values')
        self.assertEqual(
            len(res),
            5,
            'estimate returns 4 values for the initialization step')
        self.assertEqual(
            res[0],
            -1,
            'index of estimation step is -1 for initial '
            'power flow calculation')
        self.assertTrue(res[1], 'estimate succeeds')
        self.assertEqual(
            res[2].shape,
            (2,1),
            'estimate returns a two voltages')
        self.assertEqual(
            res[2][0],
            [vcx_slack],
            'first voltage is slack voltage')
        assert_array_almost_equal(
            res[3],
            [[1., 1.]],
            decimal=8,
            err_msg=
            'estimate returns two column scaling factor vectors (kp, kq) '
            'of [[1., 1.]] for initialization step')

    def test_pvalue_objQ(self):
        """one slacknode, one scalabel injection, optimize reactive power
        without Qvalue, yields just power flow result"""
        vcx_slack = 0.95+0.02j
        s = 30.+10.j
        model = make_model(
            grid.Slacknode('n_0', V=vcx_slack),
            grid.Branch('line', 'n_0', 'n_1', y_lo=1e3-1e3j, y_tr=1e-6+1e-6j),
            grid.Injection('consumer', 'n_1', P10=s.real, Q10=s.imag),
            # scaling
            grid.Defk(id='kp', step=0),
            grid.Klink(
                id_of_injection='consumer',
                part='p',
                id_of_factor='kp',
                step=0),
            # measurement
            grid.PValue(id_of_batch='p_at_line', P=9, direction=1.),
            grid.Output(
                id_of_batch='p_at_line',
                id_of_device='line',
                id_of_node='n_0'))
        init, res = estim.estimate(
            model,
            step_params=[dict(objectives='Q')])
        self.assertIsInstance(
            res, tuple, 'estimate returns tuple')
        self.assertEqual(
            len(res),
            5,
            'estimate returns 4 values for the initialization step')
        self.assertEqual(
            res[0],
            0,
            'index of estimation step is 0 for first '
            'optimization')
        self.assertTrue(res[1], 'estimate succeeds')
        self.assertEqual(
            res[2].shape,
            (2,1),
            'estimate returns a two voltages')
        self.assertEqual(
            res[2][0],
            [vcx_slack],
            'first voltage is slack voltage')
        assert_array_almost_equal(
            res[3],
            [[1., 1.]],
            decimal=8,
            err_msg=
            'estimate returns two column scaling factor vectors (kp, kq) '
            'of [[1., 1.]] equivalent to initialization step')

    def test_qvalue_objQ(self):
        """one slacknode, one scalabel injection, optimize reactive power
        with Qvalue"""
        vcx_slack = 0.95+0.02j
        s = 30.+10.j
        model = make_model(
            grid.Slacknode('n_0', V=vcx_slack),
            grid.Branch('line', 'n_0', 'n_1', y_lo=1e3-1e3j, y_tr=1e-6+1e-6j),
            grid.Injection('consumer', 'n_1', P10=s.real, Q10=s.imag),
            # scaling
            grid.Defk(id='kq'),
            grid.Klink(
                id_of_injection='consumer', part='q', id_of_factor='kq'),
            # measurement
            grid.QValue(id_of_batch='q_at_line', Q=4, direction=1.),
            grid.Output(
                id_of_batch='q_at_line',
                id_of_device='line',
                id_of_node='n_0'))
        init, res = estim.estimate(model, step_params=[dict(objectives='Q')])
        self.assertIsInstance(res, tuple, 'estimate returns tuple')
        self.assertEqual(
            len(res),
            5,
            'estimate returns 4 values for the initialization step')
        self.assertEqual(
            res[0],
            0,
            'index of estimation step is 0 for first '
            'optimization')
        self.assertTrue(res[1], 'estimate succeeds')
        self.assertEqual(
            res[2].shape,
            (2,1),
            'estimate returns a two voltages')
        self.assertEqual(
            res[2][0],
            [vcx_slack],
            'first voltage is slack voltage')
        assert_array_almost_equal(
            res[3],
            [[1., 0.38290188]],
            decimal=8,
            err_msg=
            'estimate returns two column scaling factor vectors (kp, kq) '
            'of [[1., 1.]] equivalent to initialization step')

    def test_pvalue_objP(self):
        """two nodes (one slacknode), one branch, one scalabel injection,
        optimize active power with Pvalue and scalable active power P"""
        vcx_slack = 0.95+0.02j
        S = 30.+10.j
        Pval = 20
        model = make_model(
            grid.Slacknode('n_0', V=vcx_slack),
            grid.Branch('line', 'n_0', 'n_1', y_lo=1e3-1e3j, y_tr=1e-6+1e-6j),
            grid.Injection('consumer', 'n_1', P10=S.real, Q10=S.imag),
            # scaling, define scaling factor
            grid.Defk(id='kp', step=0),
            # link scaling factor to active power of consumer
            grid.Klink(
                id_of_injection='consumer',
                part='p',
                id_of_factor='kp',
                step=0),
            # measurement
            grid.PValue(id_of_batch='p_of_line', P=Pval),
            grid.Output(
                id_of_batch='p_of_line',
                id_of_device='line',
                id_of_node='n_0'))
        init, res = estim.estimate(model, step_params=[dict(objectives='P')])
        # check
        self.assertTrue(res[1], 'estimate succeeds')
        # maximum of residual node currents without slacknode
        max_dev = pfc.max_residual_current(
            model, res[2], positions=res[4], kpq=res[3],
            loadcurve='interpolated')
        self.assertLess(max_dev, 1e-8, 'residual node current is 0')
        branch_res = rt.calculate_branch_results(
            model, res[2], positions=res[4])
        P_n0_line_pu = branch_res.loc['line','P0_pu']
        self.assertAlmostEqual(
            P_n0_line_pu,
            Pval,
            places=8,
            msg='estimated value of active power equals measured value '
            'of active power at branch')
        # scaling factor for reactive power kq
        kq = res[3][0,1]
        self.assertAlmostEqual(kq, 1., 'reactive power is not scaled')

    def test_pvalue_objP2(self):
        """two nodes (one slacknode), one branch, one scalabel injection,
        optimize active power with Pvalue and scalable active power P,
        P and Q are scaled with the same factor"""
        vcx_slack = 0.95+0.02j
        S = 30.+10.j
        Pval = 20
        model = make_model(
            grid.Slacknode('n_0', V=vcx_slack),
            grid.Branch('line', 'n_0', 'n_1', y_lo=1e3-1e3j, y_tr=1e-6+1e-6j),
            grid.Injection('consumer', 'n_1', P10=S.real, Q10=S.imag),
            # scaling, define scaling factor
            grid.Defk(id=('kpq'), step=0),
            # link scaling factor to active power of consumer
            grid.Klink(
                id_of_injection='consumer',
                part='pq',
                id_of_factor=('kpq', 'kpq'),
                step=0),
            # measurement
            grid.PValue(id_of_batch='p_of_line', P=Pval),
            grid.Output(
                id_of_batch='p_of_line',
                id_of_device='line',
                id_of_node='n_0'))
        init, res = estim.estimate(model, step_params=[dict(objectives='P')])
        # check
        self.assertTrue(res[1], 'estimate succeeds')
        max_dev = pfc.max_residual_current(
            model, res[2], positions=res[4], kpq=res[3],
            loadcurve='interpolated')
        self.assertLess(max_dev, 1e-8, 'residual node current is 0')
        branch_res = rt.calculate_branch_results(
            model, res[2], positions=res[4])
        P_n0_line_pu = branch_res.loc['line','P0_pu']
        self.assertAlmostEqual(
            P_n0_line_pu,
            Pval,
            places=8,
            msg='estimated value of active power equals measured value '
            'of active power at branch')
        # scaling factor for reactive power kq
        kp, kq = res[3][0]
        self.assertAlmostEqual(
            kp,
            kq,
            'active power and reactive power are scaled with the same factor')

    def test_pqvalue_objPQ(self):
        """scale active power P and reactive power Q in order to match
        measurements P and Q"""
        vcx_slack = 0.95+0.02j
        s = 30.+10.j
        Pval = 20.
        Qval = 4.
        model = make_model(
            grid.Slacknode('n_0', V=vcx_slack),
            grid.Branch('line', 'n_0', 'n_1', y_lo=1e3-1e3j, y_tr=1e-6+1e-6j),
            grid.Injection('consumer', 'n_1', P10=s.real, Q10=s.imag),
            # scaling, define scaling factors
            grid.Defk(id='kp', step=0),
            grid.Defk(id='kq', step=0),
            # link scaling factors to active and reactive power of consumer
            grid.Klink(
                id_of_injection='consumer',
                part=('p','q'),
                id_of_factor=('kp', 'kq'),
                step=0),
            # measurements
            grid.PValue(id_of_batch='pq_of_line', P=Pval),
            grid.QValue(id_of_batch='pq_of_line', Q=Qval),
            grid.Output(
                id_of_batch='pq_of_line',
                id_of_device='line',
                id_of_node='n_0'))
        init, res = estim.estimate(
            model,
            step_params=[dict(objectives='PQ')])
        # check
        self.assertTrue(
            res[1], 'estimate succeeds')
        max_dev = pfc.max_residual_current(
            model, res[2], positions=res[4], kpq=res[3],
            loadcurve='interpolated')
        self.assertLess(max_dev, 1e-8, 'residual node current is 0')
        branch_res = rt.calculate_branch_results(
            model, res[2], positions=res[4])
        P_n0_line_pu, Q_n0_line_pu = branch_res.loc['line',['P0_pu', 'Q0_pu']]
        self.assertAlmostEqual(
            P_n0_line_pu,
            Pval,
            places=8,
            msg='estimated value of active power equals measured value '
            'of active power at branch')
        self.assertAlmostEqual(
            Q_n0_line_pu,
            Qval,
            places=8,
            msg='estimated value of reactive power equals measured value '
            'of reactive power at branch')

    def test_pqvalue_objPQ_two_steps(self):
        """scale active power P and reactive power Q in two steps in order
        to match measurements P and Q"""
        vcx_slack = 0.95+0.02j
        s = 30.+10.j
        Pval = 20.
        Qval = 4.
        model = make_model(
            grid.Slacknode('n_0', V=vcx_slack),
            grid.Branch('line', 'n_0', 'n_1', y_lo=1e3-1e3j, y_tr=1e-6+1e-6j),
            grid.Injection('consumer', 'n_1', P10=s.real, Q10=s.imag),
            # scaling, define scaling factors
            grid.Defk(id='kp', step=(0,1)),
            grid.Defk(id='kq', step=(0,1)),
            # link scaling factors to active and reactive power of consumer
            grid.Klink(
                id_of_injection='consumer',
                part=('p','q'),
                id_of_factor=('kp', 'kq'),
                step=(0,1)),
            # measurements
            grid.PValue(id_of_batch='pq_of_line', P=Pval),
            grid.QValue(id_of_batch='pq_of_line', Q=Qval),
            grid.Output(
                id_of_batch='pq_of_line',
                id_of_device='line',
                id_of_node='n_0'))
        init, res0, res = estim.estimate(
            model,
            step_params=[
                dict(objectives='P'),
                dict(objectives='Q', constraints='P')])
        # check
        self.assertTrue(res[1], 'estimate succeeds')
        max_dev = pfc.max_residual_current(
            model, res[2], positions=res[4], kpq=res[3],
            loadcurve='interpolated')
        self.assertLess(max_dev, 1e-8, 'residual node current is 0')
        branch_res = rt.calculate_branch_results(
            model, res[2], positions=res[4])
        P_n0_line_pu, Q_n0_line_pu = branch_res.loc['line',['P0_pu', 'Q0_pu']]
        self.assertAlmostEqual(
            P_n0_line_pu,
            Pval,
            places=8,
            msg='estimated value of active power equals measured value '
            'of active power at branch')
        self.assertAlmostEqual(
            Q_n0_line_pu,
            Qval,
            places=8,
            msg='estimated value of reactive power equals measured value '
            'of reactive power at branch')

    def test_pqvalue_objI(self):
        """scale active power P and reactive power Q in order to match
        measurements for electric current I"""
        vcx_slack = 0.95+0.02j
        pq_abs = 30/(2**.5)
        # apparent power, |s| == 30
        s = complex(pq_abs, pq_abs)  # for three phases
        Ival = 5.                    # for one phase
        model = make_model(
            grid.Slacknode('n_0', V=vcx_slack),
            grid.Branch('line', 'n_0', 'n_1', y_lo=1e3-1e3j, y_tr=1e-6+1e-6j),
            grid.Injection('consumer', 'n_1', P10=s.real, Q10=s.imag),
            # scaling, define scaling factors
            grid.Defk(id='kpq', step=0),
            # link scaling factors to active and reactive power of consumer
            grid.Klink(
                id_of_injection='consumer',
                part=('p','q'),
                id_of_factor=('kpq', 'kpq'),
                step=0),
            # measurements
            grid.IValue(id_of_batch='i_of_consumer', I=Ival),
            grid.Output(id_of_batch='i_of_consumer', id_of_device='consumer'))
        init, res = estim.estimate(
            model,
            step_params=[dict(objectives='I')])
        # check
        self.assertTrue(res[1], 'estimate succeeds')
        max_dev = pfc.max_residual_current(
            model, res[2], positions=res[4], kpq=res[3],
            loadcurve='interpolated')
        self.assertLess(max_dev, 1e-8, 'residual node current is 0')
        branch_res = rt.calculate_branch_results(
            model, res[2], positions=res[4])
        I_n0_line_pu = branch_res.loc['line', 'I0_pu']
        self.assertAlmostEqual(
            I_n0_line_pu,
            Ival,
            places=8,
            msg='estimated value of current magnitude equals measured value '
            'of current magnitude at branch')

    def test_qvalue_objV(self):
        """scale reactive power Q in order to match given V"""
        vcx_slack = 1.+0.j
        # apparent power, |s| == 30
        Vval = 1.02
        model = make_model(
            grid.Slacknode('n_0', V=vcx_slack),
            grid.Branch('line_0', 'n_0', 'n_1', y_lo=1e3-1e3j),
            grid.Branch('line_1', 'n_1', 'n_2', y_lo=1e3-1e3j),
            grid.Injection('consumer', 'n_2', P10=30.0, Q10=10.0),
            # scaling, define scaling factors
            grid.Defk(id='kq'),
            # link scaling factors to active and reactive power of consumer
            grid.Klink(
                id_of_injection='consumer', part='q', id_of_factor='kq'),
            # measurement/setpoint
            grid.Vvalue(id_of_node='n_2', V=Vval))
        init, res = estim.estimate(
            model, step_params=[dict(objectives='V')])
        # check
        self.assertTrue(res[1], 'estimate succeeds')
        max_dev = pfc.max_residual_current(
            model, res[2], positions=res[4], kpq=res[3],
            loadcurve='interpolated')
        self.assertLess(max_dev, 1e-8, 'residual node current is 0')
        # check voltage
        given_V_at_node = model.vvalues.set_index('id_of_node').loc['n_2']
        self.assertAlmostEqual(
            np.abs(res[2][given_V_at_node.index_of_node])[0],
            Vval,
            places=10,
            msg='estimated voltage equals given voltage')

    def test_qvalue_objV_discrete(self):
        """scale reactive power Q in order to match given V,
        discrete scaling factor for injection"""
        vcx_slack = 1.+0.j
        # apparent power, |s| == 30
        Vval = 1.02
        model = make_model(
            grid.Slacknode('n_0', V=vcx_slack),
            grid.Branch('line_0', 'n_0', 'n_1', y_lo=1e3-1e3j),
            grid.Branch('line_1', 'n_1', 'n_2', y_lo=1e3-1e3j),
            grid.Injection('consumer', 'n_2', P10=30.0, Q10=20.0),
            # scaling, define scaling factors
            grid.Defk(id='kq', is_discrete=True),
            # link scaling factors to active and reactive power of consumer
            grid.Klink(
                id_of_injection='consumer', part='q', id_of_factor='kq'),
            # measurement/setpoint
            grid.Vvalue(id_of_node='n_2', V=Vval))
        init, res = estim.estimate(
            model, step_params=[dict(objectives='V')])
        # check
        self.assertTrue(res[1], 'estimate succeeds')
        max_dev = pfc.max_residual_current(
            model, res[2], positions=res[4], kpq=res[3],
            loadcurve='interpolated')
        self.assertLess(max_dev, 1e-8, 'residual node current is 0')
        # check voltage
        given_V_at_node = model.vvalues.set_index('id_of_node').loc['n_2']
        self.assertAlmostEqual(
            np.abs(res[2][given_V_at_node.index_of_node])[0],
            Vval,
            places=2,
            msg='estimated voltage equals given voltage')
        self.assertAlmostEqual(
            res[3][0,1] % 1,
            0.,
            places=14,
            msg='kq shall be discrete')

    def test_qvalue_objV_discrete_term_factor(self):
        """select tap position (discrete terminal factor) in order to match
        given V"""
        vcx_slack = 1.+0.j
        # apparent power, |s| == 30
        Vval = 1.02
        model = make_model(
            grid.Slacknode('n_0', V=vcx_slack),
            grid.Branch('branch_0', 'n_0', 'n_1', y_lo=1e3-1e3j),
            grid.Branch('branch_1', 'n_1', 'n_2', y_lo=1e3-1e3j),
            grid.Injection('consumer', 'n_2', P10=30.0, Q10=5.0),
            #  taps
            grid.Deft(
                'tap_branch_1', type='var', min=-16, max=16,
                value=0, m=-.1/16, n=1., is_discrete=True),
            grid.Tlink(
                id_of_node='n_1',
                id_of_branch='branch_1',
                id_of_factor='tap_branch_1'),
            # measurement/setpoint
            grid.Vvalue(id_of_node='n_2', V=Vval))
        init, res = estim.estimate(
            model, step_params=[dict(objectives='V')])
        # check
        self.assertTrue(res[1], 'estimate succeeds')
        num_res = rt.calculate_electric_data(
            model, res[2], kpq=res[3], positions=res[4])
        # maximum of residual node currents without slacknode
        max_dev = pfc.max_residual_current(
            model, res[2], kpq=res[3], positions=res[4])
        self.assertLess(max_dev, 1e-8, 'residual node current is 0')
        # check voltage
        self.assertAlmostEqual(
            # get absolute of calculated voltage at node 'n_2'
            num_res['nodes'].V_pu.n_2,
            Vval,
            places=2,
            msg='estimated voltage equals given voltage')
        self.assertAlmostEqual(
            res[4][0,0] % 1,
            0.,
            places=14,
            msg='tap_branch_1 shall be discrete')

class Term(unittest.TestCase):
    """
    schema:
    ::
        slack ---------> injection
                P=4       P10=5
                Q=3       Q10=2
        #. Defk(id=(kp kq))
        #. Klink(id_of_injection=injection part=(p q) id_of_factor=(kp kq))
        #. Defoterm(args=(kp kq))
        #. Defoterm(args=(kp kq), weight=3.0, step=2)"""
    model = make_model(
        grid.Slacknode('slack'),
        grid.Injection('injection', 'slack', P10=5., Q10=2.),
        grid.PValue(id_of_batch='b', P=4),
        grid.QValue(id_of_batch='b', Q=3),
        grid.Output(id_of_batch='b', id_of_device='injection'),
        grid.Defk(id=('kp', 'kq')),
        grid.Klink(
            id_of_injection='injection',
            part=('p', 'q'),
            id_of_factor=('kp', 'kq')),
        grid.Defoterm(args=('kp', 'kq')),
        grid.Defoterm(args=('kp', 'kq'), weight=3.0, step=2))

    def test_diff_pq(self):
        """runs basic tests with diff-term, also test weight"""
        res_ = estim.estimate(
            self.model,
            step_params=[
                # meet P and Q
                dict(objectives='PQ'),
                # meet P and Q + keep diff 'kp - kq' small
                dict(objectives='PQT'),
                dict(objectives='PQT')])
        ini, *res = rt.get_printable_results(self.model, res_)
        inj_ini = ini['injections']
        assert_array_almost_equal(
            inj_ini[['P_pu', 'Q_pu', 'kp', 'kq']].to_numpy()[0],
            #P   Q   kp  kq
            [5., 2., 1., 1.])
        # optimized to meet P=4 and Q=3
        inj_0 = res[0]['injections']
        P_pu_0, Q_pu_0, kp_0, kq_0 = (
            inj_0[['P_pu', 'Q_pu', 'kp', 'kq']].to_numpy()[0])
        assert_array_almost_equal(
            [P_pu_0, Q_pu_0],
            #P   Q
            [4., 3.])
        self.assertLess(kp_0, 1.)
        self.assertGreater(kq_0, 1.)
        # meet P and Q + keep diff 'kp - kq' small
        #   scales neither P nor Q sufficiently in order to meet given P and Q,
        #   kp is to great, kq to small
        inj_1 = res[1]['injections']
        P_pu_1, Q_pu_1, kp_1, kq_1 = (
            inj_1[['P_pu', 'Q_pu', 'kp', 'kq']].to_numpy()[0])
        self.assertLess(kp_1, 1.)
        self.assertLess(kp_0, kp_1)
        self.assertGreater(kq_1, 1.)
        self.assertGreater(kq_0, kq_1)
        # meet P and Q + keep diff 'kp - kq' small
        #   scales neither P nor Q sufficiently kp is to great, kq to small
        #   deviation is greater than previous optimization step
        inj_2 = res[2]['injections']
        P_pu_2, Q_pu_2, kp_2, kq_2 = (
            inj_2[['P_pu', 'Q_pu', 'kp', 'kq']].to_numpy()[0])
        self.assertLess(kp_2, 1.)
        self.assertLess(kp_1, kp_2)
        self.assertGreater(kq_2, 1.)
        self.assertGreater(kq_1, kq_2)

if __name__ == '__main__':
    unittest.main()
