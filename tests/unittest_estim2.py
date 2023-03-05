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
import dssex.estim as estim
import dssex.pfcnum as pfc
from numpy.testing import assert_array_almost_equal
from numpy.linalg import norm
from egrid import make_model

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
            4,
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
            4,
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
    """runs basic tests with one (slack-) node and one injection,
    flow measurements are placed at injection"""

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
            4,
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
            grid.Link(objid='consumer', id='kp', part='p', step=0),
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
            4,
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
            grid.Link(objid='consumer', id='kp', part='p', step=0),
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
            4,
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
            grid.Link(
                objid='consumer',
                id=('kpq', 'kpq'),
                part=('p','q'),
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
            4,
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
            grid.Link(
                objid='consumer',
                id=('kpq', 'kpq'),
                part=('p','q'),
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
            4,
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
            grid.Link(
                objid='consumer',
                id=('kp', 'kq'),
                part=('p','q'),
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
            4,
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
            grid.Link(
                objid='consumer',
                id=('kpq', 'kpq'),
                part=('p','q'),
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
            4,
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
            4,
            'estimate returns 4 values for the initialization step')
        self.assertEqual(
            res[0],
            -1,
            'index of estimation step is -1 for initial '
            'power flow calculation')
        self.assertEqual(
            res[1], True, 'estimate succeeds')
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
            grid.Link(objid='consumer', id='kp', part='p', step=0),
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
            4,
            'estimate returns 4 values for the initialization step')
        self.assertEqual(
            res[0],
            0,
            'index of estimation step is 0 for first '
            'optimization')
        self.assertEqual(
            res[1], True, 'estimate succeeds')
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
            grid.Defk(id='kp'),
            # link scaling factor to active power of consumer
            grid.Link(objid='consumer', id='kp', part='p'),
            # measurement
            grid.PValue(id_of_batch='p_of_line', P=Pval),
            grid.Output(
                id_of_batch='p_of_line',
                id_of_device='line',
                id_of_node='n_0'))
        init, res = estim.estimate(model, step_params=[dict(objectives='P')])
        # check
        self.assertEqual(
            res[1], True, 'estimate succeeds')
        ed = pfc.calculate_electric_data(model, res[2], res[3])
        # maximum of residual node currents without slacknode
        max_dev = norm(
            ed.residual_node_current()[model.count_of_slacks:], np.inf)
        self.assertLess(max_dev, 1e-8, 'residual node current is 0')
        P_n0_line_pu = ed.branch().loc['line','P0_pu']
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
            grid.Defk(id=('kpq')),
            # link scaling factor to active power of consumer
            grid.Link(objid='consumer', id=('kpq', 'kpq'), part='pq'),
            # measurement
            grid.PValue(id_of_batch='p_of_line', P=Pval),
            grid.Output(
                id_of_batch='p_of_line',
                id_of_device='line',
                id_of_node='n_0'))
        init, res = estim.estimate(model, step_params=[dict(objectives='P')])
        # check
        self.assertEqual(
            res[1], True, 'estimate succeeds')
        ed = pfc.calculate_electric_data(model, res[2], res[3])
        # maximum of residual node currents without slacknode
        max_dev = norm(
            ed.residual_node_current()[model.count_of_slacks:], np.inf)
        self.assertLess(max_dev, 1e-8, 'residual node current is 0')
        P_n0_line_pu = ed.branch().loc['line','P0_pu']
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
        """scale active power P and reactive power Q in order to match them
        with measurements P and Q"""
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
            grid.Link(
                objid='consumer',
                id=('kp', 'kq'),
                part=('p','q'),
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
        self.assertEqual(
            res[1], True, 'estimate succeeds')
        ed = pfc.calculate_electric_data(model, res[2], res[3])
        # maximum of residual node currents without slacknode
        max_dev = norm(
            ed.residual_node_current()[model.count_of_slacks:], np.inf)
        self.assertLess(max_dev, 1e-8, 'residual node current is 0')
        P_n0_line_pu, Q_n0_line_pu = ed.branch().loc['line',['P0_pu', 'Q0_pu']]
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
        """scale active power P and reactive power Q in order to match them
        with measurements P and Q"""
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
            grid.Link(
                objid='consumer',
                id=('kpq', 'kpq'),
                part=('p','q'),
                step=0),
            # measurements
            grid.IValue(id_of_batch='i_of_consumer', I=Ival),
            grid.Output(id_of_batch='i_of_consumer', id_of_device='consumer'))
        init, res = estim.estimate(
            model,
            step_params=[dict(objectives='I')])
        # check
        self.assertEqual(
            res[1], True, 'estimate succeeds')
        ed = pfc.calculate_electric_data(model, res[2], res[3])
        # maximum of residual node currents without slacknode
        max_dev = norm(
            ed.residual_node_current()[model.count_of_slacks:], np.inf)
        self.assertLess(max_dev, 1e-8, 'residual node current is 0')
        I_n0_line_pu = ed.branch().loc['line', 'I0_pu']
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
            grid.Defk(id='kq', step=0),
            # link scaling factors to active and reactive power of consumer
            grid.Link(
                objid='consumer',
                id='kq',
                part='q',
                step=0),
            # measurements
            grid.Vvalue(id_of_node='n_2', V=Vval))
        init, res = estim.estimate(
            model,
            step_params=[dict(objectives='V')])
        # check
        self.assertEqual(
            res[1], True, 'estimate succeeds')
        ed = pfc.calculate_electric_data(model, res[2], res[3])
        # maximum of residual node currents without slacknode
        max_dev = norm(
            ed.residual_node_current()[model.count_of_slacks:], np.inf)
        self.assertLess(max_dev, 1e-8, 'residual node current is 0')
        # check voltage
        given_V_at_node = model.vvalues.set_index('id_of_node').loc['n_2']
        self.assertAlmostEqual(
            np.abs(res[2][given_V_at_node.index_of_node])[0],
            Vval,
            places=10,
            msg='estimated voltage equals given voltage')

if __name__ == '__main__':
    unittest.main()
