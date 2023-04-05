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

Created on Wed Mar 22 20:25:45 2023

@author: pyprg
"""

import unittest
import context # adds parent folder of dssex to search path
import numpy as np
import pandas as pd
import dssex.factors as ft
import egrid.builder as grid
from egrid import make_model
from numpy.testing import assert_array_equal

class Get_scaling_factor_data(unittest.TestCase):

    def test_no_data(self):
        """'get_scaling_factor_data' processes empty input"""
        model = make_model()
        factordefs = ft.make_factordefs(model)
        factors, injection_factor = ft._get_scaling_factor_data(
            model, factordefs, [2, 3], None)
        self.assertTrue(
            factors.empty,
            "get_scaling_factor_data returns no data for factors")
        self.assertTrue(
            injection_factor.empty,
            "get_scaling_factor_data returns no data for association "
            "injection_factor")

    def test_default_scaling_factors(self):
        """'get_scaling_factor_data' creates default scaling factors
        if factors are not given explicitely"""
        model = make_model(
            grid.Injection(id='injid0', id_of_node='n_0'))
        factordefs = ft.make_factordefs(model)
        index_of_step = 3
        steps = [index_of_step-1, index_of_step]
        factors, injection_factor = ft._get_scaling_factor_data(
            model, factordefs, steps, None)
        assert_array_equal(
            [idx[0] for idx in factors.index],
            steps,
            err_msg="one factor per step")
        self.assertTrue(
            all(idx[1]=='const' for idx in factors.index),
            "all factors are const parameters")
        self.assertTrue(
            all(idx[2]==grid.DEFAULT_FACTOR_ID for idx in factors.index),
            f"all factors are '{grid.DEFAULT_FACTOR_ID}'")

    def test_step1_without_scaling_factordef(self):
        """'get_scaling_factor_data' creates default scaling factors
        if factors are not given explicitely"""
        model = make_model(
            grid.Slacknode('n_0'),
            grid.Injection('consumer', 'n_0'),
            # scaling, define scaling factors
            grid.Deff(id=('kp', 'kq'), step=0),
            # link scaling factors to active and reactive power of consumer
            grid.Link(objid='consumer', id=('kp', 'kq'), part='pq', step=0))
        factordefs = ft.make_factordefs(model)
        index_of_step = 1
        steps = [index_of_step-1, index_of_step]
        factors, injection_factor = ft._get_scaling_factor_data(
            model, factordefs, steps, None)
        self.assertEqual(
            factors.loc[0].shape[0],
            2,
            "two factors for step 0")
        self.assertTrue(
            all(factors.loc[0].reset_index().type == 'var'),
            "all factors of step 0 are decision variables ('var')")
        self.assertEqual(
            factors.loc[1].shape[0],
            1,
            "one factors for step 1")
        self.assertTrue(
            all(factors.loc[1].reset_index().type == 'const'),
            "all factors of step 1 are parameters ('const')")
        self.assertTrue(
            all(factors.loc[1].reset_index().id == grid.DEFAULT_FACTOR_ID),
            f"id of factor for step 1 is '{grid.DEFAULT_FACTOR_ID}'")
        self.assertTrue(
            all(factors.index_of_source == -1),
            "there are no source factors")

    def test_generic_scaling_factor(self):
        """'get_scaling_factor_data' creates default factors if factors are not
        given explicitely"""
        model = make_model(
            grid.Slacknode('n_0'),
            grid.Injection('consumer', 'n_0'),
            # scaling, define scaling factors
            grid.Deff(id=('kp', 'kq'), step=-1),
            # link scaling factors to active and reactive power of consumer
            grid.Link(objid='consumer', id=('kp', 'kq'), part='pq', step=-1))
        factordefs = ft.make_factordefs(model)
        index_of_step = 1
        steps = [index_of_step-1, index_of_step]
        factors, injection_factor = ft._get_scaling_factor_data(
            model, factordefs, steps, None)
        factors_step_0 = factors.loc[0]
        self.assertEqual(
            len(factors_step_0),
            2,
            "two factors for step 0")
        factors_step_1 = factors.loc[1]
        self.assertEqual(
            len(factors_step_1),
            2,
            "two factors for step 1")

    def test_scaling_factor_with_terminallink(self):
        """'get_scaling_factor_data' creates factors for injections
        if linked with Injectionlink only"""
        model = make_model(
            grid.Slacknode('n_0'),
            grid.Injection('consumer', 'n_0'),
            # scaling, define scaling factors
            grid.Deff(id=('kp', 'kq'), step=-1),
            # link scaling factors to active and reactive power of consumer
            grid.Link(
                objid='consumer',
                id=('kp', 'kq'),
                part='pq',
                cls=grid.Terminallink,
                step=-1))
        factordefs = ft.make_factordefs(model)
        index_of_step = 0
        steps = [index_of_step]
        factors, injection_factor = ft._get_scaling_factor_data(
            model, factordefs, steps, None)
        self.assertEqual(
            len(factors),
            1,
            "one factor")
        self.assertEqual(
            factors.index[0][2],
            grid.DEFAULT_FACTOR_ID,
            "'get_scaling_factor_data' creates a scaling factor with ID "
            f"{grid.DEFAULT_FACTOR_ID} as link is a Terminallink")

class Get_taps_factor_data(unittest.TestCase):

    def test_terminal_factor(self):
        """"""
        model = make_model(
            grid.Slacknode('n_0'),
            grid.Branch(
                id='branch',
                id_of_node_A='n_0',
                id_of_node_B='n_1'),
            grid.Injection('injection', 'n_1'),
            # scaling, define scaling factors
            grid.Deff(id='taps', step=-1),
            # link scaling factors to active and reactive power of consumer
            grid.Link(
                objid='branch',
                id='taps',
                nodeid='n_0',
                cls=grid.Terminallink,
                step=-1))
        factordefs = ft.make_factordefs(model)
        index_of_step = 1
        steps = [index_of_step-1, index_of_step]
        factors, injection_factor = ft._get_scaling_factor_data(
            model, factordefs, steps, start=[3, 5])
        self.assertEqual(
            len(factors.loc[0]),
            1,
            "one factor for step 0")
        self.assertEqual(
            factors.loc[0].index_of_symbol[0],
            3,
            "step-0 symbol has index 3")
        self.assertEqual(
            len(factors.loc[1]),
            1,
            "one factor for step 1")
        self.assertEqual(
            factors.loc[1].index_of_symbol[0],
            5,
            "step-1 symbol has index 5")

class Make_get_factor_data(unittest.TestCase):

    def test_default_factors(self):
        """assign (default) scaling factors for active and reactive power to
        each injection"""
        model = make_model(
            grid.Slacknode('n_0'),
            grid.Injection('consumer0', 'n_0'),
            grid.Injection('consumer1', 'n_0'))
        self.assertIsNotNone(model, "make_model makes models")
        factordefs = ft.make_factordefs(model)
        factor_data = ft.make_factor_data2(model, factordefs, step=0)
        self.assertEqual(
            factor_data.kpq.shape,
            (2,2),
            "P and Q factors for two injections")
        self.assertEqual(
            factor_data.vars.shape,
            (0,1),
            "no decision variables")
        self.assertEqual(
            factor_data.values_of_vars.shape,
            (0,1),
            "no initial values for decision variables")
        self.assertEqual(
            factor_data.var_min.shape,
            (0,1),
            "no minimum values for decision variables")
        self.assertEqual(
            factor_data.var_max.shape,
            (0,1),
            "no maximum values for decision variables")
        self.assertEqual(
            factor_data.consts.shape,
            (1,1),
            "one constant (parameter)")
        self.assertEqual(
            factor_data.consts[0,0].name(),
            '_default_',
            "name of constant is '_default_'")
        assert_array_equal(
            factor_data.values_of_consts,
            [[1.]],
            err_msg="values of consts shall be [[1.]] "
            "(default factor shall be 1.)")
        assert_array_equal(
            factor_data.var_const_to_factor,
            [0],
            err_msg="const indices shall be [0] "
            "(default factor shall have index 0)")
        assert_array_equal(
            factor_data.var_const_to_kp,
            [0, 0],
            err_msg="indices of active power scaling factors shall be [0,0] "
            "(active power scaling factors are mapped to [0,0])")
        assert_array_equal(
            factor_data.var_const_to_kq,
            [0, 0],
            err_msg="indices of reactive power scaling factors shall be [0,0] "
            "(reactive power scaling factors are mapped to [0,0])")

    def test_default_factors_terminal_factor(self):
        """taps factor at terminal"""
        model = make_model(
            grid.Slacknode('n_0'),
            grid.Branch(
                id='branch',
                id_of_node_A='n_0',
                id_of_node_B='n_1'),
            #grid.Injection('injection', 'n_1'),
            grid.Deff(
                'taps', value=0, min=-16, max=16, m=-10/16, is_discrete=True),
            grid.Link(
                objid='branch',
                id='taps',
                nodeid='n_0',
                cls=grid.Terminallink))
        self.assertIsNotNone(model, "make_model makes models")
        factordefs = ft.make_factordefs(model)
        factor_data = ft.make_factor_data2(model, factordefs, step=0)
        self.assertEqual(
            factor_data.kpq.shape,
            (0,2),
            "no scaling factors")
        self.assertEqual(
            factor_data.ftaps.shape,
            (1,1),
            "one taps factor")
        self.assertEqual(
            factor_data.ftaps[0,0].name(),
            'taps',
            "taps factor has name 'taps'")
        assert_array_equal(
            factor_data.is_discrete,
            [True],
            err_msg="taps factor shall be discrete")
        assert_array_equal(
            factor_data.var_min.toarray(),
            [[-16.]],
            err_msg="var_min shall be [[-16.]]")
        assert_array_equal(
            factor_data.var_max.toarray(),
            [[16.]],
            err_msg="var_max shall be [[16.]]")
        assert_array_equal(
            factor_data.var_const_to_factor,
            [0],
            err_msg="var_const_to_factor shall be [0]")
        assert_array_equal(
            factor_data.var_const_to_kp,
            [],
            err_msg="var_const_to_kp shall be []")
        assert_array_equal(
            factor_data.var_const_to_kq,
            [],
            err_msg="var_const_to_kq shall be []")
        assert_array_equal(
            factor_data.var_const_to_ftaps,
            [0],
            err_msg="var_const_to_ftaps shall be [0]")

    def test_one_decision_variable_for_pq(self):
        """assign decision variables to scaling factors
        for active and reactive power, use default parameters for decision
        variables"""
        model = make_model(
            grid.Slacknode('n_0'),
            grid.Injection('consumer', 'n_0'),
            # scaling, define scaling factors
            grid.Deff(id=('kp', 'kq'), step=0),
            # link scaling factors to active and reactive power of consumer
            grid.Link(objid='consumer', id=('kp', 'kq'), part='pq', step=0))
        self.assertIsNotNone(model, "make_model makes models")
        factordefs = ft.make_factordefs(model)
        factor_data = ft.make_factor_data2(model, factordefs, step=0)
        self.assertEqual(
            factor_data.kpq.shape,
            (1,2),
            "P and Q factors for one injection")
        self.assertEqual(
            factor_data.vars.shape,
            (2,1),
            "two decision variables")
        assert_array_equal(
            factor_data.values_of_vars,
            [[1.], [1.]],
            "initial values of decision variables are [[1.], [1.]]")
        assert_array_equal(
            factor_data.var_min,
            [[-np.inf], [-np.inf]],
            "minimum values of decision variables is [[-inf], [-inf]]")
        assert_array_equal(
            factor_data.var_max,
            [[np.inf], [np.inf]],
            "minimum values of decision variables is [[inf], [inf]]")
        self.assertEqual(
            factor_data.consts.shape,
            (0,1),
            "no constants (no parameters)")
        self.assertEqual(
            factor_data.values_of_consts.shape,
            (0,1),
            "no values for constants (parameters)")
        self.assertEqual(
            factor_data.var_const_to_factor.shape,
            (2,),
            "separate indices for active and reactiv power factor")
        assert_array_equal(
            factor_data.var_const_to_kp,
            [0],
            err_msg="indices of active power scaling factors shall be [0] "
            "(active power scaling factor is mapped to index 0)")
        assert_array_equal(
            factor_data.var_const_to_kq,
            [1],
            err_msg="indices of reactive power scaling factors shall be [1] "
            "(reactive power scaling factors is mapped to index 1)")

    def test_variables_for_step1(self):
        """assign decision variables to scaling factors
        for active and reactive power, use default parameters for decision
        variables"""
        model = make_model(
            grid.Slacknode('n_0'),
            grid.Injection('consumer', 'n_0'),
            # scaling, define scaling factors
            grid.Deff(
                id=('kp', 'kq'),
                value=.42,
                min=-1.5,
                max=10.3,
                step=1),
            # link scaling factors to active and reactive power of consumer
            grid.Link(objid='consumer', id=('kp', 'kq'), part='pq', step=1))
        self.assertIsNotNone(model, "make_model makes models")
        factordefs = ft.make_factordefs(model)
        factor_data = ft.make_factor_data2(model, factordefs, step=1)
        self.assertEqual(
            factor_data.kpq.shape,
            (1,2),
            "P and Q factors for one injection")
        self.assertEqual(
            factor_data.vars.shape,
            (2,1),
            "two decision variables")
        assert_array_equal(
            factor_data.values_of_vars,
            [[.42], [.42]],
            "initial values of decision variables are [[.42], [.42]]")
        assert_array_equal(
            factor_data.var_min,
            [[-1.5], [-1.5]],
            "minimum values of decision variables is [[-1.5], [-1.5]]")
        assert_array_equal(
            factor_data.var_max,
            [[10.3], [10.3]],
            "minimum values of decision variables is [[10.3], [10.3]]")
        self.assertEqual(
            factor_data.consts.shape,
            (0,1),
            "no constants (parameters)")
        self.assertEqual(
            factor_data.values_of_consts.shape,
            (0,1),
            "no values for const parameters")
        self.assertEqual(
            factor_data.var_const_to_factor.shape,
            (2,),
            "separate indices for active and reactiv power factor")
        assert_array_equal(
            factor_data.var_const_to_kp,
            [0],
            err_msg="indices of active power scaling factors shall be [0] "
            "(active power scaling factor is mapped to index 0)")
        assert_array_equal(
            factor_data.var_const_to_kq,
            [1],
            err_msg="indices of reactive power scaling factors shall be [1] "
            "(reactive power scaling factors is mapped to index 1)")

if __name__ == '__main__':
    unittest.main()
