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
import dssex.factors as ft
import egrid.builder as grid
from egrid import make_model
from numpy.testing import assert_array_equal

class Make_get_factor_data(unittest.TestCase):

    def test_default_factors(self):
        """assign (default) scaling factors for active and reactive power to
        each injection"""
        model = make_model(
            grid.Slacknode('n_0'),
            grid.Injection('consumer0', 'n_0'),
            grid.Injection('consumer1', 'n_0'))
        self.assertIsNotNone(model, "make_model makes models")
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factordata.index)
        factor_data = ft.make_factor_data(model, gen_factor_symbols, 0)
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
                id_of_node_B='n_1',
                y_lo=1e4),
            #grid.Injection('injection', 'n_1'),
            grid.Deft(
                'taps', type='var', value=0, min=-16, max=16, m=-10/16,
                is_discrete=True),
            grid.Tlink(
                id_of_node='n_0',
                id_of_branch='branch',
                id_of_factor='taps'))
        self.assertIsNotNone(model, "make_model makes models")
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factordata.index)
        factor_data = ft.make_factor_data(model, gen_factor_symbols, 0)
        self.assertEqual(
            factor_data.kpq.shape,
            (0,2),
            "no scaling factors")
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
            grid.Defk(id=('kp', 'kq'), step=0),
            # link scaling factors to active and reactive power of consumer
            grid.Klink(
                id_of_injection='consumer',
                part='pq',
                id_of_factor=('kp', 'kq'),
                step=0))
        self.assertIsNotNone(model, "make_model makes models")
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factordata.index)
        factor_data = ft.make_factor_data(model, gen_factor_symbols, 0)
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
            grid.Defk(
                id=('kp', 'kq'),
                value=.42,
                min=-1.5,
                max=10.3,
                step=1),
            # link scaling factors to active and reactive power of consumer
            grid.Klink(
                id_of_injection='consumer',
                part='pq',
                id_of_factor=('kp', 'kq'),
                step=1))
        self.assertIsNotNone(model, "make_model makes models")
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factordata.index)
        factor_data = ft.make_factor_data(model, gen_factor_symbols, 1)
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
