# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 20:25:45 2023

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

@author: pyprg
"""

import unittest
import context # adds parent folder of dssex to search path
import numpy as np
import pandas as pd
import dssex.factors as factors
import egrid.builder as grid
from egrid import make_model
from numpy.testing import assert_array_equal, assert_array_almost_equal

class Make_get_factor_data(unittest.TestCase):

    def test_make_default_parameters(self):
        """assign (default) scaling factors for active and reactive power to
        each injection"""
        vcx_slack = 0.95+0.02j
        s = 30.+10.j
        model = make_model(
            grid.Slacknode('n_0', V=vcx_slack),
            grid.Injection('consumer0', 'n_0', P10=s.real, Q10=s.imag),
            grid.Injection('consumer1', 'n_0', P10=s.real, Q10=s.imag))
        self.assertIsNotNone(model, "make_model makes models")
        get_factor_data = factors.make_get_factor_data(model)
        factor_data = get_factor_data(step=0)
        self.assertEqual(
            factor_data.kpq.shape,
            (2,2),
            "P and Q factors for two injections")
        self.assertEqual(
            factor_data.kvars.shape,
            (0,1),
            "no decision variables")
        self.assertEqual(
            factor_data.values_of_vars.shape,
            (0,1),
            "no initial values for decision variables")
        self.assertEqual(
            factor_data.kvar_min.shape,
            (0,1),
            "no minimum values for decision variables")
        self.assertEqual(
            factor_data.kvar_max.shape,
            (0,1),
            "no maximum values for decision variables")
        self.assertEqual(
            factor_data.kconsts.shape,
            (1,1),
            "one constant (parameter)")
        self.assertEqual(
            factor_data.kconsts[0,0].name(),
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

    def test_one_decision_variable_for_pq(self):
        """assign decision variables to scaling factors
        for active and reactive power, use default parameters for decision
        variables"""
        vcx_slack = 0.95+0.02j
        s = 30.+10.j
        model = make_model(
            grid.Slacknode('n_0', V=vcx_slack),
            grid.Injection('consumer', 'n_0', P10=s.real, Q10=s.imag),
            # scaling, define scaling factors
            grid.Deff(id=('kp', 'kq'), step=0),
            # link scaling factors to active and reactive power of consumer
            grid.Link(objid='consumer', id=('kp', 'kq'), part='pq'))
        self.assertIsNotNone(model, "make_model makes models")
        get_factor_data = factors.make_get_factor_data(model)
        factor_data = get_factor_data(step=0)
        self.assertEqual(
            factor_data.kpq.shape,
            (1,2),
            "P and Q factors for one injection")
        self.assertEqual(
            factor_data.kvars.shape,
            (2,1),
            "no decision variables")
        assert_array_equal(
            factor_data.values_of_vars,
            [[1.], [1.]],
            "initial values of decision variables are [[1.], [1.]]")
        assert_array_equal(
            factor_data.kvar_min,
            [[-np.inf], [-np.inf]],
            "minimum values of decision variables is [[-inf], [-inf]]")
        assert_array_equal(
            factor_data.kvar_max,
            [[np.inf], [np.inf]],
            "minimum values of decision variables is [[inf], [inf]]")
        self.assertEqual(
            factor_data.kconsts.shape,
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
