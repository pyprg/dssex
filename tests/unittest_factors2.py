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
import dssex.factors as ft
import egrid.builder as grid
from egrid import make_model
from numpy.testing import assert_array_equal

class Separate_factors(unittest.TestCase):

    def test_no_data(self):
        """"""
        model = make_model()
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factordata.index)
        factordata = ft.make_factordata(model, gen_factor_symbols, 0)
        fk, ftaps, factors = ft.separate_factors(
            factordata, np.zeros((0,1), dtype=float))
        self.assertEqual(fk.shape, (0, 2), 'no scaling factors')
        self.assertEqual(ftaps.shape, (0, 1), 'no taps (terminal) factors')
        self.assertEqual(factors.shape, (0, 1), 'no factors')

    def test_default_scaling_factors(self):
        """
        default scaling factors are constants
        """
        model = make_model(
            grid.Slacknode('n_0'),
            grid.Branch(
                id='branch',
                id_of_node_A='n_0',
                id_of_node_B='n_1'),
            grid.Injection('consumer', 'n_1'),
            grid.Injection('consumer2', 'n_1'))
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factordata.index)
        factordata = ft.make_factordata(model, gen_factor_symbols, 0)
        fk, ftaps, factors = ft.separate_factors(
            factordata, np.zeros((0,1), dtype=float))
        self.assertEqual(fk.shape, (2, 2), '2x2 scaling factors')
        assert_array_equal(
            fk,
            np.ones((2,2), dtype=float),
            err_msg='all scaling factors are 1.')
        self.assertEqual(ftaps.shape, (0, 1), 'no taps (terminal) factors')
        self.assertEqual(factors.shape, (1, 1), 'one (default) factor only')

    def test_explicit_scaling_factors(self):
        """
        default scaling factor, and two defined scaling factors
        """
        model = make_model(
            grid.Slacknode('n_0'),
            grid.Branch(
                id='branch',
                id_of_node_A='n_0',
                id_of_node_B='n_1'),
            grid.Injection('consumer', 'n_1'),
            grid.Injection('consumer2', 'n_1'),
            grid.Defk(id=('kp','kq')),
            grid.Klink(
                id_of_injection='consumer',
                id_of_factor=('kp','kq'),
                part='pq'),
            grid.Klink(
                id_of_injection='consumer2',
                id_of_factor='kq',
                part='q'))
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factordata.index)
        factordata = ft.make_factordata(model, gen_factor_symbols, 0)
        solution_vector = np.array([27., 42.]).reshape(-1,1)
        fk, ftaps, factors = ft.separate_factors(
            factordata, solution_vector)
        self.assertEqual(fk.shape, (2, 2), '2x2 scaling factors')
        assert_array_equal(
            fk,
            # first column is kp, second kq
            np.array([[27., 42.], [ 1., 42.]]),
            err_msg='scaling factors are [[27., 42.], [ 1., 42.]]')
        self.assertEqual(ftaps.shape, (0, 1), 'no taps (terminal) factors')
        self.assertEqual(
            factors.shape,
            (3, 1),
            'two decision variables, one constant (default) factor')

    def test_taps_factor(self):
        """
        default scaling factor, and one defined taps factor
        """
        model = make_model(
            grid.Slacknode('n_0'),
            grid.Branch(
                id='branch',
                id_of_node_A='n_0',
                id_of_node_B='n_1',
                y_lo=1e4),
            grid.Injection('consumer', 'n_1'),
            grid.Injection('consumer2', 'n_1'),
            grid.Deft(id='taps'),
            grid.Tlink(
                id_of_branch='branch', id_of_factor='taps',
                id_of_node='n_0'))
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factordata.index)
        factordata = ft.make_factordata(model, gen_factor_symbols, 0)
        solution_vector = np.array([-3.]).reshape(-1,1)
        fk, ftaps, factors = ft.separate_factors(
            factordata, solution_vector)
        self.assertEqual(fk.shape, (2, 2), '2x2 taps factors')
        assert_array_equal(
            fk,
            np.zeros((2,2), dtype=float),
            err_msg='all taps factors are 1.')
        self.assertEqual(
            ftaps,
            np.array([-3.]).reshape(-1,1),
            'taps (terminal) factors are [[-3.]]')
        self.assertEqual(
            factors.shape,
            (2, 1),
            'one decision variables, one constant (default) factor')

if __name__ == '__main__':
    unittest.main()
