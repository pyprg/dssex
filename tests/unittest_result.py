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

Created on Wed May 24 21:51:35 2023

@author: pyprg
"""

import unittest
import context # adds parent folder of dssex to search path
import numpy as np
import pandas as pd
import dssex.result as rt
import egrid.builder as grid
from egrid import make_model
from numpy.testing import assert_array_equal, assert_array_almost_equal

class Calculate_electric_data(unittest.TestCase):

    def test_empty(self):
        model = make_model()
        vcx = np.zeros((0,1), dtype=complex)
        res = rt.calculate_electric_data(model, vcx)
        self.assertIsInstance(
            res['branches'],
            pd.DataFrame,
            'branches is a pandas.DataFrame')
        self.assertIsInstance(
            res['injections'],
            pd.DataFrame,
            'injections is a pandas.DataFrame')
        self.assertIsInstance(
            res['nodes'],
            pd.DataFrame,
            'nodes is a pandas.DataFrame')

    def test_slacknode(self):
        vslack = .994+.023j
        model = make_model(grid.Slacknode('n_0', vslack))
        vcx = np.array([[vslack]])
        res = rt.calculate_electric_data(model, vcx)
        self.assertIsInstance(
            res['branches'],
            pd.DataFrame,
            'branches is a pandas.DataFrame')
        self.assertIsInstance(
            res['injections'],
            pd.DataFrame,
            'injections is a pandas.DataFrame')
        self.assertIsInstance(
            res['nodes'],
            pd.DataFrame,
            'nodes is a pandas.DataFrame')
        assert_array_equal(
            res['nodes'].Vcx_pu.to_numpy().reshape(-1,1),
            np.array([[vslack]]),
            err_msg='calculate_electric_data shall return the slack voltage')

    def test_slacknode_injection(self):
        """
        n_0----------->> consumer
          slack=True       P10=30
          V=.994+.023j     Q10=10
        """
        vslack = .994+.023j
        Sconsumer = 30.0+10.j
        model = make_model(
            grid.Slacknode('n_0', vslack),
            grid.Injection(
                'consumer', 'n_0', P10=Sconsumer.real, Q10=Sconsumer.imag))
        vcx = np.array([[vslack]])
        res = rt.calculate_electric_data(model, vcx)
        self.assertIsInstance(
            res['branches'],
            pd.DataFrame,
            'branches is a pandas.DataFrame')
        self.assertIsInstance(
            res['injections'][['Icx_pu']],
            pd.DataFrame,
            'injections is a pandas.DataFrame')
        self.assertIsInstance(
            res['nodes'],
            pd.DataFrame,
            'nodes is a pandas.DataFrame')
        vnode = res['nodes'].Vcx_pu.to_numpy().reshape(-1,1)
        assert_array_equal(
            vnode,
            np.array([[vslack]]),
            err_msg='calculate_electric_data shall return the slack voltage')
        Icx_pu = res['injections'][['Icx_pu']].to_numpy()
        S_pu = 3 * vnode * Icx_pu.conj()
        assert_array_almost_equal(
            S_pu,
            np.array([[Sconsumer]], dtype=np.complex128),
            err_msg='calculate_electric_data shall return the complex current')

if __name__ == '__main__':
    unittest.main()
