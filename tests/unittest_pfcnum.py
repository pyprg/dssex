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

Created on Sat Mar  4 12:08:52 2023

@author: pyprg
"""

import unittest
import numpy as np
import pandas as pd
import dssex.pfcnum as pfc
from egrid import make_model


class Calculate_electric_data(unittest.TestCase):

    def test_empty(self):
        model = make_model()
        vcx = np.zeros((0,1), dtype=complex)
        ed = pfc.calculate_electric_data(model, vcx)
        self.assertIsInstance(
            ed.branch(),
            pd.DataFrame,
            'ed.branch() returns a pandas.DataFrame')
        self.assertIsInstance(
            ed.injection(),
            pd.DataFrame,
            'ed.injection() returns a pandas.DataFrame')
        self.assertIsInstance(
            ed.node(),
            pd.DataFrame,
            'ed.node() returns a pandas.DataFrame')
        self.assertIsInstance(
            ed.residual_node_current(),
            np.ndarray,
            'ed.residual_node_current() returns a numpy.array')

if __name__ == '__main__':
    unittest.main()
