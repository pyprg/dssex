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
import context # adds parent folder of dssex to search path
import egrid.builder as grid
import dssex.estim as estim
from egrid import make_model

class Estimate(unittest.TestCase):
    
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
            res[1], True, 'estimate succeeds for empty model')
        self.assertEqual(
            res[2],
            [vcx_slack], 
            'estimate returns a one-element voltage vector with given '
            'slack-voltage for a model with just one slacknode')
        self.assertTrue(
            (res[3]==[[1., 1.]]).all(), 
            'estimate returns two column scaling factor vectors (kp, kq) '
            'of [[1., 1.]] for initialization step')

if __name__ == '__main__':
    unittest.main()
