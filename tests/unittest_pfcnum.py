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
import context # adds parent folder of dssex to search path
import numpy as np
import pandas as pd
import dssex.pfcnum as pfc
import egrid.builder as grid
from egrid import make_model
from numpy.testing import assert_array_equal, assert_array_almost_equal

class Calculate_power_flow(unittest.TestCase):

    def test_empty(self):
        model = make_model()
        success, vcx = pfc.calculate_power_flow(model)
        self.assertTrue(
            success,
            'calculate_power_flow succeeds with empty model')
        assert_array_equal(
            vcx,
            np.empty((0,1), dtype=complex),
            err_msg='calculate_power_flow shall return a size-0 vector with '
            'empty model')

    def test_slacknode(self):
        vslack = .994+.023j
        model = make_model(grid.Slacknode('n_0', vslack))
        success, vcx = pfc.calculate_power_flow(model)
        self.assertTrue(
            success,
            'calculate_power_flow succeeds with slack node only')
        assert_array_equal(
            vcx,
            np.array([[vslack]]),
            err_msg='calculate_power_flow shall return the slack voltage')

    def test_slacknode_injection(self):
        vslack = .994+.023j
        model = make_model(
            grid.Slacknode('n_0', vslack),
            grid.Injection('consumer', 'n_0', P10=30.0))
        success, vcx = pfc.calculate_power_flow(model)
        self.assertTrue(
            success,
            'calculate_power_flow succeeds with slack node and injection')
        assert_array_equal(
            vcx,
            np.array([[vslack]]),
            err_msg='calculate_power_flow shall return the slack voltage')

    def test_slacknode_branch(self):
        vslack = .994+.023j
        model = make_model(
            grid.Slacknode('n_0', vslack),
            grid.Branch('branch', 'n_0', 'n_1', y_lo=1e3))
        success, vcx = pfc.calculate_power_flow(model)
        self.assertTrue(
            success,
            'calculate_power_flow succeeds with slack node and branch')
        assert_array_equal(
            vcx,
            np.array([[vslack], [vslack]]),
            err_msg='calculate_power_flow shall return the slack voltage')

    def test_slacknode_branch_tapfactor(self):
        """with taps, taps are modeled with a terminal factor which is a
        factor + Terminallink"""
        vslack = .994+.023j
        model = make_model(
            grid.Slacknode('n_0', vslack),
            grid.Branch('branch', 'n_0', 'n_1', y_lo=1e3),
            grid.Deft(id='taps', type='const', value=-16, m=-.00625, n=1.),
            grid.Tlink(
                id_of_node='n_0',
                id_of_branch='branch',
                id_of_factor='taps'))
        success, vcx = pfc.calculate_power_flow(model)
        self.assertTrue(
            success,
            'calculate_power_flow succeeds with slack node,'
            ' branch and tapchanger')
        assert_array_almost_equal(
            vcx,
            np.array([[vslack], [1.1 * vslack]]),
            err_msg=
            'calculate_power_flow shall return the slack voltage and '
            'voltage increased by tap factor')

    def test_slacknode_branch_many_tapfactors(self):
        """with taps, multiple taps are modeled with factors, checks
        the correct assignment of tapposition to terminal"""
        vslack = .994+.023j
        model = make_model(
            grid.Slacknode('n_0', vslack),
            grid.Branch('branch', 'n_0', 'n_1', y_lo=1e3),
            grid.Deft(id='taps', type='const', value=-16, m=-.00625, n=1.),
            grid.Tlink(
                id_of_node='n_0',
                id_of_branch='branch',
                id_of_factor='taps'),
            grid.Branch('branch2', 'n_0', 'n_2', y_lo=1e3),
            grid.Deft(id='taps2', type='const', value=0., m=-.00625, n=1.),
            grid.Tlink(
                id_of_node='n_0',
                id_of_branch='branch2',
                id_of_factor='taps2'),
            grid.Branch('branch3', 'n_0', 'n_3', y_lo=1e3),
            grid.Deft(id='taps3', type='var', value=16, m=-.00625, n=1.),
            grid.Tlink(
                id_of_node='n_0',
                id_of_branch='branch3',
                id_of_factor='taps3'),
            grid.Branch('branch4', 'n_0', 'n_4', y_lo=1e3))
        success, vcx = pfc.calculate_power_flow(model)
        self.assertTrue(
            success,
            'calculate_power_flow succeeds with slack node,'
            ' branches and tapchangers')
        Vexpected = pd.Series(
            [vslack, 1.1 * vslack, vslack, .9 * vslack, vslack],
            index=['n_0', 'n_1', 'n_2', 'n_3', 'n_4'])
        assert_array_almost_equal(
            vcx,
            # Vexpected ordered according to nodes in model
            Vexpected.reindex(model.nodes.index).array.reshape(-1,1),
            err_msg=
            'calculate_power_flow shall return the slack voltage and '
            'voltages increased/decreased by tap factors')

class Calculate_residual_current(unittest.TestCase):

    def test_positions_parameter(self):
        """positions parameter overrides position value of model"""
        vslack = .994+.023j
        model_pos0 = make_model(
            grid.Slacknode('n_0', vslack),
            grid.Branch('branch', 'n_0', 'n_1', y_lo=1e3),
            grid.Injection('consumer', 'n_1', P10=30, Q10=10),
            # value is different than in calcualted model
            grid.Deft(id='taps', type='const', value=0, m=-.00625, n=1.),
            grid.Tlink(
                id_of_node='n_0', id_of_branch='branch', id_of_factor='taps'))
        model = make_model(
            grid.Slacknode('n_0', vslack),
            grid.Branch('branch', 'n_0', 'n_1', y_lo=1e3),
            grid.Injection('consumer', 'n_1', P10=30, Q10=10),
            grid.Deft(id='taps', type='const', value=-16, m=-.00625, n=1.),
            grid.Tlink(
                id_of_node='n_0', id_of_branch='branch', id_of_factor='taps'))
        success, vcx = pfc.calculate_power_flow(model)
        self.assertTrue(
            success,
            'calculate_power_flow succeeds with slack node,'
            ' branch with tapchanger and injection')
        Iresidual_0 = pfc.calculate_residual_current(
            model_pos0, vcx, positions=np.array([-16], dtype=float))
        Iresidual = pfc.calculate_residual_current(model, vcx)
        assert_array_almost_equal(
            Iresidual_0,
            Iresidual,
            err_msg="residual current shall be equal")

if __name__ == '__main__':
    unittest.main()
