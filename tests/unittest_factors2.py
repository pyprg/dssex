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
import dssex.factors2 as ft
import egrid.builder as grid
from itertools import repeat
from egrid import make_model
from numpy.testing import assert_array_equal

class Get_factor_data(unittest.TestCase):

    def test_no_data(self):
        model = make_model()
        factordefs = ft._get_factordefs(model)
        factordata = ft.get_factor_data(
            model, factordefs, 1)
        self.assertEqual(
            factordata.kpq.shape, 
            (0,2), 
            "no scaling factors")
        self.assertEqual(
            factordata.ftaps.shape, 
            (0,1), 
            "no taps factors")
        self.assertEqual(
            factordata.index_of_term.shape, 
            (0,), 
            "no terminals with taps")
        self.assertEqual(
            factordata.vars.shape, 
            (0,1), 
            "no decision variables")
        self.assertEqual(
            factordata.values_of_vars.shape, 
            (0,1), 
            "no values for decision variables")
        self.assertEqual(
            factordata.var_min.shape, 
            (0,1), 
            "no minimum values for decision variables")
        self.assertEqual(
            factordata.var_max.shape, 
            (0,1), 
            "no maximum values for decision variables")
        self.assertEqual(
            factordata.consts.shape, 
            (0,1), 
            "no paramters")
        self.assertEqual(
            factordata.values_of_consts.shape, 
            (0,1), 
            "no values for paramters")
    
    def test_generic_specific_factor(self):
        """
        one generic injection factor 'kp',
        one generic terminal factor'taps',
        default scaling factors,
        scaling factor 'kq' is specific for step 0
        """
        model = make_model(
            grid.Slacknode('n_0'),
            grid.Branch(
                id='branch',
                id_of_node_A='n_0',
                id_of_node_B='n_1'),
            grid.Injection('consumer', 'n_1'),
            grid.Injection('consumer2', 'n_1'),
            # scaling, define scaling factors
            grid.Deff(id='kp', step=-1),
            grid.Deff(id='kq', step=0),
            # link scaling factors to active and reactive power of consumer
            #   factor for each step (generic, step=-1)
            grid.Link(objid='consumer', id='kp', part='p', step=-1),
            #   factor for step 0 (specific, step=0)
            grid.Link(objid='consumer', id='kq', part='q', step=0),
            # tap factor, for each step (generic, step=-1)
            grid.Deff(id='taps', is_discrete=True, step=-1),
            # tap factor, for step 1
            grid.Deff(id='taps', type='const', is_discrete=True, step=1),
            # link generic tap factor to terminal
            grid.Link(
                objid='branch',
                id='taps',
                nodeid='n_0',
                cls=grid.Terminallink,
                step=-1),
            grid.Link(
                objid='branch',
                id='taps',
                nodeid='n_1',
                cls=grid.Terminallink,
                step=-1),
            # # link step specific tap factor to terminal
            # grid.Link(
            #     objid='branch',
            #     id='taps',
            #     nodeid='n_0',
            #     cls=grid.Terminallink,
            #     step=1)
            )
        factordefs = ft._get_factordefs(model)
        assert_array_equal(
            factordefs.gen_factor_data.index,
            ['kp', 'taps'],
            err_msg="IDs of generic factors shall be ['kp', 'taps']")
        assert_array_equal(
            factordefs.gen_factor_data.index_of_symbol,
            [0, 1],
            err_msg="indices of generic factor symbols shall be [0, 1]")
        factordata = ft.get_factor_data(
            model, factordefs, 0)
        print(factordata)

class Get_taps_factor_data(unittest.TestCase):

    def test_no_data(self):
        model = make_model()
        factordefs = ft._get_factordefs(model)
        factors, termfactor_crossref = ft._get_taps_factor_data(
            model, factordefs, [0, 1])
        self.assertTrue(factors.empty)
        self.assertTrue(termfactor_crossref.empty)

    def test_generic_specific_factor(self):
        """
        one generic injection factor 'kp',
        one generic terminal factor'taps',
        default scaling factors,
        scaling factor 'kq' is specific for step 0
        """
        model = make_model(
            grid.Slacknode('n_0'),
            grid.Branch(
                id='branch',
                id_of_node_A='n_0',
                id_of_node_B='n_1'),
            grid.Injection('consumer', 'n_1'),
            grid.Injection('consumer2', 'n_1'),
            # scaling, define scaling factors
            grid.Deff(id='kp', step=-1),
            grid.Deff(id='kq', step=0),
            # link scaling factors to active and reactive power of consumer
            #   factor for each step (generic, step=-1)
            grid.Link(objid='consumer', id='kp', part='p', step=-1),
            #   factor for step 0 (specific, step=0)
            grid.Link(objid='consumer', id='kq', part='q', step=0),
            # tap factor, for each step (generic, step=-1)
            grid.Deff(id='taps', is_discrete=True, step=-1),
            # tap factor, for step 1
            grid.Deff(id='taps', type='const', is_discrete=True, step=1),
            # link generic tap factor to terminal
            grid.Link(
                objid='branch',
                id='taps',
                nodeid='n_0',
                cls=grid.Terminallink,
                step=-1),
            grid.Link(
                objid='branch',
                id='taps',
                nodeid='n_1',
                cls=grid.Terminallink,
                step=-1),
            # # link step specific tap factor to terminal
            # grid.Link(
            #     objid='branch',
            #     id='taps',
            #     nodeid='n_0',
            #     cls=grid.Terminallink,
            #     step=1)
            )
        factordefs = ft._get_factordefs(model)
        assert_array_equal(
            factordefs.gen_factor_data.index,
            ['kp', 'taps'],
            err_msg="IDs of generic factors shall be ['kp', 'taps']")
        assert_array_equal(
            factordefs.gen_factor_data.index_of_symbol,
            [0, 1],
            err_msg="indices of generic factor symbols shall be [0, 1]")
        factors, termfactor_crossref = ft._get_taps_factor_data(
            model, factordefs, [0,1])
        print('add tests')

class Get_scaling_factor_data(unittest.TestCase):

    def test_no_data(self):
        """well, """
        model = make_model()
        factordefs = ft._get_factordefs(model)
        factors, injfactor_crossref = ft._get_scaling_factor_data(
            model, factordefs, [0, 1], repeat(len(factordefs.gen_factor_data)))
        self.assertTrue(factors.empty)
        self.assertTrue(injfactor_crossref.empty)

    def test_default_scaling_factors(self):
        """
        one generic injection factor 'kp',
        one generic terminal factor'taps',
        default scaling factors,
        scaling factor 'kq' is specific for step 0
        """
        model = make_model(
            grid.Slacknode('n_0'),
            grid.Branch(
                id='branch',
                id_of_node_A='n_0',
                id_of_node_B='n_1'),
            grid.Injection('consumer', 'n_1'),
            grid.Injection('consumer2', 'n_1'),
            # scaling, define scaling factors
            grid.Deff(id='kp', step=-1),
            grid.Deff(id='kq', step=0),
            # link scaling factors to active and reactive power of consumer
            #   factor for each step (generic, step=-1)
            grid.Link(objid='consumer', id='kp', part='p', step=-1),
            #   factor for step 0 (specific, step=0)
            grid.Link(objid='consumer', id='kq', part='q', step=0),
            # tap factor, for each step (generic, step=-1)
            grid.Deff(id='taps', is_discrete=True, step=-1),
            # link scaling factors to active and reactive power of consumer
            grid.Link(
                objid='branch',
                id='taps',
                nodeid='n_0',
                cls=grid.Terminallink,
                step=-1))
        factordefs = ft._get_factordefs(model)
        assert_array_equal(
            factordefs.gen_factor_data.index,
            ['kp', 'taps'],
            err_msg="IDs of generic factors shall be ['kp', 'taps']")
        assert_array_equal(
            factordefs.gen_factor_data.index_of_symbol,
            [0, 1],
            err_msg="indices of generic factor symbols shall be [0, 1]")
        factors, crossref = ft._get_scaling_factor_data(
            model, factordefs, [0, 1], repeat(len(factordefs.gen_factor_data)))
        assert_array_equal(
            factors.loc[0].index.get_level_values('id').sort_values(),
            ['_default_', 'kp', 'kq'],
            err_msg="factor ids of step 0 are ['_default_', 'kp', 'kq']")
        assert_array_equal(
            factors.loc[0].index_of_symbol,
            [2, 0, 3],
            err_msg="indices of symbols for step 0 are [2, 0, 3]")
        assert_array_equal(
            factors.loc[0].index_of_source,
            [-1, -1, -1],
            err_msg="indices of symbols for step 0 are [-1, -1, -1]")
        assert_array_equal(
            factors.loc[1].index.get_level_values('id').sort_values(),
            ['_default_', 'kp'],
            err_msg="factor ids of step 1 are ['_default_', 'kp']")
        assert_array_equal(
            factors.loc[1].index_of_symbol,
            [2, 0],
            err_msg="indices of symbols for step 1 are [2, 0]")
        assert_array_equal(
            factors.loc[1].index_of_source,
            [2, 0],
            err_msg="indices of symbols for step 1 are [2, 0]")

    def test_missing_generic_links(self):
        """
        no generic injection factor 'kp' as link is missing,
        one generic terminal factor'taps',
        default scaling factors,
        scaling factor 'kq' is specific for step 0
        """
        model = make_model(
            grid.Slacknode('n_0'),
            grid.Branch(
                id='branch',
                id_of_node_A='n_0',
                id_of_node_B='n_1'),
            grid.Injection('consumer', 'n_1'),
            grid.Injection('consumer2', 'n_1'),
            # scaling, define scaling factors
            grid.Deff(id='kp', step=-1),
            grid.Deff(id='kq', step=0),
            # link scaling factors to active and reactive power of consumer
            #   factor for each step (generic, step=-1)
            # grid.Link(objid='consumer', id='kp', part='p', step=-1),
            #   factor for step 0 (specific, step=0)
            grid.Link(objid='consumer', id='kq', part='q', step=0),
            # tap factor, for each step (generic, step=-1)
            grid.Deff(id='taps', is_discrete=True, step=-1),
            # link scaling factors to active and reactive power of consumer
            grid.Link(
                objid='branch',
                id='taps',
                nodeid='n_0',
                cls=grid.Terminallink,
                step=-1))
        factordefs = ft._get_factordefs(model)
        assert_array_equal(
            factordefs.gen_factor_data.index,
            ['taps'],
            err_msg="IDs of generic factors shall be ['taps']")
        assert_array_equal(
            factordefs.gen_factor_data.index_of_symbol,
            [0],
            err_msg="indices of generic factor symbols shall be [0]")
        factors, crossref = ft._get_scaling_factor_data(
            model, factordefs, [0, 1], repeat(len(factordefs.gen_factor_data)))
        assert_array_equal(
            factors.loc[0].index.get_level_values('id').sort_values(),
            ['_default_', 'kq'],
            err_msg="factor ids of step 0 are ['_default_', 'kq']")
        assert_array_equal(
            factors.loc[0].index_of_symbol,
            [1, 2],
            err_msg="indices of symbols for step 0 are [1, 2]")
        assert_array_equal(
            factors.loc[0].index_of_source,
            [-1, -1],
            err_msg="indices of symbols for step 0 are [-1, -1]")
        assert_array_equal(
            factors.loc[1].index.get_level_values('id').sort_values(),
            ['_default_'],
            err_msg="factor ids of step 1 are ['_default_']")
        assert_array_equal(
            factors.loc[1].index_of_symbol,
            [1],
            err_msg="indices of symbols for step 1 are [1]")
        assert_array_equal(
            factors.loc[1].index_of_source,
            [1],
            err_msg="indices of symbols for step 1 are [1]")

if __name__ == '__main__':
    unittest.main()
