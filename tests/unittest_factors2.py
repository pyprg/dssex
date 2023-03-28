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

class Make_factordefs(unittest.TestCase):

    def test_no_data(self):
        model = make_model()
        factordefs = ft.make_factordefs(model)
        self.assertTrue(
            factordefs.gen_factor_data.empty, 'no generic factors')
        self.assertEqual(
            factordefs.gen_factor_symbols.size1(), 0, 'no symbols')
        self.assertTrue(
            factordefs.gen_injfactor.empty, 'no generic injection factors')
        self.assertTrue(
            factordefs.gen_termfactor.empty, 'no generic terminal factors')
        self.assertEqual(
            factordefs.factorgroups.groups.groups, {}, 'no groups of factors')
        self.assertEqual(
            factordefs.injfactorgroups.groups.groups,
            {},
            'no groups of injection factors')

    def test_generic_injection_factor(self):
        """basic test with one generic injection factor"""
        model = make_model(
            grid.Slacknode('n_0'),
            grid.Branch(
                id='branch',
                id_of_node_A='n_0',
                id_of_node_B='n_1'),
            grid.Injection('consumer', 'n_1'),
            # scaling, define scaling factors
            grid.Deff(id='kp', step=-1),
            # link scaling factors to active power of consumer
            #   factor for each step (generic, step=-1)
            grid.Link(objid='consumer', id='kp', part='p', step=-1))
        factordefs = ft.make_factordefs(model)
        self.assertEqual(
            dict(
                zip(factordefs.gen_factor_data.columns,
                    factordefs.gen_factor_data.iloc[0].to_numpy())),
            {'step': -1, 'type': 'var', 'id_of_source': 'kp', 'value': 1.0,
             'min': -np.inf, 'max': np.inf, 'is_discrete': False, 'm': 1.0,
             'n': 0.0, 'index_of_symbol': 0})
        self.assertEqual(
            factordefs.gen_factor_symbols.name(),
            'kp',
            "generic factor has name 'kp'")
        assert_array_equal(
            factordefs.gen_injfactor.to_numpy(),
            np.array([[-1, 'kp']], dtype=object),
            err_msg="expected one generic factor (-1) with id 'kp'")
        assert_array_equal(
            factordefs.gen_injfactor.index.to_numpy()[0],
            ('consumer', 'p'),
            err_msg="expected index is ('consumer', 'p')")
        assert_array_equal(
            factordefs.gen_termfactor,
            np.zeros((0,3), dtype=object),
            err_msg="no taps (terminal) factor"),
        self.assertEqual(
            factordefs.factorgroups.groups.groups,
            {-1: [0]},
            "one generic factor")
        self.assertEqual(
            factordefs.injfactorgroups.groups.groups,
            {-1: [0]},
            "one generic injection_factor relation")

    def test_taps_injection_factor(self):
        """basic test with one generic taps (terminal) factor"""
        model = make_model(
            grid.Slacknode('n_0'),
            grid.Branch(
                id='branch',
                id_of_node_A='n_0',
                id_of_node_B='n_1'),
            grid.Injection('consumer', 'n_1'),
            # scaling, define scaling factors
            grid.Deff(
                id='taps', value=0., type='const', is_discrete=True,
                m=-0.00625, step=-1),
            # link scaling factors to active power of consumer
            #   factor for each step (generic, step=-1)
            grid.Link(
                objid='branch', id='taps', nodeid='n_0',
                cls=grid.Terminallink, step=-1))
        factordefs = ft.make_factordefs(model)
        self.assertEqual(
            dict(
                zip(factordefs.gen_factor_data.columns,
                    factordefs.gen_factor_data.iloc[0].to_numpy())),
            {'step': -1, 'type': 'const', 'id_of_source': 'taps', 'value': 0.,
             'min': -np.inf, 'max': np.inf, 'is_discrete': True, 'm': -0.00625,
             'n': 0.0, 'index_of_symbol': 0})
        self.assertEqual(
            factordefs.gen_factor_symbols.name(),
            'taps',
            "generic factor has name 'taps'")
        assert_array_equal(
            factordefs.gen_injfactor,
            np.zeros((0,2), dtype=object),
            err_msg="expected no generic injection_factor relation")
        assert_array_equal(
            factordefs.gen_termfactor.to_numpy(),
            np.array([[-1, 'taps', 0]], dtype=object),
            err_msg="expected taps (terminal) factor [-1, 'taps', 0]"),
        assert_array_equal(
            factordefs.gen_termfactor.index.to_numpy()[0],
            ('branch', 'n_0'),
            err_msg="expected index is ('branch', 'n_0')")
        self.assertEqual(
            factordefs.factorgroups.groups.groups,
            {-1: [0]},
            "one generic factor")

class Make_factor_data2(unittest.TestCase):

    def test_no_data(self):
        model = make_model()
        factordefs = ft.make_factordefs(model)
        factordata = ft.make_factor_data2(
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
        assert_array_equal(
            factordata.var_const_to_factor,
            [],
            err_msg="no var_const crossreference")
        assert_array_equal(
            factordata.var_const_to_kp,
            [],
            err_msg="no active power scaling factor crossreference")
        assert_array_equal(
            factordata.var_const_to_kq,
            [],
            err_msg="no reactive power scaling factor crossreference")
        assert_array_equal(
            factordata.var_const_to_ftaps,
            [],
            err_msg="no taps factor crossreference")

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
        factordefs = ft.make_factordefs(model)
        assert_array_equal(
            factordefs.gen_factor_data.index,
            ['kp', 'taps'],
            err_msg="IDs of generic factors shall be ['kp', 'taps']")
        assert_array_equal(
            factordefs.gen_factor_data.index_of_symbol,
            [0, 1],
            err_msg="indices of generic factor symbols shall be [0, 1]")
        factordata = ft.make_factor_data2(
            model, factordefs, 0)

class Get_taps_factor_data(unittest.TestCase):

    def test_no_data(self):
        model = make_model()
        factordefs = ft.make_factordefs(model)
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
        factordefs = ft.make_factordefs(model)
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
        factordefs = ft.make_factordefs(model)
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
        factordefs = ft.make_factordefs(model)
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
        factordefs = ft.make_factordefs(model)
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
