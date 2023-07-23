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

Created on Sun Apr  2 13:15:05 2023

@author: pyprg
"""
import unittest
import context # adds parent folder of dssex to search path
import egrid.builder as grid
import dssex.result as rt
import dssex.estim as estim
from egrid import make_model
from numpy.testing import assert_array_almost_equal

class VVC_transformer(unittest.TestCase):
    """
    schema_vmax (Exp_v_p=0):
    ::
          +---------------------( () )-----------+------------->
        slack                     Tr           node          consumer
          V=1.+.0j   Tlink=taps    y_lo=0.9k-0.95kj           P10=30
                                   y_tr=1.3µ+1.5µj            Exp_v_p=0

        #. Deft(id=taps value=0 type=var min=-16 max=16 m=-.00625 n=1)

    schema_vmin (Exp_v_p=2):
    ::
          +---------------------( () )-----------+------------->
        slack                     Tr           node          consumer
          V=1.+.0j   Tlink=taps    y_lo=0.9k-0.95kj           P10=30
                                   y_tr=1.3µ+1.5µj            Exp_v_p=2

        #. Deft(id=taps value=0 type=var min=-16 max=16 m=-.00625 n=1)
    """
    _devs_vvc = [
        grid.Slacknode(id_of_node='slack', V=1.+.0j),
        grid.Branch(
            id='Tr', id_of_node_A='slack', id_of_node_B='node',
            y_lo=.9e3-.95e3j, y_tr=1.3e-6+1.5e-6j),
        grid.Deft(
            id='taps', type='var', value=0, min=-16, max=16, m=-.1/16, n=1),
        grid.Tlink(id_of_branch='Tr', id_of_node='slack', id_of_factor='taps')]
    _devs_vvc_vmax = [
        _devs_vvc,
        grid.Injection(id='consumer', id_of_node='node', P10=30, Exp_v_p=0)]
    _devs_vvc_vmin = [
        _devs_vvc,
        grid.Injection(id='consumer', id_of_node='node', P10=30, Exp_v_p=2)]

    def test_min_losses_with_taps(self):
        """Voltage at secondary is driven to maximum possible value.
        """
        model_vvc = make_model(self._devs_vvc_vmax)
        # optimize according to losses
        res = estim.estimate(model_vvc, step_params=[dict(objectives='L')])
        res_vvc = list(rt.make_printables(model_vvc, res))
        tappos = res_vvc[1]['branches'].loc['Tr','Tap0']
        expected = model_vvc.factors.gen_factordata.loc['taps', 'min']
        self.assertEquals(tappos, expected)

    def test_min_losses_with_taps_limit(self):
        """Voltage at secondary is driven to maximum limit.
        """
        model_vvc = make_model(self._devs_vvc_vmax, grid.Vlimit(max=1.05))
        # optimize according to losses
        res = estim.estimate(model_vvc, step_params=[dict(objectives='L')])
        res_vvc = list(rt.make_printables(model_vvc, res))
        tappos = res_vvc[1]['branches'].loc['Tr','Tap0']
        min_position = model_vvc.factors.gen_factordata.loc['taps', 'min']
        self.assertGreater(tappos, min_position)
        self.assertLess(res_vvc[1]['nodes'].loc['node'].V_pu , 1.05)

    def test_min_losses_with_taps2(self):
        """Voltage at secondary is driven to minimum possible value.
        """
        model_vvc = make_model(self._devs_vvc_vmin)
        # optimize according to losses
        res = estim.estimate(model_vvc, step_params=[dict(objectives='L')])
        res_vvc = list(rt.make_printables(model_vvc, res))
        tappos = res_vvc[1]['branches'].loc['Tr','Tap0']
        expected = model_vvc.factors.gen_factordata.loc['taps', 'max']
        self.assertEquals(tappos, expected)

    def test_min_losses_with_taps2_limit(self):
        """Voltage at secondary is driven to minimum limit.
        """
        model_vvc = make_model(self._devs_vvc_vmin, grid.Vlimit(min=.95))
        # optimize according to losses
        res = estim.estimate(model_vvc, step_params=[dict(objectives='L')])
        res_vvc = list(rt.make_printables(model_vvc, res))
        tappos = res_vvc[1]['branches'].loc['Tr','Tap0']
        max_position = model_vvc.factors.gen_factordata.loc['taps', 'max']
        self.assertLess(tappos, max_position)
        self.assertGreater(res_vvc[1]['nodes'].loc['node'].V_pu, .95)

class VVC_shuntcapacitor(unittest.TestCase):
    """
    schema:
    ::
                                                      Q10=-5
                                                      Exp_v_q=2
                                      node          cap
                                        +-------------||
                                        |
          +-----------[ -- ]-----------+------------->
        slack           Br           node          consumer
          V=1.+.0j       y_lo=0.9k-0.95kj           P10=30
                         y_tr=1.3µ+1.5µj            Q10=15

        #.Klink(id_of_injection=cap id_of_factor=taps part=q)
    """
    _mygrid = [
        grid.Slacknode(id_of_node='slack', V=1.+.0j),
        grid.Branch(
            id='Br', id_of_node_A='slack', id_of_node_B='node',
            y_lo=.9e3-.95e3j, y_tr=1.3e-6+1.5e-6j),
        grid.Injection(id='consumer', id_of_node='node', P10=30, Q10=15),
        grid.Injection(id='cap', id_of_node='node', Q10=-5, Exp_v_q=2),
        grid.Klink(id_of_injection='cap', id_of_factor='taps', part='q')]

    def test_min_losses(self):
        """
        schema:
        ::
                                                          Q10=-5
                                                          Exp_v_q=2
                                          node          cap
                                            +-------------||
                                            |
              +-----------[ -- ]-----------+------------->
            slack           Br           node          consumer
              V=1.+.0j       y_lo=0.9k-0.95kj           P10=30
                             y_tr=1.3µ+1.5µj            Q10=15

            #.Defk(id=taps min=0 max=5 is_discrete=True)
            #.Klink(id_of_injection=cap id_of_factor=taps part=q)
        """
        model = make_model(
            self._mygrid,
            grid.Defk(id='taps', min=0, max=5, is_discrete=True))
        # optimize according to losses
        res = estim.estimate(model, step_params=[dict(objectives='L')])
        res_vvc = list(rt.make_printables(model, res))
        tappos = res_vvc[1]['injections'].loc['cap','kq']
        self.assertEqual(tappos, 3)

    def test_min_losses2(self):
        """Upper limit test.
        schema:
        ::
                                                          Q10=-5
                                                          Exp_v_q=2
                                          node           cap
                                            +-------------||
                                            |
              +-----------[ -- ]-----------+------------->
            slack           Br           node          consumer
              V=1.+.0j       y_lo=0.9k-0.95kj           P10=30
                             y_tr=1.3µ+1.5µj            Q10=15

            #.Defk(id=taps min=0 max=2 is_discrete=True)
            #.Klink(id_of_injection=cap id_of_factor=taps part=q)
        """
        model = make_model(
            self._mygrid,
            grid.Defk(id='taps', min=0, max=2, is_discrete=True))
        # optimize according to losses
        res = estim.estimate(model, step_params=[dict(objectives='L')])
        res_vvc = list(rt.make_printables(model, res))
        tappos = res_vvc[1]['injections'].loc['cap','kq']
        self.assertEqual(tappos, 2)

    def test_min_cost(self):
        """minimize cost, consider cost of losses, cost of tap change

        test with cost for change:
            * smaller than savings due to decreased losses
            * exceeding potential savings
        """
        model = make_model(
            self._mygrid,
            grid.Defk(id='taps', value=1, min=0, max=5, is_discrete=True))
        # optimize according to losses
        _, res = estim.estimate(model, step_params=[dict(objectives='L')])
        tappos = (
            rt.calculate_electric_data2(model, res)
            ['injections'].loc['cap','kq'])
        self.assertGreater(tappos, 0)
        # make initial tappos = final tappos - 1
        tappos_initial = tappos-1
        model2 = make_model(
            self._mygrid,
            grid.Defk(
                id='taps', value=tappos_initial, min=0, max=5,
                is_discrete=True))
        ini2, res2 = estim.estimate(model2, step_params=[dict(objectives='L')])
        ed_ini2 = rt.calculate_electric_data2(model2, ini2)
        tappos_ini2 = ed_ini2['injections'].loc['cap','kq']
        ed_res2 = rt.calculate_electric_data2(model2, res2)
        tappos_optimized2 = ed_res2['injections'].loc['cap','kq']
        diff_tappos2 = tappos_optimized2 - tappos_ini2
        # check that diff of tappos is indeed 1
        self.assertEquals(diff_tappos2, 1)
        losses_ini2 = ed_ini2['branches'].loc['Br','Ploss_pu']
        losses_res2 = ed_res2['branches'].loc['Br','Ploss_pu']
        diff_losses2 = losses_ini2 - losses_res2
        self.assertGreater(diff_losses2, 0)
        #
        loss_factor = 100
        savings_of_losses = loss_factor * diff_losses2
        #
        # cost of change are smaller than savings of losses
        #
        model2.factors.gen_factordata.loc['taps','cost'] = (
            .999 * savings_of_losses)
        ini3, res3 = estim.estimate(
            model2, step_params=[dict(objectives='LC', floss=loss_factor)])
        ed_res3 = rt.calculate_electric_data2(model2, res3)
        tappos_optimized3 = ed_res3['injections'].loc['cap','kq']
        self.assertEquals(tappos_optimized3, tappos_optimized2)
        #
        # cost of change are greater than savings by potential decrease of
        #   losses ==> optimal position of taps will not change
        #
        model2.factors.gen_factordata.loc['taps','cost'] = (
            1.001 * savings_of_losses)
        ini4, res4 = estim.estimate(
            model2, step_params=[dict(objectives='LC', floss=loss_factor)])
        ed_res4 = rt.calculate_electric_data2(model2, res4)
        tappos_optimized4 = ed_res4['injections'].loc['cap','kq']
        self.assertEquals(tappos_optimized4, tappos_initial)

class Flow_cost(unittest.TestCase):
    """Upper limit test.
    schema:
    ::
                                                            P10=-60
                                               P.cost=-.4   Q10=-15
                                           n---------((~)) gen
                                           |
        slack-------------[ Br ]-----------n----------->   consumer
               P.cost=.5     y_lo=0.9k-0.95kj               P10=60
                             y_tr=1.3µ+1.5µj                Q10=15

        #.Defk(id=k value=1 min=-1.2 max=+1.2)
        #.Klink(id_of_injection=gen id_of_factor=(k k) part=(p q))
    """
    def test_minimize_pcost(self):
        model_vvc = make_model(
           grid.Slacknode('slack'),
           grid.Branch(
               'Br',
               id_of_node_A='slack',
               id_of_node_B='node',
               y_lo=.9e3-.95e3j,
               y_tr=1.3e-6+1.5e-6j),
           grid.Injection('gen', 'node', P10=-60, Q10=-15),
           grid.Injection('consumer', 'node', P10=60, Q10=15),
           grid.Defk(id='k', min=-1.2, max=+1.2),
           grid.Klink(
               id_of_injection='gen',
               id_of_factor=('k', 'k'),
               part=('p', 'q')),
           grid.PValue(id_of_batch='slack_Br', cost=.5),
           grid.Output(
               id_of_batch='slack_Br',
               id_of_device='Br',
               id_of_node='slack'),
           grid.PValue(id_of_batch='n_gen', cost=.4),
           grid.Output(id_of_batch='n_gen', id_of_device='gen'))
        # optimize according to cost
        res0_ = estim.estimate(
            model_vvc,
            step_params=[dict(objectives='C')])
        ini_0, res_0 = rt.make_printables(model_vvc, res0_)
        # max generation
        assert_array_almost_equal(
            res_0['injections'].loc['gen',['kp', 'kq']].to_numpy(),
            [1.2, 1.2])
        # increase the cost of generation
        model_vvc.pvalues\
            .loc[model_vvc.pvalues.id_of_batch=='n_gen', 'cost'] = -.6
        res1_ = estim.estimate(
            model_vvc,
            step_params=[dict(objectives='C')])
        ini_1, res_1 = rt.make_printables(model_vvc, res1_)
        # max consumption
        assert_array_almost_equal(
            res_1['injections'].loc['gen',['kp', 'kq']].to_numpy(),
            [-1.2, -1.2])

if __name__ == '__main__':
    unittest.main()








