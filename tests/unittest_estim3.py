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
import numpy as np
import pandas as pd
import egrid.builder as grid
import dssex.pfcnum as pfc
import dssex.estim as estim
import dssex.estimnext as estimnext
import dssex.batch as batch
import dssex.factors2 as ft
from functools import partial
from numpy.linalg import norm
from egrid import make_model


# square of voltage magnitude, minimum value for load curve,
#   if value is below _VMINSQR the load curves for P and Q converge
#   towards a linear load curve which is 0 when V=0; P(V=0)=0, Q(V=0)=0
_VMINSQR = 0.8**2
get_injected_power_fn = partial(pfc.get_calc_injected_power_fn, _VMINSQR)

# node: 0               1               2
#
#       |     line_0    |     line_1    |
#       +-----=====-----+-----=====-----+
#       |               |               |
#                       |              \|/ consumer_1
#                       |               '
#                       |
# node:                 |               3
#                       |
#                       |     line_2    |
#                       +-----=====-----+---------------+
#                       |               |               |
#                                      \|/ consumer_2  \|/ consumer_3
#                                       '               '

grid1 = (
    grid.Slacknode('n_0', V=1.+0.j),
    grid.Branch('line_0', 'n_0', 'n_1', y_lo=1e3-1e3j),
    grid.Branch('line_1', 'n_1', 'n_2', y_lo=1e3-1e3j),
    grid.Branch('line_2', 'n_1', 'n_3', y_lo=1e3-1e3j),
    grid.Injection('consumer_1', 'n_2', P10=10.0, Q10=10.0),
    grid.Injection('consumer_2', 'n_3', P10=20.0, Q10=15.0),
    grid.Injection('consumer_3', 'n_3', P10=30.0, Q10=20.0))

order_of_nodes = pd.Series(
    range(4), dtype=np.int64, index=[f'n_{idx}' for idx in range(4)])

class Calculate_power_flow(unittest.TestCase):
    
    def test_taps_factor(self):
        pq_factors = np.ones((3,2), dtype=float)
        # first: tapsposition == 0 -> no impact to voltage        
        model0 = make_model(
            grid1,
            grid.Deff(
                'taps', type='const', min=-16, max=16, value=0,
                m=-0.00625, n=1., is_discrete=True),
            grid.Link(
                objid='line_0', id='taps', nodeid='n_0', 
                cls=grid.Terminallink))
        factordefs0 = ft.make_factordefs(model0)
        # calculate power flow
        expr0 = estimnext.create_v_symbols_gb_expressions(model0, factordefs0)
        success0, vnode_ri0 = estim.calculate_power_flow(
            model0, factordefs0, expr0, vminsqr=_VMINSQR)
        self.assertTrue(success0, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx0 = estim.ri_to_complex(vnode_ri0)
        get_injected_power0 = get_injected_power_fn(
            model0.injections,
            pq_factors=pq_factors,
            loadcurve='interpolated')
        Inode0 = pfc.eval_residual_current(
            model0, get_injected_power0, Vnode=vnode_cx0)
        # without slack node, slack is at index 0
        max_dev0 = norm(Inode0[model0.count_of_slacks:], np.inf)
        self.assertLess(max_dev0, 3e-8, 'residual node current is 0')
        ed = pfc.calculate_electric_data(model0, vnode_cx0, pq_factors)
        idx_v = model0.nodes.reindex(order_of_nodes.index).index_of_node
        # df_Vcx0 = pd.DataFrame(
        #     {'id_of_node': order_of_nodes.index, 
        #       'Vcx': vnode_cx0[idx_v].reshape(-1)})
        # print(df_Vcx0)
        #
        # second: tapsposition == -16 -> voltage increase ~10%        
        model1 = make_model(
            grid1,
            grid.Deff(
                'taps', type='const', min=-16, max=16, value=-16, # <- V +10%
                m=-0.00625, n=1., is_discrete=True),
            grid.Link(
                objid='line_0', id='taps', nodeid='n_0', 
                cls=grid.Terminallink))
        factordefs1 = ft.make_factordefs(model1)
        # calculate power flow
        expr1 = estimnext.create_v_symbols_gb_expressions(model1, factordefs1)
        success1, vnode_ri1 = estim.calculate_power_flow(
            model1, factordefs1, expr1, vminsqr=_VMINSQR)
        self.assertTrue(success1, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx1 = estim.ri_to_complex(vnode_ri1)
        get_injected_power1 = get_injected_power_fn(
            model1.injections,
            pq_factors=pq_factors,
            loadcurve='interpolated')
        Inode1 = pfc.eval_residual_current(
            model1, get_injected_power1, Vnode=vnode_cx1)
        # without slack node, slack is at index 0
        max_dev1 = norm(Inode1[model1.count_of_slacks:], np.inf)
        self.assertLess(max_dev1, 3e-8, 'residual node current is 0')
        ed = pfc.calculate_electric_data(model1, vnode_cx1, pq_factors)
        idx_v = model1.nodes.reindex(order_of_nodes.index).index_of_node
        # df_Vcx1 = pd.DataFrame(
        #     {'id_of_node': order_of_nodes.index, 
        #      'Vcx': vnode_cx1[idx_v].reshape(-1)})
        # print(df_Vcx1)
        diff_max = norm(
            #                                       10% | no slack
            (vnode_cx0[idx_v] - (vnode_cx1[idx_v] / 1.1))[1:], 
            np.inf)
        self.assertLess(diff_max, 6e-3, "voltage increase about 10%")

if __name__ == '__main__':
    unittest.main()








