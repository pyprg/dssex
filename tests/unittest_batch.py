# -*- coding: utf-8 -*-
"""
Copyright (C) 2022, 2023 pyprg

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
Created on Fri Dec 16 00:14:07 2022

@author: pyprg
"""
import unittest
import context # adds parent folder of dssex to search path
import numpy as np
import pandas as pd
import egrid.builder as grid
import dssex.pfcnum as pfc
import dssex.estim as estim
import dssex.batch as batch
import dssex.factors as ft
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
    grid.Injection('consumer_3', 'n_3', P10=30.0, Q10=20.0),
    grid.Deft(
        'taps', type='const', min=-16, max=16, value=-16,
        m=-0.00625, n=1., is_discrete=True), # <- 10 percent increase
    grid.Tlink(id_of_branch='line_0', id_of_node='n_0', id_of_factor='taps')
    )

class Batch(unittest.TestCase):

    def test_ipq(self):
        # prepare
        ipq_batches = [
            # given value of I, P, Q at n_0/line_0
            grid.IValue('batch_0'),
            grid.PValue('batch_0'),
            grid.QValue('batch_0'),
            grid.Output('batch_0', id_of_device='line_0', id_of_node='n_0'),
            # given value of I, P, Q at consumer_1
            grid.IValue('batch_1'),
            grid.PValue('batch_1'),
            grid.QValue('batch_1'),
            grid.Output('batch_1', id_of_device='consumer_1'),
            # given value of I, P, Q at n_1/line_1 and n_1/line_2
            grid.IValue('batch_2'),
            grid.PValue('batch_2'),
            grid.QValue('batch_2'),
            grid.Output('batch_2', id_of_device='line_1', id_of_node='n_1'),
            grid.Output('batch_2', id_of_device='line_2', id_of_node='n_1'),
            # given value of I, P, Q at consumer_2 and consumer_3
            grid.IValue('batch_3'),
            grid.PValue('batch_3'),
            grid.QValue('batch_3'),
            grid.Output('batch_3', id_of_device='consumer_2'),
            grid.Output('batch_3', id_of_device='consumer_3'),
            # given voltage at node n_3
            grid.Vvalue('n_2'),
            grid.Vvalue('n_3')]
        pq_factors = np.ones((3,2), dtype=float)
        model = make_model(grid1, ipq_batches)
        #factordefs = ft.make_factordefs(model)
        # calculate power flow
        gen_factor_symbols = ft._create_symbols_with_ids(
            model.factors.gen_factor_data.index)
        expr = estim.create_v_symbols_gb_expressions(model, gen_factor_symbols)
        success, vnode_ri = estim.calculate_power_flow(
            model, gen_factor_symbols, expr, vminsqr=_VMINSQR)
        vnode_ri2 = np.hstack(np.vsplit(vnode_ri.toarray(),2))
        self.assertTrue(success, "calculate_power_flow shall succeed")
        # check residual current
        vnode_cx = estim.ri_to_complex(vnode_ri)
        get_injected_power = get_injected_power_fn(
            model.injections,
            pq_factors=pq_factors,
            loadcurve='interpolated')
        Inode = pfc.eval_residual_current(
            model, get_injected_power, Vnode=vnode_cx)
        # without slack node, slack is at index 0
        max_dev = norm(Inode[model.count_of_slacks:], np.inf)
        self.assertLess(max_dev, 6e-8, 'residual node current is 0')
        ed = pfc.calculate_electric_data(model, vnode_cx, pq_factors)
        # act
        batch_values = batch.get_batch_values(
            model, vnode_ri2, pq_factors, None, 'IPQV')
        # test
        df_batch = (
            pd.DataFrame(
                dict(zip(['Qu', 'id_of_batch', 'value'], batch_values)))
            .set_index(['id_of_batch', 'Qu']))
        for result_key, batch_key in \
            [(('line_0', 'I0_pu'), ('batch_0', 'I')),
             (('line_0', 'P0_pu'), ('batch_0', 'P')),
             (('line_0', 'Q0_pu'), ('batch_0', 'Q'))]:
            qu = batch_key[1]
            val = (
                (3. if qu in 'PQ' else 1.)
                * df_batch.loc[batch_key[0], qu].value)
            self.assertAlmostEqual(
                val,
                ed.branch().loc[result_key[0], result_key[1]],
                delta=1e-12,
                msg=f'{batch_key[1]} of batch {batch_key[0]} equals '
                    f'{result_key[1]} of device {result_key[0]}')
        for result_key, batch_key in \
            [(('line_0', 'I1_pu'), ('batch_2', 'I')),
             (('line_0', 'P1_pu'), ('batch_2', 'P')),
             (('line_0', 'Q1_pu'), ('batch_2', 'Q'))]:
            qu = batch_key[1]
            val = (
                (-3. if qu in 'PQ' else 1.)
                * df_batch.loc[batch_key[0], qu].value)
            self.assertAlmostEqual(
                val,
                ed.branch().loc[result_key[0], result_key[1]],
                delta=1e-11,
                msg=f'{batch_key[1]} of batch {batch_key[0]} equals '
                    f'{result_key[1]} of device {result_key[0]}')
        for result_key, batch_key in \
            [(('consumer_1', 'I_pu'), ('batch_1', 'I')),
             (('consumer_1', 'P_pu'), ('batch_1', 'P')),
             (('consumer_1', 'Q_pu'), ('batch_1', 'Q'))]:
            qu = batch_key[1]
            val = (
                (3. if qu in 'PQ' else 1.)
                * df_batch.loc[batch_key[0], qu].value)
            self.assertAlmostEqual(
                val,
                ed.injection().loc[result_key[0], result_key[1]],
                delta=1e-12,
                msg=f'{batch_key[1]} of batch {batch_key[0]} equals '
                    f'{result_key[1]} of device {result_key[0]}')
        for result_key, batch_key in \
            [(('line_2', 'I1_pu'), ('batch_3', 'I')),
             (('line_2', 'P1_pu'), ('batch_3', 'P')),
             (('line_2', 'Q1_pu'), ('batch_3', 'Q'))]:
            qu = batch_key[1]
            val = (
                (-3. if qu in 'PQ' else 1.)
                * df_batch.loc[batch_key[0], qu].value)
            self.assertAlmostEqual(
                val,
                ed.branch().loc[result_key[0], result_key[1]],
                delta=2e-7,
                msg=f'{batch_key[1]} of batch {batch_key[0]} equals '
                    f'{result_key[1]} of device {result_key[0]}')
        self.assertAlmostEqual(
            df_batch.loc['n_2', 'V'].value,
            ed.node().loc['n_2', 'V_pu'],
            delta=1e-12,
            msg='voltage at n_2 is result of power flow calculation')
        self.assertAlmostEqual(
            df_batch.loc['n_3', 'V'].value,
            ed.node().loc['n_3', 'V_pu'],
            delta=1e-12,
            msg='voltage at n_3 is result of power flow calculation')

if __name__ == '__main__':
    unittest.main()
