# -*- coding: utf-8 -*-
"""
Copyright (C) 2021, 2022, 2023 pyprg

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

Created on Sun Aug  8 08:36:10 2021

@author: pyprg
"""
schema = """
                                                                                                   Q10=-2 Exp_v_q=2
                                                                                            n4-|| cap_4_
                                                                                            |
                                                                                            |
                                                     Exp_v_p=1.2                            |
                                                     Exp_v_q=1                              |
                              P10=4 Q10=4            P10=8.3 Q10=4          P10=4 Q10=4     |      P10=4 Q10=1            P10=4 Q10=.3
                       n1--> load_1_          n2--> load_2_          n3--> load_3_          n4--> load_4_          n5--> load_51_
                       |                      |                      |                      |                      |
                       |                      |                      |                      |                      |
    Tlink=taps         |                      |                      |                      |                      |
    I=11               |                      |                      |                      |                      |
    P=35 Q=10          |                      |                      |                      |                      |
n0(--------line_1-----)n1(--------line_2-----)n2(--------line_3-----)n3(--------line_4-----)n4(------line_5-------)n5-------> load_52_
slack=True  y_lo=1e3-1e3j          y_lo=1k-1kj            y_lo=0.9k-0.95kj       y_lo=1k-1kj           y_lo=1k-1kj |           P10=2 Q10=1
V=1.00      y_tr=1e-6+1e-6j        y_tr=1µ+1µj            y_tr=1.3µ+1.5µj        y_tr=1e-6+1e-6j       y_tr=1e-6+1e-6j
                       |                                                                                           |
                       |                                                                                           |
                       |                                                                                           |
                       |                                                                                           |
                       |           y_lo=1e3-1e3j          y_lo=1e3-1e3j                       y_lo=1e3-1e3j        |
                       |   I=10    y_tr=1e-6+1e-6j        y_tr=1e-6+1e-6j           V=1.0     y_tr=1e-6+1e-6j      |
                       n1(--------line_6-----)n6(--------line_7--------------------)n7(------line_8---------------)n5
                                              |                                     |
                                              |                                     |
                                              |                                     |
                                              n6--> load_6_          _load_7 <------n7---((~)) Gen_7_
                                                     P10=8             P10=8                    P10=-12
                                                     Q10=8             Q10=4                    Q10=-10


#. Defk(id=(kp kq))
#. Klink(id_of_injection=(load_2 load_3) part=(p q) id_of_factor=(kp kq))
#. Klink(id_of_injection=(load_51 load_6) part=q id_of_factor=kq)
#. Defk(id=kq_Gen7 max=7)
#. Klink(id_of_injection=Gen_7 part=q id_of_factor=kq_Gen7)
# Deft(id=taps value=0)
"""

import numpy as np
import dssex.pfcnum as pfc
import dssex.result as rt
from egrid import make_model

model = make_model(schema)
#%% Power Flow Calculation
# manual input
kpq = np.full((len(model.injections), 2), 1., dtype=float)
pos = []#[('taps', -16)]
positions = pfc.get_positions(model.factors, pos)
success, vcx = pfc.calculate_power_flow(model, kpq=kpq, positions=positions)
# results power flow calculation
calc_pf = rt.make_printable(
    rt.calculate_electric_data(model, vcx, kpq=kpq, positions=positions))
# accuracy
residual_current = pfc.calculate_residual_current(
    model, vcx, positions=positions, kpq=kpq)

#%% State Estimation
import dssex.estim as estim
init, res, res2 = estim.estimate(
    model,
    step_params=[
        # first step: optimize measured PQ
        dict(objectives='PQ'),
        # second step: optimize measured V,
        #   keep PQ at locations of measurement constant
        dict(objectives='V', constraints='PQ')])
calc_init = rt.make_printable(
    rt.calculate_electric_data(model, init[2], kpq=init[3], positions=init[4]))
calc_estim = rt.make_printable(
    rt.calculate_electric_data(model, res[2], kpq=res[3], positions=res[4]))
calc_estim2 = rt.make_printable(
    rt.calculate_electric_data(model, res2[2], kpq=res2[3], positions=res2[4]))
#%% VVC
schema_vvc = """
  +---------------------( () )-----------+------------->
slack                     Tr           node          consumer
 V=1.+.0j   Tlink=taps     y_lo=0.9k-0.95kj           P10=30
                           y_tr=1.3µ+1.5µj            Exp_v_p=0

#. Deft(id=taps value=0 type=var min=-16 max=16 m=-.00625 n=1)
#. Defvl(max=1.05)
"""
model_vvc = make_model(schema_vvc)
# optimize according to losses
res = estim.estimate(model_vvc, step_params=[dict(objectives='L')])
res_vvc = list(rt.make_printables(model_vvc, res))

tappos = res_vvc[1]['branches'].loc['Tr','Tap0']
expected = model_vvc.factors.gen_factordata.loc['taps', 'min']
