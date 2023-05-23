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
                              P10=4 Q10=4            P10=8.3 Q10=4          P10=4 Q10=1     |      P10=4 Q10=1            P10=4 Q10=.3
                       n1--> load_1_          n2--> load_2_          n3--> load_3_          n4--> load_4_          n5--> load_51_
                       |                      |                      |                      |                      |
                       |                      |                      |                      |                      |
    Tlink=taps         |                      |                      |                      |                      |
    I=31               |                      |                      |                      |                      |
    P=30 Q=10          |                      |                      |                      |                      |
n0(--------line_1-----)n1(--------line_2-----)n2(--------line_3-----)n3(--------line_4-----)n4(------line_5-------)n5-------> load_52_
slack=True  y_lo=1e3-1e3j          y_lo=1k-1kj            y_lo=0.9k-0.95kj       y_lo=1k-1kj           y_lo=1k-1kj |           P10=2 Q10=1
V=1.00      y_tr=1e-6+1e-6j        y_tr=1µ+1µj            y_tr=1.3µ+1.5µj        y_tr=1e-6+1e-6j       y_tr=1e-6+1e-6j
                       |                                                                                           |
                       |                                                                                           |
                       |                                                                                           |
                       |                                                                                           |
                       |           y_lo=1e3-1e3j          y_lo=1e3-1e3j                       y_lo=1e3-1e3j        |
                       |   I=10    y_tr=1e-6+1e-6j        y_tr=1e-6+1e-6j           V=.974    y_tr=1e-6+1e-6j      |
                       n1(--------line_6-----)n6(--------line_7--------------------)n7(------line_8---------------)n5
                                              |                                     |
                                              |                                     |
                                              |                                     |
                                              n6--> load_6_          _load_7 <------n7---((~)) Gen_7_
                                                     P10=8             P10=8                    P10=-12
                                                     Q10=8             Q10=4                    Q10=-10


#. Deft(id=taps value=0)

"""

import numpy as np
import pandas as pd
import pfcnum as pfc
from egrid import make_model

model = make_model(schema)



# manual input
kpq = np.ones((len(model.injections), 2), dtype=float)
kpq[:,:] = 1
pos = []#('taps', -16)]
positions = pfc.get_positions(model.factors, pos)

success, vcx = pfc.calculate_power_flow(model, kpq=kpq, positions=positions)
ed = pfc.calculate_electric_data(model, vcx, kpq=kpq, positions=positions)
vals_nodes = ed.node()
vals_branches = ed.branch()
vals_injections = ed.injection()
residual_current = ed.residual_node_current()
vals_nodees = ed.node()

print(np.abs(vcx))

#%%
import dssex.result as rt

branch_res = rt.calculate_branch_results(model, vcx)
injection_res = rt.calculate_injection_results(model, vcx)

res = rt.calculate_electric_data(model, vcx)

#%%
import dssex.result as rt
switch_flow = list(
    rt.get_switch_flow2(model, Vnode=vcx, kpq=kpq, positions=positions))
Icx = pd.concat(switch_flow) if switch_flow else pd.Series([], name='Iterm')

bridgeterminals = model.bridgeterminals

bridgeterminals['Icx'] = Icx
bridgeterminals['S'] = 3 * vcx[bridgeterminals.index_of_node].reshape(-1) * np.conjugate(Icx)

