# -*- coding: utf-8 -*-
"""
Copyright (C) 2022 pyprg

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
from egrid import make_model
from egrid.builder import (
    Slacknode, Branch, Branchtaps, Injection, PValue, QValue, IValue, Output, 
    Vvalue, Defk, Link)
import src.dssex.present as pr
from src.dssex.estim import calculate
import src.dssex.pfcnum as pfc

# Always use a decimal point for floats. Now and then processing ints
# fails with casadi/pandas/numpy.

# node: 0               1               2
#
#       |     line_0    |     line_1    |
#       +-----=====-----+-----=====-----+
#       |               |               |
#                                      \|/ consumer
#                                       '
# model_entities = [
#     Slacknode(id_of_node='n_0', V=1.+0.j),
#     Branch(
#         id='line_0',
#         id_of_node_A='n_0',
#         id_of_node_B='n_1',
#         y_lo=1e3-1e3j,
#         y_tr=1e-6+1e-6j),
#     Branch(
#         id='line_1',
#         id_of_node_A='n_1',
#         id_of_node_B='n_2',
#         y_lo=1e3-1e3j,
#         y_tr=1e-6+1e-6j),
#     Branchtaps(
#         id='tap_line1',
#         id_of_node='n_1',
#         id_of_branch='line_1',
#         Vstep=.1/16,
#         positionmin=-16,
#         positionneutral=0,
#         positionmax=16,
#         position=0),
#     Injection(
#         id='consumer_0',
#         id_of_node='n_2',
#         P10=30.0,
#         Q10=10.0,
#         Exp_v_p=2.0,
#         Exp_v_q=2.0),
#     # define a scaling factor
#     Defk(id='kp'),
#     # link the factor to the loads
#     Link(
#         objid=('consumer_0'), 
#         part='p', 
#         id='kp'),
#     PValue(
#         id_of_batch='P_line_0',
#         P=42.0),
#     Output(
#         id_of_batch='P_line_0',
#         id_of_device='line_0',
#         id_of_node='n_0'),
#     PValue(
#         id_of_batch='P_line_1',
#         P=42.0),
#     Output(
#         id_of_batch='P_line_1',
#         id_of_device='line_1',
#         id_of_node='n_1')]

model_entities = [
    Slacknode(id_of_node='n_0', V=1.+0.j),
    Branch(
        id='line_0',
        id_of_node_A='n_0',
        id_of_node_B='n_1',
        y_lo=1e3-1e3j,
        y_tr=1e-6+1e-6j
        ),
    Branch(
        id='line_1',
        id_of_node_A='n_1',
        id_of_node_B='n_2',
        y_lo=1e3-1e3j,
        y_tr=1e-6+1e-6j),
    Branchtaps(
        id='tap_line1',
        id_of_node='n_1',
        id_of_branch='line_1',
        Vstep=.1/16,
        positionmin=-16,
        positionneutral=0,
        positionmax=16,
        position=0),
    Injection(
        id='consumer_0',
        id_of_node='n_2',
        P10=30.0,
        Q10=10.0,
        Exp_v_p=0.0,
        Exp_v_q=0.0),
    Injection(
        id='consumer_1',
        id_of_node='n_2',
        P10=30.0,
        Q10=10.0,
        Exp_v_p=2.0,
        Exp_v_q=1.0),
    # define a scaling factor
    Defk(id='kp'),
    # link the factor to the loads
    Link(
        objid=('consumer_0', 'consumer_1'), 
        part='p', 
        id='kp'),
    # measurement
    PValue(
        id_of_batch='PQ_line_0',
        P=40.0),
    # QValue(
    #     id_of_batch='PQ_line_0',
    #     Q=10.0),
    Output(
        id_of_batch='PQ_line_0',
        id_of_device='line_0',
        id_of_node='n_0'),
    # # measurement
    # PValue(
    #     id_of_batch='P_line_1',
    #     P=40.0),
    # Output(
    #     id_of_batch='P_line_1',
    #     id_of_device='line_1',
    #     id_of_node='n_1')
    ]

model00 = make_model(model_entities)

import casadi
from src.dssex.estim2 import (
    make_get_scaling_and_injection_data, 
    vstack, 
    calculate_power_flow2,
    get_batch_expressions,
    get_diff_expressions,
    calculate_power_flow,
    create_v_symbols_gb_expressions,
    get_scaling_factors,
    get_k,
    ri_to_complex,
    get_calculate_from_result)
import numpy as np
import numpy.ma as ma

mymodel = model00
v_syms_gb_ex = create_v_symbols_gb_expressions(mymodel)
get_scaling_and_injection_data = make_get_scaling_and_injection_data(
    mymodel, v_syms_gb_ex['Vnode_syms'], vminsqr=0.8**2, count_of_steps=2)
scaling_data, Iinj_data = get_scaling_and_injection_data(step=0)
inj_to_node = casadi.SX(mymodel.mnodeinj)
Inode = inj_to_node @ Iinj_data[:,:2]
# power flow calculation for initial voltages
succ, Vnode_ri_vals = calculate_power_flow2(
    mymodel, v_syms_gb_ex, scaling_data, Inode)
Vnode_cx_vals = ri_to_complex(Vnode_ri_vals)
print(f'\nVnode_cx_vals:\n{Vnode_cx_vals}')
#%%
succ, vcx = pfc.calculate_power_flow(
    1e-8, 10, mymodel, loadcurve='interpolated')
print(f'\nvcx:\n{vcx}')

selector = 'P'
qu_ids_vals_exprs = get_diff_expressions(
    mymodel, v_syms_gb_ex, Iinj_data, selector)
#%%
# setup solver
Vnode_ri_syms = vstack(v_syms_gb_ex['Vnode_syms'], 2)
syms = casadi.vertcat(Vnode_ri_syms, scaling_data.kvars)
objective = casadi.sumsqr(qu_ids_vals_exprs[2] - qu_ids_vals_exprs[3])

params = casadi.vertcat(
    vstack(v_syms_gb_ex['Vslack_syms']), 
    v_syms_gb_ex['position_syms'], 
    scaling_data.kconsts)
constraints = v_syms_gb_ex['Y_by_V'] + vstack(Inode) 
nlp = {'x': syms, 'f': objective, 'g': constraints, 'p': params}

solver = casadi.nlpsol('solver', 'ipopt', nlp)
# initial values
ini = casadi.vertcat(Vnode_ri_vals, scaling_data.values_of_vars)
# values of parameters
#   Vslack must be negative as Vslack_result + Vslack_in_Inode = 0
#   because the root is searched for with formula: Y * Vresult + Inode = 0
Vslacks_neg = -mymodel.slacks.V
values_of_parameters = casadi.vertcat(
    np.real(Vslacks_neg), np.imag(Vslacks_neg), 
    mymodel.branchtaps.position,
    scaling_data.values_of_consts)
# calculate
r = solver(x0=ini, lbg=0, ubg=0, p=values_of_parameters)
succ = solver.stats()['success']
print('\n',('SUCCESS' if succ else 'F A I L E D'))
x = r['x']
if succ:
    count_of_v_ri = Vnode_ri_vals.size1()
    voltages_ri1 = x[:count_of_v_ri].toarray()
    voltages_ri2 = np.hstack(np.vsplit(voltages_ri1,2))
    voltages_cx = voltages_ri2.view(dtype=np.complex128)
    print(f'\nvoltages_cx:\n{voltages_cx}')
    x_scaling = x[count_of_v_ri:]
    scaling_factors = get_scaling_factors(scaling_data, x_scaling)
    print(f'\nscaling_factors:\n{scaling_factors}')
    k = get_k(scaling_data, x_scaling)
    print(f'\nk:\n{k}')
    get_injected_power = pfc.get_calc_injected_power_fn(
        0.8**2, mymodel.injections, pq_factors=k, loadcurve='interpolated')   
    ed = pfc.calculate_electric_data(
        mymodel, get_injected_power, mymodel.branchtaps.position, voltages_cx) 
#%%
from src.dssex.injections import calculate_cubic_coefficients


# x is constant -> cm is constant
_vminsqr = 0.8**2

Vinj_abs = np.array(
    [[.8], 
     [.7], 
     [0.6], 
     [0.5]])
Vinj_abs_sqr = Vinj_abs * Vinj_abs
Vinj_abs_cub = Vinj_abs_sqr * Vinj_abs
Vvector = np.hstack([Vinj_abs_cub, Vinj_abs_sqr, Vinj_abs])


exp = np.array(
    [[0., 0.], 
      [1., 1.], 
      [2., 2.], 
      [0., 0.]])

# exp = np.array(
#     [[0.], 
#      [1.], 
#      [2.], 
#      [0.]])

c = calculate_cubic_coefficients(_vminsqr, exp)
f_pq = (np.expand_dims(Vvector,axis=1) @ c).reshape(exp.shape)
print(f_pq)

#%%
from src.dssex.estim2 import (_current_into_injection_n)
Vabs_sqr_ = np.sum(np.power(voltages_ri2, 2), axis=1).reshape(-1, 1)
V_ = np.hstack([voltages_ri2, Vabs_sqr_])

Iinj_ri2 = _current_into_injection_n(
    mymodel.injections, 
    mymodel.mnodeinj.T,
    V_,
    k)

#%%
def _create_gb_of_terminals_n(branchterminals):
    """Creates a numpy array of branch-susceptances and branch-conductances.
    
    Parameters
    ----------
    branchterminals: pandas.DataFrame (index_of_terminal)
        * .g_lo, float, longitudinal conductance
        * .b_lo, float, longitudinal susceptance
        * .g_tr_half, float, transversal conductance
        * .b_tr_half, float, transversal susceptance
    
    Returns
    -------
    numpy.array (shape n,4)
        float
        * [:,0] g_mn, mutual conductance
        * [:,1] b_mn, mutual susceptance
        * [:,2] g_mm, self conductance
        * [:,3] b_mn, self susceptance"""
    return (
        branchterminals
        .loc[:,['g_lo', 'b_lo', 'g_tr_half', 'b_tr_half']]
        .to_numpy())

def calculate_factors_of_positions_n(branchtaps, positions):
    """Calculates longitudinal factors of branches.

    Parameters
    ----------
    branchtaps: pandas.DataFrame (index of taps)
        * .Vstep, float voltage diff per tap
        * .positionneutral, int
    position: array_like
        int, vector of positions for branch-terminals with taps

    Returns
    -------
    numpy.array (shape n,1)"""
    return ((1 - branchtaps.Vstep * (positions - branchtaps.positionneutral))
           .to_numpy()
           .reshape(-1,1))

def create_gb_of_terminals_n(branchterminals, branchtaps, positions=None):
    """Creates a vectors (as a numpy array) of branch-susceptances and 
    branch-conductances.
    The intended use is calculating a subset of terminal values. 
    Arguments 'branchtaps' and 'positions' will be selected
    accordingly, hence, it is appropriate to pass the complete branchtaps 
    and positions.

    Parameters
    ----------
    branchterminals: pandas.DataFrame (index_of_terminal)
        * .g_lo, float, longitudinal conductance
        * .b_lo, float, longitudinal susceptance
        * .g_tr_half, float, transversal conductance
        * .b_tr_half, float, transversal susceptance
    branchtaps: pandas.DataFrame (index_of_taps)
        * .index_of_term, int
        * .index_of_other_term, int
        * .Vstep, float voltage diff per tap
        * .positionneutral, int
        * .position, int (if argument positions is None)
    position: array_like (optional, accepts None)
        int, vector of positions for branch-terminals with taps

    Returns
    -------
    numpy.array (shape n,4)
        gb_mn_tot[:,0] - g_mn
        gb_mn_tot[:,1] - b_mn
        gb_mn_tot[:,2] - g_tot
        gb_mn_tot[:,3] - b_tot"""
    index_of_branch_terminals = branchterminals.index
    taps_at_term = branchtaps.index_of_term.isin(index_of_branch_terminals)
    idx_of_term = branchtaps[taps_at_term].index_of_term.to_numpy()
    positions_ = branchtaps.position if positions is None else positions
    flo = calculate_factors_of_positions_n(branchtaps, positions_)
    taps_at_other_term = (
        branchtaps.index_of_other_term.isin(index_of_branch_terminals))
    idx_of_other_term = (
        branchtaps[taps_at_other_term].index_of_other_term.to_numpy())
    # g_lo, b_lo, g_trans, b_trans
    gb_mn_mm = _create_gb_of_terminals_n(branchterminals)
    flo_at_terms = flo[taps_at_term]
    # longitudinal and transversal
    gb_mn_mm[idx_of_term] *= flo_at_terms
    # transversal
    gb_mn_mm[idx_of_term, 2:] *= flo_at_terms
    # longitudinal
    gb_mn_mm[idx_of_other_term, :2] *= flo[taps_at_other_term]
    # gb_mn_mm -> gb_mn_tot
    gb_mn_mm[:, 2:] += gb_mn_mm[:, :2] 
    return gb_mn_mm

gb_mn_tot = create_gb_of_terminals_n(
    mymodel.branchterminals, mymodel.branchtaps)
#%%
# calculate residual node current for solution of optimization
# symbolic
calculate_from_result = get_calculate_from_result(
    mymodel, v_syms_gb_ex, scaling_data, x)
vals_calc = (
    calculate_from_result(qu_ids_vals_exprs[3])
    .toarray()
    .reshape(-1))
import pandas as pd
val_calc = pd.DataFrame(
    {'qu': qu_ids_vals_exprs[0],
     'id': qu_ids_vals_exprs[1],
     'given': qu_ids_vals_exprs[2],
     'calculated': vals_calc})
is_power = val_calc.qu.isin(('P','Q'))
val_calc.loc[is_power, ['given', 'calculated']] *= 3
Inode_sol = calculate_from_result(constraints).toarray().reshape(-1)
print(f'\nInode_sol:\n{Inode_sol}')
# numeric
get_injected_power = pfc.get_calc_injected_power_fn(
    0.8**2, mymodel.injections, pq_factors=k, loadcurve='interpolated')   
ed2 = pfc.calculate_electric_data(
    mymodel, get_injected_power, mymodel.branchtaps.position, voltages_cx) 
#%% calculate power flow
results = [*calculate(model00)]
# print the result
pr.print_estim_results(results)
pr.print_measurements(results)
#%% scale load in order to meet values for active power P
model_PQ_measurements = [
     # measured P/Q pair
     PValue(id_of_batch='pq_line_0', P=30.),
     QValue(id_of_batch='pq_line_0', Q=8.),
     # assign pq_line_0 to terminal
     Output(id_of_batch='pq_line_0', id_of_node='n_0', id_of_device='line_0')]
model_scale_p = [
     # define a scaling factor
     Defk(step=0, id='kp'),
     # link the factor to an injection
     Link(step=0, objid='consumer_0', part='p', id='kp')]
model01 = make_model(model_entities, model_PQ_measurements, model_scale_p)
results01 = [*calculate(model01, parameters_of_steps=[{'objectives': 'P'}])]
# print the result
pr.print_estim_results(results01)
pr.print_measurements(results01)
#%%  scale load in order to meet values for reactive power Q
model_scale_q = [
     # define a scaling factor
     Defk(step=0, id='kq'),
     # link the factor to an injection
     Link(step=0, objid='consumer_0', part='q', id='kq')]
model02 = make_model(
    model_entities,
    model_PQ_measurements,
    model_scale_q)
results02 = [*calculate(model02, parameters_of_steps=[{'objectives': 'Q'}])]
# print the result
pr.print_estim_results(results02)
pr.print_measurements(results02)
#%% scale load with active power P and reactive power Q
model03 = make_model(
    model_entities,
    model_PQ_measurements,
    model_scale_p,
    model_scale_q)
results03 = [*calculate(model03, parameters_of_steps=[{'objectives': 'PQ'}])]
# print the result
pr.print_estim_results(results03)
pr.print_measurements(results03)
#%% PQ-generator
# node: 0               1               2
#
#       |     line_0    |     line_1    |
#       +-----=====-----+-----=====-----+
#       |               |               |
#                                     ((~)) Gen
#
model04_devices = [
    Slacknode(id_of_node='n_0', V=1.+0.j),
    Branch(
        id='line_0',
        id_of_node_A='n_0',
        id_of_node_B='n_1',
        y_lo=1e3-1e3j,
        y_tr=1e-6+1e-6j),
    Branch(
        id='line_1',
        id_of_node_A='n_1',
        id_of_node_B='n_2',
        y_lo=1e3-1e3j,
        y_tr=1e-6+1e-6j),
    Injection(
        id='Gen',
        id_of_node='n_2',
        P10=-5.0,
        Q10=-1.0)]
model04 = make_model(model04_devices)
results04 = [*calculate(model04)]
# print the result
pr.print_estim_results(results04)
pr.print_measurements(results04)
#%% PV-generator, scale Q to meet the voltage setpoint
#   (power flow calculation with PV-generator)
model05_V_setpoint = [Vvalue(id_of_node='n_2', V=1.,)]
model05_scale_q = [
     # define a scaling factor
     Defk(id='kq'),
     # link the factor to the generator
     Link(objid='Gen', part='q', id='kq')]
model05 = make_model(
    model04_devices,
    model05_V_setpoint,
    model05_scale_q)
results05 = [*calculate(model05, parameters_of_steps=[{'objectives': 'V'}])]
# print the result
pr.print_estim_results(results05)
pr.print_measurements(results05)
#%% power flow meshed configuration, consumers, capacitor, PV-generator
#
# leading and trailing underscores are not part of the IDs
#
schema06 = """
                                                                                                       Q10=-4 Exp_v_q=2
                                                                                                n4-|| cap_4_
                                                                                                |
                                                                                                |
                                                         Exp_v_p=1.2                            |      
                                                         Exp_v_q=1                              |     
                                  P10=4 Q10=4            P10=8.3 Q10=4          P10=4 Q10=1     |      P10=4 Q10=1            P10=4 Q10=2      
                           n1--> load_1_          n2--> load_2_          n3--> load_3_          n4--> load_4_          n5--> load_51_          
                           |                      |                      |                      |                      |                      
                           |                      |                      |                      |                      |                      
                           |                      |                      |                      |                      |                      
        I=31               |                      |                      |                      |                      |                      
        P=30 Q=10          |                      |                      |                      |                      |           
    n0(--------line_1-----)n1(--------line_2-----)n2(--------line_3-----)n3(--------line_4-----)n4(--------line_5-----)n5-------> load_52_
    slack=True  y_lo=1e3-1e3j          y_lo=1k-1kj            y_lo=0.9k-0.95kj       y_lo=1k-1kj            y_lo=1k-1kj            P10=4 Q10=2
    V=1.00      y_tr=1e-6+1e-6j        y_tr=1µ+1µj            y_tr=1.3µ+1.5µj        y_tr=1e-6+1e-6j        y_tr=1e-6+1e-6j   
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
    """
model06 = make_model(
    schema06,
    # define a scaling factor
    Defk(id='kq'),
    # link the factor to the generator
    Link(objid='Gen_7', part='q', id='kq'))
result06 = [*calculate(model06, parameters_of_steps=[{'objectives': 'V'}])]
# print the result
pr.print_estim_results(result06)
pr.print_measurements(result06)
#%% optimize for PQV
model07 = make_model(
    schema06,
    # load scaling
    Defk(id=('kp_load', 'kq_load'), step=(0, 1)),
    Link(
        objid=('load_1', 'load_2', 'load_3', 'load_4', 'load_51'), 
        step=(0, 1), 
        part='pq', 
        id=('kp_load', 'kq_load')),
    # generator scaling
    Defk(step=(0, 1), id='kq_gen_7'),
    Link(
        step=(0, 1), 
        objid='Gen_7', 
        part='q',
        id='kq_gen_7'))
result07 = [
    *calculate(
        model07,
        parameters_of_steps=[
            {'objectives': 'PQV'},
            #{'objectives': 'V'},
            ])]
# print the result
pr.print_estim_results(result07)
pr.print_measurements(result07)
#%% power flow meshed configuration, consumers, capacitor, PV-generator,
#   line6 has very hight admittance and is treated as a short circuit
#
# leading and trailing underscores are not part of the IDs, they avoid 
#   the device or node being connected to the adjacent entity
#
schema08 = """
                                                                                                       Q10=-4 Exp_v_q=2
                                                                                                n4-|| cap_4_
                                                                                                |
                                                                                                |
                                                         Exp_v_p=1.2                            |      
                                                         Exp_v_q=1                              |     
                                  P10=4 Q10=4            P10=8.3 Q10=4          P10=4 Q10=1     |      P10=4 Q10=1            P10=4 Q10=2      
                           n1--> load_1_          n2--> load_2_          n3--> load_3_          n4--> load_4_          n5--> load_51_          
                           |                      |                      |                      |                      |                      
                           |                      |                      |                      |                      |                      
                           |                      |                      |                      |                      |                      
        I=31               |                      |                      |                      |                      |                      
        P=30 Q=10          |                      |                      |                      |                      |           
    n0(------- line_1 ----)n1(------- line_2 ----)n2(------- line_3 ----)n3(------- line_4 ----)n4(------- line_5 ----)n5-------> load_52_
    slack=True  y_lo=1e3-1e3j          y_lo=1k-1kj            y_lo=0.9k-0.95kj       y_lo=1k-1kj            y_lo=1k-1kj            P10=4 Q10=2
    V=1.00      y_tr=1e-6+1e-6j        y_tr=1µ+1µj            y_tr=1.3µ+1.5µj        y_tr=1e-6+1e-6j        y_tr=1e-6+1e-6j   
                           |                                                                                           |
                           |                                                                                           |
                           |                                                                                           |
                           |                                                                                           |
                           |           y_lo=1e9-1e9j          y_lo=1e3-1e3j                       y_lo=1e3-1e3j        |
                           |   I=10    y_tr=1e-6+1e-6j        y_tr=1e-6+1e-6j           V=.97     y_tr=1e-6+1e-6j      |
    n0--> load_0_          n1(------- line_6 ----)n6(------- line_7 -------------------)n7(----- line_8 --------------)n5
           P10=8                                  |                                     |
                                                  |                                     |
                                                  |                                     |
                                                  n6--> load_6_          _load_7 <------n7---((~)) Gen_7_
                                                         P10=8             P10=8                    P10=-12
                                                         Q10=8             Q10=4                    Q10=-10 
    """
model08 = make_model(
    schema08,
    # define a scaling factor
    Defk(step=0, id='kq'),
    # link the factor to the generator
    Link(step=0, objid='Gen_7', part='q', id='kq'))
result08 = [*calculate(model08, parameters_of_steps=[{'objectives': 'V'}])]
# print the result
pr.print_estim_results(result08)
pr.print_measurements(result08)
#%% power flow meshed configuration, consumers, capacitor, PV-generator,
#   line6 has very high admittance and is treated as a short circuit
#
# leading and trailing underscores are not part of the IDs
#
schema09 = """
                                                                                                       Q10=-4 Exp_v_q=2
                                                                                                n4-|| cap_4_
                                                                                                |
                                                                                                |
                                                         Exp_v_p=1.2                            |      
                                                         Exp_v_q=1                              |     
                                  P10=4 Q10=4            P10=8.3 Q10=4          P10=4 Q10=1     |      P10=4 Q10=1            P10=4 Q10=2      
                           n1--> load_1_          n2--> load_2_          n3--> load_3_          n4--> load_4_          n5--> load_51_          
                           |                      |                      |                      |                      |                      
                           |                      |                      |                      |                      |                      
                           |                      |                      |                      |                      |                      
        I=31               |                      |                      |                      |                      |                      
        P=30 Q=10          |                      |                      |                      |                      |           
    n0(------- line_1 ----)n1(------- line_2 ----)n2(------- line_3 ----)n3(------- line_4 ----)n4(------- line_5 ----)n5-------> load_52_
    slack=True  y_lo=1e3-1e3j          y_lo=1k-1kj            y_lo=0.9k-0.95kj       y_lo=1k-1kj            y_lo=1k-1kj            P10=4 Q10=2
    V=1.00      y_tr=1e-6+1e-6j        y_tr=1µ+1µj            y_tr=1.3µ+1.5µj        y_tr=1e-6+1e-6j        y_tr=1e-6+1e-6j   
                           |                                                                                           |
                           |                                                                                           |
                           |                                                                                           |
                           |                                                                                           |
                           |           y_lo=1k-1kj            y_lo=1e3-1e3j                       y_lo=1e3-1e3j        |
                           |   I=10    y_tr=1µ+1µj            y_tr=1e-6+1e-6j           V=.97     y_tr=1e-6+1e-6j      |
    n0--> load_0_          n1(------- line_6 ----)n6(------- line_7 -------------------)n7(----- line_8 --------------)n5
           P10=8                                  |                                     |
                                                  |                                     |
                                                  |                                     |
                                                  n6--> load_6_          _load_7 <------n7---((~)) Gen_7_
                                                         P10=8             P10=8                    P10=-12
                                                         Q10=8             Q10=4                    Q10=-10 
    """
model09 = make_model(
    schema09,
    # define a scaling factor
    Defk(step=0, id='kp'),
    # link the factor to the generator
    Link(
        step=0,
        objid=('load_1', 'load_2', 'load_3', 'load_4', 'load_51'), 
        part='p', 
        id='kp'))
# result09 = [*calculate(model09, parameters_of_steps=[{'objectives': 'P'}])]
# pr.print_estim_results(result09)
# pr.print_measurements(result09)
#%%
import casadi
import numpy as np
from src.dssex.estim2 import (ri_to_complex, 
  create_expressions, make_calculate, get_branch_flow_expressions, 
  make_get_branch_expressions, get_node_values, vstack)

model10 = make_model(
    schema09,
    # define a scaling factor
    Defk(id='kp'),
    # link the factor to the loads
    Link(
        objid=('load_1', 'load_2', 'load_3', 'load_4', 'load_51'), 
        part='p', 
        id='kp'),
    PValue(
        id_of_batch='a',
        P=42.0),
    PValue(
        id_of_batch='cap_4',
        P=42.0),
    PValue(
        id_of_batch='load_4',
        P=42.0),
    PValue(
        id_of_batch='line_5',
        P=42.0),
    QValue(
        id_of_batch='a',
        Q=42.0),
    QValue(
        id_of_batch='cap_4',
        Q=42.0),
    QValue(
        id_of_batch='load_4',
        Q=42.0),
    QValue(
        id_of_batch='line_5',
        Q=42.0),
    IValue(
        id_of_batch='a',
        I=42.0),
    IValue(
        id_of_batch='cap_4',
        I=42.0),
    IValue(
        id_of_batch='load_4',
        I=42.0),
    IValue(
        id_of_batch='line_5',
        I=42.0),
    Output(
        id_of_batch='a',
        id_of_device='cap_4'),
    Output(
        id_of_batch='a',
        id_of_device='load_4'),
    Output(
        id_of_batch='a',
        id_of_device='line_5',
        id_of_node='n4'),
    Output(
        id_of_batch='cap_4',
        id_of_device='cap_4'),
    Output(
        id_of_batch='load_4',
        id_of_device='load_4'),
    Output(
        id_of_batch='line_5',
        id_of_device='line_5',
        id_of_node='n4')
    )
mymodel = model10
expr = create_expressions(mymodel)
success, voltages_ri = calculate_power_flow(mymodel, expr=expr)

print("\n","SUCCESS" if success else ">--F-A-I-L-E-D--<", "\n")
if success:
    voltages_complex = ri_to_complex(voltages_ri)
    print(voltages_complex)
    e_data = pfc.calculate_electric_data(
        mymodel, 'interpolated', mymodel.branchtaps.position, voltages_complex)
    # pfc-result processing
    _calculate = make_calculate(
        (vstack(expr['Vnode_syms'][:,0:2]), expr['position_syms']),
        (voltages_ri, mymodel.branchtaps.position))
    branch_PQ_expr = get_branch_flow_expressions(
        expr, 'PQ', mymodel.branchterminals)
    PQ_branch = _calculate(casadi.vcat(branch_PQ_expr))
    branch_I_expr = make_get_branch_expressions(expr, 'I')(mymodel.branchterminals)
    I_branch = _calculate(branch_I_expr)
    Vnode_abs = get_node_values(mymodel.injections.index_of_node, voltages_ri)
#%%
import casadi
import pandas as pd
from src.dssex.estim2 import (
    make_get_scaling_and_injection_data, 
    vstack, make_calculate, 
    calculate_power_flow2,
    get_batch_expressions,
    get_flow_diffs)

mymodel = model00
expr = create_expressions(mymodel)
get_scaling_and_injection_data = make_get_scaling_and_injection_data(
    mymodel, expr['Vnode_syms'], vminsqr=0.8**2, count_of_steps=2)
scaling_data, Iinj_data = get_scaling_and_injection_data(step=0)
inj_to_node = casadi.SX(mymodel.mnodeinj)
Inode = inj_to_node @ Iinj_data[:,0], inj_to_node @ Iinj_data[:,1]
succ, Vnode_ri = calculate_power_flow2(mymodel, expr, scaling_data, Inode)

_calculate = make_calculate(
    (scaling_data.symbols, 
     vstack(expr['Vnode_syms'][:,0:2]), 
     expr['position_syms']),
    (scaling_data.values, 
     Vnode_ri, 
     mymodel.branchtaps.position))

selector = 'I'
batch_expressions = get_batch_expressions(mymodel, expr, Iinj_data, selector)

get_flow_diffs(mymodel, expr, Iinj_data, selector)

#%%


batch_values_ = _calculate(casadi.vcat(batch_expressions.values()))
batch_values = pd.DataFrame(
    {'id_of_batch': batch_expressions.keys(),
     'value': (3. if selector in 'PQ' else 1.) * batch_values_.toarray().reshape(-1)})
print()
print(selector)
print(batch_values.to_markdown())

Vnode_complex = ri_to_complex(Vnode_ri)
e_data = pfc.calculate_electric_data(
    mymodel, 'interpolated', mymodel.branchtaps.position, Vnode_complex)
#%%
from src.dssex.pfcnum import calculate_power_flow

success, voltages = calculate_power_flow(1e-8, 30, mymodel)
print("\n","SUCCESS" if success else ">--F-A-I-L-E-D--<", "\n")
if success:
    Vcomp = voltages.view(dtype=np.complex128)
    print(Vcomp)
    