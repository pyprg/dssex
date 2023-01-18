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
import os, sys
_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _path)
import dssex.present as pr
import dssex.pfcnum as pfc
from functools import partial
from egrid.builder import (
    Slacknode, Branch, Branchtaps, Injection, PValue, QValue, IValue, Output, 
    Vvalue, Defk, Link)
from egrid import make_model


# square of voltage magnitude, minimum value for load curve, 
#   if value is below _VMINSQR the load curves for P and Q converge
#   towards a linear load curve which is 0 when V=0; P(V=0)=0, Q(V=0)=0
_VMINSQR = 0.8**2
get_injected_power_fn = partial(pfc.get_calc_injected_power_fn, _VMINSQR)

# Always use a decimal point for floats. Now and then processing ints
# fails with casadi/pandas/numpy.

# node: 0               1               2
#
#       |     line_0    |     line_1    |
#       +-----=====-----+-----=====-----+
#       |               |               |
#                                      \|/ consumer_1/2
#                                       '

model_entities = [
    Slacknode(id_of_node='n_0', V=1.+0.j),
    Branch(
        id='line_0',
        id_of_node_A='n_0',
        id_of_node_B='n_1',
        y_lo=1e3-1e3j,
        y_tr=1e-6+1e-6j
        ),
    # Branch(
    #     id='line_1',
    #     id_of_node_A='n_1',
    #     id_of_node_B='n_2',
    #     y_lo=1e3-1e3j,
    #     y_tr=1e-6+1e-6j),
    # Branchtaps(
    #     id='tap_line1',
    #     id_of_node='n_1',
    #     id_of_branch='line_1',
    #     Vstep=.1/16,
    #     positionmin=-16,
    #     positionneutral=0,
    #     positionmax=16,
    #     position=0),
    Injection(
        id='consumer_0',
        id_of_node='n_1',
        P10=100.0,
        Q10=50.0,
        Exp_v_p=0.0,
        Exp_v_q=2.0),
    # Injection(
    #     id='consumer_1',
    #     id_of_node='n_2',
    #     P10=30.0,
    #     Q10=10.0,
    #     Exp_v_p=0.0,
    #     Exp_v_q=2.0),
    # define a scaling factor
    # Defk(id='kp', step=(0,1,2)),
    # Defk(id='kq', step=(0,1,2)),
    # link the factor to the loads
    Link(
        objid='consumer_0', 
        part='p', 
        id='kp',
        step=(0,1,2)),
    Link(
        objid='consumer_0', 
        part='q', 
        id='kq',
        step=(0,1,2)),
    # measurement
    PValue(
        id_of_batch='n_0_line_0',
        P=40.0),
    QValue(
        id_of_batch='n_0_line_0',
        Q=10.0),
    # IValue(
    #     id_of_batch='n_0_line_0',
    #     I=42.0),
    Output(
        id_of_batch='n_0_line_0',
        id_of_device='line_0',
        id_of_node='n_0'),
    # measurement
    # PValue(
    #     id_of_batch='n_1_line_1',
    #     P=40.0),
    # Output(
    #     id_of_batch='n_1_line_1',
    #     id_of_device='line_1',
    #     id_of_node='n_1'),
    # Vvalue(
    #     id_of_node='n_1',
    #     V=.95)
    ]


from dssex.estim2 import (estimate, get_Vcx_kpq, get_step_data_fns)


count_of_steps = 3
objectives = ''

model = make_model(model_entities)
make_step_data, next_step_data = get_step_data_fns(model, 3)



step_data = make_step_data(step=0)
succ, voltages_ri, k = estimate(*step_data)
print('\n',(">> SUCCESS <<" if succ else 'F A I L E D'),'\n')
if succ:
    voltages_cx, kpq = get_Vcx_kpq(step_data.scaling_data, voltages_ri, k)
    print(f'\nvoltages_cx:\n{voltages_cx}')
    print(f'\nkpq:\n{kpq}')
    result = pfc.calculate_electric_data2(model, voltages_cx, kpq)
    result_branch = result.branch()
    result_injection = result.injection()
    # print(f'\nbranches:\n{result.branch()}')
    # print(f'\ninjections:\n{result.injection()}')

#%%
 
step_data_1 = next_step_data(1, step_data, voltages_ri, k)
succ_1, voltages_ri_1, k_1 = estimate(*step_data_1, Vnode_ri_ini=voltages_ri)

print('\n',(">> SUCCESS <<" if succ else 'F A I L E D'),'\n')
if succ:
    voltages_cx_1, kpq_1 = get_Vcx_kpq(step_data.scaling_data, voltages_ri, k)
    print(f'\nvoltages_cx:\n{voltages_cx}')
    print(f'\nkpq:\n{kpq}')
    result_1 = pfc.calculate_electric_data2(model, voltages_cx_1, kpq_1)
    result_branch_1 = result.branch()
    result_injection_1 = result.injection()
#%% scale load in order to meet values for active power P
# model_PQ_measurements = [
#      # measured P/Q pair
#      PValue(id_of_batch='pq_line_0', P=30.),
#      QValue(id_of_batch='pq_line_0', Q=8.),
#      # assign pq_line_0 to terminal
#      Output(id_of_batch='pq_line_0', id_of_node='n_0', id_of_device='line_0')]
# model_scale_p = [
#      # define a scaling factor
#      Defk(step=0, id='kp'),
#      # link the factor to an injection
#      Link(step=0, objid='consumer_0', part='p', id='kp')]
# model01 = make_model(model_entities, model_PQ_measurements, model_scale_p)
# #%%
# results01 = [*calculate(model01, parameters_of_steps=[{'objectives': 'P'}])]
# # print the result
# pr.print_estim_results(results01)
# pr.print_measurements(results01)
# #%%  scale load in order to meet values for reactive power Q
# model_scale_q = [
#      # define a scaling factor
#      Defk(step=0, id='kq'),
#      # link the factor to an injection
#      Link(step=0, objid='consumer_0', part='q', id='kq')]
# model02 = make_model(
#     model_entities,
#     model_PQ_measurements,
#     model_scale_q)
# results02 = [*calculate(model02, parameters_of_steps=[{'objectives': 'Q'}])]
# # print the result
# pr.print_estim_results(results02)
# pr.print_measurements(results02)
# #%% scale load with active power P and reactive power Q
# model03 = make_model(
#     model_entities,
#     model_PQ_measurements,
#     model_scale_p,
#     model_scale_q)
# results03 = [*calculate(model03, parameters_of_steps=[{'objectives': 'PQ'}])]
# # print the result
# pr.print_estim_results(results03)
# pr.print_measurements(results03)
# #%% PQ-generator
# # node: 0               1               2
# #
# #       |     line_0    |     line_1    |
# #       +-----=====-----+-----=====-----+
# #       |               |               |
# #                                     ((~)) Gen
# #
# model04_devices = [
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
#     Injection(
#         id='Gen',
#         id_of_node='n_2',
#         P10=-5.0,
#         Q10=-1.0)]
# model04 = make_model(model04_devices)
# results04 = [*calculate(model04)]
# # print the result
# pr.print_estim_results(results04)
# pr.print_measurements(results04)
# #%% PV-generator, scale Q to meet the voltage setpoint
# #   (power flow calculation with PV-generator)
# model05_V_setpoint = [Vvalue(id_of_node='n_2', V=1.,)]
# model05_scale_q = [
#      # define a scaling factor
#      Defk(id='kq'),
#      # link the factor to the generator
#      Link(objid='Gen', part='q', id='kq')]
# model05 = make_model(
#     model04_devices,
#     model05_V_setpoint,
#     model05_scale_q)
# results05 = [*calculate(model05, parameters_of_steps=[{'objectives': 'V'}])]
# # print the result
# pr.print_estim_results(results05)
# pr.print_measurements(results05)
# #%% power flow meshed configuration, consumers, capacitor, PV-generator
# #
# # leading and trailing underscores are not part of the IDs
# #
# schema06 = """
#                                                                                                        Q10=-4 Exp_v_q=2
#                                                                                                 n4-|| cap_4_
#                                                                                                 |
#                                                                                                 |
#                                                          Exp_v_p=1.2                            |      
#                                                          Exp_v_q=1                              |     
#                                   P10=4 Q10=4            P10=8.3 Q10=4          P10=4 Q10=1     |      P10=4 Q10=1            P10=4 Q10=2      
#                            n1--> load_1_          n2--> load_2_          n3--> load_3_          n4--> load_4_          n5--> load_51_          
#                            |                      |                      |                      |                      |                      
#                            |                      |                      |                      |                      |                      
#                            |                      |                      |                      |                      |                      
#         I=31               |                      |                      |                      |                      |                      
#         P=30 Q=10          |                      |                      |                      |                      |           
#     n0(--------line_1-----)n1(--------line_2-----)n2(--------line_3-----)n3(--------line_4-----)n4(--------line_5-----)n5-------> load_52_
#     slack=True  y_lo=1e3-1e3j          y_lo=1k-1kj            y_lo=0.9k-0.95kj       y_lo=1k-1kj            y_lo=1k-1kj            P10=4 Q10=2
#     V=1.00      y_tr=1e-6+1e-6j        y_tr=1µ+1µj            y_tr=1.3µ+1.5µj        y_tr=1e-6+1e-6j        y_tr=1e-6+1e-6j   
#                            |                                                                                           |
#                            |                                                                                           |
#                            |                                                                                           |
#                            |                                                                                           |
#                            |           y_lo=1e3-1e3j          y_lo=1e3-1e3j                       y_lo=1e3-1e3j        |
#                            |   I=10    y_tr=1e-6+1e-6j        y_tr=1e-6+1e-6j           V=.974    y_tr=1e-6+1e-6j      |
#                            n1(--------line_6-----)n6(--------line_7--------------------)n7(------line_8---------------)n5
#                                                   |                                     |
#                                                   |                                     |
#                                                   |                                     |
#                                                   n6--> load_6_          _load_7 <------n7---((~)) Gen_7_
#                                                          P10=8             P10=8                    P10=-12
#                                                          Q10=8             Q10=4                    Q10=-10 
#     """
# model06 = make_model(
#     schema06,
#     # define a scaling factor
#     Defk(id='kq'),
#     # link the factor to the generator
#     Link(objid='Gen_7', part='q', id='kq'))
# result06 = [*calculate(model06, parameters_of_steps=[{'objectives': 'V'}])]
# # print the result
# pr.print_estim_results(result06)
# pr.print_measurements(result06)
# #%% optimize for PQV
# model07 = make_model(
#     schema06,
#     # load scaling
#     Defk(id=('kp_load', 'kq_load'), step=(0, 1)),
#     Link(
#         objid=('load_1', 'load_2', 'load_3', 'load_4', 'load_51'), 
#         step=(0, 1), 
#         part='pq', 
#         id=('kp_load', 'kq_load')),
#     # generator scaling
#     Defk(step=(0, 1), id='kq_gen_7'),
#     Link(
#         step=(0, 1), 
#         objid='Gen_7', 
#         part='q',
#         id='kq_gen_7'))
# result07 = [
#     *calculate(
#         model07,
#         parameters_of_steps=[
#             {'objectives': 'PQV'},
#             #{'objectives': 'V'},
#             ])]
# # print the result
# pr.print_estim_results(result07)
# pr.print_measurements(result07)
# #%% power flow meshed configuration, consumers, capacitor, PV-generator,
# #   line6 has very hight admittance and is treated as a short circuit
# #
# # leading and trailing underscores are not part of the IDs, they avoid 
# #   the device or node being connected to the adjacent entity
# #
# schema08 = """
#                                                                                                        Q10=-4 Exp_v_q=2
#                                                                                                 n4-|| cap_4_
#                                                                                                 |
#                                                                                                 |
#                                                          Exp_v_p=1.2                            |      
#                                                          Exp_v_q=1                              |     
#                                   P10=4 Q10=4            P10=8.3 Q10=4          P10=4 Q10=1     |      P10=4 Q10=1            P10=4 Q10=2      
#                            n1--> load_1_          n2--> load_2_          n3--> load_3_          n4--> load_4_          n5--> load_51_          
#                            |                      |                      |                      |                      |                      
#                            |                      |                      |                      |                      |                      
#                            |                      |                      |                      |                      |                      
#         I=31               |                      |                      |                      |                      |                      
#         P=30 Q=10          |                      |                      |                      |                      |           
#     n0(------- line_1 ----)n1(------- line_2 ----)n2(------- line_3 ----)n3(------- line_4 ----)n4(------- line_5 ----)n5-------> load_52_
#     slack=True  y_lo=1e3-1e3j          y_lo=1k-1kj            y_lo=0.9k-0.95kj       y_lo=1k-1kj            y_lo=1k-1kj            P10=4 Q10=2
#     V=1.00      y_tr=1e-6+1e-6j        y_tr=1µ+1µj            y_tr=1.3µ+1.5µj        y_tr=1e-6+1e-6j        y_tr=1e-6+1e-6j   
#                            |                                                                                           |
#                            |                                                                                           |
#                            |                                                                                           |
#                            |                                                                                           |
#                            |           y_lo=1e9-1e9j          y_lo=1e3-1e3j                       y_lo=1e3-1e3j        |
#                            |   I=10    y_tr=1e-6+1e-6j        y_tr=1e-6+1e-6j           V=.97     y_tr=1e-6+1e-6j      |
#     n0--> load_0_          n1(------- line_6 ----)n6(------- line_7 -------------------)n7(----- line_8 --------------)n5
#            P10=8                                  |                                     |
#                                                   |                                     |
#                                                   |                                     |
#                                                   n6--> load_6_          _load_7 <------n7---((~)) Gen_7_
#                                                          P10=8             P10=8                    P10=-12
#                                                          Q10=8             Q10=4                    Q10=-10 
#     """
# model08 = make_model(
#     schema08,
#     # define a scaling factor
#     Defk(step=0, id='kq'),
#     # link the factor to the generator
#     Link(step=0, objid='Gen_7', part='q', id='kq'))
# result08 = [*calculate(model08, parameters_of_steps=[{'objectives': 'V'}])]
# # print the result
# pr.print_estim_results(result08)
# pr.print_measurements(result08)
# #%% power flow meshed configuration, consumers, capacitor, PV-generator,
# #   line6 has very high admittance and is treated as a short circuit
# #
# # leading and trailing underscores are not part of the IDs
# #
# schema09 = """
#                                                                                                        Q10=-4 Exp_v_q=2
#                                                                                                 n4-|| cap_4_
#                                                                                                 |
#                                                                                                 |
#                                                          Exp_v_p=1.2                            |      
#                                                          Exp_v_q=1                              |     
#                                   P10=4 Q10=4            P10=8.3 Q10=4          P10=4 Q10=1     |      P10=4 Q10=1            P10=4 Q10=2      
#                            n1--> load_1_          n2--> load_2_          n3--> load_3_          n4--> load_4_          n5--> load_51_          
#                            |                      |                      |                      |                      |                      
#                            |                      |                      |                      |                      |                      
#                            |                      |                      |                      |                      |                      
#         I=31               |                      |                      |                      |                      |                      
#         P=30 Q=10          |                      |                      |                      |                      |           
#     n0(------- line_1 ----)n1(------- line_2 ----)n2(------- line_3 ----)n3(------- line_4 ----)n4(------- line_5 ----)n5-------> load_52_
#     slack=True  y_lo=1e3-1e3j          y_lo=1k-1kj            y_lo=0.9k-0.95kj       y_lo=1k-1kj            y_lo=1k-1kj            P10=4 Q10=2
#     V=1.00      y_tr=1e-6+1e-6j        y_tr=1µ+1µj            y_tr=1.3µ+1.5µj        y_tr=1e-6+1e-6j        y_tr=1e-6+1e-6j   
#                            |                                                                                           |
#                            |                                                                                           |
#                            |                                                                                           |
#                            |                                                                                           |
#                            |           y_lo=1k-1kj            y_lo=1e3-1e3j                       y_lo=1e3-1e3j        |
#                            |   I=10    y_tr=1µ+1µj            y_tr=1e-6+1e-6j           V=.97     y_tr=1e-6+1e-6j      |
#     n0--> load_0_          n1(------- line_6 ----)n6(------- line_7 -------------------)n7(----- line_8 --------------)n5
#            P10=8                                  |                                     |
#                                                   |                                     |
#                                                   |                                     |
#                                                   n6--> load_6_          _load_7 <------n7---((~)) Gen_7_
#                                                          P10=8             P10=8                    P10=-12
#                                                          Q10=8             Q10=4                    Q10=-10 
#     """
# model09 = make_model(
#     schema09,
#     # define a scaling factor
#     Defk(step=0, id='kp'),
#     # link the factor to the generator
#     Link(
#         step=(0, 1, 2),
#         objid=('load_1', 'load_2', 'load_3', 'load_4', 'load_51'), 
#         part='pq',
#         id=['kp','kq']))
# # result09 = [*calculate(model09, parameters_of_steps=[{'objectives': 'P'}])]
# # pr.print_estim_results(result09)
# # pr.print_measurements(result09)
# #%%
# from src.dssex.estim2 import (
#   create_expressions, make_calculate, get_branch_flow_expressions, 
#   make_get_branch_expressions, get_node_expressions)

# model10 = make_model(
#     schema09,
#     # define a scaling factor
#     Defk(id='kp'),
#     # link the factor to the loads
#     Link(
#         objid=('load_1', 'load_2', 'load_3', 'load_4', 'load_51'), 
#         part='p', 
#         id='kp'),
#     PValue(
#         id_of_batch='a',
#         P=42.0),
#     PValue(
#         id_of_batch='cap_4',
#         P=42.0),
#     PValue(
#         id_of_batch='load_4',
#         P=42.0),
#     PValue(
#         id_of_batch='line_5',
#         P=42.0),
#     QValue(
#         id_of_batch='a',
#         Q=42.0),
#     QValue(
#         id_of_batch='cap_4',
#         Q=42.0),
#     QValue(
#         id_of_batch='load_4',
#         Q=42.0),
#     QValue(
#         id_of_batch='line_5',
#         Q=42.0),
#     IValue(
#         id_of_batch='a',
#         I=42.0),
#     IValue(
#         id_of_batch='cap_4',
#         I=42.0),
#     IValue(
#         id_of_batch='load_4',
#         I=42.0),
#     IValue(
#         id_of_batch='line_5',
#         I=42.0),
#     Output(
#         id_of_batch='a',
#         id_of_device='cap_4'),
#     Output(
#         id_of_batch='a',
#         id_of_device='load_4'),
#     Output(
#         id_of_batch='a',
#         id_of_device='line_5',
#         id_of_node='n4'),
#     Output(
#         id_of_batch='cap_4',
#         id_of_device='cap_4'),
#     Output(
#         id_of_batch='load_4',
#         id_of_device='load_4'),
#     Output(
#         id_of_batch='line_5',
#         id_of_device='line_5',
#         id_of_node='n4')
#     )
# mymodel = model10
# expr = create_expressions(mymodel)
# success, voltages_ri = calculate_power_flow(mymodel, expr=expr)

# print("\n","SUCCESS" if success else ">--F-A-I-L-E-D--<", "\n")
# if success:
#     voltages_complex = ri_to_complex(voltages_ri)
#     print(voltages_complex)
#     e_data = pfc.calculate_electric_data(
#         mymodel, 'interpolated', mymodel.branchtaps.position, voltages_complex)
#     # pfc-result processing
#     _calculate = make_calculate(
#         (vstack(expr['Vnode_syms'][:,0:2]), expr['position_syms']),
#         (voltages_ri, mymodel.branchtaps.position))
#     branch_PQ_expr = get_branch_flow_expressions(
#         expr, 'PQ', mymodel.branchterminals)
#     PQ_branch = _calculate(casadi.vcat(branch_PQ_expr))
#     branch_I_expr = make_get_branch_expressions(expr, 'I')(mymodel.branchterminals)
#     I_branch = _calculate(branch_I_expr)
#     Vnode_abs = get_node_expressions(
#         mymodel.injections.index_of_node, voltages_ri)
# #%%
# import casadi
# import pandas as pd
# from src.dssex.estim2 import (
#     make_get_scaling_and_injection_data, 
#     vstack, make_calculate, 
#     get_flow_diffs)

# expr = create_expressions(model)
# get_scaling_and_injection_data = make_get_scaling_and_injection_data(
#     mymodel, expr['Vnode_syms'], vminsqr=0.8**2, count_of_steps=2)
# scaling_data, Iinj_data = get_scaling_and_injection_data(step=0)
# inj_to_node = casadi.SX(mymodel.mnodeinj)
# Inode = inj_to_node @ Iinj_data[:,0], inj_to_node @ Iinj_data[:,1]
# succ, Vnode_ri = calculate_power_flow2(mymodel, expr, scaling_data, Inode)

# _calculate = make_calculate(
#     (scaling_data.symbols, 
#      vstack(expr['Vnode_syms'][:,0:2]), 
#      expr['position_syms']),
#     (scaling_data.values, 
#      Vnode_ri, 
#      mymodel.branchtaps.position))

# selector = 'I'
# batch_expressions = get_batch_expressions(mymodel, expr, Iinj_data, selector)

# get_flow_diffs(mymodel, expr, Iinj_data, selector)

# #%%


# batch_values_ = _calculate(casadi.vcat(batch_expressions.values()))
# batch_values = pd.DataFrame(
#     {'id_of_batch': batch_expressions.keys(),
#      'value': (3. if selector in 'PQ' else 1.) * batch_values_.toarray().reshape(-1)})
# print()
# print(selector)
# print(batch_values.to_markdown())

# Vnode_complex = ri_to_complex(Vnode_ri)
# e_data = pfc.calculate_electric_data(
#     mymodel, 'interpolated', mymodel.branchtaps.position, Vnode_complex)
# #%%
# from src.dssex.pfcnum import calculate_power_flow

# success, voltages = calculate_power_flow(1e-8, 30, mymodel)
# print("\n","SUCCESS" if success else ">--F-A-I-L-E-D--<", "\n")
# if success:
#     Vcomp = voltages.view(dtype=np.complex128)
#     print(Vcomp)
    