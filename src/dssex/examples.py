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
    Slacknode, Branch, Injection, PValue, QValue, Output, Vvalue, Defk, Link)
from estim import calculate
import present as pr

# Always use a decimal point for floats. Now and then processing ints
# fails with casadi/pandas/numpy.

# node: 0               1               2
#
#       |     line_0    |     line_1    |
#       +-----=====-----+-----=====-----+
#       |               |               |
#                                      \|/ consumer
#                                       '
model_devices = [
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
        id='consumer_0',
        id_of_node='n_2',
        P10=30.0,
        Q10=10.0,
        Exp_v_p=2.0,
        Exp_v_q=2.0)]
model00 = make_model(model_devices)
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
model01 = make_model(model_devices, model_PQ_measurements, model_scale_p)
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
    model_devices,
    model_PQ_measurements,
    model_scale_q)
results02 = [*calculate(model02, parameters_of_steps=[{'objectives': 'Q'}])]
# print the result
pr.print_estim_results(results02)
pr.print_measurements(results02)
#%% scale load with active power P and reactive power Q
model03 = make_model(
    model_devices,
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
     Defk(step=0, id='kq'),
     # link the factor to the generator
     Link(step=0, objid='Gen', part='q', id='kq')]
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
    Defk(step=0, id='kq'),
    # link the factor to the generator
    Link(step=0, objid='Gen_7', part='q', id='kq'))
result06 = [*calculate(model06, parameters_of_steps=[{'objectives': 'V'}])]
# print the result
pr.print_estim_results(result06)
pr.print_measurements(result06)
#%% optimize for PQV
model07 = make_model(
    schema06,
    # load scaling
    Defk(step=(0, 1), id=('kp_load', 'kq_load')),
    Link(
        step=(0, 1), 
        objid=('load_1', 'load_2', 'load_3', 'load_4', 'load_51'), 
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
                           |           y_lo=1e9-1e9j          y_lo=1e3-1e3j                       y_lo=1e3-1e3j        |
                           |   I=10    y_tr=1e-6+1e-6j        y_tr=1e-6+1e-6j           V=.974    y_tr=1e-6+1e-6j      |
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
result09 = [*calculate(model09, parameters_of_steps=[{'objectives': 'P'}])]
pr.print_estim_results(result09)
pr.print_measurements(result09)

