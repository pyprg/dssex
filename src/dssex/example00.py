# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 08:36:10 2021

@author: pyprg
"""
from egrid import make_model
from egrid.builder import (
    Slacknode, Branch, Injection, PQValue, Output, Vvalue, Defk, Link)
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
        y_mn=1e3-1e3j,
        y_mm_half=1e-6+1e-6j),
    Branch(
        id='line_1',
        id_of_node_A='n_1',
        id_of_node_B='n_2',
        y_mn=1e3-1e3j,
        y_mm_half=1e-6+1e-6j),
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
     PQValue(
        id_of_batch='pq_line_0',
        P=30.,
        Q=8.),
     # assign pq_line_0 to terminal
     Output(
        id_of_batch='pq_line_0',
        id_of_node='n_0',
        id_of_device='line_0')]
model_scale_p = [
     # define a scaling factor
     Defk(step=0, id='kp'),
     # link the factor to an injection
     Link(step=0, objid='consumer_0', part='p', id='kp')]
model01 = make_model(
    model_devices,
    model_PQ_measurements,
    model_scale_p)
results01 = [
    *calculate(
        model01,
        parameters_of_steps=[{'objectives': 'P'}])]
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
results02 = [
   *calculate(
        model02,
        parameters_of_steps=[{'objectives': 'Q'}])]
# print the result
pr.print_estim_results(results02)
pr.print_measurements(results02)
#%% scale load with active power P and reactive power Q
model03 = make_model(
    model_devices,
    model_PQ_measurements,
    model_scale_p,
    model_scale_q)
results03 = [
   *calculate(
        model03,
        parameters_of_steps=[{'objectives': 'PQ'}])]
# print the result
pr.print_estim_results(results03)
pr.print_measurements(results03)
#%% PV-generator
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
        y_mn=1e3-1e3j,
        y_mm_half=1e-6+1e-6j),
    Branch(
        id='line_1',
        id_of_node_A='n_1',
        id_of_node_B='n_2',
        y_mn=1e3-1e3j,
        y_mm_half=1e-6+1e-6j),
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
#%% scale Q to meet the voltage setpoint
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
results05 = [
    *calculate(
        model05,
        parameters_of_steps=[{'objectives': 'V'}])]
# print the result
pr.print_estim_results(results05)
pr.print_measurements(results05)
