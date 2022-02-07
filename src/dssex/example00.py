# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 08:36:10 2021

@author: pyprg
"""
from egrid import make_model
from egrid.builder import (
    Slacknode, PQValue, Output, Branch, Injection, Defk, Link)
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

example00 = [
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
model00 = make_model(example00)
#%% run power flow calculation only
results = list(calculate(model00))
# print the result
pr.print_estim_result(results)
pr.print_measurements(results)
#%% scale load in order to meet values for active power P and reactive power Q
example01 = example00 + [
     # measured P/Q pair
     PQValue(
        id_of_batch='pq_line_0',
        P=30.,
        Q=8.),
     # assign pq_line_0 to terminal
     Output(
        id_of_batch='pq_line_0',
        id_of_node='n_0',
        id_of_device='line_0'),
     # define a load scaling factor
     Defk(step=0, id='k'),
     # link the factor to a current injection
     #  in order to scale its active (p) and reactive (q) power
     Link(step=0, objid='consumer_0', part='pq', id='k')]
model01 = make_model(example01)
# scale in order to meet active power (P) measurement
results = [*calculate(
    model01,
    parameters_of_steps=[{'objectives': 'P'}])]
# print the result
pr.print_estim_result(results)
pr.print_measurements(results)
#%%
# scale in order to meet reactive power (Q) measurement
results = list(calculate(
    model01,
    parameters_of_steps=[{'objectives': 'Q'}]))
# print the result
pr.print_estim_result(results)
pr.print_measurements(results)
