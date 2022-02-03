# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 08:36:10 2021

@author: Carsten Laves
"""
import present as pr
import debug as debug
from egrid import (get_model, 
    Slacknode, PQValue, IValue, Output, Branch, Branchtaps,
    Injection, Defk, Link, Vvalue, IValue, PQValue)          
from estim import estimation_steps, Term

# Always use a decimal point for floats. Now and then processing ints
# fails with casadi/pandas/numpy.
 
"""
node: 0               1               2
		 
      |      line     |     line      |
      +-----=====-----+-----=====-----+
      |               |               |
                                     \|/ consumer
                                      '
"""
grid_a = ([
    Slacknode(id_of_node='n_0', V=1.+0.j),
    PQValue(
        id_of_batch='pq_line_0', 
        P=30., 
        Q=8.),
    Output(
        id_of_batch='pq_line_0', 
        id_of_node='n_0',  
        id_of_device='line_0'),
    IValue(
        id_of_batch='i_line_0',
        I=40.0),
    Output(
        id_of_batch='i_line_0', 
        id_of_node='n_0',  
        id_of_device='line_0'),
    Branch(
        id='line_0',
        id_of_node_A='n_0', 
        id_of_node_B='n_1',
        y_mn=1e3-1e3j,
        y_mm_half=1e-6+1e-6j),
    Branchtaps(
        id='taps_0',
        id_of_node='n_0', 
        id_of_branch='line_0', 
        Vstep=.2/33, 
        positionmin=-16, 
        positionneutral=0, 
        positionmax=16),
    Branch(
        id='line_1',
        id_of_node_A='n_1', 
        id_of_node_B='n_2',
        y_mn=1e3-1e3j,
        y_mm_half=1e-6+1e-6j),
    Output(
        id_of_batch='pq_consumer_0', 
        id_of_device='consumer_0'),
    Output(
        id_of_batch='i_consumer_0', 
        id_of_device='consumer_0'),
    Injection(
        id='consumer_0', 
        id_of_node='n_2', 
        P10=30.0, 
        Q10=10.0, 
        Exp_v_p=2.0, 
        Exp_v_q=2.0),
    Defk(step=(0, 1, 2), id=('kp', 'kq')),
    Link(step=(0, 1, 2), objid='consumer_0', part='pq', id=('kp', 'kq'))])
"""
node: 0               1               2               3
                                       
      |      line     |     line      |     line      |
      +-----=====-----+-----=====-----+-----=====-----+
      |               |               |               |
                                     \|/ consumer    \|/ generator
                                      '               '
"""
grid_c = (grid_a + [
    Branch(
        id='line_2',
        id_of_node_A='n_2', 
        id_of_node_B='n_3',
        y_mn=1e3-1e3j,
        y_mm_half=1e-6+1e-6j),
    Vvalue(
        id_of_node='n_3',
        V=0.98),
    Injection(
        id='generator_0', 
        id_of_node='n_3', 
        P10=-10.0, 
        Q10=-10.0, 
        Exp_v_p=0.0, 
        Exp_v_q=0.0),
    Defk(step=(1, 2), id='kq_generator_0'),
    Link(step=(1, 2), objid='generator_0', part='q', id='kq_generator_0')])

# debugging
kstr, injk_str = debug.show_factors(grid_c, 3)
tap_positions = [('taps_0', 0)]
# define objectives and constraints
parameters_of_steps =[
    {'objectives': ('P', 'Q', Term(type='V', a='n_3'))}
    #{'objectives': Term(type='V', a='n_3'),  'constraints': 'PQ'}
    ]
# run estimation steps
model = get_model(grid_c)
estim_results = list(       
    estimation_steps(model, parameters_of_steps, tap_positions))
# print the final result
pr.print_estim_result(estim_results)
pr.print_measurements(estim_results)


