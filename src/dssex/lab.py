# -*- coding: utf-8 -*-
"""
Created on Wed May  4 14:16:30 2022

@author: pyprg
"""
import numpy as np
import pandas as pd
from egrid import make_model
from egrid.builder import (
    Slacknode, Branch, Injection, PValue, QValue, Output, Vvalue, Defk, Link)
from pfcsymb import calculate_power_flow as cpfsymb
from pfcsymb import eval_residual_current as eval_symb
from pfcnum import calculate_power_flow as cpfnum
from pfcnum import get_injected_power_fn
from src.dssex.util import \
    get_results, get_residual_current_fn, get_residual_current_fn2

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
        y_tr=1e-6+1e-6j
        ),
    Branch(
        id='line_1',
        id_of_node_A='n_1',
        id_of_node_B='n_2',
        y_lo=1e3-1e3j,
        y_tr=1e-6+1e-6j
        ),
    Injection(
        id='consumer_0',
        id_of_node='n_2',
        P10=30.0,
        Q10=10.0,
        Exp_v_p=1.0,
        Exp_v_q=2.0
        )]
model = make_model(model_devices)
get_injected_power = get_injected_power_fn(
    model.injections, 
    loadcurve='original')
pq_factors = np.ones((len(model.injections), 2))
if len(model.errormessages):
    print(model.errormessages)
else:
    success, V = cpfsymb(1e-10, 20, model)
    print('SUCCESS' if success else '_F_A_I_L_E_D_')
    Ires = get_residual_current_fn(model, get_injected_power)(V)
    print('\nIres:\n', Ires)
    print('\nV:\n', pd.DataFrame(V, columns=['V']))
    res = get_results(model, get_injected_power, model.branchtaps.position, V)  
    print('\ninjections:\n', res['injections'])
    print('\nbranches:\n', res['branches'])  
#%%
from dnadb import egrid_frames
from dnadb.ifegrid import decorate_injection_results, decorate_branch_results

from egrid import model_from_frames
from egrid.model import _Y_LO_ABS_MAX
import casadi
from util import eval_residual_current, max_ri

#path = r"C:\UserData\deb00ap2\OneDrive - Siemens AG\Documents\defects\SP7-219086\eus1_loop\eus1_loop.db"
#path = r"C:\UserData\deb00ap2\OneDrive - Siemens AG\Documents\defects\SP7-219086\eus1_loop"
#path = r"C:\Users\live\OneDrive\Dokumente\py_projects\data\eus1_loop.db"
#path = r"K:\Siemens\Power\Temp\DSSE\Subsystem_142423"


fv = 1. # voltage
fl = 1 # loads
loadcurve = 'interpolated' # 'original' | 'interpolated' | 'square'
powerflowfn = cpfsymb # cpfnum | cpfsymb

path = r"D:\eus1_loop"
frames = egrid_frames(_Y_LO_ABS_MAX, path)
model = model_from_frames(frames)
pq_factors = fl * np.ones((len(model.injections), 2))


Vslack = fv * model.slacks.V
values_of_params = casadi.horzcat(
    np.real(Vslack), np.imag(Vslack), model.branchtaps.position.copy())
if len(model.errormessages):
    print(model.errormessages)
else:
    Vnode_initial = fv * (
        np.array([1.+0j]*model.shape_of_Y[0], dtype=np.complex128)
        .reshape(-1, 1))
    success, V = powerflowfn(
        1e-10, 
        20, 
        model, 
        Vslack=Vslack, 
        Vinit=Vnode_initial, 
        pq_factors=pq_factors,
        loadcurve=loadcurve)
    print()
    print('SUCCESS' if success else '_F_A_I_L_E_D_', '\n')
    Ires_symb = eval_symb(
        model, pq_factors=pq_factors, loadcurve=loadcurve, V=V)
    Ires_symb_max = max_ri(Ires_symb[1:])
    print('Ires_symb_max: ', Ires_symb_max, '\n')
    get_injected_power = get_injected_power_fn(
        model.injections, pq_factors=pq_factors, loadcurve=loadcurve)
    Ires_max = max_ri(
        eval_residual_current(model, get_injected_power, V=V)[1:])
    print('Ires_max: ', Ires_max, '\n')
    print('V:\n', V, '\n')
    res = get_results(model, get_injected_power, model.branchtaps.position, V)  
    names = frames['Names']
    result_inj = decorate_injection_results(names, res['injections'])
    print('Injections:\n', result_inj, '\n')
    result_br = decorate_branch_results(names, res['branches'])
    print('Branches:\n', result_br, '\n')
#%%
# Always use a decimal point for floats. Now and then processing ints
# fails with casadi/pandas/numpy.

# node: 0               1
#
#       |     line_0
#       +-----=====-----+
#       |               |
#                      \|/ consumer
#                       '
model_devices2 = [
    Slacknode(id_of_node='n_0', V=1.+0.j),
    Branch(
        id='line_0',
        id_of_node_A='n_0',
        id_of_node_B='n_1',
        y_lo=1e3-1e3j,
        y_tr=1e-6+1e-6j
        ),
    Injection(
        id='consumer_0',
        id_of_node='n_1',
        P10=30.0,
        Q10=10.0,
        Exp_v_p=1.0,
        Exp_v_q=2.0
        )]
model2 = make_model(
    model_devices2,
    Vvalue(id_of_node='n_1', V=1.,),
    Defk(step=0, id='kq'),
    Link(step=0, objid='Gen', part='q', id='kq'))
if len(model2.errormessages):
    print(model2.errormessages)
else:
    success2, V2 = cpfsymb(1e-10, 20, model2)
    print('\n--->>>','SUCCESS' if success2 else '_F_A_I_L_E_D_', '<<<---\n')
    print('V:\n', V2, '\n')
    get_injected_power2 = get_injected_power_fn(model2.injections)
    res2 = get_results(model2, get_injected_power2, model2.branchtaps.position, V2)
    print('\ninjections:\n', res2['injections'])
    print('\branches:\n', res2['branches'])
