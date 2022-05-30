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
from pfcnum import calculate_power_flow as cpfnum
from pfcnum import get_injected_power
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
if len(model.errormessages):
    print(model.errormessages)
else:
    success, V = cpfnum(1e-10, 20, model)
    print('SUCCESS' if success else '_F_A_I_L_E_D_')
    Ires = get_residual_current_fn(model, get_injected_power)(V)
    print('Ires: ', Ires)
    print('V: ', pd.DataFrame(V, columns=['V']))
    res = get_results(model, get_injected_power, model.branchtaps.position, V)  
    print(res['injections'])
    print(res['branches'])  
#%%
from dnadb import egrid_frames
from dnadb.ifegrid import decorate_injection_results, decorate_branch_results
from egrid import model_from_frames
from egrid.model import _Y_LO_ABS_MAX

#path = r"C:\UserData\deb00ap2\OneDrive - Siemens AG\Documents\defects\SP7-219086\eus1_loop\eus1_loop.db"
#path = r"C:\UserData\deb00ap2\OneDrive - Siemens AG\Documents\defects\SP7-219086\eus1_loop"
#path = r"C:\Users\live\OneDrive\Dokumente\py_projects\data\eus1_loop.db"
#path = r"K:\Siemens\Power\Temp\DSSE\Subsystem_142423"
path = r"D:\eus1_loop"
frames = egrid_frames(_Y_LO_ABS_MAX, path)
model = model_from_frames(frames)
if len(model.errormessages):
    print(model.errormessages)
else:
    Vnode_initial = (
        np.array([1.+0j]*model.shape_of_Y[0], dtype=np.complex128)
        .reshape(-1, 1))
    success, V = cpfnum(1e-10, 20, model, Vinit=Vnode_initial)
    print('SUCCESS' if success else '_F_A_I_L_E_D_')
    Ires = get_residual_current_fn(model, get_injected_power)(V)
    print('Ires: ', Ires)
    names = frames['Names']
    print('V: ', V)
    res = get_results(model, get_injected_power, model.branchtaps.position, V)  
    result_inj = decorate_injection_results(names, res['injections'])
    print(result_inj)
    result_br = decorate_branch_results(names, res['branches'])
    print(result_br)
#%%
from scipy.optimize import root

Vnode_initial = (
    np.array([1.+0j]*model.shape_of_Y[0], dtype=np.complex128)
    .reshape(-1, 1))
fn_res = get_residual_current_fn(model, get_injected_power)
res = root(fn_res, x0=Vnode_initial, method='krylov')
Vnode_res = res.x
#%%
Ires = get_residual_current_fn(model, get_injected_power)(Vnode_res)
print('Ires: ', Ires)
#%%
fn_res2 = get_residual_current_fn2(model, get_injected_power)
res2 = root(fn_res2, x0=Vnode_initial[model.count_of_slacks:], method='broyden1')
Ires2 = get_residual_current_fn(model, get_injected_power)(np.vstack([model.slacks.V, res2.x]))
