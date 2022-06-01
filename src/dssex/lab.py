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
get_injected_power = get_injected_power_fn(model.injections)
pq_factors = np.ones((len(model.injections), 2))
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

from pfcsymb import build_residual_fn, build_injected_current_fn
from util import get_injected_current_per_node
import casadi

#path = r"C:\UserData\deb00ap2\OneDrive - Siemens AG\Documents\defects\SP7-219086\eus1_loop\eus1_loop.db"
#path = r"C:\UserData\deb00ap2\OneDrive - Siemens AG\Documents\defects\SP7-219086\eus1_loop"
#path = r"C:\Users\live\OneDrive\Dokumente\py_projects\data\eus1_loop.db"
#path = r"K:\Siemens\Power\Temp\DSSE\Subsystem_142423"


fv = .8 # voltage
fl = 2 # loads
loadcurve = 'sq' # 'original' | 'interpolated' | 'square'
powerflowfn = cpfsymb # cpfnum | cpfsymb

path = r"D:\eus1_loop"
frames = egrid_frames(_Y_LO_ABS_MAX, path)
model = model_from_frames(frames)
pq_factors = fl * np.ones((len(model.injections), 2))
calc_Inode_inj = build_injected_current_fn(model, pq_factors, loadcurve)


fn_Iresidual = build_residual_fn(model, pq_factors, loadcurve)


#
# check current calculaiton function
#

fVprobe = 1.
Vslack = fv * model.slacks.V
values_of_params = casadi.horzcat(
    np.real(Vslack), np.imag(Vslack), model.branchtaps.position.copy())
Vprobe = fVprobe * (
        np.array([1.+0j]*model.shape_of_Y[0], dtype=np.complex128))
# symb
Vprobe_ri = np.concatenate([np.real(Vprobe), np.imag(Vprobe)]).reshape(-1)
Inode_inj_ri = np.array(calc_Inode_inj(Vprobe_ri))
Inode_inj = np.hstack(np.split(Inode_inj_ri, 2)).view(dtype=np.complex128)
print('\nInode_inj:\n', Inode_inj)
# num
get_injected_power = get_injected_power_fn(
    model.injections, pq_factors=pq_factors, loadcurve=loadcurve)
# util
Inode_inj2 = get_injected_current_per_node(get_injected_power, model, Vprobe.reshape(-1, 1))
print('\nInode_inj2:\n', Inode_inj2)

#%%
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
    Vri = np.concatenate([np.real(V), np.imag(V)]).reshape(-1)
    res = fn_Iresidual(Vri, values_of_params)
    Ires_symb = np.hstack(np.split(np.array(res), 2)).view(dtype=np.complex128)
    print('Ires_symb:\n', Ires_symb, '\n')
    Ires_symb_max = np.linalg.norm(
        np.hstack([np.real(Ires_symb[1:]), np.imag(Ires_symb[1:])])
        .reshape(-1), 
        np.inf)
    print('Ires_symb_max: ', Ires_symb_max, '\n')
    Ires = get_residual_current_fn(
        model, 
        get_injected_power)(V).reshape(-1, 1)
    print('Ires:\n', Ires, '\n')
    Ires_max = np.linalg.norm(
        np.hstack([np.real(Ires[1:]), np.imag(Ires[1:])]).reshape(-1), 
        np.inf)
    print('Ires_max: ', Ires_max, '\n')
    print('V:\n', V, '\n')
    # res = get_results(model, get_injected_power, model.branchtaps.position, V)  
    # names = frames['Names']
    # result_inj = decorate_injection_results(names, res['injections'])
    # print('Injections:\n', result_inj, '\n')
    # result_br = decorate_branch_results(names, res['branches'])
    # print('Branches:\n', result_br, '\n')
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
