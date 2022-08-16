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
from src.dssex.util import get_results, get_residual_current_fn

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
count_of_nodes = model.shape_of_Y[0]
init = np.array([1.]*count_of_nodes + [0.]*count_of_nodes).reshape(-1, 1)
if len(model.errormessages):
    print(model.errormessages)
else:
    success, V = cpfsymb(1e-10, 20, model, Vinit=init)
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
    count_of_nodes = model.shape_of_Y[0]
    init = np.array([1.]*count_of_nodes + [0.]*count_of_nodes).reshape(-1, 1)
    success, V = powerflowfn(
        1e-10, 
        20, 
        model, 
        Vslack=Vslack, 
        Vinit=init, 
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
    count_of_nodes = model2.shape_of_Y[0]
    init = np.array([1.]*count_of_nodes + [0.]*count_of_nodes).reshape(-1, 1)
    success2, V2 = cpfsymb(1e-10, 20, model2, Vinit=init)
    print('\n--->>>','SUCCESS' if success2 else '_F_A_I_L_E_D_', '<<<---\n')
    print('V:\n', V2, '\n')
    get_injected_power2 = get_injected_power_fn(model2.injections)
    res2 = get_results(
        model2, get_injected_power2, model2.branchtaps.position, V2)
    print('\ninjections:\n', res2['injections'])
    print('\branches:\n', res2['branches'])
#%% root finding with casadi
from pfcsymb import create_vars, create_gb_matrix, get_injected_current, find_root
V, Vslack, pos = create_vars(model)
# branch gb-matrix, g:conductance, b:susceptance
gb = create_gb_matrix(model, pos)
# injected current
injections = model.injections
Inode_re, Inode_im = get_injected_current(
    model.mnodeinj, V, injections[~injections.is_slack], 
    pq_factors, loadcurve)
# modify Inode of slacks
index_of_slack = model.slacks.index_of_node
Inode_re[index_of_slack] = -Vslack.re
Inode_im[index_of_slack] = -Vslack.im
# equation of node current
Ires = (gb @ V.reim) + casadi.vertcat(Inode_re, Inode_im)
# parameters, vertcat returns wrong shape for count_of_branch_taps==0
param = (casadi.vertcat(Vslack.reim, pos) if pos.size1() else Vslack.reim)
# create node current function
fn_Iresidual = casadi.Function('fn_Iresidual', [V.reim, param], [Ires])
# calculate
Vslack_ = model.slacks.V
tappositions_ = model.branchtaps.position.copy()
count_of_nodes = model.shape_of_Y[0]
init = np.array([1.]*count_of_nodes + [0.]*count_of_nodes).reshape(-1, 1)
success, voltages = find_root(fn_Iresidual, tappositions_, Vslack_, init)
Vcomp = np.hstack(np.vsplit(voltages, 2)).view(dtype=np.complex128)
#%% calculate voltage 'manually'
# node: 0               1
#
#       |     line_0
#       +-----=====-----+
#       |               |
#                      -+- 
#                      --- shunt
g0 = 1e3
b0 = -1e3
y0 = complex(g0, b0)
b1 = 1e2
y1 = complex(0., b1)
V0 = 1.+.0j
# helper
I = V0 / (1/y0 + 1/y1)
V1 = I / y1
S1 = V1 * I.conjugate()
Q1 = S1.imag
V0r = V0.real
V0j = V0.imag
V1r = V1.real
V1j = V1.imag
V01 = I / y0
f = Q1 / (g0**2 + b0**2)
# real
V1sqr = abs(V1 * V1.conjugate())
Null_r = V0r*V1r + V0j*V1j + b0 * f - V1sqr
print('Null_r: ', Null_r)
# imaginary
Null_j = V0j*V1r - V0r*V1j + g0 * f
print('Null_j: ', Null_j)
#%% find V1.real, V1.imag with given V0, Q
V1ri = casadi.SX.sym('V1ri', 2, 1)
res_ = casadi.SX(2, 1)
res_[0, 0] = V0r*V1ri[0] + V0j*V1ri[1] + b0 * f - V1sqr
res_[1, 0] = V0j*V1ri[0] - V0r*V1ri[1] + g0 * f
fn_res_ = casadi.Function('fn_Iresidual', [V1ri], [res_])
rf_ = casadi.rootfinder('rf_', 'newton', fn_res_)
result = rf_(casadi.vertcat(1., 0.))
print('\nresult: ', result, '\n')
#%% find V1.real, V1.imag, Q with given V0, abs(V1)
V1ri = casadi.SX.sym('V1ri', 2)
Q1_ = casadi.SX.sym('Q1')
f = Q1_ /(g0**2 + b0**2)
Q_ = casadi.SX.sym('Q')
VV1sqr = casadi.SX(2,1)
VV1sqr[0, 0] = V1sqr
res_ = casadi.SX(4, 1)
res_[0, 0] = V0r*V1ri[0] + V0j*V1ri[1] + b0*f - V1sqr
res_[1, 0] = V0j*V1ri[0] - V0r*V1ri[1] + g0*f
res_[2, 0] = casadi.sumsqr(V1ri) - V1sqr
res_[3, 0] = Q_ - Q1_
fn_res_ = casadi.Function('fn_Iresidual', [casadi.vertcat(V1ri, Q_, Q1_)], [res_])
rf_ = casadi.rootfinder('rf_', 'newton', fn_res_)
result = rf_(casadi.vertcat(1., 0., 0., 0.))
print('\nresult: ', result, '\n')
#%%
# node: 0               1               2
#
#       |     line_0          line_1
#       +-----=====-----+-----=====-----+
#       |               |               |
#                      ---             \|/ consumer
#                      --- shunt        '
g0 = 1e3
b0 = -1e3
y0 = complex(g0, b0)
b1 = 1e2
y1 = complex(0., b1)
V0 = 1.+.0j
# helper
I = V0 / (1/y0 + 1/y1)
V1 = I / y1
S1 = V1 * I.conjugate()
Q1 = S1.imag
V0r = V0.real
V0j = V0.imag
V1r = V1.real
V1j = V1.imag
V01 = I / y0
f = Q1 / (g0**2 + b0**2)
# real
V1sqr = abs(V1 * V1.conjugate())
Null_r = V0r*V1r + V0j*V1j + b0 * f - V1sqr
print('Null_r: ', Null_r)
# imaginary
Null_j = V0j*V1r - V0r*V1j + g0 * f
print('Null_j: ', Null_j)
#%% model with generator, search for V
# Always use a decimal point for floats. Now and then processing ints
# fails with casadi/pandas/numpy.

# node: 0               1               2
#
#       |     line_0    |     line_1    |
#       +-----=====-----+-----=====-----+
#       |               |               |
#                      (~) generator   \|/ consumer
#                                       '
model_devices3 = [
    Slacknode(id_of_node='n_0', V=1.+0.j),
    Branch(
        id='line_0',
        id_of_node_A='n_0',
        id_of_node_B='n_1',
        y_lo=1e3-1e3j,
        y_tr=1e-6+1e-6j),
    Injection(
        id='generator',
        id_of_node='n_1',
        Q10=10.0,
        Exp_v_q=2.0),
    Branch(
        id='line_1',
        id_of_node_A='n_1',
        id_of_node_B='n_2',
        y_lo=1e3-1e3j,
        y_tr=1e-6+1e-6j),
    Injection(
        id='consumer',
        id_of_node='n_2',
        P10=30.0,
        Q10=10.0,
        Exp_v_p=2.0,
        Exp_v_q=2.0)]
model3 = make_model(
    model_devices3,
    Vvalue(id_of_node='n_1', V=1.0),
    # define a scaling factor
    Defk(id='kq_generator'),
    # link the factor to the generator
    Link(objid='generator', part='q', id='kq_generator'))
injections = model3.injections
pq_factors = fl * np.ones((len(injections), 2))
V, Vslack, pos = create_vars(model3)
# branch gb-matrix, g:conductance, b:susceptance
gb = create_gb_matrix(model3, pos)
# injected current
Inode_re, Inode_im = get_injected_current(
    model3.mnodeinj, V, injections[~injections.is_slack], 
    pq_factors, loadcurve)
# modify Inode of slacks
index_of_slack = model3.slacks.index_of_node
Inode_re[index_of_slack] = -Vslack.re
Inode_im[index_of_slack] = -Vslack.im
# equation of node current
Ires = (gb @ V.reim) + casadi.vertcat(Inode_re, Inode_im)
# parameters, vertcat returns wrong shape for count_of_branch_taps==0
param = (casadi.vertcat(Vslack.reim, pos) if pos.size1() else Vslack.reim)
# create node current function
fn_Iresidual = casadi.Function('fn_Iresidual', [V.reim, param], [Ires])
# calculate
Vslack_ = model3.slacks.V
tappositions_ = model3.branchtaps.position.copy() 
count_of_nodes = model3.shape_of_Y[0]
init = np.array([1.]*count_of_nodes + [0.]*count_of_nodes).reshape(-1, 1)
success, voltages = find_root(
    fn_Iresidual, tappositions_, Vslack_, init)
Vcomp = np.hstack(np.vsplit(voltages, 2)).view(dtype=np.complex128)
#%% search for V and Q
def get_v_setpoints(model):
    """Extracts voltage setpoints having a variable Q at their 
    power flow calculation nodes.
    
    Parmeters
    ---------
    model: egrid.model.Model
    
    Returns
    -------
    pandas.DataFrame (id_of_source)
        * .value, initial value
        * .min, float
        * .max, float
        * .injid, str, identifier of injection
        * .id_of_node, str
        * .index_of_node, int
        * .V, float, setpoint"""
    factors_step0 = (
        model.load_scaling_factors.droplevel('id').filter(items=[0], axis=0))
    var_step0 = factors_step0[factors_step0.type=='var']
    inj_to_factor = (
        model.injection_factor_associations.reset_index().set_index('id'))
    inj_to_qfactor = (
        inj_to_factor[(inj_to_factor.step==0) & (inj_to_factor.part=='q')])
    injs = (
        model3.injections
        .reset_index()
        .rename(columns={'index': 'index_of_injection'})
        .loc[:,['id', 'index_of_injection', 'id_of_node', 'index_of_node']]
        .set_index('id'))
    qvars_ = (
        var_step0
        .join(inj_to_qfactor, on='id_of_source')
        .drop(columns=['type', 'step', 'part'])
        .set_index('id_of_source'))
    qvars = qvars_.join(injs, on='injid', how='inner')
    vsetpoints = (
        model.vvalues[~model.vvalues.is_slack][['id_of_node', 'V']]
        .set_index('id_of_node'))
    return qvars.join(vsetpoints, on='id_of_node', how='inner')

vsetpoints = get_v_setpoints(model3)
injections = model3.injections
f_pq = casadi.SX.ones(len(injections), 2)
kq = casadi.SX.sym('kq', len(vsetpoints))
f_pq[vsetpoints.index_of_injection, 1] = kq
Inode_re, Inode_im = get_injected_current(
    model3.mnodeinj, V, injections[~injections.is_slack], 
    f_pq, loadcurve)
# modify Inode of slacks
index_of_slack = model3.slacks.index_of_node
Inode_re[index_of_slack] = -Vslack.re
Inode_im[index_of_slack] = -Vslack.im
# equation of node current
Ires = (gb @ V.reim) + casadi.vertcat(Inode_re, Inode_im)
# difference of voltage to setpoint
diff_v = V.sqr[vsetpoints.index_of_node] - np.power(vsetpoints.V, 2)
# residuum
res_expr = casadi.vertcat(Ires, diff_v) if diff_v.size1() else Ires
# decision variables
vars_ = casadi.vertcat(V.reim, kq) if kq.size1() else V.reim
# create residual function
fn_residual = casadi.Function('fn_residual', [vars_, param], [res_expr])
#
count_of_nodes = model3.shape_of_Y[0]
Vinit = np.array([1.]*count_of_nodes + [0.]*count_of_nodes).reshape(-1, 1)
init = casadi.vertcat(Vinit, vsetpoints.V) if len(vsetpoints) else Vinit
tappositions = model3.branchtaps.position.copy()
success, vals_ = find_root(fn_residual, tappositions, Vslack_, init)
# process results
vals = np.array(vals_)
voltages = (
    np.hstack(np.split(vals[:(2*count_of_nodes)],2)).view(dtype=np.complex128))
qfactors = vals[(2*count_of_nodes):]
#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import numpy as np

mpl.rcParams['figure.figsize']  = 18, 18
mpl.rcParams['legend.fontsize'] = 12

Vsetpoint = 1

# smallest possible reactive power
q_min = -10 
# greatest possible reactive power
q_max = 5   
vdiff_ = -0.2
# half of q-residuum function exponent 
n_half = 10


def res_vdiff(v_diff):
    """Calculates residuum for voltage part.
    
    Parameters
    ==========
    v_diff: float
        setpoint - value
    
    Returns
    =======
    float"""
    return 3*v_diff

def get_res_q(q_min, q_max, res_qmin, n_half):
    """Returns a function calculating the residuum of the q-part.
    Calculation of coefficient c:
    ::        
        target: 
          c for: res_q(q_min) = ((q_min-q_centre)*q_range_size)/(2*c) ** (2*n)
        solution:
          res(q_min, vdiff_) = 0
          res_vdiff(vdiff_) + res_q(q_min) = 0
          res_vdiff(vdiff_) + (((q_min-q_centre)*q_range_size)/(2*c))**(2*n) = 0
        
          ((q_min-q_centre)*q_range_size)/(2*c)**(2*n) = 
              -res_vdiff(vdiff_)
          ((q_min-q_centre)*q_range_size)/(2*c) = 
              (-res_vdiff(vdiff_))**(1/(2*n))
        
          c = ((q_min-q_centre)*q_range_size) / 2*((-res_vdiff(vdiff_))**(1/(2*n)))
          c = q_min / (-res_v(vdiff_))**(1/(2*n))
        
    
    Parameters
    ==========
    q_min: float
        smallest possible reactive power
    q_max: float
        greates possible reactive power
    res_qmin: float
        value of function at q_min (and q_max)
    n_half: int
        half exponent of residuum function
    
    Returns
    =======
    function (float)->(float)
        (reactive_power_Q)->(residuum)"""
    q_centre = (q_min + q_max) / 2
    q_range_size = q_max - q_min
    assert 0 <= res_qmin
    c = ((q_min - q_centre)*(q_range_size/2)) / (res_qmin**(1/(2*n_half)))
    return lambda q: ( ((q - q_centre)*(q_range_size/2))/c )**( 2*n_half )


res_q = get_res_q(q_min, q_max, -res_vdiff(vdiff_), n_half)

fn_res = lambda q, vdiff: res_q(q) + res_vdiff(vdiff)




q_text = 'Q'
vdiff_text = 'Vdiff'
residuum_text = 'residuum'

volt_padding = .2
volt_limit = volt_padding + Vsetpoint
v_limits = [-volt_limit, volt_limit]
q_padding = 0.2
q_limits = [-q_padding + q_min, q_max + q_padding]
residuum_limits = [-0.2, 1.] # residuum-axis


# Make data.
Vdiff_range = np.linspace(
    start=-volt_limit, stop=volt_limit, num=1000, endpoint=True)
Qrange = np.linspace(start=q_limits[0], stop=q_limits[1], num=1000, endpoint=True)
Vdiff, Q = np.meshgrid(Vdiff_range, Qrange)

vdiff_res = res_vdiff(Vdiff)
q_res = res_q(Q)

res = fn_res(Q, Vdiff)

levels = np.linspace(-1, 1, 3, endpoint=True)

fig = plt.figure()



elev = 50
azim = -40
alpha=0.3
color = cm.cool
norm = colors.CenteredNorm(vcenter=0)

def plot_vdiff_q_res_contour(ax):
    ax.set_title('Residuum contour over Q and Vdiff')
    ax.grid()
    ax.set_ylim(q_limits)
    ctf = ax.contourf(Vdiff, Q, res, levels=100, norm=norm, cmap=color, alpha=alpha, antialiased=True)
    cs = ax.contour(Vdiff, Q, res, levels=levels, norm=norm,  cmap=color, antialiased=True, linewidths=2)
    cbar = fig.colorbar(ctf, ticks=ticker.AutoLocator())
    ax.clabel(cs, inline=True, fontsize=15)
    ax.set_xlabel(vdiff_text, fontsize=10)
    ax.set_ylabel(q_text, fontsize=10)

def plot_vdiff_res(ax):
    ax.set_title('Vdiff-Residuum vs. Vdiff')
    ax.grid()
    ax.plot(Vdiff_range, res_vdiff(Vdiff_range))
    ax.set_xlabel(vdiff_text, fontsize=10)
    ax.set_ylabel(residuum_text, fontsize=10)

def plot_q_res(ax):
    ax.set_title('Q-Residuum vs. Q')
    ax.grid()
    ax.plot(Qrange, res_q(Qrange))
    ax.set_xlim(q_limits)
    ax.set_ylim(residuum_limits)
    ax.set_xlabel(q_text, fontsize=10)
    ax.set_ylabel(residuum_text, fontsize=10)

def plot3d_vdiff_res(ax):
    ax.set_title('Vdiff-Residuum vs. Q and Vdiff')
    ax.set_xlim(v_limits)
    ax.set_ylim(q_limits)
    ax.set_zlim(-4, 4.0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type('ortho')
    surf = ax.plot_surface(Vdiff, Q, vdiff_res, alpha=alpha, norm=norm, cmap=color, antialiased=True)
    cnt = ax.contour(Vdiff, Q, vdiff_res, levels=levels, norm=norm, cmap=color, antialiased=True)
    ax.set_xlabel(vdiff_text , fontsize=10)
    ax.set_ylabel(q_text, fontsize=10)
    ax.set_zlabel(residuum_text, fontsize=10)

def plot3d_q_res(ax):
    ax.set_title('Q-Residuum vs. Q and Vdiff')
    ax.set_xlim(v_limits)
    ax.set_ylim(q_limits)
    ax.set_zlim(-4, 4.0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type('ortho')
    surf = ax.plot_surface(Vdiff, Q, q_res, alpha=alpha, norm=norm, cmap=color, antialiased=True)
    cnt = ax.contour(Vdiff, Q, q_res, levels=levels, norm=norm, cmap=color, antialiased=True)
    ax.set_xlabel(vdiff_text , fontsize=10)
    ax.set_ylabel(q_text, fontsize=10)
    ax.set_zlabel(residuum_text, fontsize=10)

def plot3d_vdiff_q_res(ax):
    ax.set_title('Residuum vs. Q and Vdiff')
    ax.set_xlim(v_limits)
    ax.set_ylim(q_limits)
    ax.set_zlim(-4, 4.0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type('ortho')
    surf = ax.plot_surface(Vdiff, Q, res, alpha=alpha, norm=norm, cmap=color, antialiased=True)
    cnt = ax.contour(Vdiff, Q, res, levels=levels, norm=norm, cmap=color, antialiased=True)
    ax.set_xlabel(vdiff_text, fontsize=10)
    ax.set_ylabel(q_text, fontsize=10)
    ax.set_zlabel(residuum_text, fontsize=10)



gs = gridspec.GridSpec(5, 5, fig, wspace=0.5, hspace=0.5)
plot_vdiff_q_res_contour(fig.add_subplot(gs[0:3, 0:3]))
plot_q_res(fig.add_subplot(gs[0, 3:5]))
plot3d_q_res(fig.add_subplot(gs[1:3, 3:5], projection='3d'))
plot3d_vdiff_q_res(fig.add_subplot(gs[3:5, 3:5], projection='3d'))
plot_vdiff_res(fig.add_subplot(gs[3:5, 0]))
plot3d_vdiff_res(fig.add_subplot(gs[3:5, 1:3], projection='3d'))
