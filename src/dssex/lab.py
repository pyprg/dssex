# -*- coding: utf-8 -*-
"""
Created on Wed May  4 14:16:30 2022

@author: pyprg
"""
import casadi
import numpy as np
from operator import itemgetter
from egrid import make_model
from egrid.builder import Slacknode, Branch, Injection, Branchtaps
from src.dssex.pfc import get_polynomial_coefficients, create_branch_gb_matrix

# square of voltage magnitude, minimum value for load curve, 
#   if value is below _VMINSQR the load curves for P and Q converge
#   towards a linear load curve which is 0 when V=0; P(V=0)=0, Q(V=0)=0
_VMINSQR = 0.8
# value of zero check, used for load curve calculation    
_EPSILON = 1e-12

def _add_interpol_coeff_to_injections(injections, vminsqr):
    """Adds polynomial coefficients to injections for linear interpolation
    of power around |V|² ~ 0.
    
    Parameters
    ----------
    injections: pandas.DataFrame
        * .Exp_v_p
        * .Exp_v_q
    vminsqr: float
        upper limit of interpolation
        
    Returns
    -------
    pandas.DataFrame
        modified injections"""
    p_coeffs = get_polynomial_coefficients(vminsqr, injections.Exp_v_p)
    injections['c3p'] = p_coeffs[:, 0]
    injections['c2p'] = p_coeffs[:, 1]
    injections['c1p'] = p_coeffs[:, 2]
    q_coeffs = get_polynomial_coefficients(vminsqr, injections.Exp_v_q)
    injections['c3q'] = q_coeffs[:, 0]
    injections['c2q'] = q_coeffs[:, 1]
    injections['c1q'] = q_coeffs[:, 2]
    return injections

_current_props = itemgetter(
    'P10', 'Q10', 'Exp_v_p', 'Exp_v_q',
    'c3p', 'c2p', 'c1p', 'c3q', 'c2q', 'c1q')

def _injected_current(injections, vminsqr, Vre, Vim, V_abs_sqr):
    """Calculates current flowing into an injection. Returns separate
    real and imaginary parts.
    Injected power is calculated this way
    (P = |V|**Exvp * P10, Q = |V|**Exvq * Q10; with |V| - magnitude of V):
    ::
        +- -+   +-                                           -+
        | P |   | (Vre ** 2 + Vim ** 2) ** (Expvp / 2) * P_10 |
        |   | = |                                             |
        | Q |   | (Vre ** 2 + Vim ** 2) ** (Expvq / 2) * Q_10 |
        +- -+   +-                                           -+

    How to calculate current from given complex power and from voltage:
    ::
                S_conjugate
        I_inj = -----------
                V_conjugate

    How to calculate current with separated real and imaginary parts:
    ::
                              +-        -+ -1  +-    -+
                S_conjugate   |  Vre Vim |     |  P Q |
        I_inj = ----------- = |          |  *  |      |
                V_conjugate   | -Vim Vre |     | -Q P |
                              +-        -+     +-    -+

                                  +-        -+   +-    -+
                       1          | Vre -Vim |   |  P Q |
              = --------------- * |          | * |      |
                Vre**2 + Vim**2   | Vim  Vre |   | -Q P |
                                  +-        -+   +-    -+

                                   +-              -+
                          1        |  P Vre + Q Vim |
        I_inj_ri = --------------- |                |
                   Vre**2 + Vim**2 | -Q Vre + P Vim |
                                   +-              -+

    How to calculate injected real and imaginary current from voltage:
    ::
        Ire =  (Vre ** 2 + Vim ** 2) ** (Expvp / 2 - 1) * P_10 * Vre
             + (Vre ** 2 + Vim ** 2) ** (Expvq / 2 - 1) * Q_10 * Vim

        Iim = -(Vre ** 2 + Vim ** 2) ** (Expvq / 2 - 1) * Q_10 * Vre
             + (Vre ** 2 + Vim ** 2) ** (Expvp / 2 - 1) * P_10 * Vim

    Parameters
    ----------
    injections: pandas.DataFrame
        * .V_abs_sqr, float, Vre**2 + Vim**2
        * .Vre, float, node voltage, real part
        * .Vim:, float, node voltage, imaginary part
        * .Exp_v_p, float, voltage exponent, active power
        * .Exp_v_q, float, voltage exponent, active power
        * .P10, float, active power at voltage 1 per unit
        * .Q10, float, reactive power at voltage 1 per unit
        * .c3p, .c2p, .c1p 
        * .c3q, .c2q, .c1q 
    vminsqr: float
        upper limit of interpolation, interpolates if |V|² < ul
    Vre: casadi.SX
        real part of node voltage, vector
    Vim: casadi.SX
        imaginary part of node voltage, vector
    V_abs_sqr: casadi.SX
        square of node voltage magnitude, vector


    Returns
    -------
    tuple
        * Ire, real part of injected current
        * Iim, imaginary part of injected current"""
    if injections.size:
        (P10, Q10, Exp_v_p, Exp_v_q, 
         c3p, c2p, c1p, c3q, c2q, c1q) = map(
             casadi.vcat, _current_props(injections))
        # interpolated function
        V_abs = casadi.power(V_abs_sqr, .5)
        V_abs_cub = V_abs_sqr * V_abs
        p_expr = (c3p*V_abs_cub + c2p*V_abs_sqr + c1p*V_abs) * P10
        q_expr = (c3q*V_abs_cub + c2q*V_abs_sqr + c1q*V_abs) * Q10
        Ire_ip = casadi.if_else(
            _EPSILON < V_abs_sqr,  
            (p_expr * Vre + q_expr * Vim) / V_abs_sqr, 
            0.0)
        Iim_ip = casadi.if_else(
            _EPSILON < V_abs_sqr, 
            (-q_expr * Vre + p_expr * Vim) / V_abs_sqr, 
            0.0)
        # original function
        y_p = casadi.power(V_abs_sqr, Exp_v_p/2 -1) * P10 
        y_q = casadi.power(V_abs_sqr, Exp_v_q/2 -1) * Q10
        Ire =  y_p * Vre + y_q * Vim
        Iim = -y_q * Vre + y_p * Vim
        # compose load functions from original and interpolated
        interpolate = V_abs_sqr < vminsqr
        return (
            casadi.if_else(interpolate, Ire_ip, Ire), 
            casadi.if_else(interpolate, Iim_ip, Iim))
    return casadi.DM(0.), casadi.DM(0.)

def calculate(model, parameters_of_steps=(), tap_positions=()):
    # gb-matrix
    branchdata = create_branch_gb_matrix(model)
    # V-vector
    count_of_nodes = branchdata.count_of_nodes
    Vre = casadi.SX.sym('Vre', count_of_nodes)
    Vim = casadi.SX.sym('Vim', count_of_nodes)
    V = casadi.vertcat(Vre, Vim)
    # I-vector
    V_abs_sqr = Vre.constpow(2) + Vim.constpow(2)
    injections = _add_interpol_coeff_to_injections(
        model.injections.copy(), _VMINSQR)
    index_of_node = injections.index_of_node
    Iinj_re, Iinj_im = _injected_current(
        injections, _VMINSQR, 
        Vre[index_of_node], Vim[index_of_node], V_abs_sqr[index_of_node])
    Inode_re = casadi.SX(count_of_nodes, 1)
    Inode_im = casadi.SX(count_of_nodes, 1)
    for index_of_injection, inj in injections.iterrows():
        index_of_node = inj.index_of_node
        Inode_re[index_of_node] += Iinj_re[index_of_injection]
        Inode_im[index_of_node] += Iinj_im[index_of_injection]
    I = casadi.vertcat(Inode_re, Inode_im)
    # equation
    Expr = (branchdata.GB @ V) - I
    return Expr, Vre, Vim
#%%
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
model00 = make_model(
    model_devices,
    Branchtaps(
        id='tap_1',
        id_of_branch='nix_1',
        id_of_node='n_0'),
    Branchtaps(
        id='tapLine_1',
        id_of_branch='line_1',
        id_of_node='n_1'),
    Branchtaps(
        id='tapLine_0',
        id_of_branch='line_0',
        id_of_node='n_1')
    )
# calculate power flow
results = calculate(model00)
#%%
# syms
g = casadi.SX(3, 3)
b = casadi.SX(3, 3)
# vals
y_mn_0 = 1e3-1e3j
y_mm_half_0 = 1e-6+1e-6j
y_mn_1 = 1e3-1e3j
y_mm_half_1 = 1e-6+1e-6j
# line_0
g_mn_0 = y_mn_0.real
b_mn_0 = y_mn_0.imag
g_mm_half_0 = y_mm_half_0.real
b_mm_half_0 = y_mm_half_0.imag
# line_1
g_mn_1 = y_mn_1.real
b_mn_1 = y_mn_1.imag
g_mm_half_1 = y_mm_half_1.real
b_mm_half_1 = y_mm_half_1.imag
# slack
g[0, 0] = 1.
# other
g[1, 0] = -g_mn_0
g[1, 1] = g_mn_0 + g_mm_half_0 + g_mn_1 + g_mm_half_1
g[1, 2] = -g_mn_1
g[2, 1] = -g_mn_1
g[2, 2] = g_mn_1 + g_mm_half_1
b[1, 0] = -b_mn_0
b[1, 1] = b_mn_0 + b_mm_half_0 + b_mn_1 + b_mm_half_1
b[1, 2] = -b_mn_1
b[2, 1] = -b_mn_1
b[2, 2] = b_mn_1 + b_mm_half_1
#
gb = casadi.blockcat([[g, -b], [b,  g]])
count_of_nodes = 3
Vre = casadi.SX.sym('Vre', count_of_nodes)
Vim = casadi.SX.sym('Vim', count_of_nodes)
Vreim = casadi.vertcat(Vre, Vim)
# injections
count_of_injections = 2
# syms
P10inj = casadi.SX.sym('P10', count_of_injections)
Q10inj = casadi.SX.sym('Q10', count_of_injections)
exp_v_p = casadi.SX.sym('exp_v_p', count_of_injections)
exp_v_q = casadi.SX.sym('exp_v_q', count_of_injections)
# vals
P10inj[0] = 30
Q10inj[0] = 10
exp_v_p[0] = 2
exp_v_q[0] = 2
P10inj[1] = 30
Q10inj[1] = 10
exp_v_p[1] = 2
exp_v_q[1] = 2
# node <= injection
Mnodeinj = casadi.SX(count_of_nodes, count_of_injections)
Mnodeinj[2, 0] = 1
Mnodeinj[2, 1] = 1
# Inode
# Ire =  (Vre ** 2 + Vim ** 2) ** (Expvp / 2 - 1) * P_10 * Vre
#      + (Vre ** 2 + Vim ** 2) ** (Expvq / 2 - 1) * Q_10 * Vim
# Iim = -(Vre ** 2 + Vim ** 2) ** (Expvq / 2 - 1) * Q_10 * Vre
#      + (Vre ** 2 + Vim ** 2) ** (Expvp / 2 - 1) * P_10 * Vim

Vnode_sqr = casadi.power(Vre, 2) + casadi.power(Vim, 2)
Vinj_sqr = casadi.transpose(Mnodeinj) @ Vnode_sqr
Gexpr_node = Mnodeinj @ (casadi.power(Vinj_sqr, exp_v_p/2 - 1) * P10inj)
Bexpr_node = Mnodeinj @ (casadi.power(Vinj_sqr, exp_v_q/2 - 1) * Q10inj)
Ire_node =  Gexpr_node * Vre + Bexpr_node * Vim
Iim_node = -Bexpr_node * Vre + Gexpr_node * Vim

# slacks
Ire_node[0] = -1. # -Vre slack
Iim_node[0] = 0.  # -Vim slack
# vector
Inode = casadi.vertcat(Ire_node, Iim_node)
# equation for root finding
Ires = (gb @ Vreim) + Inode
#%%
# solve root-finding equation
fn_Iresidual = casadi.Function('fn_Iresidual', [Vreim], [Ires])
rf = casadi.rootfinder('rf', 'nlpsol', fn_Iresidual, {'nlpsol':'ipopt'})
voltages = rf([1., 1., 1., 0., 0., 0.])
Vcalc = np.array(
    casadi.hcat(
        casadi.vertsplit(
            voltages, [0, voltages.size1()//2, voltages.size1()])))
Vcomp = Vcalc.view(dtype=np.complex128)
print()
print('V: ', Vcomp)
