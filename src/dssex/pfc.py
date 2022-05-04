# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:49:05 2022

@author: pyprg
"""
import casadi
import numpy as np
from numpy.linalg import solve
from egrid import make_model
from egrid.builder import Slacknode, Branch, Injection, Branchtaps
from collections import namedtuple
from operator import itemgetter
from functools import lru_cache
from scipy.sparse import coo_matrix

# square of voltage magnitude, minimum value for load curve, 
#   if value is below _VMINSQR the load curves for P and Q converge
#   towards a linear load curve which is 0 when V=0; P(V=0)=0, Q(V=0)=0
_VMINSQR = 0.8
# value of zero check, used for load curve calculation    
_EPSILON = 1e-12

Branchdata = namedtuple(
    'Branchdata',
    'pos branchterminals g_tot b_tot g_mn b_mn count_of_nodes GB')

@lru_cache(maxsize=200)
def get_coefficients(x, y, dydx_0, dydx):
    """Calculates the coefficients A, B, C of the polynomial
    ::
        f(x) = Ax³ + Bx² + Cx
        with
            f(0) = 0, 
            df(0)/dx = dydx_0,
            f(x) = y
            df(x)/dx = dydx
            
    Parameters
    ----------
    x: float
        x of point
    y: float
        value at x
    dydx_0: float 
        dy / dx at 0
    dydx: float 
        dy / dx at x

    Returns
    -------
    numpy.ndarray (shape=(3,))
        float, float, float (A, B, C)"""
    x_sqr = x * x
    x_cub = x * x_sqr
    cm = np.array([
        [   x_cub, x_sqr,  x],
        [      0.,    0., 1.],
        [3.*x_sqr,  2.*x, 1.]])
    return solve(cm, np.array([y, dydx_0, dydx]))

def calc_dpower_dvsqr(v_sqr, _2p):
    """Calculates the derivative of exponential power function at v_sqr.
    
    Parameters
    ----------
    v_sqr: float
        square of voltage magnitude
    _2p: float
        double of voltage exponent
    
    Returns
    -------
    float"""
    p = _2p/2
    return p * np.power(v_sqr, p-1)

def get_polynomial_coefficients(ul, exp):
    """Calculates coefficients of polynomials for interpolation.
    
    Parameters
    ----------
    ul: float
        upper limit of interpolation range
    exp: numpy.array
        float, exponents of function 'load over voltage'
    
    Returns
    -------
    numpy.array
        float, shape(n,3)"""
    return np.vstack([
        get_coefficients(ul, 1., 1., dp)
        for dp in calc_dpower_dvsqr(ul, exp)])

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

def create_branch_gb_matrix(model):
    """Generates a conductance-susceptance matrix of branches equivalent to
    branch-admittance matrix."""
    branchtaps = model.branchtaps
    pos = casadi.SX.sym('pos', len(branchtaps), 1)
    # factor longitudinal
    if pos.size1():
        flo = (1 - branchtaps.Vstep.to_numpy() * (
            pos - branchtaps.positionneutral.to_numpy()))
    else:
        flo = casadi.SX(0, 1)
    # factor transversal
    ftr = casadi.constpow(flo, 2)
    #branchterminals
    terms = (
        model.branchterminals[~model.branchterminals.is_bridge].reset_index())
    terms_with_taps = terms[terms.index_of_taps.notna()]
    idx_of_tap = terms_with_taps.index_of_taps
    # y_tot
    g_tot = casadi.SX(terms.g_tot)
    b_tot = casadi.SX(terms.b_tot)
    g_tot[terms_with_taps.index] *= ftr[idx_of_tap]
    b_tot[terms_with_taps.index] *= ftr[idx_of_tap]
    # y_mn
    g_mn = casadi.SX(terms.g_mn)
    b_mn = casadi.SX(terms.b_mn)
    g_mn[terms_with_taps.index] *= flo[idx_of_tap]
    b_mn[terms_with_taps.index] *= flo[idx_of_tap]
    terms_with_other_taps = terms[terms.index_of_other_taps.notna()]
    idx_of_other_tap = terms_with_other_taps.index_of_other_taps
    g_mn[terms_with_other_taps.index] *= flo[idx_of_other_tap]
    b_mn[terms_with_other_taps.index] *= flo[idx_of_other_tap]
    # Y
    count_of_nodes = model.shape_of_Y[0]
    G = casadi.SX(count_of_nodes, count_of_nodes)
    B = casadi.SX(count_of_nodes, count_of_nodes)
    for data_idx, idxs in \
        terms.loc[
            :, 
            ['index_of_node', 'index_of_other_node', 'at_slack']].iterrows():
        index_of_node = idxs.index_of_node
        if idxs.at_slack:
            G[index_of_node, index_of_node] = 1
        else:
            index_of_other_node = idxs.index_of_other_node
            G[index_of_node, index_of_node] += g_tot[data_idx]
            G[index_of_node, index_of_other_node] -= g_mn[data_idx]
            B[index_of_node, index_of_node] += b_tot[data_idx]
            B[index_of_node, index_of_other_node] -= b_mn[data_idx]
    GB = casadi.blockcat([[G, -B], [B,  G]])
    # remove rows/columns with GB.remove, 
    #   e.g. remove first row GB.remove([0],[])
    return Branchdata(
        pos=pos,
        branchterminals=terms,
        g_tot=g_tot,
        b_tot=b_tot,
        g_mn=g_mn,
        b_mn=b_mn,
        count_of_nodes=count_of_nodes,
        GB=GB)

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


#%%
from dnadb import egrid_frames
from egrid import model_from_frames

from collections import namedtuple

Vvar = namedtuple(
    'Vvar',
    're im reim node_sqr')

def v_var(count_of_nodes):
    """Creates casadi.SX for voltages.
    
    Parameters
    ----------
    count_of_nodes: int
        number of pfc-nodes
    
    Returns
    -------
    Vvar
        * .re
        * .im
        * .reim
        * .node_sqr"""
    Vre = casadi.SX.sym('Vre', count_of_nodes)
    Vim = casadi.SX.sym('Vim', count_of_nodes)
    Vreim = casadi.vertcat(Vre, Vim)
    Vnode_sqr = casadi.power(Vre, 2) + casadi.power(Vim, 2)
    return Vvar(
        re=Vre,
        im=Vim,
        reim=Vreim,
        node_sqr=Vnode_sqr)

def get_node_inj_matrix(count_of_nodes, injections):
    count_of_injections = len(injections)
    return coo_matrix(
        ([1] * count_of_injections, 
         (injections.index_of_node, injections.index)),
        shape=(count_of_nodes, count_of_injections),
        dtype=np.int8).tocsc()

def get_injected_interpolated_current(
        V, Vinj_sqr, Mnodeinj, exp_v_p, P10, exp_v_q, Q10):
    """Interpolates injected current for V < _VMINSQR. Calculates values
    per node.
    
    Parameters
    ----------
    V: Vvar
        casadi.SX vectors for node voltages
    Vinj_sqr: casadi.SX
        vector, voltage at injection squared
    Mnodeinj: casadi.SX
        matrix
    exp_v_p: numpy.array, float
        voltage exponents of active power per injection
    P10: numpy.array, float
        active power per injection
    exp_v_q: numpy.array, float
        voltage exponents of reactive power per injection
    Q10: numpy.array, float
        active power per injection
        
    Returns
    -------
    tuple
        * injected current per node, real part
        * injected current per node, imaginary part"""
    # per injection
    Vinj = casadi.power(Vinj_sqr, 0.5)
    Vinj_cub = Vinj_sqr * Vinj
    pc = get_polynomial_coefficients(_VMINSQR, exp_v_p)
    qc = get_polynomial_coefficients(_VMINSQR, exp_v_q)
    fpinj = (pc[:,0]*Vinj_cub + pc[:,1]*Vinj_sqr + pc[:,2]*Vinj)
    fqinj = (qc[:,0]*Vinj_cub + qc[:,1]*Vinj_sqr + qc[:,2]*Vinj) 
    # per node
    pnodeexpr = Mnodeinj @ (fpinj * P10)
    qnodeexpr = Mnodeinj @ (fqinj * Q10)
    gt_zero = _EPSILON < V.node_sqr
    Ire_ip = casadi.if_else(
        gt_zero, (pnodeexpr * V.re + qnodeexpr * V.im) / V.node_sqr, 0.0)
    Iim_ip = casadi.if_else(
        gt_zero, (-qnodeexpr * V.re + pnodeexpr * V.im) / V.node_sqr, 0.0)
    return Ire_ip, Iim_ip

def get_injected_original_current(
        V, Vinj_sqr, Mnodeinj, exp_v_p, P10, exp_v_q, Q10):
    """Calculates current flowing into injections. Returns separate
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
    V: Vvar
        casadi.SX vectors for node voltages
    Vinj_sqr: casadi.SX
        vector, voltage at injection squared
    Mnodeinj: casadi.SX
        matrix
    exp_v_p: numpy.array, float
        voltage exponents of active power per injection
    P10: numpy.array, float
        active power per injection
    exp_v_q: numpy.array, float
        voltage exponents of reactive power per injection
    Q10: numpy.array, float
        active power per injection
        
    Returns
    -------
    tuple
        * vector of injected current per node, real part
        * vector of injected current per node, imaginary part"""
    Gexpr_node = Mnodeinj @ (casadi.power(Vinj_sqr, exp_v_p/2 - 1) * P10)
    Bexpr_node = Mnodeinj @ (casadi.power(Vinj_sqr, exp_v_q/2 - 1) * Q10)
    Ire =  Gexpr_node * V.re + Bexpr_node * V.im
    Iim = -Bexpr_node * V.re + Gexpr_node * V.im
    return Ire, Iim

def get_injected_current(count_of_nodes, V, injections):
    """Creates a vector of injected node current.
    
    Parameters
    ----------
    count_of_nodes: int
        number of pfc-nodes
    V: Vvar
        * .re
        * .im
        * .reim
        * node_sqr
    injections: pandas.DataFrame
        * .P10
        * .Q10
        * .Exp_v_p
        * .Exp_v_q
    Returns
    -------
    casadi.SX
        vector, injected current per node (real parts, imaginary parts),
        shape (2*count_of_nodes, 1)"""
    P10 = injections.P10 / 3 # per phase
    Q10 = injections.Q10 / 3 # per phase
    exp_v_p = injections.Exp_v_p
    exp_v_q = injections.Exp_v_q
    Mnodeinj = casadi.SX(get_node_inj_matrix(count_of_nodes, injections))
    Vinj_sqr = casadi.transpose(Mnodeinj) @ V.node_sqr # V**2 per injection
    Ire_ip, Iim_ip = get_injected_interpolated_current(
        V, Vinj_sqr, Mnodeinj, exp_v_p, P10, exp_v_q, Q10)
    Ire, Iim = get_injected_original_current(
        V, Vinj_sqr, Mnodeinj, exp_v_p, P10, exp_v_q, Q10)
    # compose functions from original and interpolated
    interpolate = V.node_sqr < _VMINSQR
    Iinj_re = casadi.if_else(interpolate, Ire_ip, Ire)
    Iinj_im = casadi.if_else(interpolate, Iim_ip, Iim)
    # slacks
    slacks = model.slacks
    Vslack_re = np.real(slacks.V)
    Vslack_im = np.imag(slacks.V)
    index_of_slack = slacks.index_of_node
    Iinj_re[index_of_slack] = -Vslack_re
    Iinj_im[index_of_slack] = -Vslack_im
    return casadi.vertcat(Iinj_re, Iinj_im)

# model
path = r"C:\Users\live\OneDrive\Dokumente\py_projects\data\eus1_loop.db"
frames = egrid_frames(path)
model = model_from_frames(frames)
# branch gb-matrix
gb = create_branch_gb_matrix(model).GB
# variables of voltages
count_of_nodes = model.shape_of_Y[0]
V = v_var(count_of_nodes)
# injected current
injections_ = model.injections
Inode = get_injected_current(
    count_of_nodes, V, injections_[~injections_.is_slack])
# equation for root finding
Ires = (gb @ V.reim) + Inode
# solve root-finding equation
fn_Iresidual = casadi.Function('fn_Iresidual', [V.reim], [Ires])
rf = casadi.rootfinder('rf', 'nlpsol', fn_Iresidual, {'nlpsol':'ipopt'})
start = [1.0] * count_of_nodes + [0.] * count_of_nodes
voltages = rf(start)
Vcalc = np.array(
    casadi.hcat(
        casadi.vertsplit(
            voltages, [0, voltages.size1()//2, voltages.size1()])))
Vcomp = Vcalc.view(dtype=np.complex128)
print()
print('V: ', Vcomp)
