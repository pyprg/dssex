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

Created on Fri May  6 20:44:05 2022

@author: pyprg
"""

import numpy as np
from numpy.linalg import norm
from functools import partial
from operator import itemgetter
from scipy.sparse import \
    csc_array, coo_matrix, bmat, diags, csc_matrix, vstack#, hstack
from scipy.sparse.linalg import splu
from injections import get_polynomial_coefficients
from src.dssex.util import get_tap_factors

# square of voltage magnitude, minimum value for load curve, 
#   if value is below _VMINSQR the load curves for P and Q converge
#   towards a linear load curve which is 0 when V=0; P(V=0)=0, Q(V=0)=0
_VMINSQR = 0.8

_zeros = np.zeros((0, 1), dtype=np.longdouble)
_power_props = itemgetter('P10', 'Q10', 'Exp_v_p', 'Exp_v_q')

def get_gb_terms(terms, flo, ftr):
    """Multiplies conductance/susceptance of branches with factors retrieved
    from tap positions.
    
    Parameters
    ----------
    terms: pandas.DataFrame
    
    flo: pandas.Series, float
        factor for longitudinal admittance
    ftr: pandas.Series, float
        factor transversal admittance"""
    terms_with_taps = terms[terms.index_of_taps.notna()]
    idx_of_tap = terms_with_taps.index_of_taps
    g_tr = terms.g_tr_half.to_numpy()
    b_tr = terms.b_tr_half.to_numpy()
    g_tr[terms_with_taps.index] *= ftr[idx_of_tap]   
    b_tr[terms_with_taps.index] *= ftr[idx_of_tap]    
    g_lo = terms.g_lo.to_numpy()
    b_lo = terms.b_lo.to_numpy()
    g_lo[terms_with_taps.index] *= flo[idx_of_tap]
    b_lo[terms_with_taps.index] *= flo[idx_of_tap]
    terms_with_other_taps = terms[terms.index_of_other_taps.notna()]
    idx_of_other_tap = terms_with_other_taps.index_of_other_taps
    g_lo[terms_with_other_taps.index] *= flo[idx_of_other_tap]
    b_lo[terms_with_other_taps.index] *= flo[idx_of_other_tap]
    return g_tr, g_lo, b_tr, b_lo

def create_gb(terms, count_of_nodes, flo, ftr):
    """Generates a conductance-susceptance matrix of branches equivalent to
    branch-admittance matrix. M[n,n] of slack nodes is set to 1, other
    values of slack nodes are zero.
    
    Parameters
    ----------
    terms: pandas.DataFrame
    
    count_of_nodes: int
        number of power flow calculation nodes
    flo: array_like
        double, longitudinal taps factor, sparse for terminals with taps
    ftr: array_like
        transversal taps factor, sparse for terminals with taps
    
    Returns
    -------
    tuple
        * sparse matrix of branch conductances G
        * sparse matrix of branch susceptances B"""
    index_of_node = terms.index_of_node
    index_of_other_node = terms.index_of_other_node
    row = np.concatenate([index_of_node, index_of_node])
    col = np.concatenate([index_of_node, index_of_other_node])
    rowcol = row, col
    g_tr, g_lo, b_tr, b_lo = get_gb_terms(terms, flo, ftr)
    gvals = np.concatenate([(g_tr + g_lo), -g_lo])
    bvals = np.concatenate([(b_tr + b_lo), -b_lo])
    shape = count_of_nodes, count_of_nodes
    g = coo_matrix((gvals, rowcol), shape=shape, dtype=float)
    b = coo_matrix((bvals, rowcol), shape=shape, dtype=float)
    return g, b

def create_gb_matrix(model, pos):
    """Generates a conductance-susceptance matrix of branches equivalent to
    branch-admittance matrix. M[n,n] of slack nodes is set to 1, other
    values of slack nodes are zero. Hence, the returned 
    matrix is unsymmetrical.
    
    Parameters
    ----------
    model: egrid.model.Model
    
    pos: numpy.array, int
        vector of position, one variable for each terminal with taps
    
    Returns
    -------
    scipy   sparse   matrix"""
    flo, ftr = get_tap_factors(model.branchtaps, pos)
    count_of_nodes = model.shape_of_Y[0]
    terms = (
        model.branchterminals[~model.branchterminals.is_bridge].reset_index())
    G, B = create_gb(terms, count_of_nodes, flo, ftr)
    count_of_slacks = model.count_of_slacks
    diag = diags(
        [1.] * count_of_slacks, 
        shape=(count_of_slacks, count_of_nodes), 
        dtype=float)
    G_ = vstack([diag.tocsc(), G.tocsc()[count_of_slacks:, :]])
    B_ = vstack([
        csc_matrix((count_of_slacks, count_of_nodes), dtype=float), 
        B.tocsc()[count_of_slacks:, :]])
    return bmat([[G_, -B_], [B_,  G_]])

def _injected_power(vminsqr, injections):
    """Calculates power flowing through injections.
    Injected power is calculated this way
    (P = |V|**Exvp * P10, Q = |V|**Exvq * Q10; with |V| - magnitude of V):
    ::
        +- -+   +-                                           -+
        | P |   | (V_r ** 2 + V_i ** 2) ** (Expvp / 2) * P_10 |
        |   | = |                                             |
        | Q |   | (V_r ** 2 + V_i ** 2) ** (Expvq / 2) * Q_10 |
        +- -+   +-                                           -+

    Parameters
    ----------
    vminsqr: float
        upper limit of interpolation, interpolates if |V|² < vminsqr
    injections: pandas.DataFrame
        * .kp
        * .P10
        * .kq
        * .Q10
        * .Exp_v_p
        * .Exp_v_q
        * .V_abs_sqr
        * .c3p, .c2p, .c1p, polynomial coefficients for active power P
        * .c3q, .c2q, .c1q, polynomial coefficients for reactive power Q 

    Returns
    -------
    function: (numpy.array<float>) -> (numpy.array<float>, numpy.array<float>)
        tuple
            * active power P
            * reactive power Q"""
    P10, Q10, Exp_v_p, Exp_v_q = _power_props(injections)
    P10 = P10.copy() / 3 # calculate per phase
    Q10 = Q10.copy() / 3 # calculate per phase
    p_coeffs = get_polynomial_coefficients(vminsqr, injections.Exp_v_p)
    q_coeffs = get_polynomial_coefficients(vminsqr, injections.Exp_v_q)
    coeffs = np.hstack([p_coeffs, q_coeffs])
    Exp_v_p_half = Exp_v_p / 2.
    Exp_v_q_half = Exp_v_q / 2.
    # Exp_v_p_half = Exp_v_p.to_numpy() / 2.
    # Exp_v_q_half = Exp_v_q.to_numpy() / 2.
    #        
    def calc_injected_power(Vinj_abs_sqr):
        """Calculates injected power per injection.
        
        Parameters
        ----------
        Vinj_abs_sqr: numpy.array, float, shape (n,1)
            vector of squared voltage-magnitudes at injections, 
            n: number of injections

        Returns
        -------
        tuple
            * active power P
            * reactive power Q"""
        Vinj_abs_sqr2 = np.array(Vinj_abs_sqr).reshape(-1)
        Pres = np.array(P10)
        Qres = np.array(Q10)
        interpolate = np.array(Vinj_abs_sqr2 < vminsqr).reshape(-1)
        # original
        Vsqr_orig = Vinj_abs_sqr2[~interpolate]
        Pres[~interpolate] *= np.power(Vsqr_orig, Exp_v_p_half[~interpolate])
        Qres[~interpolate] *= np.power(Vsqr_orig, Exp_v_q_half[~interpolate])
        # polynomial interpolated
        Vsqr_inter = Vinj_abs_sqr2[interpolate]
        cinterpolate = coeffs[interpolate]
        V_abs = np.power(Vsqr_inter, .5)
        V321 = (
            np.hstack([Vsqr_inter * V_abs, Vsqr_inter, V_abs])
            .reshape(-1, 3))
        Pres[interpolate] *= np.sum(V321 * cinterpolate[:, :3], axis=1)
        Qres[interpolate] *= np.sum(V321 * cinterpolate[:, 3:], axis=1)
        return Pres, Qres
    return calc_injected_power

def calculate_injected_power(vminsqr, injections, Vinj_abs_sqr):
    """Calculates injected power per injection.
    Injected power is calculated this way
    (P = |V|**Exvp * P10, Q = |V|**Exvq * Q10; with |V| - magnitude of V):
    ::
        +- -+   +-                                           -+
        | P |   | (V_r ** 2 + V_i ** 2) ** (Expvp / 2) * P_10 |
        |   | = |                                             |
        | Q |   | (V_r ** 2 + V_i ** 2) ** (Expvq / 2) * Q_10 |
        +- -+   +-                                           -+

    Parameters
    ----------
    vminsqr: float
        upper limit of interpolation, interpolates if |V|² < vminsqr
    injections: pandas.DataFrame
        * .kp
        * .P10
        * .kq
        * .Q10
        * .Exp_v_p
        * .Exp_v_q
        * .V_abs_sqr
        * .c3p, .c2p, .c1p, polynomial coefficients for active power P
        * .c3q, .c2q, .c1q, polynomial coefficients for reactive power Q 
    Vinj_abs_sqr: numpy.array, float, shape (n,1)
        vector of squared voltage-magnitudes at injections, 
        n: number of injections

    Returns
    -------
    tuple
        * active power P
        * reactive power Q"""
    return _injected_power(vminsqr, injections)(Vinj_abs_sqr)

get_injected_power = partial(calculate_injected_power, _VMINSQR)

def calculate_injected_node_current(
        mnodeinj, mnodeinjT, calc_injected_power, idx_slack, Vslack, Vnode_ri):
    Vnode_ri2 = np.hstack(np.vsplit(Vnode_ri, 2))
    Vnode_ri2_sqr = np.power(Vnode_ri2, 2)
    Vnode_abs_sqr = (Vnode_ri2_sqr[:, 0] + Vnode_ri2_sqr[:, 1]).reshape(-1, 1)
    Vinj_abs_sqr = mnodeinjT @ Vnode_abs_sqr
    Pinj, Qinj = calc_injected_power(Vinj_abs_sqr)
    Sinj = (
        np
        .hstack([Pinj.reshape(-1, 1), Qinj.reshape(-1, 1)])
        .view(dtype=np.complex128))
    Sinj_node = mnodeinj @ csc_array(Sinj)
    Vnode = Vnode_ri2.view(dtype=np.complex128)
    Iinj_node = -Sinj_node / Vnode #current is negative for positive power
    Iinj_node[idx_slack] = Vslack
    return np.vstack([np.real(Iinj_node), np.imag(Iinj_node)])

def next_voltage(
        mnodeinj, mnodeinjT, calc_injected_power, gb_lu, 
        idx_slack, Vslack, Vnode_ri):
    """Solves linear equation.
    
    Parameters
    ----------
    mnodeinj: scipy.sparse.matrix
        
    mnodeinjT: scipy.sparse.matrix
    
    calc_injected_power: function 
        (numpy.array, float) -> 
            (numpy.array<float>, numpy.array<float>)
        (square of absolute voltages at injection terminals) ->
            (active power, reactive power)
    gb_lu: scipy.linalg.SolveLU
        LU-decomposition of gb-matrix (conductance, susceptance)
    idx_slack: array_like, int
        indices of slack nodes
    Vslack: array_like, float
        voltages at slack nodes
    Vnode_ri: array_like
        voltage of previous iteration
    
    Yields
    ------
    tuple
        * array_like, float, node voltages
        * array_like, float, injected node currents"""
    while True:
        Iinj_node_ri = calculate_injected_node_current(
            mnodeinj, mnodeinjT, calc_injected_power, 
            idx_slack, Vslack, Vnode_ri)
        yield Vnode_ri, Iinj_node_ri
        Vnode_ri = gb_lu.solve(Iinj_node_ri)

def solved(precision, gb, Vnode_ri, Iinj_node_ri):
    """Success function. Evaluates solution Vnode_ri, Iinj_node_ri.
    
    Parameters
    ----------
    precision: float
        tolerance of node current to be reached
    gb: scipy.sparse.matrix
        conductance, susceptance matrix
    Vnode_ri: numpy.array
        vector of node voltages, real parts then imaginary parts
    Iinj_node_ri: numpy.array
        vector of injected node currents, real parts then imaginary parts
    
    Returns
    -------
    bool"""
    Ires = gb.dot(Vnode_ri) - Iinj_node_ri
    Ires_max = norm(Ires, np.inf)
    return Ires_max < precision

def calculate_power_flow(
        precision, max_iter, model, 
        Vslack=None, tappositions=None, Vinit=None):
    """Power flow calculating function. The function solves the non-linear
    power flow problem by solving the linear equations Y * U_n+1 = I(U_n) 
    iteratively. U_n+1 is computed from Y and I(U_n) - n: index of iteration.
    
    Parameters
    ----------
    precision: float
        tolerance for node current
    max_iter: int
        limit of iteration count
    model: egrid.model.Model
        
    Vslack: array_like, complex
        vector of voltages at slacks, default model.slacks.V
    tappositions: array_like, int
        vector of tap positions, default model.branchtaps.position
    Vinit: array_like, complex
        start value of iteration, node voltage vector
    
    Returns
    -------
    tuple
        * bool, success?
        * array_like, float, node voltages, real parts then imaginary parts
        * array_like, float, injected node currents, 
          real parts then imaginary parts"""
    count_of_nodes = model.shape_of_Y[0]
    Vinit_ = (np.array([1.0]*count_of_nodes + [0.0]*count_of_nodes)
              .reshape(-1, 1)
              if Vinit is None else
              np.vstack([np.real(Vinit), np.imag(Vinit)]))
    Vslack_ = model.slacks.V if Vslack is None else Vslack
    tappositions_ = model.branchtaps.position.copy() \
        if tappositions is None else tappositions
    gb = create_gb_matrix(model, tappositions_)
    mnodeinj = model.mnodeinj
    _next_voltage = partial(
        next_voltage, 
        mnodeinj,
        mnodeinj.T,
        _injected_power(_VMINSQR, model.injections), 
        splu(gb),
        model.slacks.index_of_node,
        Vslack_) 
    _solved = partial(solved, precision, gb)    
    iter_counter = 0
    for V, I in _next_voltage(Vinit_):
        if _solved(V, I):
            return True, np.hstack(np.vsplit(V, 2)).view(dtype=np.complex128)
        if max_iter <= iter_counter:
            break
        ++iter_counter;
    return False, np.hstack(np.vsplit(V, 2)).view(dtype=np.complex128)
