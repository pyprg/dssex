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


Numeric power flow calculation with separated real and imaginary parts
with function 'calculate_power_flow'.

Result processing with functions 'calculate_results' and
'calculate_electric_data'.
"""

import numpy as np
import pandas as pd
from numpy.linalg import norm
from functools import partial
from collections import namedtuple
from operator import itemgetter
from scipy.sparse import \
    csc_array, coo_matrix, bmat, diags, csc_matrix, vstack#, hstack
from scipy.sparse.linalg import splu
from dssex.injections import get_polynomial_coefficients

# square of voltage magnitude, minimum value for load curve,
#   if value is below _VMINSQR the load curves for P and Q converge
#   towards a linear load curve which is 0 when V=0; P(V=0)=0, Q(V=0)=0
_VMINSQR = 0.8**2

_zeros = np.zeros((0, 1), dtype=np.longdouble)
_power_props = itemgetter('P10', 'Q10', 'Exp_v_p', 'Exp_v_q')

def get_tap_factors(branchtaps, pos):
    """Creates factors for tap positions, values for off-diagonal and
    diagonal factors of branches.

    Parameters
    ----------
    branchtaps: pandas.DataFrame (id of taps)
        * .Vstep, float voltage diff per tap
        * .positionneutral, int
    pos: array_like, int
        vector of positions for terms with tap

    Returns
    -------
    numpy array, float
        off-diagonal factors"""
    # factor longitudinal
    foffd = (1 - branchtaps.Vstep.to_numpy() * (
        pos - branchtaps.positionneutral.to_numpy()))
    return foffd

#
# calculation with separated real and imaginary parts
#

def get_gb_terms(terms, foffd):
    """Multiplies conductance/susceptance of branches with factors retrieved
    from tap positions.

    Parameters
    ----------
    terms: pandas.DataFrame

    foffd: pandas.Series, float
        tap-factor for off-diagonal admittance y_mn

    Returns
    -------
    tuple
        * numpy.array, float, g_lo
        * numpy.array, float, g_tot
        * numpy.array, float, b_lo
        * numpy.array, float, b_tot"""
    terms_with_taps = terms[terms.index_of_taps.notna()]
    idx_of_tap = terms_with_taps.index_of_taps
    g_lo = terms.g_lo.to_numpy()
    b_lo = terms.b_lo.to_numpy()
    g_tot = terms.g_tr_half.to_numpy() + g_lo
    b_tot = terms.b_tr_half.to_numpy() + b_lo
    foffd_of_tap = foffd[idx_of_tap]
    fdiag_of_tap = foffd_of_tap * foffd_of_tap
    g_tot[terms_with_taps.index] *= fdiag_of_tap
    b_tot[terms_with_taps.index] *= fdiag_of_tap
    g_lo = terms.g_lo.to_numpy()
    b_lo = terms.b_lo.to_numpy()
    g_lo[terms_with_taps.index] *= foffd_of_tap
    b_lo[terms_with_taps.index] *= foffd_of_tap
    terms_with_other_taps = terms[terms.index_of_other_taps.notna()]
    idx_of_other_tap = terms_with_other_taps.index_of_other_taps
    foffd_of_other_tap = foffd[idx_of_other_tap]
    g_lo[terms_with_other_taps.index] *= foffd_of_other_tap
    b_lo[terms_with_other_taps.index] *= foffd_of_other_tap
    return g_lo, g_tot, b_lo, b_tot

def create_gb(terms, count_of_nodes, foffd):
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
    g_lo, g_tot, b_lo, b_tot = get_gb_terms(terms, foffd)
    gvals = np.concatenate([g_tot, -g_lo])
    bvals = np.concatenate([b_tot, -b_lo])
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
    foffd = get_tap_factors(model.branchtaps, pos)
    count_of_nodes = model.shape_of_Y[0]
    terms = (
        model.branchterminals[~model.branchterminals.is_bridge].reset_index())
    G, B = create_gb(terms, count_of_nodes, foffd)
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

def _get_squared_injected_power_fn(injections, pq_factors=None):
    """Calculates power flowing into injections.
    ::
        +- -+   +-                            -+
        | P |   | (V_r ** 2 + V_i ** 2) * P_10 |
        |   | = |                              |
        | Q |   | (V_r ** 2 + V_i ** 2) * Q_10 |
        +- -+   +-                            -+

    Parameters
    ----------
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
    pq_factors: numpy.array, float, (nx2)
        optional
        factors for active and reactive power
    loadcurve: 'original' | 'interpolated' | 'square'

    Returns
    -------
    function: (numpy.array<float>) -> (numpy.array<float>, numpy.array<float>)
        tuple
            * active power P
            * reactive power Q"""
    P10, Q10, _, __ = _power_props(injections)
    P10 = P10.copy() / 3 # calculate per phase, assumes P10 is a 3-phase-value
    Q10 = Q10.copy() / 3 # calculate per phase, assumes Q10 is a 3-phase-value
    if not pq_factors is None:
        P10 *= pq_factors[:,0]
        Q10 *= pq_factors[:,1]
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
        Vsqr = np.array(Vinj_abs_sqr).reshape(-1)
        return np.array(P10) * Vsqr, np.array(Q10) * Vsqr
    return calc_injected_power

def _get_original_injected_power_fn(injections, pq_factors=None):
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
    injections: pandas.DataFrame
        * .kp
        * .P10
        * .kq
        * .Q10
        * .Exp_v_p
        * .Exp_v_q
        * .V_abs_sqr
    pq_factors: numpy.array, float, (nx2)
        optional
        factors for active and reactive power
    loadcurve: 'original' | 'interpolated' | 'square'

    Returns
    -------
    function: (numpy.array<float>) -> (numpy.array<float>, numpy.array<float>)
        tuple
            * active power P
            * reactive power Q"""
    P10, Q10, Exp_v_p, Exp_v_q = _power_props(injections)
    P10 = P10.copy() / 3 # calculate per phase
    Q10 = Q10.copy() / 3 # calculate per phase
    if not pq_factors is None:
        P10 *= pq_factors[:,0]
        Q10 *= pq_factors[:,1]
    Exp_v_p_half = Exp_v_p.to_numpy() / 2.
    Exp_v_q_half = Exp_v_q.to_numpy() / 2.
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
        Vsqr = np.array(Vinj_abs_sqr).reshape(-1)
        Pres = np.array(P10) * np.power(Vsqr, Exp_v_p_half)
        Qres = np.array(Q10) * np.power(Vsqr, Exp_v_q_half)
        return Pres, Qres
    return calc_injected_power

def _get_interpolated_injected_power_fn(vminsqr, injections, pq_factors=None):
    """Calculates power flowing through injections.

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
    pq_factors: numpy.array, float, (nx2)
        optional
        factors for active and reactive power
    loadcurve: 'original' | 'interpolated' | 'square'

    Returns
    -------
    function: (numpy.array<float>) -> (numpy.array<float>, numpy.array<float>)
        tuple
            * active power P
            * reactive power Q"""
    P10, Q10, Exp_v_p, Exp_v_q = _power_props(injections)
    P10 = P10.copy() / 3 # calculate per phase
    Q10 = Q10.copy() / 3 # calculate per phase
    if not pq_factors is None:
        P10 *= pq_factors[:,0]
        Q10 *= pq_factors[:,1]
    p_coeffs = get_polynomial_coefficients(vminsqr, injections.Exp_v_p)
    q_coeffs = get_polynomial_coefficients(vminsqr, injections.Exp_v_q)
    coeffs = np.hstack([p_coeffs, q_coeffs])
    Exp_v_p_half = Exp_v_p / 2.
    Exp_v_q_half = Exp_v_q / 2.
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
        Pres = np.array(P10, dtype=float)
        Qres = np.array(Q10, dtype=float)
        interpolate = np.array(Vinj_abs_sqr2 < vminsqr).reshape(-1)
        # original
        Vsqr_orig = Vinj_abs_sqr2[~interpolate]
        Pres[~interpolate] *= np.power(Vsqr_orig, Exp_v_p_half[~interpolate])
        Qres[~interpolate] *= np.power(Vsqr_orig, Exp_v_q_half[~interpolate])
        # polynomial interpolated
        Vsqr_inter = Vinj_abs_sqr2[interpolate].reshape(-1, 1)
        cinterpolate = coeffs[interpolate]
        V_abs = np.power(Vsqr_inter, .5)
        V321 = np.hstack([Vsqr_inter * V_abs, Vsqr_inter, V_abs])
        Pres[interpolate] *= np.sum(V321 * cinterpolate[:, :3], axis=1)
        Qres[interpolate] *= np.sum(V321 * cinterpolate[:, 3:], axis=1)
        return Pres, Qres
    return calc_injected_power

def get_calc_injected_power_fn(
        vminsqr, injections, pq_factors=None, loadcurve='original'):
    """Returns a function calculating power flowing through injections.

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
    pq_factors: numpy.array, float, (nx2)
        optional
        factors for active and reactive power
    loadcurve: 'original' | 'interpolated' | 'square'

    Returns
    -------
    function: (numpy.array<float>) -> (numpy.array<float>, numpy.array<float>)
        tuple
            * active power P
            * reactive power Q"""
    control_character = loadcurve[:1].lower()
    if control_character == 's': # square
        return _get_squared_injected_power_fn(injections, pq_factors)
    if control_character == 'o': # original
        return _get_original_injected_power_fn(injections, pq_factors)
    return _get_interpolated_injected_power_fn(vminsqr, injections, pq_factors)

get_injected_power_fn = partial(get_calc_injected_power_fn, _VMINSQR)

def calculate_injected_node_current(
        mnodeinj, mnodeinjT, calc_injected_power, idx_slack, Vslack, Vnode_ri):
    """Calculates injected current per injection. Special processing of
    slack nodes.

    Parameters
    ----------
    mnodeinj: scipy.sparse.matrix
        mnodeinj @ values_per_injection -> values_per_node
    mnodeinjT: scipy.sparse.matrix
        mnodeinjT @ values_per_node -> values_per_injection
    calc_injected_power: function

    idx_slack: array_like, int
        indices of slack nodes
    Vslack: array_like, float
        complex, voltages at slack nodes
    Vnode_ri: pandas.Series
        vector of node voltages (real and imaginary separated)

    Returns
    -------
    numpy.array
        float, shape (2*number_of_nodes,)"""
    Vnode_ri2 = np.hstack(np.vsplit(Vnode_ri, 2))
    Vnode_ri2_sqr = np.power(Vnode_ri2, 2)
    Vnode_abs_sqr = Vnode_ri2_sqr.sum(axis=1).reshape(-1, 1)
    Vinj_abs_sqr = mnodeinjT @ Vnode_abs_sqr
    Pinj, Qinj = calc_injected_power(Vinj_abs_sqr)
    Sinj = (
        np.hstack([Pinj.reshape(-1, 1), Qinj.reshape(-1, 1)])
        .view(dtype=np.complex128))
    Sinj_node = mnodeinj @ csc_array(Sinj)
    Vnode = Vnode_ri2.view(dtype=np.complex128)
    #current is negative for positive power
    Iinj_node = -np.conjugate(Sinj_node / Vnode)
    Iinj_node[idx_slack] = Vslack
    return np.vstack([np.real(Iinj_node), np.imag(Iinj_node)])

def next_voltage(
        mnodeinj, mnodeinjT, calc_injected_power, gb_lu,
        idx_slack, Vslack, Vnode_ri):
    """Solves linear equation.

    Parameters
    ----------
    mnodeinj: scipy.sparse.matrix
        mnodeinjT @ values_per_injection -> values_per_node
    mnodeinjT: scipy.sparse.matrix
        mnodeinjT @ values_per_node -> values_per_injection
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
    """Success-predicate function. Evaluates solution Vnode_ri, Iinj_node_ri.

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
        Vslack=None, tappositions=None, Vinit=None,
        pq_factors=None, loadcurve='original'):
    """Power flow calculating function. The function solves the non-linear
    power flow problem by solving the linear equations Y * U_n+1 = I(U_n)
    iteratively. U_n+1 is computed from Y and I(U_n). n: index of iteration.

    Parameters
    ----------
    precision: float
        tolerance for node current
    max_iter: int
        limit of iteration count
    model: egrid.model.Model
        data of electric grid
    Vslack: array_like, complex
        vector of voltages at slacks, default model.slacks.V
    tappositions: array_like, int
        vector of tap positions, default model.branchtaps.position
    Vinit: array_like, complex
        start value of iteration, node voltage vector
    pq_factors: numpy.array, float, (nx2)
        factors for active and reactive power of loads
    loadcurve: 'original' | 'interpolated' | 'square'
        default is 'original', just first letter is used

    Returns
    -------
    tuple
        * bool, success?
        * array_like, complex, node voltages"""
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
        get_calc_injected_power_fn(
            _VMINSQR, model.injections, pq_factors, loadcurve),
        splu(gb),
        model.slacks.index_of_node,
        Vslack_)
    _solved = partial(solved, precision, gb) # success predicate
    iter_counter = 0
    for V, I in _next_voltage(Vinit_):
        if _solved(V, I):
            return True, np.hstack(np.vsplit(V, 2)).view(dtype=np.complex128)
        if max_iter <= iter_counter:
            break
        iter_counter += 1;
    return False, np.hstack(np.vsplit(V, 2)).view(dtype=np.complex128)

#
# calculation with complex values
#

def get_y_terms(terms, foffd):
    """Multiplies admittances of branches with factors retrieved
    from tap positions.

    Parameters
    ----------
    terms: pandas.DataFrame

    foffd: pandas.Series, float
        factor for off-diagonal admittance -y_mn

    Returns
    -------
    tuple
        * numpy.array, complex, y_lo, longitudinal admittance, y_mn
        * numpy.array, complex, y_tot, diagonal admittance y_mm"""
    terms_with_taps = terms[terms.index_of_taps.notna()]
    idx_of_tap = terms_with_taps.index_of_taps
    y_tr = terms.y_tr_half.to_numpy()
    y_lo = terms.y_lo.to_numpy()
    y_tot = y_tr + y_lo
    foffd_of_tap = foffd[idx_of_tap]
    y_lo[terms_with_taps.index] *= foffd_of_tap
    y_tot[terms_with_taps.index] *= (foffd_of_tap*foffd_of_tap)
    terms_with_other_taps = terms[terms.index_of_other_taps.notna()]
    idx_of_other_tap = terms_with_other_taps.index_of_other_taps
    y_lo[terms_with_other_taps.index] *= foffd[idx_of_other_tap]
    return y_lo, y_tot

def create_y(terms, count_of_nodes, foffd):
    """Generates the branch-admittance matrix.

    Parameters
    ----------
    terms: pandas.DataFrame

    count_of_nodes: int
        number of power flow calculation nodes
    foffd: array_like
        double, off-diagonal factor for y_mn and y_nm admittances,
        sparse for terminals with taps

    Returns
    -------
    tuple
        * sparse matrix of branch admittances Y"""
    index_of_node = terms.index_of_node
    index_of_other_node = terms.index_of_other_node
    row = np.concatenate([index_of_node, index_of_node])
    col = np.concatenate([index_of_node, index_of_other_node])
    rowcol = row, col
    y_lo, y_tot = get_y_terms(terms, foffd)
    yvals = np.concatenate([y_tot, -y_lo])
    shape = count_of_nodes, count_of_nodes
    return coo_matrix((yvals, rowcol), shape=shape, dtype=np.complex128)

def create_y_matrix(model, pos):
    """Generates the branch-admittance matrix.
    M[n,n] of slack nodes is set to 1, other values of slack nodes are zero.
    Hence, the returned matrix is unsymmetrical.

    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    pos: numpy.array, int
        vector of position, one variable for each terminal with taps

    Returns
    -------
    scipy.sparse.matrix"""
    foffd = get_tap_factors(model.branchtaps, pos)
    count_of_nodes = model.shape_of_Y[0]
    terms = (
        model.branchterminals[~model.branchterminals.is_bridge].reset_index())
    Y = create_y(terms, count_of_nodes, foffd)
    count_of_slacks = model.count_of_slacks
    diag = diags(
        [1.+0.j] * count_of_slacks,
        shape=(count_of_slacks, count_of_nodes),
        dtype=np.complex128)
    return vstack([diag.tocsc(), Y.tocsc()[count_of_slacks:, :]])

def create_y_matrix2(model, pos):
    """Generates admittance matrix of branches without rows for slacks.

    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    pos: numpy.array, int
        vector of position, one variable for each terminal with taps

    Returns
    -------
    scipy.sparse.matrix"""
    foffd = get_tap_factors(model.branchtaps, pos)
    count_of_nodes = model.shape_of_Y[0]
    terms = (
        model.branchterminals[~model.branchterminals.is_bridge].reset_index())
    Y = create_y(terms, count_of_nodes, foffd)
    count_of_slacks = model.count_of_slacks
    return Y.tocsc()[count_of_slacks:, :]

def get_injected_power_per_injection(calculate_injected_power, Vinj):
    """

    Parameters
    ----------
    calculate_injected_power: function
        (numpy.array<float>) -> (numpy.array<float>, numpy.array<float>)
        (square_of_absolute_node-voltage) -> (active power P, reactive power Q)
    Vinj: numpy.array
        complex, voltage per injection

    Returns
    -------
    tuple
        * numpy.array, float, real power per injection
        * numpy.array, float, imaginary power per injection
        * numpy.array, float, voltage per injection"""
    Vinj_abs_sqr = np.power(np.real(Vinj), 2) + np.power(np.imag(Vinj), 2)
    return *calculate_injected_power(Vinj_abs_sqr), Vinj

def get_injected_current_per_node(calculate_injected_power, model, Vnode):
    """Calculates injected current per power flow calculation node.

    Parameters
    ----------
    calculate_injected_power: function
        (numpy.array<float>) ->
        (numpy.array<float>, numpy.array<float>)
        (square_of _absolute_node-voltage) ->
        (active power P, reactive power Q)
    model: egrid.model.Model
        model of grid for calculation
    Vnode: numpy.array
        complex, voltage per node

    Returns
    -------
    numpy.array, complex, injected current per node"""
    Pinj, Qinj, _ = get_injected_power_per_injection(
        calculate_injected_power, model.mnodeinj.T @ Vnode)
    Sinj = (
        np.hstack([Pinj.reshape(-1, 1), Qinj.reshape(-1, 1)])
        .view(dtype=np.complex128))
    return np.conjugate((model.mnodeinj @ Sinj) / Vnode)

#
# result processing
#

def calculate_injection_results(calculate_injected_power, model, Vnode):
    """Calculates electric data of injections according to node voltage.
    Returns active and reactive power in pu.

    Parameters
    ----------
    calculate_injected_power: function
        (injections, square of absolute voltages at injection terminals) ->
            (active power, reactive power)
    model: egrid.model.Model
        data of the electric power network
    Vnode: numpy.array
        complex, vector of node voltages

    Returns
    -------
    pandas.DataFrame
        id, Exp_v_p, Exp_v_q, P10, Q10, P_pu, Q_pu, V_pu,
        I_pu, Vcx_pu, Scx_pu, Icx_pu"""
    df = model.injections.loc[
        :, ['id', 'Exp_v_p', 'Exp_v_q', 'P10', 'Q10']]
    df['P_pu'], df['Q_pu'], Vinj = get_injected_power_per_injection(
        calculate_injected_power, model.mnodeinj.T @ Vnode)
    df['V_pu'] = np.abs(Vinj)
    n = df['P_pu'].size
    S = np.empty((n,1), dtype=np.complex128)
    S.real = df['P_pu'].array.reshape(-1,1)
    S.imag = df['Q_pu'].array.reshape(-1,1)
    Icx_pu = (S / Vinj).conjugate()
    df['I_pu'] = np.abs((S / Vinj).conjugate()) # abs of conjugate
    df['P_pu'] *= 3 # converts from single phase calculation to 3-phase system
    df['Q_pu'] *= 3 # converts from single phase calculation to 3-phase system
    df['Vcx_pu'] = Vinj
    df['Scx_pu'] = 3 * S
    df['Icx_pu'] = Icx_pu
    return df

def get_crossrefs(terms, count_of_nodes):
    """Creates connectivity matrices
    ::
        index_of_terminal, index_of_node -> 1
        index_of_terminal, index_of_other_node -> 1

    Parameters
    ----------
    terms: pandas.DataFrame, index of terminals
        * .index_of_node
        * .index_of_other_node
    count_of_nodes: int
        number of power flow calculation nodes

    Returns
    -------
    tuple
        * scipy.sparse.matrix, index_of_terminal, index_of_node
        * scipy.sparse.matrix, index_of_terminal, index_of_other_node"""
    count_of_terms = len(terms)
    mtermnode = coo_matrix(
        ([1] * count_of_terms,
         (terms.index, terms.index_of_node)),
        shape=(count_of_terms, count_of_nodes),
        dtype=np.int8).tocsc()
    #other
    mtermothernode = coo_matrix(
        ([1] * count_of_terms,
         (terms.index, terms.index_of_other_node)),
        shape=(count_of_terms, count_of_nodes),
        dtype=np.int8).tocsc()
    return mtermnode, mtermothernode

def get_branch_admittance_matrices(y_lo, y_tot, term_is_at_A):
    """Creates a 2x2 branch-admittance matrix for each branch.

    Parameters
    ----------
    y_lo: numpy.array, complex
        y_mn admittance, per branch
    y_tot: numpy.array, complex
        (y_mn + y_mm) admittance, per branch
    term_is_at_A: numpy.array, bool, index of terminal
        True if terminal is at side A of a branch

    Returns
    -------
    numpy.darray, complex, shape=(:, 2, 2)"""
    y_tot_A = y_tot[term_is_at_A]
    y_tot_B = y_tot[~term_is_at_A]
    y_lo_AB = y_lo[term_is_at_A]
    y_11 = y_tot_A
    y_12 = -y_lo_AB
    y_21 = -y_lo_AB
    y_22 = y_tot_B
    return (np.hstack([
         y_11.reshape(-1, 1),
         y_12.reshape(-1, 1),
         y_21.reshape(-1, 1),
         y_22.reshape(-1, 1)])
        .reshape(-1, 2, 2))

def get_y_branches(model, terms, term_is_at_A, pos):
    """Creates one admittance matrix per branch.

    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    terms: pandas.DataFrame

    term_is_at_A: numpy.array, bool, index of terminal
       True if terminal is at side A of a branch
    pos: pandas.Series
        int, positions of taps

    Returns
    -------
    numpy.array, complex, shape=(:, 2, 2)"""
    foffd = get_tap_factors(model.branchtaps, pos)
    y_lo, y_tot = get_y_terms(terms, foffd)
    return get_branch_admittance_matrices(y_lo, y_tot, term_is_at_A)

def get_v_branches(terms, voltages):
    """Creates a voltage vector 2x1 per branch.

    Parameters
    ----------
    terms: pandas.DataFrame

    voltages: numpy.array, complex
        voltages at nodes

    Returns
    -------
    numpy.array, complex, shape=(:, 2, 1)"""
    mtermnode, mtermothernode = get_crossrefs(terms, len(voltages))
    Vterm = np.asarray(mtermnode @ voltages)
    Votherterm = np.asarray(mtermothernode @ voltages)
    return np.hstack([Vterm, Votherterm]).reshape(-1, 2, 1)

def calculate_branch_results(model, Vnode, pos):
    """Calculates P, Q per branch terminal. Calculates Ploss, Qloss per branch.

    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    Vnode: numpy.array, complex
        voltages at nodes
    pos: pandas.Series
        int, positions of taps

    Returns
    -------
    pandas.DataFrame
        id, I0_pu, I1_pu, P0_pu, Q0_pu, P1_pu, Q1_pu, Ploss_pu, Qloss_pu,
        I0cx_pu, I1cx_pu, V0cx_pu, V1cx_pu, V0_pu, V1_pu"""
    branchterminals = model.branchterminals
    terms = branchterminals[(~branchterminals.is_bridge)].reset_index()
    term_is_at_A = terms.side == 'A'
    Ybr = get_y_branches(model, terms, term_is_at_A, pos)
    Vbr = get_v_branches(terms[term_is_at_A], Vnode)
    Ibr = Ybr @ Vbr
    # converts from single phase calculation to 3-phase system
    Sbr = 3 * Vbr * Ibr.conjugate()            # S0, S1
    PQbr= Sbr.view(dtype=float).reshape(-1, 4) # P0, P1, Q0, Q1
    Sbr_loss = Sbr.sum(axis=1)
    dfbr = (
        terms.loc[term_is_at_A, ['id_of_branch']]
        .rename(columns={'id_of_branch': 'id'}))
    dfv = pd.DataFrame(
        Vbr.reshape(-1, 2),
        columns=['V0cx_pu', 'V1cx_pu'])
    dfv_abs = dfv.abs()
    dfv_abs.columns = 'V0_pu', 'V1_pu'
    res = np.hstack(
        [np.abs(Ibr).reshape(-1,2),
         PQbr,
         Sbr_loss.view(dtype=float)])
    dfres = pd.DataFrame(
        res,
        columns=[
            'I0_pu', 'I1_pu', 'P0_pu', 'Q0_pu', 'P1_pu', 'Q1_pu', 'Ploss_pu',
            'Qloss_pu'])
    dfi = pd.DataFrame(Ibr.reshape(-1,2), columns=('I0cx_pu', 'I1cx_pu'))
    return pd.concat([dfbr, dfres, dfi, dfv, dfv_abs], axis=1)

def calculate_results(model, power_fn, tappositions, Vnode):
    """Calculates and arranges electric data of injections and branches
    for a given voltage vector which is typically the result of a power
    flow calculation.

    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    power_fn: function or 'original' | 'interpolated' | 'square'
        if function:
            (pandas.DataFrame, numpy.array<float>) ->
            (numpy.array<float>, numpy.array<float>)
            (injections, square_of _absolute_node-voltage) ->
            (active power P, reactive power Q)
    tappositions: array_like, int
        positions of taps
    Vnode: array_like, complex
        node voltage vector

    Returns
    -------
    dict
        * 'injections': pandas.DataFrame
        * 'branches': pandas.DataFrame"""
    power_fn_ = (get_injected_power_fn(model.injections, loadcurve=power_fn)
                 if isinstance(power_fn, str) else power_fn)
    return {
        'injections': calculate_injection_results(power_fn_, model, Vnode),
        'branches': calculate_branch_results(model, Vnode, tappositions)}

def get_residual_current(model, get_injected_power, Y, Vnode):
    """Calculates the complex residual current per node.

    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    get_injected_power: function
        (numpy.array<float>) -> (numpy.array<float>, numpy.array<float>)
        (square_of_absolute_node-voltage) -> (active power P, reactive power Q)
    Y: scipy.sparse.matrix, complex
        branch admittance matrix
    Vnode: array_like, complex
        node voltage vector

    Returns
    -------
    numpy.array
        complex, residual of node current"""
    V_ =  Vnode.reshape(-1, 1)
    Inode = Y @ V_
    Iinj = get_injected_current_per_node(get_injected_power, model, V_)
    return (Inode + Iinj).reshape(-1)

def get_residual_current2(model, get_injected_power, Vslack, Y, Vnode):
    """Calculates the complex residual current per node without slack nodes.

    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    get_injected_power: function
        (numpy.array<float>) -> (numpy.array<float>, numpy.array<float>)
        (square_of_absolute_node-voltage) -> (active power P, reactive power Q)
    Vslack: numpy.array
        complex, voltages at slack nodes
    tappositions: array_like, int
        positions of taps
    Vnode: array_like, complex
        node voltage vector

    Returns
    -------
    numpy.array
        complex, residual of node current"""
    count_of_slacks = model.count_of_slacks
    V_ =  np.hstack([Vslack.reshape(-1), Vnode.reshape(-1)]).reshape(-1, 1)
    Inode = Y @ V_
    Iinj = (
        get_injected_current_per_node(get_injected_power, model, V_)
        [count_of_slacks:])
    return (Inode + Iinj).reshape(-1)

def get_residual_current_fn(model, get_injected_power, tappositions=None):
    """Parameterizes function get_residual_current.

    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    get_injected_power: function
        (numpy.array<float>) -> (numpy.array<float>, numpy.array<float>)
        (square_of_absolute_node-voltage) -> (active power P, reactive power Q)
    tappositions: array_like, int
        positions of taps

    Returns
    -------
    function
        (numpy.array<complex>) -> (numpy.array<complex>)
        (voltage_of_nodes) -> (current_of_node)"""
    tappositions_ = model.branchtaps.position \
        if tappositions is None else tappositions
    Y = create_y_matrix(model, tappositions_).tocsc()
    return partial(get_residual_current, model, get_injected_power, Y)

def get_residual_current_fn2(
        model, get_injected_power, Vslack=None, tappositions=None):
    """Parameterizes function get_residual_current2.

    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    get_injected_power: function
        (numpy.array<float>) -> (numpy.array<float>, numpy.array<float>)
        (square_of_absolute_node-voltage) -> (active power P, reactive power Q)
    Vslack: array_like
        complex, voltages at slack nodes
    tappositions: array_like, int
        positions of taps

    Returns
    -------
    function
        (numpy.array<complex>) -> (numpy.array<complex>)
        (voltage_at_nodes) -> (current_into_nodes)"""
    tappositions_ = model.branchtaps.position \
        if tappositions is None else tappositions
    Vslack_ = model.slacks.V.to_numpy() if Vslack is None else Vslack
    Y = create_y_matrix2(model, tappositions_).tocsc()
    return partial(
        get_residual_current2, model, get_injected_power, Vslack_, Y)

def eval_residual_current(
        model, get_injected_power, tappositions=None, Vnode=None):
    """Convenience function for evaluation of a power flow calculation result.
    Calls function get_residual_current_fn and get_residual_current

    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    get_injected_power: function
        (numpy.array<float>) -> (numpy.array<float>, numpy.array<float>)
        (square_of_absolute_node-voltage) -> (active power P, reactive power Q)
    tappositions: array_like, int
        positions of taps
    Vnode: array_like, complex
        node voltage vector

    Returns
    -------
    numpy.array
        complex, residual node current"""
    return (
        get_residual_current_fn(model, get_injected_power, tappositions)(Vnode)
        .reshape(-1, 1))

Electric_data = namedtuple(
    'Electric_data',
    'branch injection node residual_node_current')
Electric_data.__doc__ = """Functions for calculating electric data for
branches, injections and nodes from power flow or estimation results.

Parameters
----------
branch: function
    (array_like<str>)->(pandas.DataFrame)
injection: function
    (array_like<str>)->(pandas.DataFrame)
node: function
    ()->(pandas.DataFrame)
residual_node_current: function
    ()->(numpy.array<complex>)"""

def calculate_electric_data(
        model, voltages_cx, pq_factors=None,
        tappositions=None, vminsqr=_VMINSQR, loadcurve='interpolated'):
    """Calculates and arranges electric data of injections and branches
    for a given voltage vector which is e.g. the result of a power
    flow calculation.

    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    voltages_cx : array_like, complex
        node voltage vector
    pq_factors: numpy.array, float, (nx2)
        optional
        factors for active and reactive power
    tappositions : array_like, int, optional
        Positions of taps. The default is model.branchtaps.position.
    vminsqr : float, optional
        Upper limit of interpolation, interpolates if |V|² < vminsqr.
        The default is _VMINSQR.
    loadcurve : str, optional
        'original'|'interpolated'|'square'. The default is 'interpolated'.

    Returns
    -------
    Electric_data
        * .branch, function (array_like<str>)->(pandas.DataFrame)
        * .injection, function (array_like<str>)->(pandas.DataFrame)
        * .residual_node_current, function ()->(numpy.array<complex>)"""
    from pandas import DataFrame as DF
    tappositions_ = (
        model.branchtaps.position if tappositions is None else tappositions)
    get_injected_power = get_calc_injected_power_fn(
        vminsqr, model.injections, pq_factors, loadcurve)
    result_data = calculate_results(
        model, get_injected_power, tappositions_, voltages_cx)
    def br_data(columns=None):
        """Returns calculated electric data of branches.

        Parameters
        ----------
        columns: array_like, optional
            default is ('P0_pu', 'P1_pu', 'Q0_pu', 'Q1_pu',
             'V0_pu', 'V1_pu', 'I0_pu', 'I1_pu', 'Ploss_pu', 'Qloss_pu')
            column names, for list all possible names see function
            calculate_branch_results

        Returns
        -------
        pandas.DataFrame"""
        res = result_data['branches'].set_index('id')
        return res.reindex(
            columns=(
                ('P0_pu', 'P1_pu', 'Q0_pu', 'Q1_pu',
                 'V0_pu', 'V1_pu', 'I0_pu', 'I1_pu', 'Ploss_pu', 'Qloss_pu')
                if columns is None else columns),
            copy=False)
    def inj_data(columns=None):
        """Returns calculated electric data of injections.

        Parameters
        ----------
        columns: array_like, optional
            default is ('P_pu', 'Q_pu', 'V_pu', 'I_pu', 'kp', 'kq',
            'P10', 'Q10', 'Exp_v_p', 'Exp_v_q')
            column names, for list all possible names see function
            calculate_injection_results

        Returns
        -------
        pandas.DataFrame"""
        res = result_data['injections'].set_index('id')
        res['kp'], res['kq'] =  np.hsplit(
            (np.ones(shape=(len(model.injections),2), dtype=float)
             if pq_factors is None else pq_factors),
            2)
        return res.reindex(
            columns=(
                ('P_pu', 'Q_pu', 'V_pu', 'I_pu', 'kp', 'kq', 'P10', 'Q10',
                 'Exp_v_p', 'Exp_v_q')
                if columns is None else columns),
            copy=False)
    return Electric_data(
        branch=br_data,
        injection=inj_data,
        node=lambda:DF(
            {'V_pu':np.abs(voltages_cx.reshape(-1)),
             'Vcx_pu':voltages_cx.reshape(-1)},
            index=model.nodes.index,
            columns=['V_pu', 'Vcx_pu']),
        residual_node_current=lambda:eval_residual_current(
            model, get_injected_power, Vnode=voltages_cx))

