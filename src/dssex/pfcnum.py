# -*- coding: utf-8 -*-
"""
Copyright (C) 2022, 2023 pyprg

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

def calculate_term_to_factor_n(factors, positions=None):
    """Calculates values for off-diagonal factors of branches having taps.

    Diagonal factors are just the square of the off-diagonal factors.
    The applied formula for the factor is
    ::
        m * position + n

    Parameters
    ----------
    factors: Factors
        * .gen_factor_data, pandas.DataFrame
        * .gen_injfactor, pandas.DataFrame
        * .terminalfactors, pandas.DataFrame
        * .factorgroups: function
            (iterable_of_int)-> (pandas.DataFrame)
        * .injfactorgroups: function
            (iterable_of_int)-> (pandas.DataFrame)
    pos: numpy.array
        float, tap-position

    Returns
    -------
    pandas.DataFrame (index_of_terminal) ->
        * .ftaps, float"""
    termfactor = factors.terminalfactors
    pos_ = termfactor.value.to_numpy() if positions is None else positions
    if termfactor.empty:
        # empty pandas.DataFrame
        return pd.DataFrame(
            [],
            columns=['ftaps'],
            dtype=np.float64,
            index=pd.Index([], dtype=np.int64, name='index_of_terminal'))
    return pd.DataFrame(
        # mx + n
        ((termfactor.m * pos_) + termfactor.n).array,
        columns=['ftaps'],
        index=termfactor.index_of_terminal)

def _calculate_f_mn_tot(index_of_other_terminal, term_to_factor):
    """Calculates terminal factors (taps factors) for each terminal.

    Parameters
    ----------
    index_of_other_terminal: pandas.DataFrame (index_of_terminal) ->
        * .index_of_other_terminal, int
    term_to_factor: pandas.DataFrame (index_of_terminal) ->
        * .ftaps, float, taps-factor

    Returns
    -------
    numpy.array, float (shape n,2)"""
    array = (
        # adds factor of terminal to terminal
        index_of_other_terminal
        .join(term_to_factor, how='left')
        .set_index('index_of_other_terminal')
        # adds factor of other terminal
        .join(term_to_factor, how='left', rsuffix='_other')
        .fillna(1.)
        # also swaps columns
        .to_numpy()[:,[1,0]])
    return array * array[:,[1]]

def create_gb_of_terminals_n(branchterminals, term_to_factor):
    """Creates vectors of branch-susceptances and branch-conductances.

    Parameters
    ----------
    branchterminals: pandas.DataFrame (index_of_terminal)
        * .g_lo, float, longitudinal conductance
        * .b_lo, float, longitudinal susceptance
        * .g_tr_half, float, transversal conductance
        * .b_tr_half, float, transversal susceptance
    term_to_factor: pandas.DataFrame (index_of_terminal) ->
        * .ftaps, float

    Returns
    -------
    numpy.array (shape n,4)
        * gb_mn_tot[:,0] - g_mn
        * gb_mn_tot[:,1] - b_mn
        * gb_mn_tot[:,2] - g_tot
        * gb_mn_tot[:,3] - b_tot"""
    # g_lo, b_lo, g_trans, b_trans
    gb_mn_tot = (
        branchterminals
        .loc[:,['g_lo', 'b_lo', 'g_tr_half', 'b_tr_half']]
        .to_numpy())
    # gb_mn_mm -> gb_mn_tot
    gb_mn_tot[:, 2:] += gb_mn_tot[:, :2]
    f_mn_tot = _calculate_f_mn_tot(
        branchterminals[['index_of_other_terminal']], term_to_factor)
    gb_mn_tot[:, :2] *= f_mn_tot[:,[0]]
    gb_mn_tot[:, 2:] *= f_mn_tot[:,[1]]
    return gb_mn_tot.copy()

def create_gb(branchterminals, count_of_nodes, term_to_factor):
    """Generates a conductance-susceptance matrix of branches.

    The conductance-susceptance matrix is equivalent to a
    branch-admittance matrix.

    Parameters
    ----------
    branchterminals: pandas.DataFrame

    count_of_nodes: int
        number of power flow calculation nodes
    term_to_factor: pandas.DataFrame (index_of_terminal) ->
        * .ftaps, float

    Returns
    -------
    tuple
        * sparse matrix of branch conductances G
        * sparse matrix of branch susceptances B"""
    gb_mn_tot = create_gb_of_terminals_n(branchterminals, term_to_factor)
    index_of_node = branchterminals.index_of_node
    index_of_other_node = branchterminals.index_of_other_node
    row = np.concatenate([index_of_node, index_of_node])
    col = np.concatenate([index_of_node, index_of_other_node])
    rowcol = row, col
    gvals = np.concatenate([gb_mn_tot[:,2], -gb_mn_tot[:,0] ])
    bvals = np.concatenate([gb_mn_tot[:,3], -gb_mn_tot[:,1]])
    shape = count_of_nodes, count_of_nodes
    g = coo_matrix((gvals, rowcol), shape=shape, dtype=float)
    b = coo_matrix((bvals, rowcol), shape=shape, dtype=float)
    return g, b

def create_gb_matrix(model, term_to_factor):
    """Generates a conductance-susceptance matrix of branches.

    The result is equivalent to a branch-admittance matrix.
    M[n,n] of slack nodes is set to 1, other values of slack nodes are zero.
    Hence, the returned matrix is unsymmetrical.

    Parameters
    ----------
    model: egrid.model.Model
        data of power network
    term_to_factor: pandas.DataFrame (index_of_terminal) ->
        * .ftaps, float

    Returns
    -------
    scipy.sparse.matrix"""
    count_of_nodes = model.shape_of_Y[0]
    terms = (
      model.branchterminals[~model.branchterminals.is_bridge].reset_index())
    G, B = create_gb(terms, count_of_nodes, term_to_factor)
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

    Formula:
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
    """Calculates injected current per injection.

    Special processing of slack nodes.

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
    """Success-predicate function.

    Evaluates solution Vnode_ri, Iinj_node_ri.

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
    return norm(Ires, np.inf) < precision if 0 < Ires.shape[0] else True

def calculate_power_flow(
        model, Vslack=None, Vinit=None,
        pq_factors=None, loadcurve='original', precision=1e-8, max_iter=30):
    """Power flow calculating function.

    The function solves the non-linear power flow problem by solving the linear
    equations Y * V_n+1 = I(V_n) iteratively. V_n+1 is computed from
    Y and I(V_n). n: index of iteration. While in- and output use complex
    values the solver uses separated values for real and imaginary parts.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    Vslack: numpy.array (nx1), optional
        complex, vector of voltages at slacks, default model.slacks.V
    Vinit: array_like, optional
        float, start value of iteration, node voltage vector,
        real parts then imaginary parts
    pq_factors: numpy.array (nx2), optional
        float, factors for active and reactive power of loads
    loadcurve: 'original' | 'interpolated' | 'square', optional
        default is 'original', just first letter is used
    precision: float, optional
        tolerance for node current
    max_iter: int, optional
        limit of iteration count

    Returns
    -------
    tuple
        * bool, success?
        * numpy.ndarray, complex, node voltages"""
    count_of_nodes = model.shape_of_Y[0]
    Vinit_ = (np.array([1.0]*count_of_nodes + [0.0]*count_of_nodes)
              .reshape(-1, 1)
              if Vinit is None else
              np.vstack([np.real(Vinit), np.imag(Vinit)]))
    Vslack_ = (
        model.slacks.V.to_numpy().reshape(-1,1) if Vslack is None else Vslack)
    term_to_factor = calculate_term_to_factor_n(model.factors)
    gb = create_gb_matrix(model, term_to_factor)
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

def get_y_terms(branchterminals, term_to_factor):
    """Creates y_mn and y_tot of terminals.

    Multiplies admittances of branches with factors retrieved from
    tap positions.

    Parameters
    ----------
    branchterminals: pandas.DataFrame (index_of_terminal)
        * .g_lo, float, longitudinal conductance
        * .b_lo, float, longitudinal susceptance
        * .g_tr_half, float, transversal conductance
        * .b_tr_half, float, transversal susceptance
    term_factor: pandas.DataFrame (index_of_terminal) ->
        * .ftaps, float

    Returns
    -------
    tuple
        * numpy.array, complex, y_mn, longitudinal admittance, y_mn
        * numpy.array, complex, y_tot, diagonal admittance ymn + y_mm"""
    y_tr = branchterminals.y_tr_half.to_numpy().reshape(-1,1)
    y_mn = branchterminals.y_lo.to_numpy().reshape(-1,1)
    y_tot = y_tr + y_mn
    f_mn_tot = _calculate_f_mn_tot(
        branchterminals[['index_of_other_terminal']], term_to_factor)
    return y_mn * f_mn_tot[:,[0]], y_tot * f_mn_tot[:,[1]]

def create_y(terms, count_of_nodes, term_factor):
    """Generates the branch-admittance matrix.

    Parameters
    ----------
    terms: pandas.DataFrame

    count_of_nodes: int
        number of power flow calculation nodes
    term_factor: pandas.DataFrame (index_of_terminal) ->
        * .ftaps, float

    Returns
    -------
    tuple
        * sparse matrix of branch admittances Y"""
    index_of_node = terms.index_of_node
    index_of_other_node = terms.index_of_other_node
    row = np.concatenate([index_of_node, index_of_node])
    col = np.concatenate([index_of_node, index_of_other_node])
    rowcol = row, col
    y_mn, y_tot = get_y_terms(terms, term_factor)
    yvals = np.concatenate([y_tot.reshape(-1), -y_mn.reshape(-1)])
    shape = count_of_nodes, count_of_nodes
    return coo_matrix((yvals, rowcol), shape=shape, dtype=np.complex128)

def create_y_matrix(model, term_factor):
    """Generates the branch-admittance matrix.

    M[n,n] of slack nodes is set to 1, other values of slack nodes are zero.
    Hence, the returned matrix is unsymmetrical.

    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    term_factor: pandas.DataFrame (index_of_terminal) ->
        * .ftaps, float

    Returns
    -------
    scipy.sparse.matrix"""
    count_of_nodes = model.shape_of_Y[0]
    terms = (
        model.branchterminals[~model.branchterminals.is_bridge].reset_index())
    Y = create_y(terms, count_of_nodes, term_factor)
    count_of_slacks = model.count_of_slacks
    diag = diags(
        [1.+0.j] * count_of_slacks,
        shape=(count_of_slacks, count_of_nodes),
        dtype=np.complex128)
    return vstack([diag.tocsc(), Y.tocsc()[count_of_slacks:, :]])

def create_y_matrix2(model, term_factor):
    """Generates admittance matrix of branches without rows for slacks.

    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    term_factor: pandas.DataFrame (index_of_terminal) ->
        * .ftaps, float

    Returns
    -------
    scipy.sparse.matrix"""
    count_of_nodes = model.shape_of_Y[0]
    terms = (
        model.branchterminals[~model.branchterminals.is_bridge].reset_index())
    Y = create_y(terms, count_of_nodes, term_factor)
    count_of_slacks = model.count_of_slacks
    return Y.tocsc()[count_of_slacks:, :]

def get_injected_power_per_injection(calculate_injected_power, Vinj):
    """Calculates active and reactive power for each injection.

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
    """Creates connectivity matrices.

    In particular:
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
    numpy.darray, complex, shape=(n, 2, 2)"""
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
    pos: numpy.array
        float, tap-position

    Returns
    -------
    numpy.array, complex, shape=(n, 2, 2)"""
    term_to_factor = calculate_term_to_factor_n(model.factors, pos)
    y_lo, y_tot = get_y_terms(terms, term_to_factor)
    return get_branch_admittance_matrices(y_lo, y_tot, term_is_at_A)

def get_v_branches(terms, voltages):
    """Creates a voltage vector 2x1 per branch.

    Parameters
    ----------
    terms: pandas.DataFrame, index of terminals
        * .index_of_node
        * .index_of_other_node
    voltages: numpy.array, complex
        voltages at nodes

    Returns
    -------
    numpy.array, complex, shape=(n, 2, 1)"""
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
    pos: numpy.array
        float, tap-positions

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

def calculate_results(model, power_fn, Vnode, pos):
    """Calculates and arranges electric data of injections and branches.

    Uses a given voltage vector which is typically the result of a power
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
    Vnode: array_like, complex
        node voltage vector
    pos: numpy.array
        float, tap position

    Returns
    -------
    dict
        * 'injections': pandas.DataFrame
        * 'branches': pandas.DataFrame"""
    power_fn_ = (get_injected_power_fn(model.injections, loadcurve=power_fn)
                 if isinstance(power_fn, str) else power_fn)
    return {
        'injections': calculate_injection_results(power_fn_, model, Vnode),
        'branches': calculate_branch_results(model, Vnode, pos)}

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
    numpy.ndarray
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

def get_residual_current_fn(model, get_injected_power):
    """Parameterizes function get_residual_current.

    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    get_injected_power: function
        (numpy.array<float>) -> (numpy.array<float>, numpy.array<float>)
        (square_of_absolute_node-voltage) -> (active power P, reactive power Q)

    Returns
    -------
    function
        (numpy.array<complex>) -> (numpy.ndarray<complex>)
        (voltage_of_nodes) -> (current_of_node)"""
    term_to_factor = calculate_term_to_factor_n(model.factors)
    Y = create_y_matrix(model, term_to_factor).tocsc()
    return partial(get_residual_current, model, get_injected_power, Y)

def get_residual_current_fn2(model, get_injected_power, Vslack=None):
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

    Returns
    -------
    function
        (numpy.array<complex>) -> (numpy.array<complex>)
        (voltage_at_nodes) -> (current_into_nodes)"""
    Vslack_ = model.slacks.V.to_numpy() if Vslack is None else Vslack
    term_to_factor = calculate_term_to_factor_n(model.factors)
    Y = create_y_matrix2(model, term_to_factor).tocsc()
    return partial(
        get_residual_current2, model, get_injected_power, Vslack_, Y)

def eval_residual_current(
        model, get_injected_power, Vnode=None):
    """Convenience function for evaluation of a power flow calculation result.

    Calls function get_residual_current_fn and get_residual_current

    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    get_injected_power: function
        (numpy.array<float>) -> (numpy.array<float>, numpy.array<float>)
        (square_of_absolute_node-voltage) -> (active power P, reactive power Q)
    Vnode: array_like, complex
        node voltage vector

    Returns
    -------
    numpy.ndarray
        complex, residual node current"""
    return (
        get_residual_current_fn(model, get_injected_power)(Vnode)
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
        model, voltages_cx, pq_factors=None, pos=None,
        vminsqr=_VMINSQR, loadcurve='interpolated'):
    """Calculates and arranges electric data of injections and branches.

    Uses a given voltage vector, e.g. the result of a power flow calculation.

    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    voltages_cx : array_like, complex
        node voltage vector
    pq_factors: numpy.array, float, (nx2)
        optional
        factors for active and reactive power
    pos: numpy.array, float, (nx1)
        optional
        factors for terminals (taps)
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
        * .node, function ()->(pandas.DataFrame)
        * .residual_node_current, function ()->(numpy.ndarray<complex>)"""
    from pandas import DataFrame as DF
    get_injected_power = get_calc_injected_power_fn(
        vminsqr, model.injections, pq_factors, loadcurve)
    result_data = calculate_results(
        model, get_injected_power, voltages_cx, pos)
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
