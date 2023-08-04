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
from operator import itemgetter
from scipy.sparse import \
    csc_array, coo_matrix, bmat, diags, csc_matrix, vstack
from scipy.sparse.linalg import splu
from dssex.injections import get_polynomial_coefficients

# square of voltage magnitude, minimum value for load curve,
#   if value is below _VMINSQR the load curves for P and Q converge
#   towards a linear load curve which is 0 when V=0; P(V=0)=0, Q(V=0)=0
_VMINSQR = 0.8**2

_zeros = np.zeros((0, 1), dtype=np.longdouble)
_power_props = itemgetter('P10', 'Q10', 'Exp_v_p', 'Exp_v_q')

def get_term_to_factor_n(terminalfactors, positions):
    """Calculates one tap factor for each row in DataFrame terminalfactors.

    Positions must provide a value for each row in terminalfactors if not None.

    The applied formula for the factor is
    ::
        m * position + n

    Parameters
    ----------
    terminalfactors : pandas.DataFrame
        * .value, float
        * .m, float
        * .n, float
    positions : array_like | None
        float, tap positions, one value for each row in terminalfactors,
        ordered according to terminalfactors

    Returns
    -------
    numpy.array, float"""
    if terminalfactors.empty:
        return np.empty((0,1), dtype=float)
    pos = (
        terminalfactors.value.to_numpy()
        if positions is None else
        positions.reshape(-1))
    return ((terminalfactors.m * pos) + terminalfactors.n).to_numpy()

def values_per_terminal(arr, terminalfactors, positions):
    """Distributes values of terminals associated to terminalfactors
    over terminals.

    Parameters
    ----------
    arr: numpy.array (shape n,2)

    terminalfactors: pandas.DataFrame (index_of_factor) ->
        * .index_of_terminal
        * .index_of_other_terminal
    positions: array_like
        float, one position for each terminal factor

    Returns
    -------
    numpy.array (shape n,2)"""
    if positions.size:
        index_of_terminal = terminalfactors.index_of_terminal
        arr[index_of_terminal, 1] = positions
        arr[terminalfactors.index_of_other_terminal, 0] = (
            arr[index_of_terminal, 1])
    return arr

def calculate_f_mn_tot_n(count_of_terminals, terminalfactors, term_to_factor):
    """Calculates off-diagonal and diagonal factors for all branches.

    Off-diagonal factors (mn) are the product of both terminal factors of one
    branch, diagonal factors (tot) are the square of the terminal factors.
    The function calculates two factors for each terminal.

    Parameters
    ----------
    count_of_terminals: int
        number of branch terminals
    terminalfactors: pandas.DataFrame
        * .index_of_terminal, int
        * .index_of_other_terminal, int
    term_to_factor: numpy.array
        float, voltage factor (m * position + n)

    Returns
    -------
    numpy.array (shape n,2)
        float, f_mn - [:, 0], f_tot - [:, 1]"""
    ft = values_per_terminal(
        np.ones((count_of_terminals, 2), dtype=float),
        terminalfactors,
        term_to_factor)
    return ft * ft[:,[1]]

def get_f_mn_tot_n(terminalfactors, count_of_terminals, positions):
    """Calculates off-diagonal and diagonal factors for all branches.

    Off-diagonal factors (mn) are the product of both terminal factors of one
    branch, diagonal factors (tot) are the square of the terminal factors.
    The function calculates two factors for each terminal.

    Parameters
    ----------
    model: egrid.model.Model
        data of an electric distribtion network
    count_of_terminals: int
        number of branch terminals
    positions : array_like | None
        float, tap positions, one value for each row in terminalfactors,
        ordered according to terminalfactors

    Returns
    -------
    numpy.array (shape n,2)
        float, f_mn - [:, 0], f_tot - [:, 1]"""
    term_to_factor = get_term_to_factor_n(terminalfactors, positions)
    return calculate_f_mn_tot_n(
        count_of_terminals, terminalfactors, term_to_factor)

def create_gb_of_terminals_n(branchterminals, f_mn_tot):
    """Creates vectors of branch-susceptances and branch-conductances.

    Parameters
    ----------
    branchterminals: pandas.DataFrame (index_of_terminal)
        * .g_lo, float, longitudinal conductance
        * .b_lo, float, longitudinal susceptance
        * .g_tr_half, float, transversal conductance
        * .b_tr_half, float, transversal susceptance
    f_mn_tot: numpy.array (shape n,2)
        float, f_mn - [:, 0], f_tot - [:, 1]

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
    # f_mn_tot = _calculate_f_mn_tot(
    #     branchterminals[['index_of_other_terminal']], term_to_factor)
    gb_mn_tot[:, :2] *= f_mn_tot[:,[0]]
    gb_mn_tot[:, 2:] *= f_mn_tot[:,[1]]
    return gb_mn_tot.copy()

def create_gb(branchterminals, count_of_nodes, f_mn_tot):
    """Generates a conductance-susceptance matrix of branches.

    The conductance-susceptance matrix is equivalent to a
    branch-admittance matrix.

    Parameters
    ----------
    branchterminals: pandas.DataFrame

    count_of_nodes: int
        number of power flow calculation nodes
    f_mn_tot: numpy.array (shape n,2)
        float, f_mn - [:, 0], f_tot - [:, 1]

    Returns
    -------
    tuple
        * sparse matrix of branch conductances G
        * sparse matrix of branch susceptances B"""
    gb_mn_tot = create_gb_of_terminals_n(branchterminals, f_mn_tot)
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

def create_gb_matrix(model, f_mn_tot):
    """Generates a conductance-susceptance matrix of branches.

    The result is equivalent to a branch-admittance matrix.
    M[n,n] of slack nodes is set to 1, other values of slack nodes are zero.
    Hence, the returned matrix is unsymmetrical.

    Parameters
    ----------
    model: egrid.model.Model
        data of power network
    f_mn_tot: numpy.array (shape n,2)
        float, f_mn - [:, 0], f_tot - [:, 1]

    Returns
    -------
    scipy.sparse.matrix"""

    count_of_nodes = model.shape_of_Y[0]
    terms = model.branchterminals[~model.branchterminals.is_bridge]
    factors = f_mn_tot[terms.index,:]
    G, B = create_gb(terms, count_of_nodes, factors)
    count_of_slacks = model.count_of_slacks
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

def _get_squared_injected_power_fn(injections, kpq):
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
    injections: pandas.DataFrame (index_of_terminal)
        * .P10
        * .Q10
        * .Exp_v_p
        * .Exp_v_q
    kpq: numpy.array, float, (nx2) | None
        scaling factors for active and reactive power

    Returns
    -------
    function: (numpy.array<float>) -> (numpy.array<float>, numpy.array<float>)
        tuple
            * active power P
            * reactive power Q"""
    P10, Q10, _, __ = _power_props(injections)
    P10 = P10.copy() / 3 # calculate per phase, assumes P10 is a 3-phase-value
    Q10 = Q10.copy() / 3 # calculate per phase, assumes Q10 is a 3-phase-value
    if not kpq is None:
        P10 *= kpq[:,0]
        Q10 *= kpq[:,1]
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

def _get_original_injected_power_fn(injections, kpq):
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
    injections: pandas.DataFrame (index_of_terminal)
        * .P10
        * .Q10
        * .Exp_v_p
        * .Exp_v_q
    kpq: numpy.array, float, (nx2) | None
        scaling factors for active and reactive power

    Returns
    -------
    function: (numpy.array<float>) -> (numpy.array<float>, numpy.array<float>)
        tuple
            * active power P
            * reactive power Q"""
    P10, Q10, Exp_v_p, Exp_v_q = _power_props(injections)
    P10 = P10.copy() / 3 # calculate per phase
    Q10 = Q10.copy() / 3 # calculate per phase
    if not kpq is None:
        P10 *= kpq[:,0]
        Q10 *= kpq[:,1]
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

def _get_interpolated_injected_power_fn(vminsqr, injections, kpq):
    """Calculates power flowing through injections.

    Parameters
    ----------
    vminsqr: float
        upper limit of interpolation, interpolates if |V|² < vminsqr
    injections: pandas.DataFrame (index_of_terminal)
        * .P10
        * .Q10
        * .Exp_v_p
        * .Exp_v_q
    kpq: numpy.array, float, (nx2) | None
        factors for active and reactive power

    Returns
    -------
    function: (numpy.array<float>) -> (numpy.array<float>, numpy.array<float>)
        tuple
            * active power P
            * reactive power Q"""
    P10, Q10, Exp_v_p, Exp_v_q = _power_props(injections)
    P10 = P10.copy() / 3 # calculate per phase
    Q10 = Q10.copy() / 3 # calculate per phase
    if not kpq is None:
        P10 *= kpq[:,0]
        Q10 *= kpq[:,1]
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
        vminsqr, injections, kpq=None, loadcurve='interpolated'):
    """Returns a function calculating power flowing through injections.

    Parameters
    ----------
    vminsqr: float
        upper limit of interpolation, interpolates if |V|² < vminsqr
    injections: pandas.DataFrame (index_of_terminal)
        * .P10
        * .Q10
        * .Exp_v_p
        * .Exp_v_q
    kpq: numpy.array, float, (nx2)
        optional
        scaling factors for active and reactive power
    loadcurve: 'original' | 'interpolated' | 'square'
        optional, default 'interpolated'

    Returns
    -------
    function: (numpy.array<float>) -> (numpy.array<float>, numpy.array<float>)
        tuple
            * active power P
            * reactive power Q"""
    control_character = loadcurve[:1].lower()
    if control_character == 's': # square
        return _get_squared_injected_power_fn(injections, kpq)
    if control_character == 'o': # original
        return _get_original_injected_power_fn(injections, kpq)
    return _get_interpolated_injected_power_fn(vminsqr, injections, kpq)

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
    try:
        Vinj_abs_sqr = mnodeinjT @ Vnode_abs_sqr
    except ValueError:
        raise ValueError('faulty connections from nodes to injections')
    Pinj, Qinj = calc_injected_power(Vinj_abs_sqr)
    Sinj = (
        np.hstack([Pinj.reshape(-1, 1), Qinj.reshape(-1, 1)])
        .view(dtype=np.complex128))
    Sinj_node = mnodeinj @ csc_array(Sinj)
    Vnode = Vnode_ri2.view(dtype=np.complex128)
    #current is negative for positive power
    Iinj_node = -np.conjugate(Sinj_node / Vnode)
    Iinj_node[idx_slack] = Vslack
    return np.vstack([np.real(Iinj_node), np.imag(Iinj_node)]).A1.reshape(-1,1)

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
        model, /, Vslack=None, *, Vinit=None,
        kpq=None, positions=None, loadcurve='original',
        precision=1e-8, max_iter=30):
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
    kpq: numpy.array (nx2), optional
        float, scaling factors for active and reactive power of loads,
        uses 1.0 if omitted
    positions : array_like (shape n,1) | None, optional
        float, tap positions, one value for each row in terminalfactors,
        ordered according to terminalfactors,
        applies factors of model if omitted
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
    Vinit_ = (
        np.array([1.0]*count_of_nodes + [0.0]*count_of_nodes).reshape(-1, 1)
        if Vinit is None else
        np.vstack([np.real(Vinit), np.imag(Vinit)]))
    Vslack_ = (
        model.slacks.V.to_numpy().reshape(-1,1) if Vslack is None else Vslack)
    terminalfactors = model.factors.terminalfactors
    term_to_factor_ = get_term_to_factor_n(terminalfactors, positions)
    terms = model.branchterminals
    count_of_branchterminals = len(terms) - sum(terms.is_bridge)
    f_mn_tot = calculate_f_mn_tot_n(
        count_of_branchterminals, terminalfactors, term_to_factor_)
    gb = create_gb_matrix(model, f_mn_tot)
    mnodeinj = model.mnodeinj
    _next_voltage = partial(
        next_voltage,
        mnodeinj,
        mnodeinj.T,
        get_calc_injected_power_fn(
            _VMINSQR, model.injections, kpq, loadcurve),
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

def get_y_terms(branchterminals, f_mn_tot):
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
    f_mn_tot: numpy.array (shape n,2)
        float, f_mn - [:, 0], f_tot - [:, 1]

    Returns
    -------
    tuple
        * numpy.array, complex, y_mn, longitudinal admittance, y_mn
        * numpy.array, complex, y_tot, diagonal admittance ymn + y_mm"""
    y_tr = branchterminals.y_tr_half.to_numpy().reshape(-1,1)
    y_mn = branchterminals.y_lo.to_numpy().reshape(-1,1)
    y_tot = y_tr + y_mn
    return y_mn * f_mn_tot[:,[0]], y_tot * f_mn_tot[:,[1]]

    # y_mn_tot = branchterminals[['y_lo', 'y_tr_half']].to_numpy()
    # y_mn_tot[:,1] += y_mn_tot[:,0]
    # return y_mn_tot * f_mn_tot


def create_y(branchterminals, count_of_nodes, f_mn_tot):
    """Generates the branch-admittance matrix.

    Parameters
    ----------
    branchterminals: pandas.DataFrame (index_of_terminal)
        * .g_lo, float, longitudinal conductance
        * .b_lo, float, longitudinal susceptance
        * .g_tr_half, float, transversal conductance
        * .b_tr_half, float, transversal susceptance
        * .index_of_node, int
        * .index_of_other_node, int
    count_of_nodes: int
        number of power flow calculation nodes
    f_mn_tot: numpy.array (shape n,2)
        float, f_mn - [:, 0], f_tot - [:, 1]

    Returns
    -------
    tuple
        * sparse matrix of branch admittances Y"""
    index_of_node = branchterminals.index_of_node
    index_of_other_node = branchterminals.index_of_other_node
    row = np.concatenate([index_of_node, index_of_node])
    col = np.concatenate([index_of_node, index_of_other_node])
    y_mn, y_tot = get_y_terms(branchterminals, f_mn_tot)
    yvals = np.concatenate([y_tot.reshape(-1), -y_mn.reshape(-1)])
    shape = count_of_nodes, count_of_nodes
    return coo_matrix((yvals, (row, col)), shape=shape, dtype=np.complex128)

def create_y_matrix(
        branchterminals, count_of_nodes, count_of_slacks, f_mn_tot):
    """Generates the branch-admittance matrix.

    M[n,n] of slack nodes is set to 1, other values of slack nodes are zero.
    Hence, the returned matrix is unsymmetrical.

    Parameters
    ----------
    branchterminals: pandas.DataFrame (index_of_terminal)
        * .g_lo, float, longitudinal conductance
        * .b_lo, float, longitudinal susceptance
        * .g_tr_half, float, transversal conductance
        * .b_tr_half, float, transversal susceptance
        * .index_of_node, int
        * .index_of_other_node, int
    count_of_nodes: int
        number of power-flow-calulation nodes
    count_of_slacks: int
        number of slack busbars
    f_mn_tot: numpy.array (shape n,2)
        float, f_mn - [:, 0], f_tot - [:, 1]

    Returns
    -------
    scipy.sparse.matrix"""
    factors = f_mn_tot[branchterminals.index]
    Y = create_y(branchterminals, count_of_nodes, factors)
    diag = diags(
        [1.+0.j],
        shape=(count_of_slacks, count_of_nodes),
        dtype=np.complex128)
    return vstack([diag.tocsc(), Y.tocsc()[count_of_slacks:, :]])

def create_y_matrix2(
        branchterminals, count_of_nodes, count_of_slacks, f_mn_tot):
    """Generates admittance matrix of branches without rows for slacks.

    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    f_mn_tot: numpy.array (shape n,2)
        float, f_mn - [:, 0], f_tot - [:, 1]

    Returns
    -------
    scipy.sparse.matrix"""
    factors = f_mn_tot[branchterminals.index]
    Y = create_y(branchterminals, count_of_nodes, factors)
    return Y.tocsc()[count_of_slacks:, :]

def get_injected_power_per_injection(calculate_injected_power, Vinj):
    """Calculates active and reactive power for each injection.

    Powers are calculated for one phase of a balanced three phase system.

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

def get_branch_admittance_matrices(y_lo, y_tot, terminal_to_branch):
    """Creates a 2x2 branch-admittance matrix for each branch.

    Parameters
    ----------
    y_lo: numpy.array, complex
        y_mn admittance, per branch
    y_tot: numpy.array, complex
        (y_mn + y_mm) admittance, per branch
    terminal_to_branch: numpy.array, (shape 2,n)
        int indices of terminal per branch, br is index of branch
        * [0, br] index of terminal A
        * [1, br] index of terminal B

    Returns
    -------
    numpy.darray, complex, shape=(n, 2, 2)"""
    # for terminals A: terminal_to_branch[0]
    y_tot_A = y_tot[terminal_to_branch[0]].reshape(-1, 1)
    # for terminals B: terminal_to_branch[1]
    y_tot_B = y_tot[terminal_to_branch[1]].reshape(-1, 1)
    y_lo_AB = y_lo[terminal_to_branch[0]].reshape(-1, 1)
    y_11 = y_tot_A
    y_12 = -y_lo_AB
    y_21 = -y_lo_AB
    y_22 = y_tot_B
    return np.hstack([y_11, y_12, y_21, y_22]).reshape(-1, 2, 2)

def get_y_branches(terms, terminal_to_branch, f_mn_tot):
    """Creates one admittance matrix per branch.

    Parameters
    ----------
    terms: pandas.DataFrame

    terminal_to_branch: numpy.array, (shape 2,n)
        int indices of terminal per branch, br is index of branch
        * [0, br] index of terminal A
        * [1, br] index of terminal B
    f_mn_tot: numpy.array (shape n,2)
        for each terminal
        float, f_mn - [:, 0], f_tot - [:, 1]

    Returns
    -------
    numpy.array, complex, shape=(n, 2, 2)"""
    y_lo, y_tot = get_y_terms(terms, f_mn_tot)
    return get_branch_admittance_matrices(y_lo, y_tot, terminal_to_branch)

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
    Vterm = voltages[terms.index_of_node]
    Votherterm = voltages[terms.index_of_other_node]
    return np.hstack([Vterm, Votherterm]).reshape(-1, 2, 1)

def calculate_branch_results(model, Vnode, positions):
    """Calculates P, Q per branch terminal. Calculates Ploss, Qloss per branch.

    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    Vnode: numpy.array, complex
        voltages at nodes
    positions : array_like | None
        float, tap positions, one value for each row in terminalfactors,
        ordered according to model.factors.terminalfactors

    Returns
    -------
    pandas.DataFrame
        id, I0_pu, I1_pu, P0_pu, Q0_pu, P1_pu, Q1_pu, Ploss_pu, Qloss_pu,
        I0cx_pu, I1cx_pu, V0cx_pu, V1cx_pu, V0_pu, V1_pu, Tap0, Tap1"""
    branchterminals = model.branchterminals
    side_a = branchterminals.side_a
    branchterminals_a = branchterminals[side_a]
    count_of_terminals = len(branchterminals)
    terminalfactors = model.factors.terminalfactors
    if positions is None:
        posbr = np.full((count_of_terminals//2, 2), np.nan, dtype=float)
    else:
        posbr = (
            values_per_terminal(
                np.full((count_of_terminals, 2), np.nan, dtype=float),
                terminalfactors,
                positions)
            [side_a,::-1])
    dposbr = pd.DataFrame(posbr, columns=['Tap0', 'Tap1'])
    term_to_factor = get_term_to_factor_n(terminalfactors, positions)
    f_mn_tot = calculate_f_mn_tot_n(
        count_of_terminals, terminalfactors, term_to_factor)
    factors = f_mn_tot[branchterminals.index]
    Ybr = get_y_branches(branchterminals, model.terminal_to_branch, factors)
    Vbr = get_v_branches(branchterminals_a, Vnode)
    Ibr = Ybr @ Vbr
    # converts from single phase calculation to 3-phase system
    Sbr = 3 * Vbr * Ibr.conjugate()            # S0, S1
    PQbr= Sbr.view(dtype=float).reshape(-1, 4) # P0, P1, Q0, Q1
    Sbr_loss = Sbr.sum(axis=1)
    dfbr = (
        branchterminals_a[['id_of_branch']]
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
    return pd.concat([dfbr, dfres, dfi, dfv, dfv_abs, dposbr], axis=1)

def calculate_results(
        model, Vnode, kpq=None, positions=None, loadcurve='interpolated'):
    """Calculates and arranges electric data of injections and branches.

    Uses a given voltage vector which is typically the result of a power
    flow calculation.

    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    Vnode: array_like, complex
        node voltage vector
    positions : array_like
        optional
        float, tap positions, one value for each row in terminalfactors,
        ordered according to model.factors.terminalfactors
    loadcurve: 'original' | 'interpolated' | 'square'

    Returns
    -------
    dict
        * 'injections': pandas.DataFrame
        * 'branches': pandas.DataFrame"""
    power_fn = get_injected_power_fn(
        model.injections, kpq=kpq, loadcurve=loadcurve)
    return {
        'injections': calculate_injection_results(power_fn, model, Vnode),
        'branches': calculate_branch_results(model, Vnode, positions)}

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
        (voltage_of_nodes) -> (residual_node_current)"""
    terminalfactors = model.factors.terminalfactors
    branchterminals = model.branchterminals.reset_index(drop=True)
    count_of_terminals = len(branchterminals)
    count_of_nodes = model.shape_of_Y[0]
    count_of_slacks = model.count_of_slacks
    def fn(Vnode, positions):
        """Calculates the residual node current.

        Parameters
        ----------
        Vnode: array_like
            complex, voltage at nodes
        positions : array_like | None
            float, tap positions, one value for each row in terminalfactors,
            ordered according to model.factors.terminalfactors

        Returns
        -------
        numpy.array
            complex, residual of node current"""
        term_to_factor = get_term_to_factor_n(
            terminalfactors, positions)
        f_mn_tot = calculate_f_mn_tot_n(
            count_of_terminals, terminalfactors, term_to_factor)
        Y = (
            create_y_matrix(
                branchterminals, count_of_nodes, count_of_slacks, f_mn_tot)
            .tocsc())
        return get_residual_current(model, get_injected_power, Y, Vnode)
    return fn

def get_residual_current_fn2(model, get_injected_power, Vslack=None):
    """Parameterizes function get_residual_current2.

    The returned function calculates the complex residual current per node
    without slack nodes.

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
        (voltage_at_nodes) -> (residual_node_current)"""
    branchterminals = model.branchterminals.reset_index(drop=True)
    count_of_terminals = len(branchterminals)
    count_of_nodes = model.shape_of_Y[0]
    count_of_slacks = model.count_of_slacks
    def fn(Vnode, positions):
        """Calculates the residual node current.

        Parameters
        ----------
        Vnode: array_like
            complex, voltage at nodes
        positions : array_like | None
            float, tap positions, one value for each row in terminalfactors,
            ordered according to model.factors.terminalfactors

        Returns
        -------
        numpy.array
            complex, residual of node current"""
        Vslack_ = model.slacks.V.to_numpy() if Vslack is None else Vslack
        f_mn_tot = get_f_mn_tot_n(
            branchterminals, count_of_terminals, positions)
        Y = (
            create_y_matrix2(
                branchterminals, count_of_nodes, count_of_slacks, f_mn_tot)
            .tocsc())
        return get_residual_current2(
            model, get_injected_power, Vslack_, Y, Vnode)
    return fn

def eval_residual_current(model, get_injected_power, Vnode, positions=None):
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
    positions: numpy.array
        float, tap-position

    Returns
    -------
    numpy.ndarray
        complex, residual node current"""
    return (
        get_residual_current_fn(model, get_injected_power)(Vnode, positions)
        .reshape(-1, 1))

def calculate_residual_current(
        model, /, Vnode, *, positions=None, kpq=None,
        loadcurve='interpolated', vminsqr=_VMINSQR):
    """Calculates residual current per power-flow-calculation node.

    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    Vnode: array_like, complex
        node voltage vector
    positions: numpy.array
        optional, default None
        float, tap-position
    kpq: numpy.array, float, (nx2)
        optional, default None
        scaling factors for active and reactive power
    loadcurve: 'original' | 'interpolated' | 'square'
        optional, default 'interpolated'
    vminsqr: float
        optional, default _VMINSQR
        upper limit of interpolation, interpolates if |V|² < vminsqr

    Returns
    -------
    numpy.ndarray
        complex, residual node current"""
    get_injected_power = get_calc_injected_power_fn(vminsqr,
        model.injections, kpq=kpq, loadcurve=loadcurve)
    return eval_residual_current(
            model, get_injected_power, Vnode, positions)

def max_residual_current(
        model, /, Vnode, *,positions=None, kpq=None,
        loadcurve='interpolated', vminsqr=_VMINSQR):
    """Calculates the maximum of residual current per node.

    Excludes slack nodes. Helper function for testing.

    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    Vnode: array_like, complex
        node voltage vector
    positions: numpy.array
        optional, default None
        float, tap-position
    kpq: numpy.array, float, (nx2)
        optional, default None
        scaling factors for active and reactive power
    loadcurve: 'original' | 'interpolated' | 'square'
        optional, default 'interpolated'
    vminsqr: float
        optional, default _VMINSQR
        upper limit of interpolation, interpolates if |V|² < vminsqr

    Returns
    -------
    numpy.ndarray
        complex, residual node current"""
    return norm(
            calculate_residual_current(
                model, Vnode, positions=positions, kpq=kpq)
            [model.count_of_slacks:],
            np.inf)
