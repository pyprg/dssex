# -*- coding: utf-8 -*-
"""
Utilities for dssex.

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

Created on Mon May  23 21:53:10 2021

@author: pyprg
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags, vstack
from functools import partial

def get_tap_factors(branchtaps, pos):
    """Creates factors for tap positions, values for longitudinal and
    transversal factors of branches.
    
    Parameters
    ----------
    branchtaps: pandas.DataFrame (id of taps)
        * .Vstep, float voltage diff per tap
        * .positionneutral, int
    pos: array_like, int
        vector of positions for terms with tap
    
    Returns
    -------
    tuple
        * numpy array, float, longitudinal factors
        * numpy array, float, transversal factors"""
    # factor longitudinal
    flo = (1 - branchtaps.Vstep.to_numpy() * (
        pos - branchtaps.positionneutral.to_numpy()))
    return flo, np.power(flo, 2)

def get_y_terms(terms, flo, ftr):
    """Multiplies admittances of branches with factors retrieved
    from tap positions.
    
    Parameters
    ----------
    terms: pandas.DataFrame
    
    flo: pandas.Series, float
        factor for longitudinal admittance
    ftr: pandas.Series, float
        factor transversal admittance
    
    Returns
    -------
    tuple
        * numpy.array, complex, y_tr, transversal admittance
        * numpy.array, complex, y_lo, longitudinal admittance"""
    terms_with_taps = terms[terms.index_of_taps.notna()]
    idx_of_tap = terms_with_taps.index_of_taps
    y_tr = terms.y_tr_half.to_numpy()
    y_tr[terms_with_taps.index] *= ftr[idx_of_tap]   
    y_lo = terms.y_lo.to_numpy()
    y_lo[terms_with_taps.index] *= flo[idx_of_tap]
    terms_with_other_taps = terms[terms.index_of_other_taps.notna()]
    idx_of_other_tap = terms_with_other_taps.index_of_other_taps
    y_lo[terms_with_other_taps.index] *= flo[idx_of_other_tap]
    return y_lo, y_tr

def create_y(terms, count_of_nodes, flo, ftr):
    """Generates the branch-admittance matrix. 
    
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
        * sparse matrix of branch admittances Y"""
    index_of_node = terms.index_of_node
    index_of_other_node = terms.index_of_other_node
    row = np.concatenate([index_of_node, index_of_node])
    col = np.concatenate([index_of_node, index_of_other_node])
    rowcol = row, col
    y_lo, y_tr = get_y_terms(terms, flo, ftr)
    yvals = np.concatenate([(y_tr + y_lo), -y_lo])
    shape = count_of_nodes, count_of_nodes
    return coo_matrix((yvals, rowcol), shape=shape, dtype=np.complex128)

def create_y_matrix(model, pos):
    """Generates admittance matrix of branches. 
    M[n,n] of slack nodes is set to 1, other
    values of slack nodes are zero. Hence, the returned 
    matrix is unsymmetrical.
    
    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    pos: numpy.array, int
        vector of position, one variable for each terminal with taps
    
    Returns
    -------
    scipy.sparse.matrix"""
    flo, ftr = get_tap_factors(model.branchtaps, pos)
    count_of_nodes = model.shape_of_Y[0]
    terms = (
        model.branchterminals[~model.branchterminals.is_bridge].reset_index())
    Y = create_y(terms, count_of_nodes, flo, ftr)
    count_of_slacks = model.count_of_slacks
    diag = diags(
        [1.+0.j] * count_of_slacks, 
        shape=(count_of_slacks, count_of_nodes), 
        dtype=np.complex128)
    return vstack([diag.tocsc(), Y.tocsc()[count_of_slacks:, :]])

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
        * numpy.array, float, voltage at per injection"""
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
    # injected current is negative for positve power
    return -np.conjugate((model.mnodeinj @ Sinj) / Vnode)

def get_injection_results(calculate_injected_power, model, Vnode):
    """Returns active and reactive power in pu for given node voltages.
    
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
    pandas.DataFrame"""
    df = model.injections.loc[
        :, ['id', 'Exp_v_p', 'Exp_v_q', 'P10', 'Q10']]
    df['P_pu'], df['Q_pu'], Vinj = get_injected_power_per_injection(
        calculate_injected_power, model.mnodeinj.T @ Vnode)
    df['V_pu'] = np.abs(Vinj)
    df['P_pu'] *= 3
    df['Q_pu'] *= 3
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

def get_branch_admittance_matrices(y_tr, y_lo, term_is_at_A):
    """Creates a 2x2 branch-admittance matrix for each branch.
    
    Parameters
    ----------
    y_tr: numpy.array, complex
        transversal admittance according to pi-equivalent circuit, per branch
    y_lo: numpy.array, complex
        longitudinal admittance according to pi-equivalent circuit, per branch
    term_is_at_A: numpy.array, bool, index of terminal
        True if terminal with is at side A of a branch
        
    Returns
    -------
    numpy.darray, complex, shape=(:, 2, 2)"""
    y_tr_A = y_tr[term_is_at_A]
    y_tr_B = y_tr[~term_is_at_A]
    y_lo_AB = y_lo[term_is_at_A]
    y_11 = y_tr_A + y_lo_AB
    y_12 = -y_lo_AB
    y_21 = -y_lo_AB
    y_22 = y_tr_B + y_lo_AB
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
       True if terminal with is at side A of a branch
    pos: pandas.Series
        int, positions of taps
        
    Returns
    -------
    numpy.array, complex, shape=(:, 2, 2)"""
    flo, ftr = get_tap_factors(model.branchtaps, pos)
    y_tr, y_lo = get_y_terms(terms, flo, ftr)
    return get_branch_admittance_matrices(y_tr, y_lo, term_is_at_A)

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

def get_branch_results(model, Vnode, pos):
    """Calculates P, Q per branch terminal. Calculates Ploss, Qloss per branch.
    
    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    Vnode: numpy.array, complex
        voltages at nodes
    
    Returns
    -------
    pandas.DataFrame
        I0_pu, I1_pu, P0_pu, Q0_pu, P1_pu, Q1_pu, Ploss_pu, Qloss_pu, V0_pu, 
        V1_pu"""
    branchterminals = model.branchterminals
    terms = branchterminals[(~branchterminals.is_bridge)].reset_index()
    term_is_at_A = terms.side == 'A'
    Ybr = get_y_branches(model, terms, term_is_at_A, pos)
    Vbr = get_v_branches(terms[term_is_at_A], Vnode)
    Ibr = Ybr @ Vbr
    Sbr = Vbr * Ibr.conjugate()
    PQbr= Sbr.view(dtype=float).reshape(-1, 4)
    Sbr_loss = Sbr.sum(axis=1)
    dfbr = (
        terms.loc[term_is_at_A, ['id_of_branch']]
        .rename(columns={'id_of_branch': 'id'}))
    dfv = pd.DataFrame(
        Vbr.reshape(-1, 2),
        columns=['V0_pu', 'V1_pu'])
    res = np.hstack(
        [np.abs(Ibr).reshape(-1,2), PQbr, Sbr_loss.view(dtype=float)])
    dfres = pd.DataFrame(
        res, 
        columns=[
            'I0_pu', 'I1_pu', 'P0_pu', 'Q0_pu', 'P1_pu', 'Q1_pu', 
            'Ploss_pu', 'Qloss_pu'])
    return pd.concat([dfbr, dfres, dfv], axis=1)

def get_results(model, get_injected_power, tappositions, Vnode):
    """Calcualtes and arranges electric data of injections and branches
    for a given voltage vector which is typically the result of a power
    flow calculation.
    
    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    get_injected_power: function 
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
    injections = get_injection_results(get_injected_power, model, Vnode)
    branches = get_branch_results(model, Vnode, tappositions)
    return {'injections': injections, 'branches': branches}

#
# root finding with slack data in the admittance matrix
#

def get_residual_current(model, get_injected_power, Y, Vnode):
    """Calculates the complex residual current per node.
    
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
        complex, residual of node current"""
    V_ =  Vnode.reshape(-1, 1)
    Inode = Y @ V_
    Iinj = get_injected_current_per_node(get_injected_power, model, V_)
    return (Inode - Iinj).reshape(-1)

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

#
# root finding without slack data in the admittance matrix
#

def create_y_matrix2(model, pos):
    """Generates admittance matrix of branches without rows for slacks. 
    Should return a symmetric matrix.
    
    Parameters
    ----------
    model: egrid.model.Model
        model of grid for calculation
    pos: numpy.array, int
        vector of position, one variable for each terminal with taps
    
    Returns
    -------
    scipy.sparse.matrix"""
    flo, ftr = get_tap_factors(model.branchtaps, pos)
    count_of_nodes = model.shape_of_Y[0]
    terms = (
        model.branchterminals[~model.branchterminals.is_bridge].reset_index())
    Y = create_y(terms, count_of_nodes, flo, ftr)
    count_of_slacks = model.count_of_slacks
    return Y.tocsc()[count_of_slacks:, :]

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
    return (Inode - Iinj).reshape(-1)

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
        (voltage_of_nodes) -> (current_of_node)"""
    tappositions_ = model.branchtaps.position \
        if tappositions is None else tappositions
    Vslack_ = model.slacks.V.to_numpy() if Vslack is None else Vslack
    Y = create_y_matrix2(model, tappositions_).tocsc()
    return partial(
        get_residual_current2, model, get_injected_power, Vslack_, Y)