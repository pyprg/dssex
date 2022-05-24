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
from scipy.sparse import coo_matrix

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

def get_injection_results(calculate_injected_power, model, voltages):
    """Returns active and reactive power in pu for given node voltages.
    
    Parameters
    ----------
    calculate_injected_power: function
        (injections, square of absolute voltages at injection terminals) ->
            (active power, reactive power)
    model: egrid.model.Model
        data of the electric power network
    voltages: array_like, complex
        vector of node voltages
        
    Returns
    -------    
    pandas.DataFrame"""
    Vinj = model.mnodeinj.T @ voltages
    Vinj_abs_sqr = np.power(np.real(Vinj), 2) + np.power(np.imag(Vinj), 2)
    df = model.injections.loc[
        :, ['id', 'Exp_v_p', 'Exp_v_q', 'P10', 'Q10']]
    df['P_pu'], df['Q_pu'] = calculate_injected_power(
        model.injections, Vinj_abs_sqr)
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

def get_y_terms(terms, flo, ftr):
    """Multiplies admittances of branches with factors retrieved
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
    y_tr = terms.y_tr_half.to_numpy()
    y_tr[terms_with_taps.index] *= ftr[idx_of_tap]   
    y_lo = terms.g_lo.to_numpy()
    y_lo[terms_with_taps.index] *= flo[idx_of_tap]
    terms_with_other_taps = terms[terms.index_of_other_taps.notna()]
    idx_of_other_tap = terms_with_other_taps.index_of_other_taps
    y_lo[terms_with_other_taps.index] *= flo[idx_of_other_tap]
    return y_tr, y_lo

def get_y_branches(model, terms, term_is_at_A, pos):
    """Creates one admittance matrix per branch.
    
    Parameters
    ----------
    model: egrid.model.Model

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

def get_branch_results(model, voltages, pos):
    """Calculates P, Q per branch terminal. Calculates Ploss, Qloss per branch.
    
    Parameters
    ----------
    model: egrid.model.Model
    
    voltages: numpy.array, complex
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
    Vbr = get_v_branches(terms[term_is_at_A], voltages)
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

def get_results(model, get_injected_power, pos, V):
    """Calcualtes and arranges electric data of injections and branches
    for a given voltage vector which is typically the result of a power
    flow calculation.
    
    Parameters
    ----------
    model: egrid.model.Model

    get_injected_power: function

    pos: array_like, int
        positions of taps
    V: array_like, complex
        node voltage vector
    
    Returns
    -------
    dict
        * 'injections': pandas.DataFrame
        * 'branches': pandas.DataFrame"""
    injections = get_injection_results(get_injected_power, model, V)
    branches = get_branch_results(model, V, model.branchtaps.position)
    return {'injections': injections, 'branches': branches}
