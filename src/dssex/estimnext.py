# -*- coding: utf-8 -*-
"""
Copyright (C) 2023 pyprg

Created on Wed Mar 29 10:20:49 2023

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

@author: pyprg
"""

import casadi

def create_v_symbols_gb_expressions(model, factordefs):
    """Creates symbols for node and slack voltages, tappositions,
    branch conductance/susceptance and an expression for Y @ V. The symbols
    and expressions are regarded constant over multiple calculation steps.
    Diagonal of slack rows are set to 1 for conductance and 0 for susceptance,
    other values of slack rows are set to 0 in matrix 'Y @ V'.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    factordefs: Factordefs
        * .gen_factor_data, pandas.DataFrame (id (str, ID of factor)) ->
            * .step, -1
            * .type, 'const'|'var'
            * .id_of_source, str
            * .value, float
            * .min, float
            * .max, float
            * .is_discrete, bool
            * .m, float
            * .n, float
            * .index_of_symbol, int
        * .gen_factor_symbols, casadi.SX, shape(n,1)
        * .gen_termfactor, pandas.DataFrame (id_of_branch, id_of_node) ->
            * .step
            * .id
            * .index_of_symbol
            * .index_of_terminal
            * .index_of_other_terminal

    Returns
    -------
    dict
        * 'Vnode_syms', casadi.SX, expressions of node Voltages
        * 'Vslack_syms', casadi.SX, symbols of slack voltages,
           Vslack_syms[:,0] real part, Vslack_syms[:,1] imaginary part
        * 'position_syms', casadi.SX, symbols of tap positions
        * 'gb_mn_tot', conductance g / susceptance b per branch terminal
                * gb_mn_tot[:,0] g_mn, mutual conductance
                * gb_mn_tot[:,1] b_mn, mutual susceptance
                * gb_mn_tot[:,2] g_tot, self conductance + mutual conductance
                * gb_mn_tot[:,3] b_tot, self susceptance + mutual susceptance
        * 'Y_by_V', casadi.SX, expression for Y @ V"""
    Vnode_syms = create_V_symbols(model.shape_of_Y[0])
    terms = model.branchterminals

    #+++++++++ new ++++++++++

    # create gb_mn_mm
    #   g_mn, b_mn, g_mm, b_mm
    gb_mn_tot = casadi.SX(
        terms[['g_lo', 'b_lo', 'g_tr_half', 'b_tr_half']].to_numpy())
    # create gb_mn_tot
    #   g_mm + g_mn
    gb_mn_tot[:,2] += gb_mn_tot[:,0]
    #   b_mm + b_mn
    gb_mn_tot[:,3] += gb_mn_tot[:,1]
    if not factordefs.gen_termfactor.empty:
        fmn, ftot = make_fmn_ftot(len(terms), factordefs)
        gb_mn_tot[:, :2] *= fmn
        gb_mn_tot[:, 2:] *= ftot

    #+++++++++ old ++++++++++

    # position_syms = casadi.SX.sym('pos', len(model.branchtaps), 1)
    # gb_mn_tot = _create_gb_mn_tot(terms, model.branchtaps, position_syms)

    #++++++++++++++++++++++++

    G_, B_ = _create_gb_matrix(
        terms.index_of_node,
        terms.index_of_other_node,
        model.shape_of_Y,
        gb_mn_tot)
    count_of_slacks = model.count_of_slacks
    G = _reset_slack_1(G_, count_of_slacks)
    B = _reset_slack_0(B_, count_of_slacks)
    return dict(
        Vnode_syms=Vnode_syms,
        Vslack_syms=casadi.horzcat(
            casadi.SX.sym('Vre_slack', count_of_slacks),
            casadi.SX.sym('Vim_slack', count_of_slacks)),
        #position_syms=factordefs.gen_factor_symbols, #position_syms,
        gb_mn_tot=gb_mn_tot,
        Y_by_V=multiply_Y_by_V(Vnode_syms, G, B))

def get_term_factor_expressions(factordefs):
    """Creates expressions for off-diagonal factors of branches.
    Diagonal factors are just the square of the off-diagonal factors.

    Parameters
    ----------
    branchtaps: pandas.DataFrame (index of taps)
        * .Vstep, float voltage diff per tap
        * .positionneutral, int

    Returns
    -------
    tuple
        * numpy.array, int, indices of terminals
        * numpy.array, int, indices of other terminals
        * casadi.SX"""
    termfactor = (
        factordefs
        .gen_termfactor[
            ['id', 'index_of_terminal', 'index_of_other_terminal']]
        .join(factordefs.gen_factor_data, on='id', how='inner'))
    symbol = factordefs.gen_factor_symbols[termfactor.index_of_symbol]
    return (
        termfactor.index_of_terminal.to_numpy(),
        termfactor.index_of_other_terminal.to_numpy(),
        # y = mx + n
        (casadi.DM(termfactor.m) * symbol) + casadi.DM(termfactor.n))

def make_fmn_ftot(count_of_terminals, factordefs):
    """Creates expressions of terminal factors (taps factors).

    Parameters
    ----------
    count_of_terminals: int
        number of terminals
    factordefs: Factordefs

    Returns
    -------
    tuple
        * casadi.SX, fmn, factor for y_mn admittances
        * casadi.SX, ftot, factors for y_tot admittance"""
    idx_term, idx_otherterm, foffd = get_term_factor_expressions(
        factordefs)
    foffd_term = casadi.SX.ones(count_of_terminals)
    foffd_term[idx_term] = foffd
    foffd_otherterm = casadi.SX.ones(count_of_terminals)
    foffd_otherterm[idx_otherterm] = foffd
    return (foffd_term * foffd_otherterm), (foffd_term * foffd_term)


#====================================================
# from dssex.estim import (
#   create_V_symbols, _create_gb_mn_tot, _create_gb_matrix,
#   _reset_slack_1, _reset_slack_0, multiply_Y_by_V)


def get_tap_factors(branchtaps, position_syms):
    """Creates expressions for off-diagonal factors of branches.
    Diagonal factors are just the square of the off-diagonal factors.

    Parameters
    ----------
    branchtaps: pandas.DataFrame (index of taps)
        * .Vstep, float voltage diff per tap
        * .positionneutral, int
    position_syms: casadi.SX
        vector of position symbols for terms with tap

    Returns
    -------
    casadi.SX"""
    if position_syms.size1():
        return (
            1
            - casadi.SX(branchtaps.Vstep)
              * (position_syms - branchtaps.positionneutral))
    else:
        return casadi.SX(0, 1)

def create_V_symbols(count_of_nodes):
    """Creates variables for node voltages and expressions for Vre**2+Vim**2.

    Parameters
    ----------
    count_of_nodes: int
        number of pfc-nodes

    Returns
    -------
    casadi.SX
        matrix shape=(number_of_nodes, 3)
        * [:,0] Vre, real part of complex voltage
        * [:,1] Vim, imaginary part of complex voltage
        * [:,2] Vre**2 + Vim**2"""
    Vre = casadi.SX.sym('Vre', count_of_nodes)
    Vim = casadi.SX.sym('Vim', count_of_nodes)
    return casadi.horzcat(Vre, Vim, Vre.constpow(2)+Vim.constpow(2))

def _create_gb_mn_tot(branchterminals, branchtaps, position_syms):
    """Creates g_mn, b_mn, g_mm, b_mm (mutual and self conductance
    and susceptance) for each terminal taking tappositions into consideration.

    Parameters
    ----------
    branchterminals: pandas.DataFrame (index of terminal)
        * .glo, mutual conductance, longitudinal
        * .blo, mutual susceptance, longitudinal
        * .g_tr_half, half of self conductance, transversal
        * .b_tr_half, half of self susceptance, transversal
    branchtaps: pandas.DataFrame (index of taps)
        * .Vstep, float voltage diff per tap
        * .positionneutral, int
    position_syms: casadi.SX
        vector of position symbols for terms with taps (index of taps)

    Returns
    -------
    casadi.SX
        * [:,0] g_mn, mutual conductance
        * [:,1] b_mn, mutual susceptance
        * [:,2] g_tot, self conductance + mutual conductance
        * [:,3] b_tot, self susceptance + mutual susceptance"""
    # create gb_mn_mm
    #   g_mn, b_mn, g_mm, b_mm
    gb_mn_tot = casadi.SX(
        branchterminals[['g_lo', 'b_lo', 'g_tr_half', 'b_tr_half']]
        .to_numpy())
    # create gb_mn_tot
    #   g_mm + g_mn
    gb_mn_tot[:,2] += gb_mn_tot[:,0]
    #   b_mm + b_mn
    gb_mn_tot[:,3] += gb_mn_tot[:,1]
    if position_syms.size1():
        foffd = get_tap_factors(branchtaps, position_syms)
        count_of_rows = gb_mn_tot.size1()
        foffd_term = casadi.SX.ones(count_of_rows)
        foffd_term[branchtaps.index_of_terminal] = foffd
        foffd_otherterm = casadi.SX.ones(count_of_rows)
        foffd_otherterm[branchtaps.index_of_other_terminal] = foffd
        gb_mn_tot[:, :2] *= (foffd_term * foffd_otherterm)
        gb_mn_tot[:, 2:] *= (foffd_term * foffd_term)
    return gb_mn_tot

def _create_gb_matrix(index_of_node, index_of_other_node, shape, gb_mn_tot):
    """Creates conductance matrix G and susceptance matrix B.

    Parameters
    ----------
    index_of_node: array_like
        int, branchterminal -> index_of_node
    index_of_other_node
        int, branchterminal -> index_of_other_node (same branch, other side)
    shape: tuple (int,int)
        shape of branch admittance matrix
    gb_mn_mm: casadi.SX
        matrix of 4 column vectors, one row for each branch terminal
        [:,0] g_mn, mutual conductance
        [:,1] b_mn, mutual susceptance
        [:,2] g_tot, self conductance + mutual conductance
        [:,3] b_tot, self susceptance + mutual susceptance

    Returns
    -------
    tuple
        * casadi.SX - conductance matrix
        * casadi.SX - susceptance matrix"""
    G = casadi.SX(*shape)
    B = casadi.SX(G)
    for r, c, gb in zip(
        index_of_node, index_of_other_node, casadi.vertsplit(gb_mn_tot, 1)):
        G[r,c] -= gb[0] # g_mn
        B[r,c] -= gb[1] # b_mn
        G[r,r] += gb[2] # g_tot, add to diagonal
        B[r,r] += gb[3] # b_tot, add to diagonal
    return G, B

def _reset_slack_0(matrix, count_of_rows):
    """Removes entries from first count_of_slacks rows of matrix.
    Returns new casadi.SX instance.

    Parameters
    ----------
    matrix: casadi.SX
        input matrix
    count_of_rows: int
        number of rows to be processed

    Returns
    -------
    casadi.SX"""
    empty = casadi.SX(count_of_rows, matrix.size2())
    return casadi.vertcat(empty, matrix[count_of_rows:, :])

def _reset_slack_1(matrix, count_of_rows):
    """Removes entries from first count_of_slacks rows of matrix, sets diagonal
    entries of first count_of_rows rows to 1.0. Returns new casadi.SX instance.

    Parameters
    ----------
    matrix: casadi.SX
        input matrix
    count_of_rows: int
        number of rows to be processed

    Returns
    -------
    casadi.SX"""
    diag = casadi.Sparsity.diag(count_of_rows, matrix.size2())
    return casadi.vertcat(diag, matrix[count_of_rows:, :])

def multiply_Y_by_V(Vreim, G, B):
    """Creates the Y-matrix from G and B, creates a vector from Vreim.
    Multiplies Y @ V.

    Parameters
    ----------
    Vreim: casadi.SX
        two vectors
        * Vreim[:,0] Vre, real part of complex voltage
        * Vreim[:,1] Vim, imaginary part of complex voltage
    G: casadi.SX
        conductance matrix
    B: casadi.SX
        susceptance matrix

    Returns
    -------
    casadi.SX"""
    V_node = casadi.vertcat(Vreim[:,0], Vreim[:,1])
    return casadi.blockcat([[G, -B], [B,  G]]) @ V_node


#====================================================