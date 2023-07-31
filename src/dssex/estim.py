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

Created on Sat Sep 10 11:28:52 2022

@author: pyprg
"""
import casadi
import pandas as pd
import numpy as np
from functools import partial
from collections import defaultdict
from scipy.sparse import coo_matrix
from egrid.model import get_vlimits_for_step, get_terms_for_step
from dssex.injections import calculate_cubic_coefficients
from dssex.batch import (
    get_values, get_batches, value_of_voltages, get_batch_values)
import dssex.factors as ft
# square of voltage magnitude, default value, minimum value for load curve,
#   if value is below _VMINSQR the load curves for P and Q converge
#   towards a linear load curve which is 0 when V=0; P(V=0)=0, Q(V=0)=0
_VMINSQR = 0.8**2
# value of zero check, used for load curve calculation
_EPSILON = 1e-12
# empty vector of values
_DM_0r1c = casadi.DM(0,1)
_EMPTY_0r1c = np.empty((0,1), dtype=float)
# empty vector of expressions
_SX_0r1c = casadi.SX(0,1)
# options for solver IPOPT
_IPOPT_opts = {'ipopt.print_level':0, 'print_time':0}

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

def _create_gb_expressions(terms):
    g_mn = casadi.SX(terms.g_lo)
    b_mn = casadi.SX(terms.b_lo)
    g_mm = casadi.SX(terms.g_tr_half)
    b_mm = casadi.SX(terms.b_tr_half)
    return casadi.horzcat(g_mn, b_mn, g_mm, b_mm)

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

def create_mapping(from_index, to_index, shape):
    """Creates a matrix M for mapping vectors.

    Parameters
    ----------
    from_index: array_like
        int
    to_index: array_like
        int
    shape: tuple
        int,int (number_of_rows (to), number_of_columns (from))

    Returns
    -------
    scipy.sparse.coo_matrix
        matrix of float 1.0 with shape=(to_index, from_index)"""
    return coo_matrix(
        ([1.]*len(from_index), (to_index, from_index)),
        shape=shape, dtype=float)

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

#
# expressions for injected current
#

def  _calculate_injected_current(Vri, Vabs_sqr, Exp_v, PQscaled):
    """Creates expression for real and imaginary parts of injected current.

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
    Vri: casadi.SX, shape n,2
        voltage at terminals of injections
        Vri[:,0] - real part of voltage
        Vri[:,1] - imaginary part of voltage
    Vabs_sqr: casadi.SX, shape n,1
        square of voltage magnitude at terminals of injections
        Vri[:,0]**2 + Vri[:,1]**2
    Exp_v: numpy.array, shape n,2
        voltage exponents of active and reactive power
    PQscaled: casadi.SX, shape n,2
        active and reactive power at nominal voltage multiplied
        by scaling factors
        PQscaled[:,0] - active power
        PQscaled[:,1] - reactive power

    Returns
    -------
    casadi.SX, shape n,2
        [:,0] - real part of injected current
        [:,1] - imaginary part of injected current"""
    ypq = casadi.power(Vabs_sqr, (Exp_v/2)-1) * PQscaled
    Ire = casadi.sum2(ypq * Vri)
    Iim = -ypq[:,1]*Vri[:,0] + ypq[:,0]*Vri[:,1]
    return casadi.horzcat(Ire, Iim)

def _get_injectiondata(
        injections, node_to_inj, Vnode, factordata, vminsqr=_VMINSQR):
    """Creates expressions for injection processing.

    Also returns intermediate expressions used for calculation of
    current magnitude, active/reactive power flow.

    Parameters
    ----------
    injections: pandas.DataFrame (int index_of_injection)
        * .P10, float, rated active power at voltage of 1.0 pu
        * .Q10, float, rated reactive power at voltage of 1.0 pu
        * .Exp_v_p, float, voltage exponent of active power
        * .Exp_v_q, float, voltage exponent of reactive power
    node_to_inj: casadi.SX
        the matrix converts node to injection values
        injection_values = node_to_inj @ node_values
    Vnode: casadi.SX
        three vectors of node voltage expressions
        * Vnode[:,0], float, Vre vector of real node voltages
        * Vnode[:,1], float, Vim vector of imaginary node voltages
        * Vnode[:,2], float, Vre**2 + Vim**2
    factordata: Factordata
        * .kpq, casadi.SX expression, vector of injection scaling factors for
          active and reactive power
    vminsqr: float
        square of voltage, upper limit interpolation interval [0...vminsqr]

    Returns
    -------
    casadi.SX (n,8)
        [:,0] Ire, current, real part
        [:,1] Iim, current, imaginary part
        [:,2] Pscaled, active power P10 multiplied by scaling factor kp
        [:,3] Qscaled, reactive power Q10 multiplied by scaling factor kq
        [:,4] Pip, active power interpolated
        [:,5] Qip, reactive power interpolated
        [:,6] Vabs_sqr, square of voltage magnitude
        [:,7] interpolate?"""
    # voltages at injections
    Vinj = node_to_inj @ Vnode
    Vinj_abs_sqr = Vinj[:, 2]
    Vinj_abs = casadi.sqrt(Vinj_abs_sqr)
    # assumes P10 and Q10 are sums of 3 per-phase-values
    PQscaled = factordata.kpq * (injections[['P10', 'Q10']].to_numpy() / 3)
    # voltage exponents
    Exp_v = injections[['Exp_v_p', 'Exp_v_q']].to_numpy()
    # interpolated P and Q
    cpq = calculate_cubic_coefficients(vminsqr, Exp_v)
    V_321 = casadi.vertcat(
        (Vinj_abs_sqr * Vinj_abs).T, Vinj_abs_sqr.T, Vinj_abs.T)
    Pip_ = casadi.vcat(
        [(casadi.SX(cpq[row_index,:,0].reshape(1,-1)) @ V_321[:,row_index])
         for row_index in range(V_321.size2())])
    Qip_ = casadi.vcat(
        [(casadi.SX(cpq[row_index,:,1].reshape(1,-1)) @ V_321[:,row_index])
         for row_index in range(V_321.size2())])
    if PQscaled.size1():
        Pip = Pip_ * PQscaled[:,0]
        Qip = Qip_ * PQscaled[:,1]
        # current according to given load curve
        I_orig = _calculate_injected_current(
            Vinj[:,:2], Vinj_abs_sqr, Exp_v, PQscaled)
    else:
        Pip = _SX_0r1c
        Qip = _SX_0r1c
        I_orig = casadi.SX(0,2)
    # interpolated current
    calculate = _EPSILON < Vinj_abs_sqr
    Vre = Vinj[:, 0]
    Vim = Vinj[:, 1]
    Ire_ip = casadi.if_else(
        calculate, (Pip * Vre + Qip * Vim) / Vinj_abs_sqr, 0.0)
    Iim_ip = casadi.if_else(
        calculate, (-Qip * Vre + Pip * Vim) / Vinj_abs_sqr, 0.0)
    # compose current expressions
    interpolate = Vinj_abs_sqr < vminsqr
    Ire = casadi.if_else(interpolate, Ire_ip, I_orig[:,0])
    Iim = casadi.if_else(interpolate, Iim_ip, I_orig[:,1])
    return casadi.horzcat(
        Ire, Iim, PQscaled, Pip, Qip, Vinj_abs_sqr, interpolate)

def _reset_slack_current(
        slack_indices, Vre_slack_syms, Vim_slack_syms, Iinj_ri):
    Iinj_ri[slack_indices,0] = Vre_slack_syms # real(Iinj)
    Iinj_ri[slack_indices,1] = Vim_slack_syms # imag(Iinj)
    return Iinj_ri

def get_term_factor_expressions(terminalfactors, gen_factor_symbols):
    """Creates expressions for off-diagonal factors of branches.

    Diagonal factors are just the square of the off-diagonal factors.

    Parameters
    ----------
    terminalfactors: pandas.DataFrame

    gen_factor_symbols: casadi.SX

    Returns
    -------
    tuple
        * numpy.array, int, indices of terminals
        * numpy.array, int, indices of other terminals
        * casadi.SX"""
    symbol = gen_factor_symbols[terminalfactors.index_of_symbol]
    return (
        terminalfactors.index_of_terminal.to_numpy(),
        terminalfactors.index_of_other_terminal.to_numpy(),
        # y = mx + n
        (casadi.DM(terminalfactors.m) * symbol) + casadi.DM(terminalfactors.n))

def make_fmn_ftot(count_of_terminals, terminalfactors, gen_factor_symbols):
    """Creates expressions of terminal factors (taps factors).

    Parameters
    ----------
    count_of_terminals: int
        number of terminals
    terminalfactors: pandas.DataFrame

    Returns
    -------
    tuple
        * casadi.SX, fmn, factor for y_mn admittances
        * casadi.SX, ftot, factors for y_tot admittance"""
    idx_term, idx_otherterm, foffd = get_term_factor_expressions(
        terminalfactors, gen_factor_symbols)
    foffd_term = casadi.SX.ones(count_of_terminals)
    foffd_term[idx_term] = foffd
    foffd_otherterm = casadi.SX.ones(count_of_terminals)
    foffd_otherterm[idx_otherterm] = foffd
    return (foffd_term * foffd_otherterm), (foffd_term * foffd_term)

def _create_gb_mn_tot_terms(terminals, terminalfactors, gen_factor_symbols):
    """Creates conductance and  susceptance expressions for each terminal.

    Parameters
    ----------
    terminals: pandas.DataFrame

    terminalfactors: pandas.DataFrame

    gen_factor_symbols: casadi.SX, shape(m,1)
        symbols of generic decision variables or generic parameters,
        generic variables/parameters are present in each optimization step,
        they are identical in each step, whereas the collection of
        step-specific variables and parameters change from step to step

    Returns
    -------
    casadi.SX
        * 'gb_mn_tot', conductance g / susceptance b per branch terminal
                * gb_mn_tot[:,0] g_mn, mutual conductance
                * gb_mn_tot[:,1] b_mn, mutual susceptance
                * gb_mn_tot[:,2] g_tot, self conductance + mutual conductance
                * gb_mn_tot[:,3] b_tot, self susceptance + mutual susceptance
    """
    # create gb_mn_mm
    #   g_mn, b_mn, g_mm, b_mm
    gb_mn_tot = casadi.SX(
        terminals[['g_lo', 'b_lo', 'g_tr_half', 'b_tr_half']].to_numpy())
    # create gb_mn_tot
    #   g_mm + g_mn
    gb_mn_tot[:,2] += gb_mn_tot[:,0]
    #   b_mm + b_mn
    gb_mn_tot[:,3] += gb_mn_tot[:,1]
    if not terminalfactors.empty:
        fmn, ftot = make_fmn_ftot(
            len(terminals), terminalfactors, gen_factor_symbols)
        gb_mn_tot[:, :2] *= fmn
        gb_mn_tot[:, 2:] *= ftot
    return gb_mn_tot

def create_v_symbols_gb_expressions(model, gen_factor_symbols):
    """Creates symbols for node and slack voltages. Creates expressions.

    Expressions for branch conductance/susceptance and an expression for
    Y @ V are prepared for further processing. The created symbols
    and expressions are regarded constant over multiple calculation steps.
    Diagonal of slack rows are set to 1 for conductance and 0 for susceptance,
    other values of slack rows are set to 0 in matrix 'Y @ V'.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    Vnode_syms: casadi.SX, shape(n,3)
        * [:,0] Vre, real part of complex voltage
        * [:,1] Vim, imaginary part of complex voltage
        * [:,2] Vre**2 + Vim**2
    gen_factor_symbols: casadi.SX, shape(m,1)
        symbols of generic decision variables or generic parameters,
        generic variables/parameters are present in each optimization step,
        they are identical in each step, whereas the collection of
        step-specific variables and parameters change from step to step

    Returns
    -------
    dict
        * 'Vnode_syms', casadi.SX, expressions of node Voltages
        * 'Vslack_syms', casadi.SX, symbols of slack voltages,
           Vslack_syms[:,0] real part, Vslack_syms[:,1] imaginary part
        * 'gb_mn_tot', conductance g / susceptance b per branch terminal
                * gb_mn_tot[:,0] g_mn, mutual conductance
                * gb_mn_tot[:,1] b_mn, mutual susceptance
                * gb_mn_tot[:,2] g_tot, self conductance + mutual conductance
                * gb_mn_tot[:,3] b_tot, self susceptance + mutual susceptance
        * 'Y_by_V', casadi.SX, expression for Y @ V"""
    Vnode_syms = create_V_symbols(model.shape_of_Y[0])
    count_of_slacks = model.count_of_slacks
    Vslack_syms=casadi.horzcat(
        casadi.SX.sym('Vre_slack', count_of_slacks),
        casadi.SX.sym('Vim_slack', count_of_slacks))
    terminals = model.branchterminals
    gb_mn_tot = _create_gb_mn_tot_terms(
        terminals, model.factors.terminalfactors, gen_factor_symbols)
    G_, B_ = _create_gb_matrix(
        terminals.index_of_node,
        terminals.index_of_other_node,
        model.shape_of_Y,
        gb_mn_tot)
    G = _reset_slack_1(G_, count_of_slacks)
    B = _reset_slack_0(B_, count_of_slacks)
    return dict(
        Vnode_syms=Vnode_syms,
        Vslack_syms=Vslack_syms,
        gb_mn_tot=gb_mn_tot,
        Y_by_V=multiply_Y_by_V(Vnode_syms, G, B))

#
# power flow calculation
#

def make_get_factor_and_injection_data(
        model, gen_factor_symbols, Vnode_syms, vminsqr):
    """Returns a function creating factordata and injectiondata.

    The data are created by the returned function for a given step, they are
    specific for that step.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    Vnode_syms: casadi.SX
        three vectors of node voltage expressions
        * Vnode_syms[:,0], float, Vre vector of real node voltages
        * Vnode_syms[:,1], float, Vim vector of imaginary node voltages
        * Vnode_syms[:,2], float, Vre**2 + Vim**2
    vminsqr: float
        square of voltage, upper limit interpolation interval [0...vminsqr]

    Returns
    -------
    function
        (int, casadi.DM)
            -> (tuple: Factordata, casadi.SX)
        (index_of_step, factors_of_previous_step)
            -> (tuple: Factordata, injection_data)
        * Factordata
            kp: casadi.SX
                column vector, symbols for scaling factor of active power
                per injection
            kq: casadi.SX
                column vector, symbols for scaling factor of reactive power
                per injection
            vars: casadi.SX
                column vector, symbols for variables of scaling factors
            values_of_vars: casadi.DM
                column vector, initial values for vars
            var_min: casadi.DM
                lower limits of vars
            var_max: casadi.DM
                upper limits of vars
            consts: casadi.SX
                column vector, symbols for constants of scaling factors
            values_of_consts: casadi.DM
                column vector, values for consts
            symbols: casadi.SX
                vector of all symbols (variables and constants) for
                extracting values which shall be passed to next step
                (function 'get_factordata', parameter 'f_prev')
            values: casadi.DM
                float, given values of symbols (variables and constants)
        * casadi.SX (n,8)
            * [:,0] Ire, injected current, real part
            * [:,1] Iim, injected current, imaginary part
            * [:,2] Pscaled, injected active power P10
                    multiplied by scaling factor kp
            * [:,3] Qscaled, injected reactive power Q10
                    multiplied by scaling factor kq
            * [:,4] Pip, injected active power interpolated
            * [:,5] Qip, injected reactive power interpolated
            * [:,6] Vabs_sqr, square of voltage magnitude at injection
            * [:,7] interpolate?"""
    get_factordata = partial(ft.make_factordata, model, gen_factor_symbols)
    injections = model.injections
    node_to_inj = casadi.SX(model.mnodeinj).T
    def get_factor_and_injection_data(step=0, f_prev=_EMPTY_0r1c):
        factordata = get_factordata(step, f_prev)
        # injected node current
        injectiondata = _get_injectiondata(
            injections, node_to_inj, Vnode_syms, factordata, vminsqr)
        return factordata, injectiondata
    return get_factor_and_injection_data

def get_expressions(model, gen_factor_symbols, vminsqr=_VMINSQR):
    """Prepares data for estimation. Creates symbols and expressions.

    Parameters
    ----------
    model : egrid.model.Model
        data of electric grid
    factordefs: Factordefs
        * .gen_factordata, pandas.DataFrame
        * .gen_injfactor, pandas.DataFrame
        * .terminalfactors, pandas.DataFrame
        * .factorgroups: function
            (iterable_of_int)-> (pandas.DataFrame)
        * .injfactorgroups: function
            (iterable_of_int)-> (pandas.DataFrame)
    vminsqr : float, optional
        minimum voltage at loads for original load curve, squared

    Returns
    -------
    dict
        * 'Vnode_syms', casadi.SX, expressions of node Voltages
        * 'Vslack_syms', casadi.SX, symbols of slack voltages,
           Vslack_syms[:,0] real part, Vslack_syms[:,1] imaginary part
        * 'gb_mn_tot', conductance g / susceptance b per branch terminal
            * gb_mn_tot[:,0] g_mn, mutual conductance
            * gb_mn_tot[:,1] b_mn, mutual susceptance
            * gb_mn_tot[:,2] g_tot, self conductance + mutual conductance
            * gb_mn_tot[:,3] b_tot, self susceptance + mutual susceptance
        * 'Y_by_V', casadi.SX, expression for Y @ V
        * 'get_factor_and_injection_data', function
          (int, casadi.DM) -> (tuple - Factordata, casadi.SX)
          which is a function
          (index_of_step, scaling_factors_of_previous_step)
            -> (tuple - Factordata, injection_data)
        * 'inj_to_node', casadi.SX, matrix, maps from
          injections to power flow calculation nodes"""
    ed = create_v_symbols_gb_expressions(model, gen_factor_symbols)
    ed['get_factor_and_injection_data'] = (
        make_get_factor_and_injection_data(
            model, gen_factor_symbols, ed['Vnode_syms'], vminsqr))
    ed['inj_to_node'] = casadi.SX(model.mnodeinj)
    return ed

def calculate_power_flow2(model, expr, factordata, Inode, Vinit=None):
    """Solves the power flow problem using a rootfinding algorithm.

    The result can be the initial voltage vector for an optimization.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    expr: dict
        * 'Vnode_syms', casadi.SX, expressions of node Voltages
        * 'Vre_slack_syms', casadi.SX, symbols of slack voltages, real part
        * 'Vim_slack_syms', casadi.SX, symbols of slack voltages,
           imaginary part
        * 'Y_by_V', casadi.SX, expression for Y @ V
    factordata: Factordata
        * .vars, casadi.SX, symbols of factors (decision variables)
        * .values_of_vars, casadi.DM, initial values for vars
        * .var_min, casadi.DM, lower limits of vars
        * .var_max, casadi.DM, upper limits of vars
        * .is_discrete, numpy.array, bool, flag for variable
        * .consts, casadi.SX, symbols of constant factors
        * .values_of_consts, casadi.DM, values for parameters ('consts')
    Inode: casadi.SX (shape n,2)
        expressions for injected current per node
        values for slack need to be set to slack voltage
        * Inode[:,0] - Ire, real part of current injected into node
        * Inode[:,1] - Iim, imaginary part of current injected into node
    Vinit: array_like
        optional
        float, initial guess of node voltages

    Returns
    -------
    tuple
        * bool, success?
        * casadi.DM, float, voltage vector [real parts, imaginary parts]"""
    Vslack_syms = expr['Vslack_syms'][:,0], expr['Vslack_syms'][:,1]
    parameter_syms=casadi.vertcat(
        *Vslack_syms,
        factordata.vars,
        factordata.consts)
    Vnode_syms = expr['Vnode_syms']
    Inode_ri = vstack(Inode, 2)
    variable_syms = vstack(Vnode_syms, 2)
    fn_Iresidual = casadi.Function(
        'fn_Iresidual',
        [variable_syms, parameter_syms],
        [expr['Y_by_V'] + Inode_ri])
    rf = casadi.rootfinder('rf', 'nlpsol', fn_Iresidual, {'nlpsol':'ipopt'})
    count_of_pfcnodes = model.shape_of_Y[0]
    Vinit_ = (
        casadi.vertcat([1.]*count_of_pfcnodes, [0.]*count_of_pfcnodes)
        if Vinit is None else Vinit)
    # Vslack must be negative as Vslack_result + Vslack_in_Inode = 0
    #   because the root is searched for with formula: Y * Vresult + Inode = 0
    Vslack_neg = -model.slacks.V
    values_of_parameters=casadi.vertcat(
        np.real(Vslack_neg), np.imag(Vslack_neg),
        factordata.values_of_vars,
        factordata.values_of_consts)
    voltages = rf(Vinit_, values_of_parameters)
    return rf.stats()['success'], voltages

def calculate_power_flow(model, Vinit=None, vminsqr=_VMINSQR):
    """Solves the power flow problem using a rootfinding algorithm.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    Vinit: array_like
        optional
        float, initial guess of node voltages
    vminsqr: float
        optional
        square of voltage, upper limit interpolation interval [0...vminsqr]

    Returns
    -------
    tuple
        * bool, success?
        * casadi.DM, float, voltage vector [real parts, imaginary parts]"""
    gen_factor_symbols = ft._create_symbols_with_ids(
        model.factors.gen_factordata.index)
    v_syms_gb_ex = create_v_symbols_gb_expressions(model, gen_factor_symbols)
    get_factor_and_injection_data = make_get_factor_and_injection_data(
        model, gen_factor_symbols, v_syms_gb_ex['Vnode_syms'], vminsqr)
    factordata, injectiondata = get_factor_and_injection_data(step=0)
    Vslack_syms = v_syms_gb_ex['Vslack_syms']
    Inode_inj = _reset_slack_current(
        model.slacks.index_of_node,
        Vslack_syms[:,0], Vslack_syms[:,1],
        casadi.SX(model.mnodeinj) @ injectiondata[:,:2])
    return calculate_power_flow2(
        model, v_syms_gb_ex, factordata, Inode_inj, Vinit)

##############
# Estimation #
##############

# expressions for flow into injections

def get_injection_flow_expressions(ipqv, quantity, injections):
    """Creates expressions for current/power flowing into given injections.

    Parameters
    ----------
    ipqv: casadi.SX (n,8)
        [:,0] Ire, current, real part
        [:,1] Iim, current, imaginary part
        [:,2] Pscaled, active power P10 multiplied by scaling factor kp
        [:,3] Qscaled, reactive power Q10 multiplied by scaling factor kq
        [:,4] Pip, active power interpolated
        [:,5] Qip, reactive power interpolated
        [:,6] Vabs_sqr, square of voltage magnitude
        [:,7] interpolate?
    quantity: 'I'|'P'|'Q'
        addresses current magnitude, active power or reactive power
    injections: pandas.DataFrame (index of injection)
        * .Exp_v_p, float, voltage exponent for active power
        * .Exp_v_q, float, voltage exponent for reactive power

    Returns
    -------
    casadi.SX (shape n,2) for 'I', casadi.SX (shape n,1) for 'P' or 'Q'"""
    assert quantity in 'IPQ', \
        f'quantity needs to be one of "I", "P" or "Q" but is "{quantity}"'
    if quantity=='I':
        return ipqv[injections.index, :2]
    ipqv_ = ipqv[injections.index, :]
    if quantity=='P':
        # ipqv_[:,2] Pscaled
        Porig = casadi.power(ipqv_[:,6], injections.Exp_v_p/2) * ipqv_[:,2]
        # ipqv_[:,4] Pip, active power interpolated
        return casadi.if_else(ipqv_[:,7], ipqv_[:,4], Porig)
    if quantity=='Q':
        # ipqv_[:,3] Qscaled
        Qorig = casadi.power(ipqv_[:,6], injections.Exp_v_q/2) * ipqv_[:,3]
        # ipqv_[:,5] Qip, reactive power interpolated
        return casadi.if_else(ipqv_[:,7], ipqv_[:,5], Qorig)
    assert False, f'no processing for quantity "{quantity}"'

# expressions for flow into branches

def _power_into_branch(
        g_tot, b_tot, g_mn, b_mn, V_abs_sqr, Vre, Vim, Vre_other, Vim_other):
    """Calculates active and reactive power flow.

    Uses admittances of a branches and the voltages at their terminals. Assumes
    PI-equivalient circuit.
    ::
        S = VI'
    with term for I:
    ::
        I = (y_mm/2) V + y_mn(V - V_other)
        I = (y_mm/2 + y_mn) V - y_mn V_other
        I = y_tot V - y_mn V_other
    S is:
    ::

        S = V (y_tot V - y_mn V_other)'
        S = y_tot' V' V - y_mn' V_other' V = S_tot - S_mn
    matrix form of y_tot and y_tot' (== conjugate(y_tot))
    ::
                +-            -+             +-            -+
                | g_tot -b_tot |             |  g_tot b_tot |
        y_tot = |              | ;  y_tot' = |              |
                | b_tot  g_tot |             | -b_tot g_tot |
                +-            -+             +-            -+
    V' V in matrix form:
    ::
                                 +-   -+
                                 | 1 0 |
        V' V = (Vre**2 + Vim**2) |     |
                                 | 0 1 |
                                 +-   -+
    matrix form for S_tot:
    ::
        +-            -+                     +-   -+ +-            -+
        | P_tot -Q_tot |                     | 1 0 | |  g_tot b_tot |
        |              | = (Vre**2 + Vim**2) |     | |              |
        | Q_tot  P_tot |                     | 0 1 | | -b_tot g_tot |
        +-            -+                     +-   -+ +-            -+
                                             +-            -+
                                             |  g_tot b_tot |
                         = (Vre**2 + Vim**2) |              |
                                             | -b_tot g_tot |
                                             +-            -+
    vector of S_tot:
    ::
        +-     -+   +-                        -+
        | P_tot |   |  g_tot (Vre**2 + Vim**2) |
        |       | = |                          |
        | Q_tot |   | -b_tot (Vre**2 + Vim**2) |
        +-     -+   +-                        -+
    matrix for V_other' V:
    ::
                     +-                    -+ +-        -+
                     |  Vre_other Vim_other | | Vre -Vim |
        V_other' V = |                      | |          |
                     | -Vim_other Vre_other | | Vim  Vre |
                     +-                    -+ +-        -+
           +-                                                                -+
           |  (Vre Vre_other + Vim Vim_other) (Vre Vim_other - Vim Vre_other) |
         = |                                                                  |
           | (-Vre Vim_other + Vim Vre_other) (Vre Vre_other + Vim Vim_other) |
           +-                                                                -+
           +-    -+
           | A -B |    A = (Vre Vre_other + Vim Vim_other)
         = |      |
           | B  A |    B = (-Vre Vim_other + Vim Vre_other)
           +-    -+
    multiply y_mn' with V_other' V:
    ::
                           +-          -+ +-    -+
                           |  g_mn b_mn | | A -B |
        y_mn' V_other' V = |            | |      |
                           | -b_mn g_mn | | B  A |
                           +-          -+ +-    -+
    					   +-                                    -+
    					   |  (g_mn A + b_mn B) (b_mn A - g_mn B) |
    					 = |                                      |
    					   | (-b_mn A + g_mn B) (g_mn A + b_mn B) |
    					   +-                                    -+
    S_mn:
    ::
        +-    -+   +-                  -+
        | P_mn |   |  (g_mn A + b_mn B) |
        |      | = |                    |
        | Q_mn |   | (-b_mn A + g_mn B) |
        +-    -+   +-                  -+
    terms for P and Q
    ::
        P =  g_tot (Vre**2 + Vim**2)
    	    - (  g_mn ( Vre Vre_other + Vim Vim_other)
    		   + b_mn (-Vre Vim_other + Vim Vre_other))

        Q = -b_tot (Vre**2 + Vim**2)
    	    + (  b_mn ( Vre Vre_other + Vim Vim_other)
    		   - g_mn (-Vre Vim_other + Vim Vre_other))

    Parameters
    ----------
    g_tot: float
        g_mm + g_mn
    b_tot: float
         b_mm + b_mn
    g_mn: float
        longitudinal conductance
    b_mn: float
        longitudinal susceptance
    V_abs_sqr: float
        Vre**2 + Vim**2
    Vre: float
        voltage in node, real part
    Vim: float
        voltage in node, imaginary part
    Vre_other: float
        voltage in other node, real part
    Vim_other: float
        voltage in other node, imaginary part

    Returns
    -------
    tuple
        * P, active power
        * Q, reactive power"""
    A = Vre * Vre_other + Vim * Vim_other
    B = Vim * Vre_other - Vre * Vim_other
    P =  V_abs_sqr * g_tot  - (A * g_mn + B * b_mn)
    Q = -V_abs_sqr * b_tot  + (A * b_mn - B * g_mn)
    return P, Q

def power_into_branch(gb_mn_tot, Vnode, terms):
    """Creates expressions of active and reactive power flow.

    Creates expressions for a subset of branch terminals from admittances
    of branches and voltages at branch terminals. Assumes PI-equivalent
    circuits.

    Parameters
    ----------
     gb_mn_tot: casadi.SX
        conductance/susceptance of branches at terminals, (index of terminal)
        * [:,0] g_mn, mutual conductance
        * [:,1] b_mn, mutual susceptance
        * [:,2] g_tot, self conductance + mutual conductance
        * [:,3] b_tot, self susceptance + mutual susceptance
    Vnode: casadi.SX
        three vectors of node voltage expressions
        * Vnode[:,0], float, Vre vector of real node voltages
        * Vnode[:,1], float, Vim vector of imaginary node voltages
        * Vnode[:,2], float, Vre**2 + Vim**2
    terms: pandas.DataFrame (index of terminal)
        * .index_of_node, int,
           index of power flow calculation node connected to terminal
        * .index_of_other_node, int
           index of power flow calculation node connected to other terminal
           of same branch like terminal

    Returns
    -------
    tuple
        * P, active power
        * Q, reactive power"""
    if len(terms):
        Vterm = Vnode[terms.index_of_node,:]
        Vother = Vnode[terms.index_of_other_node,:]
        gb_mn_tot_ = gb_mn_tot[terms.index, :]
        return _power_into_branch(
            gb_mn_tot_[:,2], gb_mn_tot_[:,3], gb_mn_tot_[:,0], gb_mn_tot_[:,1],
            Vterm[:,2], Vterm[:,0], Vterm[:,1], Vother[:,0], Vother[:,1])
    return _SX_0r1c, _SX_0r1c

def _current_into_branch(
        g_tot, b_tot, g_mn, b_mn, Vre, Vim, Vre_other, Vim_other):
    """Computes real and imaginary current flowing into a branch.

    current flow into one branch
    ::
        +-   -+   +-                                -+ +-   -+
        | Ire |   | (g_mm/2 + g_mn) -(b_mm/2 + b_mn) | | Vre |
        |     | = |                                  | |     |
        | Iim |   | (b_mm/2 + b_mn)  (g_mm/2 + g_mn) | | Vim |
        +-   -+   +-                                -+ +-   -+

                       +-          -+ +-         -+
                       | g_mn -b_mn | | Vre_other |
                     - |            | |           |
                       | b_mn  g_mn | | Vim_other |
                       +-          -+ +-         -+
    Parameters
    ----------
    g_tot: float
        g_mm / 2 + g_mn
    b_tot: float
        b_mm / 2 + b_mn
    g_mn: float
        longitudinal conductance
    b_mn: float
        longitudinal susceptance
    Vre: float
        voltage in node, real part
    Vim: float
        voltage in node, imaginary part
    Vre_other: float
        voltage in other node, real part
    Vim_other: float
        voltage in other node, imaginary part

    Returns
    -------
    casadi.SX, shape n,2"""
    Ire = g_tot * Vre - b_tot * Vim - g_mn * Vre_other + b_mn * Vim_other
    Iim = b_tot * Vre + g_tot * Vim - b_mn * Vre_other - g_mn * Vim_other
    return casadi.horzcat(Ire, Iim)

def current_into_branch(gb_mn_tot, Vnode, terms):
    """Generates expressions for real and imaginary current into branches.

    Expects a subset of branch terminals. Uses expressions of branch
    admittances and voltages at branch terminals. Assumes PI-equivalient
    circuit.

    Parameters
    ----------
     gb_mn_tot: casadi.SX
        conductance/susceptance of branches at terminals, (index of terminal)
        * [:,0] g_mn, mutual conductance
        * [:,1] b_mn, mutual susceptance
        * [:,2] g_tot, self conductance + mutual conductance
        * [:,3] b_tot, self susceptance + mutual susceptance
    Vnode: casadi.SX
        vectors of node voltage expressions
        * Vnode[:,0], float, Vre vector of real node voltages
        * Vnode[:,1], float, Vim vector of imaginary node voltages
    terms: pandas.DataFrame (index of terminal)
        * .index_of_node, int,
           index of power flow calculation node connected to terminal
        * .index_of_other_node, int
           index of power flow calculation node connected to other terminal
           of same branch like terminal

    Returns
    -------
    casadi.SX, (shape n,2 - 0:Ire, 1:Iim)"""
    if len(terms):
        Vterm = Vnode[terms.index_of_node,:]
        Vother = Vnode[terms.index_of_other_node,:]
        gb_mn_tot_ = gb_mn_tot[terms.index, :]
        return _current_into_branch(
            gb_mn_tot_[:,2], gb_mn_tot_[:,3], gb_mn_tot_[:,0], gb_mn_tot_[:,1],
            Vterm[:,0], Vterm[:,1], Vother[:,0], Vother[:,1])
    return casadi.SX(0,2)

def get_branch_flow_expressions(v_syms_gb_ex, quantity, branchterminals):
    """Creates expressions for calculation of Ire and Iim or P and Q
    of branches.

    Parameters
    ----------
    v_syms_gb_ex: dict
        * 'Vnode_syms', casadi.SX, vectors, symbols of node voltages
        * 'position_syms', casadi.SX, vector, symbols of tap positions
        * 'gb_mn_tot', casadi.SX, vectors, conductance and susceptance of
          branches connected to terminals
    selector: 'PQ'|'I'
        selects which values to calculate 'PQ' - active and reactive power,
        'I' - real and imaginary part of current

    Returns
    -------
    casadi.SX
        expressions for Iri/PQ"""
    assert quantity=='PQ' or quantity=='I',\
        f'value of indicator must be "PQ" or "I" but is "{quantity}"'
    expr_fn = power_into_branch if quantity=='PQ' else current_into_branch
    return expr_fn(
        v_syms_gb_ex['gb_mn_tot'], v_syms_gb_ex['Vnode_syms'], branchterminals)

def _get_I_expressions(Iri_exprs):
    """Creates an expression for calculating the magnitudes of current from
    real and imaginary parts.

    Parameters
    ----------
    Iri_exprs: casadi.SX (shape n,2)
        * Iri_exprs[:,0]Ire, real part of current
        * Iri_exprs[:,1]Iim, imaginary part of current

    Returns
    -------
    casadi.SX, shape (n,1)"""
    # I = sqrt(Ire**2, Iim**2), vector
    return (
        # Ire**2 + Iim**2
        casadi.sum2(
            # Ire**2, Iim**2
            Iri_exprs * Iri_exprs)
        # I = sqrt(Ire**2, Iim**2)
        .sqrt())

def make_get_branch_expressions(v_syms_gb_ex, quantity):
    """Returns an expression building function for I/P/Q-values at
    terminals of branches.

    Parameters
    ----------
    v_syms_gb_ex: dict
        * 'Vnode_syms', casadi.SX, vectors, symbols of node voltages
        * 'position_syms', casadi.SX, vector, symbols of tap positions
        * 'gb_mn_tot', casadi.SX, vectors, conductance and susceptance of
          branches connected to terminals
    quantity: 'I'|'P'|'Q'
        selects the measurement entites to create differences for

    Returns
    -------
    function
        (pandas.DataFrame) -> (casasdi.SX, shape n,2/n,1)
        (branchterminals) -> (expressions)"""
    assert quantity in 'IPQ', \
        f'quantity needs to be one of "I", "P" or "Q" but is "{quantity}"'
    if quantity=='I':
        return (
            lambda terminals:
                get_branch_flow_expressions(v_syms_gb_ex, 'I', terminals))
    if quantity=='P':
        return (
            lambda terminals:
                get_branch_flow_expressions(v_syms_gb_ex, 'PQ', terminals)[0])
    if quantity=='Q':
        return (
            lambda terminals:
                get_branch_flow_expressions(v_syms_gb_ex, 'PQ', terminals)[1])
    assert False, f'no processing implemented for quantity "{quantity}"'

#
# expressions of differences, measured - calculated
#

def _select_and_check_for_single_current(vals, key):
    selected = vals[key]
    assert selected.size < 2, (
        'more than one current value is given for one batch, however, '
        'only a single current value is possible '
        'as addition cannot be used for magnitudes of complex currents')
    return selected

def _select_and_check_for_single_cost(vals, key):
    selected = vals[key]
    assert selected.size < 2, (
        'more than one cost value is given for one batch, however, '
        'only a single cost value is possible')
    return selected

def _make_get_value(values, quantity):
    """Helper, creates a function for retrieving a value from model.

    ivalues, pvalues, qvalues or vvalues. Returns 'per-phase' values
    (one third of given P or Q).

    Returns cost for 3-phase PQ. Keep in mind the calculation is made with
    one third of P and Q.

    Parameters
    ----------
    values: pandas.DataFrame

    quantity: 'I'|'P'|'Q'|'V'|'cost'
        selects returned values

    Returns
    -------
    function
        (str) -> (float)
        (index) -> (value)"""
    try:
        vals = values[quantity]
    except KeyError:
        return lambda _: 0
    if quantity in 'PQ':
        vals *= values.direction / 3. # convert to single phase
        return lambda key: vals[key].sum()
    elif quantity in 'V':
        return lambda key: vals[key].mean()
    elif quantity == 'cost':
        try:
            vals *= values.direction # cost for 3 phase value !
            return partial(_select_and_check_for_single_cost, vals)
        except:
            return lambda _: 0
    elif quantity == 'I':
        return partial(_select_and_check_for_single_current, vals)

def _get_batch_expressions_br(model, v_syms_gb_ex, quantity):
    """Creates a vector (casadi.SX, shape n,2/n,1) expressing calculated branch
    values for absolute current, active power or reactive power. The
    expressions are based on the batch definitions.

    Parameters
    ----------
    model: egrid.model.Model
        model of electric network for calculation
    v_syms_gb_ex: dict
        * 'Vnode_syms', casadi.SX, vectors, symbols of node voltages
        * 'gb_mn_tot', casadi.SX, vectors, conductance and susceptance of
          branches connected to terminals
    quantity: 'I'|'P'|'Q'
        addresses current magnitude, active power or reactive power

    Returns
    -------
    dict
        id_of_batch => expression for I/P/Q-calculation"""
    assert quantity in 'IPQ', \
        f'quantity needs to be one of "I", "P" or "Q" but is "{quantity}"'
    get_branch_expr = make_get_branch_expressions(v_syms_gb_ex, quantity)
    branchterminals = model.branchterminals
    return {
        id_of_batch:get_branch_expr(branchterminals.loc[df.index_of_terminal])
        for id_of_batch, df in get_batches(
            get_values(model, quantity),
            model.branchoutputs,
            'index_of_terminal')}

def _get_batch_expressions_inj(model, ipqv, quantity):
    """Creates a vector (casadi.SX, shape n,1) expressing injected absolute
    current, active power or reactive power. The expressions are based
    on the batch definitions.

    Parameters
    ----------
    model: egrid.model.Model
        model of electric network for calculation
    ipqv: casadi.SX (n,8)
        [:,0] Ire, injected current, real part
        [:,1] Iim, injected current, imaginary part
        [:,2] Pscaled, injected active power P10 multiplied by
              scaling factor kp
        [:,3] Qscaled, injected reactive power Q10 multiplied by
              scaling factor kq
        [:,4] Pip, injected active power interpolated
        [:,5] Qip, injected reactive power interpolated
        [:,6] Vabs_sqr, square of voltage magnitude
        [:,7] interpolate?
    quantity: 'I'|'P'|'Q'
        addresses current magnitude, active power or reactive power

    Returns
    -------
    dict
        id_of_batch => expression for I/P/Q-calculation"""
    assert quantity in 'IPQ', \
        f'quantity needs to be one of "I", "P" or "Q" but is "{quantity}"'
    get_inj_expr = partial(get_injection_flow_expressions, ipqv, quantity)
    injections = model.injections
    return {
        id_of_batch: get_inj_expr(injections.loc[df.index_of_injection])
        for id_of_batch, df in get_batches(
            get_values(model, quantity),
            model.injectionoutputs,
            'index_of_injection')}

def get_batch_expressions(model, v_syms_gb_ex, ipqv, quantity):
    """Expresses calculated values I/P/Q in terms of variables/parameters.

    Creates a vector (casadi.SX, shape n,1) expressing calculated
    values for absolute current, active power or reactive power. The
    expressions are based on the batch definitions.

    Parameters
    ----------
    model: egrid.model.Model
        model of electric network for calculation
    v_syms_gb_ex: dict
        * 'Vnode_syms', casadi.SX, vectors, symbols of node voltages
        * 'gb_mn_tot', casadi.SX, vectors, conductance and susceptance of
          branches connected to terminals
    ipqv: casadi.SX (n,8)
        [:,0] Ire, injected current, real part
        [:,1] Iim, injected current, imaginary part
        [:,2] Pscaled, injected active power P10 multiplied by
              scaling factor kp
        [:,3] Qscaled, injected reactive power Q10 multiplied by
              scaling factor kq
        [:,4] Pip, injected active power interpolated
        [:,5] Qip, injected reactive power interpolated
        [:,6] Vabs_sqr, square of voltage magnitude
        [:,7] interpolate?
    quantity: 'I'|'P'|'Q'
        addresses current magnitude, active power or reactive power

    Returns
    -------
    dict
        id_of_batch => expression for I/P/Q-calculation"""
    dd = defaultdict(casadi.SX)
    dd.update(_get_batch_expressions_br(model, v_syms_gb_ex, quantity))
    injectionexpr = _get_batch_expressions_inj(model, ipqv, quantity)
    for id_of_batch, expr in injectionexpr.items():
        dd[id_of_batch] = casadi.sum1(casadi.vertcat(dd[id_of_batch], expr))
    if quantity in 'PQ':
        return dd
    if quantity == 'I':
        return {id_of_batch: _get_I_expressions(Iri_exprs)
                for id_of_batch, Iri_exprs in dd.items()}
    assert False, \
        f'quantity needs to be one of "I", "P" or "Q" but is "{quantity}"'

def _get_branch_loss_expression(model, v_syms_gb_ex):
    """Creates an expression for active power losses of all branches.

    Losses are calculated for 3-phase-branches which means that the one
    phase values are multiplied by 3.

    Parameters
    ----------
    model: egrid.model.Model
        model of electric network for calculation
    v_syms_gb_ex: dict
        * 'Vnode_syms', casadi.SX, vectors, symbols of node voltages
        * 'gb_mn_tot', casadi.SX, vectors, conductance and susceptance of
          branches connected to terminals

    Returns
    -------
    casadi.SX"""
    Vnode = v_syms_gb_ex['Vnode_syms']
    gb_mn_tot = v_syms_gb_ex['gb_mn_tot']
    branchterminals = model.branchterminals
    if len(branchterminals):
        Vterm = Vnode[branchterminals.index_of_node,:]
        Vother = Vnode[branchterminals.index_of_other_node,:]
        # single phase to 3-phase
        p_term = 3 * (
            _power_into_branch(
                gb_mn_tot[:,2], gb_mn_tot[:,3], gb_mn_tot[:,0], gb_mn_tot[:,1],
                Vterm[:,2], Vterm[:,0], Vterm[:,1], Vother[:,0], Vother[:,1])
            [0])
        p_term_a = p_term[model.terminal_to_branch[0]]
        p_term_b = p_term[model.terminal_to_branch[1]]
        return casadi.sum1(p_term_a + p_term_b)
    return casadi.SX(0)

def _get_diff_expression(symbols):
    """Generates squared difference expression.

    Parameters
    ----------
    symbols: casadi.SX (shape n,1)
        symbols

    Returns
    -------
    casadi.SX (shape 1,1)"""
    size = symbols.size1()
    return (
        casadi.sumsqr(
            # value - average
            symbols - (casadi.sum1(symbols) / size))
        if 1 < size else casadi.sumsqr(symbols))

_expression_functions = {
    'diff': _get_diff_expression}

def _get_expression_function(name):
    """Returns the expression building function for the given name.

    Parameters
    ----------
    name: str
        identifier of function

    Returns
    -------
    function
        (casadi.SX (shape n,1)) -> (casadi.SX (shape 1,1))"""
    return _expression_functions.get(name, lambda _:_SX_0r1c)

def _get_symbols(symbols, id_to_idx, ids):
    """Fetches symbols for given IDs

    Parameters
    ----------
    symbols: casadi.SX
        symbols of decision variables and parameters (constants)
    id_to_idx: pandas.Series (index: id_of_symbol)
        int, index of symbol in argument symbols
    ids: iterable
        string, identifiers of symbols

    Returns
    -------
    casadi.SX
        (shape n,1)"""
    try:
        return symbols[id_to_idx.loc[list(ids)]]
    except:
        return _SX_0r1c

def _get_oterm_expressions(symbols, id_to_idx, oterms):
    """Creates a term for the objective function.

    Parameters
    ----------
    symbols: casadi.SX
        symbols of decision variables and parameters (constants)
    id_to_idx: pandas.Series (index: id_of_symbol)
        int, index of symbol in argument symbols
    oterms: pandas.DataFrame
        * ['args'], iterable string, identifiers of arguments
        * ['fn'], str, identifier of function

    Returns
    -------
    casadi.SX
        term of objective function"""
    get_symbols = partial(_get_symbols, symbols, id_to_idx)
    return casadi.sum1(
        casadi.vcat(
            [_get_expression_function(row.fn.lower())(get_symbols(row.args))
             for _, row in oterms.iterrows()]))

def get_batch_flow_expressions(model, v_syms_gb_ex, ipqv, quantity, cost):
    """Expresses differences between measured and calculated values.

    Creates a vector (casadi.SX, shape n,1) expressing the difference
    between measured and calculated values for absolute current, active power
    or reactive power. The expressions are based on the batch definitions.
    Intended use is building the objective.

    Returns the cost and expression for P or Q if cost==True.

    Parameters
    ----------
    model: egrid.model.Model
        model of electric network for calculation
    v_syms_gb_ex: dict
        * 'Vnode_syms', casadi.SX, vectors, symbols of node voltages
        * 'gb_mn_tot', casadi.SX, vectors, conductance and susceptance of
          branches connected to terminals
    ipqv: casadi.SX (n,8)
        [:,0] Ire, injected current, real part
        [:,1] Iim, injected current, imaginary part
        [:,2] Pscaled, injected active power P10 multiplied by
              scaling factor kp
        [:,3] Qscaled, injected reactive power Q10 multiplied by
              scaling factor kq
        [:,4] Pip, injected active power interpolated
        [:,5] Qip, injected reactive power interpolated
        [:,6] Vabs_sqr, square of voltage magnitude
        [:,7] interpolate?
    quantity: 'I'|'P'|'Q'
        addresses current magnitude, active power or reactive power
    cost: bool
        returns cost instead of values of active/reactive power (only for P/Q)

    Returns
    -------
    tuple:
        * array_like, str, Ids of batches
        * array_like, float, values
        * casadi.SX, (shape n,1), expressions"""
    assert quantity in 'IPQ', \
        f'quantity needs to be one of "I", "P" or "Q" but is "{quantity}"'
    values = get_values(model, quantity).set_index('id_of_batch')
    batchid_expr = get_batch_expressions(model, v_syms_gb_ex, ipqv, quantity)
    exprs = casadi.vcat(batchid_expr.values())
    get_value = _make_get_value(values, 'cost' if cost else quantity)
    batchids = batchid_expr.keys()
    vals = list(map(get_value, batchids))
    return batchids, vals, exprs if exprs.size1() else _SX_0r1c

def get_flow_cost_expression(model, v_syms_gb_ex, ipqv):
    """Creates an expression for cost of active and reactive power flow.

    Parameters
    ----------
    model: egrid.model.Model
        model of electric network for calculation
    v_syms_gb_ex: dict
        * 'Vnode_syms', casadi.SX, vectors, symbols of node voltages
        * 'gb_mn_tot', casadi.SX, vectors, conductance and susceptance of
          branches connected to terminals
    ipqv: casadi.SX (n,8)
        [:,0] Ire, injected current, real part
        [:,1] Iim, injected current, imaginary part
        [:,2] Pscaled, injected active power P10 multiplied by
              scaling factor kp
        [:,3] Qscaled, injected reactive power Q10 multiplied by
              scaling factor kq
        [:,4] Pip, injected active power interpolated
        [:,5] Qip, injected reactive power interpolated
        [:,6] Vabs_sqr, square of voltage magnitude
        [:,7] interpolate?

    Returns
    -------
    casadi.SX"""
    batchids_p, vals_p, exprs_p = get_batch_flow_expressions(
        model, v_syms_gb_ex, ipqv, 'P', True)
    batchids_q, vals_q, exprs_q = get_batch_flow_expressions(
        model, v_syms_gb_ex, ipqv, 'Q', True)
    return (
        casadi.sum1(casadi.DM(vals_p) * exprs_p)
        + casadi.sum1(casadi.DM(vals_q) * exprs_q))

def get_change_cost_expression(factordata):
    """Creates an expression for cost of factor change.

    Cost are equivalent to
    ::
        some_factor * abs(value_before_change - value_after_change)

    Parameters
    ----------
    factordata: factors.Factordata

    Returns
    -------
    casadi.SX"""
    change = factordata.values_of_vars - factordata.vars
    # sigmoid for calculation of absolute value, smooth, differentiable
    change_abs = casadi.erf(100*change) * change
    return casadi.sum1(change_abs * factordata.cost_of_change)

def get_diff_expressions(model, expressions, ipqv, objectives):
    """Expresses differences between measured and calculated values.

    Creates a vector (casadi.SX, shape n,1) expressing the difference
    between measured and calculated values for absolute current, active power
    or reactive power and voltage. The expressions are based on the batch
    definitions (P/Q/I) or referenced (V) node. Intended use is building
    the objective function.

    Parameters
    ----------
    model: egrid.model.Model
        model of electric network for calculation
    expressions: dict
        * 'Vnode_syms', casadi.SX, vectors, symbols of node voltages
        * 'position_syms', casadi.SX, vector, symbols of tap positions
        * 'gb_mn_tot', casadi.SX, vectors, conductance and susceptance of
          branches connected to terminals
    ipqv: casadi.SX (n,8)
        data of P, Q, I, and V at injections
        [:,0] Ire, current, real part
        [:,1] Iim, current, imaginary part
        [:,2] Pscaled, active power P10 multiplied by scaling factor kp
        [:,3] Qscaled, reactive power Q10 multiplied by scaling factor kq
        [:,4] Pip, active power, interpolated
        [:,5] Qip, reactive power, interpolated
        [:,6] Vabs_sqr, square of voltage magnitude
        [:,7] interpolate?
    objectives: str
        string of characters 'I'|'P'|'Q'|'V'
        addresses current magnitude, active power, reactive power or magnitude
        of voltage, other characters are ignored

    Returns
    -------
    tuple
        * quantities, numpy.array<str>
        * id_of_batch, numpy.array<str>
        * value, casadi.DM, vector (shape n,1)
        * expression, casadi.SX, vector (shape n,1)"""
    _objectives = []
    _ids = []
    _vals = []
    _exprs = casadi.SX(0, 1)
    for objective in objectives:
        if objective in 'IPQ':
            ids, vals, exprs = get_batch_flow_expressions(
                model, expressions, ipqv, objective, cost=False)
            _objectives.extend([objective]*len(ids))
            _ids.extend(ids)
            _vals.extend(vals)
            _exprs = casadi.vertcat(_exprs, exprs)
        elif objective=='V':
            vvals = value_of_voltages(model.vvalues)
            count_of_values = len(vvals)
            _objectives.extend([objective]*count_of_values)
            _ids.extend(vvals.id_of_node)
            _vals.extend(vvals.V)
            _exprs = casadi.vertcat(
                _exprs, expressions['Vnode_syms'][vvals.index,2].sqrt())
    return np.array(_objectives), np.array(_ids), casadi.DM(_vals), _exprs

def get_objective_expression(
        model, expressions, factordata, Iinj_data, floss, oterms, objectives):
    """Creates expression for objective function.

    Creates a vector (casadi.SX, shape n,1) expressing the difference
    between measured and calculated values for absolute current, active power
    or reactive power and voltage includes expressions for active power
    losses of branches. The expressions are based on the batch
    definitions or referenced node. Losses are calculated for all branches.

    Parameters
    ----------
    model: egrid.model.Model
        model of electric network for calculation
    expressions: dict
        * 'Vnode_syms', casadi.SX, vectors, symbols of node voltages
        * 'position_syms', casadi.SX, vector, symbols of tap positions
          for branches
        * 'gb_mn_tot', casadi.SX, vectors, conductance and susceptance of
          branches connected to terminals
    factordata: Factordata

    Iinj_data: casadi.SX (n,8), data of P, Q, I, and V at injections
        * [:,0] Ire, current, real part
        * [:,1] Iim, current, imaginary part
        * [:,2] Pscaled, active power P10 multiplied by scaling factor kp
        * [:,3] Qscaled, reactive power Q10 multiplied by scaling factor kq
        * [:,4] Pip, active power, interpolated
        * [:,5] Qip, reactive power, interpolated
        * [:,6] Vabs_sqr, square of voltage magnitude
        * [:,7] interpolate?
    floss: float
        multiplier for branch losses
    oterms: pandas.DataFrames
        * ['id']
        * ['args']
        * ['fn']
        * ['step']
    objectives: str
        string of characters 'I'|'P'|'Q'|'V'|'L'|'C'|'T'
        addresses current magnitude, active power, reactive power or magnitude
        of voltage, losses of branches, other characters are ignored

    Returns
    -------
    casadi.SX"""
    diff_data = get_diff_expressions(model, expressions, Iinj_data, objectives)
    objective = casadi.sumsqr(diff_data[2] - diff_data[3])
    if 'C' in objectives:
        # 'C' - cost
        objective += (
            get_change_cost_expression(factordata)
            + get_flow_cost_expression(model, expressions, Iinj_data))
    if 'L' in objectives:
        # 'L' - cost of losses
        objective += (floss * _get_branch_loss_expression(model, expressions))
    if 'T' in objectives:
        # 'T' - objective function terms explicitely given by model
        objective += _get_oterm_expressions(
            factordata.all, factordata.id_to_idx, oterms)
    return objective

def get_batch_constraints(values_of_constraints, expressions_of_batches):
    """Creates expressions of constraints for keeping batch values constant.

    Expressions are made for keeping I, P, Q, or V obtained from previous
    optimizatuion step, constant in next optimization step. Batch values
    are measured or set-point values.

    Parameters
    ----------
    values_of_constraints: tuple
        * [0], str, quantity
        * [1], str, id_of_batch
        * [3], float, value
    expressions_of_batches: tuple
        * [0], str, quantity
        * [1], str, id_of_batch
        * [3], casadi.SX (shape m,1), expression

    Returns
    -------
    casadi.SX (shape n,1)"""
    batch_index = pd.MultiIndex.from_arrays(
        [expressions_of_batches[0],expressions_of_batches[1]],
        names=['quantity', 'id_of_batch'])
    value_index = pd.MultiIndex.from_arrays(
        [values_of_constraints[0],values_of_constraints[1]],
        names=['quantity', 'id_of_batch'])
    values_ = pd.Series(values_of_constraints[2], index=value_index)
    values = values_.reindex(batch_index).reset_index(drop=True)
    values_notna = values[values.notna()]
    if values_notna.size:
        expr_indices = values_notna.index.to_numpy()
        expr = expressions_of_batches[3][expr_indices]
        return casadi.SX(values_notna.to_numpy()) - expr
    return _SX_0r1c

def get_optimize_vk(model, expressions):
    """Prepares a function which optimizes node voltages and scaling factors.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    expressions : dict
        estimation data
        * 'Vnode_syms', casadi.SX (shape n,2)
        * 'Vslack_syms', casadi.SX (shape n,2)
        * 'position_syms', casadi.SX (shape n,1)
    positions: array_like
        optional
        int, positions of taps

    Returns
    -------
    function"""
    # setup solver
    Vnode_ri_syms = vstack(expressions['Vnode_syms'], 2)
    # values of parameters
    #   Vslack must be negative as Vslack_result + Vslack_in_Inode = 0
    #   because the root is searched for with formula: Y * Vresult + Inode = 0
    Vslacks_neg = -model.slacks.V
    params_ = vstack(expressions['Vslack_syms'])
    values_of_parameters_ = casadi.vertcat(
        np.real(Vslacks_neg), np.imag(Vslacks_neg))
    count_of_vri = 2 * model.shape_of_Y[0]
    Vmin = [-np.inf] * count_of_vri
    Vmax = [np.inf] * count_of_vri
    def optimize_vk(
        Vnode_ri_ini, factordata, Inode_inj, objective, constraints, lbg, ubg):
        """Solves an optimization task.

        Parameters
        ----------
        Vnode_ri_ini: casadi.DM (shape 2n,1)
            float, initial node voltages with separated real and imaginary
            parts, first real then imaginary
        factordata: Factordata
            * .vars, casadi.SX, column vector, symbols for variables
              of scaling factors
            * .consts, casadi.SX, column vector, symbols for constants
              of scaling factors
            * .values_of_vars, casadi.DM, column vector, initial values
              for vars
            * .values_of_consts, casadi.DM, column vector, values for consts
        Inode_inj: casadi.SX (shape n,2)
            expressions for injected node current
        objective: casadi.SX
            expression to minimize
        constraints: casadi.SX (shape m,1)
            expressions for additional constraints
            (default constraints are Inode==0)
        lbg: casadi.DM (shape m,1)
            lower bound of additional constraints
        ubg: casadi.DM (shape m,1)
            upper bound of additional constraints

        Returns
        -------
        bool
            success?
        casadi.DM
            result vector of optimization"""
        # symbols of decision variables: voltages and factors variables
        syms = casadi.vertcat(Vnode_ri_syms, factordata.vars)
        # symbols of parameters
        params = casadi.vertcat(params_, factordata.consts)
        values_of_parameters = casadi.vertcat(
            values_of_parameters_, factordata.values_of_consts)
        # residual node current + additional constraints
        Y_by_V = expressions['Y_by_V']
        constraints_ = casadi.vertcat(Y_by_V + vstack(Inode_inj), constraints)
        nlp = {'x': syms, 'f': objective, 'g': constraints_, 'p': params}
        is_discrete = factordata.is_discrete
        if any(is_discrete):
            discrete = np.concatenate(
                # voltage variables are not discrete
                [np.full((Vnode_ri_syms.size1(),), False, dtype=np.bool_),
                 is_discrete])
            solver = casadi.nlpsol(
                'solver', 'bonmin', nlp, {'discrete':discrete})
        else:
            solver = casadi.nlpsol('solver', 'ipopt', nlp, _IPOPT_opts)
        # initial values of decision variables
        ini = casadi.vertcat(Vnode_ri_ini, factordata.values_of_vars)
        # limits of decision variables
        lbx = casadi.vertcat(Vmin, factordata.var_min)
        ubx = casadi.vertcat(Vmax, factordata.var_max)
        # bounds of constraints, upper and lower bounds for node currents
        #   is zero, hence, just one vector of zeros is needed for
        #   node currents
        bg = casadi.DM.zeros(Y_by_V.size1())
        lbg_ = casadi.vertcat(bg, lbg)
        ubg_ = casadi.vertcat(bg, ubg)
        # calculate
        r = solver(
            x0=ini, p=values_of_parameters, lbg=lbg_, ubg=ubg_,
            lbx=lbx, ubx=ubx)
        return solver.stats()['success'], r['x']
    return optimize_vk

def _rm_slack_entries(expr, count_of_slacks):
    """Removes rows of slack nodes.

    Parameters
    ----------
    expr: casadi.SX

    count_of_slacks: int
        number of slack nodes

    Returns
    -------
    casadi.SX"""
    rows = expr.size1()
    upper, lower = casadi.vertsplit(expr, [0, rows//2, rows])
    return casadi.vertcat(
        upper[count_of_slacks:, :], lower[count_of_slacks:, :])

def get_optimize_vk2(model, expressions, positions=None):
    """Prepares a function which optimizes node voltages and scaling factors.

    Processes slack differently than function 'get_optimize_vk'. Slackrows
    are removed from admittance matrix. Slackvoltages are parameters.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    expressions : dict
        estimation data
        * 'Vnode_syms', casadi.SX (shape n,2)
        * 'Vslack_syms', casadi.SX (shape n,2)
        * 'position_syms', casadi.SX (shape n,1)
    positions: array_like
        optional
        int, positions of taps

    Returns
    -------
    function"""
    # setup solver
    vexpr = expressions['Vnode_syms']
    count_of_slacks = model.count_of_slacks
    vparam = vexpr[:count_of_slacks, :2]
    Vparam_ri_syms = vstack(vparam)
    vvar = vexpr[count_of_slacks:, :2]
    Vnode_ri_syms = vstack(vvar)
    params_ = casadi.vertcat(
        vstack(Vparam_ri_syms), expressions['position_syms'])
    # parameters
    positions_ = model.branchtaps.position if positions is None else positions
    Vslacks = model.slacks.V
    Vre_slack = np.real(Vslacks)
    Vim_slack = np.imag(Vslacks)
    values_of_parameters_ = casadi.vertcat(Vre_slack, Vim_slack, positions_)
    # limits
    count_of_vri = Vnode_ri_syms.size1()
    Vmin = [-np.inf] * count_of_vri
    Vmax = [np.inf] * count_of_vri
    # constraints
    Y_by_V = _rm_slack_entries(expressions['Y_by_V'], count_of_slacks)
    def optimize_vk(
        Vnode_ri_ini, factordata, Inode_inj, objective,
        constraints, lbg, ubg):
        """Solves an optimization task.

        Parameters
        ----------
        Vnode_ri_ini: casadi.DM (shape 2n,1)
            float, initial node voltages with separated real and imaginary
            parts, first real then imaginary
        factordata: Factordata
            * .vars, casadi.SX, column vector, symbols for variables
              of scaling factors
            * .consts, casadi.SX, column vector, symbols for constants
              of scaling factors
            * .values_of_vars, casadi.DM, column vector, initial values
              for vars
            * .values_of_consts, casadi.DM, column vector, values for consts
        Inode_inj: casadi.SX (shape n,2)
            expressions for injected node current
        objective: casadi.SX
            expression to minimize
        constraints: casadi.SX (shape m,1)
            expressions for additional constraints
            (default constraints are Inode==0)
        lbg: casadi.DM (shape m,1)
            lower bound of additional constraints
        ubg: casadi.DM (shape m,1)
            upper bound of additional constraints

        Returns
        -------
        bool
            success?
        casadi.DM
            result vector of optimization"""
        syms = casadi.vertcat(Vnode_ri_syms, factordata.vars)
        params = casadi.vertcat(params_, factordata.consts)
        values_of_parameters = casadi.vertcat(
            values_of_parameters_, factordata.values_of_consts)
        constraints_ = casadi.vertcat(
            Y_by_V + vstack(Inode_inj[count_of_slacks:, :]),
            constraints)
        nlp = {'x': syms, 'f': objective, 'g': constraints_, 'p': params}
        is_discrete = factordata.is_discrete
        if any(is_discrete):
            discrete = np.concatenate(
                # voltage variables are not discrete
                [np.full((Vnode_ri_syms.size1(),), False, dtype=np.bool_),
                 is_discrete])
            solver = casadi.nlpsol(
                'solver', 'bonmin', nlp, {'discrete':discrete})
        else:
            solver = casadi.nlpsol('solver', 'ipopt', nlp, _IPOPT_opts)
        # initial values of decision variables
        vini = _rm_slack_entries(Vnode_ri_ini, count_of_slacks)
        ini = casadi.vertcat(vini, factordata.values_of_vars)
        # limits of decision variables
        lbx = casadi.vertcat(Vmin, factordata.var_min)
        ubx = casadi.vertcat(Vmax, factordata.var_max)
        # bounds of constraints, upper and lower bounds for node currents
        #   is zero, hence, just one vector of zeros is needed for
        #   node currents
        bg = casadi.DM.zeros(Y_by_V.size1())
        lbg_ = casadi.vertcat(bg, lbg)
        ubg_ = casadi.vertcat(bg, ubg)
        # calculate
        r = solver(
            x0=ini, p=values_of_parameters, lbg=lbg_, ubg=ubg_,
            lbx=lbx, ubx=ubx)
        # add slack voltages
        parta, partb = casadi.vertsplit(
            r['x'], [0, count_of_vri//2, ini.size1()])
        return (
            solver.stats()['success'],
            casadi.vertcat(Vre_slack, parta, Vim_slack, partb))
    return optimize_vk

#
# organize data
#

def vstack(m, column_count=0):
    """Helper, stacks columns of matrix m vertically which creates a vector.

    Parameters
    ----------
    m: casadi.DM
        shape (n,m)
    column_count: int
        only first column_count columns will be stacked vertically

    Returns
    -------
    casadi.DM shape (n*m,1)"""
    size2 = m.size2()
    cc = min(column_count, size2) if 0 < column_count else size2
    return casadi.vertcat(*(m[:,idx] for idx in range(cc))) \
        if cc else _DM_0r1c

def ri_to_ri2(arr):
    """Converts a vector with separate real and imaginary parts of shape n,1
    into a vector of shape n/2,2.

    Parameters
    ----------
    arr: numpy.array

    Returns
    -------
    numpy.array (shape n/2,2)"""
    return np.hstack(np.vsplit(arr, 2))

def ri_to_complex(arr):
    """Converts a vector with separate real and imaginary parts into a
    vector of complex values.

    Parameters
    ----------
    arr: numpy.array

    Returns
    -------
    numpy.array"""
    return ri_to_ri2(arr).view(dtype=np.complex128)

#
# with estimation result
#

def make_calculate(symbols, values):
    """Helper, returns a function which evaluates expressions numerically.

    Parameters
    ----------
    symbols: tuple
        casadi.SX, symbols which might be used in expressions to evaluate
    values: tuple
        array_like, values of symbols

    Returns
    -------
    function
        (casadi.SX) -> (casadi.DM)
        (expressions) -> (values)"""
    def _calculate(expressions):
        """Calculates values for expressions.

        Parameters
        ----------
        expressions: casadi.SX

        Returns
        -------
        casadi.DM"""
        fn = casadi.Function('fn', list(symbols), [expressions])
        return fn(*values)
    return _calculate

def get_calculate_from_result(model, vsyms, factordata, x):
    """Returns a function numerically evaluating casadi.SX-expressions.

    Creates a function which calculates the values of casadi.SX expressions
    using the result of the nlp-solver.

    Parameters
    ----------
    model: egrid.model.Model
        model of electric network for calculation
    vsyms: dict
        * 'Vnode_syms', casadi.SX, vectors, symbols of node voltages
        * 'Vslack_syms', symbols of slack voltages
    factordata: Factordata
        step specific symbols
        * .vars, casadi.SX, decision variables
        * .consts, casadi.SX, parameters
        * .values_of_consts, casadi.DM
    x: casadi.DM
        result vector calculated by nlp-solver

    Returns
    -------
    function
        (casadi.SX) -> (casadi.DM)
        (expressions) -> (values)"""
    Vnode_ri_syms = vstack(vsyms['Vnode_syms'], 2)
    count_of_v_ri = Vnode_ri_syms.size1()
    voltages_ri = x[:count_of_v_ri].toarray()
    values_of_factors = x[count_of_v_ri:]
    Vslacks_neg = -model.slacks.V
    return make_calculate(
        (factordata.vars,
         factordata.consts,
         Vnode_ri_syms,
         vstack(vsyms['Vslack_syms'], 2)),
        (values_of_factors,
         factordata.values_of_consts,
         voltages_ri,
         casadi.vertcat(np.real(Vslacks_neg), np.imag(Vslacks_neg))))

#
# convenience functions for easier handling of estimation
#

def _get_vlimits(process_vlimts, model, Vnode_sqr, step):
    """Fetches expressions, lower and upper bound of voltage limits.

    Parameters
    ----------
    process_vlimts: bool
        flag, switches voltage limits
    model: egrid.model.Model
        data of electric distribution network
    Vnode_sqr: casasdi.SX
        expressions for selected node voltages squared
    step : int
        index of optimization step

    Returns
    -------
    tuple
        * casadi.SX, expressions for square of selected node voltages
        * casadi.DM, lower bound for square of selected node voltages
        * casadi.DM, upper bound for square of selected node voltages"""
    if process_vlimts:
        vlimits = get_vlimits_for_step(model.vlimits, step)
        if len(vlimits):
            Vlimit_sqr = Vnode_sqr[vlimits.index]
            return (
                Vlimit_sqr[vlimits.index],
                casadi.DM(vlimits['min'] * vlimits['min']),
                casadi.DM(vlimits['max'] * vlimits['max']))
    return _SX_0r1c, _DM_0r1c, _DM_0r1c

def get_step_data(
    model, expressions, *, step=0, f_prev=_EMPTY_0r1c, objectives='',
    constraints='', values_of_constraints=None,  floss=1.):
    """Prepares data for call of function optimize_step.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    expressions: dict
        * 'Vnode_syms', casadi.SX, expressions of node Voltages
        * 'Vslack_syms', casadi.SX, symbols of slack voltages,
           Vslack_syms[:,0] real part, Vslack_syms[:,1] imaginary part
        * 'gb_mn_tot', conductance g / susceptance b per branch terminal
            * gb_mn_tot[:,0] g_mn, mutual conductance
            * gb_mn_tot[:,1] b_mn, mutual susceptance
            * gb_mn_tot[:,2] g_tot, self conductance + mutual conductance
            * gb_mn_tot[:,3] b_tot, self susceptance + mutual susceptance
        * 'Y_by_V', casadi.SX, expression for Y @ V
        * 'get_factor_and_injection_data', function

          (int, numpy.array<float>) -> (tuple - Factordata, casadi.SX)

          which is a function

          (index_of_step, factors_of_previous_step)
            -> (tuple - Factordata, injection_data)
        * 'inj_to_node', casadi.SX, matrix, maps from
          injections to power flow calculation nodes
    step: int
        optional, default 0
        index of estimation step
    f_prev: numpy.array
        optional
        result output of factors (only) calculated in previous step, if any
    objectives: str
        optional, default ''
        string of characters 'I'|'P'|'Q'|'V'|'L'|'C'|'T' or empty string ''
        addresses differences of calculated and given values to be minimized,
        the characters are symbols for:
            * 'I' given current magnitude
            * 'P' active power
            * 'Q' reactive power
            * 'V' magnitude of voltage
            * 'L' losses of branches
            * 'C' for cost
            * 'T' for terms from model.terms
        other characters are ignored
    constraints: str
        optional, default ''
        string of characters 'I'|'P'|'Q'|'V' or empty string ''
        addresses quantities of batches (measured values or setpoints)
        to be kept constant, the values are obtained from a previous
        calculation/initialization step, the characters are symbols for
        given current magnitude, active power, reactive power or
        magnitude of voltage, other characters are ignored,
        values must be given with argument 'values_of_constraints',
        conditions must be satisfied
    values_of_constraints: tuple
        optional, default False
        * [0] numpy.array<str>, quantities,
        * [1] numpy.array<str>, id_of_batch
        * [2] numpy.array<float>, vector (shape n,1), value
        values for constraints (refer to argument 'constraints')
    floss: float
        optional, default is 1.0
        multiplier for branch losses

    Returns
    -------
    dict
        * ['model'], egrid.model.Model
        * ['expressions'], dict
        * ['factordata'], factors.Factordata
        * ['Inode_inj'], casadi.SX
        * ['objective'], str
        * ['constraints'], str
        * ['lbg'], casadi.DM
        * ['ubg'], casadi.DM"""
    factordata, Iinj_data = (
        expressions['get_factor_and_injection_data'](step, f_prev))
    Vslack_syms = expressions['Vslack_syms']
    Inode_inj = _reset_slack_current(
        model.slacks.index_of_node,
        Vslack_syms[:,0], Vslack_syms[:,1],
        expressions['inj_to_node'] @ Iinj_data[:,:2])
    oterms = get_terms_for_step(model.terms, step)
    objective = get_objective_expression(
        model, expressions, factordata, Iinj_data, floss, oterms, objectives)
    expr_of_batches = get_diff_expressions(
        model, expressions, Iinj_data, constraints)
    batch_constraints = (
        _SX_0r1c
        if values_of_constraints is None or len(expr_of_batches[0])==0 else
        get_batch_constraints(values_of_constraints, expr_of_batches))
    process_vlimits = 'B' in constraints
    vlimits, lbg_v, ubg_v = _get_vlimits(
        process_vlimits, model, expressions['Vnode_syms'][:,2], step)
    constraints = casadi.vertcat(batch_constraints, vlimits)
    # bounds of batch constraints
    number_of_constraints = batch_constraints.size1()
    # lower and upper bound of Inode
    bg_inode = casadi.DM.zeros(number_of_constraints)
    lbg = casadi.vertcat(bg_inode, lbg_v)
    ubg = casadi.vertcat(bg_inode, ubg_v)
    return dict(
        model=model, expressions=expressions, factordata=factordata,
        Inode_inj=Inode_inj, objective=objective,
        constraints=constraints, lbg=lbg, ubg=ubg)

def get_step_data_fns(model, gen_factor_symbols):
    """Creates two functions for generating step specific data.

    Function 'ini_step_data' creates the step_data structure for the first run
    of function 'optimize_step'. Function 'next_step_data' for all
    subsequent runs.

    Parameters
    ----------
    model: egrid.model.Model
        data of a balanced distribution network
    gen_factor_symbols: casadi.SX
        generic factor symbols, decision variables or parameters for
        scaling factors of injections and terminal factors (taps factors)

    Returns
    -------
    tuple
        * ini_step_data: function
            ()->(Stepdata), optional parameters:
              objectives: str
                  optional, default ''
                  string of characters 'I'|'P'|'Q'|'V'|'L'|'C'
                  or empty string '',
                  addresses quantities (of measurements/setpoints),
                  branch losses and cost
              step: int
                  optional, default 0
              f_prev: factors calculated by previous step
                  optional, default None
              constraints: str
                  optional, default ''
                  string of characters 'I'|'P'|'Q'|'V' or empty string ''
              value_of_constraints: data made by function 'get_batch_values'
                  optional, default None
        * next_step_data: function
            (int, Stepdata, casadi.DM, casadi.DM)->(Stepdata), parameters:
              step: int
                  index of optimizatin step
              step_data: Stepdata
                  made in previous calculation step
              voltages_ri: casadi.DM
                  node voltages calculated by previous calculation step
              k: casadi.DM
                  scaling factors calculated by previous calculation step
              objectives: str (optional)
                  optional, default ''
                  string of characters 'I'|'P'|'Q'|'V'|'L'|'C'
                  or empty string ''
              constraints: str (optional)
                  optional, default ''
                  string of characters 'I'|'P'|'Q'|'V' or empty string ''"""
    expressions = get_expressions(model, gen_factor_symbols)
    make_step_data = partial(get_step_data, model, expressions)
    def next_step_data(
            *, step, voltages_ri, k, objectives='', constraints='', floss=1.):
        """Creates data for function 'optimize_step'.

        Parameters
        ----------
        step: int
            index of optimization step
        voltages_ri: casadi.DM
            node voltages calculated by previous calculation step
        k: casadi.DM
            factors calculated by previous calculation step
        objectives: str (optional)
            optional, default ''
            string of characters 'I'|'P'|'Q'|'V'|'L'|'C'|'T' or empty string ''
        constraints: str (optional)
            optional, default ''
            string of characters 'I'|'P'|'Q'|'V' or empty string ''
        floss: float
            optional, default is 1.0
            multiplier for branch losses

        Returns
        -------
        tuple
            * dict
                * ['model']
                * ['expressions']
                * ['factordata']
                * ['Inode_inj']
                * ['objective'], str
                * ['constraints'], str
                * ['lbg']
                * ['ubg']
            * numpy.array, values of val_factors"""
        # calculate values of previous step
        voltages_ri2 = ri_to_ri2(voltages_ri)
        factordata = ft.make_factordata(model, gen_factor_symbols, step, k)
        kpq, ftaps, values_of_factors = ft.separate_factors(factordata, k)
        batch_values = get_batch_values(
            model, voltages_ri2, kpq, ftaps, constraints)
        return  make_step_data(
            step=step,
            f_prev=values_of_factors,
            objectives=objectives,
            constraints=constraints,
            values_of_constraints=batch_values,
            floss=floss)
    return make_step_data, next_step_data

def optimize_step(
    *, model, expressions, factordata, Inode_inj, objective,
    constraints, lbg, ubg, Vnode_ri_ini=None):
    """Runs one optimization step.

    Calculates initial voltages if not provided.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    expressions: dict
        estimation data
        * 'Vnode_syms', casadi.SX (shape n,2)
        * 'Vslack_syms', casadi.SX (shape n,2)
    factordata: Factordata
        * .vars, casadi.SX, column vector, symbols for variables
          of scaling factors
        * .consts, casadi.SX, column vector, symbols for constants
          of scaling factors
        * .values_of_vars, casadi.DM, column vector, initial values
          for vars
        * .values_of_consts, casadi.DM, column vector, values for consts
    Inode_inj: casadi.SX (shape n,2)
        * Inode_inj[:,0] - Ire, real part of current injected into node
        * Inode_inj[:,1] - Iim, imaginary part of current injected into node
    objective: casadi.SX
        expression to minimize
    constraints: casadi.SX
        expressions for additional constraints, values to be kept zero
        (default constraints are Inode==0)
    lbg: casadi.DM (shape m,1)
        lower bound of additional constraints
    ubg: casadi.DM (shape m,1)
        upper bound of additional constraints
    Vnode_ri_ini: array_like (shape 2n,1)
        optional, default is None
        initial node voltages separated real and imaginary values

    Returns
    -------
    succ : bool
        success?
    voltages_ri : numpy.array, complex (shape 2n,1)
        calculated node voltages for n nodes,
        n values for real part then n values for imaginary part
    factors : numpy.array, float (shape m,1)
        values for factors"""
    if Vnode_ri_ini is None:
        # power flow calculation for initial voltages
        succ, Vnode_ri_ini = calculate_power_flow2(
            model, expressions, factordata, Inode_inj)
        assert succ, 'calculation of power flow failed'
    optimize = get_optimize_vk(model, expressions)
    succ, x = optimize(
        Vnode_ri_ini, factordata, Inode_inj, objective, constraints, lbg, ubg)
    # result processing
    #   node voltages are first values in result, factors follow
    count_of_Vvalues = 2 * model.shape_of_Y[0]
    if count_of_Vvalues < x.size1():
        # split result into voltage part and factor part
        v, f = casadi.vertsplit(x, count_of_Vvalues)
        return succ, v.toarray(), f.toarray()
    # just voltages, no factors
    return succ, x.toarray(), _EMPTY_0r1c

def optimize_steps(model, gen_factorsymbols, step_params=(), vminsqr=_VMINSQR):
    """Estimates grid status stepwise.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric network
    gen_factorsymbols: casadi.SX, shape(n,1)
        symbols of generic (for each step) decision variables or parameters
    step_params: array_like
        dict {'objectives': str, objectives,
              'constraints': str, constraints,
              'floss': float, factor for losses}
            if empty the function calculates power flow,
            each dict triggers an estimation step
        * objectives, ''|'P'|'Q'|'I'|'V'|'L'|'C' (also string of characters)
          'P' - objective function is created with terms for active power
          'Q' - objective function is created with terms for reactive power
          'I' - objective function is created with terms for electric current
          'V' - objective function is created with terms for voltage
          'L' - objective function is created with terms for losses in branches
          'C' - objective function is created with terms for cost
          'T' - objective function is created with terms of model.terms
        * constraints, ''|'P'|'Q'|'I'|'V'|'B' (string or tuple of characters)
          'P' - adds constraints keeping the initial values
                of active powers at the location of given values
          'Q' - adds constraints keeping the initial values
                of reactive powers at the location of given values
          'I' - adds constraints keeping the initial values
                of electric current at the location of given values
          'V' - adds constraints keeping the initial values
                of voltages at the location of given values
          'B' - consider voltage limits (if any)
    vminsqr: float (default _VMINSQR)
        minimum voltage at injection, if the voltage is below this limit
        affected injections are interpolated and do not follow the given
        curve anymore, the interploation is towards a linear function reaching
        P,Q == 0,0 when |V| == 0, the given value of the argument must be
        the squared value of the limit

    Yields
    ------
    tuple
        * int, index of estimation step,
          (-1 for initial power flow calculation, 0 for first estimation)
        * bool, success?
        * voltages_ri, casadi.DM (shape 2n,1)
          calculated node voltages, real voltages then imaginary voltages
        * k, casadi.DM (shape m,1)
          factors for injections
        * factordata, Factordata,
          factor data of step"""
    make_step_data, next_step_data = get_step_data_fns(
        model, gen_factorsymbols)
    step_data = make_step_data(step=0)
    factordata = step_data['factordata']
    # power flow calculation for initial voltages
    succ, voltages_ri = calculate_power_flow2(
        model, step_data['expressions'], factordata, step_data['Inode_inj'])
    values_of_vars = factordata.values_of_vars
    yield -1, succ, voltages_ri, values_of_vars, factordata
    for step, kv in enumerate(step_params):
        objectives = kv.get('objectives', '').upper()
        constraints = kv.get('constraints', '').upper()
        floss = kv.get('floss', 1.)
        factor_values = ft.get_factor_values(factordata, values_of_vars)
        step_data = next_step_data(
            step=step, voltages_ri=voltages_ri, k=factor_values,
            objectives=objectives, constraints=constraints, floss=floss)
        factordata = step_data['factordata']
        # estimation
        succ, voltages_ri, values_of_vars = optimize_step(
            **step_data, Vnode_ri_ini=voltages_ri)
        yield step, succ, voltages_ri, values_of_vars, factordata

def get_Vcx_factors(factordata, voltages_ri, factorvalues):
    """Helper. Arranges solver result.

    Parameters
    ----------
    factordata: Factordata
        * .values_of_consts, float
            column vector, values for consts
        * .var_const_to_factor
            int, index_of_factor=>index_of_var_const
            converts var_const to factor (var_const[var_const_to_factor])
        * .var_const_to_kp
            int, converts var_const to kp, one active power scaling factor
            for each injection (var_const[var_const_to_kp])
        * .var_const_to_kq
            int, converts var_const to kq, one reactive power scaling factor
            for each injection (var_const[var_const_to_kq])
    voltages_ri : casadi.DM (shape 2n,1)
        real part of node voltages, imaginary part of node voltages
    factorvalues: casasdi.DM
        values of decision variables

    Returns
    -------
    tuple:
        * numpy.array (shape n,1), complex, node voltages
        * numpy.array (shape m,2), float, scaling factors per injection
        * numpy.array (shape n,1), float, positions"""
    kpq, pos, vars_consts = ft.separate_factors(factordata, factorvalues)
    V = ri_to_complex(voltages_ri)
    return V, kpq, pos

def estimate(model, step_params=(), vminsqr=_VMINSQR):
    """Estimates grid status stepwise.

    Parameters
    ----------
    model: egrid.model.Model

    step_params: array_like
        dict {'objectives': objectives, 'constraints': constraints}
            if empty the function calculates power flow,
            each dict triggers an estimation step
        * objectives, ''|'P'|'Q'|'I'|'V'|'L'|'C'|'T' (also string of characters)
          'P' - objective function is created with terms for active power
          'Q' - objective function is created with terms for reactive power
          'I' - objective function is created with terms for electric current
          'V' - objective function is created with terms for voltage
          'L' - objective function is created with terms for losses of branches
          'C' - objective function is created with terms for cost
          'T' - objective function is created with terms of model.terms
        * constraints, ''|'P'|'Q'|'I'|'V'|'B' (also string of characters)
          'P' - adds constraints keeping the initial values
                of active powers at the location of given
                active power values during this step
          'Q' - adds constraints keeping the initial values
                of reactive powers at the location of given
                reactive power values during this step
          'I' - adds constraints keeping the initial values
                of electric current at the location of given
                current values during this step
          'V' - adds constraints keeping the initial values
                of voltages at the location of given
                voltage values during this step
          'B' - consider voltage limits (if any, 'B' for bounds)
    vminsqr: float (default _VMINSQR)
        minimum

    Yields
    ------
    tuple
        * int, index of estimation step,
          (initial power flow calculation result is -1, first estimation is 0)
        * bool, success?
        * voltages_cx : numpy.array, complex (shape n,1)
            calculated complex node voltages
        * pq_factors : numpy.array, float (shape m,2)
            scaling factors for injections
        * tappositions"""
    gen_factor_symbols = ft._create_symbols_with_ids(
        model.factors.gen_factordata.index)
    return (
        (step, succ, *get_Vcx_factors(factordata, v_ri, factorvalues))
        for step, succ, v_ri, factorvalues, factordata in optimize_steps(
            model, gen_factor_symbols, step_params, vminsqr))
