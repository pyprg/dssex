# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 11:28:52 2022

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

@author: pyprg
"""
import casadi
import pandas as pd
from functools import partial
from scipy.sparse import coo_matrix
from egrid.builder import DEFAULT_FACTOR_ID, defk, Loadfactor
from itertools import chain
from collections import namedtuple
# square of voltage magnitude, minimum value for load curve, 
#   if value is below _VMINSQR the load curves for P and Q converge
#   towards a linear load curve which is 0 when V=0; P(V=0)=0, Q(V=0)=0
_VMINSQR = 0.8
# value of zero check, used for load curve calculation    
_EPSILON = 1e-12
_EMPTY_TUPLE = ()

def get_tap_factors(branchtaps, position_vars):
    """Creates vars for tap positions, expressions for longitudinal and
    transversal factors of branches.
    
    Parameters
    ----------
    branchtaps: pandas.DataFrame (id of taps)
        * .Vstep, float voltage diff per tap
        * .positionneutral, int
    position_vars: casadi.SX
        vector of positions for terms with tap
    
    Returns
    -------
    tuple
        * casadi.SX, longitudinal factors
        * transversal factors"""
    if position_vars.size1():     
        # longitudinal factor 
        flo = (1 
            - casadi.SX(branchtaps.Vstep) 
              * (position_vars - branchtaps.positionneutral))
    else:
        flo = casadi.SX(0, 1)
    return casadi.horzcat(flo, casadi.constpow(flo, 2))    

def _mult_gb_by_tapfactors(
        gb_mn_mm, flo_ftr, index_of_term, index_of_other_term):
    """Multiplies conductance and susceptance of branch terminals in order
    to consider positions of taps.
    
    Parameters
    ----------
    gb_mn_mm: casadi.SX
        matrix of 4 column vectors, one row for each branch terminal
        gb_mn_mm[:,0] - g_mn
        gb_mn_mm[:,1] - b_mn
        gb_mn_mm[:,2] - g_mm
        gb_mn_mm[:,3] - b_mm
    flo_ftr: casadi.SX
        matrix of 2 column vectors, one row for each tap position
        flo_ftr[.,0] - longitudinal factor
        flo_ftr[.,1] - transversal factor
    index_of_term: array_like
        tap -> index of terminal
    index_of_other_term: array_like
        tap -> index of other terminal (of same branch)
    
    Returns
    -------
    casadi.SX
        gb_mn_mm[:,0] - g_mn
        gb_mn_mm[:,1] - b_mn
        gb_mn_mm[:,2] - g_mm
        gb_mn_mm[:,3] - b_mm"""
    assert flo_ftr.size1(), "factors required"
    assert len(index_of_term), "indices of terminals required"
    assert len(index_of_other_term), "indices of other terminals required"
    flo = flo_ftr[:, 0]
    gb_mn_mm[index_of_term, :2] *= flo
    gb_mn_mm[index_of_other_term, :2] *= flo
    # transversal factor
    gb_mn_mm[index_of_term, 2:] *= flo_ftr[:, 1]
    return gb_mn_mm 

def _create_gb_expressions(terms):
    g_mn = casadi.SX(terms.g_lo)
    b_mn = casadi.SX(terms.b_lo)
    g_mm = casadi.SX(terms.g_tr_half)
    b_mm = casadi.SX(terms.b_tr_half)
    return casadi.horzcat(g_mn, b_mn, g_mm, b_mm)

def _create_gb2(
        index_of_node, index_of_other_node, count_of_pfcnodes, gb_mn_mm):
    """Creates conductance matrix G and susceptance matrix B.
    
    Parameters
    ----------
    index_of_node: array_like
        int, branchterminal -> index_of_node
    index_of_other_node
        int, branchterminal -> index_of_other_node (same branch, other side)
    count_of_pfcnodes: int
        number of power flow calculation nodes
    gb_mn_mm: casadi.SX
        matrix of 4 column vectors, one row for each branch terminal
        gb_mn_mm[:,0] - g_mn
        gb_mn_mm[:,1] - b_mn
        gb_mn_mm[:,2] - g_mm
        gb_mn_mm[:,3] - b_mm
    
    Returns
    -------
    tuple
        * casadi.SX - conductance matrix
        * casadi.SX - susceptance matrix"""
    G = casadi.SX(count_of_pfcnodes, count_of_pfcnodes)
    B = casadi.SX(count_of_pfcnodes, count_of_pfcnodes)
    # mutual conductance/susceptance, mn
    for r, c, g, b in zip(
            index_of_node,
            index_of_other_node,
            gb_mn_mm[:,0].elements(),
            gb_mn_mm[:,1].elements()):
        G[r,c] -= g
        G[r,r] += g
        B[r,c] -= b
        B[r,r] += b
    # self conductance/susceptance, mm
    for idx, g, b in zip(
            index_of_node,
            gb_mn_mm[:,2].elements(),
            gb_mn_mm[:,3].elements()):
        G[idx,idx] += g
        B[idx,idx] += b
    return G, B

def _create_gb(model, position_vars):
    """Creates the branch conductance matrix and the branch susceptance matrix.
    
    Parameters
    ----------
    model: egrid.model.Model
    
    position_vars: casadi.SX
        one variable for each tap position
    
    Returns
    -------
    tuple
        * casadi.SX - conductance matrix
        * casadi.SX - susceptance matrix"""
    terms = model.branchterminals
    gb_mn_mm = _create_gb_expressions(terms)
    if position_vars.size1():
        branchtaps = model.branchtaps
        flo_ftr = get_tap_factors(branchtaps, position_vars)
        gb_mn_mm = _mult_gb_by_tapfactors(
            gb_mn_mm, 
            flo_ftr, 
            branchtaps.index_of_term, 
            branchtaps.index_of_other_term)
    return _create_gb2(
        terms.index_of_node, 
        terms.index_of_other_node,
        model.shape_of_Y[0],
        gb_mn_mm)
    
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

def create_gb_matrix(model, position_vars):
    """Creates conductance matrix G and susceptance matrix B.
    
    Parameters
    ----------
    model: egrid.model.Model
    
    position_vars: casadi.SX
        one variable for each tap position
    
    Returns
    -------
    tuple
        * casadi.SX - conductance matrix
        * casadi.SX - susceptance matrix"""
    G, B = _create_gb(model, position_vars)
    count_of_slacks = model.count_of_slacks
    return (
        _reset_slack_1(G, count_of_slacks), 
        _reset_slack_0(B, count_of_slacks))

def create_Vvars(count_of_nodes):
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
    
def create_mapping(from_index, to_index):
    """Creates a matrix M for mapping vectors.
    
    Parameters
    ----------
    from_index: array_like
        int
    to_index: array_like
        int
    
    Returns
    -------
    casadi.SX
        matrix of float 1.0 with shape=(to_index, from_index)"""
    number_of_rows = len(to_index)
    number_of_columns = len(from_index)
    return casadi.SX(coo_matrix(
        ([1.]*number_of_rows, (to_index, from_index)), 
        shape=(number_of_rows, number_of_columns), dtype=float))

def calculate_Y_by_V(G, B, Vreim):
    V_node = casadi.vertcat(Vreim[:,0], Vreim[:,1])
    return casadi.blockcat([[G, -B], [B,  G]]) @ V_node

#
# scaling factors
#

def _get_step_factor_to_injection_part(
        injectionids, assoc_frame, step_factors, count_of_steps):
    """Arranges ids for all steps and injections.

    Parameters
    ----------
    injectionids: pandas.Series
        str, IDs of all injecions
    assoc_frame: (str (step), str (injid), 'p'|'q' (part))
        * .id, str
    step_factors: pandas.DataFrame

    count_of_steps: int
        number of optimization steps

    Returns
    -------
    pandas.Dataframe (int (step), str (id of injection))
        * .injid, str
        * .part, 'p'|'q'"""
    # all injections, create step, id, (pq) for all injections
    index_all = pd.MultiIndex.from_product(
        [range(count_of_steps), injectionids, ('p', 'q')],
        names=('step', 'injid', 'part'))
    # step injid part => id
    return (
        assoc_frame
        .reindex(index_all, fill_value=DEFAULT_FACTOR_ID).reset_index()
        .set_index(['step', 'id'])
        .join(step_factors[[]]))

def _get_factor_ini_values(myfactors, symbols):
    """Returns expressions for initial values of scaling variables/parameters.

    Parameters
    ----------
    myfactors: pandas.DataFrame

    symbols: pandas.Series

    Returns
    -------
    pandas.Series
        casadi.SX"""
    unique_factors = myfactors.index
    prev_index = pd.MultiIndex.from_arrays(
        [unique_factors.get_level_values(0) - 1, myfactors.id_of_source.array])
    ini = pd.Series(symbols, index=unique_factors).reindex(prev_index)
    ini.index = unique_factors
    # transfer data from value in case of missing source data
    ini_isna = ini.isna()
    ini[ini_isna] = -1
    return ini.astype(dtype='Int64')

def _get_default_factors(count_of_steps):
    """Generates one default factor per step.
    
    Parameters
    ----------
    count_of_steps: int
        number of factors to generate
    
    Parameters
    ----------
    pandas.DataFrame"""
    return (
        pd.DataFrame(
            defk(id_=DEFAULT_FACTOR_ID, type_='const', value=1.0, 
                 min_=1.0, max_=1.0, step=range(count_of_steps)),
            columns=Loadfactor._fields)
        .set_index(['step', 'id']))

def _factor_index_per_step(factors):
    """Creates an index (0...n) for each step. 
    (Alternative for _get_default_factors)
    
    Parameters
    ----------
    factors: pandas.DataFrame (step, id)->...
    
    Returns
    -------
    pandas.Series"""
    return pd.Series(
        chain.from_iterable(
            (range(len(factors.loc[step]))) 
            for step in factors.index.levels[0]),
        index=factors.index,
        name='index_of_symbol')

def _factor_index(factors):
    """Creates an index for all factors (includes all steps).
    
    Parameters
    ----------
    factors: pandas.DataFrame (step, id)->...
    
    Returns
    -------
    pandas.Series"""
    return pd.Series(
        list(range(len(factors.index))),
        index=factors.index,
        name='index_of_symbol')

def get_load_scaling_factor_data(
        injectionids, given_factors, assoc, count_of_steps):
    """Creates and arranges indices for scaling factors and initialization
    of scaling factors.

    Parameters
    ----------
    injectionids: pandas.Series
        str, identifiers of injections
    given_factors: pandas.DataFrame
        * int, step
        * ...
    assoc: pandas.DataFrame (int (step), str (injid))
        * str (id of factor)
    count_of_steps: int
        number of optimization steps

    Returns
    -------
    tuple
        * pandas.DataFrame, all scaling factors
        * pandas.DataFrame, injections with scaling factors"""
    # given factors
    # step, id_of_factor => id_of_injection, 'id_p'|'id_q'
    step_injection_part_factor = _get_step_factor_to_injection_part(
        injectionids, assoc, given_factors, count_of_steps)
    # remove factors not needed, add default (nan) factors if necessary
    required_factors_index = step_injection_part_factor.index.unique()
    required_factors = given_factors.reindex(required_factors_index)
    # ensure existence of default factors when needed
    default_factors = _get_default_factors(count_of_steps)
    # replace nan with values (for required default factors)
    factors = required_factors.combine_first(default_factors)
    index_of_symbol = _factor_index_per_step(factors)
    factors['index_of_symbol'] = index_of_symbol
    # add data for initialization
    factors['index_of_source'] = _get_factor_ini_values(
        factors, index_of_symbol)
    if step_injection_part_factor.shape[0]:
        injection_factors = (
            step_injection_part_factor
            .join(factors.index_of_symbol)
            .reset_index()
            .set_index(['step', 'injid', 'part'])
            .unstack('part')
            .droplevel(0, axis=1))
        injection_factors.columns=['id_p', 'id_q', 'kp', 'kq']
    else:
        injection_factors = pd.DataFrame(
            [],
            columns=['id_p', 'id_q', 'kp', 'kq'],
            index=pd.MultiIndex.from_arrays(
                [[],[]], names=['step', 'injid']))
    injids = injection_factors.index.get_level_values(1)
    index_of_injection = (
        pd.Series(injectionids.index, index=injectionids)
        .reindex(injids)
        .array)
    injection_factors['index_of_injection'] = index_of_injection
    factors.reset_index(inplace=True)
    factors.set_index(['step', 'type', 'id'], inplace=True)
    return factors, injection_factors

def get_factors(model, count_of_steps=1):
    """Creates identifiers and indices of scaling factors. Creates mapping
    from injections to scaling factors. The mapping is specific for each
    calculation step, injection and scaled part of power (eiher active or 
    reactive power)

    Parameters
    ----------
    model: 
    
    count_of_steps: int
        number of optimization steps (default is 1)

    Returns
    -------
    tuple
        pandas.DataFrame"""
    return get_load_scaling_factor_data(
        model.injections.id,
        model.load_scaling_factors,
        model.injection_factor_associations,
        count_of_steps)

def _groupby_step(df):
    """Resets index of pandas.Dataframe df and group the resulting frame by
    column 'step'.
    
    Parameters
    ----------
    df: pandas.DataFrame
        either has column 'step' or one part of the index has name 'step'
        
    Returns
    -------
    pandas.core.groupby.generic.DataFrameGroupBy"""
    df_ = df.reset_index()
    return df_.groupby('step')

def _create_symbols_with_ids(ids):
    """Creates a column vector of casadi symbols with given identifiers.
    
    Parameters
    ----------
    ids: iterable of str
    
    Returns
    -------
    casadi.SX"""
    return casadi.vertcat(*(casadi.SX.sym(id_) for id_ in ids))

def _get_values_of_symbols(factor_data, value_of_previous_step):
    """Returns values for symbols. If the symbols is a variable the value
    is the initial value. Values can either be given explicitely or are
    calculated in the previous calculation step.
    
    Parameters
    ----------
    factor_data: pandas.DataFrame
        * .index_of_symbol, int
        * .value, float
        * .index_of_source, int
    value_of_previous_step: casadi.DM
        vector of float
    
    Returns
    -------
    casadi.DM
        column vector of float"""
    values = casadi.DM.zeros(len(factor_data), 1)
    # explicitely given values not calculated in previous step
    is_given = factor_data.index_of_source < 0
    given = factor_data[is_given]
    values[given.index_of_symbol] = factor_data.value
    # values calculated in previous step
    calc = factor_data[~is_given]
    values[calc.index_of_symbol] = value_of_previous_step[calc.index_of_source]
    return values
    
def _select_type(factor_data, vecs, type_):
    """Creates column vectors from vecs by extracting elements by their 
    indices.
    
    Parameters
    ----------
    factor_data: pandas.DataFrame
        * .type
        * .index_of_symbol
    vecs: iterable
        casadi.SX or casadi.DM, column vector
    type_: 'const' | 'var'
        
    Returns
    -------
    tuple
        * casadi.SX
        * casadi.DM"""
    indices = factor_data[factor_data.type==type_].index_of_symbol
    return (v[indices, 0] for v in vecs)

_k_prev_default = casadi.DM.zeros(0,1)

Scalingdata = namedtuple(
    'Scalingfactors',
    'kp kq vars_ values_of_vars consts values_of_consts symbols')
Scalingdata.__doc__="""
Symbols of variables and constants for scaling factors.

Parameters
----------
kp: casadi.SX
    column vector, symbols for scaling factor of active power per injection
kq: casadi.SX
    column vector, symbols for scaling factor of reactive power per injection
vars_: casadi.SX
    column vector, symbols for variables of scaling factors
values_of_vars: casadi.DM
    column vector, initial values for vars_
consts: casadi.SX
    column vector, symbols for constants of scaling factors
values_of_consts: casadi.DM
    column vector, values for consts
symbols: casadi.SX
    vector of all symbols (variables and constants) for extracting values 
    which shall be passed to next step 
    (function 'get_scaling_data', parameter 'k_prev')"""
    
def get_scaling_data(
        factor_step_groups, injection_factor_step_groups, 
        step=0, k_prev=_k_prev_default):
    """Prepares data of scaling factors per step.
    
    Parameters
    ----------
    factor_step_groups: pandas.groupby
    
    injection_factor_step_groups: pandas.groupby
    
    step: int
        step number 0, 1, ...
    k_prev: casadi.DM
        values of scaling factors from previous step, variables and constants
        
    Returns
    -------
    Scalingdata"""
    factors = factor_step_groups.get_group(step)
    injections_factors = (
        injection_factor_step_groups
        .get_group(step)
        .sort_values(by='index_of_injection'))
    symbols = _create_symbols_with_ids(factors.id)
    values = _get_values_of_symbols(factors, k_prev)
    symbols_values = partial(_select_type, factors, [symbols, values])
    symbols_of_consts, values_of_consts = symbols_values('const')
    symbols_of_vars, values_of_vars = symbols_values('var')
    return Scalingdata(
        kp=symbols[injections_factors.kp],
        kq=symbols[injections_factors.kq],
        vars_=symbols_of_vars,
        values_of_vars=values_of_vars,
        consts=symbols_of_consts,
        values_of_consts=values_of_consts,
        symbols=symbols)
    
def get_scaling_data_fn(model, count_of_steps=1):
    """Creates a function for creating Scalingdata.
    
    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    count_of_steps: int
        number of calculation steps > 0 
    
    Returns
    -------
    Scalingdata:
        kp: casadi.SX
            column vector, symbols for scaling factor 
            of active power per injection
        kq: casadi.SX
            column vector, symbols for scaling factor 
            of reactive power per injection
        vars_: casadi.SX
            column vector, symbols for variables of scaling factors
        values_of_vars: casadi.DM
            column vector, initial values for vars_
        consts: casadi.SX
            column vector, symbols for constants of scaling factors
        values_of_consts: casadi.DM
            column vector, values for consts"""
    assert 0 < count_of_steps, "count_of_steps must be an int greater than 0"
    factors, injection_factors = get_factors(model, count_of_steps)
    return partial(
        get_scaling_data,
        _groupby_step(factors),
        _groupby_step(injection_factors))
    