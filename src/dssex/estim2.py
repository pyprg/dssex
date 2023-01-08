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
import numpy as np
from functools import partial
from collections import defaultdict
from itertools import chain
from collections import namedtuple
from scipy.sparse import coo_matrix
from egrid.input import DEFAULT_FACTOR_ID, defk, Loadfactor
from src.dssex.injections import calculate_cubic_coefficients
from src.dssex.batch import get_values, get_batches, value_of_voltages
# square of voltage magnitude, default value, minimum value for load curve,
#   if value is below _VMINSQR the load curves for P and Q converge
#   towards a linear load curve which is 0 when V=0; P(V=0)=0, Q(V=0)=0
_VMINSQR = 0.8**2
# value of zero check, used for load curve calculation
_EPSILON = 1e-12
# empty vector of values
_DM_0r1c = casadi.DM(0,1)
# empty vector of expressions
_SX_0r1c = casadi.SX(0,1)

def create_symbols_with_ids(ids):
    """Creates a column vector of casadi symbols with given identifiers.

    Parameters
    ----------
    ids: iterable of str

    Returns
    -------
    casadi.SX"""
    return casadi.vertcat(*(casadi.SX.sym(id_) for id_ in ids))

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

def _mult_gb_by_tapfactors(
        gb_mn_tot, foffd, index_of_term, index_of_other_term):
    """Multiplies conductance and susceptance of branch terminals in order
    to consider positions of taps.

    Parameters
    ----------
    gb_mn_mm: casadi.SX
        matrix of 4 column vectors, one row for each branch terminal
        gb_mn_tot[:,0] - g_mn
        gb_mn_tot[:,1] - b_mn
        gb_mn_tot[:,2] - g_tot
        gb_mn_tot[:,3] - b_tot
    foffd: casadi.SX
        vector - off-diagonal factor
    index_of_term: array_like
        taps -> index of terminal
    index_of_other_term: array_like
        taps -> index of other terminal (of same branch)

    Returns
    -------
    casadi.SX
        gb_mn_tot[:,0] - g_mn
        gb_mn_tot[:,1] - b_mn
        gb_mn_tot[:,2] - g_tot
        gb_mn_tot[:,3] - b_tot"""
    assert foffd.size1(), "factors required"
    assert len(index_of_term), "indices of terminals required"
    assert len(index_of_other_term), "indices of other terminals required"
    # mn
    gb_mn_tot[index_of_term, :2] *= foffd
    gb_mn_tot[index_of_other_term, :2] *= foffd
    # tot
    gb_mn_tot[index_of_term, 2:] *= (foffd*foffd)
    return gb_mn_tot

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
# scaling factors
#

def _get_step_factor_to_injection_part(
        injectionids, assoc_frame, step_factors, count_of_steps):
    """Arranges ids for all calculation steps and injections.

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
    ini.fillna(-1, inplace=True)
    return ini.astype(dtype='Int64')

def _get_default_factors(count_of_steps):
    """Generates one default scaling factor for each step. The factor is
    of type 'const' has value 1.0, minimum and maximum are 1.0 too.

    Parameters
    ----------
    count_of_steps: int
        number of factors to generate

    Parameters
    ----------
    pandas.DataFrame (index: ['step', 'id'])
        value in column 'id' of index is the string 
        of egrid.builder.DEFAULT_FACTOR_ID,
        columns according to fields of egrid.builder.Loadfactor"""
    return (
        pd.DataFrame(
            defk(id_=DEFAULT_FACTOR_ID, type_='const', value=1.0,
                 min_=1.0, max_=1.0, step=range(count_of_steps)),
            columns=Loadfactor._fields)
        .set_index(['step', 'id']))

def _factor_index_per_step(factors):
    """Creates an index (0...n) for each step.

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

def get_load_scaling_factor_data(
        injectionids, given_factors, assoc_frame, count_of_steps):
    """Creates and arranges indices for scaling factors and initialization
    of scaling factors.

    Parameters
    ----------
    injectionids: pandas.Series
        str, identifiers of injections
    given_factors: pandas.DataFrame (step, id)
        * .type, 'var' | 'const'
        * .id_of_source , str, source from previous step for (initial) value
        * .value, float, (initial) value if no valid source from previous step
        * .min, float, smallest value, constraint during optimization
        * .max, float, greatest value, constraint during optimization
    assoc_frame: pandas.DataFrame (int (step), str (injid), 'p'|'q' (part))
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
        injectionids, assoc_frame, given_factors, count_of_steps)
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
    model: egrid.model.Model
        data of electric grid
    count_of_steps: int
        number of optimization steps (default is 1)

    Returns
    -------
    tuple
        * pandas.DataFrame, all scaling factors
        * pandas.DataFrame, injections with scaling factors"""
    return get_load_scaling_factor_data(
        model.injections.id,
        model.load_scaling_factors,
        model.injection_factor_associations,
        count_of_steps)

def _groupby_step(df):
    """Resets index of pandas.Dataframe df and groups the data in the 
    resulting frame by column 'step'.

    Parameters
    ----------
    df: pandas.DataFrame
        either has column 'step' or one part of the index has name 'step'

    Returns
    -------
    pandas.core.groupby.generic.DataFrameGroupBy"""
    df_ = df.reset_index()
    return df_.groupby('step')

def _get_values_of_symbols(factor_data, value_of_previous_step):
    """Returns values for symbols. If a symbol is a variable the value
    is the initial value. Values are either given explicitely or are
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
    if len(given):
        values[given.index_of_symbol] = given.value.to_numpy().reshape(-1,1)
    # values calculated in previous step
    calc = factor_data[~is_given]
    if len(calc):
        assert (0 < value_of_previous_step.size), \
            'missing value_of_previous_step'
        values[calc.index_of_symbol] = (
            value_of_previous_step[calc.index_of_source.astype(int)])
    return values

def _select_type(vecs, row_index):
    """Creates column vectors from vecs by extracting elements by their
    (row-) indices.

    Parameters
    ----------
    vecs: iterable
        casadi.SX or casadi.DM, column vector
    factor_data: array_like
        int

    Returns
    -------
    iterator
        * casadi.SX / casadi.DM"""
    return (v[row_index, 0] for v in vecs)

Scalingdata = namedtuple(
    'Scalingdata',
    'kpq kvars values_of_vars kvar_min kvar_max kconsts values_of_consts '
    'var_const_to_factor var_const_to_kp var_const_to_kq')
Scalingdata.__doc__="""
Symbols of variables and constants for scaling factors.

Parameters
----------
kpq: casadi.SX
    two column vectors, symbols for scaling factors of active and
    reactive power per injection
kvars: casadi.SX
    column vector, symbols for variables of scaling factors
values_of_vars: casadi.DM
    column vector, initial values for kvars
kvar_min: casadi.DM
    lower limits of kvars
kvar_max: casadi.DM
    upper limits of kvars
kconsts: casadi.SX
    column vector, symbols for constants of scaling factors
values_of_consts: casadi.DM
    column vector, values for consts
var_const_to_factor: array_like
    int, index_of_factor=>index_of_var_const
    converts var_const to factor (var_const[var_const_to_factor])
var_const_to_kp: array_like
    int, converts var_const to kp, one active power scaling factor for
    each injection (var_const[var_const_to_kp])
var_const_to_kq: array_like
    int, converts var_const to kq, one reactive power scaling factor for
    each injection (var_const[var_const_to_kq])"""

def _make_DM_vector(array_like):
    """Creates a casadi.DM vector from array_like.

    Parameters
    ----------
    array_like: array_like

    Returns
    -------
    casadi.DM"""
    return casadi.DM(array_like) if len(array_like) else _DM_0r1c

def get_scaling_data(
        factor_step_groups, injection_factor_step_groups,
        step=0, k_prev=_DM_0r1c):
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
    Scalingdata
        kpq: casadi.SX
            two column vectors, symbols for scaling factors of active and
            reactive power per injection
        kvars: casadi.SX
            column vector, symbols for variables of scaling factors
        values_of_vars: casadi.DM
            column vector, initial values for kvars
        kvar_min: casadi.DM
            lower limits of kvars
        kvar_max: casadi.DM
            upper limits of kvars
        kconsts: casadi.SX
            column vector, symbols for constants of scaling factors
        values_of_consts: casadi.DM
            column vector, values for consts
        var_const_to_factor: array_like
            int, index_of_factor=>index_of_var_const
            converts var_const to factor (var_const[var_const_to_factor])
        var_const_to_kp: array_like
            int, converts var_const to kp, one active power scaling factor
            for each injection (var_const[var_const_to_kp])
        var_const_to_kq: array_like
            int, converts var_const to kq, one reactive power scaling factor
            for each injection (var_const[var_const_to_kq])"""
    factors = factor_step_groups.get_group(step)
    # a symbol for each factor
    symbols = create_symbols_with_ids(factors.id)
    try:
        injections_factors = (
            injection_factor_step_groups
            .get_group(step)
            .sort_values(by='index_of_injection'))
    except KeyError:
        injections_factors = pd.DataFrame(
            [],
            columns=[
                'step', 'injid', 'id_p', 'id_q',
                'kp', 'kq', 'index_of_injection'])
    values = _get_values_of_symbols(factors, k_prev)
    symbols_values = partial(_select_type, [symbols, values])
    factors_var = factors[factors.type=='var']
    symbols_of_vars, values_of_vars = symbols_values(
        factors_var.index_of_symbol)
    factors_consts = factors[factors.type=='const']
    symbols_of_consts, values_of_consts = symbols_values(
        factors_consts.index_of_symbol)
    # the optimization result is provided as a vector of concatenated
    #   scaling variables and scaling constants, we prepare indices for
    #   mapping to kp/kq (which are ordered according to injections)
    var_const_to_factor = np.concatenate(
        [factors_var.index_of_symbol.array,
         factors_consts.index_of_symbol.array])
    return Scalingdata(
        # kp/kq used in expression building
        #   (columns of injections_factors.kp/kq store an index)
        kpq=casadi.horzcat(
            symbols[injections_factors.kp].reshape((-1,1)),
            symbols[injections_factors.kq].reshape((-1,1))),
        # kvars, variables for solver preparation
        kvars=symbols_of_vars,
        # initial values, argument in solver call
        values_of_vars=values_of_vars,
        # lower bound of scaling factors, argument in solver call
        kvar_min=_make_DM_vector(factors_var['min']),
        # upper bound of scaling factors, argument in solver call
        kvar_max=_make_DM_vector(factors_var['max']),
        # kconsts, constants for solver preparation
        kconsts=symbols_of_consts,
        # values of constants, argument in solver call
        values_of_consts=values_of_consts,
        # reordering of result
        var_const_to_factor=var_const_to_factor,
        var_const_to_kp=var_const_to_factor[injections_factors.kp],
        var_const_to_kq=var_const_to_factor[injections_factors.kq])

def get_scaling_factors(scaling_data, x_scaling):
    """Function for extracting scaling factors from the result provided
    by the solver.
    Enhances scaling factors calculated by optimization with constant
    scaling factors and rearranges the factors for further processing with
    method 'get_scaling_data'.

    Parameters
    ----------
    scaling_data: Scalingdata
        * .values_of_consts, float
            column vector, values for consts
        * .var_const_to_factor
            int, index_of_factor=>index_of_var_const
            converts var_const to factor (var_const[var_const_to_factor])
    x_scaling: casadi.DM
        result of optimization (subset)

    Result
    ------
    numpy.array
        float, scaling factors, ordered for further proccessing in next step as
        parameter 'k_prev' of function 'get_scaling_data'"""
    k_var_const = (
        casadi.vertcat(x_scaling, scaling_data.values_of_consts).toarray())
    return k_var_const[scaling_data.var_const_to_factor].reshape(-1)

def get_k(scaling_data, x_scaling):
    """Function for extracting scaling factors from the result provided
    by the solver.
    Enhances scaling factors calculated by optimization with constant
    scaling factors and reorders the factors according to order of injections.
    Returns kp and kq for each injection.
    The function reates a vector of values for scaling factors which are 
    decision variables and those which are constants. This vector is ordered 
    for use as initial scaling factor values in next estimation step.

    Parameters
    ----------
    scaling_data: Scalingdata
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
    x_scaling: casadi.DM
        result of optimization (subset)

    Result
    ------
    tuple
        * numpy.array (n,2), kp, kq for each injection
        * casadi.DM (m,1) kvar/const"""
    k_var_const = (
        casadi.vertcat(x_scaling, scaling_data.values_of_consts).toarray())
    kp = k_var_const[scaling_data.var_const_to_kp]
    kq = k_var_const[scaling_data.var_const_to_kq]
    return np.hstack([kp, kq]), k_var_const[scaling_data.var_const_to_factor]

def make_get_scaling_data(model, count_of_steps=1):
    """Creates a function for creating Scalingdata specific for a calculation
    step.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    count_of_steps: int
        0 < number of calculation steps

    Returns
    -------
    function
        (int, casadi.DM) -> (Scalingdata)
        (index_of_calculation_step, result_of_previous_step) -> (Scalingdata)
        Scalingdata:
            kp: casadi.SX
                column vector, symbols for scaling factor
                of active power per injection
            kq: casadi.SX
                column vector, symbols for scaling factor
                of reactive power per injection
            kvars: casadi.SX
                column vector, symbols for variables of scaling factors
            values_of_vars: casadi.DM
                column vector, initial values for kvars
            kconsts: casadi.SX
                column vector, symbols for constants of scaling factors
            values_of_consts: casadi.DM
                column vector, values for consts"""
    assert 0 < count_of_steps, \
        "count_of_steps must be an integer greater than 0"
    factors, injection_factors = get_factors(model, count_of_steps)
    return partial(
        get_scaling_data,
        _groupby_step(factors),
        _groupby_step(injection_factors))

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

def _injected_current(
        injections, node_to_inj, Vnode, scaling_data, vminsqr=_VMINSQR):
    """Creates expressions for current flowing into injections.
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
    scaling_data: Scalingdata
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
    PQscaled = scaling_data.kpq * (injections[['P10', 'Q10']].to_numpy() / 3)
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
        Pip = casadi.SX(0,1)
        Qip = casadi.SX(0,1)
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
    # greate gb_mn_mm
    gb_mn_tot = _create_gb_expressions(branchterminals)
    # create gb_mn_tot
    gb_mn_tot[:,2] += gb_mn_tot[:,0]
    gb_mn_tot[:,3] += gb_mn_tot[:,1]
    if position_syms.size1():
        foffd = get_tap_factors(branchtaps, position_syms)
        return _mult_gb_by_tapfactors(
            gb_mn_tot,
            foffd,
            branchtaps.index_of_term,
            branchtaps.index_of_other_term)
    return gb_mn_tot

def create_v_symbols_gb_expressions(model):
    """Creates symbols for node and slack voltages, tappositions,
    branch conductance/susceptance and an expression for Y @ V. The symbols
    and expressions are regarded constant over multiple calculation steps.
    Diagonal of slack rows are set to 1 for conductance and 0 for susceptance,
    other values of slack rows are set to 0 in matrix 'Y @ V'.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid

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
    count_of_pfcnodes = model.shape_of_Y[0]
    Vnode_syms = create_V_symbols(count_of_pfcnodes)
    position_syms = casadi.SX.sym('pos', len(model.branchtaps), 1)
    gb_mn_tot = _create_gb_mn_tot(
        model.branchterminals, model.branchtaps, position_syms)
    terms = model.branchterminals
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
        position_syms=position_syms,
        gb_mn_tot=gb_mn_tot,
        Y_by_V=multiply_Y_by_V(Vnode_syms, G, B))

#
# power flow calculation
#

def make_get_scaling_and_injection_data(
        model, Vnode_syms, vminsqr, count_of_steps):
    """Returns a function creating scaling_data and injection_data
    for a given step, in general expressions which are specific
    for the given step.

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
    count_of_steps: int
        0 < number of calculation steps

    Returns
    -------
    function
        (int, casadi.DM)
            -> (tuple: Scalingdata, casadi.SX)
        (index_of_step, scaling_factors_of_previous_step)
            -> (tuple: Scalingdata, injection_data)
        * Scalingdata
            kp: casadi.SX
                column vector, symbols for scaling factor of active power
                per injection
            kq: casadi.SX
                column vector, symbols for scaling factor of reactive power
                per injection
            kvars: casadi.SX
                column vector, symbols for variables of scaling factors
            values_of_vars: casadi.DM
                column vector, initial values for kvars
            kvar_min: casadi.DM
                lower limits of kvars
            kvar_max: casadi.DM
                upper limits of kvars
            kconsts: casadi.SX
                column vector, symbols for constants of scaling factors
            values_of_consts: casadi.DM
                column vector, values for consts
            symbols: casadi.SX
                vector of all symbols (variables and constants) for
                extracting values which shall be passed to next step
                (function 'get_scaling_data', parameter 'k_prev')
            values: casadi.DM
                float, given values of symbols (variables and constants)
        * casadi.SX (n,8)
            [:,0] Ire, current, real part
            [:,1] Iim, current, imaginary part
            [:,2] Pscaled, active power P10 multiplied by scaling factor kp
            [:,3] Pip, active power interpolated
            [:,4] Qscaled, reactive power Q10 multiplied by scaling factor kq
            [:,5] Qip, reactive power interpolated
            [:,6] Vabs_sqr, square of voltage magnitude at injections
            [:,7] interpolate?"""
    get_scaling_data = make_get_scaling_data(model, count_of_steps)
    injections = model.injections
    node_to_inj = casadi.SX(model.mnodeinj).T
    def get_scaling_and_injection_data(step=0, k_prev=_DM_0r1c):
        assert step < count_of_steps, \
            f'index "step" ({step}) must be smaller than '\
            f'value of paramter "count_of_steps" ({count_of_steps})'
        scaling_data = get_scaling_data(step, k_prev)
        # injected node current
        Iinj_data = _injected_current(
            injections, node_to_inj, Vnode_syms, scaling_data, vminsqr)
        return scaling_data, Iinj_data
    return get_scaling_and_injection_data

def get_expressions(model, count_of_steps, vminsqr=_VMINSQR):
    """Prepares data for estimation. Creates symbols and expressions.

    Parameters
    ----------
    model : egrid.model.Model
        data of electric grid
    count_of_steps : int
        number of estimation steps
    vminsqr : float, optional
        minimum voltage at loads for original load curve, squared

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
        * 'Y_by_V', casadi.SX, expression for Y @ V
        * 'get_scaling_and_injection_data', function
          (int, casadi.DM) -> (tuple - Scalingdata, casadi.SX)
          which is a function
          (index_of_step, scaling_factors_of_previous_step)
            -> (tuple - Scalingdata, injection_data)
        * 'inj_to_node', casadi.SX, matrix, maps from
          injections to power flow calculation nodes"""
    ed = create_v_symbols_gb_expressions(model)
    ed['get_scaling_and_injection_data'] = (
        make_get_scaling_and_injection_data(
            model, ed['Vnode_syms'], vminsqr, count_of_steps))
    ed['inj_to_node'] = casadi.SX(model.mnodeinj)
    return ed

def calculate_power_flow2(
        model, expr, scaling_data, Inode, tappositions=None, Vinit=None):
    """Solves the power flow problem using a rootfinding algorithm. The result
    is the initial voltage vector for the optimization.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    expr: dict
        optional
        * 'Vnode_syms', casadi.SX, expressions of node Voltages
        * 'Vre_slack_syms', casadi.SX, symbols of slack voltages, real part
        * 'Vim_slack_syms', casadi.SX, symbols of slack voltages,
           imaginary part
        * 'position_syms', casadi.SX, symbols of tap positions
        * 'Y_by_V', casadi.SX, expression for Y @ V
    scaling_data: Scalingdata
        optional
    Inode: casadi.SX (shape n,2)
        * Inode[:,0] - Ire, real part of current injected into node
        * Inode[:,1] - Iim, imaginary part of current injected into node
    tappositions: array_like
        optional
        int, positions of taps
    Vinit: array_like
        optional
        float, initial guess of node voltages

    Returns
    -------
    tuple
        * bool, success?
        * casadi.DM, float, voltage vector [real parts, imaginary parts]"""
    Vnode_syms = expr['Vnode_syms']
    Vslack_syms = expr['Vslack_syms'][:,0], expr['Vslack_syms'][:,1]
    parameter_syms=casadi.vertcat(
        *Vslack_syms,
        expr['position_syms'],
        scaling_data.kvars,
        scaling_data.kconsts)
    slacks = model.slacks
    Inode_ = _reset_slack_current(slacks.index_of_node, *Vslack_syms, Inode)
    Inode_ri = vstack(Inode_, 2)
    variable_syms = vstack(Vnode_syms, 2)
    fn_Iresidual = casadi.Function(
        'fn_Iresidual',
        [variable_syms, parameter_syms],
        [expr['Y_by_V'] + Inode_ri])
    rf = casadi.rootfinder(
        'rf',
        'nlpsol',
        fn_Iresidual,
        {'nlpsol':'ipopt'})
    count_of_pfcnodes = model.shape_of_Y[0]
    Vinit_ = (
        casadi.vertcat([1.]*count_of_pfcnodes, [0.]*count_of_pfcnodes)
        if Vinit is None else Vinit)
    tappositions_ = (
        model.branchtaps.position if tappositions is None else tappositions)
    # Vslack must be negative as Vslack_result + Vslack_in_Inode = 0
    #   because the root is searched for with formula: Y * Vresult + Inode = 0
    Vslack_neg = -slacks.V
    values_of_parameters=casadi.vertcat(
        np.real(Vslack_neg), np.imag(Vslack_neg),
        tappositions_,
        scaling_data.values_of_vars,
        scaling_data.values_of_consts)
    voltages = rf(Vinit_, values_of_parameters)
    return rf.stats()['success'], voltages

def calculate_power_flow(
        model, v_syms_gb_ex=None, tappositions=None, Vinit=None,
        vminsqr=_VMINSQR):
    """Solves the power flow problem using a rootfinding algorithm.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    v_syms_gb_ex: dict
        optional
        * 'Vnode_syms', casadi.SX, expressions of node Voltages
        * 'Vre_slack_syms', casadi.SX, symbols of slack voltages, real part
        * 'Vim_slack_syms', casadi.SX, symbols of slack voltages,
           imaginary part
        * 'position_syms', casadi.SX, symbols of tap positions
        * 'Y_by_V', casadi.SX, expression for Y @ V
    tappositions: array_like
        optional
        int, positions of taps
    Vinit: array_like
        optional
        float, initial guess of node voltages
    vminsqr: float
        optinal
        square of voltage, upper limit interpolation interval [0...vminsqr]

    Returns
    -------
    tuple
        * bool, success?
        * casadi.DM, float, voltage vector [real parts, imaginary parts]"""
    const_expr = (
        create_v_symbols_gb_expressions(model)
        if v_syms_gb_ex is None else v_syms_gb_ex)
    get_scaling_and_injection_data = make_get_scaling_and_injection_data(
        model, const_expr['Vnode_syms'], vminsqr, count_of_steps=1)
    scaling_data, Iinj_data = get_scaling_and_injection_data(step=0)
    inj_to_node = casadi.SX(model.mnodeinj)
    Inode = inj_to_node @ Iinj_data[:,:2]
    return calculate_power_flow2(
        model, const_expr, scaling_data, Inode, tappositions, Vinit)

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
    """Calculates active and reactive power flow
    from admittances of a branch and the voltages at its terminals. Assumes
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
                +-            -+           +-            -+
                | g_tot -b_tot |           |  g_tot b_tot |
        y_tot = |              |  y_tot' = |              |
                | b_tot  g_tot |           | -b_tot g_tot |
                +-            -+           +-            -+
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
    """Creates expressions of active and reactive power flow for a subset of
    branch terminals from admittances of branches and voltages at
    branch terminals. Assumes PI-equivalent circuits.

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
    return casadi.SX(0,1), casadi.SX(0,1)

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
    """Generates expressions for real and imaginary current flowing into
    given subset of branch terminals from expressions of branch admittances
    and voltages at branch terminals. Assumes PI-equivalient circuit.

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

def _make_get_value(values, quantity):
    """Helper, creates a function retrieving a value from
    ivalues, pvalues, qvalues or vvalues. Returns 'per-phase' values
    (one third of given P or Q).

    Parameters
    ----------
    values: pandas.DataFrame

    quantity: 'I'|'P'|'Q'|'V'

    Returns
    -------
    function
        (str) -> (float)
        (index) -> (value)"""
    vals = (
        values[quantity]
        if quantity in 'IV' else values[quantity] * values.direction / 3.)
    return lambda idx: vals[idx]

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
        * 'position_syms', casadi.SX, vector, symbols of tap positions
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
        id_of_batch:get_branch_expr(branchterminals.loc[df.index_of_term])
        for id_of_batch, df in get_batches(
            get_values(model, quantity),
            model.branchoutputs,
            'index_of_term')}

def _get_batch_expressions_inj(model, ipqv, quantity):
    """Creates a vector (casadi.SX, shape n,1) expressing injected absolute
    current, active power or reactive power. The expressions are based
    on the batch definitions.

    Parameters
    ----------
    model: egrid.model.Model
        model of electric network for calculation
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
    """Creates a vector (casadi.SX, shape n,1) expressing calculated
    values for absolute current, active power or reactive power. The
    expressions are based on the batch definitions.

    Parameters
    ----------
    model: egrid.model.Model
        model of electric network for calculation
    v_syms_gb_ex: dict
        * 'Vnode_syms', casadi.SX, vectors, symbols of node voltages
        * 'position_syms', casadi.SX, vector, symbols of tap positions
        * 'gb_mn_tot', casadi.SX, vectors, conductance and susceptance of
          branches connected to terminals
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

def get_batch_flow_expressions(model, v_syms_gb_ex, ipqv, quantity):
    """Creates a vector (casadi.SX, shape n,1) expressing the difference
    between measured and calculated values for absolute current, active power
    or reactive power. The expressions are based on the batch definitions.
    Intended use is building the objective.

    Parameters
    ----------
    model: egrid.model.Model
        model of electric network for calculation
    v_syms_gb_ex: dict
        * 'Vnode_syms', casadi.SX, vectors, symbols of node voltages
        * 'position_syms', casadi.SX, vector, symbols of tap positions
        * 'gb_mn_tot', casadi.SX, vectors, conductance and susceptance of
          branches connected to terminals
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

    Returns
    -------
    casadi.SX
        vector (shape n,1)"""
    assert quantity in 'IPQ', \
        f'quantity needs to be one of "I", "P" or "Q" but is "{quantity}"'
    values = get_values(model, quantity).set_index('id_of_batch')
    get_value = _make_get_value(values, quantity)
    batchid_expr = get_batch_expressions(model, v_syms_gb_ex, ipqv, quantity)
    batchids = batchid_expr.keys()
    vals = list(map(get_value, batchids))
    exprs = casadi.vcat(batchid_expr.values())
    return batchids, vals, exprs if exprs.size1() else _SX_0r1c

def get_node_expressions(index_of_node, Vnode_ri):
    """Returns expression of absolute voltages for addressed nodes.

    Parameters
    ----------
    index_of_node: array_like, int
        node indices for subset slicing
    Vnode_ri: casadi.DM
        * Vnode_ri[index_of_node] - Vre
        * Vnode_ri[size/2 + index_of_node] - Vim

    Returns
    -------
    casadi.SX"""
    if len(index_of_node):
        Vnode_ri_ = casadi.hcat(
            casadi.vertsplit(Vnode_ri, Vnode_ri.size1()//2))[index_of_node,:]
        Vsqr = casadi.power(Vnode_ri_, 2)
        return (Vsqr[:, 0] + Vsqr[:, 1]).sqrt()
    return _DM_0r1c

def get_diff_expressions(model, expressions, ipqv, quantities):
    """Creates a vector (casadi.SX, shape n,1) expressing the difference
    between measured and calculated values for absolute current, active power
    or reactive power and voltage. The expressions are based on the batch
    definitions or referenced node. Intended use is building the objective.

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
        [:,4] Pip, active power interpolated
        [:,5] Qip, reactive power interpolated
        [:,6] Vabs_sqr, square of voltage magnitude
        [:,7] interpolate?
    quantities: str
        string of characters 'I'|'P'|'Q'|'V'
        addresses current magnitude, active power, reactive power or magnitude
        of voltage, case insensitive, other characters are ignored

    Returns
    -------
    tuple
        * quantities, numpy.array<str>
        * id_of_batch, numpy.array<str>
        * value, casadi.DM, vector (shape n,1)
        * expression, casadi.SX, vector (shape n,1)"""
    _quantities = []
    _ids = []
    _vals = []
    _exprs = casadi.SX(0, 1)
    for quantity in quantities.upper():
        if quantity in 'IPQ':
            ids, vals, exprs = get_batch_flow_expressions(
                model, expressions, ipqv, quantity)
            _quantities.extend([quantity]*len(ids))
            _ids.extend(ids)
            _vals.extend(vals)
            _exprs = casadi.vertcat(_exprs, exprs)
        if quantity=='V':
            vvals = value_of_voltages(model.vvalues)
            count_of_values = len(vvals)
            _quantities.extend([quantity]*count_of_values)
            _ids.extend(vvals.id_of_node)
            _vals.extend(vvals.V)
            _exprs = casadi.vertcat(
                _exprs,
                expressions['Vnode_syms'][vvals.index,2].sqrt())
    return np.array(_quantities), np.array(_ids), casadi.DM(_vals), _exprs

def get_batch_constraints(values_of_constraints, expressions_of_batches):
    """Creates expressions for constraints for keeping batch values
    constant.
    
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

def get_optimize(model, expressions, positions=None):
    """Preapares the optimization function.

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
    params_ = casadi.vertcat(
        vstack(expressions['Vslack_syms']),
        expressions['position_syms'])
    positions_ = model.branchtaps.position if positions is None else positions
    values_of_parameters_ = casadi.vertcat(
        np.real(Vslacks_neg), np.imag(Vslacks_neg), positions_)
    count_of_vri = 2 * model.shape_of_Y[0]
    Vmin = [-np.inf] * count_of_vri
    Vmax = [np.inf] * count_of_vri
    def optimize(
        Vnode_ri_ini, scaling_data, Inode_inj, diff_data,
        constraints=_SX_0r1c):
        """Solves an optimization task.

        Parameters
        ----------
        Vnode_ri_ini: casadi.DM (shape 2n,1)
            float, initial node voltages with separated real and imaginary
            parts, first real then imaginary
        scaling_data: Scalingdata
            * .kvars, casadi.SX, column vector, symbols for variables
              of scaling factors
            * .kconsts, casadi.SX, column vector, symbols for constants
              of scaling factors
            * .values_of_vars, casadi.DM, column vector, initial values
              for kvars
            * .values_of_consts, casadi.DM, column vector, values for consts
        Inode_inj: casadi.SX (shape n,2)
            expressions for injected node current
        diff_data: tuple
            data for objective function
            * symbols for quantities, list of values 'P'|'Q'|'I'|'V'
            * ids of batches, list of str
            * values, list of floats
            * casadi.SX, expressions of differences (shape n,1)
        constraints: casadi.SX
            expressions for additional constraints, values to be kept zero
            (default constraints are Inode==0)

        Returns
        -------
        bool
            success?
        casadi.DM
            result vector of optimization"""
        syms = casadi.vertcat(Vnode_ri_syms, scaling_data.kvars)
        objective = casadi.sumsqr(diff_data[2] - diff_data[3])
        params = casadi.vertcat(params_, scaling_data.kconsts)
        values_of_parameters = casadi.vertcat(
            values_of_parameters_, scaling_data.values_of_consts)
        constraints_ = casadi.vertcat(
            expressions['Y_by_V'] + vstack(Inode_inj), constraints)
        nlp = {'x': syms, 'f': objective, 'g': constraints_, 'p': params}
        solver = casadi.nlpsol('solver', 'ipopt', nlp)
        # initial values of desision variables
        ini = casadi.vertcat(Vnode_ri_ini, scaling_data.values_of_vars)
        # limits of desision variables
        lbx = casadi.vertcat(Vmin, scaling_data.kvar_min)
        ubx = casadi.vertcat(Vmax, scaling_data.kvar_max)
        # calculate
        r = solver(
            x0=ini, p=values_of_parameters, lbg=0, ubg=0, lbx=lbx, ubx=ubx)
        # r = solver(
        #     x0=ini, p=values_of_parameters, lbg=0, ubg=0)
        return solver.stats()['success'], r['x']
    return optimize

#
# organize data
#

def vstack(m, column_count=0):
    """Helper, stacks columns of matrix m vertically which creates
    a vector.

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

def get_calculate_from_result(model, expressions, scaling_data, x):
    """Creates a function which calculates the values of casadi.SX expressions
    using the result of the nlp-solver.

    Parameters
    ----------
    model: egrid.model.Model
        model of electric network for calculation
    expressions: dict
        * 'Vnode_syms', casadi.SX, vectors, symbols of node voltages
        * 'position_syms', casadi.SX, vector, symbols of tap positions
        * 'Vslack_syms', symbols of slack voltages
    scaling_data: Scalingdata
        calculation step specific symbols
    x: casadi.SX
        result vector calculated by nlp-solver

    Returns
    -------
    function
        (casadi.SX) -> (casadi.DM)
        (expressions) -> (values)"""
    Vnode_ri_syms = vstack(expressions['Vnode_syms'], 2)
    count_of_v_ri = Vnode_ri_syms.size1()
    voltages_ri = x[:count_of_v_ri].toarray()
    x_scaling = x[count_of_v_ri:]
    Vslacks_neg = -model.slacks.V
    return make_calculate(
        (scaling_data.kvars,
         scaling_data.kconsts,
         Vnode_ri_syms,
         vstack(expressions['Vslack_syms'], 2),
         expressions['position_syms']),
        (x_scaling,
         scaling_data.values_of_consts,
         voltages_ri,
         casadi.vertcat(np.real(Vslacks_neg), np.imag(Vslacks_neg)),
         model.branchtaps.position))

#
# convenience functions for easier handling of estimation
#

def get_step_data(
    model, expressions, step=0, k_prev=_DM_0r1c, 
    quantities_of_objective='', quantities_of_constraints='', 
    values_of_constraints=None, positions=None):
    """Prepares data for call of function estimate.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    expressions: dict
        * 'Vnode_syms', casadi.SX, expressions of node Voltages
        * 'Vslack_syms', casadi.SX, symbols of slack voltages,
           Vslack_syms[:,0] real part, Vslack_syms[:,1] imaginary part
        * 'position_syms', casadi.SX, symbols of tap positions
        * 'gb_mn_tot', conductance g / susceptance b per branch terminal
            * gb_mn_tot[:,0] g_mn, mutual conductance
            * gb_mn_tot[:,1] b_mn, mutual susceptance
            * gb_mn_tot[:,2] g_tot, self conductance + mutual conductance
            * gb_mn_tot[:,3] b_tot, self susceptance + mutual susceptance
        * 'Y_by_V', casadi.SX, expression for Y @ V
        * 'get_scaling_and_injection_data', function
          (int, casadi.DM) -> (tuple - Scalingdata, casadi.SX)
          which is a function
          (index_of_step, scaling_factors_of_previous_step)
            -> (tuple - Scalingdata, injection_data)
        * 'inj_to_node', casadi.SX, matrix, maps from
          injections to power flow calculation nodes
    step: int
        index of estimation step
    k_prev: casadi.DM
        optional
        result output of factors (only) calculated in previous step, if any
    quantities_of_objective: str
        optional
        string of characters 'I'|'P'|'Q'|'V' or empty string ''
        addresses differences of calculated and given values to be minimized,
        the characters are symbols for given current magnitude, active power,
        reactive power or magnitude of voltage,
        case insensitive, other characters are ignored
    quantities_of_constraints: str
        optional
        string of characters 'I'|'P'|'Q'|'V' or empty string ''
        addresses quantities of batches (measured values or setpoints) 
        to be kept constant, the values are obtained from a previous
        calculation/initialization step, the characters are symbols for 
        given current magnitude, active power, reactive power or 
        magnitude of voltage, case insensitive, other characters are ignored,
        values must be given with argument 'values_of_constraints',
        conditions must be satisfied
    values_of_constraints: tuple
        * [0] numpy.array<str>, quantities, 
        * [1] numpy.array<str>, id_of_batch
        * [2] numpy.array<float>, vector (shape n,1), value
        values for constraints (refer to argument 'quantities_of_constraints')
    positions: array_like
        optional
        int, positions of taps

    Returns
    -------
    model: egrid.model.Model
        data of electric grid
    expressions: dict
        * 'Vnode_syms', casadi.SX, expressions of node Voltages
        * 'Vslack_syms', casadi.SX, symbols of slack voltages,
           Vslack_syms[:,0] real part, Vslack_syms[:,1] imaginary part
        * 'position_syms', casadi.SX, symbols of tap positions
        * 'gb_mn_tot', conductance g / susceptance b per branch terminal
            * gb_mn_tot[:,0] g_mn, mutual conductance
            * gb_mn_tot[:,1] b_mn, mutual susceptance
            * gb_mn_tot[:,2] g_tot, self conductance + mutual conductance
            * gb_mn_tot[:,3] b_tot, self susceptance + mutual susceptance
        * 'Y_by_V', casadi.SX, expression for Y @ V
        * 'get_scaling_and_injection_data', function
          (int, casadi.DM) -> (tuple - Scalingdata, casadi.SX)
          which is a function
          (index_of_step, scaling_factors_of_previous_step)
            -> (tuple - Scalingdata, injection_data)
        * 'inj_to_node', casadi.SX, matrix, maps from
          injections to power flow calculation nodes
    scaling_data: Scalingdata
        * .kvars, casadi.SX, column vector, symbols for variables
          of scaling factors
        * .kconsts, casadi.SX, column vector, symbols for constants
          of scaling factors
        * .values_of_vars, casadi.DM, column vector, initial values
          for kvars
        * .values_of_consts, casadi.DM, column vector, values for consts
    diff_data: tuple
        data for objective function to be minimized
        table, four column vectors
        * quantities, list of string
        * id_of_batch, list of string
        * value, casadi.DM, vector (shape n,1)
        * expression, casadi.SX, vector (shape n,1)
    batch_constraints: casadi.SX (shape m,1)
        constraints for keeping quantities calculated in previous step constant
    Vnode_ri_ini: array_like (shape 2n,1)
        initial node voltages separated real and imaginary values
    Inode_inj: casadi.SX (shape n,2)
        * Inode_inj[:,0] - Ire, real part of current injected into node
        * Inode_inj[:,1] - Iim, imaginary part of current injected into node
    positions: array_like
        int, positions of taps, can be None"""
    scaling_data, Iinj_data = (
        expressions['get_scaling_and_injection_data'](step, k_prev))
    Inode_inj = expressions['inj_to_node'] @ Iinj_data[:,:2]
    diff_data = get_diff_expressions(
        model, expressions, Iinj_data, quantities_of_objective)
    expr_of_batches = get_diff_expressions(
        model, expressions, Iinj_data, quantities_of_constraints)
    batch_constraints = (
        _SX_0r1c
        if values_of_constraints is None or len(expr_of_batches[0])==0 else
        get_batch_constraints(values_of_constraints, expr_of_batches))
    return (
        model, expressions, scaling_data, diff_data, batch_constraints, 
        Inode_inj, positions)

def estimate(
    model, expressions, scaling_data, diff_data, batch_constraints, Inode_inj,
    positions, Vnode_ri_ini=None):
    """Optimizes.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    expressions: dict
        estimation data
        * 'Vnode_syms', casadi.SX (shape n,2)
        * 'Vslack_syms', casadi.SX (shape n,2)
        * 'position_syms', casadi.SX (shape n,1)
    scaling_data: Scalingdata
        * .kvars, casadi.SX, column vector, symbols for variables
          of scaling factors
        * .kconsts, casadi.SX, column vector, symbols for constants
          of scaling factors
        * .values_of_vars, casadi.DM, column vector, initial values
          for kvars
        * .values_of_consts, casadi.DM, column vector, values for consts
    diff_data: tuple
        data for objective function
        table, four column vectors
        * quantities, list of string
        * id_of_batch, list of string
        * value, casadi.DM, vector (shape n,1)
        * expression, casadi.SX, vector (shape n,1)
    Inode_inj: casadi.SX (shape n,2)
        * Inode_inj[:,0] - Ire, real part of current injected into node
        * Inode_inj[:,1] - Iim, imaginary part of current injected into node
    positions: array_like
        optional
        int, positions of taps
    Vnode_ri_ini: array_like (shape 2n,1)
        initial node voltages separated real and imaginary values

    Returns
    -------
    succ : bool
        success?
    voltages_cx : numpy.array, complex (shape n,1)
        calculated complex node voltages
    pq_factors : numpy.array, float (shape m,2)
        scaling factors for injections"""
    if Vnode_ri_ini is None:
        # power flow calculation for initial voltages
        succ, Vnode_ri_ini = calculate_power_flow2(
            model, expressions, scaling_data, Inode_inj, positions)
        assert succ, 'calculation of power flow is failed'
    optimize = get_optimize(model, expressions, positions)
    succ, x = optimize(
        Vnode_ri_ini, scaling_data, Inode_inj, diff_data, batch_constraints)
    # result processing
    count_of_Vvalues = 2 * model.shape_of_Y[0]
    return (
        (succ, *casadi.vertsplit(x, count_of_Vvalues))
        if count_of_Vvalues < x.size1() else
        (succ, x, _DM_0r1c))

def get_Vcx_kpq(scaling_data, x_V, x_scaling):
    """Helper. Processes results of estimation.

    Parameters
    ----------
    scaling_data: Scalingdata
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
    x_V : casadi.DM
        real part of node voltages, imaginary part of node voltages
    x_scaling : casasdi.DM
        scaling factors

    Returns
    -------
    V : numpy.array (shape n,1), complex
        node voltages
    k : numpy.array (shape m,2), float
        scaling factors per injection"""
    k, _ = get_k(scaling_data, x_scaling)
    V = ri_to_complex(x_V)
    return V, k
