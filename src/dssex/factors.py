# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 12:17:49 2023

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
Created on Fri Dec 16 00:14:07 2022

@author: pyprg

The function 'get_factors' returns data on factors to be applied to nominal
active and reactive power of injections. Factors fall in one of the two
categories 'var' or 'const'. Factors of category 'var' are decision variables.
Factors of category 'const' are parameters. Factors are specific for
each step. A factor is initialized by a value of a factor from previous step
or - if none exists - by a factor specific 'value' (which is an attribute of
the input data).
"""
import casadi
import pandas as pd
import numpy as np
from functools import partial
from collections import namedtuple
from itertools import chain
from egrid.builder import DEFAULT_FACTOR_ID, deff, Factor
# empty vector of values
_DM_0r1c = casadi.DM(0,1)
# empty vector of expressions
_SX_0r1c = casadi.SX(0,1)

def _create_symbols_with_ids(ids):
    """Creates a column vector of casadi symbols with given identifiers.

    Parameters
    ----------
    ids: iterable of str

    Returns
    -------
    casadi.SX"""
    return casadi.vertcat(*(casadi.SX.sym(id_) for id_ in ids))

def _get_step_factor_to_injection_part(
        injectionids, assoc_frame, step_factors, indices_of_steps):
    """Arranges ids for all calculation steps and injections.

    Parameters
    ----------
    injectionids: pandas.Series
        str, IDs of all injections
    assoc_frame: (str (step), str (injid), 'p'|'q' (part))
        * .id, str, ID of factor
    step_factors: pandas.DataFrame

    indices_of_steps: array_like
        int, indices of optimization steps

    Returns
    -------
    pandas.Dataframe (int (step), str (id of injection))
        * .injid, str
        * .part, 'p'|'q'"""
    # all injections, create step, id, (pq) for all injections
    index_all = pd.MultiIndex.from_product(
        [indices_of_steps, injectionids, ('p', 'q')],
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
    # '-1' means copy initial data from column 'value' as there is no valid
    #   reference to a var/const of previous step
    ini.fillna(-1, inplace=True)
    return ini.astype(dtype='Int64')

def _get_default_factors(indices_of_steps):
    """Generates one default scaling factor for each step. The factor is
    of type 'const' has value 1.0, minimum and maximum are 1.0 too.

    Parameters
    ----------
    indices_of_steps: array_like
        int, indices of steps which to create a default factor for

    Parameters
    ----------
    pandas.DataFrame (index: ['step', 'id'])
        value in column 'id' of index is the string
        of egrid.builder.DEFAULT_FACTOR_ID,
        columns according to fields of egrid.builder.Loadfactor"""
    return (
        pd.DataFrame(
            deff(id_=DEFAULT_FACTOR_ID, type_='const', value=1.0,
                 min_=1.0, max_=1.0, step=indices_of_steps),
            columns=Factor._fields)
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

def _add_step_index(df, step_indices):
    """Copies data of generic step (index -1) for each step index
    in step_indices. Combines the data with df.

    Parameters
    ----------
    df: pandas.DataFrame
        df has a MultiIndex
    step_indices: interable
        int

    Returns
    -------
    pandas.DataFrame (extended copy of df)"""
    try:
        df_ = df.loc[-1]
    except KeyError:
        return df
    def _new_index(index_of_step):
        cp = df_.copy()
        # adds a column 'step', adds value 'index_of_step' in each row of
        #   column 'step', then adds the colum 'step' to the index
        #   (column 'step' is added to the old index in the left most position)
        return cp.assign(step=index_of_step).set_index(['step', cp.index])
    return df.combine_first(
        pd.concat([_new_index(idx) for idx in step_indices]))

def get_factor_data(
        injectionids, given_factors, assoc_frame, index_of_step):
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
    index_of_step: int
        index of optimization steps, first step has index 0

    Returns
    -------
    tuple
        * pandas.DataFrame, all scaling factors
        * pandas.DataFrame, injections with scaling factors"""
    # index for requested step and the step before requested step, data
    #   of step before are needed for initialization
    indices_of_steps = (
        [index_of_step - 1, index_of_step]
        if 0 < index_of_step else
        [0])
    # given factors are either specific for a step or defined for each step
    #   which is indicated by a '-1' step-index
    #   generate factors from those with -1 step setting
    given_factors_ = _add_step_index(given_factors.copy(), indices_of_steps)
    assoc_frame_ = _add_step_index(assoc_frame.copy(), indices_of_steps)
    # step, id_of_factor => id_of_injection, 'id_p'|'id_q'
    step_factor_injection_part = _get_step_factor_to_injection_part(
        injectionids, assoc_frame_, given_factors_, indices_of_steps)
    # remove factors not needed, add default (nan) factors if necessary
    required_factors_index = step_factor_injection_part.index.unique()
    required_factors = given_factors_.reindex(required_factors_index)
    if required_factors.isna().any(axis=None):
        # ensure existence of default factors when needed
        default_factor_steps = (
            # type is just an arbitrary column
            required_factors[required_factors.type.isna()]
            .reset_index()
            .step)
        default_factors = _get_default_factors(default_factor_steps)
        # replace nan with values (for required default factors)
        factors = required_factors.combine_first(default_factors)
    else:
        factors = required_factors
    index_of_symbol = _factor_index_per_step(factors)
    factors['index_of_symbol'] = index_of_symbol
    # add data for initialization
    factors['index_of_source'] = _get_factor_ini_values(
        factors, index_of_symbol)
    factors.sort_values(by=['step', 'type', 'id'], inplace=True)
    if not step_factor_injection_part.empty:
        injection_factors = (
            step_factor_injection_part
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
    """Returns values for symbols. When a symbol is a variable the value
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
        assert len(value_of_previous_step), 'missing value_of_previous_step'
        values[calc.index_of_symbol] = (
            value_of_previous_step[calc.index_of_source.astype(int)])
    return values

def _select_rows(vecs, row_index):
    """Creates column vectors from vecs by extracting elements by their
    (row-) indices.

    Parameters
    ----------
    vecs: iterable
        casadi.SX or casadi.DM, column vector
    row_index: array_like
        int

    Returns
    -------
    iterator
        * casadi.SX / casadi.DM"""
    return (v[row_index, 0] for v in vecs)

Factordata = namedtuple(
    'Factordata',
    'kpq kvars values_of_vars kvar_min kvar_max kconsts values_of_consts '
    'var_const_to_factor var_const_to_kp var_const_to_kq')
Factordata.__doc__="""
Symbols of variables and constants for factors.

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

_empty_factor_data = Factordata(
    kpq=casadi.horzcat(_SX_0r1c, _SX_0r1c),
    kvars=_SX_0r1c,
    values_of_vars=_DM_0r1c,
    kvar_min=_DM_0r1c,
    kvar_max=_DM_0r1c,
    kconsts=_SX_0r1c,
    values_of_consts=_DM_0r1c,
    var_const_to_factor=(),
    var_const_to_kp=(),
    var_const_to_kq=())

def _make_DM_vector(array_like):
    """Creates a casadi.DM vector from array_like.

    Parameters
    ----------
    array_like: array_like

    Returns
    -------
    casadi.DM"""
    return casadi.DM(array_like) if len(array_like) else _DM_0r1c

def _get_injections_factors(injection_factor_step_groups, step):
    """Returns the group with index step. Returns an empty pandas.DataFrame
    if the group with index step does not exist.

    Parameters
    ----------
    injection_factor_step_groups: pandas.groupby

    step: int
        index of step 0, 1, ...

    Returns
    -------
    pandas.DataFrame"""
    try:
        return (
            injection_factor_step_groups
            .get_group(step)
            .sort_values(by='index_of_injection'))
    except KeyError:
        return pd.DataFrame(
            [],
            columns=[
                'step', 'injid', 'id_p', 'id_q',
                'kp', 'kq', 'index_of_injection'])

def _get_factor_data_for_step(
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
    Factordata
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
    try:
        factors = factor_step_groups.get_group(step)
    except KeyError:
        return _empty_factor_data
    # a symbol for each factor
    symbols = _create_symbols_with_ids(factors.id)
    injections_factors = _get_injections_factors(
        injection_factor_step_groups, step)
    values = _get_values_of_symbols(factors, k_prev)
    select_symbols_values = partial(_select_rows, [symbols, values])
    factors_var = factors[factors.type=='var']
    symbols_of_vars, values_of_vars = select_symbols_values(
        factors_var.index_of_symbol)
    factors_consts = factors[factors.type=='const']
    symbols_of_consts, values_of_consts = select_symbols_values(
        factors_consts.index_of_symbol)
    # the optimization result is provided as a concatenation of
    #   values for decision variables and parameters, here we prepare indices
    #   for mapping to kp/kq (which are ordered according to injections)
    var_const_idxs = np.concatenate(
        [factors_var.index_of_symbol.array,
         factors_consts.index_of_symbol.array])
    var_const_to_factor = np.zeros_like(var_const_idxs)
    var_const_to_factor[var_const_idxs] = factors.index_of_symbol
    return Factordata(
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

def make_get_factor_data(model):
    """Creates a function for creating Factordata specific for a calculation
    step.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid

    Returns
    -------
    function
        (int, casadi.DM) -> (Factordata)
        (index_of_calculation_step, result_of_previous_step) -> (Factordata)
        Factordata:
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
    def _get_factor_data(step=0, k_prev=_DM_0r1c):
        factors, injection_factors = get_factor_data(
                model.injections.id,
                model.factors,
                model.injection_factor_associations,
                step)
        return _get_factor_data_for_step(
            _groupby_step(factors),
            _groupby_step(injection_factors),
            step,
            k_prev)
    return _get_factor_data

def get_values_of_factors(factor_data, x_factors):
    """Function for extracting factors for injections from the result provided
    by the solver.
    Enhances scaling factors calculated by optimization with constant
    scaling factors and reorders the factors according to order of injections.
    Returns kp and kq for each injection.
    The function creates a vector of values for scaling factors which are
    decision variables and those which are constants. This vector is ordered
    for use as initial scaling factor values in next estimation step.

    Parameters
    ----------
    factor_data: Factordata
        * .values_of_consts,
            array_like, float, column vector, values for consts
        * .var_const_to_factor,
            array_like int, index_of_factor=>index_of_var_const
            converts var_const to factor (var_const[var_const_to_factor])
        * .var_const_to_kp
            array_like int, converts var_const to kp, one active power
            scaling factor for each injection (var_const[var_const_to_kp])
        * .var_const_to_kq
            array_like int, converts var_const to kq, one reactive power
            scaling factor for each injection (var_const[var_const_to_kq])
    x_factors: array_like
        float, result of optimization (subset)

    Result
    ------
    tuple
        * numpy.array (n,2), kp, kq for each injection
        * numpy.array (m,1) kvar/const"""
    # concat values of decision variables and parameters
    k_var_const = np.vstack([x_factors, factor_data.values_of_consts])
    kp = k_var_const[factor_data.var_const_to_kp]
    kq = k_var_const[factor_data.var_const_to_kq]
    return np.hstack([kp, kq]), k_var_const[factor_data.var_const_to_factor]

