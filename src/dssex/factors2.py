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
from itertools import chain, repeat
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
    return (
        _SX_0r1c 
        if ids.empty else 
        casadi.vertcat(*(casadi.SX.sym(id_) for id_ in ids)))

def _get_step_injection_part_to_factor(
        injectionids, assoc_frame, indices_of_steps):
    """Arranges ids for all calculation steps and injections.

    Parameters
    ----------
    injectionids: pandas.Series
        str, IDs of all injections
    assoc_frame: (str (step), str (injid), 'p'|'q' (part))
        * .id, str, ID of factor
    indices_of_steps: array_like
        int, indices of optimization steps

    Returns
    -------
    pandas.Dataframe (int (step), str (id of injection), 'p'|'q' (part))
        * .id, str, identifier of factor"""
    # all injections, create step, id, (pq) for all injections
    index_all = pd.MultiIndex.from_product(
        [indices_of_steps, injectionids, ('p', 'q')],
        names=('step', 'injid', 'part'))
    # step injid part => id
    return (
        # do not accept duplicated links
        assoc_frame[~assoc_frame.index.duplicated()]
        .reindex(index_all, fill_value=DEFAULT_FACTOR_ID))

def _get_factor_ini_values(factors):
    """Returns indices for initial values of variables/parameters.

    Parameters
    ----------
    factors: pandas.DataFrame

    symbols: pandas.Series
        int, indices of symbols

    Returns
    -------
    pandas.Series
        int"""
    unique_factors = factors.index
    prev_index = pd.MultiIndex.from_arrays(
        [unique_factors.get_level_values(0) - 1, factors.id_of_source.array])
    ini = factors.index_of_symbol.reindex(prev_index)
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

def _factor_index_per_step(factors, start):
    """Creates an index (0...n) for each step.

    Parameters
    ----------
    factors: pandas.DataFrame (step, id)->...

    start: iterable
        int, first index for each step, if start is None the first index is 0

    Returns
    -------
    pandas.Series"""
    offsets = repeat(0) if start is None else start
    return pd.Series(
        chain.from_iterable(
            (range(off, off+len(factors.loc[step])))
            for off, step in zip(offsets, factors.index.levels[0])),
        index=factors.index,
        name='index_of_symbol',
        dtype=np.int64)

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
    'kpq ftaps index_of_term '
    'vars values_of_vars var_min var_max is_discrete '
    'consts values_of_consts '
    'var_const_to_factor var_const_to_kp var_const_to_kq var_const_to_ftaps')
Factordata.__doc__="""
Symbols of variables and constants for factors.

Parameters
----------
kpq: casadi.SX
    two column vectors, symbols for scaling factors of active and
    reactive power per injection
ftaps: casadi.SX
    vector, symbols for taps factors of terminals for terminals
    addressed by index_of_term
index_of_term: numpy.array
    int, index of terminal which ftaps is assigned to
vars: casadi.SX
    column vector, symbols for variables of scaling factors
values_of_vars: casadi.DM
    column vector, initial values for vars
var_min: casadi.DM
    lower limits of vars
var_max: casadi.DM
    upper limits of vars
is_discrete: numpy.array
    bool, flag for variable
consts: casadi.SX
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
    for each injection (var_const[var_const_to_kq])
var_const_to_ftaps: array_like
    int, converts var_const to ftaps, factor assigned to
    (selected) terminals (var_const[var_const_to_ftaps])"""

_empty_factor_data = Factordata(
    kpq=casadi.horzcat(_SX_0r1c, _SX_0r1c),
    ftaps=_SX_0r1c,
    index_of_term=(),
    vars=_SX_0r1c,
    values_of_vars=_DM_0r1c,
    var_min=_DM_0r1c,
    var_max=_DM_0r1c,
    is_discrete=(),
    consts=_SX_0r1c,
    values_of_consts=_DM_0r1c,
    var_const_to_factor=(),
    var_const_to_kp=(),
    var_const_to_kq=(),
    var_const_to_ftaps=())

def _make_DM_vector(array_like):
    """Creates a casadi.DM vector from array_like.

    Parameters
    ----------
    array_like: array_like

    Returns
    -------
    casadi.DM"""
    return casadi.DM(array_like) if len(array_like) else _DM_0r1c

def _empty_like(df, droplevel=-1):
    """Creates an empty pandas.DataFrame from a template DataFrame.

    Parameters
    ----------
    df: pandas.DataFrame

    droplevel: int
        function drops a level from index if value is greater -1

    Returns
    -------
    pandas.DataFrame"""
    idx_ = df.index.droplevel(droplevel) if -1 < droplevel else df.index
    idx = idx_.delete(range(len(idx_)))
    return pd.DataFrame([], columns=df.columns, index=idx).astype(df.dtypes)

def _loc(df, key):
    try:
        return df.loc[key]
    except KeyError:
        return _empty_like(df, 0)

Stepgroups = namedtuple(
    'Stepgroups', 'groups template')
Stepgroups.__doc__ = """pandas.Groupby object and a(n empty) DataFrame
template.

Parameters
----------
groups: pandas.Groupby
    a pandas.DataFrame grouped by columns 'step'

template: pandas.DataFrame
"""

def _create_stepgroups(df):
    """Groups df by column 'step'.

    Parameters
    ----------
    df: pandas.DataFrame
        * .step

    Returns
    -------
    Stepgroups"""
    df_ = df.reset_index()
    return Stepgroups(df_.groupby('step'), _empty_like(df_))

def _selectgroup(step, stepgroups):
    """Selects a group with index step from stepgroups. Returns a copy of
    stepgroups.template if no group with index exists.

    Parameters
    ----------
    step: int
        index of optimization step
    stepgroups: Stepgroups

    Returns
    -------
    pandas.DataFrame"""
    try:
        return stepgroups.groups.get_group(step)
    except KeyError:
        return stepgroups.template.copy()

def _selectgroups(steps, stepgroups):
    """Selects and concatenates groups from stepgroups if present.
    Returns an empty pandas.DataFrame otherwise.

    Parameters
    ----------
    steps: interable
        int, index of optimization step, first step is 0, generic data
        have step -1
    stepgroups: Stepgroups

    Returns
    -------
    pandas.DataFrame"""
    return pd.concat([_selectgroup(step, stepgroups) for step in steps])

Factordefs = namedtuple(
    'Factordefs',
    'gen_factor_data gen_factor_symbols gen_injfactor gen_termfactor '
    'factorgroups injfactorgroups')
Factordefs.__doc__ ="""
Data of generic factors (step == -1),
symbols for generic factors, references to injections and terminals for
generic factors.

Data grouped by step for factors, references to injections
and references to terminals.

Parameters
----------
* .gen_factor_data, pandas.DataFrame
* .gen_factor_symbols, casadi.SX, shape(n,1)
* .gen_injfactor, pandas.DataFrame
* .gen_termfactor, pandas.DataFrame
* .factorgroups, pandas.DataFrame
* .injfactorgroups, pandas.DataFrame
"""

def _get_factordefs(model):
    """Create casadi.SX symbols for generic factors. Filters valid data.
    Generic factors are present in each step, they have step index -1.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid

    Returns
    -------
    Factordefs
        * .gen_factor_data, pandas.DataFrame
        * .gen_factor_symbols, casadi.SX, shape(n,1)
        * .gen_injfactor, pandas.DataFrame
        * .gen_termfactor, pandas.DataFrame
        * .factorgroups, pandas.DataFrame
        * .injfactorgroups, pandas.DataFrame"""
    factorgroups = _create_stepgroups(model.factors)
    injfactorgroups = _create_stepgroups(model.injection_factor_associations)
    termfactorgroups = _create_stepgroups(model.terminal_factor_associations)
    # factors with attribute step == -1
    gen_factors = _selectgroup(-1, factorgroups)
    # terminal-factor association with step attribute == -1
    gen_termassoc = _selectgroup(-1, termfactorgroups)
    valid_termassoc = gen_termassoc.id.isin(gen_factors.id)
    termassoc_ = gen_termassoc[valid_termassoc]
    # injection-factor association with step attribute == -1
    gen_injassoc = _selectgroup(-1, injfactorgroups)
    valid_incassoc = gen_injassoc.id.isin(gen_factors.id)
    injassoc = gen_injassoc[valid_incassoc]
    # only referenced factors are necessary
    valid_factorids = np.sort(pd.concat([termassoc_.id, injassoc.id]).unique())
    factors = (
        gen_factors.set_index('id').reindex(valid_factorids).reset_index())
    factors['index_of_symbol'] = range(len(factors))
    symbols = _create_symbols_with_ids(factors.id)
    # add index of symbol to termassoc, 
    #   terminal factors are NEVER step-specific
    termassoc = (
        pd.merge(
            left=termassoc_, 
            right=factors[['id','index_of_symbol']], 
            left_on='id', 
            right_on='id'))
    return Factordefs(
        factors.set_index('id'),
        symbols,
        injassoc.set_index(['injid', 'part']),
        termassoc.set_index(['branchid', 'nodeid']),
        factorgroups, 
        injfactorgroups)

def _add_step_index(df, step_indices):
    """Copies data of df for each step index in step_indices.
    Concatenates the data.

    Parameters
    ----------
    df: pandas.DataFrame
        df has a MultiIndex
    step_indices: interable
        int

    Returns
    -------
    pandas.DataFrame (extended copy of df)"""
    if df.empty:
        return df.copy().assign(step=0).set_index(['step', df.index])
    def _new_index(index_of_step):
        cp = df.copy()
        # adds a column 'step', adds value 'index_of_step' in each row of
        #   column 'step', then adds the colum 'step' to the index
        #   (column 'step' is added to the old index in the left most position)
        return cp.assign(step=index_of_step).set_index(['step', cp.index])
    return pd.concat([_new_index(idx) for idx in step_indices])

def _get_injection_factors(step_factor_injection_part, factors):
    """Creates crossreference from (step, injection) to IDs and
    indices of scaling factors for active and reactive power.

    Parameters
    ----------
    step_factor_injection_part: pandas.DataFrame
        DESCRIPTION.
    factors: pandas.DataFrame
        DESCRIPTION.

    Returns
    -------
    pandas.DataFrame
        DESCRIPTION."""
    if not step_factor_injection_part.empty:
        injection_factors = (
            step_factor_injection_part
            .join(factors.index_of_symbol)
            .reset_index()
            .set_index(['step', 'injid', 'part'])
            .unstack('part')
            .droplevel(0, axis=1))
        injection_factors.columns=['id_p', 'id_q', 'kp', 'kq']
        return injection_factors.astype({'kp':np.int64, 'kq':np.int64})
    return pd.DataFrame(
        [],
        columns=['id_p', 'id_q', 'kp', 'kq'],
        index=pd.MultiIndex.from_arrays([[],[]], names=['step', 'injid']))

def _add_default_factors(required_factors):
    if required_factors.isna().any(axis=None):
        # ensure existence of default factors when needed
        default_factor_steps = (
            # type is just an arbitrary column
            required_factors[required_factors.type.isna()]
            .reset_index()
            .step
            .unique())
        default_factors = _get_default_factors(default_factor_steps)
        # replace nan with values (for required default factors)
        return required_factors.combine_first(default_factors)
    return required_factors

def _get_scaling_factor_data(model, factordefs, steps, start):
    """Creates and arranges indices for scaling factors and values for
    initialization of scaling factors.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    factordefs: Generic_factors
        * .gen_factor_data, pandas.DataFrame
        * .gen_factor_symbols, casadi.SX, shape(n,1)
        * .gen_injfactor, pandas.DataFrame
        * .gen_termfactor, pandas.DataFrame
        * .factorgroups, pandas.DataFrame
        * .injfactorgroups, pandas.DataFrame
    steps: iterable
        int, indices of optimization steps, first step has index 0
    start: iterable | None
        int, first index for each step, None: 0 for all steps

    Returns
    -------
    tuple
        * pandas.DataFrame, all scaling factors
          (str (step), 'const'|'var', str (id of factor)) ->
          * .id_of_source, str, source of initial value,
            factor of previous step
          * .value, float, initial value if no valid source reference
          * .min, float, smallest possible value
          * .max, float, greatest possible value
          * .is_discrete, bool, no digits after decimal point if True
            (input for MINLP solver), False for load scaling factors, True
            for capacitor bank model
          * .m, float, not used for scaling factors
          * .n, float, not used for scaling factors
          * .index_of_symbol, int, index in 1d-vector of var/const
          * .index_of_source, int, index in 1d-vector of previous step
          * .devtype, 'injection'
        * pandas.DataFrame, injections with scaling factors
          (int (step), str (injid))
          * .id_p, str, ID for scaling factor of active power
          * .id_q, str, ID for scaling factor of reactive power
          * .kp, int, index of active power scaling factor in 1d-vector
          * .kq, int, index of reactive power scaling factor in 1d-vector
          * .index_of_injection, int, index of affected injection"""
    generic_injfactor_steps = _add_step_index(factordefs.gen_injfactor, steps)
    assoc_steps = (
        _selectgroups(steps, factordefs.injfactorgroups)
        .set_index(['step', 'injid', 'part']))
    # get generic_assocs which are not in assocs of step, this allows
    #   overriding the linkage of factors
    assoc_diff = generic_injfactor_steps.index.difference(assoc_steps.index)
    generic_assoc = generic_injfactor_steps.reindex(assoc_diff)
    inj_assoc = pd.concat([generic_assoc, assoc_steps])
    # given factors are either specific for a step or defined for each step
    #   which is indicated by a '-1' step-index
    #   generate factors from those with -1 step setting for given steps
    # step, inj, part => factor
    #   copy generic_factors and add step indices
    generic_factor_steps = _add_step_index(factordefs.gen_factor_data, steps)
    # retrieve step specific factors
    factors_steps = (
        _selectgroups(steps, factordefs.factorgroups)
        .set_index(['step', 'id']))
    # select generic factors only if not part of specific factors
    req_generic_factors_index = generic_factor_steps.index.difference(
        factors_steps.index)
    # union of generic and specific injection factors
    given_factors = pd.concat(
        [generic_factor_steps
             .drop(columns=['index_of_symbol'])
             .reindex(req_generic_factors_index),
         factors_steps])
    # generate MultiIndex for:
    #   - two factors
    #   - all requested steps
    #   - all injections
    step_injection_part__factor = _get_step_injection_part_to_factor(
        model.injections.id, inj_assoc, steps)
    # step-to-factor index
    step_factor = (
        pd.MultiIndex.from_arrays(
            [step_injection_part__factor.index.get_level_values('step'),
             step_injection_part__factor.id],
            names=['step', 'id'])
        .unique())
    # given_factors, expand union of generic and specific factors to
    #   all required factors
    # this method call destroys type info of column 'is_discrete', was bool
    required_factors = given_factors.reindex(step_factor)
    factors = (
        _add_default_factors(required_factors)
        .astype({'is_discrete':bool}, copy=False))
    factors.sort_index(inplace=True)
    # indices of symbols
    factors = factors.join(
        generic_factor_steps['index_of_symbol']
        .astype('Int64'), how='left')
    no_symbol = factors['index_of_symbol'].isna()
    # range of indices for new scaling factor indices
    factor_index_per_step = _factor_index_per_step(
        factors[no_symbol], start)
    factors.loc[no_symbol, 'index_of_symbol'] = factor_index_per_step
    # add data for initialization
    factors['index_of_source'] = _get_factor_ini_values(factors)
    injection_factors = _get_injection_factors(
        step_injection_part__factor.reset_index().set_index(['step', 'id']),
        factors)
    # indices of injections ordered according to injection_factors
    injids = injection_factors.index.get_level_values(1)
    index_of_injection = (
        pd.Series(model.injections.index, index=model.injections.id)
        .reindex(injids)
        .array)
    injection_factors['index_of_injection'] = index_of_injection
    factors.reset_index(inplace=True)
    factors.set_index(['step', 'type', 'id'], inplace=True)
    return factors.assign(devtype='injection'), injection_factors

def _get_taps_factor_data(model, factordefs, steps):
    """Arranges indices for taps factors and values for initialization of
    taps factors.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    factordefs: Generic_factors
        * .gen_factor_data, pandas.DataFrame
        * .gen_factor_symbols, casadi.SX, shape(n,1)
        * .gen_injfactor, pandas.DataFrame
        * .gen_termfactor, pandas.DataFrame
        * .factorgroups, pandas.DataFrame
        * .injfactorgroups, pandas.DataFrame
    steps: iterable
        int, indices of optimization steps, first step has index 0

    Returns
    -------
    tuple
        * pandas.DataFrame, all taps factors
          (str (step), 'const'|'var', str (id of factor)) ->
          * .id_of_source, str, source of initial value,
            factor of previous step
          * .value, float, initial value if no valid source reference
          * .min, float, smallest possible value
          * .max, float, greatest possible value
          * .is_discrete, bool, no digits after decimal point if True
            (input for MINLP solver), True for taps model
          * .m, float, -Vdiff per tap-step in case of taps
          * .n, float, n = 1 - (Vdiff * index_of_neutral_position) for taps,
            e.g. when index_of_neutral_position=0 --> n=0
          * .index_of_symbol, int, index in 1d-vector of var/const
          * .index_of_source, int, index in 1d-vector of previous step
          * .devtype, 'terminal'
        * pandas.DataFrame, terminals with taps factors
          (int (step), int (index_of_term))
          * .index_of_symbol, int"""
    generic_termfactor_steps = _add_step_index(
        factordefs.gen_termfactor, steps)
    # get generic_assocs which are not in assocs of step, this allows
    #   overriding the linkage of factors
    term_assoc = generic_termfactor_steps[
        ~generic_termfactor_steps.index.duplicated()]
    # add index of terminal
    term_assoc['index_of_terminal'] = (
        model.branchterminals[['id_of_branch', 'id_of_node']]
        .reset_index()
        .set_index(['id_of_branch', 'id_of_node'])
        .reindex(term_assoc.index.droplevel('step'))
        .to_numpy())
    # given factors are generic with step -1
    #   generate factors for given steps
    # step, branch, node => factor
    #   copy generic_factors and add step indices
    generic_factor_steps = _add_step_index(factordefs.gen_factor_data, steps)
    # filter for linked taps-factors step-to-factor index
    step_factor = (
        pd.MultiIndex.from_arrays(
            [term_assoc.index.get_level_values('step'),
             term_assoc.id],
            names=['step', 'id'])
        .unique())
    factors = generic_factor_steps.reindex(step_factor)
    factors.sort_index(inplace=True)
    # retrieve step specific factors
    # step specific factors cannot introduce new taps-factors just
    #   modify generic taps-factors
    factors_steps = (
        _selectgroups(steps, factordefs.factorgroups)
        .set_index(['step', 'id']))
    override_index = factors.index.intersection(factors_steps.index)
    cols = [
        'type', 'id_of_source', 'value', 'min', 'max', 'is_discrete', 'm', 'n']
    factors.loc[override_index, cols] = factors_steps.loc[override_index, cols]
    # add data for initialization
    factors['index_of_source'] = _get_factor_ini_values(factors)
    return (
        factors.assign(devtype='terminal')
            .reset_index()
            .set_index(['step', 'type', 'id']), 
        term_assoc)

def _get_factor_data_for_step(
        factordefs, factors, injection_factors, terminal_factor, 
        k_prev=[]):
    """Prepares data of scaling factors per step.

    Arguments for solver call:

    Vectors of variables ('var') and of paramters ('const'), vectors of minimum
    and of maximum variable values, vectors of initial values for variables
    and values for paramters.

    Data to reorder the solver result:

    Sequences of indices for converting the vector of var/const to order
    of factors (symbols), to the order of active power scaling factors,
    reactive power scaling factors and taps factors.


    Parameters
    ----------
    factors: pandas.DataFrame
        sorted by 'index_of_symbol'

    injection_factors: pandas.groupby

    terminal_factors: pandas.groupby

    step: int
        step number 0, 1, ...
    k_prev: array_like
        float, values of scaling factors from previous step, 
        variables and constants

    Returns
    -------
    Factordata
        kpq: casadi.SX
            two column vectors, symbols for scaling factors of active and
            reactive power per injection
        ftaps: casadi.SX
            vector, symbols for taps factors of terminals for terminals
            addressed by index_of_term
        index_of_term: numpy.array
            int, index of terminal which ftaps is assigned to
        vars: casadi.SX
            column vector, symbols for variables of scaling factors
        values_of_vars: casadi.DM
            column vector, initial values for vars
        var_min: casadi.DM
            lower limits of vars
        var_max: casadi.DM
            upper limits of vars
        is_discrete: numpy.array
            bool, flag for variable
        consts: casadi.SX
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
            for each injection (var_const[var_const_to_kq])
        var_const_to_ftaps: array_like
            int, converts var_const to ftaps, factor assigned to
            (selected) terminals (var_const[var_const_to_ftaps])"""
    # a symbol for each factor
    count_of_generic_factors = len(factordefs.gen_factor_data)
    symbols_step = _create_symbols_with_ids(
        factors[count_of_generic_factors <= factors.index_of_symbol].id)
    symbols = casadi.vertcat(factordefs.gen_factor_symbols, symbols_step)
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
    var_const_idxs = (
        np.concatenate(
            [factors_var.index_of_symbol.array,
             factors_consts.index_of_symbol.array])
        .astype(np.int64))
    var_const_to_factor = np.zeros_like(var_const_idxs)
    var_const_to_factor[var_const_idxs] = factors.index_of_symbol
    return Factordata(
        # kp/kq used in expression building
        #   (columns of injections_factors.kp/kq store an index)
        kpq=casadi.horzcat(
            symbols[injection_factors.kp].reshape((-1,1)),
            symbols[injection_factors.kq].reshape((-1,1))),
        ftaps=symbols[terminal_factor.index_of_symbol].reshape((-1,1)),
        index_of_term=terminal_factor.index_of_terminal.to_numpy(),
        # vars, variables for solver preparation
        vars=symbols_of_vars,
        # initial values, argument in solver call
        values_of_vars=values_of_vars,
        # lower bound of scaling factors, argument in solver call
        var_min=_make_DM_vector(factors_var['min']),
        # upper bound of scaling factors, argument in solver call
        var_max=_make_DM_vector(factors_var['max']),
        # flag for variable
        is_discrete=factors_var.is_discrete.to_numpy(),
        # consts, constants for solver preparation
        consts=symbols_of_consts,
        # values of constants, argument in solver call
        values_of_consts=values_of_consts,
        # reordering of result
        var_const_to_factor=var_const_to_factor,
        var_const_to_kp=var_const_to_factor[injection_factors.kp],
        var_const_to_kq=var_const_to_factor[injection_factors.kq],
        var_const_to_ftaps=var_const_to_factor[
            terminal_factor.index_of_symbol])

def get_factor_data(model, factordefs, step=0, k_prev=[]):
    """Returns data of decision variables and on parameters for a specific
    step.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    factordefs: Factordefs
        * .gen_factor_data, pandas.DataFrame
        * .gen_factor_symbols, casadi.SX, shape(n,1)
        * .gen_injfactor, pandas.DataFrame
        * .gen_termfactor, pandas.DataFrame
        * .factorgroups, pandas.DataFrame
        * .injfactorgroups, pandas.DataFrame
    step: int
        index of optimization step, first index is 0
    k_prev: array_like optional
        float, factors of previous optimization step

    Returns
    -------
    Factordata"""
    # index for requested step and the step before requested step, 
    #   data of step before are needed for initialization
    steps = [step - 1, step] if 0 < step else [0]
    # factors assigned to terminals
    taps_factors, terminal_factor = _get_taps_factor_data(
        model, factordefs, steps)
    # scaling factors for injections
    start = repeat(len(factordefs.gen_factor_data))
    scaling_factors, injection_factors = _get_scaling_factor_data(
        model, factordefs, steps, start)
    factors = pd.concat([scaling_factors, taps_factors])
    factors.sort_values('index_of_symbol', inplace=True)
    return _get_factor_data_for_step(
        factordefs,
        _loc(factors, step).reset_index(),
        _loc(injection_factors, step).reset_index(),
        _loc(terminal_factor, step).reset_index(),
        k_prev)

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
        * numpy.array (m,1) var/const"""
    # concat values of decision variables and parameters
    var_const = np.vstack([x_factors, factor_data.values_of_consts])
    kp = var_const[factor_data.var_const_to_kp]
    kq = var_const[factor_data.var_const_to_kq]
    return np.hstack([kp, kq]), var_const[factor_data.var_const_to_factor]

