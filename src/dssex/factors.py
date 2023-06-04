# -*- coding: utf-8 -*-
"""
Copyright (C) 2023 pyprg

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

Created on Sun Mar 19 12:17:49 2023

@author: pyprg

The function 'make_factor_data' returns data on factors to be applied
to nominal active and reactive power of injections and factors to be applied
to admittances of branches. The factors fall in one of two
categories 'var' or 'const'. Factors of category 'var' are decision variables.
Factors of category 'const' are parameters. Factors are specific for an
optimization step. A factor is initialized by a value of a factor from a
previous step or - if none exists - by a factor specific 'value' (which is
an attribute of the input data). Factors with step setting of -1 are
generic factors, their data are copied for each step.
"""
import casadi
import numpy as np
import egrid.factors as ef
from collections import namedtuple
_NPARRAY_0r1c = np.zeros((0,1), dtype=float)
# empty vector of values
_DM_0r1c = casadi.DM(0,1)
# empty vector of expressions
_SX_0r1c = casadi.SX(0,1)

def _make_DM_vector(array_like):
    """Creates a casadi.DM vector from array_like.

    Parameters
    ----------
    array_like: array_like

    Returns
    -------
    casadi.DM"""
    return casadi.DM(array_like) if len(array_like) else _DM_0r1c

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

Factorsymbols = namedtuple(
    'Factorsymbols', 'kpq vars consts')
Factorsymbols.__doc__ = """Symbols for factors of one step.

Parameters
----------
kpq: casadi.SX
    shape nx2, scaling factors for active and reactive power
    (for each injection)
vars: casadi.SX
    decision variables
consts: casadi.SX
    parameters"""

def make_factor_symbols(
        gen_factor_symbols, id_of_step_symbol, index_of_kpq_symbol,
        index_of_var_symbol, index_of_const_symbol):
    """Creates symbols for factors of one step. Arranges symbols in vectors.

    The result provides access to scaling factors for injections in two
    column vectors for active and reactive power which is required by
    expression building for injected current and power.

    The result provides access to decision variables and parameter symbols
    which is required for passing the symbols to a solver.

    Parameters
    ----------
    gen_factor_symbols : casadi.SX
        symbols available for each optimization step
    id_of_step_symbol : iterable
        str, identifiers of symbols
    index_of_kpq_symbol : array_like (shape n,2)
        int, indices of symbols,
        one index of an active power scaling factor and one index of a
        reactive power scaling factor per injection
    index_of_var_symbol : array_like
        int, indices of symbols for decision variables
    index_of_const_symbol : array_like
        int, indices of symbols for parameters

    Returns
    -------
    Factorsymbols
        * .kpq, casadi.SX
            shape nx2, scaling factors for active and reactive power
            (for each injection)
        * .vars, casadi.SX
            decision variables
        * .consts, casadi.SX
            parameters"""
    symbols = casadi.vertcat(
        gen_factor_symbols,
        _create_symbols_with_ids(id_of_step_symbol))
    return Factorsymbols(
        kpq=casadi.horzcat(
            symbols[index_of_kpq_symbol[:,0]].reshape((-1,1)),
            symbols[index_of_kpq_symbol[:,1]].reshape((-1,1))),
        vars=symbols[index_of_var_symbol].reshape((-1,1)),
        consts=symbols[index_of_const_symbol].reshape((-1,1)))

Factordata = namedtuple(
    'Factordata',
    'kpq vars consts '
    'values_of_vars var_min var_max is_discrete '
    'values_of_consts '
    'var_const_to_factor var_const_to_kp var_const_to_kq var_const_to_ftaps')
Factordata.__doc__="""
Symbols of variables and constants for factors.

Parameters
----------
kpq: casadi.SX
    two column vectors, symbols for scaling factors of active and
    reactive power per injection
vars: casadi.SX
    column vector, symbols for variables of scaling factors
consts: casadi.SX
    column vector, symbols for constants of scaling factors
values_of_vars: casadi.DM
    column vector, initial values for vars
var_min: casadi.DM
    lower limits of vars
var_max: casadi.DM
    upper limits of vars
is_discrete: numpy.array
    bool, flag for variable
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

def make_factor_data(model, gen_factor_symbols, step=0, f_prev=_NPARRAY_0r1c):
    """Returns data of decision variables and of parameters for a given step.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    gen_factor_symbols: casadi.SX, shape(n,1)
        symbols of generic (for each step) decision variables or parameters
    step: int
        index of optimization step, first index is 0
    f_prev: numpy.array, optional
        float, factors of previous optimization step

    Returns
    -------
    Factordata"""
    fm = ef.make_factor_meta(model, step, f_prev)
    symbols = make_factor_symbols(
        gen_factor_symbols,
        fm.id_of_step_symbol,
        fm.index_of_kpq_symbol,
        fm.index_of_var_symbol,
        fm.index_of_const_symbol)
    return Factordata(
        kpq=symbols.kpq,
        vars=symbols.vars,
        consts=symbols.consts,
        # initial values, argument in solver call
        values_of_vars=_make_DM_vector(fm.values_of_vars),
        # lower bound of scaling factors, argument in solver call
        var_min=_make_DM_vector(fm.var_min),
        # upper bound of scaling factors, argument in solver call
        var_max=_make_DM_vector(fm.var_max),
        # flag for variable
        is_discrete=fm.is_discrete,
        # values of constants, argument in solver call
        values_of_consts=_make_DM_vector(fm.values_of_consts),
        # reordering of result
        var_const_to_factor=fm.var_const_to_factor,
        var_const_to_kp=fm.var_const_to_kp,
        var_const_to_kq=fm.var_const_to_kq,
        var_const_to_ftaps=fm.var_const_to_ftaps)

def separate_factors(factordata, factors):
    """Function for extracting factors of injections and terminals.

    Extracts  from the result provided by the solver.
    Enhances factors calculated by optimization with values of parameters and
    reorders the factors according to order of injections and terminals.
    Returns kp and kq for each injection. Returns a factor ftaps for terminals
    addressed by 'Factordefs.index_of_terminal' which is an argument to the
    function 'make_factor_data'.
    The function creates a vector of values for factors which are
    decision variables and those which are constants. This vector is ordered
    for use as initial factor values in next estimation step.

    Parameters
    ----------
    factordata: Factordata
        * .values_of_consts,
            array_like, float, column vector, values for consts
        * .var_const_to_factor,
            array_like int, index_of_factor=>index_of_var_const
            converts var_const to factor (var_const[var_const_to_factor])
        * .var_const_to_kp,
            array_like int, converts var_const to kp, one active power
            scaling factor for each injection (var_const[var_const_to_kp])
        * .var_const_to_kq,
            array_like int, converts var_const to kq, one reactive power
            scaling factor for each injection (var_const[var_const_to_kq])
        * .var_const_to_ftaps,
            array_like int, converts var_const to ftaps, factor assigned to
            (selected) terminals (var_const[var_const_to_ftaps])
    factors: numpy.array|casadi.DM, shape(k,1)
        float, result of optimization (subset) without voltages

    Result
    ------
    tuple
        * numpy.array (l,2), kp, kq for each injection
        * numpy.array (m,1), for selected terminals
        * numpy.array (n,1) var/const"""
    # concat values of decision variables and parameters
    var_const = np.vstack([factors, factordata.values_of_consts])
    kp = var_const[factordata.var_const_to_kp]
    kq = var_const[factordata.var_const_to_kq]
    return (
        np.hstack([kp, kq]),
        var_const[factordata.var_const_to_ftaps],
        var_const[factordata.var_const_to_factor])

def get_factor_values(factordata, values_of_vars):
    """Reorders factors of injections and terminals.

    Parameters
    ----------
    factordata: Factordata
        * .values_of_consts,
            array_like, float, column vector, values for consts
        * .var_const_to_factor,
            array_like int, index_of_factor=>index_of_var_const
            converts var_const to factor (var_const[var_const_to_factor])
    values_of_vars: array_like
        float, calculated values of decision variables without voltages

    Returns
    -------
    numpy.array (n,1) var/const"""
    var_const = np.vstack([values_of_vars, factordata.values_of_consts])
    return var_const[factordata.var_const_to_factor]
