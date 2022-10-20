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
from scipy.sparse import coo_matrix
from egrid.builder import DEFAULT_FACTOR_ID, defk, Loadfactor
from itertools import chain
from collections import namedtuple
from src.dssex.injections import get_polynomial_coefficients
# square of voltage magnitude, minimum value for load curve,
#   if value is below _VMINSQR the load curves for P and Q converge
#   towards a linear load curve which is 0 when V=0; P(V=0)=0, Q(V=0)=0
_VMINSQR = 0.8**2
# value of zero check, used for load curve calculation
_EPSILON = 1e-12
_EMPTY_TUPLE = ()

Vinjection = namedtuple('Vinjection', 're im abs_sqr')
Vinjection.__doc__ = """
Voltage values per injection.

Parameters
----------
re: casadi.SX
    vector, real part of voltages at injections
re: casadi.SX
    vector, imaginary part of voltages at injections
abs_sqr: casadi.SX
    vector, re**2 + im**2"""

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
    """Creates expressions for longitudinal and transversal factors
    of branches.

    Parameters
    ----------
    branchtaps: pandas.DataFrame (index of taps)
        * .Vstep, float voltage diff per tap
        * .positionneutral, int
    position_syms: casadi.SX
        vector of position symbols for terms with tap

    Returns
    -------
    casadi.SX
        * [:,0] longitudinal factors
        * [:,1] transversal factors"""
    if position_syms.size1():
        # longitudinal factor
        flo = (1
            - casadi.SX(branchtaps.Vstep)
              * (position_syms - branchtaps.positionneutral))
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
        flo_ftr[:,0] - longitudinal factor
        flo_ftr[:,1] - transversal factor
    index_of_term: array_like
        taps -> index of terminal
    index_of_other_term: array_like
        taps -> index of other terminal (of same branch)

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
    ini_isna = ini.isna()
    ini[ini_isna] = -1
    return ini.astype(dtype='Int64')

def _get_default_factors(count_of_steps):
    """Generates one default scaling factor per step.

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
    (Alternative for _factor_index)

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
    (Alternative for _factor_index_per_step)

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

def _get_values_of_symbols(factor_data, value_of_previous_step):
    """Returns values for symbols. If a symbol is a variable the value
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

def _select_type(vecs, factor_data):
    """Creates column vectors from vecs by extracting elements by their
    indices.

    Parameters
    ----------
    vecs: iterable
        casadi.SX or casadi.DM, column vector
    factor_data: pandas.DataFrame
        * .index_of_symbol

    Returns
    -------
    iterator
        * casadi.SX / casadi.DM"""
    return (v[factor_data.index_of_symbol, 0] for v in vecs)

_DM_no_row = casadi.DM(0,1)

Scalingdata = namedtuple(
    'Scalingdata',
    'kp kq kvars values_of_vars kvar_min kvar_max kconsts values_of_consts '
    'symbols values')
Scalingdata.__doc__="""
Symbols of variables and constants for scaling factors.

Parameters
----------
kp: casadi.SX
    column vector, symbols for scaling factor of active power per injection
kq: casadi.SX
    column vector, symbols for scaling factor of reactive power per injection
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
    vector of all symbols (variables and constants) for extracting values
    which shall be passed to next step
    (function 'get_scaling_data', parameter 'k_prev')
values: casadi.DM
    float, given values of symbols (variables and constants)"""

def _make_DM_vector(array_like):
    """Creates a casadi.DM vector from array_like.

    Parameters
    ----------
    array_like: array_like

    Returns
    -------
    casadi.DM"""
    return casadi.DM(array_like) if len(array_like) else casadi.DM(0,1)

def get_scaling_data(
        factor_step_groups, injection_factor_step_groups,
        step=0, k_prev=_DM_no_row):
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
    symbols = create_symbols_with_ids(factors.id)
    values = _get_values_of_symbols(factors, k_prev)
    symbols_values = partial(_select_type, [symbols, values])
    factors_consts = factors[factors.type=='const']
    symbols_of_consts, values_of_consts = symbols_values(factors_consts)
    factors_var = factors[factors.type=='var']
    symbols_of_vars, values_of_vars = symbols_values(factors_var)
    return Scalingdata(
        # columns of kp/kq store an index
        kp=symbols[injections_factors.kp],
        kq=symbols[injections_factors.kq],
        kvars=symbols_of_vars,
        values_of_vars=values_of_vars,
        kvar_min=_make_DM_vector(factors_var['min']),
        kvar_max=_make_DM_vector(factors_var['max']),
        kconsts=symbols_of_consts,
        values_of_consts=values_of_consts,
        symbols=symbols,
        values=values)

def get_scaling_data_fn(model, count_of_steps=1):
    """Creates a function for creating Scalingdata.

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
    assert 0 < count_of_steps, "count_of_steps must be an int greater than 0"
    factors, injection_factors = get_factors(model, count_of_steps)
    return partial(
        get_scaling_data,
        _groupby_step(factors),
        _groupby_step(injection_factors))

#
# injected node current
#

def get_inj_current_original(inj, Vinj, P, Q):
    """Creates expressions for real and imaginary current injected into
    power flow calculation nodes per injection (load, capacitor, generator,
    battery).
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
    inj: pandas.DataFrame (int index_of_injection)
        * .Exp_v_p, float, voltage exponent of active power
        * .Exp_v_q, float, voltage exponent of reactive power
    Vinj: Vinjection
        expression for the voltage vector at injections,
        each element expresses the voltage at one paticular injection
        * .re, float, real part of voltage
        * .im, float, imaginary part of voltage
        * .abs_sqr, float, Vinj.re**2 + Vinj.im**2
    P: casadi.SX
        expression for active power when voltage is 1 pu,
        value for one phase, 1/3 of a 3-phase-load
    Q: casadi.SX
        expression for reactive power when voltage is 1 pu,
        value for one phase, 1/3 of a 3-phase-load

    Returns
    -------
    tuple
        * Ire, casadi.SX, expression for real part of current per injection
        * Iim, casadi.SX, expression for imaginary part
          of current per injection"""
    y_p = casadi.power(Vinj.abs_sqr, inj.Exp_v_p/2 - 1) * P
    y_q = casadi.power(Vinj.abs_sqr, inj.Exp_v_q/2 - 1) * Q
    return y_p * Vinj.re + y_q * Vinj.im, -y_q * Vinj.re + y_p * Vinj.im

def get_inj_current_interpolated(vminsqr, inj, Vinj, P, Q):
    """Interpolates complex current injected into power flow calculation nodes
    per injection when the absolute voltage is below vminsqr**.5. The current
    decreases when the magnitude of voltage comes closer to zero.
    ::
        I_complex(|V|) = A|V|³ + B|V|² + C|V|

    Parameters
    ----------
    vminsqr: float
        square of voltage, upper limit interpolation interval [0...vminsqr]
    inj: pandas.DataFrame (int index_of_injection)
        * .Exp_v_p, float, voltage exponent of active power
        * .Exp_v_q, float, voltage exponent of reactive power
    Vinj: Vinjection
        expression for the voltage vector at injections,
        each element expresses the voltage at one paticular injection
        * .re, float, real part of voltage
        * .im, float, imaginary part of voltage
        * .abs_sqr, float, Vinj.re**2 + Vinj.im**2
    P: casadi.SX
        expression for active power when voltage is 1 pu,
        value for one phase, 1/3 of a 3-phase-load
    Q: casadi.SX
        expression for reactive power when voltage is 1 pu,
        value for one phase, 1/3 of a 3-phase-load

    Returns
    -------
    tuple
        * Ire, casadi.SX, expression for real part of current per injection
        * Iim, casadi.SX, expression for imaginary part
          of current per injection"""
    Vinj_abs = casadi.sqrt(Vinj.abs_sqr)
    Vinj_abs_cub = Vinj.abs_sqr * Vinj_abs
    cp = get_polynomial_coefficients(vminsqr, inj.Exp_v_p)
    p_expr = (
        (cp[:,0]*Vinj_abs_cub + cp[:,1]*Vinj.abs_sqr + cp[:,2]*Vinj_abs) * P)
    cq = get_polynomial_coefficients(vminsqr, inj.Exp_v_q)
    q_expr = (
        (cq[:,0]*Vinj_abs_cub + cq[:,1]*Vinj.abs_sqr + cq[:,2]*Vinj_abs) * Q)
    calculate = _EPSILON < Vinj.abs_sqr
    Ire_ip = casadi.if_else(
        calculate, (p_expr * Vinj.re + q_expr * Vinj.im) / Vinj.abs_sqr, 0.0)
    Iim_ip = casadi.if_else(
        calculate, (-q_expr * Vinj.re + p_expr * Vinj.im) / Vinj.abs_sqr, 0.0)
    return Ire_ip, Iim_ip

def current_into_injection(
        injections, node_to_inj, Vnode, scaling_data, vminsqr=_VMINSQR):
    """Creates casadi.SX expressions (vectors) for real and imaginary parts of
    current flowing into injections.

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
        * Vnode[:,2], float, Vre**2 + Vim**2, vector of imaginary node voltages
    scaling_data: Scalingdata
        * .kp, casadi.SX expression, vector of injection scaling factors for
          active power
        * .kq, casadi.SX expression, vector of injection scaling factors for
          reactive power
    vminsqr: float
        square of voltage, upper limit interpolation interval [0...vminsqr]

    Returns
    -------
    tuple
        * Ire, casadi.SX, expression for real part of current per injection
        * Iim, casadi.SX, expression for imaginary part
          of current per injection"""
    Vinj = Vinjection(
        re=node_to_inj @ Vnode[:,0],
        im=node_to_inj @ Vnode[:,1],
        abs_sqr=node_to_inj @ Vnode[:,2])
    # calculates per phase, assumes P10 and Q10 are sums of 3 per-phase-values
    P = casadi.vcat(injections.P10 / 3) * scaling_data.kp
    Q = casadi.vcat(injections.Q10 / 3) * scaling_data.kq
    Iinj_re_orig, Iinj_im_orig = get_inj_current_original(
        injections, Vinj, P, Q)
    Iinj_re_ip, Iinj_im_ip = get_inj_current_interpolated(
        vminsqr, injections, Vinj, P, Q)
    # compose current functions from original and interpolated
    interpolate = Vinj.abs_sqr < vminsqr
    Iinj_re = casadi.if_else(interpolate, Iinj_re_ip, Iinj_re_orig)
    Iinj_im = casadi.if_else(interpolate, Iinj_im_ip, Iinj_im_orig)
    return Iinj_re, Iinj_im

def _injected_node_current(inj_to_node, Iinj_re, Iinj_im):
    """Converts current per injection into current per power flow calculation
    node.
    
    Parameters
    ----------
    inj_to_node: casadi.SX
        the matrix converts injection values into node values
        node_values = inj_to_node @ injection_values
    Iinj_re: casadi.SX
        vector of current per injection, real part
    Iinj_im: casadi.SX
        vector of current per injection, imaginary part
    
    Returns
    -------
    tuple
        * casadi.SX, injected real current per node
        * casadi.SX, injected imaginary current per node"""
    return inj_to_node @ Iinj_re, inj_to_node @ Iinj_im

def get_injected_node_current(
        injections, node_to_inj, Vnode, scaling_data, vminsqr=_VMINSQR):
    """Creates casadi.SX expressions (vectors) for real and imaginary part of
    current injected into power flow calculation nodes. Vectors have elements
    per power flow calculation node.

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
        * Vnode[:,2], float, Vre**2 + Vim**2, vector of imaginary node voltages
    scaling_data: Scalingdata
        * .kp, casadi.SX expression, vector of injection scaling factors for
          active power
        * .kq, casadi.SX expression, vector of injection scaling factors for
          reactive power
    vminsqr: float
        square of voltage, upper limit interpolation interval [0...vminsqr]

    Returns
    -------
    tuple
        * Ire, casadi.SX, expression for real part of injected current per node
        * Iim, casadi.SX, expression for imaginary part of injected
          current per node"""
    Iinj = current_into_injection(
        injections, node_to_inj, Vnode, scaling_data, vminsqr)
    return _injected_node_current(node_to_inj.T, *Iinj)

def _reset_slack_current(
        slack_indices, Vre_slack_syms, Vim_slack_syms, Iinj_re, Iinj_im):
    Iinj_re[slack_indices] = -Vre_slack_syms
    Iinj_im[slack_indices] = -Vim_slack_syms
    return casadi.vertcat(Iinj_re, Iinj_im)

def _create_gb_mn_tot(branchterminals, branchtaps, position_syms):
    """Creates g_mn, b_mn, g_mm, b_mm (mutual and self conductance
    and susceptance) for each terminal taking tappositions into consideration.

    Parameters
    ----------
    branchterminals: pandas.DataFrame (index of terminal)
        * .glo, mutual conductance, longitudinal
        * .blo, mutual susceptance, longitudinal
        * .g_tr_half, half of self conductance, tranversal
        * .b_tr_half, half of self susceptance, tranversal
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
    gb_mn_mm = _create_gb_expressions(branchterminals)
    if position_syms.size1():
        flo_ftr = get_tap_factors(branchtaps, position_syms)
        gb_mn_mm = _mult_gb_by_tapfactors(
            gb_mn_mm,
            flo_ftr,
            branchtaps.index_of_term,
            branchtaps.index_of_other_term)
    gb_mn_mm[:,2] += gb_mn_mm[:,0]
    gb_mn_mm[:,3] += gb_mn_mm[:,1]
    return gb_mn_mm

def create_expressions(model):
    """Creates symbols for node and slack voltages, tappositions,
    branch conductance/susceptance and an expression for Y @ V. The symbols
    and expressions are regarded constant over multiple calculation steps.
    Diagonal of slack rows are set to 1 for conductance and 0 for susceptance,
    other values of slack rows are set to 0.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid

    Returns
    -------
    dict
        * 'Vnode_syms', casadi.SX, expressions of node Voltages
        * 'Vre_slack_syms', casadi.SX, symbols of slack voltages, real part
        * 'Vim_slack_syms', casadi.SX, symbols of slack voltages,
           imaginary part
        * 'position_syms', casadi.SX, symbols of tap positions
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
            casadi.SX.sym('Vre_slack', model.count_of_slacks),
            casadi.SX.sym('Vim_slack', model.count_of_slacks)),
        position_syms=position_syms,
        gb_mn_tot=gb_mn_tot,
        Y_by_V=multiply_Y_by_V(Vnode_syms, G, B))

#
# power flow calculation
#

def _calculate_power_flow(
        model, scaling_data, expr, Iinjection, tappositions=None, Vinit=None):
    """Solves the power flow problem using a rootfinding algorithm.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    scaling_data: Scalingdata
        optional
    expr: dict
        optional
        * 'Vnode_syms', casadi.SX, expressions of node Voltages
        * 'Vre_slack_syms', casadi.SX, symbols of slack voltages, real part
        * 'Vim_slack_syms', casadi.SX, symbols of slack voltages,
           imaginary part
        * 'position_syms', casadi.SX, symbols of tap positions
        * 'Y_by_V', casadi.SX, expression for Y @ V
    Iinjection: tuple, casadi.SX
        * Ire, real part of current flowing into injection
        * Ire, imaginary part of current flowing into injection
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
    Iinj_ = _injected_node_current(casadi.SX(model.mnodeinj), *Iinjection)
    Vslack_syms = expr['Vslack_syms'][:,0], expr['Vslack_syms'][:,1]
    parameter_syms=casadi.vertcat(
        *Vslack_syms, expr['position_syms'], scaling_data.symbols)
    slacks = model.slacks
    Iinj = _reset_slack_current(slacks.index_of_node, *Vslack_syms, *Iinj_)
    variable_syms = casadi.vertcat(Vnode_syms[:,0], Vnode_syms[:,1])
    fn_Iresidual = casadi.Function(
        'fn_Iresidual',
        [variable_syms, parameter_syms],
        [expr['Y_by_V'] + Iinj])
    rf = casadi.rootfinder('rf', 'nlpsol', fn_Iresidual, {'nlpsol':'ipopt'})
    count_of_pfcnodes = model.shape_of_Y[0]
    Vinit_ = (
        casadi.vertcat([1.]*count_of_pfcnodes, [0.]*count_of_pfcnodes)
        if Vinit is None else Vinit)
    tappositions_ = (
        model.branchtaps.position if tappositions is None else tappositions)
    values_of_parameters=casadi.vertcat(
        np.real(slacks.V), np.imag(slacks.V),
        tappositions_,
        scaling_data.values)
    voltages = rf(Vinit_, values_of_parameters)
    return rf.stats()['success'], voltages

def calculate_power_flow(model, expr=None, tappositions=None, Vinit=None):
    """Solves the power flow problem using a rootfinding algorithm.

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
    expr_ = create_expressions(model) if expr is None else expr
    scaling_data = get_scaling_data_fn(model, count_of_steps=1)(step=0)
    Iinjection = current_into_injection(
        model.injections, 
        casadi.SX(model.mnodeinj.T), 
        expr_['Vnode_syms'], 
        scaling_data)
    return _calculate_power_flow(
        model, scaling_data, expr_, Iinjection, tappositions, Vinit)

##############
# Estimation #
##############

# power and current flow into branches

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
    """Calculates active and reactive power flow for a subset of 
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
        * Vnode[:,2], float, Vre**2 + Vim**2, vector of imaginary node voltages
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
    tuple
        * Ire
        * Iim"""
    Ire = g_tot * Vre - b_tot * Vim - g_mn * Vre_other + b_mn * Vim_other
    Iim = b_tot * Vre + g_tot * Vim - b_mn * Vre_other - g_mn * Vim_other
    return Ire, Iim

def current_into_branch(gb_mn_tot, Vnode, terms):
    """Computes real and imaginary current flowing into given subset of branch
    terminals from branch admittances and voltages at branch terminals. Assumes
    PI-equivalient circuit.

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
    tuple
        * Ire
        * Iim"""
    if len(terms):
        Vterm = Vnode[terms.index_of_node,:]
        Vother = Vnode[terms.index_of_other_node,:]
        gb_mn_tot_ = gb_mn_tot[terms.index, :]
        return _current_into_branch(
            gb_mn_tot_[:,2], gb_mn_tot_[:,3], gb_mn_tot_[:,0], gb_mn_tot_[:,1],
            Vterm[:,0], Vterm[:,1], Vother[:,0], Vother[:,1])
    return casadi.SX(0,1), casadi.SX(0,1)

def get_branch_expressions_fn_factory(expr):
    """Returns a function returning a new funtion for expression building
    function('PQ' | 'I') -> function(branchterminals) -> (casadi.SX).
    
    Parameters
    ----------
    expr: dict
        * 'Vnode_syms', casadi.SX, vectors, symbols of node voltages
        * 'gb_mn_tot', casadi.SX, vectors, conductance and susceptance of
          branches connected to terminals
        
    Returns
    -------
    function
        ('PQ'|'I') ->((pandas.DataFrames) -> (casadi.SX))
        (indicator) -> ((branch terminals)->(power or current per terminal))"""
    Vnode_syms = expr['Vnode_syms']
    gb_mn_tot = expr['gb_mn_tot']
    def get_branch_expressions(selector):
        """Returns a function calculating either active and reactive power 
        or real and imaginary part of current flowing 
        into given branchterminals.
        
        Parameters
        ----------
        selector: 'PQ'|'I'
            selects which values to calculate 'PQ' - active and reactive power,
            'I' - real and imaginary part of current
        
        Returns
        -------
        function
            (pandas.DataFrame) -> (casadi.SX with shape n,2)
            branchterminals: pandas.DataFrame (index of terminal)
                * .index_of_node, int,
                   index of power flow calculation node connected to terminal
                * .index_of_other_node, int
                   index of power flow calculation node connected to 
                   other terminal of same branch like terminal"""
        assert selector=='PQ' or selector=='I',\
            f'value of indicator must be "PQ" or "I" but is "{selector}"'
        expr_fn = power_into_branch if selector=='PQ' else current_into_branch
        return partial(expr_fn, gb_mn_tot, Vnode_syms)
    return get_branch_expressions

def get_branch_values_fn(expr, voltages_ri, tappositions):
    """Returns a function which calculates active/reactive power for
    given terminals or real/imaginary part for given terminals. The returned
    function is intended to calculate values of a subset of all terminals
    of the model. Calculation will fail for branches with too high admittance.
    
    Parameters
    ----------
    expr: dict
        * 'Vnode_syms', casadi.SX, vectors, symbols of node voltages
        * 'position_syms', casadi.SX, vector, symbols of tap positions
        * 'gb_mn_tot', casadi.SX, vectors, conductance and susceptance of
          branches connected to terminals
    voltages_ri: casadi.DM
        float, voltages at nodes first real parts then imaginary parts
    tappositions: array_like
        int
    
    Returns
    -------
    function
        ('PQ'|'I', pandas.DataFrames) -> (casadi.DM)
        (indicator, terminals) -> (power or current per terminal)"""
    get_branch_expressions_fn = get_branch_expressions_fn_factory(expr)
    Vnode_syms = expr['Vnode_syms']
    symbols = [
        casadi.vertcat(Vnode_syms[:,0], Vnode_syms[:,1]), 
        expr['position_syms']]
    def get_branch_values(selector, branchterminals):
        """Calculates either active and reactive power or real and imaginary
        part of current flowing into given branchterminals.
        
        Parameters
        ----------
        selector: 'PQ'|'I'
            selects which values to calculate 'PQ' - active and reactive power,
            'I' - real and imaginary part of current
        branchterminals: pandas.DataFrame (index of terminal)
            * .index_of_node, int,
               index of power flow calculation node connected to terminal
            * .index_of_other_node, int
               index of power flow calculation node connected to other terminal
               of same branch like terminal
        
        Returns
        -------
        casadi.DM (shape is n,2)"""
        get_branch_expressions = get_branch_expressions_fn(selector)
        get_results = casadi.Function(
            f'get_{selector}_branch_results', 
            symbols, 
            get_branch_expressions(branchterminals))
        return get_results(voltages_ri, tappositions)
    return get_branch_values

#
# difference: measured - calculated
#

def _get_I_expressions(get_I_branch_exprs, branchterminals):
    """Creates an expression for calculating magnitudes of current at 
    given terminals.
    
    Parameters
    ----------
    get_I_branch_exprs: function
        generates a tuple of vectors, one vector for expressions of 
        real current flowing into a branch and one vector for imaginary current
        (pandas.DataFrame) -> (tuple: casadi.SX, casadi.SX)
        (branchterminals) -> (tuple: vector of Ire, vector of Iim)
    branchterminals: pandas.DataFrame
        terminals of batch
    
    Returns
    -------
    casadi.SX, shape (n,1)"""
    # I = sqrt(Ire**2, Iim**2), vector
    return casadi.sum2(
        # Ire**2, Iim**2
        casadi.power(
            # Ire, Iim
            casadi.hcat(get_I_branch_exprs(branchterminals)),
            2)
        # I = sqrt(Ire**2, Iim**2)
        .sqrt())

def _make_get_branch_expr(get_branch_expressions_fn, selector):
    """Returns an expression building function for I/P/Q-values at 
    terminals of branches.
    
    Parameters
    ----------
    get_branch_expressions_fn: function
        factory function for creating another function using the selector
        ('I'|'P'|'Q') -> function(
                (pandas.DataFrame) -> (tuple: casadi.SX, casadi.SX))
        the created function returns expression for I/P/Q flowing into
        given terminals (pandas.DataFrame).
    selector: 'I'|'P'|'Q'
        selects the measurement entites to create differences for
    
    Returns
    -------
    function
        (str, pandas.DataFrame) -> (casasdi.SX, shape 1,1)
        (id_of_batch, branchterminals) -> (difference: measured-calculated)"""
    assert selector in 'IPQ', \
        f'selector needs to be one of "I", "P" or "Q" but is "{selector}"'
    if selector=='I':
        get_I_branch_exprs = get_branch_expressions_fn('I')
        return partial(_get_I_expressions, get_I_branch_exprs)
    get_PQ_branch_exprs = get_branch_expressions_fn('PQ')
    if selector=='P':
        return lambda branchterminals: get_PQ_branch_exprs(branchterminals)[0]
    if selector=='Q':
        return lambda branchterminals: get_PQ_branch_exprs(branchterminals)[1]
    assert False, f'no processing implemented for selector "{selector}"'
        
def _get_values(model, selector):
    """Helper for returning I/P/Q-values from model using a str 'I'|'P'|'Q'.
    
    Parametes
    ---------
    model: egrid.model.Model
        model of electric network for calculation
    selector: 'I'|'P'|'Q'
        accesses model.ivalues | model.pvalues | model.qvalues
        
    Returns
    -------
    pandas.DataFrame
        model.ivalues | model.pvalues | model.qvalues"""
    assert selector in 'IPQ', \
        f'selector needs to be one of "I", "P" or "Q" but is "{selector}"'
    if selector=='I':
        return model.ivalues
    if selector=='P':
        return model.pvalues
    if selector=='Q':
        return model.qvalues
    assert False, f'no processing implemented for selector "{selector}"'

def _make_get_value(values, selector):
    """Helper, retrieves value from ivalues, pvalues or qvalues.
    
    Parameters
    ----------
    values: pandas.DataFrame
    
    selector: 'I'|'P'|'Q'
    
    Returns
    -------
    function
        (str) -> (float)
        (id_of_batch) -> (value)"""
    vals = (
        values[selector] 
        if selector=='I' else values[selector] * values.direction)
    return lambda id_of_batch: vals[id_of_batch]

def _get_terminals_of_batches(branchoutputs, branchterminals):
    """Returns id_of_batch and associated branch terminals.
    
    Parameters
    ----------
    branchoutputs: pandas.DataFrame
        * .id_of_batch
        * .index_of_term
    branchterminals: pandas.DataFrame (index of terminal)
    
    Returns
    -------
    iterator
        tuple
            * str, id_of_batch
            * pandas.DataFrame, terminals"""
    termgroups = (
         branchoutputs.loc[:, ['id_of_batch', 'index_of_term']]
         .groupby('id_of_batch'))
    return ((id_of_batch, branchterminals.loc[df.index_of_term])
            for id_of_batch, df in termgroups)

def get_batch_expressions(model, get_branch_expressions_fn, selector=''):
    """Creates a vector (casadi.SX, shape n,1) expressing the difference
    between measured and calculated values for absolute current, active power
    or reactive power. The expressions are based on the batch definitions.
    Intended use is retrieving values from power flow calculation result.
    
    Parametes
    ---------
    model: egrid.model.Model
        model of electric network for calculation
    get_branch_expressions_fn: function
        factory function for creating another function using the selector
        ('I'|'P'|'Q') -> function(
                (pandas.DataFrame) -> (tuple: casadi.SX, casadi.SX))
        the created function returns expression for I/P/Q flowing into
        given terminals (pandas.DataFrame).
    selector: 'I'|'P'|'Q'
        addresses current magnitude, active power or reactive power
    
    Returns
    -------
    dict
        id_of_batch => expression for I/P/Q-calculation"""
    assert selector in 'IPQ', \
        f'selector needs to be one of "I", "P" or "Q" but is "{selector}"'
    values = _get_values(model, selector).set_index('id_of_batch')
    branchoutputs = model.branchoutputs
    is_relevant = branchoutputs.id_of_batch.isin(values.index)
    get_branch_expr = _make_get_branch_expr(
        get_branch_expressions_fn, selector)
    return {id_of_batch: casadi.sum1(get_branch_expr(branchterminals))
            for id_of_batch, branchterminals in 
                _get_terminals_of_batches(
                    branchoutputs.loc[is_relevant], model.branchterminals)}

def get_value_diffs(model, get_branch_expressions_fn, selector=''):
    """Creates a vector (casadi.SX, shape n,1) expressing the difference
    between measured and calculated values for absolute current, active power
    or reactive power. The expressions are based on the batch definitions.
    Intended use is building the objective function or constraints.
    
    Parametes
    ---------
    model: egrid.model.Model
        model of electric network for calculation
    get_branch_expressions_fn: function
        factory function for creating another function using the selector
        ('I'|'P'|'Q') -> function(
                (pandas.DataFrame) -> (tuple: casadi.SX, casadi.SX))
        the created function returns expression for I/P/Q flowing into
        given terminals (pandas.DataFrame).
    selector: 'I'|'P'|'Q'
        addresses current magnitude, active power or reactive power
    
    Returns
    -------
    casadi.SX
        vector"""
    assert selector in 'IPQ', \
        f'selector needs to be one of "I", "P" or "Q" but is "{selector}"'
    values = _get_values(model, selector).set_index('id_of_batch')
    branchoutputs = model.branchoutputs
    is_relevant = branchoutputs.id_of_batch.isin(values.index)
    get_value = _make_get_value(values, selector)
    get_branch_expr = _make_get_branch_expr(
        get_branch_expressions_fn, selector)
    return casadi.vcat(
        [get_value(id_of_batch) - casadi.sum1(get_branch_expr(branchterminals))
         for id_of_batch, branchterminals in 
         _get_terminals_of_batches(
             branchoutputs.loc[is_relevant], model.branchterminals)])

def get_node_values(index_of_node, Vnode_ri):
    """Returns absolute voltages for addressed nodes.

    Parameters
    ----------
    index_of_node: array_like, int
        addresses nodes
    Vnode_ri: casadi.DM
        * [index_of_node, Vre]

    Returns
    -------
    casadi.SX"""
    if len(index_of_node):
        Vnode_ri_ = casadi.hcat(
            casadi.vertsplit(Vnode_ri, Vnode_ri.size1()//2))[index_of_node,:]
        Vsqr = casadi.power(Vnode_ri_, 2)
        return (Vsqr[:, 0] + Vsqr[:, 1]).sqrt()
    return casadi.DM(0,1)

def value_of_voltages(vvalues):
    """Extracts value of voltages from vvalues.

    Parameters
    ----------
    vvalues: pandas.DataFrame
        * index_of_node, int
        * V, absolute value of voltage at node

    Returns
    -------
    pandas.DataFrame
        * V, float, absolute value of voltage"""
    return vvalues[['V', 'index_of_node']].groupby('index_of_node').mean()

# def voltage_at_node(index_of_node, Vnode):
#     """Returns expression of absolute voltage at addressed nodes.

#     Parameters
#     ----------
#     index_of_node: array_like, int
#         addresses nodes
#     Vnode: casadi.SX
#         * Vnode[index_of_node, 2] => Vabs**2

#     Returns
#     -------
#     casadi.SX"""
#     return (
#         Vnode[index_of_node, 2].sqrt()
#         if len(index_of_node)
#         else casadi.SX(0,1))

def ri_to_complex(array_like):
    """Converts a vector with separate real and imaginary parts into a
    vector of complex values.

    Parameters
    ----------
    array_like: casadi.DM (and other types?)

    Returns
    -------
    numpy.array"""
    return np.hstack(np.vsplit(array_like, 2)).view(dtype=np.complex128)

