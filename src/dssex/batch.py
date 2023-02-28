# -*- coding: utf-8 -*-
"""
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
Created on Fri Dec 16 00:00:58 2022

@author: pyprg
"""
import numpy as np
from functools import partial
from collections import defaultdict
from pandas import concat
from dssex.injections import calculate_cubic_coefficients
# square of voltage magnitude, minimum value for load curve,
#   if value is below _VMINSQR the load curves for P and Q converge
#   towards a linear load curve which is 0 when V=0; P(V=0)=0, Q(V=0)=0
_VMINSQR = 0.8**2
# value of zero check, used for load curve calculation
_EPSILON = 1e-12

def get_values(model, selector):
    """Helper for returning I/P/Q/V-values from model using a
    string 'I'|'P'|'Q'|'V'.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric grid
    selector: 'I'|'P'|'Q'|'V'
        accesses model.ivalues | model.pvalues | model.qvalues | model.vvalues

    Returns
    -------
    pandas.DataFrame
        model.ivalues | model.pvalues | model.qvalues | model.vvalues"""
    assert selector in 'IPQV', \
        f'selector needs to be one of "I", "P", "Q" or "V" but is "{selector}"'
    if selector=='I':
        return model.ivalues
    if selector=='P':
        return model.pvalues
    if selector=='Q':
        return model.qvalues
    if selector=='V':
        return model.vvalues
    assert False, f'no processing implemented for selector "{selector}"'

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
    groups =  (
        vvalues[['V', 'index_of_node', 'id_of_node']].groupby('index_of_node'))
    return concat([groups['V'].mean(), groups['id_of_node'].min()], axis=1)

#
# injected power / current
#

def _calculate_injected_power_n(Vinj_abs_sqr, Exp_v, PQ):
    """Numerically calculates injected active or reactive power or both.

    Parameters
    ----------
    Vinj_abs_sqr: numpy.array, shape n,1
        square of voltage magnitude at terminals of injections
    Exp_v: numpy array (n,1 or n,2)
        voltage exponents for active or reactive powre or both
    PQ: numpy array (n,1 or n,2)
        active or reactive power or both, dimension must match dimension of
        Exp_v

    Returns
    -------
    numpy.array (shape PQ.shape)"""
    assert Exp_v.shape == PQ.shape, \
        f'shapes of Exp_v and PQ must match ' \
        f'but do not {Exp_v.shape}!={PQ.shape}'
    return (
        (np.power(Vinj_abs_sqr.reshape(-1, 1), Exp_v/2) * PQ)
        .reshape(PQ.shape))

def _interpolate_injected_power_n(Vinj_abs_sqr, Exp_v, PQ, vminsqr):
    """Interpolates values of injected power.

    Parameters
    ----------
    Vinj_abs_sqr: numpy.array, shape n,1
        square of voltage magnitude at terminals of injections
        Vri[:,0]**2 + Vri[:,1]**2
    Exp_v: numpy.array, shape n,1 or n,2
        voltage exponents of active and or reactive power
    PQ: numpy.array, shape n,1 or n,2
        active and or reactive power
    vminsqr: float
        square of voltage, upper limit of interpolation interval [0...vminsqr]

    Returns
    -------
    numpy.array, shape n,1 or n,2"""
    # interpolated P and Q
    assert Vinj_abs_sqr.shape[0] == Exp_v.shape[0] == PQ.shape[0], \
        'Vinj_abs_sqr, Exp_v and PQ must have equal number of rows'
    assert Exp_v.shape == PQ.shape, \
        f'shapes of Exp_v and PQ must match ' \
        f'but do not {Exp_v.shape}!={PQ.shape}'
    if Vinj_abs_sqr.shape[0]:
        c = calculate_cubic_coefficients(vminsqr, Exp_v)
        Vinj_abs_sqr_ = Vinj_abs_sqr.reshape(-1, 1)
        Vinj_abs = np.sqrt(Vinj_abs_sqr_)
        Vvector = np.hstack([Vinj_abs_sqr_*Vinj_abs, Vinj_abs_sqr_, Vinj_abs])
        f_pq = (np.expand_dims(Vvector,axis=1) @ c).reshape(Exp_v.shape)
        return f_pq * PQ
    else:
        return np.ndarray(Exp_v.shape, dtype=float)

def _injected_power_n(
        injections, node_to_inj, Vnode_abs_sqr, kpq, vminsqr=_VMINSQR):
    """Numerically calculates magnitudes of injected power flowing
    into injections.
    This function is intended for calculation of power for selected
    injections only.

    Parameters
    ----------
    injections: pandas.DataFrame (int index_of_injection)
        subset of the injections of the model
        * .P10, float, rated active power at voltage of 1.0 pu
        * .Q10, float, rated reactive power at voltage of 1.0 pu
        * .Exp_v_p, float, voltage exponent of active power
        * .Exp_v_q, float, voltage exponent of reactive power
    node_to_inj: casadi.SX
        the matrix converts node to injection values, matrix for all
        injecions of the model
        injection_values = node_to_inj @ node_values
    Vnode_abs_sqr: numpy.array<float>
        * Vnode_re**2 + Vnode_im**2
    kpq: numpy.array (shape n,2)
        vector of injection scaling factors for active and reactive power,
        factors for all injections of model
    vminsqr: float
        square of voltage, upper limit of interpolation interval [0...vminsqr]

    Returns
    -------
    numpy.array, shape n,2
        [:,0] - real part of injected current
        [:,1] - imaginary part of injected current"""
    # voltages at injections
    count_of_injections = len(injections)
    idx_of_injections = injections.index
    PQ_inj = np.ndarray(shape=(count_of_injections,2), dtype=float)
    Vabs_sqr = node_to_inj[idx_of_injections] @ Vnode_abs_sqr
    interpolate = Vabs_sqr < vminsqr
    # assumes P10 and Q10 are sums of 3 per-phase-values
    PQscaled = (
        kpq[idx_of_injections] * (injections[['P10', 'Q10']].to_numpy() / 3))
    Exp_v = injections[['Exp_v_p', 'Exp_v_p']].to_numpy()
    PQ_inj[~interpolate] = _calculate_injected_power_n(
        Vabs_sqr[~interpolate], Exp_v[~interpolate], PQscaled[~interpolate])
    # interpolated power
    Vabs_sqr_ip = Vabs_sqr[interpolate]
    PQ_inj[interpolate] = _interpolate_injected_power_n(
        Vabs_sqr_ip, Exp_v[interpolate], PQscaled[interpolate], vminsqr)
    return PQ_inj

def _calculate_injected_current_n(Vri, Vabs_sqr, Exp_v, PQscaled):
    """Calculates values of injected current.

    Parameters
    ----------
    Vri: numpy.array, shape n,2
        voltage at terminals of injections
        Vri[:,0] - real part of voltage
        Vri[:,1] - imaginary part of voltage
    Vabs_sqr: numpy.array, shape n,1
        square of voltage magnitude at terminals of injections
        Vri[:,0]**2 + Vri[:,1]**2
    Exp_v: numpy.array, shape n,2
        voltage exponents of active and reactive power
    PQscaled: numpy.array, shape n,2
        active and reactive power at nominal voltage multiplied
        by scaling factors
        PQscaled[:,0] - active power
        PQscaled[:,1] - reactive power

    Returns
    -------
    numpy.array, shape n,2
        [:,0] - real part of injected current
        [:,1] - imaginary part of injected current"""
    ypq = np.power(Vabs_sqr.reshape(-1, 1), Exp_v/2 - 1) * PQscaled
    Ire = np.sum(ypq * Vri, axis=1).reshape(-1, 1)
    Iim = (-ypq[:,1]*Vri[:,0] + ypq[:,0]*Vri[:,1]).reshape(-1, 1)
    return np.hstack([Ire, Iim])

def _interpolate_injected_current_n(Vinj, Vabs_sqr, PQinj):
    """Interpolates values of injected current in a voltage interval,
    S = f(|V|).

    Parameters
    ----------
    Vinj: numpy.array, shape n,2
        voltage at terminals of injections
        Vinj[:,0] - real part of voltage
        Vinj[:,1] - imaginary part of voltage
    Vabs_sqr: numpy.array, shape n,1
        square of voltage magnitude at terminals of injections
        Vinj[:,0]**2 + Vinj[:,1]**2
    PQinj: numpy.array, shape n,2
        injected active and reactive power
        Sinj[:,0] - active power
        Sinj[:,1] - reactive power

    Returns
    -------
    numpy.array, shape n,2
        [:,0] - real part of injected current
        [:,1] - imaginary part of injected current"""
    Pip = PQinj[:,0]
    Qip = PQinj[:,1]
    Vre = Vinj[:, 0]
    Vim = Vinj[:, 1]
    calculate = _EPSILON < Vabs_sqr
    Ire = np.where(calculate, (Pip * Vre + Qip * Vim) / Vabs_sqr, 0.0)
    Iim = np.where(calculate, (-Qip * Vre + Pip * Vim) / Vabs_sqr, 0.0)
    return np.hstack([Ire.reshape(-1, 1), Iim.reshape(-1, 1)])

def _injected_current_n(
        injections, node_to_inj, Vnode_ri2, Vnode_abs_sqr, kpq,
        vminsqr=_VMINSQR):
    """Numerically calculates magnitudes of injected currents flowing
    into injections.
    This function is intended for calculation of currents for selected
    injections only.

    Parameters
    ----------
    injections: pandas.DataFrame (int index_of_injection)
        subset of the injections of the model
        * .P10, float, rated active power at voltage of 1.0 pu
        * .Q10, float, rated reactive power at voltage of 1.0 pu
        * .Exp_v_p, float, voltage exponent of active power
        * .Exp_v_q, float, voltage exponent of reactive power
    node_to_inj: casadi.SX
        the matrix converts node to injection values, matrix for all
        injecions of the model
        injection_values = node_to_inj @ node_values
    Vnode_ri2: numpy.array
        * Vnode_ri2[:,0], float, Vre vector of real node voltages
        * Vnode_ri2[:,1], float, Vim vector of imaginary node voltages
    Vnode_abs_sqr: numpy.array
        * float, Vre**2 + Vim**2
    kpq: numpy.array (shape n,2)
        vector of injection scaling factors for active and reactive power,
        factors for all injections of model
    vminsqr: float
        square of voltage, upper limit interpolation interval [0...vminsqr]

    Returns
    -------
    numpy.array, shape n,2
        [:,0] - real part of injected current
        [:,1] - imaginary part of injected current"""
    # voltages at injections
    count_of_injections = len(injections)
    idx_of_injections = injections.index
    Iri = np.empty(shape=(count_of_injections,2), dtype=float)
    Vinj_ri2 = node_to_inj[idx_of_injections] @ Vnode_ri2
    Vinj_abs_sqr = node_to_inj[idx_of_injections] @ Vnode_abs_sqr
    # assumes P10 and Q10 are sums of 3 per-phase-values
    PQscaled = (
        kpq[idx_of_injections]
        * (injections[['P10', 'Q10']].to_numpy() / 3))
    Exp_v = injections[['Exp_v_p', 'Exp_v_p']].to_numpy()
    interpolate = Vinj_abs_sqr < vminsqr
    # interpolated power
    Vabs_sqr_ip = Vinj_abs_sqr[interpolate]
    PQinj_ip = _interpolate_injected_power_n(
        Vabs_sqr_ip, Exp_v[interpolate], PQscaled[interpolate], vminsqr)
    # interpolated current
    Iri[interpolate] = _interpolate_injected_current_n(
        Vinj_ri2[interpolate], Vabs_sqr_ip, PQinj_ip)
    # current according to given load curve
    orig = ~interpolate
    Iri[orig] = _calculate_injected_current_n(
        Vinj_ri2[orig], Vinj_abs_sqr[orig], Exp_v[orig], PQscaled[orig])
    return Iri

def _get_injected_value(
        node_to_inj, Vnode_ri2, Vabs_sqr, kpq, selector, vminsqr, injections):
    """Returns one of electric current I, active power P or reactive power Q
    selected by selector for given injections.

    Parameters
    ----------
    node_to_inj: casadi.SX
        the matrix converts node to injection values, matrix for all
        injecions of the model
        injection_values = node_to_inj @ node_values
    Vnode_ri2: numpy.array<float> (shape n,3)
        [:,0] Vnode_re, real part of node voltage
        [:,1] Vnode_im, imaginary part of node voltage
    Vabs_sqr: numpy.array<float> (shape n,1)
        Vnode_ri2[:,0]**2 + Vnode_ri2[:,1]**2
    kpq: numpy.array (shape n,2)
        vector of injection scaling factors for active and reactive power,
        factors for all injections of model
    selector : TYPE
        DESCRIPTION.
    vminsqr: float
        square of voltage, upper limit of interpolation interval [0...vminsqr]
    injections: pandas.DataFrame (int index_of_injection)
        subset of the injections of the model
        * .P10, float, rated active power at voltage of 1.0 pu
        * .Q10, float, rated reactive power at voltage of 1.0 pu
        * .Exp_v_p, float, voltage exponent of active power
        * .Exp_v_q, float, voltage exponent of reactive power

    Returns
    -------
    numpy.array<float> (shape 1,2) or float"""
    if selector=='I':
        return np.sum(
            _injected_current_n(
                injections, node_to_inj, Vnode_ri2, Vabs_sqr, kpq, vminsqr),
            axis=0).reshape(-1, 2)
    if selector=='P':
        return np.sum(
            _injected_power_n(injections, node_to_inj, Vabs_sqr, kpq, vminsqr)
            [:,0])
    if selector=='Q':
        return np.sum(
            _injected_power_n(injections, node_to_inj, Vabs_sqr, kpq, vminsqr)
            [:,1])
    assert False, \
        f'selector needs to be one of "I", "P" or "Q" but is "{selector}"'

def get_batches(values, outputs, column_of_device_index):
    """Creates an iterator over tuples (id_of_batch, pandas.DataFrame)
    for given values and outputs.

    Parameters
    ----------
    values : pandas.DataFrame
        model.ivalues | model.pvalues | model.qvalues | model.vvalues
    outputs : pandas.DataFrame
        model.branchoutputs|model.injectionoutputs
    column_of_device_index : str
        'index_of_term'|'index_of_injection'

    Returns
    -------
    iterator
        tuple
            * str, id_of_batch
            * pandas.DataFrame"""
    ids_of_batches = values.id_of_batch
    is_relevant = outputs.id_of_batch.isin(ids_of_batches)
    relevant_out = (
        outputs
        .loc[is_relevant, ['id_of_batch', column_of_device_index]])
    return (id_of_batch__df
        for id_of_batch__df in relevant_out.groupby('id_of_batch'))

def _get_batch_values_inj(
        model, Vnode_ri2, Vabs_sqr, kpq, selector, vminsqr=_VMINSQR):
    """Calculates a vector (numpy.array, shape n,1) of injected absolute
    current, active power or reactive power. The expressions are based
    on the batch definitions.

    Parameters
    ----------
    model: egrid.model.Model
        model of electric network for calculation
    Vnode_ri2: numpy.array<float> (shape n,3)
        [:,0] Vnode_re, real part of node voltage
        [:,1] Vnode_im, imaginary part of node voltage
    Vabs_sqr: numpy.array<float> (shape n,1)
        Vnode_ri2[:,0]**2 + Vnode_ri2[:,1]**2
    kpq: numpy.array (shape n,2)
        vector of injection scaling factors for active and reactive power,
        factors for all injections of model
    selector: 'I'|'P'|'Q'
        addresses current magnitude, active power or reactive power
    vminsqr: float
        square of voltage, upper limit of interpolation interval [0...vminsqr]

    Returns
    -------
    dict
        id_of_batch => values for I/P/Q-calculation"""
    assert selector in 'IPQ', \
        f'selector needs to be one of "I", "P" or "Q" but is "{selector}"'
    get_inj_val = partial(
        _get_injected_value,
        model.mnodeinj.T,
        Vnode_ri2,
        Vabs_sqr,
        kpq,
        selector,
        vminsqr)
    injections = model.injections
    return {
        id_of_batch:get_inj_val(injections.loc[df.index_of_injection])
        for id_of_batch, df in get_batches(
            get_values(model, selector),
            model.injectionoutputs,
            'index_of_injection')}

#
# power / current into branch-terminals
#

def _get_gb_of_terminals_n(branchterminals):
    """Creates a numpy array of branch-susceptances and branch-conductances.

    Parameters
    ----------
    branchterminals: pandas.DataFrame (index_of_terminal)
        * .g_lo, float, longitudinal conductance
        * .b_lo, float, longitudinal susceptance
        * .g_tr_half, float, transversal conductance
        * .b_tr_half, float, transversal susceptance

    Returns
    -------
    numpy.array (shape n,4)
        float
        * [:,0] g_mn, mutual conductance
        * [:,1] b_mn, mutual susceptance
        * [:,2] g_mm, self conductance
        * [:,3] b_mn, self susceptance"""
    return (
        branchterminals
        .loc[:,['g_lo', 'b_lo', 'g_tr_half', 'b_tr_half']]
        .to_numpy())

def _calculate_factors_of_positions_n(branchtaps, positions):
    """Calculates longitudinal factors of branches.

    Parameters
    ----------
    branchtaps: pandas.DataFrame (index of taps)
        * .Vstep, float voltage diff per tap
        * .positionneutral, int
    position: array_like
        int, vector of positions for branch-terminals with taps

    Returns
    -------
    numpy.array (shape n,1)"""
    return (
        (1 - branchtaps.Vstep * (positions - branchtaps.positionneutral))
        .to_numpy()
        .reshape(-1,1))

def _create_gb_of_terminals_n(branchterminals, branchtaps, positions=None):
    """Creates a vectors (as a numpy array) of branch-susceptances and
    branch-conductances.
    The intended use is calculating a subset of terminal values.
    Arguments 'branchtaps' and 'positions' will be selected
    accordingly, hence, it is appropriate to pass the complete branchtaps
    and positions.

    Parameters
    ----------
    branchterminals: pandas.DataFrame (index_of_terminal)
        * .g_lo, float, longitudinal conductance
        * .b_lo, float, longitudinal susceptance
        * .g_tr_half, float, transversal conductance
        * .b_tr_half, float, transversal susceptance
    branchtaps: pandas.DataFrame (index_of_taps)
        * .index_of_term, int
        * .index_of_other_term, int
        * .Vstep, float voltage diff per tap
        * .positionneutral, int
        * .position, int (if argument positions is None)
    position: array_like (optional, accepts None)
        int, vector of positions for branch-terminals with taps

    Returns
    -------
    numpy.array (shape n,4)
        gb_mn_tot[:,0] - g_mn
        gb_mn_tot[:,1] - b_mn
        gb_mn_tot[:,2] - g_tot
        gb_mn_tot[:,3] - b_tot"""
    index_of_branch_terminals = branchterminals.index
    is_taps_at_term = branchtaps.index_of_term.isin(
        index_of_branch_terminals)
    taps_at_term = branchtaps[is_taps_at_term]
    is_term_at_tap = index_of_branch_terminals.isin(
        taps_at_term.index_of_term)
    is_taps_at_other_term = (
        branchtaps.index_of_other_term.isin(index_of_branch_terminals))
    taps_at_other_term = branchtaps[is_taps_at_other_term]
    is_other_term_at_tap = index_of_branch_terminals.isin(
        taps_at_other_term.index_of_other_term)
    positions_ = branchtaps.position if positions is None else positions
    f_at_term = _calculate_factors_of_positions_n(
        taps_at_term, positions_[is_taps_at_term])
    f_at_other_term = _calculate_factors_of_positions_n(
        taps_at_other_term, positions_[is_taps_at_other_term])
    # g_lo, b_lo, g_trans, b_trans
    gb_mn_tot = _get_gb_of_terminals_n(branchterminals)
    # gb_mn_mm -> gb_mn_tot
    gb_mn_tot[:, 2:] += gb_mn_tot[:, :2]
    if f_at_term.size:
        # diagonal and off-diagonal
        gb_mn_tot[is_term_at_tap] *= f_at_term
        # diagonal
        gb_mn_tot[is_term_at_tap, 2:] *= f_at_term
    if f_at_other_term.size:
        # off-diagonal
        gb_mn_tot[is_other_term_at_tap, :2] *= f_at_other_term
    return gb_mn_tot.copy()

def _get_branch_flow_values(
        branchtaps, positions, vnode_ri2, branchterminals):
    """Calculates current, active and reactive power flow into branches from
    given terminals. 'branchterminals' is a subset, all other arguments are
    complete.

    Parameters
    ----------
    branchtaps: pandas.DataFrame
        data of taps
    positions: array_like<int>
        tap positions, accepts None
    vnode_ri2: numpy.array<float>, (shape 2n,1)
        voltages at nodes, real part than imaginary part
    branchterminals: pandas.DataFrame
        data of terminals
        * .index_of_node
        * .index_of_other_node

    Returns
    -------
    numpy.array<float>, (shape m,3)
        * [:,0] Ire, real part of current
        * [:,1] Iim, imaginary part of current
        * [:,2] P, active power
        * [:,3] Q, reactive power"""
    gb_mn_tot = _create_gb_of_terminals_n(
        branchterminals, branchtaps, positions)
    # reverts columns to y_tot, y_mn
    y_mn_tot = gb_mn_tot.view(dtype=np.complex128)[:,np.newaxis,::-1]
    # complex voltage
    voltages_cx = vnode_ri2.view(dtype=np.complex128)
    # voltages per node
    Vcx_node = voltages_cx[branchterminals.index_of_node]
    Vcx_other_node = voltages_cx[branchterminals.index_of_other_node]
    Vcx = np.hstack([Vcx_node, -Vcx_other_node]).reshape(-1,2,1)
    # current into terminal (branch)
    Icx = (y_mn_tot @ Vcx).reshape(-1,1)
    Sterm = Vcx_node * Icx.conj()
    return np.hstack([Icx.view(dtype=float), Sterm.view(dtype=float)])

# slices I/P/Q-vectors from result of function _get_branch_flow_values
_branch_flow_slicer = dict(I=np.s_[:,:2], P=np.s_[:,2], Q=np.s_[:,3])

def _get_batch_values_br(
        model, vnode_ri2, positions, selector, vminsqr=.8**2):
    """Calculates a vector (numpy.array, shape n,1) of injected absolute
    current, active power or reactive power. The expressions are based
    on the batch definitions.

    Parameters
    ----------
    model: egrid.model.Model
        model of electric network for calculation
    vnode_ri2: numpy.array<float> (shape n,2)
        [:,0] Vnode_re, real part of node voltage
        [:,1] Vnode_im, imaginary part of node voltage
    positions: array_like<int>
        tap positions, accepts None
    selector: 'I'|'P'|'Q'
        addresses current magnitude, active power or reactive power
    vminsqr: float
        square of voltage, upper limit of interpolation interval [0...vminsqr]

    Returns
    -------
    dict
        id_of_batch => values for I/P/Q-calculation"""
    slicer = _branch_flow_slicer[selector]
    get_branch_vals = lambda terminals: (
        _get_branch_flow_values(
            model.branchtaps, positions, vnode_ri2, terminals)[slicer])
    branchterminals = model.branchterminals
    return {
        id_of_batch:get_branch_vals(branchterminals.loc[df.index_of_term])
        for id_of_batch, df in get_batches(
            get_values(model, selector),
            model.branchoutputs,
            'index_of_term')}

def _get_batch_flow_values(
        model, Vnode_ri2, Vabs_sqr, kpq, positions, selector, vminsqr):
    """Calculates a float value for each batch id. The returned values
    are a subset of calculated values of a network model.

    Parameters
    ----------
    model: egrid.model.Model
        model of electric network for calculation
    Vnode_ri2: numpy.array<float> (shape n,3)
        [:,0] Vnode_re, real part of node voltage
        [:,1] Vnode_im, imaginary part of node voltage
    Vabs_sqr: numpy.array<float> (shape n,1)
        Vnode_ri2[:,0]**2 + Vnode_ri2[:,1]**2
    kpq: numpy.array (shape n,2)
        vector of injection scaling factors for active and reactive power,
        factors for all injections of model
    positions: array_like<int>
        tap positions, accepts None
    selector: 'I'|'P'|'Q'
        addresses current magnitude, active power or reactive power
    vminsqr: float
        square of voltage, upper limit of interpolation interval [0...vminsqr]
        for interpolation of injected values

    Returns
    -------
    dict
        id_of_batch, str => value for I/P/Q-calculation, float"""
    shape = 0, 2 if selector=='I' else 1
    dd = defaultdict(lambda:np.empty(shape, dtype=float))
    dd.update(
        _get_batch_values_br(
            model, Vnode_ri2, positions, selector, vminsqr))
    inj_vals = _get_batch_values_inj(
        model, Vnode_ri2, Vabs_sqr, kpq, selector, vminsqr)
    for id_of_batch, val in inj_vals.items():
        dd[id_of_batch] = np.vstack([dd[id_of_batch], val])
    if selector in 'PQ':
        return {id_of_batch: np.sum(arr) for id_of_batch, arr in dd.items()}
    if selector == 'I':
        return {id_of_batch:
                np.sqrt(np.sum(np.square(np.sum(Iri_vals, axis=0))))
                for id_of_batch, Iri_vals in dd.items()}
    assert False, \
        f'selector needs to be one of "I", "P" or "Q" but is "{selector}"'

def get_batch_values(
    model, Vnode_ri2, kpq, positions=None, quantities='', vminsqr=_VMINSQR):
    """Provided, node voltages, scaling factors and tappositions are results
    and parameters of a power flow calculation with the grid-data of the model,
    'get_batch_values' returns calculated values for the selected quantities.
    For instance, if quantity is 'P' the function returns the active power
    according to the power flow calculation at the terminals active power
    values (table 'PValues') are given for by the model.

    Parameters
    ----------
    model: egrid.model.Model
        model of electric network for calculation
    Vnode_ri2: numpy.array<float> (shape n,3)
        [:,0] Vnode_re, real part of node voltage
        [:,1] Vnode_im, imaginary part of node voltage
    kpq: numpy.array (shape n,2)
        vector of injection scaling factors for active and reactive power,
        factors for all injections of model
    positions: array_like<int>
        tap positions, accepts None
    quantities: str
        string of characters 'I'|'P'|'Q'|'V'
        addresses current magnitude, active power, reactive power or magnitude
        of voltage, case insensitive, other characters are ignored
    vminsqr: float
        square of voltage, upper limit of interpolation interval [0...vminsqr]
        for interpolation of injected values

    Returns
    -------
    tuple
        * quantity, numpy.array<str>
        * id_of_batch, numpy.array<str>
        * value, numpy.array<float>, vector (shape n,1)"""
    Vabs_sqr = np.sum(Vnode_ri2 * Vnode_ri2, axis=1)
    _quantities = []
    _ids = []
    _vals = []
    for sel in quantities.upper():
        if sel in 'IPQ':
            id_val = _get_batch_flow_values(
                model, Vnode_ri2, Vabs_sqr, kpq, positions, sel, vminsqr)
            _quantities.extend([sel]*len(id_val))
            _ids.extend(id_val.keys())
            _vals.extend(id_val.values())
        if sel=='V':
            vvals = value_of_voltages(model.vvalues)
            count_of_values = len(vvals)
            _quantities.extend([sel]*count_of_values)
            _ids.extend(vvals.id_of_node)
            _vals.extend(np.sqrt(Vabs_sqr[vvals.index]))
    return np.array(_quantities), np.array(_ids), np.array(_vals)
