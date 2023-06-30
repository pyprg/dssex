# -*- coding: utf-8 -*-
"""
Created on Fri May 19 21:25:36 2023

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
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.sparse.linalg import spsolve
from dssex.pfcnum import (
    get_calc_injected_power_fn, get_injected_power_per_injection)

#
# injections
#

def _calculate_injected_si(injections, Vinj, kpq, loadcurve, vminsqr):
    """Calculates complex power and current of injections.

    Complex power is calculated for one phase.

    Parameters
    ----------
    injections : TYPE
        DESCRIPTION.
    Vinj: numpy.array
        complex, voltage per injection
    kpq: numpy.array, float, (nx2)
        scaling factors for active and reactive power
    loadcurve: 'original' | 'interpolated' | 'square'

    vminsqr: float
        upper limit of interpolation, interpolates if |V|² < vminsqr

    Returns
    -------
    numpy.array (shape 2,n)
        complex, [0] - complex power (single phase), [1] - complex current"""
    power_fn = get_calc_injected_power_fn(vminsqr, injections, kpq, loadcurve)
    P_pu, Q_pu, _  = get_injected_power_per_injection(power_fn, Vinj)
    Ssinglephase = (
        np.hstack([P_pu.reshape(-1,1), Q_pu.reshape(-1,1)])
        .view(np.complex128)
        .reshape(-1))
    Icx = (Ssinglephase / Vinj.reshape(-1)).conjugate()
    return np.vstack([Ssinglephase, Icx])

def _calculate_injection_results(injections, Vinj, SI, kpq):
    """Calculates electric data of injections.

    Returns active and reactive power, current and voltage in pu.

    Parameters
    ----------
    injections: pandas.DataFrame
        * .id, str
        * .Exp_v_p, float
        * .Exp_v_q, float
        * .P10, float
        * .Q10, float
    Vinj: numpy.array
        complex, vector of voltages at terminals of injections
    SI: numpy.array (shape 2,n)
        complex
        [0] - complex power (single phase)
        [1] - complex current

    Returns
    -------
    pandas.DataFrame
        id, Exp_v_p, Exp_v_q, P10, Q10, P_pu, Q_pu, V_pu,
        I_pu, Vcx_pu, Scx_pu, Icx_pu"""
    S = 3 * SI[0] # single phase to 3-phase
    PQ = S.reshape(-1,1).view(dtype=np.float64)
    df = injections.loc[:, ['id', 'Exp_v_p', 'Exp_v_q', 'P10', 'Q10']]
    df['P_pu'] = PQ[:, 0]
    df['Q_pu'] = PQ[:, 1]
    df['V_pu'] = np.abs(Vinj)
    df['I_pu'] = np.abs(SI[1])
    df['Vcx_pu'] = Vinj
    df['Scx_pu'] = S
    df['Icx_pu'] = SI[1]
    if kpq is None:
        df['kp'] = 1.
        df['kq'] = 1.
    else:
        df['kp'] = kpq[:,0]
        df['kq'] = kpq[:,1]
    df.set_index('id', inplace=True)
    df.sort_index(inplace=True)
    return df

def calculate_injection_results(
        model, /, Vnode, *, kpq=None, loadcurve='interpolated', vminsqr=.64):
    """Calculates electric data of injections according from node voltage.

    Returns active and reactive power in pu.

    Parameters
    ----------
    model: egrid.model.Model
        data of the electric power network
    Vnode: numpy.array
        complex, vector of node voltages
    kpq: numpy.array, float, (nx2)
        optional
        scaling factors for active and reactive power
    loadcurve: 'original' | 'interpolated' | 'square'
        optional, default is 'interpolated'
    vminsqr: float
        optional, default is 0.64
        upper limit of interpolation, interpolates if |V|² < vminsqr

    Returns
    -------
    pandas.DataFrame
        id, Exp_v_p, Exp_v_q, P10, Q10, P_pu, Q_pu, V_pu,
        I_pu, Vcx_pu, Scx_pu, Icx_pu"""
    Vinj = model.mnodeinj.T @ Vnode
    injections = model.injections
    SI = _calculate_injected_si(injections, Vinj, kpq, loadcurve, vminsqr)
    return _calculate_injection_results(injections, Vinj, SI, kpq)

#
# branches
#

def _calculate_f_tot_mn_all(terminalfactors, positions, branchterminals):
    """Calculates tap factors for all branch terminals.

    Terminals without taps are included. Terminals without taps get factor 1.
    ::
        m*postion + n

    Parameters
    ----------
    terminalfactors: pandas.DataFrame
        * .index_of_terminal, int
        * .index_of_other_terminal
        * .m, float
        * .n, float
    positions: numpy.array
        float, tap positions, one entry for each record in terminalfactors
    branchterminals: pandas.DataFrame (index_of_terminal)
        * .index_of_other_terminal, int
        * .m, float
        * .n, float

    Returns
    -------
    numpy.array (shape n,2)
        factors for values of admittance, two values for each terminal,
        first is factor for terminal, second is factor for other terminal
        (row of selected_terminals)"""
    factor = terminalfactors.m * positions + terminalfactors.n
    f_term_otherterm = _distribute_over_terminals(factor, branchterminals)
    # f_mn = f * f_other, f_tot = f**2
    return f_term_otherterm * f_term_otherterm[:, 0].reshape(-1,1)

def _vterm_from_vnode(branchterminals, Vnode):
    """Reorders Vnode. Returns V_term_a, V_term_b for each branchterminal.

    Parameters
    ----------
    branchterminals: pandas.DataFrame
        * .index_of_node, int
        * .index_of_other_node, int
    Vnode: numpy.array (shape m,1)
        complex, node voltage (at power-flow-calculation node)

    Returns
    -------
    numpy.array (shape n,2), complex"""
    vcx_term_a = Vnode[branchterminals.index_of_node]
    vcx_term_b = Vnode[branchterminals.index_of_other_node]
    return np.concatenate([vcx_term_a, vcx_term_b], axis=1)

def _calc_term_current(f_tot_mn, branchterminals, vcx_term_ab):
    """Calculates current flowing into terminals of branches.

    Parameters
    ----------
    f_tot_mn: numpy.array (shape n,2)
        float, terminal factors for considering taps
        [:, 0] f_tot
        [:, 1] f_mn
    branchterminals : pandas.DataFrame
        * .y_tr_half, half of complex admittance of branch, transversal
        * .y_lo, complex admittance of branch, longitudinal
        * .index_of_node, int
        * .index_of_other_node,int
    vcx_term_ab: numpy.array (shape n,2)
        complex, Vcx_term_a, Vcx_term_b

    Returns
    -------
    numpy.array (shape n,1), complex
        current into terminal"""
    y_tot_mn = branchterminals[['y_tr_half', 'y_lo']].to_numpy()
    y_tot_mn[:,0] += y_tot_mn[:,1]
    y_tot_mn[:,1] *= -1.
    y_tot_mn = (y_tot_mn * f_tot_mn).reshape(-1,1,2)
    return (y_tot_mn @ vcx_term_ab.reshape(-1,2,1)).reshape(-1,1)

def _calculate_terminal_current(model, Vcx_term_ab, positions):
    """Calculates current flowing into terminals of branches.

    Parameters
    ----------
    model: egrid.model.Model
        data of the electric power network
    Vcx_term_ab: numpy.array (shape n,2)
        complex, vector of terminal voltages for terminals A and B
    positions: numpy.array
        float, tap positions, one entry for each record in
        model.factors.terminalfactors

    Returns
    -------
    numpy.array (shape n,1), complex
        current into terminal"""
    branchterminals = model.branchterminals
    terminalfactors = model.factors.terminalfactors
    f_tot_mn = _calculate_f_tot_mn_all(
        terminalfactors, positions, branchterminals)
    return _calc_term_current(f_tot_mn, branchterminals, Vcx_term_ab)

def _calculate_branch_results(
        branchterminals, terminalfactors, Iterm, Vcx_term_ab, positions):
    """Calculates electric values for branches.

    Parameters
    ----------
    model: egrid.model.Model
        data of the electric power network
    Vcx_term_ab: numpy.array (shape n,2)
        complex, vector of terminal voltages for terminals A and B
    positions: numpy.array
        float, tap positions, one entry for each record in
        model.factors.terminalfactors

    Returns
    -------
    pandas.DataFrame (id)
        * .S0cx_pu, complex
        * .I0cx_pu, complex
        * .V0cx_pu, complex
        * .S1cx_pu, complex
        * .I1cx_pu, complex
        * .V1cx_pu, complex
        * .Slosscx_pu, complex
        * .S0_pu, float
        * .P0_pu, float
        * .Q0_pu, float
        * .I0_pu, float
        * .V0_pu, float
        * .S1_pu, float
        * .P1_pu, float
        * .Q1_pu, float
        * .I1_pu, float
        * .V1_pu, float
        * .Ploss_pu, float
        * .Qloss_pu, float
        * .Tap0, float
        * .Tap1, float"""
    Vcx_a = Vcx_term_ab[:, 0].reshape(-1,1)
    Sterm = 3 * Vcx_a * np.conjugate(Iterm) # factor 3 for 3-phase system
    terms_a = branchterminals[branchterminals.side_a]
    siv = np.concatenate([Sterm, Iterm, Vcx_a], axis=1)
    siv_ab = (
        np.concatenate([
            siv[terms_a.index].reshape(-1,3),
            siv[terms_a.index_of_other_terminal].reshape(-1,3)],
            axis=1)
        .reshape(-1,6))
    df_siv = pd.DataFrame(
        siv_ab,
        columns=['S0cx_pu','I0cx_pu','V0cx_pu', 'S1cx_pu','I1cx_pu','V1cx_pu'],
        index=terms_a.index)
    df_siv['id_of_branch'] = terms_a.id_of_branch
    df_siv['index_of_other_terminal'] = terms_a.index_of_other_terminal
    df_siv['Slosscx_pu'] = df_siv.S0cx_pu + df_siv.S1cx_pu
    PQ_pu = Sterm.view(dtype=np.float64)
    I_pu = np.abs(Iterm)
    V_pu = np.abs(Vcx_term_ab[:,0].reshape(-1, 1))
    pqiv = np.concatenate([PQ_pu, I_pu, V_pu], axis=1)
    pqiv_ab = (
        np.concatenate([
            pqiv[terms_a.index].reshape(-1,4),
            pqiv[terms_a.index_of_other_terminal].reshape(-1,4)],
            axis=1)
        .reshape(-1,8))
    df_pqiv = pd.DataFrame(
        pqiv_ab,
        columns=[
            'P0_pu','Q0_pu','I0_pu', 'V0_pu',
            'P1_pu','Q1_pu','I1_pu', 'V1_pu'],
        index=terms_a.index)
    df_pqiv['S0_pu'] = np.abs(df_siv.S0cx_pu)
    df_pqiv['S1_pu'] = np.abs(df_siv.S1cx_pu)
    df_pqiv['Ploss_pu'] = np.real(df_siv.Slosscx_pu)
    df_pqiv['Qloss_pu'] = np.imag(df_siv.Slosscx_pu)
    tappos = pd.Series(positions, index=terminalfactors.index, name="Tap")
    df_siv_ = (
        df_siv
        .join(tappos)
        .join(
            tappos,
            on='index_of_other_terminal',
            lsuffix='0',
            rsuffix='1')
        .drop(columns='index_of_other_terminal'))
    branch_res = pd.concat([df_siv_, df_pqiv], axis=1)
    branch_res.set_index('id_of_branch', inplace=True)
    branch_res.index.name = 'id'
    return branch_res

def calculate_branch_results(model, /, Vnode, *, positions=None):
    """Calculates electric values for branches.

    Parameters
    ----------
    model: egrid.model.Model
        data of the electric power network
    Vnode: numpy.array (shape m,1)
        complex, node voltage (at power-flow-calculation node)
    positions: numpy.array
        optional, default is None
        float, tap positions, one entry for each record in
        model.factors.terminalfactors

    Returns
    -------
    pandas.DataFrame (id)
        * .S0cx_pu, complex
        * .I0cx_pu, complex
        * .V0cx_pu, complex
        * .S1cx_pu, complex
        * .I1cx_pu, complex
        * .V1cx_pu, complex
        * .Slosscx_pu, complex
        * .S0_pu, float
        * .P0_pu, float
        * .Q0_pu, float
        * .I0_pu, float
        * .V0_pu, float
        * .S1_pu, float
        * .P1_pu, float
        * .Q1_pu, float
        * .I1_pu, float
        * .V1_pu, float
        * .Ploss_pu, float
        * .Qloss_pu, float
        * .Tap0, float
        * .Tap1, float"""
    branchterminals = model.branchterminals
    terminalfactors = model.factors.terminalfactors
    Vcx_term_ab = _vterm_from_vnode(branchterminals, Vnode)
    positions_ = (
            (positions
             if positions else
             terminalfactors.value.to_numpy())
            .reshape(-1))
    Iterm = _calculate_terminal_current(model, Vcx_term_ab, positions_)
    return _calculate_branch_results(
        branchterminals, terminalfactors, Iterm, Vcx_term_ab, positions_)

#
# switch flow
#

def _distribute_over_terminals(val, terminals):
    # select and order according to selected_terminals
    val_term = (
        val.reindex(terminals.index)
        .fillna(1.) # multiply with 1. if no data given
        .to_numpy()
        .reshape(-1,1))
    # select and order according to other selected_terminals
    val_other_term = (
        val.reindex(terminals.index_of_other_terminal)
        .fillna(1.) # multiply with 1. if no data given
        .to_numpy()
        .reshape(-1,1))
    return np.concatenate([val_term, val_other_term], axis=1)

def _get_by_key(df, key):
    """Selects rows according to key from level 0 of a MultiIndex.

    Parameters
    ----------
    df: pandas.DataFrame MultiIndex

    key: type of df.index level 0

    Returns
    -------
    pandas.DataFrame"""
    try:
        return df.loc[key]
    except KeyError:
        newdf = pd.DataFrame([], columns=df.columns)
        newdf.index.names = df.index.names[1:]
        return newdf

def _get_slack_for_switchflow(terminals):
    """Selects one node of a switch flow node group to be the slack.

    Parameters
    ----------
    terminals: pandas.DataFrame (index_of_terminal)
        group of terminals sharing one power-flow-calculation node, which
        means they are connected via short circuits
        * .id_of_node
        * .switch_flow_index
        * .at_slack

    Raises
    ------
    ValueError
        if more than one connectivity node is a slack for power flow
        calculation

    Returns
    -------
    int
        index of slack"""
    idxs_of_slacks = (
        terminals[['switch_flow_index','at_slack']][terminals.at_slack]
        .switch_flow_index
        .unique())
    number_of_slacks = idxs_of_slacks.size
    if 1 < number_of_slacks:
        node_ids = ', '.join(set(
            terminals[['id_of_node','at_slack']][terminals.at_slack]
            .id_of_node))
        raise ValueError(
            'Calculation of flow through bridges not implemented if '
            'multiple slack nodes are connected, following slack nodes are '
            'connected by branches which are treated like short circuits: '
            f'{node_ids}')
    return idxs_of_slacks[0] if number_of_slacks==1 else 0

def _make_switch_flow_matrix(switch_flow_index, other):
    """Creates the 'Laplacian matrix' for switch flow calculation.

    Parameters
    ----------
    switch_flow_index: array_like
        int, index of the terminal-node for switch flow calculation
    other: array_like
        int, index of other terminal node (of same branch) for switch flow
        calculation

    Returns
    -------
    scipy.sparse.coo_matrix, complex"""
    dim = switch_flow_index.max() + 1
    vals = np.full((len(switch_flow_index),), -1., dtype=complex)
    vals_diag = np.ones_like(vals)
    return coo_matrix(
        (np.concatenate([vals_diag, vals]),
         (np.concatenate([switch_flow_index, switch_flow_index]),
          np.concatenate([switch_flow_index, other]))),
         shape=(dim, dim),
         dtype=np.complex128)

def _modify_slack(Y, slack_idx):
    """Modifies the row of the slack in matrix Y.

    Parameters
    ----------
    Y: scipy.sparse.csr_matrix
        Laplacien matrix
    slack_idx: int
        index of slack

    Returns
    -------
    scipy.sparse.csr_matrix"""
    headrows = Y[:slack_idx]
    slackrow = csr_matrix(
        ([1.], ([0], [slack_idx])), shape=(1, Y.shape[1]), dtype=float)
    tailrows = Y[slack_idx+1:]
    return vstack([headrows, slackrow, tailrows])

def _calculate_switch_flow(Iterminj, group_of_terminals):
    """Calculates electrical current flowing through terminals of short
    circuits.

    Parameters
    ----------
    Iterminj: pandas.DataFrame (switch_flow_index)
        switch_flow_index, int, index of connectivity node for
            switch-flow-calculation)
        * .Icx, complex, current flowing out of the
          switch-flow-calculation node
    group_of_terminals: pandas.DataFrame (index_of_terminal)
        * .index_of_node, int
        * .switch_flow_index, int
        * .index_of_other_terminal, int
        * .at_slack, bool

    Returns
    -------
    pandas.Series (index_of_terminal)
        Icx, complex current flowing into the terminal"""
    switch_flow_index = group_of_terminals.switch_flow_index
    other = (
        group_of_terminals.reindex(
            group_of_terminals.index_of_other_terminal).switch_flow_index)
    Y = _make_switch_flow_matrix(switch_flow_index, other)
    slack_idx = _get_slack_for_switchflow(group_of_terminals)
    A = _modify_slack(Y.tocsr(), slack_idx).tocsc()
    # create B
    B = np.zeros((A.shape[0],), dtype=np.complex128)
    # index is switch_flow_index,
    #   automatically sorts B according to switch_flow_index
    B[Iterminj.index] = Iterminj.Icx
    B[slack_idx] = 1.
    # solve linear equations
    x = spsolve(A, B.reshape(-1,1))
    return pd.Series(
        x[switch_flow_index] - x[other],
        index=group_of_terminals.index,
        name='Iterm')

def _calculate_switch_flows(Iterm_df, Iinj_df, groups_of_terminals):
    """Calculates current flowing into terminals of short circuit branches.

    Parameters
    ----------
    Iterm_df: pandas.DataFrame
        current flowing into branch terminals
        * .index_of_node, int, index of power flow calculation node
        * .switch_flow_index, ind, index of switch flow calculation node
        * .Icx, complex, current flowing out of the
          switch-flow-calculation node
    Iinj_df: pandas.DataFrame
        current flowing into injections
        * .index_of_node, int, index of power flow calculation node
        * .switch_flow_index, ind, index of switch flow calculation node
        * .Icx, complex, current flowing out of the
          switch-flow-calculation node
    groups_of_terminals: pandas.groupby
        terminals of short circuit branches which to calculate current flow for
        grouped by 'index_of_node' (index of power flow calculation node)
            * .index_of_node
            * .switch_flow_index
            * .index_of_other_terminal
            * .at_slack

    Returns
    -------
    iterator
        pandas.Series (index_of_terminal),
        Icx, complex current flowing into the terminal"""
    # I of switch_flow nodes
    Inode = (
        pd.concat([Iterm_df, Iinj_df])
        .groupby(['index_of_node', 'switch_flow_index'])
        .sum())
    return (
        _calculate_switch_flow(
            _get_by_key(Inode, index_of_pfc_node), group_of_terminals)
        for index_of_pfc_node, group_of_terminals in groups_of_terminals)

def get_switch_flow(bridgeterminals, Icx_branchterm, Icx_injection):
    """Calculates electrical current flow through short circuit branches.

    Use function 'get_switch_flow2' if current through branch terminals and
    into injections is not yet calculated.

    Parameters
    ----------
    bridgeterminals: pandas.DataFrame (index_of_terminal)
        * .index_of_node, int, index of power flow calculation node
        * .switch_flow_index, int, index of switch flow calculation node
        * .index_of_other_terminal, index of other terminal at same branch
        * .at_slack, bool, indicates if power flow calculation node is
          a slack node
    Icx_branchterm: pandas.DataFrame
        * .index_of_node
        * .switch_flow_index
        * .Icx, complex, current flowing into branch terminal
        current for all branchterminals in model
    Icx_injection: pandas.DataFrame
        * .index_of_node
        * .switch_flow_index
        * .in_super_node
        * .Icx, complex, current flowing into injection
        current for all injections in model

    Returns
    -------
    pandas.Series (index_of_terminal)
        Icx, complex current flowing into the terminal"""
    groups_of_terminals = (
        bridgeterminals[
            ['index_of_node', 'switch_flow_index',
             'index_of_other_terminal', 'at_slack']]
        .groupby('index_of_node'))
    at_pfc_node_border = Icx_branchterm.index_of_node.isin(
        groups_of_terminals.groups.keys())
    Icx_at_border = Icx_branchterm[at_pfc_node_border]
    Iterm_df = pd.DataFrame(
        {'index_of_node': Icx_at_border.index_of_node,
         'switch_flow_index': Icx_at_border.switch_flow_index,
         'Icx': Icx_at_border.Icx})
    Icx_in_super_nodes = Icx_injection.loc[Icx_injection.in_super_node]
    Iinj_df = pd.DataFrame(
        {'index_of_node': Icx_in_super_nodes.index_of_node,
         'switch_flow_index': Icx_in_super_nodes.switch_flow_index,
         'Icx': Icx_in_super_nodes.Icx})
    return _calculate_switch_flows(Iterm_df, Iinj_df, groups_of_terminals)

def _calculate_bridge_results(
        bridgeterminals, Icx_branchterm, Icx_injection, Vnode):
    """Calculates electric values for branches.

    Parameters
    ----------
    bridgeterminals: pandas.DataFrame (index_of_terminal)
        * .index_of_node, int, index of power flow calculation node
        * .switch_flow_index, int, index of switch flow calculation node
        * .index_of_other_terminal, index of other terminal at same branch
        * .at_slack, bool, indicates if power flow calculation node is
          a slack node
    Icx_branchterm: pandas.DataFrame
        * .index_of_node
        * .switch_flow_index
        * .Icx, complex, current flowing into branch terminal
        current for all branchterminals in model
    Icx_injection: pandas.DataFrame
        * .index_of_node
        * .switch_flow_index
        * .in_super_node
        * .Icx, complex, current flowing into injection
        current for all injections in model
    Vnode: numpy.array (shape m,1)
        complex, node voltage (at power-flow-calculation node)

    Returns
    -------
    pandas.DataFrame (id)
        * .S0cx_pu, complex
        * .I0cx_pu, complex
        * .V0cx_pu, complex
        * .S1cx_pu, complex
        * .I1cx_pu, complex
        * .V1cx_pu, complex
        * .Slosscx_pu, complex
        * .S0_pu, float
        * .P0_pu, float
        * .Q0_pu, float
        * .I0_pu, float
        * .V0_pu, float
        * .S1_pu, float
        * .P1_pu, float
        * .Q1_pu, float
        * .I1_pu, float
        * .V1_pu, float
        * .Ploss_pu, float
        * .Qloss_pu, float"""
    switch_flow = list(
        get_switch_flow(bridgeterminals, Icx_branchterm, Icx_injection))
    Icx_bridgeterminal = (
        pd.concat(switch_flow) if switch_flow else pd.Series([], name='Iterm'))
    bridgeterminals_a = bridgeterminals[bridgeterminals.side_a]
    bridgeterminal_res = bridgeterminals_a.loc[:, ['id_of_branch']]
    bridgeterminal_res['I0cx_pu'] = Icx_bridgeterminal
    bridgeterminal_res['I1cx_pu'] = (
        Icx_bridgeterminal[bridgeterminals_a.index_of_other_terminal].array)
    bridgeterminal_res['V0cx_pu'] = (
        Vnode[bridgeterminals_a.index_of_node])
    bridgeterminal_res['V1cx_pu'] = bridgeterminal_res.V0cx_pu
    bridgeterminal_res['S0cx_pu'] = (
        bridgeterminal_res.V0cx_pu * np.conjugate(bridgeterminal_res.I0cx_pu))
    bridgeterminal_res['S1cx_pu'] = (
        bridgeterminal_res.V1cx_pu * np.conjugate(bridgeterminal_res.I1cx_pu))
    bridgeterminal_res['I0_pu'] = np.abs(bridgeterminal_res.I0cx_pu)
    bridgeterminal_res['I1_pu'] = bridgeterminal_res.I0_pu
    bridgeterminal_res['V0_pu'] = np.abs(bridgeterminal_res.V0cx_pu)
    bridgeterminal_res['V1_pu'] = bridgeterminal_res.V0_pu
    bridgeterminal_res['S0_pu'] = np.abs(bridgeterminal_res.S0cx_pu)
    bridgeterminal_res['P0_pu'] = np.real(bridgeterminal_res.S0cx_pu)
    bridgeterminal_res['Q0_pu'] = np.imag(bridgeterminal_res.S0cx_pu)
    bridgeterminal_res['S1_pu'] = np.abs(bridgeterminal_res.S1cx_pu)
    bridgeterminal_res['P1_pu'] = np.real(bridgeterminal_res.S1cx_pu)
    bridgeterminal_res['Q1_pu'] = np.imag(bridgeterminal_res.S1cx_pu)
    bridgeterminal_res['Slosscx_pu'] = 0+0j
    bridgeterminal_res['Ploss_pu'] = 0
    bridgeterminal_res['Qloss_pu'] = 0
    bridgeterminal_res.set_index('id_of_branch', inplace=True)
    bridgeterminal_res.index.name = 'id'
    return bridgeterminal_res

def calculate_electric_data(
        model, /, Vnode, *, positions=None,
        kpq=None, loadcurve='interpolated', vminsqr=.64):
    """Calculates and arranges electric data of injections, nodes and branches.

    Parameters
    ----------
    model: egrid.model.Model
        data of the electric power network
    Vnode: numpy.array
        complex, vector of node voltages
    positions: numpy.array
        optional
        float, tap positions, one entry for each record in
        model.factors.terminalfactors
    kpq: numpy.array, float, (nx2)
        optional
        scaling factors for active and reactive power
    loadcurve: 'original' | 'interpolated' | 'square'
        optional, default is 'interpolated'
    vminsqr: float
        optional, default is 0.64
        upper limit of interpolation, interpolates if |V|² < vminsqr

    Returns
    -------
    dict
        * branches
        * injections
        * nodes"""
    nodes = model.nodes
    Vcx_cn_node = Vnode[nodes.index_of_node].reshape(-1)
    nodes_res = pd.DataFrame(
        {'Vcx_pu': Vcx_cn_node,
         'V_pu': np.abs(Vcx_cn_node),
         'Vre': np.real(Vcx_cn_node),
         'Vim': np.imag(Vcx_cn_node)},
        index=nodes.index)
    nodes_res.sort_index(inplace=True)
    nodes_res.index.name = 'id'
    # branches
    branchterminals = model.branchterminals
    terminalfactors = model.factors.terminalfactors
    Vcx_term_ab = _vterm_from_vnode(branchterminals, Vnode)
    positions_ = (
        (positions
         if positions else
         terminalfactors.value.to_numpy())
        .reshape(-1))
    Iterm = _calculate_terminal_current(model, Vcx_term_ab, positions_)
    branch_res = _calculate_branch_results(
        branchterminals, terminalfactors, Iterm, Vcx_term_ab, positions_)
    # injections
    injections = model.injections
    Vinj = model.mnodeinj.T @ Vnode
    SI = _calculate_injected_si(injections, Vinj, kpq, loadcurve, vminsqr)
    injection_res = _calculate_injection_results(injections, Vinj, SI, kpq)
    # switch flow
    Icx_branchterm = (
        branchterminals.loc[:, ['index_of_node', 'switch_flow_index']])
    Icx_branchterm['Icx'] = Iterm
    Icx_injection = injections.loc[
        :, ['index_of_node', 'switch_flow_index', 'in_super_node']]
    bridgeterminals = model.bridgeterminals
    Icx_injection['Icx'] = SI[1]
    bridgeterminal_res = _calculate_bridge_results(
        bridgeterminals, Icx_branchterm, Icx_injection, Vnode)
    branches = pd.concat([branch_res, bridgeterminal_res])
    branches.sort_index(inplace=True)
    return dict(
        branches=branches,
        injections=injection_res,
        nodes=nodes_res)

def calculate_electric_data2(model, result):
    """Calculates and arranges electric data of injections, nodes and branches.

    Parameters
    ----------
    model: egrid.model.Model
        data of the electric power network
    result : tuple
        * int, index of estimation step,
          (initial power flow calculation result is -1, first estimation is 0)
        * bool, success?
        * numpy.array, complex (shape n,1)
            calculated complex node voltages
        * numpy.array, float (shape m,2)
            scaling factors for injections
        * numpy.array, tappositions

    Returns
    -------
    dict
        * branches
        * injections
        * nodes"""
    return calculate_electric_data(
        model, result[2], kpq=result[3], positions=result[4])

def _calculate_f_tot_mn(terminalfactors, positions, selected_terminals):
    """Calculates tap factors for a subset of branch terminals.

    May include terminals without taps. Terminals without taps get factor 1.
    ::
        m*postion + n

    Parameters
    ----------
    terminalfactors: pandas.DataFrame
        * .index_of_terminal, int
        * .index_of_other_terminal
        * .m, float
        * .n, float
    positions: numpy.array
        float, tap positions, one entry for each record in terminalfactors
    selected_terminals: pandas.DataFrame (index_of_terminal)
        * .index_of_other_terminal, int

    Returns
    -------
    numpy.array (shape n,2)
        factors for values of admittance matrix 2 for each terminal
        (row of selected_terminals)"""
    # filter required factors
    is_required = (
        terminalfactors.index_of_terminal.isin(selected_terminals.index)
        | terminalfactors.index_of_other_terminal.isin(
            selected_terminals.index))
    f_req = (
        terminalfactors.loc[is_required, ['index_of_terminal', 'm', 'n']]
        .set_index('index_of_terminal'))
    # calculate value of terminal factor: m*x + n
    val = f_req.m * positions[f_req.index] + f_req.n
    f_term_otherterm = _distribute_over_terminals(val, selected_terminals)
    # f_mn = f * f_other, f_tot = f**2
    return f_term_otherterm * f_term_otherterm[:, 0].reshape(-1,1)

def calc_inj_current(injections, kpq, loadcurve, vcx, vminsqr):
    """Calculates injected current per injection.

    Parameters
    ----------
    injections: pandas.DataFrame (index_of_terminal)
        * .P10
        * .Q10
        * .Exp_v_p
        * .Exp_v_q
    kpq: numpy.array, float, (nx2)
        scaling factors for active and reactive power
    loadcurve: 'original' | 'interpolated' | 'square'

    vcx: numpy.array (shape n,1)
        complex, complex node voltages

    Returns
    -------
    numpy.array (shape n,2)
        complex"""
    calculate_injected_power = get_calc_injected_power_fn(
        vminsqr,
        injections,
        kpq=kpq[injections.index],
        loadcurve=loadcurve)
    vcx_inj = vcx[injections.index_of_node]
    P_pu, Q_pu, _ = get_injected_power_per_injection(
        calculate_injected_power,
        vcx_inj)
    Sinj = (
        np.concatenate((P_pu.reshape(-1,1), Q_pu.reshape(-1,1)), axis=1)
        .view(dtype=np.complex128))
    return np.conjugate(Sinj/vcx_inj)

def get_switch_flow2(model, /, Vnode, *, kpq, positions, vminsqr=.64):
    """Calculates electrical current flow through model.bridgeterminals.

    This function can be used without calculation of terminal and injected
    current beforehand. It calculates all needed currents at the nodes
    internally. Prefer function 'get_switch_flow' if current through
    branch-terminals and into injections is already calculated.

    Parameters
    ----------
    model: egrid.model.Model
        data of an electric distribution network for calculation
    Vnode: numpy.array (shape n,1)
        complex, node voltages
    kpq: numpy.array (shape l,2)
        scaling factors for active and reactive power for all model.injections
    positions: numpy.array (shape m,)
        float, tap positions, one entry for each record in
        model.terminalfactors

    Returns
    -------
    pandas.Series (index_of_terminal)
        Icx, complex current flowing into the terminal"""
    bridgeterminals = model.bridgeterminals
    branchterminals = model.branchterminals
    groups_of_terminals = (
        bridgeterminals[
            ['index_of_node', 'switch_flow_index',
             'index_of_other_terminal', 'at_slack']]
        .groupby('index_of_node'))
    at_pfc_node_border = branchterminals.index_of_node.isin(
        groups_of_terminals.groups.keys())
    branchterminals_at_border = branchterminals[at_pfc_node_border]
    f_tot_mn = _calculate_f_tot_mn(
        model.factors.terminalfactors,
        positions,
        branchterminals_at_border)
    vcx_term_ab = _vterm_from_vnode(branchterminals_at_border, Vnode)
    Iterm = _calc_term_current(
        f_tot_mn, branchterminals_at_border, vcx_term_ab)
    Iterm_df = pd.DataFrame(
        {'index_of_node': branchterminals_at_border.index_of_node,
         'switch_flow_index': branchterminals_at_border.switch_flow_index,
         'Icx': Iterm.reshape(-1)})
    injections = model.injections.loc[model.injections.in_super_node]
    Iinj = calc_inj_current(injections, kpq, 'interpolated', Vnode, vminsqr)
    Iinj_df = pd.DataFrame(
        {'index_of_node': injections.index_of_node,
         'switch_flow_index': injections.switch_flow_index,
         'Icx': Iinj.reshape(-1)})
    return _calculate_switch_flows(Iterm_df, Iinj_df, groups_of_terminals)

def filter_columns(df, *, regex=r'cx', positive=False):
    """Filters columns of df.

    Parameters
    ----------
    df : pandas.DataFrame

    regex: string|re
        optional, default 'cx'
        expression for filter
    positive: bool
        optional, default False

    Returns
    -------
    pandas.DataFrame"""
    import re
    return (
        df[[name for name in df.columns if re.search(regex, name)]]
        if positive else
        df[[name for name in df.columns if re.search(regex, name) is None]])

def make_printable(dict_of_frames):
    """Removes columns containing complex values, fills nan with '-'.

    Parameters
    ----------
    dict_of_frames : dictionary
        pandas.DataFrame

    Returns
    -------
    dict
        pandas.DataFrame"""
    return {
        k:filter_columns(df).sort_index(axis=1).fillna('-').round(3)
        for k,df in dict_of_frames.items()}

def make_printables(model, results):
    """Calculates and arranges electric data for each optimization step.

    Parameters
    ----------
    model: egrid.model.Model
        data of the electric power network
    result: iterator
        * tuple
            * int, index of estimation step,
              (initial power flow calculation result is -1,
               first estimation is 0)
            * bool, success?
            * numpy.array, complex (shape n,1)
                calculated complex node voltages
            * numpy.array, float (shape m,2)
                scaling factors for injections
            * numpy.array, tappositions

    Returns
    -------
    iterator
        dict
            pandas.DataFrame"""
    return (
        make_printable(calculate_electric_data2(model, res))
        for res in results)


