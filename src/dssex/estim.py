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

Created on Sun Aug  8 08:36:10 2021

@author: pyprg
"""
import casadi
import numpy as np
import pandas as pd
from functools import partial, singledispatch
from operator import itemgetter
from collections import namedtuple
from egrid.builder import DEFAULT_FACTOR_ID, defk, Loadfactor
from egrid.model import get_pfc_nodes
from injections import add_interpol_coeff_to_injections
# helper

# square of voltage magnitude, minimum value for load curve, 
#   if value is below _VMINSQR the load curves for P and Q converge
#   towards a linear load curve which is 0 when V=0; P(V=0)=0, Q(V=0)=0
_VMINSQR = 0.8
# value of zero check, used for load curve calculation    
_EPSILON = 1e-12
_EMPTY_TUPLE = ()

def _create_symbols(prefix, names):
    """Creates symbols."""
    return names.map(lambda x:casadi.SX.sym(prefix + x))

_get_xp = itemgetter('x', 'p')

Term = namedtuple('Term', 'a b factor type', defaults=('', '', 1.0, 'k'))
Term.__doc__ = """Relation for use in objective function. One
Relation adds the term '(a-b)**2' to the objective function.

Parameters
----------
a: str
    id of first element
b: str
    id of second element
type: 'k'
    k - scaling factor relation"""

#
# expressions
#

def _get_branch_tap_factors(branchtaps):
    """Arranges data of taps for branches.

    Parameters
    ----------
    branchtaps: pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        * .Vstep, float
        * .positionmin, int
        * .positionneutral, int
        * .positionmax, int
        * .pos, casadi.SX, symbol of position
        * .flo, casadi.SX, factor for longitudinal admittance
        * .ftr, casadi.SX, factor for transversal admittance"""
    branchtaps['flo'] = (
        1 - branchtaps.Vstep * (branchtaps.pos - branchtaps.positionneutral))
    branchtaps['ftr'] = np.power(branchtaps.flo, 2)
    return (
        branchtaps
        .reindex([
            'Vstep', 'positionmin', 'positionneutral', 'positionmax',
            'pos', 'flo', 'ftr'],
            axis=1))

def _calculate_current_into_branch(
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

def _calculate_branch_terminal_current(branch_terminals):
    """Calculates real and imaginary current flowing into one branch.

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
    branch_terminals: pandas.DataFrame
        with columns
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
    return _calculate_current_into_branch(
        branch_terminals.g_tot, branch_terminals.b_tot,
        branch_terminals.g_mn, branch_terminals.b_mn,
        branch_terminals.Vre, branch_terminals.Vim,
        branch_terminals.Vre_other, branch_terminals.Vim_other)

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
           +-                                                               -+
           |  (Vre Vre_other + Vim Vim_other) (Vre Vim_other - Vim Vre_other)|
         = |                                                                 |
           | (-Vre Vim_other + Vim Vre_other) (Vre Vre_other + Vim Vim_other)|
           +-                                                               -+
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
    					   +-                                   -+
    					   |  (g_mn A + b_mn B) (b_mn A - g_mn B)|
    					 = |                                     |
    					   | (-b_mn A + g_mn B) (g_mn A + b_mn B)|
    					   +-                                   -+
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
    P =  g_tot * V_abs_sqr - (g_mn * A + b_mn * B)
    Q = -b_tot * V_abs_sqr + (b_mn * A - g_mn * B)
    return P, Q

_power_props = itemgetter(
    'kp', 'kq', 'P10', 'Q10', 'V_abs_sqr', 'Exp_v_p', 'Exp_v_q',
    'c3p', 'c2p', 'c1p', 'c3q', 'c2q', 'c1q')

def _injected_power(injections, vminsqr):
    """Calculates power flowing through injections.
    Injected power is calculated this way
    (P = |V|**Exvp * P10, Q = |V|**Exvq * Q10; with |V| - magnitude of V):
    ::
        +- -+   +-                                           -+
        | P |   | (V_r ** 2 + V_i ** 2) ** (Expvp / 2) * P_10 |
        |   | = |                                             |
        | Q |   | (V_r ** 2 + V_i ** 2) ** (Expvq / 2) * Q_10 |
        +- -+   +-                                           -+

    Parameters
    ----------
    injections: pandas.DataFrame
        * .kp
        * .P10
        * .kq
        * .Q10
        * .Exp_v_p
        * .Exp_v_q
        * .V_abs_sqr
        * .c3p, .c2p, .c1p, polynomial coefficients for active power P
        * .c3q, .c2q, .c1q, polynomial coefficients for reactive power Q 
    vminsqr: float
        upper limit of interpolation, interpolates if |V|² < vminsqr

    Returns
    -------
    tuple
        * active power P
        * reactive power Q"""
    if injections.size:
        (kp, kq, P10, Q10, V_abs_sqr, Exp_v_p, Exp_v_q,
         c3p, c2p, c1p, c3q, c2q, c1q)= map(
            casadi.vcat, _power_props(injections))
        # interpolated
        V_abs = casadi.power(V_abs_sqr, .5)
        V_abs_cub = V_abs_sqr * V_abs
        fp_interpol = c3p*V_abs_cub + c2p*V_abs_sqr + c1p*V_abs
        fq_interpol = c3q*V_abs_cub + c2q*V_abs_sqr + c1q*V_abs
        # original
        fp_original = casadi.power(V_abs_sqr, Exp_v_p/2) 
        fq_original = casadi.power(V_abs_sqr, Exp_v_q/2)
        return (
            casadi.if_else(
                V_abs_sqr < vminsqr, fp_interpol, fp_original) * kp * P10, 
            casadi.if_else(
                V_abs_sqr < vminsqr, fq_interpol, fq_original) * kq * Q10)
    return casadi.DM(0.), casadi.DM(0.)

_current_props = itemgetter(
    'kp', 'kq', 'P10', 'Q10', 'Vre', 'Vim', 'V_abs_sqr', 'Exp_v_p', 'Exp_v_q',
    'c3p', 'c2p', 'c1p', 'c3q', 'c2q', 'c1q')

def _injected_current(injections, vminsqr):
    """Calculates current flowing into an injection. Returns separate
    real and imaginary parts.
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
    injections: pandas.DataFrame
        * .V_abs_sqr, float, Vre**2 + Vim**2
        * .Vre, float, node voltage, real part
        * .Vim:, float, node voltage, imaginary part
        * .Exp_v_p, float, voltage exponent, active power
        * .Exp_v_q, float, voltage exponent, active power
        * .P10, float, active power at voltage 1 per unit
        * .Q10, float, reactive power at voltage 1 per unit
        * .kp, float, scaling factor for active power
        * .kq, float, scaling factor for reactive power
        * .c3p, .c2p, .c1p 
        * .c3q, .c2q, .c1q 
    vminsqr: float
        upper limit of interpolation, interpolates if |V|² < vminsqr

    Returns
    -------
    tuple
        * Ire, real part of injected current
        * Iim, imaginary part of injected current"""
    if injections.size:
        (kp, kq, P10, Q10, Vre, Vim, V_abs_sqr, Exp_v_p, Exp_v_q, 
         c3p, c2p, c1p, c3q, c2q, c1q) = map(
             casadi.vcat, _current_props(injections))
        # interpolated function
        V_abs = casadi.power(V_abs_sqr, .5)
        V_abs_cub = V_abs_sqr * V_abs
        p_expr = (c3p*V_abs_cub + c2p*V_abs_sqr + c1p*V_abs) * kp * P10
        q_expr = (c3q*V_abs_cub + c2q*V_abs_sqr + c1q*V_abs) * kq * Q10
        Ire_ip = casadi.if_else(
            _EPSILON < V_abs_sqr,  
            (p_expr * Vre + q_expr * Vim) / V_abs_sqr, 
            0.0)
        Iim_ip = casadi.if_else(
            _EPSILON < V_abs_sqr, 
            (-q_expr * Vre + p_expr * Vim) / V_abs_sqr, 
            0.0)
        # original function
        y_p = casadi.power(V_abs_sqr, Exp_v_p/2 -1) * kp * P10 
        y_q = casadi.power(V_abs_sqr, Exp_v_q/2 -1) * kq * Q10
        Ire =  y_p * Vre + y_q * Vim
        Iim = -y_q * Vre + y_p * Vim
        # compose load functions from original and interpolated
        return (
            casadi.if_else(V_abs_sqr < vminsqr, Ire_ip, Ire), 
            casadi.if_else(V_abs_sqr < vminsqr, Iim_ip, Iim))
    return casadi.DM(0.), casadi.DM(0.)

#
# decision variables
#

def _add_vk_to_injections(injections, Vnode, k, default_value):
    """Joins injection data with symbols of node voltage and scaling factors.

    Parameters
    ----------
    injections: pandas.DataFrame (index of injection)
        * .id, str
        * .id_of_node, str
        * .P10, float, active power at voltage 1.0 pu
        * .Q10, float, reactive power at voltage 1.0 pu
        * .Exp_v_p, float, voltage exponent for active power
        * .Exp_v_q, float, voltage exponent for reactive power
        * .index_of_node, str, index of node the injection is connected to
    Vnode: pandas.DataFrame (index of node)
        * .id_of_node, str
        * .Vre, float, real part of voltage
        * .Vim, float, imaginary part of voltage
        * .V_abs_sqr, float, Vre**2 + Vim**2
    k: pandas.DataFrame (index of injection)
        * .kp, casadi.SX, symbol of scaling factor for active power
        * .kq, casadi.SX, symbol of scaling factor for reactive power
    default_value: float
        * substitue for not existing scaling factors

    Returns
    -------
    pandas.DataFrame (index of injection)
        with columns: 'id', 'id_of_node', 'P10', 'Q10', 'Exp_v_p', 'Exp_v_q',
            'index_of_node', 'kp', 'kq', 'Vre', 'Vim', 'V_abs_sqr'"""
    tmp = injections.join(k)
    tmp.fillna(default_value, inplace=True)
    return tmp.join(
        Vnode.loc[:, Vnode.columns != 'id_of_node'], on='index_of_node')
    
def _add_v_to_branch_terminals(branch_terminals, Vnode):
    """Adds node voltages to branch terminal data.

    Parameters
    ----------
    branches: pandas.DataFrame (index of branch)
        * .index_of_node
        * .index_of_other_node
    Vnode: pandas.DataFrame (index of node)
        * .Vre, real part of voltage
        * .Vim, imaginary part of voltage

    Returns
    -------
    pandas.DataFrame (index of terminal)
        * .index_of_node
        * .index_of_other_node
        * .Vre
        * .Vim
        * .Vre_other
        * .Vim_other
        * .V_abs_sqr"""
    _Vnode = Vnode.loc[:, Vnode.columns != 'id_of_node']
    branch_terminal_data = (
        branch_terminals
        .join(_Vnode, on='index_of_node')
        .join(_Vnode, on='index_of_other_node', rsuffix='_other'))
    return branch_terminal_data

def _create_v_symbols(nodes):
    """Creates symbols for real and imaginary part of node voltages.

    Parameters
    ----------
    nodes: pandas.DataFrame (str, id_of_node)
        * .idx, int, index of node

    Returns
    -------
    pandas.DataFrame (index of node)
        * .Vre
        * .Vim
        * .V_abs_sqr"""
    id_of_node = pd.Series(nodes.index, dtype=str)
    Vre = pd.Series(_create_symbols('Vre_', id_of_node))
    Vim = pd.Series(_create_symbols('Vim_', id_of_node))
    return pd.DataFrame(
        {'id_of_node': id_of_node,
         'Vre': Vre,
         'Vim': Vim,
         'V_abs_sqr': Vre**2 + Vim**2},
        index=nodes.index_of_node)

def _create_factor_symbols(unique_factors):
    """Creates symbols for variables and parameters of scaling factors.

    Parameters
    ----------
    unique_factors: pandas.MultiIndex
        int (step), str (id_of_factor)

    Returns
    -------
    pandas.Series (nt (step), str (id_of_factor))
        casadi.SX"""
    return(
        pd.Series(
#            (f"{id}:{step}" for step, id in unique_factors.values),
            (f"{id}" for step, id in unique_factors.values),
            index=unique_factors,
            name='symbol')
        .apply(casadi.SX.sym)
        )

#
# measurements
#

def _power_into_measured_branches(
        branchoutputs, branch_terminal_data):
    """Calculates power flowing into measured branches.

    Parameters
    ----------
    branchoutputs: pandas.DataFrame
        * .id_of_batch, str
        * .id_of_branch, str
        * .id_of_node, str
        * .index_of_node, int
        * .index_of_branch, int
    branch_terminal_data: pandas.DataFrame (index of branch terminal)
        * .index_of_branch, int
        * .id_of_branch, str
        * .index_of_node, int
        * .index_of_other_node, int
        * .g_tot, .b_tot, float, conductance/susceptance
        * .g_mn, .b_mn, float, conductance/susceptance
        * .g_mm_half, .b_mm_half, float, conductance/susceptance
        * .index_of_taps, int
        * .index_of_other_taps, int
        * .side, 'A'|'B'
        * .Vre, .Vim, casadi.SX, real/imaginary voltage
        * .V_abs_sqr, casadi.SX, Vre**2 + Vim**2
        * .Vre_other, .Vim_other, casadi.SX, real/imaginary voltage
        * .V_abs_sqr_other, casadi.SX, Vre_other**2 + Vim_other**2
        * .flo, .ftr, casadi.SX, factor of longitudinal/transversal admittance
        * .flo_other, casadi.SX, factor of longitudinal admittance other side
        * .Ire, .Iim, casadi.SX, real/imaginary current entering the branch
        * .P, .Q, casadi.SX, real/imaginary power entering the branch
    Returns
    -------
    pandas.DataFrame (id_of_batch)
        * .P, casadi.SX, active power
        * .Q, casadi.SX, reactive power"""
    return (
        branchoutputs.merge(
            right=branch_terminal_data,
            how='inner',
            on=['index_of_node', 'index_of_branch'])
        [['id_of_batch', 'P', 'Q']])

def _current_into_measured_branches(
        branchoutputs, branch_terminal_data):
    """Calculates current flowing into measured branches.

    Parameters
    ----------
    branchoutputs: pandas.DataFrame
        * .id_of_batch, str
        * .id_of_branch, str
        * .id_of_node, str
        * .index_of_node, int
        * .index_of_branch, int
    branch_terminal_data: pandas.DataFrame (index of branch terminal)
        * .index_of_branch, int
        * .id_of_branch, str
        * .index_of_node, int
        * .index_of_other_node, int
        * .g_tot, .b_tot, float, conductance/susceptance
        * .g_mn, .b_mn, float, conductance/susceptance
        * .g_mm_half, .b_mm_half, float, conductance/susceptance
        * .index_of_taps, int
        * .index_of_other_taps, int
        * .side, 'A'|'B'
        * .Vre, .Vim, casadi.SX, real/imaginary voltage
        * .V_abs_sqr, casadi.SX, Vre**2 + Vim**2
        * .Vre_other, .Vim_other, casadi.SX, real/imaginary voltage
        * .V_abs_sqr_other, casadi.SX, Vre_other**2 + Vim_other**2
        * .flo, .ftr, casadi.SX, factor of longitudinal/transversal admittance
        * .flo_other, casadi.SXX, factor of longitudinal admittance other side
        * .Ire, .Iim, casadi.SX, real/imaginary current entering the branch
        * .P, .Q, casadi.SX, real/imaginary power entering the branch
    Returns
    -------
    pandas.DataFrame (id_of_batch)
        * .Ire, casadi.SX, real part of electric current
        * .Im, casadi.SX, imaginary part of electric current"""
    return (
        branchoutputs.merge(
            right=branch_terminal_data,
            how='inner',
            on=['index_of_node', 'index_of_branch'])
        [['id_of_batch', 'Ire', 'Iim']])

def _power_into_measured_injection(injectionoutputs, injection_data):
    """Calculates injected active and reactive power at terminals of measured
    injections.

    Parameters
    ----------
    injectionoutputs: pandas.DataFrame

    injection_data: pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        * .id_of_batch, str, id of measurement point
        * .P, float, active power
        * .Q, float, reactive power"""
    return (
        injectionoutputs.join(
            injection_data[['P', 'Q']],
            on='index_of_injection',
            how='inner')
        [['id_of_batch', 'P', 'Q']])

def _current_into_measured_injection(injectionoutputs, injection_data):
    """Calculates injected current at measured injections.

    Parameters
    ----------
    injectionoutputs: pandas.DataFrame

    injection_data: pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        * .id_of_batch, str, id of measurement point
        * .Ire, float, active power
        * .Iim, float, reactive power"""
    return (
        injectionoutputs.join(
            injection_data, on='index_of_injection', how='inner')
        [['id_of_batch', 'Ire', 'Iim']])

def _measured_values_per_batch(id_of_batch, values, x__):
    """Sums up active and reactive power values per measurement point.

    Parameters
    ----------
    id_of_batch: str
        identifier of collection of measurements
    values: pandas.DataFrame
        * .id_of_batch
        * .values

    Returns
    -------
    pandas.DataFrame (id_of_batch)
        * P
        * Q"""
    return (
        pd.DataFrame({'id_of_batch': id_of_batch, x__: values})
        .groupby('id_of_batch')
        .sum())

def _get_measured_and_calculated_value(
        branchoutputs, branch_terminal_data,
        injectionoutputs, injection_data,
        values, x__='P'):
    """Arranges calculated (or expression of casadi.SX for calculation)
    and measured power per id_of_batch.

    Parameters
    ----------
    branchoutputs: pandas.DataFrame

    branch_terminal_data: pandas.DataFrame

    injectionoutputs: pandas.DataFrame

    injection_data: pandas.DataFrame

    values: pandas.DataFrame
    
    x__: 'P'|'Q'

    Returns
    -------
    pandas.DataFrame
        * .('P'|'Q')_measured
        * .('P'|'Q')_calculated"""     
    point_ids = values.id_of_batch.unique()
    indexer = branchoutputs.id_of_batch.isin(point_ids)
    calculated_br = (
        _power_into_measured_branches(
            branchoutputs.loc[indexer, :], branch_terminal_data)
        [['id_of_batch', x__]])
    calculated_inj = (
        _power_into_measured_injection(
            injectionoutputs, injection_data)
        [['id_of_batch', x__]])
    calculated = (
        pd.concat([calculated_br, calculated_inj])
        .groupby('id_of_batch')
        .sum())
    given = _measured_values_per_batch(
        values.id_of_batch, values[x__] * values.direction, x__)
    result = given.join(
        calculated, lsuffix='_measured', rsuffix='_calculated', how='inner')
    return result if len(result) else pd.DataFrame(
        _EMPTY_TUPLE,
        columns=[f'{x__}_measured', f'{x__}_calculated'])

def _get_measured_and_calculated_current(
        branchoutputs, branch_terminal_data,
        injectionoutputs, injection_data,
        ivalues):
    """Arranges calculated and measured current per id_of_batch.

    Parameters
    ----------
    branchoutputs: pandas.DataFrame

    branch_terminal_data: pandas.DataFrame

    injectionoutputs: pandas.DataFrame

    injection_data: pandas.DataFrame

    ivalues: pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        * .I_measured
        * .I_calculated"""
    point_ids = ivalues.id_of_batch.unique()
    indexer = branchoutputs.id_of_batch.isin(point_ids)
    Icalculated_br = _current_into_measured_branches(
        branchoutputs.loc[indexer, :], branch_terminal_data)
    Icalculated_inj = _current_into_measured_injection(
        injectionoutputs, injection_data)
    Icalculated_ri = (
        pd.concat([Icalculated_br, Icalculated_inj])
        .groupby('id_of_batch')
        .sum())
    try:
        Icalculated = (
            (Icalculated_ri.Ire.pow(2) + Icalculated_ri.Iim.pow(2)).pow(0.5)
            .rename('I'))
    except:
        return pd.DataFrame(
            _EMPTY_TUPLE, columns=['I_measured', 'I_calculated'])
    Igiven = _measured_values_per_batch(ivalues.id_of_batch, ivalues.I, 'I')
    return Igiven.join(
        Icalculated, lsuffix='_measured', rsuffix="_calculated", how='inner')

def _get_measured_and_calculated_voltage(Vvalues, Vnode):
    """Arranges calculated and measured current per node id.

    Parameters
    ----------
    Vvalues: pandas.DataFrame
        * .index_of_node
        * .V
    Vnode: pandas.DataFrame (index of node)
        * .id_of_node
        * .V_abs_sqr

    Returns
    -------
    pandas.DataFrame
        * .V_measured
        * .V_calculated"""
    if Vvalues.size:
        V = Vvalues[['V', 'index_of_node']].groupby('index_of_node').mean()
        Vcalc = Vnode.V_abs_sqr.reindex(V.index).pow(0.5).rename('V')
        ids_of_nodes = (
            Vvalues[['id_of_node', 'index_of_node']]
            .groupby('index_of_node')
            .first())
        return (
            V
            .join(Vcalc, lsuffix='_measured', rsuffix='_calculated')
            .join(ids_of_nodes)
            .set_index('id_of_node'))
    return pd.DataFrame(
        _EMPTY_TUPLE,
        columns=['V_measured', 'V_calculated'],
        index=pd.Index([], name='id_of_node'),
        dtype=float)

#
# slack
#

def _get_Vslacks(slacks):
    """Calculates real and imaginary part of given slack voltages.

    Parameters
    ----------
    slacks: pandas.DataFrame (index of node)
        * .V, complex

    Returns
    -------
    pandas.DataFrame (index of node)
        * Vre, float
        * Vim, float"""
    return (
        slacks.V.apply([np.real, np.imag])
        .rename(columns={'real': 'Vre', 'imag': 'Vim'})
        if len(slacks) else pd.DataFrame((), columns=['Vre', 'Vim']))

def _get_Vslack_symbols(Vsymbols, slacks):
    """Filters symbols of slack voltages.

    Parameters
    ----------
    Vsymbols: pandas.DataFrame (index of node)
        * .Vre, symbol
        * .Vim, symbol
    Vslacks: pandas.DataFrame (index of node)
        * .V, complex

    Returns
    -------
    tuple
        * numpy array, symbols of slack voltages"""
    return (
        Vsymbols.loc[slacks.index_of_node][['Vre', 'Vim']]
        .to_numpy()
        .reshape(-1))

def _get_slack_data(Vsymbols, slacks):
    """Filters symbols of slack voltages and calculates real and imaginary
    part from given slack voltages

    Parameters
    ----------
    Vsymbols: pandas.DataFrame (index of node)
        * .Vre, symbol
        * .Vim, symbol
    Vslacks: pandas.DataFrame (index of node)
        * .V, complex

    Returns
    -------
    tuple
        * numpy array, symbols of slack voltages
        * pandas.DataFrame (index of node)
            * Vre, float
            * Vim, float"""
    return _get_Vslack_symbols(Vsymbols, slacks), _get_Vslacks(slacks)

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
    ini[ini_isna] = myfactors.value[ini_isna]
    return ini

def get_load_scaling_factors(injectionids, given_factors, assoc, count_of_steps):
    """Creates and arranges symbols for scaling factors and initialization
    data.

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
    default_factors = (
        pd.DataFrame(
            defk(range(count_of_steps), DEFAULT_FACTOR_ID,
                 type_='const', value=1.0, min_=1.0, max_=1.0),
            columns=Loadfactor._fields)
        .set_index(['step', 'id']))
    # replace nan with values (for required default factors)
    factors = required_factors.combine_first(default_factors)
    symbols = _create_factor_symbols(factors.index)
    factors['symbol'] = symbols
    # add data for initialization
    factors['ini'] = _get_factor_ini_values(factors, symbols)
    if step_injection_part_factor.shape[0]:
        injection_factors = (
            step_injection_part_factor
            .join(factors.symbol)
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

#
# arrange input for estimation
#

def _get_branch_terminal_data(branchterminals, branchtapfactors, Vnode):
    """Arranges data of branchterminals in a DataFrame. Adds voltages of nodes
    and current to data given by branchterminals.

    Parameters
    ----------
    branchterminals: pandas.DataFrame
        power flow calculation model
    Vnode: pandas.Series, casadi.SX
        node voltages

    Returns
    -------
    pandas.DataFrame
        * .index_of_branch
        * .id_of_branch
        * .id_of_node
        * .id_of_other_node
        * .index_of_node
        * .index_of_other_node
        * .g_tot
        * .b_tot
        * .g_mn
        * .b_mn
        * .g_mm_half
        * .b_mm_half
        * .side
        * .Vre
        * .Vim
        * .V_abs_sqr
        * .Vre_other
        * .Vim_other
        * .V_abs_sqr_other
        * .Ire
        * .Iim
        * .P
        * .Q"""
    term_data = (
        _add_v_to_branch_terminals(branchterminals, Vnode)
        .fillna(-1)
        .join(branchtapfactors[['flo', 'ftr']], on='index_of_taps')
        .join(
            branchtapfactors[['flo']],
            on='index_of_other_taps',
            rsuffix='_other'))
    branch_with_taps = (
        term_data.flo.notna() | term_data.flo_other.notna())
    term_data.fillna(1.0, inplace=True)
    terms = term_data[branch_with_taps]
    g_tot = (terms.g_tr_half + terms.g_lo) * terms.ftr
    b_tot = (terms.b_tr_half + terms.b_lo) * terms.ftr
    g_mn = terms.g_lo * terms.flo * terms.flo_other
    b_mn = terms.b_lo * terms.flo * terms.flo_other
    term_data.loc[branch_with_taps, ['g_mn']] = g_mn.to_numpy()
    term_data.loc[branch_with_taps, ['b_mn']] = b_mn.to_numpy()
    term_data.loc[branch_with_taps, ['g_tot']] = g_tot.to_numpy()
    term_data.loc[branch_with_taps, ['b_tot']] = b_tot.to_numpy()
    Ire, Iim = _calculate_branch_terminal_current(term_data)
    term_data['Ire'] = Ire
    term_data['Iim'] = Iim
    P, Q = _power_into_branch(
        term_data.g_tot, term_data.b_tot, term_data.g_mn, term_data.b_mn,
        term_data.V_abs_sqr, term_data.Vre, term_data.Vim,
        term_data.Vre_other, term_data.Vim_other)
    term_data['P'] = P
    term_data['Q'] = Q
    return term_data

def _get_injection_data(injections, Vnode, k, vminsqr):
    """Arranges data of injections in a DataFrame. Adds voltages of nodes
    and current to data given by injections.

    Parameters
    ----------
    injections: pandas.DataFrame
        power flow calculation model
    Vnode: pandas.Series, casadi.SX
        node voltages
    k: pandas.Series, casadi.SX
        scaling factors
    vminsqr: float
        upper limit of interpolated load curves

    Returns
    -------
    pandas.DataFrame
        * .id, str
        * .id_of_node, str
        * .P10, float
        * .Q10, float
        * .Exp_v_p, float
        * .Exp_v_q, float
        * .scalingp, float
        * .scalingq, float
        * .kp_min, float
        * .kp_max, float
        * .kq_min, float
        * .kq_max, float
        * .index_of_node, int
        * .kp, casadi.SX
        * .kq, casadi.SX
        * .Vre, casadi.SX
        * .Vim, casadi.SX
        * .V_abs_sqr, casadi.SX
        * .Ire, casadi.SX
        * .Iim, casadi.SX
        * .P, casadi.SX
        * .Q, casadi.SX"""
    injection_data = _add_vk_to_injections(injections, Vnode, k, 1.0)
    injection_data = add_interpol_coeff_to_injections(injection_data, vminsqr)
    Ire, Iim = _injected_current(injection_data, vminsqr)
    injection_data['Ire'] = Ire.elements()
    injection_data['Iim'] = Iim.elements()
    P, Q = _injected_power(injection_data, vminsqr)
    injection_data['P'] = P.elements()
    injection_data['Q'] = Q.elements()
    return injection_data

Estimation_data = namedtuple(
    'Estimation_data',
    'slacks slack_indexer Vsymbols Vsymbols_var '
    'branch_terminal_data branch_taps '
    'kvars kconsts '
    'injection_data V P Q I Inode')
Estimation_data.__doc__ = """Data for estimating the state of an electric
network.

Parameters
----------
slacks: pandas.DataFrame
    * .id_of_node, str
    * .V, complex, voltage at slack bus
    * .index_of_node, int
slack_indexer: numpy.ndarray (bool)
    indexer for Vsymbols, selects slacks
Vsymbols: pandas.DataFrame
    * .id_of_node, str
    * .Vre, casadi.SX
    * .Vim, casadi.SX
    * .V_abs_sqr, casadi.SX
Vsymbols_var: ??
    *  ...
kvars: pandas.DataFrame
    *  ...
kconsts: pandas.DataFrame
    *  ...
branch_terminal_data: pandas.DataFrame
    * .index_of_branch, int
    * .id_of_branch, str
    * .id_of_node, str
    * .id_of_other_node, str
    * .index_of_node, int
    * .index_of_other_node, int
    * .g_tot, float
    * .b_tot, float
    * .g_mn, float
    * .b_mn, float
    * .g_mm_half, float
    * .b_mm_half, float
    * .side, str
    * .Vre, casadi.SX
    * .Vim, casadi.SX
    * .V_abs_sqr, casadi.SX
    * .Vre_other, casadi.SX
    * .Vim_other, casadi.SX
    * .V_abs_sqr_other, casadi.SX
    * .Ire, casadi.SX
    * .Iim, casadi.SX
    * .P, casadi.SX
    * .Q, casadi.SX
branch_taps: pandas.DataFrame
    * .id, str
    * .Vstep, float
    * .positionmin, int
    * .positionneutral, int
    * .positionmax, int
    * .index_of_terminal, int
    * .pos, casadi.SX
injection_data: pandas.DataFrame
    * .id, str
    * .id_of_node, str
    * .P10, float
    * .Q10, float
    * .Exp_v_p, float
    * .Exp_v_q, float
    * .scalingp, float
    * .scalingq, float
    * .kp_min, float
    * .kp_max, float
    * .kq_min, float
    * .kq_max, float
    * .index_of_node, int
    * .kp, casadi.SX
    * .kq, casadi.SX
    * .Vre, casadi.SX
    * .Vim, casadi.SX
    * .V_abs_sqr, casadi.SX
    * .Ire, casadi.SX
    * .Iim, casadi.SX
    * .P, casadi.SX
    * .Q, casadi.SX
P: pandas.DataFrame
    * .P_measured, float
    * .P_calculated, casadi.SX
Q: pandas.DataFrame
    * .Q_measured, float
    * .Q_calculated, casadi.SX
I: pandas.DataFrame
    * .I_measured, float
    * .I_calculated, casadi.SX
V: pandas.DataFrame
    * .V_measured, float
    * .V_calculated, casadi.SX
Inode: casadi.SX
    expressions for calculation of node current without slack nodes"""

def _calculate_Y_by_V(node_index, Vsymbols, gb):
    """Creates a vector of node currents (selected nodes only) from
    branches and node voltages.

    Parameters
    ----------
    node_index: pandas.Index
        identifies nodes to be considered
    Vsymbols: casadi.SX
        vector of node voltage symbols
    gb: casadi.SX
        branch admittance matrix with separated conductance and susceptance

    Returns
    -------
    casadi.SX
        shape 2n,1"""
    node_index_ri = _get_index_ri(node_index)
    V_node = casadi.vertcat(Vsymbols[['Vre', 'Vim']].to_numpy().reshape(-1, 1))
    return casadi.mtimes(gb[node_index_ri, :], V_node)

def _add_I_injected(node_index, Y_by_V, injection_data):
    """Creates a vector of node current casadi.SX-expressions
    (selected nodes only).

    Parameters
    ----------
    node_index: pandas.Index
        identifies nodes to be considered
    Y_by_V: casadi.SX (2nx1)
        current vector
    injection_data: pandas.DataFrame

    Returns
    -------
    casadi.SX
        shape (2nx1)"""
    I_injected = _get_injected_node_current(injection_data, node_index)
    return casadi.simplify(Y_by_V + I_injected)

def _get_branch_estimation_data(model, Vsymbols):
    """Arranges data for estimation.

    Parameters
    ----------
    model: gridmodel.Model

    Vsymbols: pandas.DataFrame (index of node)
        * .id_of_node, str, identifier of node
        * .Vre, casadi.SX, symbol of node voltage, real part
        * .Vim, casadi.SX, symbol of node voltage, imaginary part
        * .V_abs_sqr, casadi.SX, expression Vre**2 + Vim**2

    Returns
    -------
    tuple
        * pandas.DataFrame, terminal data with columns:
            'index_of_branch', 'id_of_branch', 'index_of_node',
            'index_of_other_node', 'g_tot', 'b_tot', 'g_mn', 'b_mn',
            'g_mm_half', 'b_mm_half', 'index_of_taps', 'index_of_other_taps',
            'side', 'Vre', 'Vim', 'V_abs_sqr', 'Vre_other', 'Vim_other',
            'V_abs_sqr_other', 'flo', 'ftr', 'flo_other', 'Ire', 'Iim',
            'P', 'Q'
        * pandas.DataFrame, tap data with columns:
            'id', 'Vstep', 'positionmin', 'positionneutral', 'positionmax',
            'pos'"""
    branchtaps = model.branchtaps.copy()
    possymbols = _create_symbols('pos_', model.branchtaps.id)
    branchtaps['pos'] = possymbols
    branchtapfactors = _get_branch_tap_factors(branchtaps)
    terminals = model.branchterminals
    branch_terminal_data = _get_branch_terminal_data(
        terminals[~terminals.is_bridge], branchtapfactors, Vsymbols)
    return (
        branch_terminal_data,
        branchtaps[
            ['id', 'Vstep', 'positionmin', 'positionneutral',
             'positionmax', 'position', 'pos']])

_EMPTY_LOAD_SCALING_FACTORS = pd.DataFrame(
    [],
    columns=(Loadfactor._fields + ('symbol', 'ini')))

def _query_load_scaling_factors_of_type(type_, load_scaling_factors_step):
    """Computes load_scaling_factors_step.loc[type_], returns an empty
    pandas.DataFrame if key 'type_' is not in data frame

    Parameters
    ----------
    type_: 'var'|'const'

    load_scaling_factors_step: pandas.DataFrame
        ...

    Returns
    -------
    pandas.DataFrame
        with columns: 'id', 'Vstep', 'positionmin', 'positionneutral',
             'positionmax', 'pos'"""
    try:
        return load_scaling_factors_step.loc[type_].reset_index()
    except:
        return _EMPTY_LOAD_SCALING_FACTORS.copy();

def get_estimation_data(model, count_of_steps, vminsqr=_VMINSQR):
    """Prepares data for estimation.
    Vsymbols and expressions of V are reused in all steps.
    ksymbols are specific for each step, hence, expressions of PQ, I and Inode
    are (potentially) specific for each step.
    k_ini expressions (for initial load scaling factors) reference the
    previous optimization step. Therefore, data retrieval needs the
    evaluation function of the previous step.

    Parameters
    ----------
    model: gridmodel.Model

    count_of_steps: int
        number of optimization steps
    vminsqr: float
        upper limit of interpolated load curves (default _VMINSQR)

    Returns
    -------
    function (int) -> (Estimation_data)"""
    pfc_nodes = get_pfc_nodes(model.nodes)
    Vsymbols = _create_v_symbols(pfc_nodes)
    branch_terminal_data, branch_taps = _get_branch_estimation_data(
        model, Vsymbols)
    slack_indexer = pfc_nodes.reset_index().set_index('index_of_node').is_slack
    Vsymbols_var =  (
        Vsymbols.loc[~slack_indexer, ['Vre', 'Vim']].to_numpy().reshape(-1))
    node_index = Vsymbols.index[~slack_indexer]
    Y_by_V = _calculate_Y_by_V(
        node_index, Vsymbols, _create_gb_branch_matrix(branch_terminal_data))
    load_scaling_factors, injection_factors = get_load_scaling_factors(
        model.injections.id,
        model.load_scaling_factors,
        model.injection_factor_associations,
        count_of_steps)
    def of_step(step):
        """Arranges data for estimation of one step.

        Parameters
        ----------
        step: int
            optimization step which the data shall be prepared for

        Returns
        -------
        Estimation_data"""
        try:
            load_scaling_factors_step = load_scaling_factors.loc[step]
        except:
            load_scaling_factors_step = None
        kvars = _query_load_scaling_factors_of_type(
            'var', load_scaling_factors_step)
        kvars.rename(
            columns={'id': 'id_of_k', 'min': 'kmin', 'max': 'kmax'},
            inplace=True)
        kvars.set_index('id_of_k', inplace=True)
        kconsts = _query_load_scaling_factors_of_type(
            'const', load_scaling_factors_step)
        kconsts.rename(
            columns={'id': 'id_of_k', 'min': 'kmin', 'max': 'kmax'},
            inplace=True)
        kconsts.set_index('id_of_k', inplace=True)
        try:
            kpq = (
                injection_factors.loc[step, ['index_of_injection', 'kp', 'kq']]
                .set_index('index_of_injection'))
        except:
            kpq = pd.DataFrame(
                (),
                columns=['kp', 'kq'],
                index=pd.Index([], name='index_of_injection'))
        injection_data = _get_injection_data(
            model.injections, Vsymbols, kpq, vminsqr)
        V = _get_measured_and_calculated_voltage(model.vvalues, Vsymbols)
        P = _get_measured_and_calculated_value(
            model.branchoutputs,
            branch_terminal_data,
            model.injectionoutputs,
            injection_data,
            model.pvalues,
            'P')
        Q = _get_measured_and_calculated_value(
            model.branchoutputs,
            branch_terminal_data,
            model.injectionoutputs,
            injection_data,
            model.qvalues,
            'Q')
        I = _get_measured_and_calculated_current(
            model.branchoutputs,
            branch_terminal_data,
            model.injectionoutputs,
            injection_data,
            model.ivalues)
        Inode = _add_I_injected(node_index, Y_by_V, injection_data)
        return Estimation_data(
            # constant part
            slacks=model.slacks,
            slack_indexer=slack_indexer,
            Vsymbols=Vsymbols,
            Vsymbols_var=Vsymbols_var,
            branch_terminal_data=branch_terminal_data,
            branch_taps=branch_taps,
            # step-specific part
            kvars=kvars,
            kconsts=kconsts,
            injection_data=injection_data,
            V=V, # constant, not step specific
            P=P,
            Q=Q,
            I=I,
            Inode=Inode)
    return of_step

def _calculate_norm(ars):
    """Calculates a norm.

    Parameters
    ----------
    ars: list
        numpy.array (nx2)

    Returns
    -------
    float"""
    if not ars:
        return casadi.DM()
    ar = casadi.vcat(ars)
    return casadi.sum1((ar[:,0] - ar[:,1]) ** 2)

def _create_constraints(target_frames):
    """Creates expressions for constraints of target values (measurements and
    setpoint).

    Parameters
    ----------
    target_frames: iterable
        tuple
            * pandas.DataFrame (str, id_of_measurements)
                * float, target value (e.g. measurement)
                * expression for calculating the value from the state
                  variables
            * str, 'P'|'Q'|'I'|'V'

    Returns
    -------
    tuple
        * casadi.SX (nx1), variables of parameters
        * casadi.SX (nx1), expressions of constraints"""
    list_var = []
    list_expressions = []
    for target_frame, prefix in target_frames:
        list_var += (
            casadi.SX.sym(prefix + id) for id in target_frame.index)
        # column with index 1 is expression of how to calculate the value
        list_expressions += target_frame.iloc[:,1].to_list()
    vars = casadi.vcat(list_var)
    return ((vars, vars - casadi.vcat(list_expressions))
        if vars.size1() else
        (vars, vars))

def _create_gb_branch_matrix(terms):
    """Creates a conductance-susceptance branch matrix for power flow
    calculation. The matrix is equivalent to an admittance branch matrix.
    However the admittances are split into real and imaginary parts.

    Before power flow calculation the slack rows need modification.

    Parameters
    ----------
    terms: pandas.DataFrame
        * .index_of_node, int
        * .index_of_other_node, int
        * .g_mn, float, longitudinal conductance
        * .b_mn, float, longitudinal susceptance
        * .g_tot, float, half of self conductance plus g_mn
        * .b_tot, float, half of self susceptance plus b_mn

    Returns
    -------
    casadi.SX"""
    idx_a = 2 * terms.index_of_node
    idx_a_p_1 = idx_a + 1
    #index is zero based, unlike shape
    size = idx_a_p_1.max() + 1 if len(idx_a_p_1) else 0
    idx_b = 2 * terms.index_of_other_node
    idx_b_p_1 = idx_b + 1
    rows = pd.concat([idx_a, idx_a_p_1, idx_a_p_1, idx_a])
    columns_mn = pd.concat([idx_b, idx_b_p_1, idx_b, idx_b_p_1])
    vals_mn = pd.concat([-terms.g_mn, -terms.g_mn, -terms.b_mn, terms.b_mn])
    gb = casadi.SX(size, size)
    for r, c, v in zip(rows, columns_mn, vals_mn):
        gb[r, c] += v
    columns_mm = pd.concat([idx_a, idx_a_p_1, idx_a, idx_a_p_1])
    vals_mm = pd.concat([terms.g_tot, terms.g_tot, terms.b_tot, -terms.b_tot])
    for r, c, v in zip(rows, columns_mm, vals_mm):
        gb[r, c] += v
    return gb

def _get_index_ri(index):
    """Creates an index for separate real and imaginary parts from
    an index of complex.

    Parameters
    ----------
    index: pandas.Series
        int

    Returns
    -------
    pandas.Series
        int"""
    index_r = 2 * index.to_numpy().reshape(-1,1)
    index_i = index_r + 1
    return np.hstack([index_r, index_i]).reshape(-1)

def _get_injected_node_current(injection_data, node_index):
    """Creates a vector of injected node currents (index_of_node->current).

    Parameters
    ----------
    injection_data: pandas.DataFrame
        * .index_of_node
        * .Ire
        * .Iim
    V_node_index: pandas.Series
        bool

    Returns
    -------
    casasdi.SX
        shape (2nx1)"""
    return casadi.vertcat(
        injection_data[['index_of_node', 'Ire', 'Iim']]
        .groupby('index_of_node')
        .sum()
        .reindex(node_index)
        .fillna(0.0)
        .stack()
        .to_numpy())

#
# power flow calculation
#

def _get_symbols_pf(estimation_data):
    """Extracts symbols from estimation_data for power flow calculation.

    Returns
    -------
    list
        * [0]: symbols of decision (voltage) variables
        * [1]: symbols of slack voltages, tap positions, scaling factors"""
    Vnode_vars = casadi.vertcat(
        estimation_data
        .Vsymbols
        .loc[~estimation_data.slack_indexer, ['Vre', 'Vim']]
        .to_numpy()
        .reshape(-1,1))
    # slack voltages
    Vslack_const = casadi.vertcat(
        estimation_data
        .Vsymbols
        .loc[estimation_data.slack_indexer, ['Vre', 'Vim']]
        .to_numpy()
        .reshape(-1,1))
    branch_tap_pos = estimation_data.branch_taps.pos.to_numpy()
    params = casadi.vertcat(
        Vslack_const,
        branch_tap_pos,
        casadi.vcat(estimation_data.kconsts.symbol),
        casadi.vcat(estimation_data.kvars.symbol))
    return [Vnode_vars, params]

def _create_rootfinder(estimation_data):
    """Creates a root finding function which solves the power flow problem.

    Parameters
    ----------
    estimation_data: Estimation_data

    Returns
    -------
    casadi.casadi.Function"""
    vars_and_consts_symbols = _get_symbols_pf(estimation_data)
    fn_Iresidual = casadi.Function(
        'fn_Iresidual', vars_and_consts_symbols, [estimation_data.Inode])
    return casadi.rootfinder('rf', 'nlpsol', fn_Iresidual, {'nlpsol':'ipopt'})

def _calculate_pf(rootfinder, values_of_parameters, Vguess):
    """Function for power flow calculation.
    (parameters: Vslack, positions_of_branch_taps, kconsts, kvars)

    Parameters
    ----------
    rootfinder: casadi.casadi.Function
        function ([float], [float]) -> casadi.casadi.DM,
        (in particular:
             (initial_node_voltages_ri, values_of_parameter) ->
             (node_voltages_ri))
    values_of_parameters: array_like
        value of parameters
    Vguess: array_like
        float, start value of unknown voltage vector
    Vslack: array_like
        float, slack voltage, parameters
    positions_of_branch_taps: array_like
        int, parameters
    kconsts: array_like
        float, load scaling factors, parameters
    kvars: array_like
        float, load scaling factors, parameters

    Returns
    -------
    tuple
        * bool, success
        * casadi.casadi.DM, vector of voltages,
            sequence of separated, alternating real and imaginary parts"""
    voltages = rootfinder(Vguess, values_of_parameters)
    return rootfinder.stats()['success'], voltages

def _get_calculate_power_flow(estimation_data, values_of_parameters):
    """Parameterizes function _calculate_pf for power flow calculation.

    Parameters
    ----------
    estimation_data: Estimation_data

    values_of_parameters: array_like

    Returns
    -------
    casadi.casadi.Function
        (initial_voltages, positions_of_branch_taps, load_scaling_factors)
        ->(bool, casadi.casadi.DM) which is (success, vector of voltages)"""
    return partial(
        _calculate_pf,
        _create_rootfinder(estimation_data),
        values_of_parameters)

#
# input data
#

def update_branch_tap_positions(branch_taps, tap_positions):
    """Returns tap position. Updates column 'position'.

    Parameters
    ----------
    branch_taps: pandas.DataFrame
        * .id
        * .positionneutral
    tap_positions: array_like
        tuple (str (id_of_taps), int (position))

    Returns
    -------
    numpy.array
        int"""
    modified_taps = (
        pd.DataFrame(tap_positions, columns=('id', 'position'))
        .set_index('id'))
    taps = branch_taps.loc[:, ['id', 'position']]
    taps.set_index('id', inplace=True)
    taps.update(modified_taps)
    taps.reset_index(inplace=True)
    return taps.position.to_numpy()

#
# set-point and measurement
#

def get_calculated(target_frames):
    """Arranges expressions of target values (measurements and setpoint).

    Parameters
    ----------
    target_arrays: list
        numpy.array<float>

    Returns
    -------
    casadi.SX(n, 1)"""
    return casadi.vcat(
        [target_frame.iloc[:,1].to_numpy()
         for target_frame, _ in target_frames])

target_value_getter = {
    'P':lambda ed:(ed.P.loc[:, ['P_measured', 'P_calculated']]),
    'Q':lambda ed:(ed.Q.loc[:, ['Q_measured', 'Q_calculated']]),
    'I':lambda ed:(ed.I.loc[:, ['I_measured', 'I_calculated']]),
    'V':lambda ed:(ed.V.loc[:, ['V_measured', 'V_calculated']])}

@singledispatch
def get_objective_array(key, estimation_data):
    msg = ("function get_objective_array received parameter 'key' "
           f"of unexpected type. (type:{type(key)}, value:{str(key)}) "
           "possible types are <class str> (\'PQIV\') and <class Term>")
    raise RuntimeError(msg)

@get_objective_array.register(str)
def _(key, estimation_data):
    try:
        return casadi.vertcat(
            target_value_getter.get(key)(estimation_data).to_numpy())
    except:
        msg = ("Error, something went wrong, please check indicators of "
               "objective types, function get_objective_array received "
               f"a value for parameter key of {str(key)}, possible "
               f"values are {target_value_getter.keys()}.")
        raise RuntimeError(msg)

@get_objective_array.register(Term)
def _(key, estimation_data):
    if key.type == 'k': # k - scaling factor
        both = estimation_data.kvars.reindex([key.a, key.b]).symbol
        if both.isna().any():
            msg = (
                "Error, attributes 'a' and 'b' of 'Term'-instances "
                "having type 'k' need to adress defined 'var' "
                "scaling factors, however, given Term does not provide "
                f"valid references to such scaling factors: {key}")
            raise RuntimeError(msg)
        return key.factor * both.to_numpy().reshape(1,2)
    elif key.type == 'V':
        measurement_vid = key.a
        try:
            return key.factor * casadi.vertcat(
                estimation_data.V.loc[
                    measurement_vid,['V_measured', 'V_calculated']]
                .to_numpy()
                .reshape(-1,2))
        except:
            msg = (
                "Error, attribute 'a' of a 'Term' instance having type 'V' "
                "needs to adress a connectivity node with an associated "
                "voltage value, however, given Term-instance does not "
                "provide a valid reference to such a "
                f"connectivity node: {key}")
            raise ValueError(msg)
    elif key.type == 'kavg':
        try:
            kvars = estimation_data.kvars.reindex(key.a).symbol
            if kvars.isna().any() or kvars.shape[0] < 1:
                msg = (
                    "Error, attribute 'a' of 'Term'-instances "
                    "having type 'kavg' must be a tuple of references to "
                    "'var' scaling factors, however, given Term does not "
                    "provide multiple references to such scaling factors or "
                    f"includes invalid references: {key}")
                raise RuntimeError(msg)
            symbols = kvars.to_numpy()
            count_of_symbols = len(symbols)
            avg = casadi.rdivide(
                casadi.sum1(casadi.vcat(symbols)), count_of_symbols)
            return key.factor * casadi.horzcat(
                casadi.SX.ones(count_of_symbols, 1) * avg, symbols)
        except:
            msg = (
                "Error, while processing 'Term'-instance having type 'kavg': "
                f"{key}")
            raise ValueError(msg)
    elif key.type == 'k0':
        try:
            kvars = estimation_data.kvars.reindex(key.a).symbol
            if kvars.isna().any() or kvars.shape[0] < 1:
                msg = (
                    "Error, attribute 'a' of 'Term'-instances "
                    "having type 'k0' must be a tuple of references to "
                    "'var' scaling factors, however, given Term does not "
                    "provide multiple references to such scaling factors or "
                    f"includes invalid references: {key}")
                raise RuntimeError(msg)
            symbols = kvars.to_numpy()
            count_of_symbols = len(symbols)
            refvals = np.array([0.0] * count_of_symbols).reshape(-1, 1)
            return key.factor * np.hstack([refvals, symbols.reshape(-1,1)])
        except:
            msg = (
                "Error, while processing 'Term'-instance having type 'kavg': "
                f"{key}")
            raise ValueError(msg)

@singledispatch
def get_objective_arrays(include, estimation_data):
    return [get_objective_array(key, estimation_data) for key in include]

@get_objective_arrays.register(Term)
def _(include, estimation_data):
    return [get_objective_array(include, estimation_data)]

def get_target_values(include, estimation_data):
    return ((target_value_getter.get(key)(estimation_data), key)
            for key in include)

def get_target_calculated(estimation_data, include):
    """Arranges expressions (formulas) for calculation of values from node
    voltages, power of injections and branch admittances. The expressions
    are retrieved for measurements and set-points (targets). The function
    returns for instance the expression for calculating the active power
    flowing into a branch from voltges at both terminals and admittances
    if there are data for an active power measurement at this particular
    terminal.

    Parameters
    ----------
    estimation_data: Estimation_data

    include: str
        concatenation of characters 'P', 'Q', 'I', 'V', each character
        is a symbol for the type of measurements to include in the output

    Returns
    -------
    casadi.SX(n, 1)"""
    return get_calculated(get_target_values(include, estimation_data))

#
# NLP
#

Nlp = namedtuple( 'Nlp', 'nlp lbx ubx Vslacks')

def get_nlp(estimation_data, objectives='PQIV', constraints=''):
    """Creates a none linear program for electrical network state estimation.

    Parameters
    ----------
    model: gridmodel.Model
        grid data prepared for power flow calculation
    objectives: str
        "P?Q?I?V?", default "PQIV'"
        terms of targets value norms (norms of measurements and setpoints;
        (value_measured - value_calculated) ** 2)
        to include into the objective function
        P - active power
        Q - reactive power
        I - electric current
        V - electric voltage
    constraints: str
         terms of targets values to include in constraints
        "P?Q?I?V?", default ""

    Returns
    -------
    Nlp
        * .nlp, none-linear-program for the casadi solver
        * .is_slack, pandas.Indexer filters slack nodes or voltages
        * .Vslacks, pandas.DataFrame of slack voltages (.Vre, .Vim)"""
    objective = _calculate_norm(
        get_objective_arrays(objectives, estimation_data))
    constraint_parameters, measurement_constraints = _create_constraints(
        get_target_values(constraints, estimation_data))
    myconstraints = casadi.vertcat(
        estimation_data.Inode, measurement_constraints)
    # voltage symbols, decision variables only
    Vsymbols_var = estimation_data.Vsymbols_var
    count_of_voltage_vars = Vsymbols_var.size
    # lower bound of decision variables
    lbx = np.concatenate(
        [[-np.inf] * count_of_voltage_vars, estimation_data.kvars.kmin])
    # upper bound of decision variables
    ubx = np.concatenate(
        [[np.inf] * count_of_voltage_vars, estimation_data.kvars.kmax])
    # decision variables
    sym_x = casadi.vertcat(
        Vsymbols_var,
        casadi.vcat(estimation_data.kvars.symbol))
    # parameters
    slack_voltage_symbols, Vslacks = _get_slack_data(
        estimation_data.Vsymbols,
        estimation_data.slacks)
    sym_p = casadi.vertcat(
        slack_voltage_symbols,
        estimation_data.branch_taps.pos.to_numpy(),
        casadi.vcat(estimation_data.kconsts.symbol),
        constraint_parameters)
    return Nlp(
        nlp={'x': sym_x, 'f': objective, 'g': myconstraints, 'p': sym_p},
        lbx=lbx,
        ubx=ubx,
        Vslacks=Vslacks)

#
# estimation
#

def calculate_values(xp, values_of_params, x, expressions):
    """Extracts values for expressions from optimization result vector.

    Parameters
    ----------
    xp: tuple
        * casadi.casadi.SX, decision variables
        * casadi.casadi.SX, parameters
    values_of_params: casadi.DM
        values of parameters
    x: casadi.DM
        result vector of optimization
    expressions: numpy.array
        casadi.casadi.SX

    Returns
    -------
    casadi.casadi.DM"""
    fn = casadi.Function('fn', xp, expressions)
    return fn(x, values_of_params)

def create_evaluating_function(nlp, values_of_params, x):
    """Creates a function which calculates values for expression.

    Parameters
    ----------
    nlp: dict
        'non-linear program'
    values_of_params: casadi.DM
        values for parameters
    x: casadi.DM
        values for variables

    Returns
    -------
    function
        (casadi.SX) -> (casadi.DM)"""
    return partial(calculate_values, _get_xp(nlp), values_of_params, x)

def prepare_initial_data(values_of_parameters, estimation_data):
    """Calculates power flow on given data in order to create
    initial data for the first optimization step.

    Parameters
    ----------
    values_of_parameters: array_like
        float/int, values for parameters of mynlp
        (slack voltages, positions of taps, values of constant scaling factors)
    estimation_data: Estimation_data
        source of decision variables, parameter variables and
        target data (measurement and set-point data)

    Returns
    -------
    tuple
        * bool, success
        * funtion (casadi.SX) -> (casadi.DM) for result retrieval"""
    count_of_none_slack_nodes = np.sum(~estimation_data.slack_indexer)
    Vguess = [1., .0] * count_of_none_slack_nodes
    kvars_guess = estimation_data.kvars.value.to_numpy()
    calculate_power_flow = _get_calculate_power_flow(
        estimation_data, casadi.vertcat(values_of_parameters, kvars_guess))
    success, voltages_pf_calc = calculate_power_flow(Vguess)
    # create evaluating function
    # decision variables
    sym_x = casadi.vertcat(
        estimation_data.Vsymbols_var,
        casadi.vcat(estimation_data.kvars.symbol))
    voltages_k_pf = casadi.vertcat(voltages_pf_calc, kvars_guess)
    # do not include measurement constraints, create constraints especially
    #   for initial power flow calculation
    slack_voltage_symbols = _get_Vslack_symbols(
        estimation_data.Vsymbols,
        estimation_data.slacks)
    sym_p = casadi.vertcat(
        slack_voltage_symbols,
        estimation_data.branch_taps.pos.to_numpy(),
        casadi.vcat(estimation_data.kconsts.symbol))
    evaluate_expression = partial(
        calculate_values,
        (sym_x, sym_p),
        values_of_parameters,
        voltages_k_pf)
    return success, evaluate_expression

def estimate(vslack_tappos, estimation_data, previous_data, mynlp,
             evaluate_expression, constraint_types):
    """Runs an optimization step.

    Parameters
    ----------
    vslack_tappos: array_like
        float/int, values for parameters of mynlp
        (slack voltages, positions of taps)
    estimation_data: Estimation_data
        source of scaling factor variables (decisision variables)
    previous_data: Estimation_data
        source of initial values for voltage (decision) variables and
        constraint parameters for target data (measurement and set-point data)
    mynlp: Nlp
        non-linear program returned by function 'get_nlp'
    evaluate_expression: function
        (casadi.SX) -> (casadi.DM)
        function returns values for given expressions, the function
        is created by a previous or initial optimization step, it is used
        for calculating initial values for decision and parameter variables
    constraint_types: str
        indicates type of target values to fix in optimization by constraints,
        the string may contain characters 'P', 'Q'. 'I', 'V'
        for active power, reactive power, electric current and voltage

    Returns
    -------
    tuple
        * bool, success
        * funtion (casadi.SX) -> (casadi.DM) for result retrieval"""
    # 'estimation_data.kvars.ini' and 'estimation_data.kconsts.ini' provide
    #   scaling factor symbols of previous run (or skalar values - especially
    #   for the initial run), 'evaluate_expression' provides
    #   the context of the previous run, thus the initial values are
    #   retrieved from previous run
    initial_values = casadi.vcat(
        evaluate_expression(
            [previous_data.Vsymbols_var,
             casadi.vcat(estimation_data.kvars.ini)]))
    kconst_ini = evaluate_expression(
        [casadi.vcat(estimation_data.kconsts.ini)])
    values_of_constraint_parameters = evaluate_expression(
        [get_target_calculated(previous_data, constraint_types)])
    values_of_params = casadi.vertcat(
        vslack_tappos, kconst_ini, values_of_constraint_parameters)
    # check if we have an objective function
    myf = mynlp.nlp['f']
    if(myf.is_empty() or myf.is_zero()):
        return (
            True,
            create_evaluating_function(
                mynlp.nlp, values_of_params, initial_values))
    #solver = casadi.nlpsol('solver', 'bonmin', mynlp0.nlp, {'discrete': [False]*9})
    solver = casadi.nlpsol('solver', 'ipopt', mynlp.nlp)
    # solve
    r = solver(
        x0=initial_values,
        lbg=0, ubg=0,
        lbx=mynlp.lbx, ubx=mynlp.ubx,
        p=values_of_params)
    return (
        solver.stats()['success'],
        create_evaluating_function(mynlp.nlp, values_of_params, r['x']))

def calculate(
        model=None, parameters_of_steps=(), tap_positions=(), vminsqr=_VMINSQR):
    """Estimates grid status stepwise.

    Parameters
    ----------
    model: egrid.model.Model

    parameters_of_steps: array_like
        dict {'objectives': objectives, 'constraints': constraints}
            if empty the function calculates power flow,
            each dict triggersan estimization step
        * objectives, ''|'P'|'Q'|'I'|'V' (string or tuple of characters)
          'P' - objective function is created with terms for active power
          'Q' - objective function is created with terms for reactive power
          'I' - objective function is created with terms for electric current
          'V' - objective function is created with terms for voltage
        * constraints, ''|'P'|'Q'|'I'|'V' (string or tuple of characters)
          'P' - adds constraints keeping the initial values
                of active powers at the location of given values
          'Q' - adds constraints keeping the initial values
                of reactive powers at the location of given values
          'I' - adds constraints keeping the initial values
                of electric current at the location of given values
          'V' - adds constraints keeping the initial values
                of voltages at the location of given values
    tap_positions: array_like
        tuple str, int - (ID of Branchtap, position)
    vminsqr: float (default _VMINSQR)
        minimum

    Yields
    ------
    tuple
        * int, estimation step,
          (initial power flow calculation result is -1, first estimation is 0)
        * bool, success?
        * Estimation_data,
        * function, (casadi.SX) -> (casadi.DM), for result retrieval"""
    get_estimation_data_for_step = get_estimation_data(
        model, len(parameters_of_steps))
    # create initial values for first step
    estimation_data = get_estimation_data_for_step(0)
    positions_of_branch_taps = update_branch_tap_positions(
        estimation_data.branch_taps, tap_positions)
    values_of_parameters = casadi.vertcat(
        _get_Vslacks(estimation_data.slacks).to_numpy().reshape(-1),
        positions_of_branch_taps)
    # initial power flow calculation
    success, evaluate_expression = prepare_initial_data(
        casadi.vertcat(
            values_of_parameters, estimation_data.kconsts.value),
        estimation_data)
    yield -1, success, estimation_data, evaluate_expression
    if success:
        for step, parameters in enumerate(parameters_of_steps):
            previous_data = estimation_data
            if step:
                estimation_data = get_estimation_data_for_step(step)
            types_of_constraints = parameters.get('constraints','')
            mynlp = get_nlp(
                estimation_data,
                objectives=parameters.get('objectives',''),
                constraints=types_of_constraints)
            success, evaluate_expression = estimate(values_of_parameters,
                estimation_data, previous_data, mynlp, evaluate_expression,
                types_of_constraints)
            yield step, success, estimation_data, evaluate_expression
            if not success:
                return
