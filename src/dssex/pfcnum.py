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

Created on Fri May  6 20:44:05 2022

@author: pyprg
"""

import numpy as np
from operator import itemgetter
from scipy.sparse import csc_array
from injections import get_polynomial_coefficients, get_node_inj_matrix

# square of voltage magnitude, minimum value for load curve, 
#   if value is below _VMINSQR the load curves for P and Q converge
#   towards a linear load curve which is 0 when V=0; P(V=0)=0, Q(V=0)=0
_VMINSQR = 0.8

_zeros = np.zeros((0, 1), dtype=np.longdouble)
_power_props = itemgetter('P10', 'Q10', 'Exp_v_p', 'Exp_v_q')

def _injected_power(vminsqr, injections):
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
    function: (numpy.array<float>) -> (numpy.array<float>, numpy.array<float>)
        tuple
            * active power P
            * reactive power Q"""
    if injections.size:
        P10, Q10, Exp_v_p, Exp_v_q = _power_props(injections)
        p_coeffs = get_polynomial_coefficients(vminsqr, injections.Exp_v_p)
        q_coeffs = get_polynomial_coefficients(vminsqr, injections.Exp_v_q)
        coeffs = np.hstack([p_coeffs, q_coeffs])
        Exp_v_p_half = Exp_v_p.to_numpy() / 2.
        Exp_v_q_half = Exp_v_q.to_numpy() / 2.
        #        
        def calc_injected_power(Vinj_abs_sqr):
            """Calculates injected power per injection.
            
            Parameters
            ----------
            Vinj_abs_sqr: numpy.array, float, shape (n,1)
                vector of squared voltage-magnitudes at injections, 
                n: number of injections

            Returns
            -------
            tuple
                * active power P
                * reactive power Q"""
            Vinj_abs_sqr2 = Vinj_abs_sqr.reshape(-1)
            Pres = np.array(P10)
            Qres = np.array(Q10)
            interpolate = (Vinj_abs_sqr2 < vminsqr)
            # original
            Vorig = Vinj_abs_sqr2[~interpolate]
            Pres[~interpolate] *= np.power(Vorig, Exp_v_p_half[~interpolate])
            Qres[~interpolate] *= np.power(Vorig, Exp_v_q_half[~interpolate])
            # polynomial interpolated
            Vsqr_inter = Vinj_abs_sqr2[interpolate]
            cinterpolate = coeffs[interpolate]
            V_abs = np.power(Vsqr_inter, .5)
            V321 = (
                np.hstack([Vsqr_inter * V_abs, Vsqr_inter, V_abs])
                .reshape(-1, 3))
            Pres[interpolate] *= np.sum(V321 * cinterpolate[:, :3], axis=1)
            Qres[interpolate] *= np.sum(V321 * cinterpolate[:, 3:], axis=1)
            return Pres, Qres
        return calc_injected_power
    return lambda _, __: _zeros, _zeros
#%%

from dnadb import egrid_frames
from egrid import model_from_frames

path = r"C:\UserData\deb00ap2\OneDrive - Siemens AG\Documents\defects\SP7-219086\eus1_loop\eus1_loop.db"
frames = egrid_frames(path)
model = model_from_frames(frames)
Vslack = model.slacks.V
idx_slack = model.slacks.index_of_node
mnodeinj = get_node_inj_matrix(model.shape_of_Y[0], model.injections)
mnodeinjT = mnodeinj.T
calc_injected_power = _injected_power(_VMINSQR, model.injections)


#%%

Vnode = np.array(
    [1.+0j] * model.shape_of_Y[0], 
    dtype=np.complex128).reshape(-1,1)
Vnode_abs_sqr = np.power(abs(Vnode), 2)



Vinj_abs_sqr = mnodeinjT @ Vnode_abs_sqr
Pinj, Qinj = calc_injected_power(Vinj_abs_sqr.T)
Sinj = np.hstack([Pinj.reshape(-1, 1), Qinj.reshape(-1, 1)]).view(dtype=np.complex128)
Sinj_node = mnodeinj @ csc_array(Sinj)

Inode = Sinj_node / Vnode
Inode[idx_slack] = Vslack


