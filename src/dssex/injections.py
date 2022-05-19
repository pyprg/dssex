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

Created on Fri May  6 20:49:20 2022

@author: pyprg
"""

import numpy as np
from functools import lru_cache
from numpy.linalg import solve

@lru_cache(maxsize=200)
def get_coefficients(x, y, dydx_0, dydx):
    """Calculates the coefficients A, B, C of the polynomial
    ::
        f(x) = Ax³ + Bx² + Cx
        with
            f(0) = 0, 
            df(0)/dx = dydx_0,
            f(x) = y
            df(x)/dx = dydx
            
    Parameters
    ----------
    x: float
        x of point
    y: float
        value at x
    dydx_0: float 
        dy / dx at 0
    dydx: float 
        dy / dx at x

    Returns
    -------
    numpy.ndarray (shape=(3,))
        float, float, float (A, B, C)"""
    x_sqr = x * x
    x_cub = x * x_sqr
    cm = np.array([
        [   x_cub, x_sqr,  x],
        [      0.,    0., 1.],
        [3.*x_sqr,  2.*x, 1.]])
    return solve(cm, np.array([y, dydx_0, dydx]))

def calc_dpower_dvsqr(v_sqr, _2p):
    """Calculates the derivative of exponential power function at v_sqr.
    
    Parameters
    ----------
    v_sqr: float
        square of voltage magnitude
    _2p: float
        double of voltage exponent
    
    Returns
    -------
    float"""
    p = _2p/2
    return p * np.power(v_sqr, p-1)

def get_polynomial_coefficients(ul, exp):
    """Calculates coefficients of polynomials for interpolation.
    
    Parameters
    ----------
    ul: float
        upper limit of interpolation range
    exp: numpy.array
        float, exponents of function 'load over voltage'
    
    Returns
    -------
    numpy.array
        float, shape(n,3)"""
    return np.vstack([
        get_coefficients(ul, 1., 1., dp)
        for dp in calc_dpower_dvsqr(ul, exp)])

def add_interpol_coeff_to_injections(injections, vminsqr):
    """Adds polynomial coefficients to injections for linear interpolation
    of power around |V|² ~ 0.
    
    Parameters
    ----------
    injections: pandas.DataFrame
        * .Exp_v_p
        * .Exp_v_q
    vminsqr: float
        upper limit of interpolation
        
    Returns
    -------
    pandas.DataFrame
        modified injections"""
    p_coeffs = get_polynomial_coefficients(vminsqr, injections.Exp_v_p)
    injections['c3p'] = p_coeffs[:, 0]
    injections['c2p'] = p_coeffs[:, 1]
    injections['c1p'] = p_coeffs[:, 2]
    q_coeffs = get_polynomial_coefficients(vminsqr, injections.Exp_v_q)
    injections['c3q'] = q_coeffs[:, 0]
    injections['c2q'] = q_coeffs[:, 1]
    injections['c1q'] = q_coeffs[:, 2]
    return injections
