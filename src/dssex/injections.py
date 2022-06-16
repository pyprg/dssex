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
        [   x_cub, x_sqr,  x],  # f(x1)
        [      0.,    0., 1.],  # df(x0)/dx
        [3.*x_sqr,  2.*x, 1.]]) # df(x1)/dx
    return solve(cm, np.array([y, dydx_0, dydx]))

def calc_dx(x, exp):
    """Calculates the derivative of exponential power function at x.
    
    Parameters
    ----------
    x: float
        magnitude
    exp: float
        exponent
    
    Returns
    -------
    float"""
    return exp * np.power(x, exp-1)

def get_polynomial_coefficients(x1, exp, dydx_0=1.):
    """Calculates coefficients of polynomials for interpolation.
    
    Parameters
    ----------
    x1: float
        upper limit of interpolation range
    exp: numpy.array
        float, exponents of function 'load over voltage'
    dydx_0: float
        df(0) / dx
    
    Returns
    -------
    numpy.array
        float, shape(n,3)"""
    return np.vstack([
        get_coefficients(x1, y1, dydx_0, dx) 
        for y1, dx in zip(np.power(x1, exp), calc_dx(x1, exp))])

def polynome(x1, exp, x):
    """Function for checking the polynome.
    
    Parameters
    ----------
    x1: float
        coordinate at which y1 = x1**exp
    exp: float
        exponent of original function
    x: float
        value to calculate y for"""
    xsqr = x*x
    coeffs = get_coefficients(
        x1, 
        x1**exp, 1., 
        calc_dx(x1, exp))
    return (coeffs[0] * xsqr * x) + (coeffs[1] * xsqr) + (coeffs[2] * x)

def interpolate(x1, exp, x):
    """Function for checking the interpolation concept.
    the function x**exp shall be interpolated near x~0 that y->0 for x=0.
    The function is interpolated with a polynome of third order.
    
    Parameters
    ----------
    x1: float
        upper limit of interpolation, above the limit y is according to 
        the original function, below is interpolated
    exp: float
        exponent of original function
    x: float
        value to calculate y for"""
    if x < x1:
        return polynome(x1, exp, x)
    else:
        return x**exp

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
