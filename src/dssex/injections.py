# -*- coding: utf-8 -*-
"""
Copyright (C) 2022, 2023 pyprg

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

def get_coefficients_matrix(x1_sqr):
    """Creates a matrix for the calculation of coefficients for a cubic
    function interpolation.
    ::
        f(x) = Ax³ + Bx² + Cx; f(0)=0
    The returned matrix M is used for the calculation of the coefficents
    A, B and C.
    ::
        M x coefficients = V
        with V = [y, dy/dx_0, dy/dx_1]
             dy/dx_0 derivative at x=0
             dy/dx_1 derivative at x=x1

    Parameters
    ----------
    x1_sqr: float
        square of x1 (== x1 * x1)

    Returns
    -------
    numpy.array (shape 3,3)"""
    x1 = np.sqrt(x1_sqr)
    x1_cub = x1_sqr*x1
    return np.array([
        [   x1_cub, x1_sqr, x1],  # f(x1)
        [       0.,     0., 1.],  # df(x0)/dx
        [3.*x1_sqr,  2.*x1, 1.]]) # df(x1)/dx

def make_calculate_coefficients(x1_sqr):
    """Creates a function calculating coefficients of cubic polynomials.

    (numpy.array<float> shape r,c) -> (numpy.array<float> (shape r,3,c)).
    The returned function creates an array of shape (3,1) for each scalar.

    Parameters
    ----------
    x1_sqr: float

    Returns
    -------
    function
        (float) -> (numpy.array<float> (shape 3,1))
        (exponent) -> (3_coefficients_for_interpolated_function)"""
    m = get_coefficients_matrix(x1_sqr)
    @lru_cache(maxsize=200)
    def calculate_coefficients(exp):
        """Calculates three coefficients for a cubic polynomial.
        ::
            p(x) = Ax³ + Bx² + Cx

        The following parameters of the function are given.
        ::
            f(x) = x**exp function used for calculation of value and
                            derivative in x1 (x1 is upper limit of the
                            interpolation interval [0...x1])
            p(x1) = y1
            dp(x1)/dx = exp * x1**(exp-1)
            p(0) = 0
            dp(0)/dx = 1

        Parameters
        ----------
        exp: float
            exponent for function x**exp

        Returns
        -------
        numpy.array (3,1)"""
        y = np.power(x1_sqr, exp/2) # == x1**exp
        dydx = exp * np.power(x1_sqr, (exp-1)/2) # == exp*x1**exp-1
        return solve(m, np.array([y, 1, dydx]).reshape(-1,1))
    # creates a NumPy ufunc which calculates 3 coefficients for each float
    #   scalar of an array
    calc_coeffs = np.frompyfunc(calculate_coefficients, 1, 1)
    return lambda exp: (
        np.apply_along_axis(np.hstack, 1, calc_coeffs(exp))
        if exp.shape[0] else np.ndarray(exp.shape, dtype=float))

def calculate_cubic_coefficients(x1_sqr, exp):
    """Calculates three coefficients for a cubic polynomial.
    ::
        p(x) = Ax³ + Bx² + Cx

    The following parameters of the function are given.
    ::
        f(x) = x**exp function used for calculation of value and
                        derivative in x1 (x1 is upper limit of the
                        interpolation interval [0...x1])
        p(x1) = y1
        dp(x1)/dx = exp * x1**(exp-1)
        p(0) = 0
        dp(0)/dx = 1

    Parameters
    ----------
    x1_sqr: float

    exp: float
        exponent for function x**exp

    Returns
    -------
    numpy.array (3,1)"""
    return make_calculate_coefficients(x1_sqr)(exp)

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

def get_polynomial_coefficients(x1_sqr, exp, dydx_0=1.):
    """Calculates coefficients of polynomials for interpolation.

    Parameters
    ----------
    x1_sqr: float
        upper limit of interpolation range, squared
    exp: numpy.array
        float, exponents of function 'load over voltage'
    dydx_0: float
        df(0) / dx

    Returns
    -------
    numpy.array
        float, shape(n,3)"""
    if len(exp):
        x1 = np.sqrt(x1_sqr)
        cm = get_coefficients_matrix(x1_sqr)
        return np.vstack([
            solve(cm, np.array([y1, dydx_0, dx]))
            for y1, dx in zip(np.power(x1, exp), calc_dx(x1, exp))])
    return np.ndarray(shape=(0,3), dtype=float)

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
    x_sqr = x*x
    cm = get_coefficients_matrix(x_sqr)
    coeffs = solve(cm, np.array([x1**exp, 1, calc_dx(x1, exp)]))
    return (coeffs[0] * x_sqr * x) + (coeffs[1] * x_sqr) + (coeffs[2] * x)

def interpolate(x1, exp, x):
    """Function for checking the interpolation concept.

    The function x**exp shall be interpolated near x~0 that y->0 for x=0.
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
    return polynome(x1, exp, x) if x < x1 else x**exp

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
