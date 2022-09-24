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

Created on Sun Aug  19 08:36:10 2021

@author: pyprg
"""
from dssex.estim import get_load_scaling_factors
from egrid import make_model, make_data_frames
from functools import singledispatch

@singledispatch
def show_factors(model, count_of_steps=3):
    """For debugging. Create tables of scaling factors for 'manual' checking.

    Returns
    -------
    tuple
        pandas.DataFrame of strings for debugging only,
        makes scaling factors visible"""
    factors, injection_factors = get_load_scaling_factors(
        model.injections.id,
        model.load_scaling_factors,
        model.injection_factor_associations,
        count_of_steps)
    return factors.applymap(str), injection_factors.applymap(str)

@show_factors.register(list)
def _(grid, count_of_steps=3):
    model = make_model(make_data_frames(grid))
    return show_factors(model, count_of_steps=3)