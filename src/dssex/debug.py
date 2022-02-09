# -*- coding: utf-8 -*-
"""
Created on Sun Aug  19 08:36:10 2021

@author: pyprg
"""
from dssex.estim import get_load_scaling_factors
from egrid import make_model, make_data_frames
from functools import singledispatch

@singledispatch
def show_factors(model, count_of_steps=3):
    """
    
    Returns
    -------
    tuple
        pandas.DataFrame of strings for debugging only,
        makes scaling factors visible"""
    factors, injection_factors = get_load_scaling_factors(
        model.injections.id, 
        model.load_scaling_factors,
        model.injection_factor_association,
        count_of_steps)
    return factors.applymap(str), injection_factors.applymap(str)

@show_factors.register(list)
def _(grid, count_of_steps=3):
    model = make_model(make_data_frames(grid))
    return show_factors(model, count_of_steps=3)