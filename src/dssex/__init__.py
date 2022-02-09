# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 08:36:10 2021

@author: pyprg
"""

__doc__ = """

"""

from .estim import calculate

def print_power_flow(*args):
    """Calculates the power flow of given network model. 
    
    For instance this multiline string can be the model
    (it is the default input if no args are given)
    ::
        n0<-------------cable------------>n1--((~)) motor
         slack=True      y_mn=1e3-1e3j               P10=42        
    
    Parameters
    ----------
    args: iterable
        egrid.builder.(
            Branch | Slacknode | Injection | Output | 
            PQValue | IValue | Vvalue | Branchtaps | Defk | Link) | str"""
    from egrid import make_model
    from .present import print_estim_result
    model = make_model(
        args if args else
        """
        n0<-------------cable------------>n1--((~)) motor
         slack=True      y_mn=1e3-1e3j               P10=42""")
    *_, res = calculate(model) # res is last value from generator
    print_estim_result(res)
