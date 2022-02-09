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
    msg = model.errormessages
    if len(msg):
        print()
        print(msg.to_markdown())
    else:
        *_, res = calculate(model) # res is last value from generator
        print_estim_result(res)
