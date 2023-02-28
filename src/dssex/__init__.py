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

Created on Sun Aug  8 08:36:10 2021

@author: pyprg
"""

from .estim import estimate
from .pfcnum import calculate_electric_data

DEFAULT_NETWORK = """
         slack=True   y_lo=1e3-1e3j             y_lo=2e3-2e3j
        n0<----------cable--------->n1<------line------>n2
                                    |                   |
                                    |                   |
                     motor_a ((~))--n1--((~)) motor_b  _n2--> oven
                      P10=42                   P10=7           P10=27
                      Q10=4.2                  Q10=.7          Exp_v_p=1
                      Exp_v_p=2
                      Exp_v_q=2

# letters, numbers, '=', and '.' are evaluated data
# names of connectivity nodes start with 'n'
# adjacent entities are connected unless the name has a leading or
#   trailing underscore '_'
# lines with first character '#' are not evaluated
"""

def calculate_power_flow(*args):
    """Calculates the power flow of given network model.

    A multiline string can be the model e.g
    (the default input if no args are given)
    ::
         slack=True   y_lo=1e3-1e3j             y_lo=2e3-2e3j
        n0<----------cable--------->n1<------line------>n2
                                    |                   |
                                    |                   |
                     motor_a ((~))--n1--((~)) motor_b  _n2--> oven
                      P10=42                   P10=7           P10=27
                      Q10=4.2                  Q10=.7          Exp_v_p=1
                      Exp_v_p=2
                      Exp_v_q=2

    Parameters
    ----------
    args: iterable (optional)
        egrid.builder.(
            Branch | Slacknode | Injection | Output |
            PValue | QValue | IValue | Vvalue | Branchtaps |
            Defk | Link | Message) | str

    Returns
    -------
    dict"""
    from egrid import \
        create_objects, make_data_frames, get_failure, model_from_frames
    try:
        frames = make_data_frames(
            create_objects(args if args else DEFAULT_NETWORK))
        error_msg = get_failure(frames)
        if error_msg is None:
            model = model_from_frames(frames)
            # res is last value generated by estimate
            *_, res = estimate(model)
            step, succ, v_cx, kpq = res
            if succ:
                ed = calculate_electric_data(model, v_cx, kpq)
                return dict(
                    success=succ,
                    branches=ed.branch(),
                    injections=ed.injection())
            else:
                return dict(success=False, message='')
        else:
            dict(success=False, message=error_msg)
    except Exception as e:
        return dict(success=False, message=str(e))

def print_power_flow(*args):
    """Calculates power flow and prints results.

    A multiline string can be the model e.g
    (the default input if no args are given)
    ::
         slack=True   y_lo=1e3-1e3j             y_lo=2e3-2e3j
        n0<----------cable--------->n1<------line------>n2
                                    |                   |
                     motor_a ((~))--n1--((~)) motor_b  _n2--> oven
                      P10=42                   P10=7           P10=27
                      Q10=4.2                  Q10=.7          Exp_v_p=1
                      Exp_v_p=2
                      Exp_v_q=2

    Parameters
    ----------
    args: iterable (optional)
        egrid.builder.(
            Branch | Slacknode | Injection | Output | PValue |
            QValue | IValue | Vvalue | Branchtaps |
            Defk | Link | Message) | str"""
    from pandas import DataFrame
    if args is None:
        args = [DEFAULT_NETWORK]
    res = calculate_power_flow(*args)
    for arg in args:
        if isinstance(arg, str):
            print()
            print(arg)
    for k, v in res.items():
        print()
        if isinstance(v, DataFrame):
            if len(v):
                print(f'{k}:')
                print(v.to_markdown())
        else:
            print(f'{k}: {v}')
