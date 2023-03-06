# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 22:50:14 2023

@author: pyprg
"""
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

letters, numbers, '=', and '.' are evaluated data
names of connectivity nodes start with 'n'
adjacent entities are connected unless the name has a leading or
  trailing underscore '_'
lines with first character '#' are not evaluated
"""

def calculate_pf(model, step_params=()):
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

    letters, numbers, '=', and '.' are evaluated data
    names of connectivity nodes start with 'n'
    adjacent entities are connected unless the name has a leading or
      trailing underscore '_'
    lines with first character '#' are not evaluated

    Parameters
    ----------
    args: iterable (optional)
        egrid.builder.(
            Branch | Slacknode | Injection | Output |
            PValue | QValue | IValue | Vvalue | Branchtaps |
            Defk | Link | Message) | str

    Returns
    -------
    dict
        * messages
        * branches
        * injections"""
    from dssex.estim import estimate
    from dssex.pfcnum import calculate_electric_data
    from egrid import make_model
    from pandas import DataFrame, concat
    from numpy import zeros
    messages = model.messages
    if all(messages.level < 2):
        # res is last value generated by estimate
        *_, res = estimate(model, step_params)
        step, succ, v_cx, kpq = res
        if succ:
            txt = 'calculation successful'
            m = concat([messages, DataFrame([dict(message=txt, level=0)])])
            ed = calculate_electric_data(model, v_cx, kpq)
            return dict(
                messages=m, branches=ed.branch(), injections=ed.injection())
        else:
            txt = 'calculation failed'
            m = concat([messages, DataFrame([dict(message=txt, level=2)])])
    else:
        txt = 'not calculated, error(s) in model'
        m = concat([messages, DataFrame([dict(message=txt, level=2)])])
    ed = calculate_electric_data(
        make_model(), zeros((0,1), dtype=complex))
    return dict(messages=m, branches=ed.branch(), injections=ed.injection())

def print_power_flow(*args):
    """Calculates the power flow of given network model. Prints the result.

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

    letters, numbers, '=', and '.' are evaluated data
    names of connectivity nodes start with 'n'
    adjacent entities are connected unless the name has a leading or
      trailing underscore '_'
    lines with first character '#' are not evaluated

    Parameters
    ----------
    args: iterable (optional)
        egrid.builder.(
            Branch | Slacknode | Injection | Output |
            PValue | QValue | IValue | Vvalue | Branchtaps |
            Defk | Link | Message) | str"""
    from egrid import make_model_checked
    if not args:
        args = DEFAULT_NETWORK
        print(args)
    model = make_model_checked(args)
    for title, df in calculate_pf(model).items():
        print(f'\n>{title.upper()}>')
        print(df.to_markdown())