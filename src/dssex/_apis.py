# -*- coding: utf-8 -*-
"""
Copyright (C) 2023 pyprg

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

Created on Mon Mar  6 22:50:14 2023

@author: pyprg
"""
DEFAULT_NETWORK = """
         slack=True   y_lo=1e3-1e3j           y_lo=2e3-2e3j
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
# lines with first character '#' are not part of the schema
"""

def estimate(model, step_params=(), vmin=.8):
    """Calculates the power flow of given network model.

    With given step_params the function estimates the network state.

    Parameters
    ----------
    model: egrid.model.Model
        data of electric distribution network

    step_params: array_like (default ())
        dict
        ::
            {'objectives': objectives,
             'constraints': constraints,
             'floss': float, factor for losses}

            if empty the function calculates power flow,
            each dict triggers an estimation step

        * objectives, string of ''|'P'|'Q'|'I'|'V'|'U'|'L'|'C'|'T',
          objective function is created with terms:

          * 'P' for active power
          * 'Q' for reactive power
          * 'I' for electric current
          * 'V' for voltage
          * 'U' for voltage violation
          * 'L' for losses of branches
          * 'C' for cost
          * 'T' of model.terms

        * constraints, string of ''|'P'|'Q'|'I'|'V'|'U', adds constraints:

          * 'P' keeping the initial values of active power at the location of
             given active power values during this step
          * 'Q' keeping the initial values of reactive power at the location of
             given reactive power values during this step
          * 'I' keeping the initial values of electric current at the location
             of given current values during this step
          * 'V' keeping the initial values of voltages at the location of given
             voltage values during this step
          * 'U' considering voltage limits

    vmin: float (default 0.8)
        minimum magnitude of voltage at injection for load curve defined by
        model, power will be interpolated below this value, squared value

    Returns
    -------
    tuple
        * pandas.DataFrame, messages
        * array_like
          dict, calculated electric data
            * 'optimization_result', tuple
                * int, index of step
                * bool, success?
                * numpy.array, complex, voltages at nodes
                * numpy.array, float (shape m,2), values of factors
                * numpy.array, float (shape n,1), tappositions
            * 'branches', pandas.DataFrame
            * 'injections', pandas.DataFrame
            * 'nodes', pandas.DataFrame"""
    from dssex.estim import estimate_stepwise
    from dssex.result import get_printable_results, get_printable_result
    from egrid import make_model
    from pandas import DataFrame, concat
    from numpy import empty
    messages = model.messages
    if all(messages.level < 2):
        results = list(estimate_stepwise(
            model, step_params=step_params, vminsqr=vmin**2))
        printables = get_printable_results(model, results)
        return messages, list(
            p | {'optimization_result': r}
            for r, p in zip(results, printables))
    # return messages and empty result structure
    txt = 'not calculated, error(s) in model'
    msgs = concat([messages, DataFrame([dict(message=txt, level=2)])])
    v_cx = empty(shape=(0,1), dtype=complex)
    res = get_printable_result(make_model(), v_cx)
    factors = empty(shape=(0,2), dtype=float)
    tappositions = empty(shape=(0,1), dtype=float)
    return msgs, [res | {
        'optimization_result':(-1, False, v_cx, factors, tappositions)}]

def create_model_and_estimate(args=(), step_params=(), vmin=.8):
    """Calculates the power flow of a given network model.

    Parameters
    ----------
    args: iterable (optional)
        egrid.builder.(
            Branch | Slacknode | Injection | Output |
            PValue | QValue | IValue | Vvalue | Vlimit |
            Defk | Deft | Defoterm | Klink | Tlink | Message) | str

    step_params: array_like (default ())
        dict
        ::
            {'objectives': objectives,
             'constraints': constraints,
             'floss': float, factor for losses}

            if empty the function calculates power flow,
            each dict triggers an estimation step

        * objectives, string of ''|'P'|'Q'|'I'|'V'|'U'|'L'|'C'|'T',
          objective function is created with terms:

          * 'P' for active power
          * 'Q' for reactive power
          * 'I' for electric current
          * 'V' for voltage
          * 'U' for voltage violation
          * 'L' for losses of branches
          * 'C' for cost
          * 'T' of model.terms

        * constraints, string of ''|'P'|'Q'|'I'|'V'|'U', adds constraints:

          * 'P' keeping the initial values of active power at the location of
             given active power values during this step
          * 'Q' keeping the initial values of reactive power at the location of
             given reactive power values during this step
          * 'I' keeping the initial values of electric current at the location
             of given current values during this step
          * 'V' keeping the initial values of voltages at the location of given
             voltage values during this step
          * 'U' considering voltage limits

    vmin: float (default 0.8)
        minimum magnitude of voltage at injection for load curve defined by
        model, power will be interpolated below this value, squared value

    Returns
    -------
    tuple
        * pandas.DataFrame, messages
        * array_like
          dict, calculated electric data
            * 'optimization_result', tuple
                * int, index of step
                * bool, success?
                * numpy.array, complex, voltages at nodes
                * numpy.array, float (shape m,2), values of factors
                * numpy.array, float (shape n,1), tappositions
            * 'branches', pandas.DataFrame
            * 'injections', pandas.DataFrame
            * 'nodes', pandas.DataFrame"""
    from egrid import make_model_checked
    model = make_model_checked(args)
    return estimate(model, step_params, vmin)

def print_powerflow(*args):
    """Calculates the power flow of a given network model. Prints the result.

    A multiline string can be the model e.g
    (the default input if no args are given)
    ::
         slack=True   y_lo=1e3-1e3j           y_lo=2e3-2e3j
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

    lines with first character '#' are not part of input graph

    lines starting with '#.' provide non-graph input data e.g. scaling factors

    Parameters
    ----------
    args: iterable (optional)
        egrid.builder.(
            Branch | Slacknode | Injection | Output |
            PValue | QValue | IValue | Vvalue | Vlimit |
            Defk | Deft | Defoterm | Klink | Tlink | Message) | str"""
    if not args:
        args = DEFAULT_NETWORK
        print(args)
    msg, res = create_model_and_estimate(args)
    print('\n>MESSAGES>')
    print(msg.to_markdown())
    *_, result = res
    for cl, df in (
            (cl,result[cl]) for cl in ['branches','injections','nodes']):
        print(f'\n>{cl.upper()}>')
        print(df.fillna('-').to_markdown())

if __name__ == '__main__':
    print_powerflow()
