# -*- coding: utf-8 -*-
"""
Created on Sun Aug  19 08:36:10 2021

@author: Carsten Laves
"""
import pandas as pd
import numpy as np
from collections import namedtuple
from operator import attrgetter, methodcaller


get_IPQV = attrgetter('I', 'PQ', 'V')

#
# results related to network elements
#

def _create_vnode_value_frame(Vsymbols, vnode_vals):
    """Creates a pandas.DataFrame (id of node) with voltage.
    
    Parameters
    ----------
    Vsymbols: pandas.DataFrame (index of node)
        * .id_of_node, str
        * .Vre, casadi.SX, real part of node voltage
        * .Vim, casadi.SX, imaginary part of node voltage
        * .V_abs_sqr, casadi.SX, Vre**2 + Vim**2
    vnode_vals: np.array (nx1)
        float
    Returns
    -------
    pandas.DataFrame (id_of_node)
        * .Vabs, float"""
    return pd.DataFrame(
        {'id_of_node': Vsymbols.id_of_node,
         'Vabs': vnode_vals.reshape(-1)},
        index=Vsymbols.index.rename('index_of_node'))

def _add_VIPQ(df, values):
    Isqr = np.power(values[:,1:3], 2)
    df['Vabs'] = np.sqrt(values[:,0])
    df['Iabs'] = np.sqrt(Isqr[:, 0] + Isqr[:, 1])
    df['P'] = values[:, 3]
    df['Q'] = values[:, 4]
    return df

def _add_term_results(df, values):
    Isqr = np.power(values[:,1:3], 2)
    df['Vabs_sqr'] = values[:,0]
    df['Vabs_sqr_other'] = values[:,5]
    df['Vabs'] = np.sqrt(values[:,0])
    df['Iabs'] = np.sqrt(Isqr[:, 0] + Isqr[:, 1])
    df['P'] = values[:, 3]
    df['Q'] = values[:, 4]
    return df

def _add_VIPQkpkq(df, values):
    df2 = _add_VIPQ(df, values)
    df2['kp'] = values[:, 5]
    df2['kq'] = values[:, 6]
    return df2

def _create_injection_value_frame(injection_data, injection_vals):
    """Creates a pandas.DataFrame with result values of injections 
    (index of injection).
    
    Parameters
    ----------
    injection_data: pandas.DataFrame
        * .id, str, unique identifier
        * .id_of_node, str, unique identifier of connected node
        * .P10, float, scheduled active power if voltage is 1.0 p.u.
        * .Q10, float, scheduled reactive power if voltage is 1.0 p.u.
        * .Exp_v_p, float, voltage exponent of active power
        * .Exp_v_q, float, voltage exponent of reactive power
        * .index_of_node, int, index of connected node
        * .kp, casadi.SX, symbol of decision variable for scaling factor of
            active power
        * .kq, casadi.SX, symbol of decision variable for scaling factor of
            reactive power
        * .Vre, casadi.SX, symbol of decision variable for voltage at
            connected node, real part
        * .Vim, casadi.SX, symbol of decision variable for voltage at
            connected node, imaginary part
        * .V_abs_sqr, casadi.SX, expression Vre**2 + Vim**2
        * .Irem, casadi.SX, real current flow
        * .Iim, casadi.SX, imaginary current flow
        * .P, casadi.SX, active power flow
        * .Q, casadi.SX, reactive power flow
    injection_vals: numpy.array<float> (index of injection)
        * [0] - V_abs_sqr
        * [1] - Ire
        * [2] - Iim
        * [3] - P
        * [4] - Q
        * [5] - kp
        * [6] - kq
    
    Returns
    -------
    pandas.DataFrame (index of injection)
        * .id_of_injection, str, unique identifier
        [all input fields]"""
    injection_values = _add_VIPQkpkq(
        injection_data[['id']].copy(), injection_vals)
    injection_values.index.set_names('index_of_injection', inplace=True)
    return injection_values

def get_branch_values(terminal_values):
    """Arranges branch terminal data per branch. Calculates active and
    reactive 'losses' P_loss, Q_loss.
    
    Parameters
    ----------
    terminal_values: pandas.DataFrame
        * .id_of_branch, str
        * .side, str
        * .Vabs, float
        * .Iabs, float
        * .P, float
        * .Q, float
    
    Returns
    -------
    pandas.DataFrame (index of branch)
        * .id_of_branch, str, unique identifier
        * .Vabs_A, float, voltage magnitude at node A
        * .Vabs_B, float, voltage magnitude at node A
        * .Iabs_A, float, magnitude of current flow into terminal A
        * .Iabs_B, float, magnitude of current flow into terminal B
        * .P_A, float, active power flow into terminal A
        * .P_B, float, active power flow into terminal B
        * .Q_A, float, reactive power flow into terminal A
        * .Q_B, float, reactive power flow into terminal B
        * .P_loss, float, active power losses
        * .Q_loss, float, reactive power losses"""
    branch_losses = (
        terminal_values[['index_of_branch', 'P', 'Q']]
        .groupby('index_of_branch')
        .sum()
        .rename(columns={'P': 'P_loss', 'Q': 'Q_loss'}))
    terminal_values.set_index(['index_of_branch', 'side'], inplace=True)
    branches = terminal_values.unstack('side')
    branches.columns = np.array([f'{p}_{q}' for p, q in branches.columns])
    branches.drop('id_of_branch_B', axis=1, inplace=True)
    branches.rename(
        columns={'id_of_branch_A': 'id_of_branch'}, 
        inplace=True)
    return branches.join(branch_losses)

def _create_branch_value_frame(branch_terminal_data, terminal_vals):
    terminal_values = _add_VIPQ(
        (branch_terminal_data[['id_of_branch', 'index_of_branch', 'side']]
         .copy()), 
        terminal_vals)
    return get_branch_values(terminal_values)

def _create_terminal_value_frame(branch_terminal_data, terminal_vals):
    return _add_term_results(
        (branch_terminal_data[['id_of_branch', 'index_of_branch', 'side']]
         .copy()), 
        terminal_vals)
    return get_branch_values(terminal_values)

Result_factory = namedtuple(
    'Result_factory', 
    'node_values injection_values branch_values terminal_values')
Result_factory.__doc__ = """Collection of functions returning
arranged results.

Parameters
----------
node_values: function
    ()->(pandas.DataFrame)
injection_values: function
    ()->(pandas.DataFrame)
branch_values: function
    ()->(pandas.DataFrame)
terminal_values: function
    ()->(pandas.DataFrame)"""

VNODE_RESULT_COLUMNS = [
    'V_abs_sqr']
INJECTION_RESULT_COLUMNS = [
    'V_abs_sqr', 'Ire', 'Iim', 'P', 'Q', 'kp', 'kq']
BRANCH_TERMINAL_RESULT_COLUMNS = [
    'V_abs_sqr', 'Ire', 'Iim', 'P', 'Q', 'V_abs_sqr_other']

def _get_vnode_formulas(estimation_data):
    return np.sqrt(
        estimation_data.Vsymbols[VNODE_RESULT_COLUMNS].to_numpy())    

def _get_injection_formulas(estimation_data):
    return (
        estimation_data
        .injection_data[INJECTION_RESULT_COLUMNS]
        .to_numpy())

def _get_terminal_formulas(estimation_data):
    return (
        estimation_data
        .branch_terminal_data[BRANCH_TERMINAL_RESULT_COLUMNS]
        .to_numpy())

def create_value_factory(estimation_data, evaluate_expression):
    """Creates a factory object for estimation results.
    
    Parameters
    ----------
    estimation_data: Estimation_data
    
    evaluate_expression: function
        (array_like<casadi.SX>) -> (array_like<casadi.DM>)
    
    Returns
    -------
    Result_factory
        * .node_values, function, () -> (pandas.DataFrame)
        * .injection_values, function, () -> (pandas.DataFrame)
        * .branch_values, function, () -> (pandas.DataFrame)
        * .terminal_values, function, () -> (pandas.DataFrame)"""
    Vnode_formulas = _get_vnode_formulas(estimation_data)
    injection_formulas = _get_injection_formulas(estimation_data)
    terminal_formulas = _get_terminal_formulas(estimation_data)
    vnode_vals, injection_vals, terminal_vals = evaluate_expression(
        [Vnode_formulas, injection_formulas, terminal_formulas])
    return Result_factory(
        node_values=lambda:_create_vnode_value_frame(
            estimation_data.Vsymbols, np.array(vnode_vals)),
        injection_values=lambda:_create_injection_value_frame(
            estimation_data.injection_data, np.array(injection_vals)),
        branch_values=lambda:_create_branch_value_frame(
            estimation_data.branch_terminal_data, np.array(terminal_vals)),
        terminal_values=lambda:_create_terminal_value_frame(
            estimation_data.branch_terminal_data, np.array(terminal_vals)))

def get_branch_data(estimation_data, evaluate_expression):
    """Extracts branch values from result of estimation.
    
    Parameters
    ----------
    estimation_data: Estimation_data
    
    evaluate_expression: function
        (array_like<casadi.SX>) -> (array_like<casadi.DM>)
    
    Returns
    -------
    pandas.DataFrame"""
    terminal_formulas = _get_terminal_formulas(estimation_data)
    terminal_vals = evaluate_expression([terminal_formulas])
    return _create_branch_value_frame(
        estimation_data.branch_terminal_data, np.array(terminal_vals))
#
# presentation of results at points of measurements/set-points
#

Measurement_Result_factory = namedtuple(
    'Measurement_Result_factory', 
    'Vmeasured_calculated PQmeasured_calculated Imeasured_calculated')
Measurement_Result_factory.__doc__ = """Collection of functions returning
arranged results for measured data.

Parameters
----------
Vmeasured_calculated: function
    ()->(pandas.DataFrame)
PQmeasured_calculated: function
    ()->(pandas.DataFrame)
Imeasured_calculated: function
    ()->(pandas.DataFrame)"""

def _add_values(df, vals, column_names):
    """Adds a new column to pandas.DataFrame df.
    
    Parameters
    ----------
    df: pandas.DataFrame
    
    vals: casadi.DM
        values of new column
    column_name: array_like
        str, name of new columns
        
    Returns
    -------
    pandas.DataFrame"""
    if len(df):
        df_copy = df.copy()
        for idx in range(len(column_names)):
            df_copy[column_names[idx]] = np.array(vals[:, idx])
        return df_copy
    return pd.DataFrame(columns=df.columns.append(pd.Index(column_names)))

def create_measurement_result_factory(estimation_data, evaluate_expression):
    """Creates a factory object for estimation results of measured values.
    
    Parameters
    ----------
    estimation_data: Estimation_data
    
    evaluate_expression: function
        (array_like<casadi.SX>) -> (array_like<casadi.DM>)
        the function is returned by the estimation function, it is the
        container for accessing estimation results
    
    Returns
    -------
    Measurement_Result_factory
        * .Vmeasured_calculated, function, () -> (pandas.DataFrame)
        * .PQmeasured_calculated, function, () -> (pandas.DataFrame)
        * .Imeasured_calculated, function, () -> (pandas.DataFrame)"""
    Idata, PQdata, Vdata = get_IPQV(estimation_data)
    # retrieve values from estimation result
    Imeasured_calc, PQmeasured_calc, Vmeasured_calc = evaluate_expression(
        [Idata.I_calculated.to_numpy(),
         PQdata[['P_calculated', 'Q_calculated']].to_numpy(),
         Vdata.V_calculated.to_numpy()])
    return Measurement_Result_factory(
        Vmeasured_calculated=lambda:_add_values(
            Vdata[['V_measured']].copy(),
            Vmeasured_calc,
            ['V_calculated']),
        PQmeasured_calculated=lambda:_add_values(
                PQdata[['P_measured', 'Q_measured']].copy(),
                PQmeasured_calc,
                ['P_calculated', 'Q_calculated'])
            [['P_measured', 'P_calculated', 'Q_measured', 'Q_calculated']],
        Imeasured_calculated=lambda:_add_values(
            Idata[['I_measured']].copy(), 
            Imeasured_calc, 
            ['I_calculated']))

#
# result printing
#

def print_results(grid_values):
    """(Generic) printing function. Assumes grid_values fields are funtions
    and calls one after another. Prints the return values.
    
    Parameters
    ----------
    grid_values: namedtuple
        all fields are callable functions: ()->(printable object)"""
    for method_name in grid_values._fields:
        print()
        print(method_name)
        print(methodcaller(method_name)(grid_values).to_markdown())

def print_estim_result(estim_results):
    """Prints results of estimation  for each step.
    
    Parameters
    ----------
    estim_result: itarable
        tuple 
            * int
            * bool
            * function (casadi.SX) -> (casadi.DM)
            * Estimation_data"""
    for step, success, estimation_data, evaluate_expression in estim_results:
        print()
        # :1 is the format
        print(f"===[ step: {step:1} ]========================================")
        if success:
            results = create_value_factory(
                estimation_data, evaluate_expression)
            print_results(results)
            measurement_results = create_measurement_result_factory(
                estimation_data, evaluate_expression)
            print_results(measurement_results)
            # analysis of residual node currents
            Inode_calc = evaluate_expression([estimation_data.Inode])
            print()
            print('residual Inode: ', Inode_calc)
        else:
            print('failed')

_caller_name = {
    'P': 'PQmeasured_calculated', 
    'Q': 'PQmeasured_calculated',
    'I': 'Imeasured_calculated', 
    'V': 'Vmeasured_calculated'}

def arrange_measurement_results(step_resultfn, quantity):
    measured_quantity = f"{quantity}_measured"
    calculated_quantity = f"{quantity}_calculated"
    level_names=['quantity', 'step']
    get_results = methodcaller(_caller_name[quantity])
    first_step, results_first_step = step_resultfn[0]
    measured_calculated = get_results(results_first_step)
    df = measured_calculated[[measured_quantity, calculated_quantity]]
    df.columns = pd.MultiIndex.from_tuples(
        [(measured_quantity, -1), (calculated_quantity, -1)], 
        names=level_names)
    calculated = pd.DataFrame(
        {step:get_results(result).loc[:,calculated_quantity] 
         for step, result in step_resultfn[1:]})
    calculated.columns = pd.MultiIndex.from_tuples(
        [(calculated_quantity, step) for step in calculated.columns], 
        names=level_names)
    return pd.concat([df, calculated], axis=1)

def print_measurements(estim_results, quantities='VPQI'):
    step_resultfn = [
        (step, create_measurement_result_factory(
            estimation_data, evaluate_expression))
        for step, success, estimation_data, evaluate_expression 
        in estim_results]
    for quantity in quantities:
        print()
        print(
            arrange_measurement_results(step_resultfn, quantity)
            .to_markdown())
