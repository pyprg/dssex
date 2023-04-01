# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 22:10:12 2023

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
Created on Fri Dec 16 00:00:58 2022

@author: pyprg
"""

import pandas as pd
import numpy as np

# def calculate_term_factor_n(factordefs, positions):
#     """Calculates values for off-diagonal factors of branches having taps.
#     Diagonal factors are just the square of the off-diagonal factors.
#     The formula for the factor is
#     ::
#         m * position + n

#     Parameters
#     ----------
#     factordefs: Factordefs
#         * .gen_termfactor, pandas.DataFrame (id_of_branch, id_of_node) ->
#             * .id, str, ID of factor
#             * .index_of_terminal
#             * .index_of_other_terminal
#         * .gen_factor_data, pandas.DataFrame (id (str, ID of factor)) ->
#             * .value, float
#             * .m, float
#             * .n, float
#     positions: numpy.array, int (shape n,1)
#         vector of position values for terms with taps

#     Returns
#     -------
#     pandas.DataFrame (index_of_terminal) ->
#         * .ftaps, float"""
#     termfactor = (
#         factordefs
#         .gen_termfactor[['id', 'index_of_terminal']]
#         .join(factordefs.gen_factor_data, on='id', how='inner'))
#     if termfactor.empty:
#         return pd.DataFrame(
#             [], 
#             columns=['ftaps'], 
#             dtype=np.float64, 
#             index=pd.Index([], dtype=np.int64, name='index_of_terminal'))
#     return pd.DataFrame(
#         # mx + n
#         ((termfactor.m * positions) + termfactor.n).array,
#         columns=['ftaps'],
#         index=termfactor.index_of_terminal)

# def _calculate_f_mn_tot(index_of_other_terminal, term_factor):
#     """Calculates terminal factors (taps factors) for each terminal.

#     Parameters
#     ----------
#     index_of_other_terminal: pandas.DataFrame (index_of_terminal) ->
#         * .index_of_other_terminal, int
#     term_factor: pandas.DataFrame (index_of_terminal) ->
#         * .ftaps, float, taps-factor

#     Returns
#     -------
#     numpy.array, float (shape n,2)"""
#     f = (
#         index_of_other_terminal
#         .join(term_factor, how='left')
#         .set_index('index_of_other_terminal')
#         .join(term_factor, how='left', rsuffix='_other')
#         .fillna(1.))
#     return f.to_numpy()[:,[1,0]] * f.ftaps.to_numpy()

# def create_gb_of_terminals_n(branchterminals, term_factor):
#     """Creates a vectors (as a numpy array) of branch-susceptances and
#     branch-conductances.
#     The intended use is calculating a subset of terminal values.
#     Arguments 'branchtaps' and 'positions' will be selected
#     accordingly, hence, it is appropriate to pass the complete branchtaps
#     and positions.

#     Parameters
#     ----------
#     branchterminals: pandas.DataFrame (index_of_terminal)
#         * .g_lo, float, longitudinal conductance
#         * .b_lo, float, longitudinal susceptance
#         * .g_tr_half, float, transversal conductance
#         * .b_tr_half, float, transversal susceptance
#     term_factor: pandas.DataFrame (index_of_terminal) ->
#         * .ftaps, float

#     Returns
#     -------
#     numpy.array (shape n,4)
#         gb_mn_tot[:,0] - g_mn
#         gb_mn_tot[:,1] - b_mn
#         gb_mn_tot[:,2] - g_tot
#         gb_mn_tot[:,3] - b_tot"""
#     # g_lo, b_lo, g_trans, b_trans
#     gb_mn_tot = (
#         branchterminals
#         .loc[:,['g_lo', 'b_lo', 'g_tr_half', 'b_tr_half']]
#         .to_numpy())
#     # gb_mn_mm -> gb_mn_tot
#     gb_mn_tot[:, 2:] += gb_mn_tot[:, :2]
#     f_mn_tot = _calculate_f_mn_tot(
#         branchterminals[['index_of_other_terminal']], term_factor)
#     gb_mn_tot[:, :2] *= f_mn_tot[:,[0]]
#     gb_mn_tot[:, 2:] *= f_mn_tot[:,[1]]
#     return gb_mn_tot
