# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:49:05 2022

@author: pyprg
"""
import casadi
import numpy as np
from egrid import make_model
from egrid.builder import Slacknode, Branch, Injection, Branchtaps
from scipy.sparse import coo_matrix
from collections import namedtuple

Branchdata = namedtuple(
    'Branchdata',
    'pos branchterminals g_tot b_tot g_mn b_mn count_of_nodes GB')

def create_branch_gb_matrix(model):
    """Generates a conductance-susceptance matrix of branches equivalent to
    branch-admittance matrix."""
    branchtaps = model.branchtaps
    pos = casadi.SX.sym('pos', len(branchtaps), 1)
    # factor longitudinal
    flo = 1 - branchtaps.Vstep.to_numpy() * (
        pos - branchtaps.positionneutral.to_numpy())
    # factor transversal
    ftr = casadi.constpow(flo, 2)
    #branchterminals
    terms = (
        model.branchterminals[~model.branchterminals.is_bridge].reset_index())
    terms_with_taps = terms[terms.index_of_taps.notna()]
    idx_of_tap = terms_with_taps.index_of_taps
    # y_tot
    g_tot = casadi.SX(terms.g_tot)
    b_tot = casadi.SX(terms.b_tot)
    g_tot[terms_with_taps.index] *= ftr[idx_of_tap]
    b_tot[terms_with_taps.index] *= ftr[idx_of_tap]
    # y_mn
    g_mn = casadi.SX(terms.g_mn)
    b_mn = casadi.SX(terms.b_mn)
    g_mn[terms_with_taps.index] *= flo[idx_of_tap]
    b_mn[terms_with_taps.index] *= flo[idx_of_tap]
    terms_with_other_taps = terms[terms.index_of_other_taps.notna()]
    idx_of_other_tap = terms_with_other_taps.index_of_other_taps
    g_mn[terms_with_other_taps.index] *= flo[idx_of_other_tap]
    b_mn[terms_with_other_taps.index] *= flo[idx_of_other_tap]
    # Y
    count_of_nodes = model.shape_of_Y[0]
    G = casadi.SX(count_of_nodes, count_of_nodes)
    B = casadi.SX(count_of_nodes, count_of_nodes)
    for data_idx, idxs in \
        terms.loc[:, ['index_of_node', 'index_of_other_node']].iterrows():
        index_of_node = idxs.index_of_node
        index_of_other_node = idxs.index_of_other_node
        G[index_of_node, index_of_node] += g_tot[data_idx]
        G[index_of_node, index_of_other_node] -= g_mn[data_idx]
        B[index_of_node, index_of_node] += b_tot[data_idx]
        B[index_of_node, index_of_other_node] -= b_mn[data_idx]
    GB = casadi.blockcat([[G, -B], [B,  G]])
    # remove rows/columns with GB.remove, 
    #   e.g. remove first row GB.remove([0],[])
    return Branchdata(
        pos=pos,
        branchterminals=terms,
        g_tot=g_tot,
        b_tot=b_tot,
        g_mn=g_mn,
        b_mn=b_mn,
        count_of_nodes=count_of_nodes,
        GB=GB)


def calculate(model, parameters_of_steps=(), tap_positions=()):
    
    
    branchdata = create_branch_gb_matrix(model)
    
    GBroot = casadi.SX(branchdata.GB) # copy
    # slack indices for gb-matrix
    slackidx = model.nodes[model.nodes.is_slack].index_of_node.to_list()
    slackidx.extend([2*idx for idx in slackidx])
    GBroot.remove(slackidx, [])
    
    
    vector_size = 2 * model.shape_of_Y[0]
    Vnode = casadi.SX.sym('Vnode', vector_size)
    BG = casadi.SX.sym('BG', vector_size, vector_size)
    Inode = casadi.SX.zeros(vector_size)
    return None


model_devices = [
    Slacknode(id_of_node='n_0', V=1.+0.j),
    Branch(
        id='line_0',
        id_of_node_A='n_0',
        id_of_node_B='n_1',
        y_mn=1e3-1e3j,
        y_mm_half=1e-6+1e-6j),
    Branch(
        id='line_1',
        id_of_node_A='n_1',
        id_of_node_B='n_2',
        y_mn=1e3-1e3j,
        y_mm_half=1e-6+1e-6j),
    Injection(
        id='consumer_0',
        id_of_node='n_2',
        P10=30.0,
        Q10=10.0,
        Exp_v_p=2.0,
        Exp_v_q=2.0)]
model00 = make_model(
    model_devices,
    Branchtaps(
        id='tap_1',
        id_of_branch='nix_1',
        id_of_node='n_0'),
    Branchtaps(
        id='tapLine_1',
        id_of_branch='line_1',
        id_of_node='n_1'),
    Branchtaps(
        id='tapLine_0',
        id_of_branch='line_0',
        id_of_node='n_1')
    )
# calculate power flow
results = calculate(model00)

