# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:49:05 2022

@author: pyprg
"""
import casadi
import numpy as np
from collections import namedtuple
from injections import get_polynomial_coefficients

# square of voltage magnitude, minimum value for load curve, 
#   if value is below _VMINSQR the load curves for P and Q converge
#   towards a linear load curve which is 0 when V=0; P(V=0)=0, Q(V=0)=0
_VMINSQR = 0.8
# value of zero check, used for load curve calculation    
_EPSILON = 1e-12

Vvar = namedtuple(
    'Vvar',
    're im reim node_sqr decvars slack')

def get_tap_factors(branchtaps, pos):
    """Creates vars for tap positions, expressions for longitudinal and
    transversal factors of branches.
    
    Parameters
    ----------
    branchtaps: pandas.DataFrame (id of taps)
        * .Vstep, float voltage diff per tap
        * .positionneutral, int
    pos: casadi.SX
        vector of positions for terms with tap
    
    Returns
    -------
    tuple
        * casadi.SX, longitudinal factors
        * transversal factors"""
    # factor longitudinal
    if pos.size1():
        flo = (1 - branchtaps.Vstep.to_numpy() * (
            pos - branchtaps.positionneutral.to_numpy()))
    else:
        flo = casadi.SX(0, 1)
    return flo, casadi.constpow(flo, 2)

def create_gb(terms, count_of_nodes, flo, ftr):
    """Generates a conductance-susceptance matrix of branches equivalent to
    branch-admittance matrix. M[n,n] of slack nodes is set to 1, other
    values of slack nodes are zero.
    
    Parameters
    ----------
    terms: pandas.DataFrame
    
    count_of_nodes: int
        number of power flow calculation nodes
    flo: casadi.SX, vector
        longitudinal taps factor, sparse for terminals with taps
    ftr: casadi.SX, vector
        transversal taps factor, sparse for terminals with taps
    
    Returns
    -------
    tuple
        * casadi.SX, sparse matrix of branch conductances G
        * casadi.SX, sparse matrix of branch susceptances B"""
    terms_with_taps = terms[terms.index_of_taps.notna()]
    idx_of_tap = terms_with_taps.index_of_taps
    # y_tot
    g_mm = casadi.SX(terms.g_tr_half)
    b_mm = casadi.SX(terms.b_tr_half)
    g_mm[terms_with_taps.index] *= ftr[idx_of_tap]
    b_mm[terms_with_taps.index] *= ftr[idx_of_tap]
    # y_mn
    g_mn = casadi.SX(terms.g_lo)
    b_mn = casadi.SX(terms.b_lo)
    g_mn[terms_with_taps.index] *= flo[idx_of_tap]
    b_mn[terms_with_taps.index] *= flo[idx_of_tap]
    terms_with_other_taps = terms[terms.index_of_other_taps.notna()]
    idx_of_other_tap = terms_with_other_taps.index_of_other_taps
    g_mn[terms_with_other_taps.index] *= flo[idx_of_other_tap]
    b_mn[terms_with_other_taps.index] *= flo[idx_of_other_tap]
    # Y
    G = casadi.SX(count_of_nodes, count_of_nodes)
    B = casadi.SX(count_of_nodes, count_of_nodes)
    for data_idx, idxs in \
        terms.loc[:, ['index_of_node', 'index_of_other_node']].iterrows():
        index_of_node = idxs.index_of_node
        index_of_other_node = idxs.index_of_other_node
        G[index_of_node, index_of_node] += (g_mn[data_idx] + g_mm[data_idx])
        G[index_of_node, index_of_other_node] -= g_mn[data_idx]
        B[index_of_node, index_of_node] += (b_mn[data_idx] + b_mm[data_idx])
        B[index_of_node, index_of_other_node] -= b_mn[data_idx]
    return G, B

def create_gb_matrix(model, pos):
    """Generates a conductance-susceptance matrix of branches equivalent to
    branch-admittance matrix. M[n,n] of slack nodes is set to 1, other
    values of slack nodes are zero. Hence, the returned 
    matrix is unsymmetrical.
    
    Parameters
    ----------
    model: egrid.model.Model
    
    pos: casadi.SX
        vector of position variables, one variable for each terminal with taps
    
    Returns
    -------
    casadi.SX"""
    flo, ftr = get_tap_factors(model.branchtaps, pos)
    count_of_nodes = model.shape_of_Y[0]
    terms = (
        model.branchterminals[~model.branchterminals.is_bridge].reset_index())
    G, B = create_gb(terms, count_of_nodes, flo, ftr)
    count_of_slacks = model.count_of_slacks
    diag = casadi.Sparsity.diag(count_of_slacks, count_of_nodes)
    G_ = casadi.vertcat(
        diag, 
        G[count_of_slacks:, :])
    B_ = casadi.vertcat(
        casadi.SX(count_of_slacks, count_of_nodes), 
        B[count_of_slacks:, :])
    return  casadi.blockcat([[G_, -B_], [B_,  G_]])

def create_branch_gb_matrix2(model, pos):
    """Generates a conductance-susceptance matrix of branches equivalent to
    branch-admittance matrix. Removes slack rows and columns
    The returned matrix is symmetric. Additionally returns columns of slacks.
    
    Parameters
    ----------
    model: egrid.model.Model
    
    pos: casadi.SX
        vector of position variables, one variable for each terminal with taps
    
    Returns
    -------
    tuple
        * casadi.SX, gb-matrix,
        * casadi.SX, g-columns of slacks
        * casadi.SX, b-columns of slacks"""
    flo, ftr = get_tap_factors(model.branchtaps, pos)
    terms = (
        model.branchterminals[~model.branchterminals.is_bridge].reset_index())
    G, B = create_gb(model, terms, model.shape_of_Y[0], flo, ftr)
    count_of_slacks = model.count_of_slacks
    G_ = G[count_of_slacks:, count_of_slacks:]
    B_ = B[count_of_slacks:, count_of_slacks:]
    return  (
        casadi.blockcat([[G_, -B_], [B_,  G_]]), 
        G[count_of_slacks:, :count_of_slacks],
        B[count_of_slacks:, :count_of_slacks])

_no_slacks = casadi.DM(0,2)

def create_Vvars(count_of_nodes, slacks=_no_slacks):
    """Creates casadi.SX for voltages.
    
    Parameters
    ----------
    count_of_nodes: int
        number of pfc-nodes
    
    Returns
    -------
    Vvar
        * .re
        * .im
        * .reim
        * .node_sqr"""
    count_of_slacks = slacks.size1()
    count = count_of_nodes - count_of_slacks
    Vre_dec = casadi.SX.sym('Vre', count)
    Vim_dec = casadi.SX.sym('Vim', count)
    Vre_slack = slacks[:, 0]
    Vim_slack = slacks[:, 1]
    Vre = casadi.vertcat(Vre_slack, Vre_dec)
    Vim = casadi.vertcat(Vim_slack, Vim_dec)
    Vreim = casadi.vertcat(Vre, Vim)
    Vnode_sqr = casadi.power(Vre, 2) + casadi.power(Vim, 2)
    return Vvar(
        re=Vre,
        im=Vim,
        reim=Vreim,
        node_sqr=Vnode_sqr,
        decvars=casadi.vertcat(Vre_dec, Vim_dec),
        slack=casadi.vertcat(Vre_slack, Vim_slack))

def get_injected_squared_current(V, Vinj_sqr, Mnodeinj, P10, Q10):
    """Calculates current flowing into injections. Returns separate
    real and imaginary parts.
    Injected power is calculated this way
    (P = |V|**2 * P10, Q = |V|**2 * Q10; with |V| - magnitude of V):
    ::
        +- -+   +-                           -+
        | P |   | (Vre ** 2 + Vim ** 2) * P10 |
        |   | = |                             |
        | Q |   | (Vre ** 2 + Vim ** 2) * Q10 |
        +- -+   +-                           -+

    How to calculate current from given complex power and from voltage:
    ::
                S_conjugate
        I_inj = -----------
                V_conjugate

    How to calculate current with separated real and imaginary parts:
    ::
                              +-        -+ -1  +-    -+
                S_conjugate   |  Vre Vim |     |  P Q |
        I_inj = ----------- = |          |  *  |      |
                V_conjugate   | -Vim Vre |     | -Q P |
                              +-        -+     +-    -+

                                  +-        -+   +-    -+
                       1          | Vre -Vim |   |  P Q |
              = --------------- * |          | * |      |
                Vre**2 + Vim**2   | Vim  Vre |   | -Q P |
                                  +-        -+   +-    -+

                                   +-              -+
                          1        |  P Vre + Q Vim |
        I_inj_ri = --------------- |                |
                   Vre**2 + Vim**2 | -Q Vre + P Vim |
                                   +-              -+

    How to calculate injected real and imaginary current from voltage:
    ::
        Ire =  P10 * Vre + Q10 * Vim
        Iim = -Q10 * Vre + P10 * Vim

    Parameters
    ----------
    V: Vvar
        casadi.SX vectors for node voltages
    Vinj_sqr: casadi.SX
        vector, voltage at injection squared
    Mnodeinj: casadi.SX
        matrix
    exp_v_p: numpy.array, float
        voltage exponents of active power per injection
    P10: numpy.array, float
        active power per injection
    exp_v_q: numpy.array, float
        voltage exponents of reactive power per injection
    Q10: numpy.array, float
        active power per injection
        
    Returns
    -------
    tuple
        * vector of injected current per node, real part
        * vector of injected current per node, imaginary part"""
    Gexpr_node = Mnodeinj @ P10
    Bexpr_node = Mnodeinj @ Q10
    Ire =  (Gexpr_node * V.re) + (Bexpr_node * V.im)
    Iim = (-Bexpr_node * V.re) + (Gexpr_node * V.im)
    return Ire, Iim

def get_injected_original_current(
        V, Vinj_sqr, Mnodeinj, exp_v_p, P10, exp_v_q, Q10):
    """Calculates current flowing into injections. Returns separate
    real and imaginary parts.
    Injected power is calculated this way
    (P = |V|**Exvp * P10, Q = |V|**Exvq * Q10; with |V| - magnitude of V):
    ::
        +- -+   +-                                          -+
        | P |   | (Vre ** 2 + Vim ** 2) ** (Expvp / 2) * P10 |
        |   | = |                                            |
        | Q |   | (Vre ** 2 + Vim ** 2) ** (Expvq / 2) * Q10 |
        +- -+   +-                                          -+

    How to calculate current from given complex power and from voltage:
    ::
                S_conjugate
        I_inj = -----------
                V_conjugate

    How to calculate current with separated real and imaginary parts:
    ::
                              +-        -+ -1  +-    -+
                S_conjugate   |  Vre Vim |     |  P Q |
        I_inj = ----------- = |          |  *  |      |
                V_conjugate   | -Vim Vre |     | -Q P |
                              +-        -+     +-    -+

                                  +-        -+   +-    -+
                       1          | Vre -Vim |   |  P Q |
              = --------------- * |          | * |      |
                Vre**2 + Vim**2   | Vim  Vre |   | -Q P |
                                  +-        -+   +-    -+

                                   +-              -+
                          1        |  P Vre + Q Vim |
        I_inj_ri = --------------- |                |
                   Vre**2 + Vim**2 | -Q Vre + P Vim |
                                   +-              -+

    How to calculate injected real and imaginary current from voltage:
    ::
        Ire =  (Vre ** 2 + Vim ** 2) ** (Expvp / 2 - 1) * P10 * Vre
             + (Vre ** 2 + Vim ** 2) ** (Expvq / 2 - 1) * Q10 * Vim

        Iim = -(Vre ** 2 + Vim ** 2) ** (Expvq / 2 - 1) * Q10 * Vre
             + (Vre ** 2 + Vim ** 2) ** (Expvp / 2 - 1) * P10 * Vim

    Parameters
    ----------
    V: Vvar
        casadi.SX vectors for node voltages
    Vinj_sqr: casadi.SX
        vector, voltage at injection squared
    Mnodeinj: casadi.SX
        matrix
    exp_v_p: numpy.array, float
        voltage exponents of active power per injection
    P10: numpy.array, float
        active power per injection
    exp_v_q: numpy.array, float
        voltage exponents of reactive power per injection
    Q10: numpy.array, float
        active power per injection
        
    Returns
    -------
    tuple
        * vector of injected current per node, real part
        * vector of injected current per node, imaginary part"""
    Gexpr_node = Mnodeinj @ (casadi.power(Vinj_sqr, exp_v_p/2 - 1) * P10)
    Bexpr_node = Mnodeinj @ (casadi.power(Vinj_sqr, exp_v_q/2 - 1) * Q10)
    Ire =  Gexpr_node * V.re + Bexpr_node * V.im
    Iim = -Bexpr_node * V.re + Gexpr_node * V.im
    return Ire, Iim

def get_injected_interpolated_current(
        V, Vinj_sqr, Mnodeinj, exp_v_p, P10, exp_v_q, Q10):
    """Interpolates injected current for V < _VMINSQR. Calculates values
    per node.
    
    Parameters
    ----------
    V: Vvar
        casadi.SX vectors for node voltages
    Vinj_sqr: casadi.SX
        vector, voltage at injection squared
    Mnodeinj: casadi.SX
        matrix
    exp_v_p: numpy.array, float
        voltage exponents of active power per injection
    P10: numpy.array, float
        active power per injection
    exp_v_q: numpy.array, float
        voltage exponents of reactive power per injection
    Q10: numpy.array, float
        active power per injection
        
    Returns
    -------
    tuple
        * injected current per node, real part
        * injected current per node, imaginary part"""
    # per injection
    Vinj = casadi.power(Vinj_sqr, 0.5)
    Vinj_cub = Vinj_sqr * Vinj
    pc = get_polynomial_coefficients(_VMINSQR, exp_v_p)
    qc = get_polynomial_coefficients(_VMINSQR, exp_v_q)
    fpinj = (pc[:,0]*Vinj_cub + pc[:,1]*Vinj_sqr + pc[:,2]*Vinj)
    fqinj = (qc[:,0]*Vinj_cub + qc[:,1]*Vinj_sqr + qc[:,2]*Vinj) 
    # per node
    pnodeexpr = Mnodeinj @ (fpinj * P10)
    qnodeexpr = Mnodeinj @ (fqinj * Q10)
    gt_zero = _EPSILON < V.node_sqr
    Ire_ip = casadi.if_else(
        gt_zero, (pnodeexpr * V.re + qnodeexpr * V.im) / V.node_sqr, 0.0)
    Iim_ip = casadi.if_else(
        gt_zero, (-qnodeexpr * V.re + pnodeexpr * V.im) / V.node_sqr, 0.0)
    return Ire_ip, Iim_ip

def get_injected_current(matrix_nodeinj, V, injections, 
                         pq_factors=None, loadcurve='original'):
    """Creates a vector of injected node current.
    
    Parameters
    ----------
    Mnodeinj: matrix
        
    V: Vvar
        * .re, casadi.SX, vector, real part of node voltage
        * .im, casadi.SX, vector, imaginary part of node voltage
        * .reim (not used)
        * .node_sqr, casadi.SX, vector, .re**2  + .im**2
    injections: pandas.DataFrame
        * .P10, float, active power at |V| = 1.0 pu, sum of all 3 phases
        * .Q10, float, reactive power at |V| = 1.0 pu, sum of all 3 phases
        * .Exp_v_p, float, voltage exponent of active power
        * .Exp_v_q, float, voltage exponent of reactive power
    pq_factors: numpy.array, float, (nx2)
        factors for active and reactive power of loads
    loadcurve: 'original' | 'interpolated' | 'square'
        default is 'original', just first letter is used

    Returns
    -------
    casadi.SX
        vector, injected current per node (real parts, imaginary parts),
        shape (2*count_of_nodes, 1)"""
    control_character = loadcurve[:1].lower()
    P10 = injections.P10 / 3 # for one phase only
    Q10 = injections.Q10 / 3 # for one phase only
    Mnodeinj = casadi.SX(matrix_nodeinj)
    Vinj_sqr = casadi.transpose(Mnodeinj) @ V.node_sqr # V**2 per injection
    if not pq_factors is None:
        P10 *= pq_factors[:,0]
        Q10 *= pq_factors[:,1]
    control_character = loadcurve[:1].lower()
    if control_character == 's':
        return get_injected_squared_current(V, Vinj_sqr, Mnodeinj, P10, Q10)         
    exp_v_p = injections.Exp_v_p.copy()
    exp_v_q = injections.Exp_v_q.copy()
    Ire, Iim = get_injected_original_current(
        V, Vinj_sqr, Mnodeinj, exp_v_p, P10, exp_v_q, Q10)
    if control_character == 'o':
        return Ire, Iim
    Ire_ip, Iim_ip = get_injected_interpolated_current(
        V, Vinj_sqr, Mnodeinj, exp_v_p, P10, exp_v_q, Q10)
    # compose functions from original and interpolated
    interpolate = V.node_sqr < _VMINSQR
    Inode_re = casadi.if_else(interpolate, Ire_ip, Ire)
    Inode_im = casadi.if_else(interpolate, Iim_ip, Iim)
    return Inode_re, Inode_im    

def build_injected_current_fn(model, pq_factors=None, loadcurve='original'):
    """Creates a function for calculating the injected current per node.
    
    Parameters
    ----------
    model: egrid.model.Model
    
    pq_factors: numpy.array, float, (nx2)
        factors for active and reactive power of loads
    loadcurve: 'original' | 'interpolated' | 'square'
        default is 'original', just first letter is used
    
    Returns
    -------
    casadi.Function
        (casadi.SX)->(casadi.DM)
        (vector_of_Vnode_ri)->(vector_of_Inode_inj_ri)"""
    # variables of voltages
    count_of_nodes = model.shape_of_Y[0]
    V = create_Vvars(count_of_nodes)
    # injected current per node
    Inode_re, Inode_im = get_injected_current(
        model.mnodeinj, V, model.injections, pq_factors, loadcurve)
    Inode = casadi.vertcat(Inode_re, Inode_im)
    return casadi.Function('fn_Inode_inj_ri', [V.reim], [Inode])

def build_residual_fn(model, pq_factors=None, loadcurve='original'):
    """Creates function for calculating the residual node current. The
    returned function can be used for root-finding.
    
    Parameters
    ----------
    model: egrid.model.Model
    
    pq_factors: numpy.array, float, (nx2)
        factors for active and reactive power of loads
    loadcurve: 'original' | 'interpolated' | 'square'
        default is 'original', just first letter is used
    
    Returns
    -------
    casadi.Function"""
    count_of_branch_taps = len(model.branchtaps)
    pos = casadi.SX.sym('pos', count_of_branch_taps)
    # branch gb-matrix, g:conductance, b:susceptance
    gb = create_gb_matrix(model, pos)
    # variables of voltages
    count_of_nodes = gb.size1() // 2
    V = create_Vvars(count_of_nodes)
    # injected current
    injections = model.injections
    Inode_re, Inode_im = get_injected_current(
        model.mnodeinj, V, injections[~injections.is_slack], 
        pq_factors, loadcurve)
    vslack_var = casadi.SX.sym('Vslack', len(model.slacks), 2)# 0:real, 1:imag
    # modify Inode of slacks
    index_of_slack = model.slacks.index_of_node
    Inode_re[index_of_slack] = -vslack_var[:, 0]
    Inode_im[index_of_slack] = -vslack_var[:, 1]
    # equation of node current
    Ires = (gb @ V.reim) + casadi.vertcat(Inode_re, Inode_im)
    # parameters, horzcat returns wrong shape for count_of_branch_taps==0
    param = (casadi.horzcat(vslack_var[:, 0], vslack_var[:, 1], pos) 
             if count_of_branch_taps else 
             casadi.horzcat(vslack_var[:, 0], vslack_var[:, 1]))
    # create node current function
    return casadi.Function('fn_Iresidual', [V.reim, param], [Ires])

def build_objective(model, gb_matrix, V, count_of_slacks, pq_factors=None):
    """Creates expression for solving the power flow problem by minimization.
    
    Parameters
    ----------
    model: egrid.model.Model
    gb_matrix: casadi.SX
        sparse branch-matrix, g:conductance, b:susceptance
    V: Vvar
        decision variables, voltage vectors for real and imaginary parts of
        node voltages
    count_of_slacks: int
        number of slack-nodes
    pq_factors: numpy.array, float, (nx2)
        factors for active and reactive power of loads
    
    Returns
    -------
    casadi.SX, expression to be minimized"""
    # injected current
    injections = model.injections
    Inode_re, Inode_im = get_injected_current(
        model.mnodeinj, V, injections[~injections.is_slack], pq_factors)
    # equation for root finding
    I = casadi.vertcat(Inode_re, Inode_im)[count_of_slacks:, :]
    Ires = ((gb_matrix[count_of_slacks:, :] @ V.reim) + I)
    return casadi.norm_2(Ires)
    
def find_root(
        fn_Iresidual, tappositions, Vslack, count_of_nodes=0, Vinit=None):
    """Finds root of fn_Iresidual.
    
    Parameters
    ----------
    fn_Iresidual: casadi.Function
        function to find a root for
    tappositions: numpy.array, int
        positions of taps
    Vslack: array_like, shape (n, 1) of complex
        values for slack voltages (parameter), n: number of slacks
    count_of_nodes: int
        number of nodes, defaults to 0
    Vinit: array_like, shape (n, 1) of complex
        vector of initial complex node voltages
        n: number of nodes, defaults to None
    
    Returns
    -------
    tuple
        * success?, bool
        * casadi.DM, voltage vector of floats, shape (2n,1), 
          n real parts followed by n imaginary parts"""
    Vinit_ = (
        np.vstack([np.real(Vinit), np.imag(Vinit)]) 
        if not Vinit is None else
        np.array([1.]*count_of_nodes + [0.]*count_of_nodes).reshape(-1, 1))
    values_of_params = casadi.horzcat(
        np.real(Vslack), np.imag(Vslack), tappositions)
    rf = casadi.rootfinder('rf', 'nlpsol', fn_Iresidual, {'nlpsol':'ipopt'})
    #rf = casadi.rootfinder('rf', 'newton', fn_Iresidual)
    try:
        return True, rf(Vinit_, values_of_params)
    except:
        return False, casadi.DM(Vinit_)
    
def calculate_power_flow(
        precision, max_iter, model, 
        Vslack=None, tappositions=None, Vinit=None, 
        pq_factors=None, loadcurve='original'):
    """Power flow calculating function.
    
    Parameters
    ----------
    precision: float (not used)
        tolerance for node current
    max_iter: int (not used)
        limit of iteration count
    model: egrid.model.Model
    
    Vslack: array_like, complex
        vector of voltages at slacks, default model.slacks.V
    tappositions: array_like, int
        vector of tap positions, default model.branchtaps.position
    Vinit: array_like, float
        start value of iteration, node voltage vector, 
        real parts then imaginary parts
    pq_factors: numpy.array, float, (nx2)
        factors for active and reactive power of loads
    loadcurve: 'original' | 'interpolated' | 'square'
        default is 'original', just first letter is used
    
    Returns
    -------
    tuple
        * bool, success?
        * array_like, float, node voltages, real parts then imaginary parts"""
    Vslack_ = model.slacks.V if Vslack is None else Vslack
    tappositions_ = model.branchtaps.position.copy() \
        if tappositions is None else tappositions
    fn_Iresidual = build_residual_fn(model, pq_factors, loadcurve)
    success, voltages = find_root(
        fn_Iresidual, tappositions_, Vslack_, 
        count_of_nodes=model.shape_of_Y[0], Vinit=Vinit)
    return success, np.hstack(np.vsplit(voltages, 2)).view(dtype=np.complex128)
    
def eval_residual_current(
        model, pq_factors=None, loadcurve=None, 
        tappositions=None, Vslack=None, V=None):
    """Function for evaluating a power flow solution. Calculates
    the complex residual node current. 
    Not intended to be used inside a solver loop.
    
    Parameters
    ----------
    model: egrid.model.Model
    
    pq_factors: numpy.array
        float
    loadcurve: 'original' | 'interpolated' | 'square'
    
    tappositions: numyp.array
        int
    Vslack: array_like
        complex
    V: numpy.array
        complex
    
    Returns
    -------
    numpy.array
        complex"""
    tappositions_ = (
        model.branchtaps.position.copy()
        if tappositions is None else tappositions)
    Vslack_ = model.slacks.V if Vslack is None else Vslack
    values_of_params = casadi.horzcat(
        np.real(Vslack_), np.imag(Vslack_), tappositions_)
    Vri = V.view(dtype=np.float64).T.reshape(-1)
    fn_Iresidual = build_residual_fn(model, pq_factors, loadcurve)
    res = fn_Iresidual(Vri, values_of_params)
    return np.hstack(np.split(np.array(res), 2)).view(dtype=np.complex128)
