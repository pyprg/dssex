# dssex

## Experimental Distribution System State Estimator

Prototype of a balanced medium voltage network state estimator 
(distribution system state estimator - DSSE). The prototype is intended
for studying state estimation with tiny details of network configurations.

Values are processed 'per unit'.

## Domain

**dssex** focusses on problem formulation. Automatic Differentiation and 
control of numerical computation is provided by the CasADi package 
https://web.casadi.org/ which delegates solving of the non-linear programs to
IPOPT https://coin-or.github.io/Ipopt/ which in turn deploys MUMPS as a solver
for linear subproblems. IPOPT and MUMPS are installed with CasADi. 
(CasADi supports other solvers, including commercial ones.)
Pre- and post-processing is supported by Pandas https://pandas.pydata.org/.

Non-goals:
 
    * detection of faulty measurements 
    * performance

## Abstract Elements

**dssex** processes two types of network devices and connectivity nodes.

Transformers and lines share the PI-equivalent circuit. Both are mapped to the 
**Branch** type. The same is true for series capacitors.

Loads, PV/PQ-generators, batteries, and shunt capacitors are instances of 
**Injection**. Injections have attributes P10 and Q10 for active and reactive 
power at voltage of 1&nbsp;pu and Exp_v_p and Exp_v_q for modeling the voltage 
dependency. For PQ-generators, constant and measured loads the voltage
exponents of the injection instances Exp_v_p, Exp_v_q are set to 0. This 
injections and the injections of shunt capacitors are not scaled. Injections 
of shunt capacitors have an active power setting P10 of 0 and a negative 
reactive power setting Q10. Its voltage exponent for reactive power Exp_v_q 
is 2 (constant reactance). Injections of loads can have different voltage 
exponents and can be scaled during estimation.

The optimization problem is setup using instance of:

    * branch (with one or two taps) (line, transformer (winding), series capacitor)
    * injection (consumer, generator, shunt capacitor)
    * node (given by reference id from other elements only)
    * slack node, marker of a node, source of given complex voltage
    * given flow value (PQI-measurements, real value)
    * given node value (measurement or setpoint of voltage, real value)

## Math Aproach

IPOPT is an Inerior Point solver which minimizes a scalar-valued objective 
function with respect to multiple constraints of different types 
(equality, smaller than, ...). 

The estimation goal is to scale injections in order to meet the measured 
values or setpoints as accurately as possible. **dssex** creates a 
'non-linear program' consisting of constraint formulations and an 
objective function.

The objective funtion is the sum of terms having the structure
(measured_value_or_setpoint - calculated_value)\*\*2. The search is
an iterative process requiring a start value. A part of the constraints
are the power flow relations. For each node&nbsp;- without the slack&nbsp;-
is the sum of current flowing through branches into the node equal to the 
injected current at this node.

    Y * V - I = 0          Y - complex branch admittance matrix
                           V - complex node voltage vector
                           I - complex node current vector

The results of the optimization process are the node voltages and the
scaling factors. Variables for voltages and factors are named decision
variables. The node current I is expressed in terms of the node voltages
and scaling factors. The objective function is also expressed in decision 
variables and attributes of the devices and the topology. P_caluculated 
in the term of an active power measurement at the terminals of a branch
(P_measured - P_calculated)\*\*2, for example, is expressed in decision 
variables of the node voltages and admittance values of the branch.

Initial node voltages for the optimization process are currently calculated 
by a rootfinding function created and solved by CasADi (using IPOPT).

The complete estimation process can consist of several minimization steps. 
Each step has a specific objective funtion. E.g. after 
opimizing towards P/Q measurements the next step might optimize to meet the 
voltage measurements/setpoints. P and Q values of the first step can be 
fixed by adding constraints accordingly. A separate handling of flow and 
voltage measurements and optimization of additional criteria is thus possible. 
This split avoids numeric problems created by different magnitudes of 
voltages, powers and currents. Weighting factors shall be avoided. 
Including additional (consistent) measurements shall yield better results.

## Separate Real and Imaginary Part

The non-linear solver and the CasADi-package do not support complex numbers.
That is why the complex calculation is processed with separate real and
imaginary parts. Each complex number is transformed in a 2x2-matrix:

         +-      -+
         | Re -Im |          C  - complex number
    C -> |        |          Re - real part of complex number
         | Im  Re |          Im - imaginary part of complex number
         +-      -+

Multiplication of two complex number expressed as matrices is equivalent to
multiplication of two matrices.

    +-        -+   +-        -+   +-                                         -+
    | Yre -Yim |   | Vre -Vim |   | (Yre*Vre - Yim*Vim)  -(Yre*Vim + Yim*Vre) |
    |          | * |          | = |                                           |
    | Yim  Yre |   | Vim  Vre |   | (Yim*Vre + Yre*Vim)   (Yre*Vre - Yim*Vim) |
    +-        -+   +-        -+   +-                                         -+
