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

## Kick Start

__dssex.print_powerflow(*args)__ calculates power flow for a given network
and prints the results. If args are not provided the function calculates
the power flow for a simple configuration, prints this simple configuration
and the results.

## Abstract Network Elements

**dssex** processes two types of network devices and connectivity nodes.

Transformers and lines share the PI-equivalent circuit. Both are mapped to the
**Branch** type. The same is true for series capacitors.

Loads, PV/PQ-generators, batteries, and shunt capacitors are instances of
**Injection**. Injections have attributes P10 and Q10 for active and reactive
power at voltage of 1.0&nbsp;pu and Exp_v_p and Exp_v_q for modeling the
voltage dependency. The voltage exponents of the injection instances Exp_v_p,
Exp_v_q are set to 0 for PQ-generators, constant and measured loads. Those
injections and the injections of shunt capacitors are not scaled. Injections
of shunt capacitors have an active power setting P10 of 0 and a negative
reactive power setting Q10. Its voltage exponent for reactive power Exp_v_q
is 2 (constant reactance). Injections of loads can have different voltage
exponents and can be scaled during estimation.

The optimization problem is setup using instance of:

    * branch (with one or two taps) (line, transformer (winding), series capacitor)
    * injection (consumer, generator, shunt capacitor, battery)
    * node (given by reference id from other elements only)
    * slack node, marker of a node, source of given complex voltage
    * given flow value (PQI-measurements, real value)
    * given node value (measurement or setpoint of voltage, real value)

As there are separate decision variables for real and imaginary parts of
node voltages and internally build expressions of real and imaginary parts
for currents of both device abstractions (branches and injections) using
**phasor measurements** would be a natural fit. However, the focus was up to
now including the more difficult to process but widely-used measurement
values of V/I magnitude. Hence, including phasor measurements is not yet done.

## Math Approach

IPOPT is an Interior Point solver which minimizes a scalar-valued objective
function with respect to multiple constraints of different types
(equality, smaller than, ...).

The estimation goal is to scale injections in order to meet the measured
values or setpoints as accurately as possible. **dssex** creates a
'non-linear program' consisting of constraint formulations and an
objective function.

The objective function is the sum of terms having the structure
(measured_value_or_setpoint - calculated_value)\*\*2. The search is
an iterative process requiring a start value. A part of the constraints
are the power flow relations. The sum of current flowing through branches
into a node equals the injected current&nbsp;- except for slacks
(current injection method).

    Y * V - I = 0          Y - complex branch admittance matrix
                           V - complex node voltage vector
                           I - complex node current vector

(Scaling) factors and node voltages are the results of the optimization
process. Variables for voltages and factors are named decision variables.
The injected node current I is expressed in terms of the node voltages and
scaling factors. The objective function is also expressed in decision variables
and attributes of the devices and the topology. P_calculated in the term of an
active power measurement at the terminals of a branch
(P_measured&nbsp;-&nbsp;P_calculated)\*\*2, for example, is expressed in
decision variables of the node voltages and admittance values of the branch.

Initial node voltages for the optimization process are currently calculated
by a rootfinding function created and solved by CasADi (using IPOPT).

One of the goals is to **avoid weighting factors** and not to trade off e.g.
meeting power measurement values for meeting voltage measurements or
vice versa. Selected sets of weighting factors tend to fit for a particular
configuration only and selecting requires some kind of magic. Additionally
they introduce some tradeoffs which hopefully can be avoided when using
multiple optimization steps. Therefore, the complete estimation process of
this implementation can consist of several minimization steps.

Each step has a specific objective funtion. E.g. after opimizing towards
P/Q measurements the next step might optimize to meet the current/voltage
measurements. P and Q values of the first step can be
fixed by adding constraints accordingly. A separate handling of flow and
voltage measurements and optimization of additional criteria is thus possible.

The split avoids numeric problems created by different magnitudes of
voltages, powers and currents.

Function **estim.estimate_stepwise** implements described method.

## Separate Real and Imaginary Parts

The non-linear solver and the CasADi-package do not support complex numbers.
Hence, complex values are calculated with separate real and imaginary parts.
Each complex number is transformed in a 2x2-matrix:

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

The separate processing of real and imaginary parts in the described manner
enables a straight forward integration of phasor measurements in the future.

## Numeric Power Flow Calculation and Result Processing

Function **pfcnum.calculate_power_flow** solves the non-linear
power-flow-problem. It is an experimental implementation deploying the
schema of separated real and imaginary parts of complex values internally while
accepting an egrid.model.Model as input and returning a vector of
complex voltages.

Function **pfcnum.calculate_results** accepts an egrid.model.Model and
a voltage vector as input and calculates the power and current flow for
branches and injections as well as losses for branches. Function
**calculate_residual_current** is a tool for checking the obtained result
numerically.

## Volt-Var-Control (VVC)

Features added for studying of Volt-Var-Control (VVC) problems:

    * discrete decision variables (shunt capacitors)
    * decision variables for tap positions (also discrete)
    * minimum and maximum limits for absolute voltages at nodes
    * expressions for power losses of branches used in objective function
    * cost for active and reactive power (e.g. import/generation)
    * cost for change of factors (e.g. position of taps)

Control variables of VVC could be discrete or continuous values for active and
reactive power of injections (shunt capacitors with taps, batteries,
generators, PV-units, ...) and load-tap-changers of transformers.

The VVC-Algorithm also benefits from the described multi step approach when
for instance initially resolving voltage violations and then minimizing costs.
