# dssex

Exprimental Distribution System State Estimator

Prototype of a balanced medium voltage network state estimator 
(distruibution system state estimator - DSSE). The prototype is intended
for studying state estimation with tiny details of network configurations.
The network is modeled with abstraction elements:

    * branch (with one or two taps) (line, transformer (winding), series capacitor)
    * injection (consumer, generator, shunt capacitor)
    * node (given by reference id from other elements only)
    * slack node, marker of a node, source of given complex voltage
    * given flow value (PQI-measurements, real value)
    * given node value (measurement or setpoint of voltage, real value)

The intention is to be able to model any topology and use measured/given
values associated to any terminal of the real network. The abstraction
shall provide all means to model any real (balanced) network. A transforming
process shall be possible to create the correct abstraction from data
of a real network in an easy way.

The estimation goal is to scale injections in order to meet the measured values
or setpoints as accurately as possible. A minimization problem is created 
which minimizes the deviation of measured and calculated values. The power 
flow equations, residual node currents, are (a part of) the constraints. 
Differences in the topologic quality (radial/meshed) of network shall not 
require an adaptation of the solving algorithm. Initial values are obtained 
from a power flow calculation. The complete estimation might have several 
minimization steps. Each step has a specific objective funtion. E.g. after 
opimizing towards P/Q measurements the next step might optimize to meet the 
voltage measurements/setpoints. P and Q values of the first step can be 
fixed by adding constraints accordingly. (Adaption of the steps is the 
topic of this study.) A separate handling of flow and voltage measurements 
and optimization of additional criteria is thus possible. This split avoids 
numeric problems created by different magnitudes of voltages, powers and 
currents. Weighting factors shall be avoided. Including additional 
(consistent) measurements shall yield better results.
Heart of the formula system is the power flow equation YV=I, 
with Y:(branch-)admittance-Matrix, V:node-voltages, I:node-current.
All complex values are processed with separated real and imaginary parts. The
node current I has in general a non-linear dependency upon the node voltage V.
Values are processed 'per unit'.
Variables of node-voltages and injection scaling factors build up the
network state enhanced by slack voltage and tap positions.
The terms of the objective function to be minimized are of the 
structure '(measured - calculated)**2'. Evaluation of calculated and 
measured P, Q and I requires the expression of the values with the state 
(NLP-term: decision) variables.  

The prototype focusses on problem formulation. Automatic Differentiation and 
control of numerical computation is provided by the CasADi package 
https://web.casadi.org/ which delegates solving of the non-linear programs to
IPOPT https://coin-or.github.io/Ipopt/ which in turn deploys MUMPS as a solver
for linear subproblems. IPOPT and MUMPS are installed with CasADi. 
(CasADi supports other solvers, including commercial ones.)
Pre- and post-processing is supported by Pandas https://pandas.pydata.org/.

Non-goals: 
    detection of faulty measurements, 
    performance
