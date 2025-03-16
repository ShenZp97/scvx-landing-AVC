import numpy as np
# Trajectory points
K_3dof = 20

# Max solver iterations
iterations = 50

epsilon_vc = 1e-3
epsilon_tr = 1e-5

solver = ['ECOS', 'MOSEK', 'GUROBI', 'SCS'][1]
verbose_solver = False

weight_nu = 10
w_tr = np.diag([0.001, 0.001, 0.001])

# weight_nu = 1e-8
# w_tr = np.diag([0.01, 0.01, 0.01])

# Weight constants
# w_nu = 1e6  # virtual control
# w_f = 10
# w_tr = 0.01 * np.ones((K, ))
# mass

# delta_x_tol = 0.01
# epsilon_vc = 5e-4
# epsilon_tr = 5e-4
# initial trust region radius
# tr_radius = 5

# trust region variables
# rho_0 = 0.0
# rho_1 = 0.25
# rho_2 = 0.9
# alpha = 2.0
# beta = 3.2
