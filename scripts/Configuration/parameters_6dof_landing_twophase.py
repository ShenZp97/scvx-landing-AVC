import cvxpy as cp
import numpy as np
import os
# Trajectory points
K = 20

# Recorded AVC trajectory used to build the time-varying landing cone. Columns
# 1 to 4 are read as time, x, y and z.
carrier_traj_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 os.pardir, os.pardir, 'data', 'loop_traj_platform_new.csv')

# Max solver iterations
iterations = 10

epsilon_vc = 1e-4
epsilon_tr = 1e-2
# epsilon_tr = 0.5

# epsilon_tr = 1e-3

solver = [cp.ECOS, cp.MOSEK, cp.GUROBI, cp.SCS][1]
# verbose_solver = False

# Obstacle
# obs_num = 1
# R1 = 1
#
# H1 = np.diag([1., 1., 1.])
# p1 = np.array((3., 3.2, 2.5))

weight_nu = 1e4
# weight_nu = 1e2
# w_tr = np.diag([0.1, 0.1, 0.1])
# w_tr = np.diag([0.1, 0.1, 0.1, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
# w_tr = np.diag([0.1, 0.1, 0.1, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])

weight_dx = 10

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
