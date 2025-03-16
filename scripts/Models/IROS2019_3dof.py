import sympy as sp
import numpy as np
import cvxpy as cvx
# import open3d as pcd
# from utils import euler_to_quat
from Configuration.parameters_iros2019 import K
# from Configuration.parameters_iros2019_hoop import K, rho_h, rho_c, rho_g, l_c, p1, n1

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.ticker as ticker

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class Model:
    """
    A 3 degree of freedom uav obstacle avoidance problem.
    """
    n_x = 6
    n_u = 3

    # Mass
    mass = 0.5  # kg

    # Flight time guess
    # tf_min = 1
    # tf_max = 8
    # t_f_guess = 3.  # s

    # State constraints
    r_I_init = np.array((1., 1., 1.))  # x m, y m, z m, z is altitude
    v_I_init = np.array((0., 0., 0.))  # x m/s, y m/s, z m/s
    # q_B_I_init = euler_to_quat((-30, 30, 0))
    # w_B_init = np.deg2rad(np.array((0., 0., 0.)))

    r_I_final = np.array((6., 6., 1.))
    v_I_final = np.array((0., 0., 0.))
    # q_B_I_final = euler_to_quat((0, 0, 0))
    # w_B_final = np.deg2rad(np.array((0., 0., 0.)))

    # Angles
    max_gimbal = 80
    # max_angle = 80
    # glidelslope_angle = 20
    #
    cos_theta_max = np.cos(np.deg2rad(max_gimbal))
    # cos_theta_max = np.cos(np.deg2rad(max_angle))
    # tan_gamma_gs = np.tan(np.deg2rad(glidelslope_angle))

    # Thrust limits
    T_max = 15.  # [kg*m/s^2]
    T_min = 3.

    # Angular moment of inertia
    # J_B = np.diag([4000000., 4000000., 100000.])  # 100000 [kg*m^2], 4000000 [kg*m^2], 4000000 [kg*m^2]

    # Gravity
    g_I = np.array((0., 0., -9.81))  # -9.81 [m/s^2]

    u_init = mass*np.array((0., 0., 9.81))
    u_final = mass*np.array((0., 0., 9.81))

    # Aerodynamic model
    # rho0 = 1.225  # [kg/m^3]
    # ca_x = 1.
    # ca_yz = 3.
    # S_A = 10
    # C_A = np.diag([ca_yz, ca_yz, ca_x])

    # Vector from thrust point to CoM
    # r_T_B = np.array([0., 0., -14.])  # -20 m
    # r_cp_B = np.array([0., 0., 2.])  # positive for static stable

    # Obstacle

    # ------------------------------------------ Start normalization stuff
    def __init__(self):
        """
        A large r_scale for a small scale problem will
        ead to numerical problems as parameters become excessively small
        and (it seems) precision is lost in the dynamics.
        """

        # self.set_random_initial_state()

        self.x_init = np.concatenate((self.r_I_init, self.v_I_init))
        self.x_final = np.concatenate((self.r_I_final, self.v_I_final))

        # slack variable for linear constraint relaxation
        # self.s_prime = cvx.Variable((K, 1), nonneg=True)
        self.sigma = cvx.Variable(K, nonneg=True)
        self.a_max = cvx.Parameter(nonneg=True)
        self.a_min = cvx.Parameter(nonneg=True)

        # slack variable for lossless convexification
        # self.gamma = cvx.Variable(K, nonneg=True)

    def get_equations(self):
        """
        :return: Functions to calculate A, B and f given state x and input u
        """
        f = sp.zeros(6, 1)

        x = sp.Matrix(sp.symbols('rx ry rz vx vy vz', real=True))
        u = sp.Matrix(sp.symbols('ux uy uz', real=True))

        g_I = sp.Matrix(self.g_I)
        # r_T_B = sp.Matrix(self.r_T_B)
        # r_cp_B = sp.Matrix(self.r_cp_B)
        # J_B = sp.Matrix(self.J_B)

        # C_B_I = dir_cosine(x[7:11, 0])
        # C_I_B = C_B_I.transpose()
        # C_A = sp.Matrix(self.C_A)

        # A_B = -0.5*self.rho0*(f[4:7, 0].norm())*self.S_A*C_A*C_B_I*f[4:7, 0]
        f[0:3, 0] = x[3:6, 0]
        f[3:6, 0] = ((1 / self.mass) * u) + g_I
        # f[7:11, 0] = 1 / 2 * omega(x[11:14, 0]) * x[7: 11, 0]
        # f[11:14, 0] = J_B ** -1 * (skew(r_T_B) * u + skew(r_cp_B) * A_B) - skew(x[11:14, 0]) * x[11:14, 0]

        f = sp.simplify(f)
        A = sp.simplify(f.jacobian(x))
        B = sp.simplify(f.jacobian(u))

        f_func = sp.lambdify((x, u), f, 'numpy')
        A_func = sp.lambdify((x, u), A, 'numpy')
        B_func = sp.lambdify((x, u), B, 'numpy')

        return f_func, A_func, B_func
        # return f_func

    def initialize_trajectory(self, X, U):
        """
        Initialize the trajectory.

        :param X: Numpy array of states to be initialized
        :param U: Numpy array of inputs to be initialized
        :return: The initialized X and U
        """

        for k in range(K):
            alpha1 = (K - k - 1) / (K-1)
            alpha2 = k / (K-1)

            r_I_k = alpha1 * self.x_init[0:3] + alpha2 * self.x_final[0:3]
            v_I_k = alpha1 * self.x_init[3:6] + alpha2 * self.x_final[3:6]
            # q_B_I_k = np.array([1, 0, 0, 0])
            # w_B_k = alpha1 * self.x_init[11:14] + alpha2 * self.x_final[11:14]

            X[:, k] = np.concatenate((r_I_k, v_I_k))
            U[:, k] = (self.T_max - 0) / 2 * np.array([0, 0, 1])

        return X, U


    def get_objective(self, X_v, U_v, X_last_p, U_last_p):
        """
        Get model specific objective to be minimized.

        :param X_v: cvx variable for current states
        :param U_v: cvx variable for current inputs
        :param X_last_p: cvx parameter for last states
        :param U_last_p: cvx parameter for last inputs
        :return: A cvx objective function.
        """

        # return cvx.Minimize(1e5 * cvx.sum(self.s_prime))
        return 0

    def get_constraints(self, X_v, U_v):
        """
        Get model specific constraints.

        :param X_v: cvx variable for current states
        :param U_v: cvx variable for current inputs
        :param X_last_p: cvx parameter for last states
        :param U_last_p: cvx parameter for last inputs
        :return: A list of cvx constraints
        """
        # Boundary conditions:
        constraints = [
            X_v[:, 0] == self.x_init[:],
            # X_v[7:11, 0] == self.x_init[7:11],
            # X_v[11:14, 0] == self.x_init[11:14],

            X_v[:, -1] == self.x_final[:],

            U_v[:, 0] == self.u_init,

            U_v[:, -1] == self.u_final,

            # - X_v[2, :] <= 0,
        ]

        self.a_max.value = self.T_max
        self.a_min.value = self.T_min
        constraints += [
            # State constraints:
            # cvx.norm(X_v[1: 3, :], axis=0) <= X_v[3, :] / self.tan_gamma_gs,  # glideslope
            # cvx.norm(X_v[8:10, :], axis=0) <= np.sqrt((1 - self.cos_theta_max) / 2),  # maximum angle
            # cvx.norm(X_v[11: 14, :], 'inf', axis=0) <= self.w_B_max,  # maximum angular velocity

            # Control constraints:
            self.cos_theta_max * self.sigma <= U_v[2, :],  # gimbal angle constraint
            # self.cos_delta_max * self.gamma <= U_v[2, :],

            cvx.norm(U_v, axis=0) <= self.sigma,  # upper thrust constraint
            self.sigma <= self.a_max,
            -self.sigma <= -self.a_min
            # U_v[2, :] >= self.T_min  # simple lower thrust constraint

            # # Lossless convexification:
            # self.gamma <= self.T_max,
            # self.T_min <= self.gamma,
            # cvx.norm(U_v, axis=0) <= self.gamma

        ]

        # linearized lower thrust constraint
        # lhs = [B_g[k, :] @ U_v[:, k] for k in range(K)]
        # constraints += [
        #     self.T_min - cvx.vstack(lhs) <= 0
        # ]

        # linearized obstacle avoidance
        # oac = [xi[k] + zeta[k, :] @ (X_v[0:3, k]) - zeta_dpp[k] for k in range(K)]
        # constraints += [
        #     self.R1 - xi[k] - zeta[k, :] @ (X_v[0:3, k]) + zeta_dpp[k] <= 0 for k in range(K)
        # ]
        return constraints

    # def get_linear_cost(self):
    #     cost = np.sum(self.s_prime.value)
    #     return cost
    #
    # def get_nonlinear_cost(self, X=None, U=None):
    #     magnitude = np.linalg.norm(U, 2, axis=0)
    #     is_violated = magnitude < self.T_min
    #     violation = self.T_min - magnitude
    #     cost = np.sum(is_violated * violation)
    #     return cost

    def plot_trajectory(self, all_X, all_U, all_sigma, X_nl_full, scene=0, label='Nonlinear open loop', color='xkcd:grey',
                        initial_label='Initial trajectory', initial_color='xkcd:grey'):

        fig_new = plt.figure(figsize=(6, 6))
        ax1 = fig_new.add_subplot(1, 1, 1, projection='3d')
        # new_X = all_X[-1, :, :]
        # new_U = all_U[-1, :, :]
        new_X = all_X[:, :]
        new_U = all_U[:, :]
        new_sigma = all_sigma
        # initial_X = all_X[0, :, :]
        # initial_U = all_U[0, :, :]

        ax1.plot(X_nl_full[0, :], X_nl_full[1, :], X_nl_full[2, :], label=label, color=color,
                 linestyle='-', linewidth=1)
        # ax1.plot(initial_X[0, :], initial_X[1, :], initial_X[2, :], label=initial_label, color=initial_color,
        #          linestyle='-', linewidth=1)
        ax1.plot(new_X[0, :], new_X[1, :], new_X[2, :], label='Planning result',
                 linestyle='-', linewidth=1)

        ax1.set_xlim([0, 6])
        ax1.set_ylim([0, 6])
        ax1.set_zlim([0, 4])

        ax1.set_xlabel('X, east')
        ax1.set_ylabel('Y, north')
        ax1.set_zlabel('Z, up')

        ax1.legend()
        self.plot_thrust(all_U, all_sigma)

        new_sigma = all_sigma
        k_sum = np.linspace(0, K - 1, K)
        time_real = (k_sum * new_sigma / K)


    def plot_thrust(self, all_U, all_sigma):
        fig1, axs = plt.subplots(1, 1)
        fig2, axs2 = plt.subplots(1, 1)
        fig3, axs3 = plt.subplots(1, 1)

        new_sigma = all_sigma
        k_sum = np.linspace(0, K - 1, K)
        # fig1, axs = plt.subplots(1, 1)
        time_real = (k_sum * new_sigma / (K-1))
        # axs.plot(time_real, all_U[-1, 0, :], '.--')
        # axs.plot(time_real, all_U[-1, 1, :], '.-.')
        # axs.plot(time_real, all_U[-1, 2, :], '.-')
        axs.plot(time_real, all_U[0, :], '.--')
        axs.plot(time_real, all_U[1, :], '.-.')
        axs.plot(time_real, all_U[2, :], '.-')

        # axs.plot(time_real, m.u_redim(m.T_min)*np.ones_like(time_real), '--', color='red')
        # axs.plot(time_real, m.u_redim(m.T_max)*np.ones_like(time_real), '--', color='red')
        axs.grid(True)
        axs.set_xlabel('$t$ $[s]$')
        axs.set_ylabel('Thrusts vector $[N]$')

        # fig2, axs2 = plt.subplots(1, 1)
        # axs2.plot(time_real, np.linalg.norm(all_U[-1, :, :], axis=0), '.-')
        axs2.plot(time_real, np.linalg.norm(all_U[:, :], axis=0), '.-')
        axs2.plot(time_real, self.T_min * np.ones_like(time_real), '--', color='red')
        axs2.plot(time_real, self.T_max * np.ones_like(time_real), '--', color='red')
        axs2.grid(True)
        axs2.set_xlabel('$t$ $[s]$')
        axs2.set_ylabel('Total thrust $[N]$')

        # axs3.plot(time_real, all_U[-1, 2, :] / np.linalg.norm(all_U[-1, :, :], axis=0), '.-')
        axs3.plot(time_real, all_U[2, :] / np.linalg.norm(all_U[:, :], axis=0), '.-')
        axs3.plot(time_real, self.cos_theta_max * np.ones_like(time_real), '--', color='red')
        axs3.set_xlabel('$t$ $[s]$')
        axs3.set_ylabel('Thrust tilt constraint $[N]$')

    def rotation_with_vector(self, a, b):
        # a and b are unit vectors
        vs = np.cross(a, b)
        v = vs / np.linalg.norm(vs)
        ca = np.dot(a, b)  # notice a and b are unit vectors
        vt = (1 - ca) * v
        rm = np.zeros((3, 3))
        vtx, vty, vtz = vt[0], vt[1], vt[2]
        vx, vy, vz = v[0], v[1], v[2]
        vsx, vsy, vsz = vs[0], vs[1], vs[2]
        rm[0, 0] = vtx * vx + ca
        rm[1, 1] = vty * vy + ca
        rm[2, 2] = vtz * vz + ca
        vtx *= vy
        vtz *= vx
        vty *= vz
        rm[0, 1] = vtx - vsz
        rm[0, 2] = vtz - vsy
        rm[1, 0] = vtx - vsz
        rm[1, 2] = vty - vsx
        rm[2, 0] = vtz - vsy
        rm[2, 1] = vty - vsx
        return rm
