import sympy as sp
import numpy as np
import cvxpy as cvx
from scipy import interpolate
# import open3d as pcd
from scripts.utils import euler_to_quat
from scripts.Configuration.parameters_6dof_landing_twophase import K
# from Configuration.parameters_iros2019_hoop import K, rho_h, rho_c, rho_g, l_c, p1, n1
import csv
import pandas as pd
import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d.art3d as art3d
# import matplotlib.ticker as ticker
#
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm


def skew(v):
    return sp.Matrix([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def dir_cosine(q):
    return sp.Matrix([
        [1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])],
        [2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])],
        [2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)]
    ])


def omega(w):
    return sp.Matrix([
        [0, -w[0], -w[1], -w[2]],
        [w[0], 0, w[2], -w[1]],
        [w[1], -w[2], 0, w[0]],
        [w[2], w[1], -w[0], 0],
    ])


class Model:
    """
    A 3 degree of freedom uav obstacle avoidance problem.
    """
    n_x = 13
    n_u = 4

    # Mass
    mass = 0.68  # kg

    # Flight time guess
    # tf_min = 1
    # tf_max = 8
    # t_f_guess = 3.  # s

    # State constraints
    r_I_init = np.array((1., 1., 1.))  # x m, y m, z m, z is altitude
    v_I_init = np.array((0., 0., 0.))  # x m/s, y m/s, z m/s
    q_B_I_init = euler_to_quat((0, 0, 0))
    w_B_init = np.deg2rad(np.array((0., 0., 0.)))

    # r_I_init = np.array((20., 20., 2.))  # x m, y m, z m, z is altitude
    # v_I_init = np.array((7.42173885, 6.25086656, -1.07100202))  # x m/s, y m/s, z m/s
    # q_B_I_init = np.array((5.81132495e-01, -2.76826531e-01, -2.66548820e-01, -7.17868154e-01))
    # w_B_init = np.array((7.31719067e-02, -1.16067689e-01, 8.80443784e-03))

    r_I_final = np.array((6., 6, 1.))
    # r_I_final = np.array((1., 1., 1.))
    v_I_final = np.array((0., 0., 0.))
    q_B_I_final = euler_to_quat((0, 0, 0))
    w_B_final = np.deg2rad(np.array((0., 0., 0.)))

    # Angles
    # max_gimbal = 30
    # max_angle = 80
    glidelslope_angle = 45
    # glidelslope_angle = 75

    # w_B_max = np.deg2rad(90)

    # omega_max_xy = 1.5
    # omega_max_z = 0.3
    omega_max_xy = 1.0
    omega_max_z = 0.3

    # v_max = 20
    #
    # cos_theta_max = np.cos(np.deg2rad(max_gimbal))
    # cos_theta_max = np.cos(np.deg2rad(max_angle))
    tan_gamma_gs = np.tan(np.deg2rad(glidelslope_angle))

    # Thrust limits
    # T_max = 3.3 * 9.81 * mass  # [kg*m/s^2]
    # T_min = 0.
    T_max = 20.  # [kg*m/s^2]
    T_min = 5.

    # Angular moment of inertia
    # J_B = np.diag([0.001, 0.001, 0.0017])  # [kg*m^2]
    # J_B = np.diag([0.00264, 0.00264, 0.00496])
    J_B = np.diag([0.007, 0.007, 0.012])
    J_B_inv = np.linalg.inv(J_B)

    # Gravity
    g = 9.81
    g_I = np.array((0., 0., -g))  # -9.81 [m/s^2]
    f_balance = g/4.
    u_init = mass*f_balance*np.ones(4)
    u_final = u_init

    # Aerodynamic model
    # rho0 = 1.225  # [kg/m^3]
    # ca_x = 1.
    # ca_yz = 3.
    # S_A = 10
    # C_A = np.diag([ca_yz, ca_yz, ca_x])

    # Vector from thrust point to CoM
    # r_T_B = np.array([0., 0., -14.])  # -20 m
    # r_cp_B = np.array([0., 0., 2.])  # positive for static stable

    # Matrix for single motor force
    m_T = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 1, 1, 1]])
    # arm_length = 0.15
    arm_length = 0.17
    # ctau = 0.5
    ctau = 0.05

    # X:
    # arm_torque = arm_length / np.sqrt(2)
    # m_tau = np.array([[arm_torque, arm_torque, -arm_torque, -arm_torque],
    #                   [-arm_torque, arm_torque, arm_torque, -arm_torque],
    #                   [ctau, -ctau, ctau, -ctau]])
    
    # shizi:
    arm_torque = arm_length
    m_tau = np.array([[arm_torque, 0, -arm_torque, 0],
                      [0, arm_torque, 0, -arm_torque],
                      [-ctau, ctau, -ctau, ctau]])

    # Obstacle

    # ------------------------------------------ Start normalization stuff
    def __init__(self, quad_name):
        """
        A large r_scale for a small scale problem will
        ead to numerical problems as parameters become excessively small
        and (it seems) precision is lost in the dynamics.
        """

        if quad_name == "iris":
            self.mass  =1.52
            self.T_max = 21.
            self.T_min = 5.
            self.J_B = np.diag([0.0347563, 0.0458929, 0.0977])
            self.J_B_inv = np.linalg.inv(self.J_B)
            self.u_init = self.mass*self.f_balance*np.ones(4)
            self.u_final = self.u_init

            self.length_fx = 0.13
            self.length_bx = 0.13
            self.length_fy = 0.22
            self.length_by = 0.2
            self.ctau = 0.016

            self.m_tau = np.array([[self.length_fy, -self.length_fy, -self.length_by, self.length_by],
                      [-self.length_fx, -self.length_fx, self.length_bx, self.length_bx],
                      [-self.ctau, self.ctau, -self.ctau, self.ctau]])
            
        if quad_name == "aims1":
            self.mass  = 0.661
            self.T_max = 30.
            self.T_min = 4.
            self.J_B = np.diag([0.0021055, 0.0025566, 0.0041921])
            self.J_B_inv = np.linalg.inv(self.J_B)
            self.u_init = self.mass*self.f_balance*np.ones(4)
            self.u_final = self.u_init

            self.arm_length = 0.125
            self.ctau = 0.016

            arm_torque = self.arm_length / np.sqrt(2)
            self.m_tau = np.array([[arm_torque, -arm_torque, -arm_torque, arm_torque],
                              [-arm_torque, -arm_torque, arm_torque, arm_torque],
                              [self.ctau, -self.ctau, self.ctau, -self.ctau]])
            
        if quad_name == "aims3":
            self.mass  = 0.463
            self.T_max = 30.
            self.T_min = 4.
            self.J_B = np.diag([0.000931844, 0.000931844, 0.000793978])  # [kg*m^2]
            self.J_B_inv = np.linalg.inv(self.J_B)
            self.u_init = self.mass*self.f_balance*np.ones(4)
            self.u_final = self.u_init

            self.arm_length = 0.125
            self.ctau = 0.016

            arm_torque = self.arm_length / np.sqrt(2)

            lenth_fx = 0.0635
            lenth_bx = 0.0635
            lenth_fy = 0.049
            lenth_by = 0.049
            self.m_tau = np.array([[lenth_fx, -lenth_fx, -lenth_bx, lenth_bx],
                            [-lenth_fy, -lenth_fy, lenth_by, lenth_by],
                            [self.ctau, -self.ctau, self.ctau, -self.ctau]])

        # self.set_random_initial_state()

        self.x_init = np.concatenate((self.r_I_init, self.v_I_init, self.q_B_I_init, self.w_B_init))
        self.x_final = np.concatenate((self.r_I_final, self.v_I_final, self.q_B_I_final, self.w_B_final))

        # slack variable for linear constraint relaxation
        # self.s_prime = cvx.Variable((K, 1), nonneg=True)
        # self.sigma = cvx.Variable(K, nonneg=True)
        self.rotor_T_max = cvx.Parameter(nonneg=True)
        self.rotor_T_min = cvx.Parameter(nonneg=True)

        # slack variable for lossless convexification
        # self.gamma = cvx.Variable(K, nonneg=True)

    def get_equations(self):
        """
        :return: Functions to calculate A, B and f given state x and input u
        """
        f = sp.zeros(13, 1)

        x = sp.Matrix(sp.symbols('rx ry rz vx vy vz q0 q1 q2 q3 wx wy wz', real=True))
        u = sp.Matrix(sp.symbols('u1 u2 u3 u4', real=True))

        g_I = sp.Matrix(self.g_I)
        M_T = sp.Matrix(self.m_T)
        M_tau = sp.Matrix(self.m_tau)
        # r_T_B = sp.Matrix(self.r_T_B)
        # r_cp_B = sp.Matrix(self.r_cp_B)
        J_B = sp.Matrix(self.J_B)
        J_B_inv = sp.Matrix(self.J_B_inv)

        C_B_I = sp.Matrix(dir_cosine(x[6:10, 0]))
        C_I_B = sp.Matrix(C_B_I.transpose())
        # C_A = sp.Matrix(self.C_A)

        # A_B = -0.5*self.rho0*(f[4:7, 0].norm())*self.S_A*C_A*C_B_I*f[4:7, 0]
        f[0:3, 0] = x[3:6, 0]
        f[3:6, 0] = 1 / self.mass * (C_I_B @ M_T @ u) + g_I
        f[6:10, 0] = 1 / 2 * omega(x[10:13, 0]) * x[6: 10, 0]
        f[10:13, 0] = J_B_inv * ((M_tau @ u) - skew(x[10:13, 0]) @ J_B @ x[10:13, 0])

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
            q_B_I_k = np.array([1, 0, 0, 0])
            w_B_k = alpha1 * self.x_init[10:13] + alpha2 * self.x_final[10:13]

            X[:, k] = np.concatenate((r_I_k, v_I_k, q_B_I_k, w_B_k))
            # U[:, k] = (self.T_max-self.T_min)/2 * np.array([1, 1, 1, 1])
            U[:, k] = self.f_balance * np.array([1, 1, 1, 1])
            # U[:, k] = (self.T_max-self.T_min)/8 * np.array([1, 1, 1, 1])

        return X, U
    
    def initialize_trajectory_landing_only(self, X, U, K_landing):
        """
        Initialize the trajectory.

        :param X: Numpy array of states to be initialized
        :param U: Numpy array of inputs to be initialized
        :return: The initialized X and U
        """

        for k in range(K_landing):
            alpha1 = (K_landing - k - 1) / (K_landing-1)
            alpha2 = k / (K_landing-1)

            r_I_k = alpha1 * self.x_init[0:3] + alpha2 * self.x_final[0:3]
            v_I_k = alpha1 * self.x_init[3:6] + alpha2 * self.x_final[3:6]
            q_B_I_k = np.array([1, 0, 0, 0])
            w_B_k = alpha1 * self.x_init[10:13] + alpha2 * self.x_final[10:13]

            X[:, k] = np.concatenate((r_I_k, v_I_k, q_B_I_k, w_B_k))
            # U[:, k] = (self.T_max-self.T_min)/2 * np.array([1, 1, 1, 1])
            U[:, k] = self.f_balance * np.array([1, 1, 1, 1])
            # U[:, k] = (self.T_max-self.T_min)/8 * np.array([1, 1, 1, 1])

        return X, U

    def initialize_trajectory_waypoint(self, X, U, gates):
        """
        Initialize the trajectory.

        :param X: Numpy array of states to be initialized
        :param U: Numpy array of inputs to be initialized
        :return: The initialized X and U
        """
        gate_num = gates.shape[0]
        k_gate = np.zeros(gate_num+2)
        d_waypoints = np.zeros(gate_num+1)
        for k in range(gate_num+1):
            if k == 0:
                d_waypoints[k] = np.linalg.norm(self.x_init[0:3]-gates[k, :])
            elif k == gate_num:
                d_waypoints[k] = np.linalg.norm(self.x_final[0:3]-gates[k-1, :])
            else:
                d_waypoints[k] = np.linalg.norm(gates[k, :]-gates[k-1, :])

        k_gate[0] = 0
        k_gate[-1] = K-1
        for k in range(gate_num):
            k_gate[k+1] = np.floor(K*sum(d_waypoints[0:k+1])/sum(d_waypoints))

        all_waypoints = np.concatenate((self.x_init[0:3].reshape(1, -1), gates, self.x_final[0:3].reshape(1, -1)))
        X[0, :] = interpolate.interp1d(k_gate, all_waypoints[:, 0], kind='linear')(range(K))
        X[1, :] = interpolate.interp1d(k_gate, all_waypoints[:, 1], kind='linear')(range(K))
        X[2, :] = interpolate.interp1d(k_gate, all_waypoints[:, 2], kind='linear')(range(K))

        for k in range(K):
            alpha1 = (K - k - 1) / (K-1)
            alpha2 = k / (K-1)
            v_I_k = alpha1 * self.x_init[3:6] + alpha2 * self.x_final[3:6]
            q_B_I_k = np.array([1, 0, 0, 0])
            w_B_k = alpha1 * self.x_init[10:13] + alpha2 * self.x_final[10:13]

            X[3:, k] = np.concatenate((v_I_k, q_B_I_k, w_B_k))
            U[:, k] = self.T_max/4 * np.array([1, 1, 1, 1])
            # U[:, k] = self.f_balance * np.array([1, 1, 1, 1])

        tf_guess = sum(d_waypoints)/1.5

        return X, U, k_gate, tf_guess

    def initialize_trajectory_hoop(self, X, U, p_h):
        """
        Initialize the trajectory.

        :param X: Numpy array of states to be initialized
        :param U: Numpy array of inputs to be initialized
        :return: The initialized X and U
        """
        k_h = np.floor(K / 2)
        for k in range(K):
            alpha1 = (K - k - 1) / (K-1)
            alpha2 = k / (K-1)

            if k < k_h:
                beta1 = (k_h - k - 1) / (k_h - 1)
                beta2 = k / (k_h - 1)
                r_I_k = beta1 * self.x_init[0:3] + beta2 * p_h
            else:
                beta1 = (K - k - 1) / (K - k_h - 1)
                beta2 = (k - k_h) / (K - k_h - 1)
                r_I_k = beta1 * p_h + beta2 * self.x_final[0:3]

            # r_I_k = alpha1 * self.x_init[0:3] + alpha2 * self.x_final[0:3]
            v_I_k = alpha1 * self.x_init[3:6] + alpha2 * self.x_final[3:6]
            # q_B_I_k = np.array([1, 0, 0, 0])
            # w_B_k = alpha1 * self.x_init[11:14] + alpha2 * self.x_final[11:14]

            X[:, k] = np.concatenate((r_I_k, v_I_k))
            U[:, k] = ((self.T_min - 0) / 2) * np.array([0, 0, 1])

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

    def get_constraints(self, X_v, U_v, p_init, v_init, p_final, v_final):
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
            # X_v[0:3, 0] == self.x_init[0:3],
            # X_v[3:6, 0] == self.x_init[3:6],
            X_v[6:10, 0] == self.x_init[6:10],
            X_v[10:13, 0] == self.x_init[10:13],

            # X_v[:, -1] == self.x_final[:],

            X_v[0:3, 0] == p_init,
            X_v[3:6, 0] == v_init,
            X_v[0:3, -1] == p_final,
            X_v[3:6, -1] == v_final,

            # X_v[0:3, -1] == self.x_final[0:3],
            # X_v[3:6, -1] == self.x_final[3:6],
            X_v[6:10, -1] == self.x_final[6:10],
            X_v[10:13, -1] == self.x_final[10:13],

            U_v[:, 0] == self.u_init,
            
            U_v[:, -1] == self.u_final,

            # - X_v[2, :] <= 0,
        ]

        self.rotor_T_max.value = self.T_max/4
        self.rotor_T_min.value = self.T_min/4

        constraints += [
            # State constraints:
            # cvx.norm(X_v[1: 3, :], axis=0) <= (X_v[3, :]) * self.tan_gamma_gs,  # glideslope
            # cvx.norm(X_v[8:10, :], axis=0) <= np.sqrt((1 - self.cos_theta_max) / 2),  # maximum angle
            # cvx.norm(X_v[11: 14, :], 'inf', axis=0) <= self.w_B_max,  # maximum angular velocity
            # cvx.norm(X_v[3: 6, :], axis=0) <= self.v_max,

            X_v[10: 12, :] <= self.omega_max_xy,
            X_v[10: 12, :] >= -self.omega_max_xy,
            X_v[12, :] <= self.omega_max_z,
            X_v[12, :] >= -self.omega_max_z,
            # cvx.norm(X_v[10: 13, :], axis=0) <= self.w_B_max,

            # Control constraints:
            U_v[:, :] <= self.rotor_T_max,
            -U_v[:, :] <= -self.rotor_T_min

            # # Lossless convexification:
            # self.cos_theta_max * self.sigma <= U_v[2, :],  # gimbal angle constraint
            # cvx.norm(U_v, axis=0) <= self.sigma,  # upper thrust constraint
            # self.sigma <= self.a_max,
            # -self.sigma <= -self.a_min

        ]

        # constraints += [cvx.norm(X_v[0: 2, k]-p_final[0:2], axis=0) <= (X_v[2, k]-p_final[2]) * self.tan_gamma_gs for k in range(K)]

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

    def plot_trajectory(self, all_X, all_U, initialX, initialU, all_sigma, X_nl_full, scene=0, label='Nonlinear open loop', color='xkcd:grey',
                        initial_label='Initial trajectory', initial_color='xkcd:grey'):

        fig_new = plt.figure(figsize=(6, 6))
        ax1 = fig_new.add_subplot(1, 1, 1, projection='3d')
        new_X = all_X[:, :]
        new_U = all_U[:, :]
        new_sigma = all_sigma
        initial_X = initialX
        initial_U = initialU

        ax1.plot(X_nl_full[0, :], X_nl_full[1, :], X_nl_full[2, :], label=label, color=color,
                 linestyle='-', linewidth=1)
        ax1.plot(initial_X[0, :], initial_X[1, :], initial_X[2, :], label=initial_label, color=initial_color,
                 linestyle='-', linewidth=1)
        ax1.plot(new_X[0, :], new_X[1, :], new_X[2, :], label='Planning result',
                 linestyle='-', linewidth=1)

        # ax1.set_xlim([min(new_X[0, :]), max(new_X[0, :])])
        # ax1.set_ylim([min(new_X[1, :]), max(new_X[1, :])])
        # ax1.set_zlim([min(new_X[2, :]), max(new_X[2, :])])
        ax1.set_xlim([0, max(new_X[0, :])])
        ax1.set_ylim([0, 3])
        ax1.set_zlim([0, 3])

        ax1.set_xlabel('X, east')
        ax1.set_ylabel('Y, north')
        ax1.set_zlabel('Z, up')

        ax1.legend()
        self.plot_thrust(all_U, all_sigma)

    def plot_thrust(self, all_U, all_sigma):
        fig1, axs = plt.subplots(1, 1)
        # fig2, axs2 = plt.subplots(1, 1)
        # fig3, axs3 = plt.subplots(1, 1)

        new_sigma = all_sigma
        k_sum = np.linspace(0, K - 1, K)
        # fig1, axs = plt.subplots(1, 1)
        time_real = (k_sum * new_sigma / (K-1))
        axs.plot(time_real, all_U[0, :], '.--')
        axs.plot(time_real, all_U[1, :], '.-.')
        axs.plot(time_real, all_U[2, :], '.-')
        axs.plot(time_real, all_U[3, :], '.')
        axs.plot(time_real, self.T_min * np.ones_like(time_real)/4, '--', color='red')
        axs.plot(time_real, self.T_max * np.ones_like(time_real)/4, '--', color='red')
        # axs.plot(time_real, m.u_redim(m.T_min)*np.ones_like(time_real), '--', color='red')
        # axs.plot(time_real, m.u_redim(m.T_max)*np.ones_like(time_real), '--', color='red')
        axs.grid(True)
        axs.set_xlabel('$t$ $[s]$')
        axs.set_ylabel('Thrusts vector $[N]$')

        # fig2, axs2 = plt.subplots(1, 1)
        # axs2.plot(time_real, np.linalg.norm(all_U[-1, :, :], axis=0), '.-')
        # axs2.plot(time_real, self.T_min * np.ones_like(time_real), '--', color='red')
        # axs2.plot(time_real, self.T_max * np.ones_like(time_real), '--', color='red')
        # axs2.grid(True)
        # axs2.set_xlabel('$t$ $[s]$')
        # axs2.set_ylabel('Total thrust $[N]$')

        # axs3.plot(time_real, all_U[-1, 2, :] / np.linalg.norm(all_U[-1, :, :], axis=0), '.-')
        # axs3.plot(time_real, self.cos_theta_max * np.ones_like(time_real), '--', color='red')
        # axs3.set_xlabel('$t$ $[s]$')
        # axs3.set_ylabel('Thrust tilt constraint $[N]$')

    def save_tra(self, all_X, all_U, new_sigma, mu, lamb, filename):
        new_X = all_X[-1, :, :]
        new_U = all_U[-1, :, :]
        k_sum = np.linspace(0, K - 1, K)
        time_real = (k_sum * new_sigma / (K - 1))
        p = new_X[0:3, :]
        v = new_X[3:6, :]
        q = new_X[6:10, :]
        w = new_X[10:13, :]

        NW = 1

        dt = new_sigma / (K-1)
        a_lin = np.zeros((3, K))
        a_rot = np.zeros((3, K))
        a_lin[:, 0:-1] = np.diff(v) / dt
        a_rot[:, 0:-1] = np.diff(w) / dt

        assert type(filename) == str
        with open(filename+'.csv', 'w') as csvfile:

            traj_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            labels = ['t', 'p_x', 'p_y', 'p_z',
                      'q_w', 'q_x', 'q_y', 'q_z',
                      'v_x', 'v_y', 'v_z',
                      'w_x', 'w_y', 'w_z',
                      'a_lin_x', 'a_lin_y', 'a_lin_z',
                      'a_rot_x', 'a_rot_y', 'a_rot_z',
                      'u_1', 'u_2', 'u_3', 'u_4']
            for i in range(NW):
                labels += ['mu_' + str(i)]
                # labels += ['nu_' + str(i)]
                labels += ['lambda_' + str(i)]

            traj_writer.writerow(labels)
            for i in range(K):
                # States
                row = [time_real[i],
                       p[0, i], p[1, i], p[2, i],
                       q[0, i], q[1, i], q[2, i], q[3, i],
                       v[0, i], v[1, i], v[2, i],
                       w[0, i], w[1, i], w[2, i],
                       a_lin[0, i], a_lin[1, i], a_lin[2, i],
                       a_rot[0, i], a_rot[1, i], a_rot[2, i]]

                # Inputs
                if i < K - 1:
                    for j in range(self.n_u):
                        row += [new_U[j, i]]
                else:
                    row += [0] * self.n_u

                # Progress
                for j in range(NW):
                    row += [lamb[j, i]]
                    if i > 0:
                        # row += [self.nu[j, i - 1]]
                        row += [mu[j, i - 1]]
                    else:
                        row += [0] * 2

                traj_writer.writerow(row)

        #  all trajectory
        all_traj = all_X.reshape(-1, all_X.shape[-1])
        pd.DataFrame(all_traj).to_csv(filename+'_all.csv')
