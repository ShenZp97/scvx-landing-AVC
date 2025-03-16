#! /usr/bin/env python3
# import rospy
# import actionlib

import numpy as np
import pandas as pd

from time import time
import matplotlib.pyplot as plt

from scripts.Configuration.parameters_3dof_landing_twophase import *
from scripts.Configuration.parameters_6dof_landing_twophase import *
from scripts.SCP.discretization_3dof_landing_twophase import FirstOrderHold
from scripts.SCP.discretization_noloop_x_landing_twophase import FirstOrderHold as FirstOrderHold_6dof
# from SCP.discretization_6dof import FirstOrderHold as FirstOrderHold_6dof
from scripts.SCP.scproblem_3dof_landing_twophase import SCProblem
from scripts.SCP.scproblem_3dof_landing_twophase_obstacle import SCProblem as SCProblem_obstacle
from scripts.SCP.scproblem_quad_6dof_socp_landing_twophase import SCProblem as SCProblem_6dof
from scripts.SCP.scproblem_quad_6dof_socp_landing_twophase_moving_exp import SCProblem as SCProblem_6dof_moving_exp
# from utils import format_line
from scripts.Models.quadrotor_3dof_landing_twophase import Model
from scripts.Models.quadrotor_6dof_x_landing_twophase import Model as Model_6dof
from scripts.Models.quadrotor_6dof_x_landing_twophase_moving import Model as Model_6dof_moving

from scripts.utils import obstacle_constraint_violation, plot_obstacle


class lcvx_traj():
    def __init__(self,quad_name, K):
        self.m = Model(quad_name)

        self.seq = None

        # INITIALIZATION--------------------------------------------------------------------------------------------------------
        self.tf_guess_1 = 3.
        self.tf_guess_2 = 2.
        # self.delta_t = self.tf_guess / (K - 1)

        # self.integrator = FirstOrderHold(self.m, K)

        self.K = K
        self.K_phase = int(np.ceil(K/2))
        # START LCVX--------------------------------------------------------------------------------------

        self.problem = SCProblem(self.m, K)

        # pre warm
        self.initial_p = np.array([1, 1, 1])
        self.initial_v = np.array([0, 0, 0])
        self.final_p = np.array([4, 4, 1])
        self.final_v = np.array([0, 0, 0])

        self.initial_u = self.m.mass*np.array((0., 0., 9.81))
        self.final_u = self.m.mass*np.array((0., 0., 9.81))

        self.problem.set_parameters(delta_t=(self.tf_guess_1 / (self.K_phase - 1)), delta_t2=(self.tf_guess_1 / (self.K_phase - 1)) ** 2, 
                                    delta_t_2=(self.tf_guess_2 / (self.K - self.K_phase)), delta_t2_2=(self.tf_guess_2 / (self.K - self.K_phase)) ** 2, initial_p=self.initial_p,
                                    initial_v=self.initial_v, final_p=self.final_p, final_v=self.final_v,
                                    initial_u=self.initial_u, final_u=self.final_u)
        error = self.problem.solve(verbose=False, solver=solver, ignore_dpp=False)

        # ax = plt.figure().add_subplot(projection='3d')
        # X = self.problem.get_variable('X')
        # ax.plot(X[0, :], X[1,:], X[2,:])
        # plt.show()

        # message
        # self.msg = PositionCommand()

    def set_parameters(self):
        self.problem.set_parameters(delta_t=(self.tf_guess_1 / (self.K - 1)), delta_t2=(self.tf_guess_1 / (self.K - 1)) ** 2, 
                                    delta_t_2=(self.tf_guess_2 / (self.K - self.K_phase)), delta_t2_2=(self.tf_guess_2 / (self.K - self.K_phase)) ** 2, initial_p=self.initial_p,
                                    initial_v=self.initial_v, final_p=self.final_p, final_v=self.final_v,
                                    initial_u=self.initial_u, final_u=self.final_u)

    def solve(self):
        self.problem.solve(verbose=False, solver=solver, ignore_dpp=False)

    def sub_odom(self, msg):
        # self.seq = msg.header.seq
        self.odom_current = msg
        self.pos_current = np.array((msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z))

class lcvx_traj_obstacle():
    def __init__(self,quad_name, K):
        self.m = Model(quad_name)

        self.seq = None

        # INITIALIZATION--------------------------------------------------------------------------------------------------------
        self.tf_guess_1 = 3.
        self.tf_guess_2 = 2.
        # self.delta_t = self.tf_guess / (K - 1)

        # self.integrator = FirstOrderHold(self.m, K)

        self.K = K
        self.K_phase = int(np.ceil(K/2))

        self.X = np.zeros(shape=[self.m.n_x, K])
        self.U = np.zeros(shape=[self.m.n_u, K])
        self.X, self.U = self.m.initialize_trajectory(self.X, self.U)

        # obstacle
        self.obs_num = 1
        self.m.obs_num = self.obs_num
        self.R1 = 1.
        self.H1 = np.diag([1., 1., 1.])
        self.p1 = np.array((4, 4, 2))
        # START LCVX--------------------------------------------------------------------------------------

        self.problem = SCProblem_obstacle(self.m, K)

        # pre warm
        self.initial_p = np.array([1, 1, 1])
        self.initial_v = np.array([0, 0, 0])
        self.final_p = np.array([4, 4, 1])
        self.final_v = np.array([0, 0, 0])

        self.initial_u = self.m.mass*np.array((0., 0., 9.81))
        self.final_u = self.m.mass*np.array((0., 0., 9.81))

        A = self.R1-obstacle_constraint_violation(self.X[0:3, :self.K_phase-1].copy(), self.H1, self.p1)
        delta_r = (self.X[0:3, :self.K_phase-1] - np.tile(self.p1.reshape(3, 1), (1, self.K_phase-1)))
        xi = np.linalg.norm(np.dot(self.H1, delta_r), axis=0)
        zeta = np.dot(self.H1.transpose(), np.dot(self.H1, delta_r)).transpose()
        B = zeta / np.tile(np.expand_dims(xi, axis=-1), (1, 3))
        C = np.array([B[k, :] @ self.X[0:3, k] for k in range(self.K_phase-1)])
        if self.obs_num == 1:
            A = np.expand_dims(A, axis=-1)
            C = np.expand_dims(C, axis=-1)

        self.problem.set_parameters(delta_t=(self.tf_guess_1 / (K - 1)), delta_t2=(self.tf_guess_1 / (K - 1)) ** 2, 
                                    delta_t_2=(self.tf_guess_2 / (K - 1)), delta_t2_2=(self.tf_guess_2 / (K - 1)) ** 2, initial_p=self.initial_p,
                                    initial_v=self.initial_v, final_p=self.final_p, final_v=self.final_v,
                                    initial_u=self.initial_u, final_u=self.final_u,
                                    A=A, B=B, C=C)
        error = self.problem.solve(verbose=False, solver=solver, ignore_dpp=False)

        # ax = plt.figure().add_subplot(projection='3d')
        # X = self.problem.get_variable('X')
        # ax.plot(X[0, :], X[1,:], X[2,:])
        # plt.show()

        # message
        # self.msg = PositionCommand()

    def set_parameters(self):
        self.problem.set_parameters(delta_t=(self.tf_guess_1 / (K - 1)), delta_t2=(self.tf_guess_1 / (K - 1)) ** 2, 
                                    delta_t_2=(self.tf_guess_2 / (K - 1)), delta_t2_2=(self.tf_guess_2 / (K - 1)) ** 2, initial_p=self.initial_p,
                                    initial_v=self.initial_v, final_p=self.final_p, final_v=self.final_v,
                                    initial_u=self.initial_u, final_u=self.final_u)

    def solve(self):
        self.problem.solve(verbose=False, solver=solver, ignore_dpp=False)

    def sub_odom(self, msg):
        # self.seq = msg.header.seq
        self.odom_current = msg
        self.pos_current = np.array((msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z))



class scvx_traj():
    def __init__(self, quad_name, K):
        self.m = Model_6dof(quad_name)

        self.seq = None

        # INITIALIZATION--------------------------------------------------------------------------------------------------------
        self.tf_guess_1 = 3.
        self.tf_guess_2 = 2.
        # self.delta_t = self.tf_guess_1 / (K - 1)

        self.obs_num = 1
        self.m.obs_num = self.obs_num
        self.R1 = 1.
        self.H1 = np.diag([1., 1., 1.])
        self.p1 = np.array((4, 4, 2))

        self.K = K
        K_phase = int(np.ceil(K/2))
        self.integrator_obstacle = FirstOrderHold_6dof(quad_name, self.m, K_phase)
        self.integrator_landing = FirstOrderHold_6dof(quad_name, self.m, self.K-K_phase+1)

        
        # START SCVX--------------------------------------------------------------------------------------

        self.problem = SCProblem_6dof(self.m, K)

        self.X = np.zeros(shape=[self.m.n_x, K])
        self.U = np.zeros(shape=[self.m.n_u, K])
        self.X, self.U = self.m.initialize_trajectory(self.X, self.U)

        self.sigma_1 = self.tf_guess_1
        self.sigma_2 = self.tf_guess_2
        # self.all_X = [self.X.copy()]
        # self.all_U = [self.U.copy()]
        # self.all_sigma = [self.sigma]

        # pre warm
        self.initial_p = np.array([1., 1, 1])
        self.initial_v = np.array([0, 0, 0])
        self.final_p = np.array([4, 4, 1])
        self.final_v = np.array([0, 0, 0])

        self.initial_q = np.array([1, 0, 0, 0])
        self.initial_w = np.array([0, 0, 0])
        self.final_q = np.array([1, 0, 0, 0])
        self.final_w = np.array([0, 0, 0])

        self.initial_u = self.m.mass*self.m.f_balance*np.ones(4)
        self.final_u = self.m.mass*self.m.f_balance*np.ones(4)

        K_phase = int(np.ceil(K/2))
        self.K_phase = K_phase
        A_bar_1, B_bar_1, C_bar_1, S_bar_1, z_bar_1 = self.integrator_obstacle.calculate_discretization(self.X[:, 0:K_phase], self.U[:, 0:K_phase], self.sigma_1, K_phase)
        A_bar_2, B_bar_2, C_bar_2, S_bar_2, z_bar_2 = self.integrator_landing.calculate_discretization(self.X[:, K_phase-1:self.K], self.U[:, K_phase-1:self.K], self.sigma_2, self.K-K_phase+1)

        A_bar = np.concatenate((A_bar_1,A_bar_2),axis=1)
        B_bar = np.concatenate((B_bar_1,B_bar_2),axis=1)
        C_bar = np.concatenate((C_bar_1,C_bar_2),axis=1)
        S_bar = np.concatenate((S_bar_1,S_bar_2),axis=1)
        z_bar = np.concatenate((z_bar_1,z_bar_2),axis=1)

        A = self.R1-obstacle_constraint_violation(self.X[0:3, :self.K_phase-1].copy(), self.H1, self.p1)
        delta_r = (self.X[0:3, :self.K_phase-1] - np.tile(self.p1.reshape(3, 1), (1, self.K_phase-1)))
        xi = np.linalg.norm(np.dot(self.H1, delta_r), axis=0)
        zeta = np.dot(self.H1.transpose(), np.dot(self.H1, delta_r)).transpose()
        B = zeta / np.tile(np.expand_dims(xi, axis=-1), (1, 3))
        C = np.array([B[k, :] @ self.X[0:3, k] for k in range(self.K_phase-1)])
        if self.obs_num == 1:
            A = np.expand_dims(A, axis=-1)
            C = np.expand_dims(C, axis=-1)

        self.problem.set_parameters(X_last=self.X, A_bar=A_bar, B_bar=B_bar, C_bar=C_bar, S_bar=S_bar, z_bar=z_bar,
                                    sigma_last_1=self.sigma_1, sigma_last_2=self.sigma_2, initial_p=self.initial_p,
                                    initial_v=self.initial_v, final_p=self.final_p, final_v=self.final_v,
                                    initial_q=self.initial_q, initial_w=self.initial_w, final_q=self.final_q, final_w=self.final_w,
                                    initial_u=self.initial_u, final_u=self.final_u,
                                    A=A,B=B,C=C)
        
        # A_bar_1, B_bar_1, S_bar_1, z_bar_1 = self.integrator.trapezoidal_rule(self.X[:, 0:K_phase], self.U[:, 0:K_phase], self.sigma_1)
        # A_bar_2, B_bar_2, S_bar_2, z_bar_2 = self.integrator.trapezoidal_rule(self.X[:, K_phase-1:self.K], self.U[:, K_phase-1:self.K], self.sigma_2)
        
        # delta_t = 1./(K_phase-1)
        # delta_t_2 = 1./(K-K_phase)

        # A_bar = np.concatenate((A_bar_1*delta_t/2,A_bar_2*delta_t_2/2),axis=1)
        # B_bar = np.concatenate((B_bar_1*delta_t/2,B_bar_2*delta_t_2/2),axis=1)
        # S_bar = np.concatenate((S_bar_1*delta_t/2,S_bar_2*delta_t_2/2),axis=1)
        # z_bar = np.concatenate((z_bar_1*delta_t/2,z_bar_2*delta_t_2/2),axis=1)

        # self.problem.set_parameters(X_last=self.X, A_bar=A_bar, B_bar=B_bar, S_bar=S_bar, z_bar=z_bar,
        #                             sigma_last_1=self.sigma_1, sigma_last_2=self.sigma_2, initial_p=self.initial_p,
        #                             initial_v=self.initial_v, final_p=self.final_p, final_v=self.final_v)
        
        
        # self.problem.get_is_dpp()
        # self.problem.get_is_dcp()
        error = self.problem.solve(verbose=False, solver=solver, ignore_dpp=False)

        # ax = plt.figure().add_subplot(projection='3d')
        # X = self.problem.get_variable('X')
        # ax.plot(X[0, :], X[1,:], X[2,:])
        # plt.show()

        # message
        # self.msg = PositionCommand()

    def set_parameters(self, A_bar, B_bar, C_bar, S_bar, z_bar):
        self.problem.set_parameters(X_last=self.X, A_bar=A_bar, B_bar=B_bar, C_bar=C_bar, S_bar=S_bar, z_bar=z_bar,
                                    sigma_last_1=self.sigma_1, sigma_last_2=self.sigma_2, initial_p=self.initial_p,
                                    initial_v=self.initial_v, final_p=self.final_p, final_v=self.final_v,
                                    initial_q=self.initial_q, initial_w=self.initial_w, final_q=self.final_q, final_w=self.final_w,
                                    initial_u=self.initial_u, final_u=self.final_u)

    def zero_initialize(self):
        self.X = np.zeros(shape=[self.m.n_x, K])
        self.U = np.zeros(shape=[self.m.n_u, K])

    def line_initialize(self):
        """
        Initialize the trajectory.

        :param X: Numpy array of states to be initialized
        :param U: Numpy array of inputs to be initialized
        :return: The initialized X and U
        """

        for k in range(K):
            alpha1 = (K - k - 1) / (K-1)
            alpha2 = k / (K-1)

            r_I_k = alpha1 * self.initial_p + alpha2 * self.final_p
            v_I_k = alpha1 * self.initial_v + alpha2 * self.final_v
            q_B_I_k = np.array([1, 0, 0, 0])
            w_B_k = alpha1 * self.initial_w + alpha2 * self.final_w

            self.X[:, k] = np.concatenate((r_I_k, v_I_k, q_B_I_k, w_B_k))
            # U[:, k] = (self.T_max-self.T_min)/2 * np.array([1, 1, 1, 1])
            self.U[:, k] = self.m.f_balance * np.array([1, 1, 1, 1])
            # U[:, k] = (self.T_max-self.T_min)/8 * np.array([1, 1, 1, 1])


    def set_initial_traj(self, X, U, tf1, tf2):
        self.X = X
        self.U = U
        self.sigma_1 = tf1
        self.sigma_2 = tf2

    def solve(self):
        converged = False

        t_integration = 0
        
        for it in range(iterations):

            # ax = plt.figure().add_subplot(projection='3d')
            # ax.plot(self.X[0, :], self.X[1,:], self.X[2,:])
            # plt.savefig('traj_test.png')

            K_phase = self.K_phase

            t_init = time()
            A_bar_1, B_bar_1, C_bar_1, S_bar_1, z_bar_1 = self.integrator_obstacle.calculate_discretization(self.X[:, 0:K_phase], self.U[:, 0:K_phase], self.sigma_1, K_phase)
            A_bar_2, B_bar_2, C_bar_2, S_bar_2, z_bar_2 = self.integrator_landing.calculate_discretization(self.X[:, K_phase-1:self.K], self.U[:, K_phase-1:self.K], self.sigma_2, self.K-K_phase+1)
            t_integration += time()-t_init

            A_bar = np.concatenate((A_bar_1,A_bar_2),axis=1)
            B_bar = np.concatenate((B_bar_1,B_bar_2),axis=1)
            C_bar = np.concatenate((C_bar_1,C_bar_2),axis=1)
            S_bar = np.concatenate((S_bar_1,S_bar_2),axis=1)
            z_bar = np.concatenate((z_bar_1,z_bar_2),axis=1)

            A = self.R1-obstacle_constraint_violation(self.X[0:3, :self.K_phase-1].copy(), self.H1, self.p1)
            delta_r = (self.X[0:3, :self.K_phase-1] - np.tile(self.p1.reshape(3, 1), (1, self.K_phase-1)))
            xi = np.linalg.norm(np.dot(self.H1, delta_r), axis=0)
            zeta = np.dot(self.H1.transpose(), np.dot(self.H1, delta_r)).transpose()
            B = zeta / np.tile(np.expand_dims(xi, axis=-1), (1, 3))
            C = np.array([B[k, :] @ self.X[0:3, k] for k in range(self.K_phase-1)])
            if self.obs_num == 1:
                A = np.expand_dims(A, axis=-1)
                C = np.expand_dims(C, axis=-1)

            self.problem.set_parameters(X_last=self.X, A_bar=A_bar, B_bar=B_bar, C_bar=C_bar, S_bar=S_bar, z_bar=z_bar,
                                        sigma_last_1=self.sigma_1, sigma_last_2=self.sigma_2, initial_p=self.initial_p,
                                        initial_v=self.initial_v, final_p=self.final_p, final_v=self.final_v,
                                        initial_q=self.initial_q, initial_w=self.initial_w, final_q=self.final_q, final_w=self.final_w,
                                        initial_u=self.initial_u, final_u=self.final_u,
                                        A=A, B=B, C=C)
            error = self.problem.solve(verbose=False, solver=solver, ignore_dpp=False)
            # print("SCvx Solve time:", self.problem.get_solver_status().solve_time)
            # get solution
            new_X = self.problem.get_variable('X')
            new_U = self.problem.get_variable('U')
            new_sigma_1 = self.problem.get_variable('sigma_1')
            new_sigma_2 = self.problem.get_variable('sigma_2')
            linear_cost_dynamics = np.linalg.norm(self.problem.get_variable('nu'), 1, axis=0)  # virtual control
            # dx = new_X[0:3, :] - X[0:3, :]
            # dxk = [np.matmul(np.matmul(dx[:, k].T, w_tr), dx[:, k]) for k in range(K)]
            # J_vc = weight_nu * np.sum(linear_cost_dynamics)
            J_vc = np.sum(linear_cost_dynamics)
            # dxk = [np.matmul(np.matmul(dx[:, k].T, np.diag([1, 1, 1])), dx[:, k]) for k in range(K)]

            J_tr = np.sum(self.problem.get_variable('p'))

            # delta_u = sum(
            #     np.linalg.norm((np.array(new_U) - np.array(U)),
            #                 ord=2,
            #                 axis=0))

            # mu = problem.get_variable('mu')
            self.X = new_X
            self.U = new_U
            self.sigma_1 = new_sigma_1
            self.sigma_2 = new_sigma_2
            # print('')
            print('J_vc:',J_vc)
            print('J_tr:',J_tr)
            # print(self.sigma_1)
            # print(self.sigma_2)

            # print(format_line('Virtual Control Cost', J_vc))
            # print(format_line('Constraint Cost', linear_cost_constraints))
            # print('')
            # print(format_line('Actual change', max_nonlinear_diff))
            # print(format_line('Predicted change', predicted_change))

            # all_X.append(self.X.copy())
            # all_U.append(self.U.copy())
            if J_vc <= epsilon_vc and J_tr <= epsilon_tr:
                converged = True
                print(f'Converged after {it + 1} iterations.')
                self.converged = True
                break
            else:
                self.converged = False

            if J_vc <= epsilon_vc:
                self.dyn_feasible = True
            else:
                self.dyn_feasible = False

        if not converged:
            print('Jvc and Jtr: ', J_vc, J_tr)
        print('Integration time: ',t_integration)
        # all_X = np.stack(all_X)
        # all_U = np.stack(all_U)
        # all_sigma = np.array(all_sigma)

    def solve_replan(self):
        converged = False

        t_integration = 0
        
        for it in range(1):

            # ax = plt.figure().add_subplot(projection='3d')
            # ax.plot(self.X[0, :], self.X[1,:], self.X[2,:])
            # plt.savefig('traj_test.png')

            K_phase = self.K_phase

            t_init = time()
            A_bar_1, B_bar_1, C_bar_1, S_bar_1, z_bar_1 = self.integrator_obstacle.calculate_discretization(self.X[:, 0:K_phase], self.U[:, 0:K_phase], self.sigma_1, K_phase)
            A_bar_2, B_bar_2, C_bar_2, S_bar_2, z_bar_2 = self.integrator_landing.calculate_discretization(self.X[:, K_phase-1:self.K], self.U[:, K_phase-1:self.K], self.sigma_2, self.K-K_phase+1)
            t_integration += time()-t_init

            A_bar = np.concatenate((A_bar_1,A_bar_2),axis=1)
            B_bar = np.concatenate((B_bar_1,B_bar_2),axis=1)
            C_bar = np.concatenate((C_bar_1,C_bar_2),axis=1)
            S_bar = np.concatenate((S_bar_1,S_bar_2),axis=1)
            z_bar = np.concatenate((z_bar_1,z_bar_2),axis=1)

            A = self.R1-obstacle_constraint_violation(self.X[0:3, :self.K_phase-1].copy(), self.H1, self.p1)
            delta_r = (self.X[0:3, :self.K_phase-1] - np.tile(self.p1.reshape(3, 1), (1, self.K_phase-1)))
            xi = np.linalg.norm(np.dot(self.H1, delta_r), axis=0)
            zeta = np.dot(self.H1.transpose(), np.dot(self.H1, delta_r)).transpose()
            B = zeta / np.tile(np.expand_dims(xi, axis=-1), (1, 3))
            C = np.array([B[k, :] @ self.X[0:3, k] for k in range(self.K_phase-1)])
            if self.obs_num == 1:
                A = np.expand_dims(A, axis=-1)
                C = np.expand_dims(C, axis=-1)

            self.problem.set_parameters(X_last=self.X, A_bar=A_bar, B_bar=B_bar, C_bar=C_bar, S_bar=S_bar, z_bar=z_bar,
                                        sigma_last_1=self.sigma_1, sigma_last_2=self.sigma_2, initial_p=self.initial_p,
                                        initial_v=self.initial_v, final_p=self.final_p, final_v=self.final_v,
                                        initial_q=self.initial_q, initial_w=self.initial_w, final_q=self.final_q, final_w=self.final_w,
                                        initial_u=self.initial_u, final_u=self.final_u,
                                        A=A, B=B, C=C)
            error = self.problem.solve(verbose=False, solver=solver, ignore_dpp=False)
            # print("SCvx Solve time:", self.problem.get_solver_status().solve_time)
            # get solution
            new_X = self.problem.get_variable('X')
            new_U = self.problem.get_variable('U')
            new_sigma_1 = self.problem.get_variable('sigma_1')
            new_sigma_2 = self.problem.get_variable('sigma_2')
            linear_cost_dynamics = np.linalg.norm(self.problem.get_variable('nu'), 1, axis=0)  # virtual control
            # dx = new_X[0:3, :] - X[0:3, :]
            # dxk = [np.matmul(np.matmul(dx[:, k].T, w_tr), dx[:, k]) for k in range(K)]
            # J_vc = weight_nu * np.sum(linear_cost_dynamics)
            J_vc = np.sum(linear_cost_dynamics)
            # dxk = [np.matmul(np.matmul(dx[:, k].T, np.diag([1, 1, 1])), dx[:, k]) for k in range(K)]

            J_tr = np.sum(self.problem.get_variable('p'))

            # delta_u = sum(
            #     np.linalg.norm((np.array(new_U) - np.array(U)),
            #                 ord=2,
            #                 axis=0))

            # mu = problem.get_variable('mu')
            self.X = new_X
            self.U = new_U
            self.sigma_1 = new_sigma_1
            self.sigma_2 = new_sigma_2
            # print('')
            print('J_vc:',J_vc)
            print('J_tr:',J_tr)
            # print(self.sigma_1)
            # print(self.sigma_2)

            # print(format_line('Virtual Control Cost', J_vc))
            # print(format_line('Constraint Cost', linear_cost_constraints))
            # print('')
            # print(format_line('Actual change', max_nonlinear_diff))
            # print(format_line('Predicted change', predicted_change))

            # all_X.append(self.X.copy())
            # all_U.append(self.U.copy())
            if J_vc <= epsilon_vc and J_tr <= epsilon_tr:
                converged = True
                print(f'Converged after {it + 1} iterations.')
                self.converged = True
                break
            else:
                self.converged = False

            if J_vc <= epsilon_vc:
                self.dyn_feasible = True
                print(f'Replan after {it + 1} iterations.')
                break
            else:
                self.dyn_feasible = False

        if not converged:
            print('Jvc and Jtr: ', J_vc, J_tr)
        print('Integration time: ',t_integration)
        # all_X = np.stack(all_X)
        # all_U = np.stack(all_U)
        # all_sigma = np.array(all_sigma)

    def sub_odom(self, msg):
        # self.seq = msg.header.seq
        self.odom_current = msg
        self.pos_current = np.array((msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z))
    
class scvx_traj_moving_exp():
    def __init__(self, quad_name, K):
        # self.m = Model_6dof(quad_name)
        self.m = Model_6dof_moving(quad_name)

        self.seq = None

        # INITIALIZATION--------------------------------------------------------------------------------------------------------
        self.tf_guess_1 = 3.
        self.tf_guess_2 = 2.
        # self.delta_t = self.tf_guess_1 / (K - 1)

        self.obs_num = 1
        self.m.obs_num = self.obs_num
        self.R1 = 1.
        self.H1 = np.diag([1., 1., 1.])
        self.p1 = np.array((4, 4, 2))

        self.K = K
        K_phase = int(np.ceil(K/2))
        self.integrator_obstacle = FirstOrderHold_6dof(quad_name, self.m, K_phase)
        self.integrator_landing = FirstOrderHold_6dof(quad_name, self.m, self.K-K_phase+1)

        
        # START SCVX--------------------------------------------------------------------------------------

        self.problem = SCProblem_6dof_moving_exp(self.m, K)

        self.X = np.zeros(shape=[self.m.n_x, K])
        self.U = np.zeros(shape=[self.m.n_u, K])
        self.X, self.U = self.m.initialize_trajectory(self.X, self.U)

        self.sigma_1 = self.tf_guess_1
        self.sigma_2 = self.tf_guess_2
        # self.all_X = [self.X.copy()]
        # self.all_U = [self.U.copy()]
        # self.all_sigma = [self.sigma]

        # pre warm
        self.initial_p = np.array([1., 1, 1])
        self.initial_v = np.array([0, 0, 0])
        self.final_p = np.array([4, 4, 1])
        self.final_v = np.array([0, 0, 0])

        self.initial_q = np.array([1, 0, 0, 0])
        self.initial_w = np.array([0, 0, 0])
        self.final_q = np.array([1, 0, 0, 0])
        self.final_w = np.array([0, 0, 0])

        self.initial_u = self.m.mass*self.m.f_balance*np.ones(4)
        self.final_u = self.m.mass*self.m.f_balance*np.ones(4)

        K_phase = int(np.ceil(K/2))
        self.K_phase = K_phase
        A_bar_1, B_bar_1, C_bar_1, S_bar_1, z_bar_1 = self.integrator_obstacle.calculate_discretization(self.X[:, 0:K_phase], self.U[:, 0:K_phase], self.sigma_1, K_phase)
        A_bar_2, B_bar_2, C_bar_2, S_bar_2, z_bar_2 = self.integrator_landing.calculate_discretization(self.X[:, K_phase-1:self.K], self.U[:, K_phase-1:self.K], self.sigma_2, self.K-K_phase+1)

        A_bar = np.concatenate((A_bar_1,A_bar_2),axis=1)
        B_bar = np.concatenate((B_bar_1,B_bar_2),axis=1)
        C_bar = np.concatenate((C_bar_1,C_bar_2),axis=1)
        S_bar = np.concatenate((S_bar_1,S_bar_2),axis=1)
        z_bar = np.concatenate((z_bar_1,z_bar_2),axis=1)

        A = self.R1-obstacle_constraint_violation(self.X[0:3, :self.K_phase-1].copy(), self.H1, self.p1)
        delta_r = (self.X[0:3, :self.K_phase-1] - np.tile(self.p1.reshape(3, 1), (1, self.K_phase-1)))
        xi = np.linalg.norm(np.dot(self.H1, delta_r), axis=0)
        zeta = np.dot(self.H1.transpose(), np.dot(self.H1, delta_r)).transpose()
        B = zeta / np.tile(np.expand_dims(xi, axis=-1), (1, 3))
        C = np.array([B[k, :] @ self.X[0:3, k] for k in range(self.K_phase-1)])
        if self.obs_num == 1:
            A = np.expand_dims(A, axis=-1)
            C = np.expand_dims(C, axis=-1)

        self.problem.set_parameters(X_last=self.X, A_bar=A_bar, B_bar=B_bar, C_bar=C_bar, S_bar=S_bar, z_bar=z_bar,
                                    sigma_last_1=self.sigma_1, sigma_last_2=self.sigma_2, initial_p=self.initial_p,
                                    initial_v=self.initial_v, final_p=self.final_p, final_v=self.final_v,
                                    initial_q=self.initial_q, initial_w=self.initial_w, final_q=self.final_q, final_w=self.final_w,
                                    initial_u=self.initial_u, final_u=self.final_u,
                                    A=A,B=B,C=C)
        
        # A_bar_1, B_bar_1, S_bar_1, z_bar_1 = self.integrator.trapezoidal_rule(self.X[:, 0:K_phase], self.U[:, 0:K_phase], self.sigma_1)
        # A_bar_2, B_bar_2, S_bar_2, z_bar_2 = self.integrator.trapezoidal_rule(self.X[:, K_phase-1:self.K], self.U[:, K_phase-1:self.K], self.sigma_2)
        
        # delta_t = 1./(K_phase-1)
        # delta_t_2 = 1./(K-K_phase)

        # A_bar = np.concatenate((A_bar_1*delta_t/2,A_bar_2*delta_t_2/2),axis=1)
        # B_bar = np.concatenate((B_bar_1*delta_t/2,B_bar_2*delta_t_2/2),axis=1)
        # S_bar = np.concatenate((S_bar_1*delta_t/2,S_bar_2*delta_t_2/2),axis=1)
        # z_bar = np.concatenate((z_bar_1*delta_t/2,z_bar_2*delta_t_2/2),axis=1)

        # self.problem.set_parameters(X_last=self.X, A_bar=A_bar, B_bar=B_bar, S_bar=S_bar, z_bar=z_bar,
        #                             sigma_last_1=self.sigma_1, sigma_last_2=self.sigma_2, initial_p=self.initial_p,
        #                             initial_v=self.initial_v, final_p=self.final_p, final_v=self.final_v)
        
        
        # self.problem.get_is_dpp()
        # self.problem.get_is_dcp()
        error = self.problem.solve(verbose=False, solver=solver, ignore_dpp=False)

        # ax = plt.figure().add_subplot(projection='3d')
        # X = self.problem.get_variable('X')
        # ax.plot(X[0, :], X[1,:], X[2,:])
        # plt.show()

        # message
        # self.msg = PositionCommand()

    def set_parameters(self, A_bar, B_bar, C_bar, S_bar, z_bar):
        self.problem.set_parameters(X_last=self.X, A_bar=A_bar, B_bar=B_bar, C_bar=C_bar, S_bar=S_bar, z_bar=z_bar,
                                    sigma_last_1=self.sigma_1, sigma_last_2=self.sigma_2, initial_p=self.initial_p,
                                    initial_v=self.initial_v, final_p=self.final_p, final_v=self.final_v,
                                    initial_q=self.initial_q, initial_w=self.initial_w, final_q=self.final_q, final_w=self.final_w,
                                    initial_u=self.initial_u, final_u=self.final_u)

    def zero_initialize(self):
        self.X = np.zeros(shape=[self.m.n_x, K])
        self.U = np.zeros(shape=[self.m.n_u, K])

    def line_initialize(self):
        """
        Initialize the trajectory.

        :param X: Numpy array of states to be initialized
        :param U: Numpy array of inputs to be initialized
        :return: The initialized X and U
        """

        for k in range(K):
            alpha1 = (K - k - 1) / (K-1)
            alpha2 = k / (K-1)

            r_I_k = alpha1 * self.initial_p + alpha2 * self.final_p
            v_I_k = alpha1 * self.initial_v + alpha2 * self.final_v
            q_B_I_k = np.array([1, 0, 0, 0])
            w_B_k = alpha1 * self.initial_w + alpha2 * self.final_w

            self.X[:, k] = np.concatenate((r_I_k, v_I_k, q_B_I_k, w_B_k))
            # U[:, k] = (self.T_max-self.T_min)/2 * np.array([1, 1, 1, 1])
            self.U[:, k] = self.m.f_balance * np.array([1, 1, 1, 1])
            # U[:, k] = (self.T_max-self.T_min)/8 * np.array([1, 1, 1, 1])


    def set_initial_traj(self, X, U, tf1, tf2):
        self.X = X
        self.U = U
        self.sigma_1 = tf1
        self.sigma_2 = tf2

    def solve(self):
        converged = False

        t_integration = 0
        
        for it in range(iterations):

            # ax = plt.figure().add_subplot(projection='3d')
            # ax.plot(self.X[0, :], self.X[1,:], self.X[2,:])
            # plt.savefig('traj_test.png')

            K_phase = self.K_phase

            t_init = time()
            A_bar_1, B_bar_1, C_bar_1, S_bar_1, z_bar_1 = self.integrator_obstacle.calculate_discretization(self.X[:, 0:K_phase], self.U[:, 0:K_phase], self.sigma_1, K_phase)
            A_bar_2, B_bar_2, C_bar_2, S_bar_2, z_bar_2 = self.integrator_landing.calculate_discretization(self.X[:, K_phase-1:self.K], self.U[:, K_phase-1:self.K], self.sigma_2, self.K-K_phase+1)
            t_integration += time()-t_init

            A_bar = np.concatenate((A_bar_1,A_bar_2),axis=1)
            B_bar = np.concatenate((B_bar_1,B_bar_2),axis=1)
            C_bar = np.concatenate((C_bar_1,C_bar_2),axis=1)
            S_bar = np.concatenate((S_bar_1,S_bar_2),axis=1)
            z_bar = np.concatenate((z_bar_1,z_bar_2),axis=1)

            A = self.R1-obstacle_constraint_violation(self.X[0:3, :self.K_phase-1].copy(), self.H1, self.p1)
            delta_r = (self.X[0:3, :self.K_phase-1] - np.tile(self.p1.reshape(3, 1), (1, self.K_phase-1)))
            xi = np.linalg.norm(np.dot(self.H1, delta_r), axis=0)
            zeta = np.dot(self.H1.transpose(), np.dot(self.H1, delta_r)).transpose()
            B = zeta / np.tile(np.expand_dims(xi, axis=-1), (1, 3))
            C = np.array([B[k, :] @ self.X[0:3, k] for k in range(self.K_phase-1)])
            if self.obs_num == 1:
                A = np.expand_dims(A, axis=-1)
                C = np.expand_dims(C, axis=-1)

            self.problem.set_parameters(X_last=self.X, A_bar=A_bar, B_bar=B_bar, C_bar=C_bar, S_bar=S_bar, z_bar=z_bar,
                                        sigma_last_1=self.sigma_1, sigma_last_2=self.sigma_2, initial_p=self.initial_p,
                                        initial_v=self.initial_v, final_p=self.final_p, final_v=self.final_v,
                                        initial_q=self.initial_q, initial_w=self.initial_w, final_q=self.final_q, final_w=self.final_w,
                                        initial_u=self.initial_u, final_u=self.final_u,
                                        A=A, B=B, C=C)
            error = self.problem.solve(verbose=False, solver=solver, ignore_dpp=False)
            # print("SCvx Solve time:", self.problem.get_solver_status().solve_time)
            # get solution
            new_X = self.problem.get_variable('X')
            new_U = self.problem.get_variable('U')
            new_sigma_1 = self.problem.get_variable('sigma_1')
            new_sigma_2 = self.problem.get_variable('sigma_2')
            linear_cost_dynamics = np.linalg.norm(self.problem.get_variable('nu'), 1, axis=0)  # virtual control
            # dx = new_X[0:3, :] - X[0:3, :]
            # dxk = [np.matmul(np.matmul(dx[:, k].T, w_tr), dx[:, k]) for k in range(K)]
            # J_vc = weight_nu * np.sum(linear_cost_dynamics)
            J_vc = np.sum(linear_cost_dynamics)
            # dxk = [np.matmul(np.matmul(dx[:, k].T, np.diag([1, 1, 1])), dx[:, k]) for k in range(K)]

            J_tr = np.sum(self.problem.get_variable('p'))

            # delta_u = sum(
            #     np.linalg.norm((np.array(new_U) - np.array(U)),
            #                 ord=2,
            #                 axis=0))

            # mu = problem.get_variable('mu')
            self.X = new_X
            self.U = new_U
            self.sigma_1 = new_sigma_1
            self.sigma_2 = new_sigma_2
            # print('')
            print('J_vc:',J_vc)
            print('J_tr:',J_tr)
            # print(self.sigma_1)
            # print(self.sigma_2)

            # print(format_line('Virtual Control Cost', J_vc))
            # print(format_line('Constraint Cost', linear_cost_constraints))
            # print('')
            # print(format_line('Actual change', max_nonlinear_diff))
            # print(format_line('Predicted change', predicted_change))

            # all_X.append(self.X.copy())
            # all_U.append(self.U.copy())
            if J_vc <= epsilon_vc and J_tr <= epsilon_tr:
                converged = True
                print(f'Converged after {it + 1} iterations.')
                self.converged = True
                break
            else:
                self.converged = False

            if J_vc <= epsilon_vc:
                self.dyn_feasible = True
            else:
                self.dyn_feasible = False

        if not converged:
            print('Jvc and Jtr: ', J_vc, J_tr)
        print('Integration time: ',t_integration)
        # all_X = np.stack(all_X)
        # all_U = np.stack(all_U)
        # all_sigma = np.array(all_sigma)

    def solve_replan(self):
        converged = False

        t_integration = 0
        
        for it in range(1):

            # ax = plt.figure().add_subplot(projection='3d')
            # ax.plot(self.X[0, :], self.X[1,:], self.X[2,:])
            # plt.savefig('traj_test.png')

            K_phase = self.K_phase

            t_init = time()
            A_bar_1, B_bar_1, C_bar_1, S_bar_1, z_bar_1 = self.integrator_obstacle.calculate_discretization(self.X[:, 0:K_phase], self.U[:, 0:K_phase], self.sigma_1, K_phase)
            A_bar_2, B_bar_2, C_bar_2, S_bar_2, z_bar_2 = self.integrator_landing.calculate_discretization(self.X[:, K_phase-1:self.K], self.U[:, K_phase-1:self.K], self.sigma_2, self.K-K_phase+1)
            t_integration += time()-t_init

            A_bar = np.concatenate((A_bar_1,A_bar_2),axis=1)
            B_bar = np.concatenate((B_bar_1,B_bar_2),axis=1)
            C_bar = np.concatenate((C_bar_1,C_bar_2),axis=1)
            S_bar = np.concatenate((S_bar_1,S_bar_2),axis=1)
            z_bar = np.concatenate((z_bar_1,z_bar_2),axis=1)

            A = self.R1-obstacle_constraint_violation(self.X[0:3, :self.K_phase-1].copy(), self.H1, self.p1)
            delta_r = (self.X[0:3, :self.K_phase-1] - np.tile(self.p1.reshape(3, 1), (1, self.K_phase-1)))
            xi = np.linalg.norm(np.dot(self.H1, delta_r), axis=0)
            zeta = np.dot(self.H1.transpose(), np.dot(self.H1, delta_r)).transpose()
            B = zeta / np.tile(np.expand_dims(xi, axis=-1), (1, 3))
            C = np.array([B[k, :] @ self.X[0:3, k] for k in range(self.K_phase-1)])
            if self.obs_num == 1:
                A = np.expand_dims(A, axis=-1)
                C = np.expand_dims(C, axis=-1)

            self.problem.set_parameters(X_last=self.X, A_bar=A_bar, B_bar=B_bar, C_bar=C_bar, S_bar=S_bar, z_bar=z_bar,
                                        sigma_last_1=self.sigma_1, sigma_last_2=self.sigma_2, initial_p=self.initial_p,
                                        initial_v=self.initial_v, final_p=self.final_p, final_v=self.final_v,
                                        initial_q=self.initial_q, initial_w=self.initial_w, final_q=self.final_q, final_w=self.final_w,
                                        initial_u=self.initial_u, final_u=self.final_u,
                                        A=A, B=B, C=C)
            error = self.problem.solve(verbose=False, solver=solver, ignore_dpp=False)
            # print("SCvx Solve time:", self.problem.get_solver_status().solve_time)
            # get solution
            new_X = self.problem.get_variable('X')
            new_U = self.problem.get_variable('U')
            new_sigma_1 = self.problem.get_variable('sigma_1')
            new_sigma_2 = self.problem.get_variable('sigma_2')
            linear_cost_dynamics = np.linalg.norm(self.problem.get_variable('nu'), 1, axis=0)  # virtual control
            # dx = new_X[0:3, :] - X[0:3, :]
            # dxk = [np.matmul(np.matmul(dx[:, k].T, w_tr), dx[:, k]) for k in range(K)]
            # J_vc = weight_nu * np.sum(linear_cost_dynamics)
            J_vc = np.sum(linear_cost_dynamics)
            # dxk = [np.matmul(np.matmul(dx[:, k].T, np.diag([1, 1, 1])), dx[:, k]) for k in range(K)]

            J_tr = np.sum(self.problem.get_variable('p'))

            # delta_u = sum(
            #     np.linalg.norm((np.array(new_U) - np.array(U)),
            #                 ord=2,
            #                 axis=0))

            # mu = problem.get_variable('mu')
            self.X = new_X
            self.U = new_U
            self.sigma_1 = new_sigma_1
            self.sigma_2 = new_sigma_2
            # print('')
            print('J_vc:',J_vc)
            print('J_tr:',J_tr)
            # print(self.sigma_1)
            # print(self.sigma_2)

            # print(format_line('Virtual Control Cost', J_vc))
            # print(format_line('Constraint Cost', linear_cost_constraints))
            # print('')
            # print(format_line('Actual change', max_nonlinear_diff))
            # print(format_line('Predicted change', predicted_change))

            # all_X.append(self.X.copy())
            # all_U.append(self.U.copy())
            if J_vc <= epsilon_vc and J_tr <= epsilon_tr:
                converged = True
                print(f'Converged after {it + 1} iterations.')
                self.converged = True
                break
            else:
                self.converged = False

            if J_vc <= epsilon_vc:
                self.dyn_feasible = True
                print(f'Replan after {it + 1} iterations.')
                break
            else:
                self.dyn_feasible = False

        if not converged:
            print('Jvc and Jtr: ', J_vc, J_tr)
        print('Integration time: ',t_integration)
        # all_X = np.stack(all_X)
        # all_U = np.stack(all_U)
        # all_sigma = np.array(all_sigma)

    def sub_odom(self, msg):
        # self.seq = msg.header.seq
        self.odom_current = msg
        self.pos_current = np.array((msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z))
    

def euler_to_quat(yaw, roll, pitch):
    # ZXY convention
    # a = np.deg2rad(a)

    # cy = np.cos(a[1] * 0.5)
    # sy = np.sin(a[1] * 0.5)
    # cr = np.cos(a[0] * 0.5)
    # sr = np.sin(a[0] * 0.5)
    # cp = np.cos(a[2] * 0.5)
    # sp = np.sin(a[2] * 0.5)

    # q = np.zeros(4)

    # q[0] = cy * cr * cp + sy * sr * sp
    # q[1] = cy * sr * cp - sy * cr * sp
    # q[3] = cy * cr * sp + sy * sr * cp
    # q[2] = sy * cr * cp - cy * sr * sp
    c1 = np.cos(yaw * 0.5)
    s1 = np.sin(yaw * 0.5)
    c2 = np.cos(roll * 0.5)
    s2 = np.sin(roll * 0.5)
    c3 = np.cos(pitch * 0.5)
    s3 = np.sin(pitch * 0.5)

    q = np.zeros((4, yaw.shape[1]))

    q[0,:] = c1 * c2 * c3 - s1 * s2 * s3
    q[1,:] = c1 * s2 * c3 - s1 * c2 * s3
    q[3,:] = c1 * c2 * s3 + s1 * s2 * c3
    q[2,:] = s1 * c2 * c3 + c1 * s2 * s3

    return q

def quat2yaw(q):
    # ZXY convention

    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    # r11 = -2*(x*y - w*z)
    # r12 = w*w - x*x + y*y - z*z
    # r21 = 2*(y*z + w*x)
    # r31 = -2*(x*z - w*y)
    # r32 = w*w - x*x - y*y + z*z

    w2 = w * w
    x2 = x * x
    y2 = y * y
    z2 = z * z
    unitLength = w2 + x2 + y2 + z2
    abcd = w * x + y * z
    eps = 1e-7  # TODO: pick from your math lib instead of hardcoding.

    if (abcd > (0.5 - eps) * unitLength):

        yaw = 2 * np.arctan2(y, w)
        pitch = np.pi
        roll = 0

    elif (abcd < (-0.5 + eps) * unitLength):

        yaw = -2 * np.arctan2(y, w)
        pitch = -np.pi
        roll = 0

    else:
        adbc = w * z - x * y
        acbd = w * y - x * z
        yaw = np.arctan2(2 * adbc, 1 - 2 * (z2 + x2))
        pitch = np.arcsin(2 * abcd / unitLength)
        roll = np.arctan2(2 * acbd, 1 - 2 * (y2 + x2))

        return yaw


def scvx_landing(quad_name, goal, tf_guess, init_p, plot=False, replan=False, goal_v=np.array([0, 0, 0])):

    # position_f = goal.waypoints[0]
    # xf = position_f.position.x
    # yf = position_f.position.y
    # zf = position_f.position.z
    # p_f = np.array([xf,yf,zf])
    # tf_guess = goal.waypoint_times[0]
    # print(p_f, tf_guess)
    cvx_tracker = lcvx_traj(quad_name, K_3dof)

    cvx6dof_tracker = scvx_traj(quad_name, K)

    p_f = goal
    v_f = goal_v

    # cvx_tracker.tf_guess_1 = 0.7*tf_guess
    # cvx_tracker.tf_guess_2 = 0.3*tf_guess
    cvx_tracker.tf_guess_1 = 0.6*tf_guess
    cvx_tracker.tf_guess_2 = 0.4*tf_guess
    print(cvx_tracker.tf_guess_1,cvx_tracker.tf_guess_2)
    # cvx6dof_tracker.tf_guess = tf_guess

    cvx_tracker.initial_p = init_p
    cvx6dof_tracker.initial_p = init_p
    
    cvx_tracker.final_p = p_f
    cvx6dof_tracker.final_p = p_f
    cvx6dof_tracker.p1 = p_f

    cvx_tracker.final_v = v_f
    cvx6dof_tracker.final_v = v_f


    print('initial position:',cvx_tracker.initial_p)
    cvx_tracker.set_parameters()
    lcvx_t0 = time()
    cvx_tracker.solve()
    print('lcvx solve time: ', time()-lcvx_t0)
    
    new_X = cvx_tracker.problem.get_variable('X')
    new_U = cvx_tracker.problem.get_variable('U')

    cvx6dof_tracker.X = np.zeros(shape=[cvx6dof_tracker.m.n_x, K])
    cvx6dof_tracker.X[0:6, :] = new_X

    yaw = np.zeros(shape=[1, K])
    roll = np.arctan(-(new_U[1,:]/new_U[2,:]))
    pitch = np.arctan((new_U[0,:]/new_U[2,:])*np.cos(roll))
    q = euler_to_quat(yaw, roll, pitch)
    T_total = np.linalg.norm(new_U, axis=0)
    T_rotor = T_total/4

    cvx6dof_tracker.X[6:10, :] = q[:, 0:K]
    cvx6dof_tracker.U[:, :] = np.tile(T_rotor, (4,1))
    cvx6dof_tracker.sigma_1 = cvx_tracker.tf_guess_1
    cvx6dof_tracker.sigma_2 = cvx_tracker.tf_guess_2

    print('initial position:',cvx6dof_tracker.initial_p)
    t_solve_start = time()

    cvx6dof_tracker.solve()

    t_solve = time()-t_solve_start
    print("Tracker"+" Solve time: "+str(t_solve*1e3)+" ms")

    if plot:
        ax = plt.figure().add_subplot(projection='3d')
        X1 = cvx_tracker.problem.get_variable('X')
        X2 = cvx6dof_tracker.problem.get_variable('X')
        ax.plot(X1[0, :], X1[1,:], X1[2,:], label='3-DoF')
        ax.plot(X2[0, :], X2[1,:], X2[2,:], label='6-DoF')
        plot_obstacle(ax, cvx6dof_tracker.R1, cvx6dof_tracker.H1, cvx6dof_tracker.p1)
        plt.legend()
        # plt.savefig('traj.png')
        plt.show()

    # Start handle trajectory for MPC interface
    X_traj = cvx6dof_tracker.problem.get_variable('X').T
    U_traj = cvx6dof_tracker.problem.get_variable('U').T
    tf_1 = cvx6dof_tracker.problem.get_variable('sigma_1')
    tf_2 = cvx6dof_tracker.problem.get_variable('sigma_2')
    k_sum_1 = np.linspace(0, cvx6dof_tracker.K_phase - 1, cvx6dof_tracker.K_phase)
    k_sum_2 = np.linspace(0, (cvx6dof_tracker.K-cvx6dof_tracker.K_phase+1) - 1, cvx6dof_tracker.K-cvx6dof_tracker.K_phase+1)
    time_traj_1 = (k_sum_1 * (tf_1) / (cvx6dof_tracker.K_phase - 1))
    time_traj_2 = (k_sum_2 * (tf_2) / ((cvx6dof_tracker.K-cvx6dof_tracker.K_phase+1) - 1))
    time_traj = np.concatenate((time_traj_1, time_traj_2[1:]+time_traj_1[-1]))
    print("final_time:", time_traj[-1], tf_1, tf_2)
    print("final_state:", X_traj[-1,:])

    result_traj = np.concatenate((X_traj, U_traj, np.expand_dims(time_traj, axis=1)), axis=1)
    pd.DataFrame(result_traj).to_csv('scvx_traj1.csv')

    result_traj_3dof = np.concatenate((new_X, new_U), axis=0)
    pd.DataFrame(result_traj_3dof).to_csv('scvx_traj1_3dof.csv')

    if replan:
        return X_traj, time_traj, U_traj, cvx6dof_tracker
    else:
        return X_traj, time_traj, U_traj
    

def scvx_landing_no_3dof(quad_name, goal, tf_guess, init_p, plot=False, replan=False, goal_v=np.array([0, 0, 0])):

    # position_f = goal.waypoints[0]
    # xf = position_f.position.x
    # yf = position_f.position.y
    # zf = position_f.position.z
    # p_f = np.array([xf,yf,zf])
    # tf_guess = goal.waypoint_times[0]
    # print(p_f, tf_guess)
    # cvx_tracker = lcvx_traj(quad_name, K_3dof)

    cvx6dof_tracker = scvx_traj(quad_name, K)

    p_f = goal
    v_f = goal_v

    # cvx_tracker.tf_guess_1 = 0.7*tf_guess
    # cvx_tracker.tf_guess_2 = 0.3*tf_guess
    # cvx_tracker.tf_guess_1 = 0.6*tf_guess
    # cvx_tracker.tf_guess_2 = 0.4*tf_guess
    # print(cvx_tracker.tf_guess_1,cvx_tracker.tf_guess_2)
    # cvx6dof_tracker.tf_guess = tf_guess

    cvx6dof_tracker.sigma_1 = 0.6*tf_guess
    cvx6dof_tracker.sigma_2 = 0.4*tf_guess

    # cvx_tracker.initial_p = init_p
    cvx6dof_tracker.initial_p = init_p
    
    # cvx_tracker.final_p = p_f
    cvx6dof_tracker.final_p = p_f
    cvx6dof_tracker.p1 = p_f

    # cvx_tracker.final_v = v_f
    cvx6dof_tracker.final_v = v_f


    # print('initial position:',cvx_tracker.initial_p)
    # cvx_tracker.set_parameters()
    # lcvx_t0 = time()
    # cvx_tracker.solve()
    # print('lcvx solve time: ', time()-lcvx_t0)
    
    # new_X = cvx_tracker.problem.get_variable('X')
    # new_U = cvx_tracker.problem.get_variable('U')

    # cvx6dof_tracker.X = np.zeros(shape=[cvx6dof_tracker.m.n_x, K])
    # cvx6dof_tracker.X[0:6, :] = new_X

    # yaw = np.zeros(shape=[1, K])
    # roll = np.arctan(-(new_U[1,:]/new_U[2,:]))
    # pitch = np.arctan((new_U[0,:]/new_U[2,:])*np.cos(roll))
    # q = euler_to_quat(yaw, roll, pitch)
    # T_total = np.linalg.norm(new_U, axis=0)
    # T_rotor = T_total/4

    # cvx6dof_tracker.X[6:10, :] = q[:, 0:K]
    # cvx6dof_tracker.U[:, :] = np.tile(T_rotor, (4,1))
    # cvx6dof_tracker.sigma_1 = cvx_tracker.tf_guess_1
    # cvx6dof_tracker.sigma_2 = cvx_tracker.tf_guess_2

    cvx6dof_tracker.line_initialize()
    # cvx6dof_tracker.zero_initialize()
    X_init = cvx6dof_tracker.X

    print('initial position:',cvx6dof_tracker.initial_p)
    t_solve_start = time()

    cvx6dof_tracker.solve()

    t_solve = time()-t_solve_start
    print("Tracker"+" Solve time: "+str(t_solve*1e3)+" ms")

    if plot:
        ax = plt.figure().add_subplot(projection='3d')
        # X1 = cvx_tracker.problem.get_variable('X')
        X2 = cvx6dof_tracker.problem.get_variable('X')
        # ax.plot(X1[0, :], X1[1,:], X1[2,:], label='3-DoF')
        ax.plot(X_init[0, :], X_init[1,:], X_init[2,:], label='initialization')
        ax.plot(X2[0, :], X2[1,:], X2[2,:], label='6-DoF')
        plot_obstacle(ax, cvx6dof_tracker.R1, cvx6dof_tracker.H1, cvx6dof_tracker.p1)
        plt.legend()
        # plt.savefig('traj.png')
        plt.show()

    # Start handle trajectory for MPC interface
    X_traj = cvx6dof_tracker.problem.get_variable('X').T
    U_traj = cvx6dof_tracker.problem.get_variable('U').T
    tf_1 = cvx6dof_tracker.problem.get_variable('sigma_1')
    tf_2 = cvx6dof_tracker.problem.get_variable('sigma_2')
    k_sum_1 = np.linspace(0, cvx6dof_tracker.K_phase - 1, cvx6dof_tracker.K_phase)
    k_sum_2 = np.linspace(0, (cvx6dof_tracker.K-cvx6dof_tracker.K_phase+1) - 1, cvx6dof_tracker.K-cvx6dof_tracker.K_phase+1)
    time_traj_1 = (k_sum_1 * (tf_1) / (cvx6dof_tracker.K_phase - 1))
    time_traj_2 = (k_sum_2 * (tf_2) / ((cvx6dof_tracker.K-cvx6dof_tracker.K_phase+1) - 1))
    time_traj = np.concatenate((time_traj_1, time_traj_2[1:]+time_traj_1[-1]))
    print("final_time:", time_traj[-1], tf_1, tf_2)
    print("final_state:", X_traj[-1,:])

    result_traj = np.concatenate((X_traj, U_traj, np.expand_dims(time_traj, axis=1)), axis=1)
    pd.DataFrame(result_traj).to_csv('scvx_traj1_6dof.csv')
    init_traj = X_init
    pd.DataFrame(init_traj).to_csv('scvx_traj1_init.csv')
    # result_traj_3dof = np.concatenate((new_X, new_U), axis=0)
    # pd.DataFrame(result_traj_3dof).to_csv('scvx_traj1_3dof.csv')

    if replan:
        return X_traj, time_traj, U_traj, cvx6dof_tracker
    else:
        return X_traj, time_traj, U_traj


def scvx_landing_moving_exp(quad_name, goal, tf_guess, init_p, plot=False, replan=False, goal_v=np.array([0, 0, 0])):

    # position_f = goal.waypoints[0]
    # xf = position_f.position.x
    # yf = position_f.position.y
    # zf = position_f.position.z
    # p_f = np.array([xf,yf,zf])
    # tf_guess = goal.waypoint_times[0]
    # print(p_f, tf_guess)
    cvx_tracker = lcvx_traj(quad_name, K_3dof)

    cvx6dof_tracker = scvx_traj_moving_exp(quad_name, K)

    p_f = goal
    v_f = goal_v

    cvx_tracker.tf_guess_1 = 0.7*tf_guess
    cvx_tracker.tf_guess_2 = 0.3*tf_guess
    # cvx_tracker.tf_guess_1 = 0.6*tf_guess
    # cvx_tracker.tf_guess_2 = 0.4*tf_guess
    print(cvx_tracker.tf_guess_1,cvx_tracker.tf_guess_2)
    # cvx6dof_tracker.tf_guess = tf_guess

    cvx_tracker.initial_p = init_p
    cvx6dof_tracker.initial_p = init_p
    
    cvx_tracker.final_p = p_f
    cvx6dof_tracker.final_p = p_f
    cvx6dof_tracker.p1 = p_f

    cvx_tracker.final_v = v_f
    cvx6dof_tracker.final_v = v_f


    print('initial position:',cvx_tracker.initial_p)
    cvx_tracker.set_parameters()
    lcvx_t0 = time()
    cvx_tracker.solve()
    print('lcvx solve time: ', time()-lcvx_t0)
    
    new_X = cvx_tracker.problem.get_variable('X')
    new_U = cvx_tracker.problem.get_variable('U')

    cvx6dof_tracker.X = np.zeros(shape=[cvx6dof_tracker.m.n_x, K])
    cvx6dof_tracker.X[0:6, :] = new_X

    yaw = np.zeros(shape=[1, K])
    roll = np.arctan(-(new_U[1,:]/new_U[2,:]))
    pitch = np.arctan((new_U[0,:]/new_U[2,:])*np.cos(roll))
    q = euler_to_quat(yaw, roll, pitch)
    T_total = np.linalg.norm(new_U, axis=0)
    T_rotor = T_total/4

    cvx6dof_tracker.X[6:10, :] = q[:, 0:K]
    cvx6dof_tracker.U[:, :] = np.tile(T_rotor, (4,1))
    cvx6dof_tracker.sigma_1 = cvx_tracker.tf_guess_1
    cvx6dof_tracker.sigma_2 = cvx_tracker.tf_guess_2

    print('initial position:',cvx6dof_tracker.initial_p)
    t_solve_start = time()

    cvx6dof_tracker.solve()

    t_solve = time()-t_solve_start
    print("Tracker"+" Solve time: "+str(t_solve*1e3)+" ms")

    if plot:
        ax = plt.figure().add_subplot(projection='3d')
        X1 = cvx_tracker.problem.get_variable('X')
        X2 = cvx6dof_tracker.problem.get_variable('X')
        ax.plot(X1[0, :], X1[1,:], X1[2,:], label='3-DoF')
        ax.plot(X2[0, :], X2[1,:], X2[2,:], label='6-DoF')
        plot_obstacle(ax, cvx6dof_tracker.R1, cvx6dof_tracker.H1, cvx6dof_tracker.p1)
        plt.legend()
        # plt.savefig('traj.png')
        plt.show()

    # Start handle trajectory for MPC interface
    X_traj = cvx6dof_tracker.problem.get_variable('X').T
    U_traj = cvx6dof_tracker.problem.get_variable('U').T
    tf_1 = cvx6dof_tracker.problem.get_variable('sigma_1')
    tf_2 = cvx6dof_tracker.problem.get_variable('sigma_2')
    k_sum_1 = np.linspace(0, cvx6dof_tracker.K_phase - 1, cvx6dof_tracker.K_phase)
    k_sum_2 = np.linspace(0, (cvx6dof_tracker.K-cvx6dof_tracker.K_phase+1) - 1, cvx6dof_tracker.K-cvx6dof_tracker.K_phase+1)
    time_traj_1 = (k_sum_1 * (tf_1) / (cvx6dof_tracker.K_phase - 1))
    time_traj_2 = (k_sum_2 * (tf_2) / ((cvx6dof_tracker.K-cvx6dof_tracker.K_phase+1) - 1))
    time_traj = np.concatenate((time_traj_1, time_traj_2[1:]+time_traj_1[-1]))
    print("final_time:", time_traj[-1], tf_1, tf_2)
    print("final_state:", X_traj[-1,:])

    result_traj = np.concatenate((X_traj, U_traj, np.expand_dims(time_traj, axis=1)), axis=1)
    pd.DataFrame(result_traj).to_csv('scvx_traj1.csv')

    result_traj_3dof = np.concatenate((new_X, new_U), axis=0)
    pd.DataFrame(result_traj_3dof).to_csv('scvx_traj1_3dof.csv')

    if replan:
        return X_traj, time_traj, U_traj, cvx6dof_tracker
    else:
        return X_traj, time_traj, U_traj


def scvx_landing_moving_exp_no_3dof(quad_name, goal, tf_guess, init_p, plot=False, replan=False, goal_v=np.array([0, 0, 0])):

    # position_f = goal.waypoints[0]
    # xf = position_f.position.x
    # yf = position_f.position.y
    # zf = position_f.position.z
    # p_f = np.array([xf,yf,zf])
    # tf_guess = goal.waypoint_times[0]
    # print(p_f, tf_guess)
    # cvx_tracker = lcvx_traj(quad_name, K_3dof)

    cvx6dof_tracker = scvx_traj_moving_exp(quad_name, K)

    p_f = goal
    v_f = goal_v

    # cvx_tracker.tf_guess_1 = 0.7*tf_guess
    # cvx_tracker.tf_guess_2 = 0.3*tf_guess
    # cvx_tracker.tf_guess_1 = 0.6*tf_guess
    # cvx_tracker.tf_guess_2 = 0.4*tf_guess
    # print(cvx_tracker.tf_guess_1,cvx_tracker.tf_guess_2)
    # cvx6dof_tracker.tf_guess = tf_guess

    cvx6dof_tracker.sigma_1 = 0.6*tf_guess
    cvx6dof_tracker.sigma_2 = 0.4*tf_guess

    # cvx_tracker.initial_p = init_p
    cvx6dof_tracker.initial_p = init_p
    
    # cvx_tracker.final_p = p_f
    cvx6dof_tracker.final_p = p_f
    cvx6dof_tracker.p1 = p_f

    # cvx_tracker.final_v = v_f
    cvx6dof_tracker.final_v = v_f


    # print('initial position:',cvx_tracker.initial_p)
    # cvx_tracker.set_parameters()
    # lcvx_t0 = time()
    # cvx_tracker.solve()
    # print('lcvx solve time: ', time()-lcvx_t0)
    
    # new_X = cvx_tracker.problem.get_variable('X')
    # new_U = cvx_tracker.problem.get_variable('U')

    # cvx6dof_tracker.X = np.zeros(shape=[cvx6dof_tracker.m.n_x, K])
    # cvx6dof_tracker.X[0:6, :] = new_X

    # yaw = np.zeros(shape=[1, K])
    # roll = np.arctan(-(new_U[1,:]/new_U[2,:]))
    # pitch = np.arctan((new_U[0,:]/new_U[2,:])*np.cos(roll))
    # q = euler_to_quat(yaw, roll, pitch)
    # T_total = np.linalg.norm(new_U, axis=0)
    # T_rotor = T_total/4

    # cvx6dof_tracker.X[6:10, :] = q[:, 0:K]
    # cvx6dof_tracker.U[:, :] = np.tile(T_rotor, (4,1))
    # cvx6dof_tracker.sigma_1 = cvx_tracker.tf_guess_1
    # cvx6dof_tracker.sigma_2 = cvx_tracker.tf_guess_2

    cvx6dof_tracker.line_initialize()
    # cvx6dof_tracker.zero_initialize()
    X_init = cvx6dof_tracker.X

    print('initial position:',cvx6dof_tracker.initial_p)
    t_solve_start = time()

    cvx6dof_tracker.solve()

    t_solve = time()-t_solve_start
    print("Tracker"+" Solve time: "+str(t_solve*1e3)+" ms")

    if plot:
        ax = plt.figure().add_subplot(projection='3d')
        # X1 = cvx_tracker.problem.get_variable('X')
        X2 = cvx6dof_tracker.problem.get_variable('X')
        # ax.plot(X1[0, :], X1[1,:], X1[2,:], label='3-DoF')
        ax.plot(X_init[0, :], X_init[1,:], X_init[2,:], label='initialization')
        ax.plot(X2[0, :], X2[1,:], X2[2,:], label='6-DoF')
        plot_obstacle(ax, cvx6dof_tracker.R1, cvx6dof_tracker.H1, cvx6dof_tracker.p1)
        plt.legend()
        # plt.savefig('traj.png')
        plt.show()

    # Start handle trajectory for MPC interface
    X_traj = cvx6dof_tracker.problem.get_variable('X').T
    U_traj = cvx6dof_tracker.problem.get_variable('U').T
    tf_1 = cvx6dof_tracker.problem.get_variable('sigma_1')
    tf_2 = cvx6dof_tracker.problem.get_variable('sigma_2')
    k_sum_1 = np.linspace(0, cvx6dof_tracker.K_phase - 1, cvx6dof_tracker.K_phase)
    k_sum_2 = np.linspace(0, (cvx6dof_tracker.K-cvx6dof_tracker.K_phase+1) - 1, cvx6dof_tracker.K-cvx6dof_tracker.K_phase+1)
    time_traj_1 = (k_sum_1 * (tf_1) / (cvx6dof_tracker.K_phase - 1))
    time_traj_2 = (k_sum_2 * (tf_2) / ((cvx6dof_tracker.K-cvx6dof_tracker.K_phase+1) - 1))
    time_traj = np.concatenate((time_traj_1, time_traj_2[1:]+time_traj_1[-1]))
    print("final_time:", time_traj[-1], tf_1, tf_2)
    print("final_state:", X_traj[-1,:])

    result_traj = np.concatenate((X_traj, U_traj, np.expand_dims(time_traj, axis=1)), axis=1)
    pd.DataFrame(result_traj).to_csv('scvx_traj1_6dof.csv')
    init_traj = X_init
    pd.DataFrame(init_traj).to_csv('scvx_traj1_init.csv')

    # result_traj_3dof = np.concatenate((new_X, new_U), axis=0)
    # pd.DataFrame(result_traj_3dof).to_csv('scvx_traj1_3dof.csv')

    if replan:
        return X_traj, time_traj, U_traj, cvx6dof_tracker
    else:
        return X_traj, time_traj, U_traj



def scvx_landing_new(quad_name, goal, tf_guess, init_p, plot=False, replan=False): # add 3 DoF obstacle

    # position_f = goal.waypoints[0]
    # xf = position_f.position.x
    # yf = position_f.position.y
    # zf = position_f.position.z
    # p_f = np.array([xf,yf,zf])
    # tf_guess = goal.waypoint_times[0]
    # print(p_f, tf_guess)
    cvx_tracker = lcvx_traj(quad_name, K_3dof)
    cvx_tracker_o = lcvx_traj_obstacle(quad_name, K_3dof)

    cvx6dof_tracker = scvx_traj(quad_name, K)

    p_f = goal

    cvx_tracker.tf_guess_1 = 0.7*tf_guess
    cvx_tracker.tf_guess_2 = 0.5*tf_guess
    print(cvx_tracker.tf_guess_1,cvx_tracker.tf_guess_2)
    # cvx6dof_tracker.tf_guess = tf_guess

    cvx_tracker.initial_p = init_p
    cvx6dof_tracker.initial_p = init_p
    
    cvx_tracker.final_p = p_f
    cvx6dof_tracker.final_p = p_f
    cvx6dof_tracker.p1 = p_f

    cvx_tracker_o.initial_p = init_p
    cvx_tracker_o.final_p = p_f
    cvx_tracker_o.p1 = p_f
    cvx_tracker_o.tf_guess_1 = 0.7*tf_guess
    cvx_tracker_o.tf_guess_2 = 0.3*tf_guess


    print('initial position:',cvx_tracker.initial_p)
    cvx_tracker.set_parameters()
    lcvx_t0 = time()
    cvx_tracker.solve()
    print('lcvx solve time: ', time()-lcvx_t0)
    
    new_X = cvx_tracker.problem.get_variable('X')
    new_U = cvx_tracker.problem.get_variable('U')

    cvx_tracker_o.X = new_X
    cvx_tracker_o.U = new_U

    print('initial position:',cvx_tracker_o.initial_p)
    cvx_tracker_o.set_parameters()
    lcvx_t0 = time()
    cvx_tracker_o.solve()
    print('lcvx solve time: ', time()-lcvx_t0)

    new_X = cvx_tracker_o.problem.get_variable('X')
    new_U = cvx_tracker_o.problem.get_variable('U')

    cvx6dof_tracker.X = np.zeros(shape=[cvx6dof_tracker.m.n_x, K])
    cvx6dof_tracker.X[0:6, :] = new_X

    yaw = np.zeros(shape=[1, K])
    roll = np.arctan(-(new_U[1,:]/new_U[2,:]))
    pitch = np.arctan((new_U[0,:]/new_U[2,:])*np.cos(roll))
    q = euler_to_quat(yaw, roll, pitch)
    T_total = np.linalg.norm(new_U, axis=0)
    T_rotor = T_total/4

    cvx6dof_tracker.X[6:10, :] = q[:, 0:K]
    cvx6dof_tracker.U[:, :] = np.tile(T_rotor, (4,1))
    cvx6dof_tracker.sigma_1 = cvx_tracker.tf_guess_1
    cvx6dof_tracker.sigma_2 = cvx_tracker.tf_guess_2

    print('initial position:',cvx6dof_tracker.initial_p)
    t_solve_start = time()

    cvx6dof_tracker.solve()

    t_solve = time()-t_solve_start
    print("Tracker"+" Solve time: "+str(t_solve*1e3)+" ms")

    if plot:
        ax = plt.figure().add_subplot(projection='3d')
        X1 = cvx_tracker.problem.get_variable('X')
        X2 = cvx6dof_tracker.problem.get_variable('X')
        X3 = cvx_tracker_o.problem.get_variable('X')
        ax.plot(X1[0, :], X1[1,:], X1[2,:], label='3-DoF')
        ax.plot(X3[0, :], X3[1,:], X3[2,:], '.', label='3-DoF-obsracle')
        ax.plot(X2[0, :], X2[1,:], X2[2,:], label='6-DoF')
        plot_obstacle(ax, cvx6dof_tracker.R1, cvx6dof_tracker.H1, cvx6dof_tracker.p1)
        plt.legend()
        # plt.savefig('traj.png')
        plt.show()

    # Start handle trajectory for MPC interface
    X_traj = cvx6dof_tracker.problem.get_variable('X').T
    U_traj = cvx6dof_tracker.problem.get_variable('U').T
    tf_1 = cvx6dof_tracker.problem.get_variable('sigma_1')
    tf_2 = cvx6dof_tracker.problem.get_variable('sigma_2')
    k_sum_1 = np.linspace(0, cvx6dof_tracker.K_phase - 1, cvx6dof_tracker.K_phase)
    k_sum_2 = np.linspace(0, (cvx6dof_tracker.K-cvx6dof_tracker.K_phase+1) - 1, cvx6dof_tracker.K-cvx6dof_tracker.K_phase+1)
    time_traj_1 = (k_sum_1 * (tf_1) / (cvx6dof_tracker.K_phase - 1))
    time_traj_2 = (k_sum_2 * (tf_2) / ((cvx6dof_tracker.K-cvx6dof_tracker.K_phase+1) - 1))
    time_traj = np.concatenate((time_traj_1, time_traj_2[1:]+time_traj_1[-1]))

    result_traj = np.concatenate((X_traj, U_traj, np.expand_dims(time_traj, axis=1)), axis=1)
    pd.DataFrame(result_traj).to_csv('scvx_traj1.csv')

    result_traj_3dof = np.concatenate((new_X, new_U), axis=1)
    pd.DataFrame(result_traj_3dof).to_csv('scvx_traj1_3dof.csv')

    if replan:
        return X_traj, time_traj, U_traj, cvx6dof_tracker
    else:
        return X_traj, time_traj, U_traj