#! /usr/bin/env python3
import rospy
import actionlib

import numpy as np
import pandas as pd

from time import time
import matplotlib.pyplot as plt

from scripts.Configuration.parameters_3dof_landing_twophase import *
from scripts.Configuration.parameters_6dof_landing_twophase import *
from scripts.SCP.discretization_3dof_landing_twophase import FirstOrderHold
from scripts.SCP.discretization_noloop_x_landing_twophase_compile import calculate_discretization
# from SCP.discretization_6dof import FirstOrderHold as FirstOrderHold_6dof
from scripts.SCP.scproblem_3dof_landing_twophase import SCProblem
from scripts.SCP.scproblem_quad_6dof_socp_landing_twophase import SCProblem as SCProblem_6dof
# from utils import format_line
from scripts.Models.quadrotor_3dof_landing_twophase import Model
from scripts.Models.quadrotor_6dof_x_landing_twophase import Model as Model_6dof

from scripts.utils import obstacle_constraint_violation, plot_obstacle
from multiprocessing.pool import Pool
import threading


class MyThread(threading.Thread):

    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result   # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None



class lcvx_traj():
    def __init__(self, K):
        self.m = Model()

        self.seq = None

        # INITIALIZATION--------------------------------------------------------------------------------------------------------
        self.tf_guess_1 = 3.
        self.tf_guess_2 = 2.
        # self.delta_t = self.tf_guess / (K - 1)
        self.integrator = FirstOrderHold(self.m, K)
        self.K = K
        # START LCVX--------------------------------------------------------------------------------------

        self.problem = SCProblem(self.m, K)

        # pre warm
        self.initial_p = np.array([1, 1, 1])
        self.initial_v = np.array([0, 0, 0])
        self.final_p = np.array([4, 4, 1])
        self.final_v = np.array([0, 0, 0])

        self.problem.set_parameters(delta_t=(self.tf_guess_1 / (K - 1)), delta_t2=(self.tf_guess_1 / (K - 1)) ** 2, 
                                    delta_t_2=(self.tf_guess_2 / (K - 1)), delta_t2_2=(self.tf_guess_2 / (K - 1)) ** 2, initial_p=self.initial_p,
                                    initial_v=self.initial_v, final_p=self.final_p, final_v=self.final_v)
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
                                    initial_v=self.initial_v, final_p=self.final_p, final_v=self.final_v)

    def solve(self):
        self.problem.solve(verbose=False, solver=solver, ignore_dpp=False)

    def sub_odom(self, msg):
        # self.seq = msg.header.seq
        self.odom_current = msg
        self.pos_current = np.array((msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z))


class scvx_traj():
    def __init__(self, K):
        self.m = Model_6dof()

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
        # self.integrator_obstacle = FirstOrderHold_6dof(self.m, K_phase)
        # self.integrator_landing = FirstOrderHold_6dof(self.m, self.K-K_phase+1)

        
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

        K_phase = int(np.ceil(K/2))
        self.K_phase = K_phase
        A_bar_1, B_bar_1, C_bar_1, S_bar_1, z_bar_1 = calculate_discretization(self.m.n_x, self.m.n_u, self.m.mass, self.X[:, 0:K_phase], self.U[:, 0:K_phase], self.sigma_1, K_phase)
        A_bar_2, B_bar_2, C_bar_2, S_bar_2, z_bar_2 = calculate_discretization(self.m.n_x, self.m.n_u, self.m.mass, self.X[:, K_phase-1:self.K], self.U[:, K_phase-1:self.K], self.sigma_2, self.K-K_phase+1)

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
                                    initial_v=self.initial_v, final_p=self.final_p, final_v=self.final_v)

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

            # pool = Pool(2)
            # results = pool.starmap(calculate_discretization, [(self.m.n_x, self.m.n_u, self.m.mass, self.X[:, 0:K_phase], self.U[:, 0:K_phase], self.sigma_1, K_phase),
            #                                               (self.m.n_x, self.m.n_u, self.m.mass, self.X[:, K_phase-1:self.K], self.U[:, K_phase-1:self.K], self.sigma_2, self.K-K_phase+1)])
            # # result1 = pool.apply_async(calculate_discretization, args=(self.m.n_x, self.m.n_u, self.m.mass, self.X[:, 0:K_phase], self.U[:, 0:K_phase], self.sigma_1, K_phase))
            # # result2 = pool.apply_async(calculate_discretization, args=(self.m.n_x, self.m.n_u, self.m.mass, self.X[:, K_phase-1:self.K], self.U[:, K_phase-1:self.K], self.sigma_2, self.K-K_phase+1))
            # pool.close()
            # pool.join()
            # A_bar_1, B_bar_1, C_bar_1, S_bar_1, z_bar_1 = results[0]
            # A_bar_2, B_bar_2, C_bar_2, S_bar_2, z_bar_2 = results[1]

            # t1=MyThread(func=calculate_discretization, args=(self.m.n_x, self.m.n_u, self.m.mass, self.X[:, 0:K_phase], self.U[:, 0:K_phase], self.sigma_1, K_phase, ))
            # t2=MyThread(func=calculate_discretization, args=(self.m.n_x, self.m.n_u, self.m.mass, self.X[:, K_phase-1:self.K], self.U[:, K_phase-1:self.K], self.sigma_2, self.K-K_phase+1, ))
            
            # t1.start()
            # t2.start()

            # t1.join()
            # t2.join()
            # A_bar_1, B_bar_1, C_bar_1, S_bar_1, z_bar_1 = t1.get_result()
            # A_bar_2, B_bar_2, C_bar_2, S_bar_2, z_bar_2 = t2.get_result()

            A_bar_1, B_bar_1, C_bar_1, S_bar_1, z_bar_1 = calculate_discretization(self.m.n_x, self.m.n_u, self.m.mass, self.X[:, 0:K_phase], self.U[:, 0:K_phase], self.sigma_1, K_phase)
            A_bar_2, B_bar_2, C_bar_2, S_bar_2, z_bar_2 = calculate_discretization(self.m.n_x, self.m.n_u, self.m.mass, self.X[:, K_phase-1:self.K], self.U[:, K_phase-1:self.K], self.sigma_2, self.K-K_phase+1)
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
            if J_vc < epsilon_vc and J_tr < epsilon_tr:
                converged = True
                print(f'Converged after {it + 1} iterations.')
                break

        if not converged:
            print('Jvc and Jtr: ', J_vc, J_tr)
        print('Integration time (ms): ',t_integration*1e3)
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


def scvx_landing(goal, tf_guess, init_p, plot=False):

    # position_f = goal.waypoints[0]
    # xf = position_f.position.x
    # yf = position_f.position.y
    # zf = position_f.position.z
    # p_f = np.array([xf,yf,zf])
    # tf_guess = goal.waypoint_times[0]
    # print(p_f, tf_guess)
    cvx_tracker = lcvx_traj(K_3dof)

    cvx6dof_tracker = scvx_traj(K)

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

    return X_traj, time_traj, U_traj

