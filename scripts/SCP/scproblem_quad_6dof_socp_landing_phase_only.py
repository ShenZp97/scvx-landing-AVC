import cvxpy as cvx
import numpy as np
# from cvxpygen import cpg
from scripts.Configuration.parameters_6dof_landing_twophase import weight_nu, weight_dx

class SCProblem:
    """
    Defines a standard Successive Convexification problem and adds the model specific constraints and objectives.

    :param m: The model object
    :param K: Number of discretization points
    """

    def __init__(self, m, K):
        # Variables:
        self.K = K
        K_phase = int(np.ceil(K/2))
        K_landing = K-K_phase+1

        self.n_x = m.n_x
        self.n_u = m.n_u
        # self.sigma = m.sigma
        self.g_I = m.g_I
        self.tan_gamma_gs = m.tan_gamma_gs
        self.mass = m.mass
        # self.get_constr = m.get_constraints
        self.var = dict()
        self.var['X'] = cvx.Variable((m.n_x, K_landing))
        self.var['U'] = cvx.Variable((m.n_u, K_landing))
        self.var['nu'] = cvx.Variable((self.n_x, K_landing - 1))
        self.var['nu_max'] = cvx.Variable((K_landing - 1,))
        # self.var['nu_max'] = cvx.Variable((1, self.K - 1))
        self.var['u_max'] = cvx.Variable((K_landing,))
        self.var['sigma_1'] = cvx.Variable(nonneg=True)
        self.var['sigma_2'] = cvx.Variable(nonneg=True)

        # waypoints
        # self.gate_num = 1
        # self.gate1 = np.array((14., 8., 1.))
        # self.gate1 = np.array((0., 4., 2.))
        # self.gate2 = np.array((4., 4., 3.))
        # self.d_tol = 0.3**2
        # self.var['lambda'] = cvx.Variable((self.gate_num, self.K), nonneg=True)
        # self.var['mu'] = cvx.Variable((self.gate_num, self.K - 1), nonneg=True)
        self.var['p'] = cvx.Variable((1, K_landing), nonneg=True)  # for soft constraints

        # Parameters:
        self.par = dict()
        # self.par['delta_t'] = cvx.Parameter(nonneg=True)
        # self.par['delta_t2'] = cvx.Parameter(nonneg=True)

        # self.var['nu'] = cvx.Variable(obs_num, nonneg=True)
        # self.par['A1'] = cvx.Parameter((self.K - 1, 1))
        # self.par['B1'] = cvx.Parameter((self.K - 1, 3))
        # self.par['C1'] = cvx.Parameter((self.K - 1, ))
        # self.par['A2'] = cvx.Parameter((self.K - 2, 1))
        # self.par['B2'] = cvx.Parameter((self.K - 2, 3))
        # self.par['C2'] = cvx.Parameter((self.K - 2, ))

        # self.par['C'] = cvx.Parameter((self.K, obs_num))
        self.par['A_bar'] = cvx.Parameter((self.n_x * self.n_x, K_landing- 1))
        self.par['B_bar'] = cvx.Parameter((self.n_x * self.n_u, K_landing - 1))
        self.par['C_bar'] = cvx.Parameter((self.n_x * self.n_u, K_landing - 1))
        self.par['S_bar'] = cvx.Parameter((self.n_x, K_landing - 1))
        self.par['z_bar'] = cvx.Parameter((self.n_x, K_landing - 1))

        # self.par['tr_radius'] = cvx.Parameter(nonneg=True)
        self.par['X_last'] = cvx.Parameter((self.n_x, K_landing))
        # self.par['sigma_last_1'] = cvx.Parameter(nonneg=True)
        self.par['sigma_last_2'] = cvx.Parameter(nonneg=True)

        self.par['initial_p'] = cvx.Parameter((3,), nonneg=False)
        self.par['initial_v'] = cvx.Parameter((3,), nonneg=False)
        self.par['final_p'] = cvx.Parameter((3,), nonneg=False)
        self.par['final_v'] = cvx.Parameter((3,), nonneg=False)

        self.par['initial_q'] = cvx.Parameter((4,), nonneg=False)
        self.par['initial_w'] = cvx.Parameter((3,), nonneg=False)
        self.par['final_q'] = cvx.Parameter((4,), nonneg=False)
        self.par['final_w'] = cvx.Parameter((3,), nonneg=False)

        self.par['initial_u'] = cvx.Parameter((4,), nonneg=False)
        self.par['final_u'] = cvx.Parameter((4,), nonneg=False)

        self.rotor_T_max = cvx.Parameter(nonneg=True)
        self.rotor_T_min = cvx.Parameter(nonneg=True)

        # self.par['A'] = cvx.Parameter((K_phase-1, m.obs_num))
        # self.par['B'] = cvx.Parameter((K_phase-1, 3*m.obs_num))
        # self.par['C'] = cvx.Parameter((K_phase-1, m.obs_num))

        # Constraints:
        constraints = []

        # Model:
        # constraints += self.get_constr(self.var['X'], self.var['U'], self.par['initial_p'], self.par['initial_v'], self.par['final_p'], self.par['final_v'])
        constraints = [
            # X_v[0:3, 0] == self.x_init[0:3],
            # X_v[3:6, 0] == self.x_init[3:6],
            # self.var['X'][6:10, 0] == m.x_init[6:10],
            # self.var['X'][10:13, 0] == m.x_init[10:13],

            # X_v[:, -1] == self.x_final[:],

            self.var['X'][0:3, 0] == self.par['initial_p'],
            self.var['X'][3:6, 0] == self.par['initial_v'],
            self.var['X'][0:3, -1] == self.par['final_p'],
            self.var['X'][3:6, -1] == self.par['final_v'],

            # X_v[0:3, -1] == self.x_final[0:3],
            # X_v[3:6, -1] == self.x_final[3:6],

            # self.var['X'][6:10, -1] == m.x_final[6:10],
            # self.var['X'][10:13, -1] == m.x_final[10:13],

            # self.var['U'][:, 0] == m.u_init,
            
            # self.var['U'][:, -1] == m.u_final,

            self.var['X'][6:10, 0] == self.par['initial_q'],
            self.var['X'][10:13, 0] == self.par['initial_w'],

            self.var['X'][6:10, -1] == self.par['final_q'],
            self.var['X'][10:13, -1] == self.par['final_w'],

            self.var['U'][:, 0] == self.par['initial_u'],
            
            self.var['U'][:, -1] == self.par['final_u'],

            # - X_v[2, :] <= 0,
        ]

        self.rotor_T_max.value = m.T_max/4
        self.rotor_T_min.value = m.T_min/4

        constraints += [
            # State constraints:
            # cvx.norm(X_v[1: 3, :], axis=0) <= (X_v[3, :]) * self.tan_gamma_gs,  # glideslope
            # cvx.norm(X_v[8:10, :], axis=0) <= np.sqrt((1 - self.cos_theta_max) / 2),  # maximum angle
            # cvx.norm(X_v[11: 14, :], 'inf', axis=0) <= self.w_B_max,  # maximum angular velocity
            # cvx.norm(X_v[3: 6, :], axis=0) <= self.v_max,

            self.var['X'][10: 12, :] <= m.omega_max_xy,
            self.var['X'][10: 12, :] >= -m.omega_max_xy,
            self.var['X'][12, :] <= m.omega_max_z,
            self.var['X'][12, :] >= -m.omega_max_z,

            # Control constraints:
            self.var['U'][:, :] <= self.rotor_T_max,
            -self.var['U'][:, :] <= -self.rotor_T_min

        ]


        # Dynamics:
        # constraints += [
        #     self.var['X'][0:3, k + 1] ==
        #     self.var['X'][0:3, k]
        #     + self.par['delta_t'] * self.var['X'][3:6, k]
        #     + ((self.par['delta_t2']) * (
        #                 (self.var['U'][:, k + 1] + 2. * self.var['U'][:, k]) / self.mass - (3. * -self.g_I))) / 6.
        #     + self.var['nu'][0:3, k]
        #     for k in range(self.K - 1)
        # ]
        # constraints += [
        #     self.var['X'][3:6, k + 1] ==
        #     self.var['X'][3:6, k]
        #     + (self.par['delta_t'] * (
        #             (self.var['U'][:, k + 1] + self.var['U'][:, k]) / self.mass - (2. * -self.g_I))) / 2.
        #     + self.var['nu'][3:6, k]
        #     for k in range(self.K - 1)
        # ]
        dx = self.var['X'][:, :] - self.par['X_last'][:, :]
        # ds = cvx.norm(self.var['sigma'] - self.par['sigma_last'])
        # dxk = cvx.vstack(dx[:, :])


        # for k in range(K_phase - 1):
        #     constraints += [
        #         self.var['X'][:, k + 1] ==
        #         cvx.reshape(self.par['A_bar'][:, k], (self.n_x, self.n_x)) @ self.var['X'][:, k]
        #         + cvx.reshape(self.par['B_bar'][:, k], (self.n_x, self.n_u)) @ self.var['U'][:, k]
        #         + cvx.reshape(self.par['C_bar'][:, k], (self.n_x, self.n_u)) @ self.var['U'][:, k + 1]
        #         + self.par['S_bar'][:, k] * self.var['sigma_1']
        #         + self.par['z_bar'][:, k]
        #         + self.var['nu'][:, k],
        #         self.par['A'][k] - self.par['B'][k, :] @ (self.var['X'][0:3, k]) + self.par['C'][k] <= 0,
                
        #     ]
        #     constraints += [cvx.norm(dx[:, k+1]) <= self.var['p'][0, k],
        #                     cvx.norm(self.var['nu'][:, k]) <= self.var['nu_max'][k]
        #                     ]

        for k in range(0, self.K - K_phase):
            constraints += [
                self.var['X'][:, k + 1] ==
                cvx.reshape(self.par['A_bar'][:, k], (self.n_x, self.n_x)) @ self.var['X'][:, k]
                + cvx.reshape(self.par['B_bar'][:, k], (self.n_x, self.n_u)) @ self.var['U'][:, k]
                + cvx.reshape(self.par['C_bar'][:, k], (self.n_x, self.n_u)) @ self.var['U'][:, k + 1]
                + self.par['S_bar'][:, k] * self.var['sigma_2']
                + self.par['z_bar'][:, k]
                + self.var['nu'][:, k],
                # self.var['X'][2, k] >= self.par['final_p'][2]
            ]
            
            # landing cone constraint for replanning
            if k >= 100:
                constraints += [cvx.norm(self.var['X'][0: 2, k] - self.par['final_p'][0:2], axis=0) <= (self.var['X'][2, k] - self.par['final_p'][2]) * self.tan_gamma_gs
                                ]

            # constraints += [
            #     self.par['A1'][k] + self.par['B1'][k, :] @ self.var['X'][0:3, k + 1] + self.par['C1'][k] *
            #     self.var['mu'][
            #         0, k] <= 0,
            #     # self.par['A2'][k] + self.par['B2'][k, :] @ self.var['X'][0:3, k + 1] + self.par['C2'][k] * self.var['mu'][
            #     #     1, k] <= 0
            # ]
            constraints += [
                cvx.norm(dx[:, k+1]) <= self.var['p'][0, k],
                cvx.norm(self.var['nu'][:, k]) <= self.var['nu_max'][k]
            ]
            # constraints += [
            #     cvx.norm(self.var['nu'][:, k]) <= self.var['nu_max'][0, k],
            #
            # ]

        # terminal position for 1st phase
        # height = 1.
        # constraints += [cvx.norm(self.var['X'][0: 3, K_phase-1] - (self.par['final_p'][0:3] + np.array([0, 0, height])), axis=0) <= height*np.sin(np.arctan(self.tan_gamma_gs)),
        #                 # -self.var['X'][2, :] <= -0.5
        #                 ]

        # constraints += [cvx.norm(self.var['sigma_1'] - self.par['sigma_last_1']) <= self.var['p'][0, -2]]
        constraints += [cvx.norm(self.var['sigma_2'] - self.par['sigma_last_2']) <= self.var['p'][0, -1]]

        # constraints += [self.var['lambda'][:, 1:] == self.var['lambda'][:, 0:-1] - self.var['mu'][:, :],
        #                 # self.var['lambda'][0, :] <= self.var['lambda'][1, :] # sequence
        #                 ]

        # for k in range(self.K - 2):
        #     constraints += [
        #         self.par['A1'][k] + self.par['B1'][k, :] @ self.var['X'][0:3, k + 1] + self.par['C1'][k] * self.var['mu'][
        #             0, k] <= 0,
        #         # self.par['A2'][k] + self.par['B2'][k, :] @ self.var['X'][0:3, k + 1] + self.par['C2'][k] * self.var['mu'][
        #         #     1, k] <= 0
        #     ]

        # waypoints racing
        # constraints += [
        #     self.var['mu'] >= 0,
        #     # self.var['mu'] <= 1,
        #     # self.var['lambda'] >= 0,
        #     # self.var['lambda'] <= 1,
        #     self.var['lambda'][:, 0] == 1,
        #     self.var['lambda'][:, -1] == 0
        #     ]

        # constraints += [
        #     self.par['A'][k] - self.par['B'][k, :] @ (self.var['X'][0:3, k]) + self.par['C'][k] <= 0 for k in
        #     range(self.K)
        # ]

        # constraints += [
        #     self.var['sigma'] <= 8.,
        #     self.var['sigma'] >= 0.1
        # ]

        constraints += [cvx.norm(self.var['U'], 2, axis=0)[k] <= self.var['u_max'][k] for k in range(K_landing)]

        # dx = self.var['X'][0:3, :] - self.par['X_last'][0:3, :]
        # constraints += [cvx.norm(dx, 1) <= self.par['tr_radius']]

        model_objective = None



        sc_objective = cvx.Minimize(
                                    cvx.sum(self.var['u_max']) +
                                    # self.var['sigma']
                                    # + weight_nu * cvx.sum(cvx.norm(self.var['nu'], 1, axis=0))
                                    + weight_nu * cvx.sum(self.var['nu_max'])
                                    + weight_dx * cvx.sum(self.var['p'])
                                    # + weight_dx * ds
            # self.par['weight_f'] * cvx.sum(cvx.norm(self.var['U'], axis=0))
            # + self.var['slack_o']
        )

        objective = sc_objective if model_objective is None else sc_objective + model_objective

        self.prob = cvx.Problem(objective, constraints)

    def set_parameters(self, **kwargs):
        """
        All parameters have to be filled before calling solve().
        """

        for key in kwargs:
            if key in self.par:
                self.par[key].value = kwargs[key]
            else:
                print(f'Parameter \'{key}\' does not exist.')

    def print_available_parameters(self):
        print('Parameter names:')
        for key in self.par:
            print(f'\t {key}')
        print('\n')

    def print_available_variables(self):
        print('Variable names:')
        for key in self.var:
            print(f'\t {key}')
        print('\n')

    def get_variable(self, name):
        """
        :param name: Name of the variable.
        :return The value of the variable.
        """

        if name in self.var:
            return self.var[name].value
        else:
            print(f'Variable \'{name}\' does not exist.')
            return None

    def solve(self, **kwargs):
        error = False
        try:
            self.prob.solve(**kwargs)
        except cvx.SolverError:
            error = True

        return error

    def get_objective(self):
        return self.prob.value

    def get_status(self):
        return self.prob.status

    def get_solver_status(self):
        return self.prob.solver_stats

    # def solve_cvxpygen(self, **kwargs):
    #     error = False
    #     try:
    #         self.prob.solve(**kwargs)
    #     except cvx.SolverError:
    #         error = True
    #
    #     return error

    # def generate_cvxpygen(self, code_dir='landing_problem', solver='ECOS'):
    #     cpg.generate_code(self.prob, code_dir=code_dir, solver=solver)
    #
    # def load_cvxpygen(self, method_name='cpg'):
    #     from landing_problem.cpg_solver import cpg_solve
    #     self.prob.register_solve(method_name, cpg_solve)

    def get_is_dcp(self):
        print("Problem is DCP:", self.prob.is_dcp())

    def get_is_dpp(self):
        print("Problem is DPP:", self.prob.is_dcp(dpp=True))


