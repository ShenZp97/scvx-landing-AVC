import cvxpy as cvx
import numpy as np
# from cvxpygen import cpg
from scripts.Configuration.parameters_3dof_landing_twophase import weight_nu, w_tr

class SCProblem:
    """
    Defines a standard Successive Convexification problem and adds the model specific constraints and objectives.

    :param m: The model object
    :param K: Number of discretization points
    """

    def __init__(self, m, K):
        # Variables:
        self.K = K
        self.n_x = m.n_x
        self.n_u = m.n_u
        self.tan_gamma_gs = m.tan_gamma_gs
        # self.sigma = m.sigma
        self.g_I = m.g_I
        self.mass = m.mass
        self.get_constr = m.get_constraints
        self.var = dict()
        self.var['X'] = cvx.Variable((m.n_x, K))
        self.var['U'] = cvx.Variable((m.n_u, K))
        # self.var['nu'] = cvx.Variable((m.n_x, K - 1))
        # self.var['sigma'] = cvx.Variable(nonneg=True)
        # self.var['eta'] = cvx.Variable((K, ), nonneg=True)

        # self.var['slack_o'] = cvx.Variable(nonneg=True)
        # Parameters:
        self.par = dict()
        # self.par['delta_t'] = cvx.Parameter(nonneg=True)
        # self.par['delta_t2'] = cvx.Parameter(nonneg=True)
        self.par['delta_t_2'] = cvx.Parameter(nonneg=True)
        self.par['delta_t2_2'] = cvx.Parameter(nonneg=True)
        self.par['initial_p'] = cvx.Parameter((3, ), nonneg=False)
        self.par['initial_v'] = cvx.Parameter((3,), nonneg=False)
        self.par['final_p'] = cvx.Parameter((3, ), nonneg=False)
        self.par['final_v'] = cvx.Parameter((3, ), nonneg=False)

        self.par['initial_u'] = cvx.Parameter((3,), nonneg=False)
        self.par['final_u'] = cvx.Parameter((3,), nonneg=False)


        # self.par['A_bar'] = cvx.Parameter((m.n_x * m.n_x, K - 1))
        # self.par['B_bar'] = cvx.Parameter((m.n_x * m.n_u, K - 1))
        # self.par['C_bar'] = cvx.Parameter((m.n_x * m.n_u, K - 1))
        # self.par['S_bar'] = cvx.Parameter((m.n_x, K - 1))
        # self.par['z_bar'] = cvx.Parameter((m.n_x, K - 1))

        # self.par['X_last'] = cvx.Parameter((m.n_x, K))
        # self.par['U_last'] = cvx.Parameter((m.n_u, K))
        # self.par['sigma_last'] = cvx.Parameter(nonneg=True)
        # self.par['B_g'] = cvx.Parameter((K, m.n_u))

        # self.par['weight_f'] = cvx.Parameter(nonneg=True)
        # self.par['weight_nu'] = cvx.Parameter(nonneg=True)
        # self.par['weight_tr'] = cvx.Parameter((K, ), nonneg=True)
        # self.par['tr_radius'] = cvx.Parameter(nonneg=True)

        # self.par['xi'] = cvx.Parameter((K, ), nonneg=True)
        # self.par['zeta'] = cvx.Parameter((K, 3))
        # self.par['zeta_dpp'] = cvx.Parameter((K, ))

        # Constraints:
        constraints = []

        # Model:
        constraints += m.get_constraints(self.var['X'], self.var['U'], self.par['initial_p'], self.par['initial_v'], self.par['final_p'], self.par['final_v'],
                                         self.par['initial_u'], self.par['initial_u'])

        # Dynamics:
        # K_phase = int(np.ceil(K/2))
        # for k in range(K_phase - 1):
        #     constraints += [
        #         self.var['X'][0:3, k + 1] ==
        #         self.var['X'][0:3, k]
        #         + self.par['delta_t'] * self.var['X'][3:6, k]
        #         + ((self.par['delta_t2']) * ((self.var['U'][:, k + 1] + 2. * self.var['U'][:, k])/self.mass - (3. * -m.g_I)))/6.,

        #         self.var['X'][3:6, k + 1] ==
        #         self.var['X'][3:6, k]
        #         + (self.par['delta_t'] * (
        #                 (self.var['U'][:, k + 1] + self.var['U'][:, k]) / self.mass - (2. * -m.g_I))) / 2.,

        #     ]

        for k in range(0, self.K - 1):
            constraints += [
                self.var['X'][0:3, k + 1] ==
                self.var['X'][0:3, k]
                + self.par['delta_t_2'] * self.var['X'][3:6, k]
                + ((self.par['delta_t2_2']) * ((self.var['U'][:, k + 1] + 2. * self.var['U'][:, k])/self.mass - (3. * -m.g_I)))/6.,

                self.var['X'][3:6, k + 1] ==
                self.var['X'][3:6, k]
                + (self.par['delta_t_2'] * (
                        (self.var['U'][:, k + 1] + self.var['U'][:, k]) / self.mass - (2. * -m.g_I))) / 2.,
                #landing cone:        
                # cvx.norm(self.var['X'][0: 2, k] - self.par['final_p'][0:2], axis=0) <= (
                #             self.var['X'][2, k] - self.par['final_p'][2]) * self.tan_gamma_gs,

                # self.var['X'][2, k] >= self.par['final_p'][2]

            ]

        # height = 1.
        # constraints += [cvx.norm(self.var['X'][0: 3, K_phase-1] - (self.par['final_p'][0:3] + np.array([0, 0, height])), axis=0) <= height*np.sin(np.arctan(self.tan_gamma_gs))]

        # i = ((self.par['delta_t']) * (self.var['U'][:, K-1] + 2. * self.var['U'][:, K-2] - (3. * -m.g_I)))/6.
        # print(i.is_dcp(dpp=True))
        # for i in constraints:
        #     print(i.is_dcp(dpp=True))

        # Trust region:
        # du = self.var['U'] - self.par['U_last']
        # dx = self.var['X'] - self.par['X_last']
        # ds = self.var['sigma'] - self.par['sigma_last']
        # constraints += [cvx.norm(dx, 1) + cvx.norm(du, 1) + cvx.norm(ds, 1) <= self.par['tr_radius']]
        # constraints += [cvx.norm(dx, 2, axis=0)**2 + cvx.norm(du, 2, axis=0)**2 <= self.var['eta']]
        # constraints += [cvx.norm(dx, 2, axis=0) ** 2 + cvx.norm(du, 2, axis=0) ** 2 <= self.var['eta']]

        # Objective:
        # model_objective = m.get_objective(self.var['X'], self.var['U'], self.par['X_last'], self.par['U_last'])
        model_objective = None
        # sc_objective = cvx.Minimize(
        #     self.par['weight_sigma'] * self.var['sigma']
        #     + self.par['weight_nu'] * cvx.norm(self.var['nu'], 1)
        # )
        sc_objective = cvx.Minimize(
            cvx.sum(m.sigma)
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

