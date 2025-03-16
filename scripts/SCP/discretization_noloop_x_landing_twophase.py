import numpy as np
from scipy.integrate import solve_ivp
# from numba import jit


class ODE_solution(object) :
    def __init__(self) :
        pass
    def setter(self,y,t) :
        self.y = y
        self.t = t


def RK4(odefun, tspan, y0, args, N_RK=10) :
    t = np.linspace(tspan[0],tspan[-1],N_RK)
    h = t[1] - t[0]
    iy = len(y0)
    y_sol = np.zeros((N_RK,iy))
    y_sol[0] = y0
    for idx in range(0,N_RK-1) :
        tk = t[idx]
        yk = y_sol[idx]
        k1 = odefun(tk,yk,*args)
        k2 = odefun(tk + h/2,yk + h/2*k1,*args)
        k3 = odefun(tk + h/2,yk + h/2*k2,*args)
        k4 = odefun(tk+h,yk+h*k3,*args)
        y_sol[idx+1] = yk + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    sol = ODE_solution()
    sol.setter(y_sol.T,t)
    return sol


class FirstOrderHold:
    def __init__(self, quad_name, m, K):
        self.K = K
        self.m = m
        self.n_x = m.n_x
        self.n_u = m.n_u
        self.mass = m.mass
        self.quad_name = quad_name

        self.A_bar = np.zeros([m.n_x * m.n_x, K - 1])
        self.B_bar = np.zeros([m.n_x * m.n_u, K - 1])
        self.C_bar = np.zeros([m.n_x * m.n_u, K - 1])
        self.S_bar = np.zeros([m.n_x, K - 1])
        self.z_bar = np.zeros([m.n_x, K - 1])

        # vector indices for flat matrices
        x_end = m.n_x
        A_bar_end = m.n_x * (1 + m.n_x)
        B_bar_end = m.n_x * (1 + m.n_x + m.n_u)
        C_bar_end = m.n_x * (1 + m.n_x + m.n_u + m.n_u)
        S_bar_end = m.n_x * (1 + m.n_x + m.n_u + m.n_u + 1)
        z_bar_end = m.n_x * (1 + m.n_x + m.n_u + m.n_u + 2)
        self.x_ind = slice(0, x_end)
        self.A_bar_ind = slice(x_end, A_bar_end)
        self.B_bar_ind = slice(A_bar_end, B_bar_end)
        self.C_bar_ind = slice(B_bar_end, C_bar_end)
        self.S_bar_ind = slice(C_bar_end, S_bar_end)
        self.z_bar_ind = slice(S_bar_end, z_bar_end)

        self.f, self.A, self.B = m.get_equations()

        self.dt = 1. / (K - 1)

    def calculate_discretization(self,x,u,tf,K) :
        x = x.T
        u = u.T
        delT = tf/(K-1)
        self.K = K
        self.dt = 1. / (K - 1)
        assert abs(delT*(K-1) - tf) < 1e-6
        # differentiate dynamics and cost
        # if self.type_discretization == 'zoh' :
        #     A,B,s,z,x_prop_n = self.model.diff_discrete_zoh(x[0:self.N,:],u[0:self.N,:],delT,tf)
        #     x_prop = np.squeeze(A@np.expand_dims(x[0:self.N,:],2) +
        #                     B@np.expand_dims(u[0:self.N,:],2) +
        #                     np.expand_dims(tf*s+z,2))
        #     Bm = np.copy(B)
        #     Bp = np.copy(B)
        # elif self.type_discretization == 'foh' :
        #     # A,Bm,Bp,s,z,x_prop_n = self.model.diff_discrete_foh(x[0:self.N,:],u,delT,tf)
        #     # A,Bm,Bp,s,z,x_prop_n = self.model.diff_discrete_foh_tau(x[0:self.N,:],u,1/self.N,tf)
        #     A,Bm,Bp,s,z,x_prop_n = self.model.diff_discrete_foh_variational(x[0:self.N,:],u,1/self.N,tf)
        #     x_prop = np.squeeze(A@np.expand_dims(x[0:self.N,:],2) +
        #                     Bm@np.expand_dims(u[0:self.N,:],2) +
        #                     Bp@np.expand_dims(u[1:self.N+1,:],2) +
        #                     np.expand_dims(tf*s+z,2))
        #     B = np.copy(Bm)

        A,Bm,Bp,s,z,x_prop_n = self.diff_discrete_foh_variational(x[0:(K-1),:],u,tf)
        # A, Bm, Bp, s, z, x_prop_n = self.diff_discrete_foh_var_vectorized(x[0:(self.K - 1), :], u, tf)

        # remove small element
        eps_machine = np.finfo(float).eps
        A[np.abs(A) < eps_machine] = 0
        # B[np.abs(B) < eps_machine] = 0
        Bm[np.abs(Bm) < eps_machine] = 0
        Bp[np.abs(Bp) < eps_machine] = 0
        for k in range(K - 1):
            self.A_bar[:, k] = A[k, :, :].flatten(order='F')
            self.B_bar[:, k] = Bm[k, :, :].flatten(order='F')
            self.C_bar[:, k] = Bp[k, :, :].flatten(order='F')
            self.S_bar = s.T
            self.z_bar = z.T
        return self.A_bar,self.B_bar,self.C_bar,self.S_bar,self.z_bar

    # def _ode_dVdt(self, V, t, u_t0, u_t1, sigma):
    def _ode_dVdt(self, t, V, u_t0, u_t1, sigma):
        """
        ODE function to compute dVdt.

        :param V: Evaluation state V = [x, Phi_A, B_bar, C_bar, S_bar, z_bar]
        :param t: Evaluation time
        :param u_t0: Input at start of interval
        :param u_t1: Input at end of interval
        :param sigma: Total time
        :return: Derivative at current time and state dVdt
        """
        alpha = (self.dt - t) / self.dt
        beta = t / self.dt
        x = V[self.x_ind]
        u = u_t0 + (t / self.dt) * (u_t1 - u_t0)

        # using \Phi_A(\tau_{k+1},\xi) = \Phi_A(\tau_{k+1},\tau_k)\Phi_A(\xi,\tau_k)^{-1}
        # and pre-multiplying with \Phi_A(\tau_{k+1},\tau_k) after integration
        Phi_A_xi = np.linalg.inv(V[self.A_bar_ind].reshape((self.n_x, self.n_x)))

        A_subs = sigma * self.A(x, u)
        B_subs = sigma * self.B(x, u)
        f_subs = self.f(x, u)

        dVdt = np.zeros_like(V)
        dVdt[self.x_ind] = sigma * f_subs.transpose()
        dVdt[self.A_bar_ind] = np.matmul(A_subs, V[self.A_bar_ind].reshape((self.n_x, self.n_x))).reshape(-1)
        dVdt[self.B_bar_ind] = np.matmul(Phi_A_xi, B_subs).reshape(-1) * alpha
        dVdt[self.C_bar_ind] = np.matmul(Phi_A_xi, B_subs).reshape(-1) * beta
        dVdt[self.S_bar_ind] = np.matmul(Phi_A_xi, f_subs).transpose()
        z_t = -np.matmul(A_subs, x) - np.matmul(B_subs, u)
        dVdt[self.z_bar_ind] = np.matmul(Phi_A_xi, z_t)

        return dVdt

    def integrate_nonlinear_piecewise(self, X_l, U, sigma):
        """
        Piecewise integration to verfify accuracy of linearization.
        :param X_l: Linear state evolution
        :param U: Linear input evolution
        :param sigma: Total time
        :return: The piecewise integrated dynamics
        """
        X_nl = np.zeros_like(X_l)
        X_nl[:, 0] = X_l[:, 0]

        for k in range(self.K - 1):
            # X_nl[:, k + 1] = odeint(self._dx, X_l[:, k],
            #                         (0, self.dt * sigma),
            #                         args=(U[:, k], U[:, k + 1], sigma), tfirst=True)[1, :]
            X_nl[:, k + 1] = solve_ivp(self._dx, (0, self.dt * sigma), X_l[:, k], args=(U[:, k], U[:, k + 1], sigma)).y[
                             :, -1]

        return X_nl

    def integrate_nonlinear_full(self, x0, U, sigma):
        """
        Simulate nonlinear behavior given an initial state and an input over time.
        :param x0: Initial state
        :param U: Linear input evolution
        :param sigma: Total time
        :return: The full integrated dynamics
        """
        X_nl = np.zeros([x0.size, self.K])
        X_nl[:, 0] = x0

        for k in range(self.K - 1):
            # X_nl[:, k + 1] = odeint(self._dx, X_nl[:, k],
            #                         (0, self.dt * sigma),
            #                         args=(U[:, k], U[:, k + 1], sigma), tfirst=True)[1, :]
            X_nl[:, k + 1] = solve_ivp(self._dx, (0, self.dt * sigma), X_nl[:, k],
                                       args=(U[:, k], U[:, k + 1], sigma)).y[:, -1]

        return X_nl

    def _dx(self, t, x, u_t0, u_t1, sigma):
        u = u_t0 + (t / (self.dt * sigma)) * (u_t1 - u_t0)

        return np.squeeze(self.f(x, u))

    def diff_discrete_foh_variational(self, x, u, tf):

        dtau = self.dt
        ix = self.n_x
        iu = self.n_u

        ndim = np.ndim(x)
        if ndim == 1:  # 1 step state & input
            N = 1
            x = np.expand_dims(x, axis=0)
            u = np.expand_dims(u, axis=0)
        else:
            N = np.size(x, axis=0)
        idx_state = slice(0, ix)
        idx_A = slice(ix, ix + ix * ix)
        idx_Bm = slice(ix + ix * ix, ix + ix * ix + ix * iu)
        idx_Bp = slice(ix + ix * ix + ix * iu, ix + ix * ix + 2 * ix * iu)
        idx_s = slice(ix + ix * ix + 2 * ix * iu, ix + ix * ix + 2 * ix * iu + ix)

        # idx_z = slice(ix+ix*ix+2*ix*iu+ix,ix+ix*ix+2*ix*iu+ix+ix)
        def dvdt(t, V, um, up, length):
            assert len(um) == len(up)
            assert len(um) == length
            alpha = (dtau - t) / dtau
            beta = t / dtau
            u = alpha * um + beta * up
            # V = V.reshape((length,ix + ix*ix + 2*ix*iu + ix + ix)).transpose()
            V = V.reshape((length, ix + ix * ix + 2 * ix * iu + ix)).transpose()
            x = V[:ix].transpose()
            Phi = V[ix:ix * ix + ix]
            Phi = Phi.transpose().reshape((length, ix, ix))
            x3 = V[idx_Bm].transpose().reshape(length, ix, iu)
            x4 = V[idx_Bp].transpose().reshape(length, ix, iu)
            x5 = V[idx_s].transpose().reshape(length, ix, 1)
            # x6 = V[idx_z].transpose().reshape(length,ix,1)
            f = self.forward(x, u)
            # if self.type_linearization == "numeric_central":
            #     A, B = self.diff_numeric_central(x, u)
            # elif self.type_linearization == "numeric_forward":
            #     A, B = self.diff_numeric(x, u)
            # elif self.type_linearization == "analytic":
            #     A, B = self.diff(x, u)

            A, B = self.diff_numeric_central(x, u)
            # A, B = self.diff(x, u)
            A, B = tf * A, tf * B
            dpdt = np.matmul(A, Phi).reshape((length, ix * ix)).transpose()
            dbmdt = (A @ x3 + B * alpha).reshape((length, ix * iu)).transpose()
            dbpdt = (A @ x4 + B * beta).reshape((length, ix * iu)).transpose()
            dsdt = np.squeeze(A @ x5 + np.expand_dims(f, 2)).transpose()
            # dzdt = np.squeeze(A@x6 - A@np.expand_dims(x,2) - B@np.expand_dims(u,2)).transpose()
            # dv = np.vstack((tf*f.transpose(),dpdt,dbmdt,dbpdt,dsdt,dzdt))
            dv = np.vstack((tf * f.transpose(), dpdt, dbmdt, dbpdt, dsdt))
            return dv.flatten(order='F')

        A0 = np.eye(ix).flatten()
        Bm0 = np.zeros((ix * iu))
        Bp0 = np.zeros((ix * iu))
        s0 = np.zeros(ix)
        # z0 = np.zeros(ix)
        # V0 = np.array([np.hstack((x[i],A0,Bm0,Bp0,s0,z0)) for i in range(N)]).transpose()
        V0 = np.array([np.hstack((x[i], A0, Bm0, Bp0, s0)) for i in range(N)]).transpose()
        V0_repeat = V0.flatten(order='F')

        # sol = solve_ivp(dvdt,(0,dtau),V0_repeat,args=(u[0:N],u[1:],N),rtol=1e-6,atol=1e-10)
        # sol = solve_ivp(dvdt,(0,dtau),V0_repeat,args=(u[0:N],u[1:],N))
        sol = RK4(dvdt, (0, dtau), V0_repeat, args=(u[0:N], u[1:], N), N_RK=6)

        sol = sol.y[:, -1].reshape((N, -1))
        x_prop = sol[:, idx_state].reshape((-1, ix))
        A = sol[:, idx_A].reshape((-1, ix, ix))
        Bm = sol[:, idx_Bm].reshape((-1, ix, iu))
        Bp = sol[:, idx_Bp].reshape((-1, ix, iu))
        s = sol[:, idx_s].reshape((-1, ix, 1)).squeeze()
        # z = sol[:,idx_z].reshape((-1,ix,1)).squeeze()
        z = x_prop - np.squeeze(A @ np.expand_dims(x[0:N, :], 2) +
                                Bm @ np.expand_dims(u[0:N, :], 2) +
                                Bp @ np.expand_dims(u[1:N + 1, :], 2) +
                                np.expand_dims(tf * s, 2))

        return A, Bm, Bp, s, z, x_prop

    def forward(self, x, u, idx=None):
        xdim = np.ndim(x)
        if xdim == 1:  # 1 step state & input
            N = 1
            x = np.expand_dims(x, axis=0)
        else:
            N = np.size(x, axis=0)
        udim = np.ndim(u)
        if udim == 1:
            u = np.expand_dims(u, axis=0)

        # state & input
        rx = x[:, 0]
        ry = x[:, 1]
        rz = x[:, 2]
        vx = x[:, 3]
        vy = x[:, 4]
        vz = x[:, 5]
        q0 = x[:, 6]
        q1 = x[:, 7]
        q2 = x[:, 8]
        q3 = x[:, 9]
        wx = x[:, 10]
        wy = x[:, 11]
        wz = x[:, 12]

        u1 = u[:, 0]
        u2 = u[:, 1]
        u3 = u[:, 2]
        u4 = u[:, 3]

        # output
        f = np.zeros_like(x)
        f[:, 0] = vx
        f[:, 1] = vy
        f[:, 2] = vz
        f[:, 3] = (2/self.mass)*(q0*q2 + q1*q3)*(u1 + u2 + u3 + u4)
        f[:, 4] = (2/self.mass)*(q0*q1 - q2*q3)*(-u1 - u2 - u3 - u4)
        f[:, 5] = -(1 / self.mass) * u1 * (2 * q1 ** 2 + 2 * q2 ** 2 - 1) - (1 / self.mass) * u2 * (
                    2 * q1 ** 2 + 2 * q2 ** 2 - 1) - (1 / self.mass) * u3 * (2 * q1 ** 2 + 2 * q2 ** 2 - 1) - (
                              1 / self.mass) * u4 * (2 * q1 ** 2 + 2 * q2 ** 2 - 1) - 9.81
        f[:, 6] = -0.5*q1*wx - 0.5*q2*wy - 0.5*q3*wz
        f[:, 7] = 0.5*q0*wx + 0.5*q2*wz - 0.5*q3*wy
        f[:, 8] = 0.5*q0*wy - 0.5*q1*wz + 0.5*q3*wx
        f[:, 9] = 0.5*q0*wz + 0.5*q1*wy - 0.5*q2*wx

        f[:, 10] = 24.2857142857143*u1 - 24.2857142857143*u3 - 0.714285714285714*wy*wz
        f[:, 11] = 24.2857142857143*u2 - 24.2857142857143*u4 + 0.714285714285714*wx*wz
        f[:, 12] =  -4.16666666666667*u1 + 4.16666666666667*u2 - 4.16666666666667*u3 + 4.16666666666667*u4

        if self.quad_name=="iris":
            f[:, 10] = 6.32978769316642*u1 - 6.32978769316642*u2 - 5.75435244833311*u3 + 5.75435244833311*u4 - 1.49058156363019*wy*wz
            f[:, 11] = -2.83268217959641*u1 - 2.83268217959641*u2 + 2.83268217959641*u3 + 2.83268217959641*u4 + 1.37153459467586*wx*wz
            f[:, 12] =  -0.163766632548618*u1 + 0.163766632548618*u2 - 0.163766632548618*u3 + 0.163766632548618*u4 - 0.113987717502559*wx*wy
        if self.quad_name=="aims1":
            f[:, 10] = 41.9797424119299*u1 - 41.9797424119299*u2 - 41.9797424119299*u3 + 41.9797424119299*u4 - 0.77677511279981*wy*wz
            f[:, 11] = -34.5726150544936*u1 - 34.5726150544936*u2 + 34.5726150544936*u3 + 34.5726150544936*u4 + 0.816162090276148*wx*wz
            f[:, 12] =  3.81670284582906*u1 - 3.81670284582906*u2 + 3.81670284582906*u3 - 3.81670284582906*u4 - 0.107607165859593*wx*wy
        if self.quad_name=="aims3":
            f[:, 10] = 68.1444533634385*u1 - 68.1444533634385*u2 - 68.1444533634385*u3 + 68.1444533634385*u4 + 0.147949656809509*wy*wz
            f[:, 11] = -52.5839088946218*u1 - 52.5839088946218*u2 + 52.5839088946218*u3 + 52.5839088946218*u4 - 0.147949656809509*wx*wz
            f[:, 12] =  20.1516918604798*u1 - 20.1516918604798*u2 + 20.1516918604798*u3 - 20.1516918604798*u4


        return f

    def diff_numeric_central(self, x, u):
        # state & input size
        ix = self.n_x
        iu = self.n_u

        ndim = np.ndim(x)
        if ndim == 1:  # 1 step state & input
            N = 1
            x = np.expand_dims(x, axis=0)
            u = np.expand_dims(u, axis=0)
        else:
            N = np.size(x, axis=0)

        # numerical difference
        h = pow(2, -18)
        eps_x = np.identity(ix)
        eps_u = np.identity(iu)

        # expand to tensor
        x_mat = np.expand_dims(x, axis=2)
        u_mat = np.expand_dims(u, axis=2)

        # diag
        x_diag = np.tile(x_mat, (1, 1, ix))
        u_diag = np.tile(u_mat, (1, 1, iu))

        # augmented = [x_aug x], [u, u_aug]
        x_aug_m = x_diag - eps_x * h
        x_aug_m = np.dstack((x_aug_m, np.tile(x_mat, (1, 1, iu))))
        x_aug_m = np.reshape(np.transpose(x_aug_m, (0, 2, 1)), (N * (iu + ix), ix))

        u_aug_m = u_diag - eps_u * h
        u_aug_m = np.dstack((np.tile(u_mat, (1, 1, ix)), u_aug_m))
        u_aug_m = np.reshape(np.transpose(u_aug_m, (0, 2, 1)), (N * (iu + ix), iu))

        # augmented = [x_aug x], [u, u_aug]
        x_aug_p = x_diag + eps_x * h
        x_aug_p = np.dstack((x_aug_p, np.tile(x_mat, (1, 1, iu))))
        x_aug_p = np.reshape(np.transpose(x_aug_p, (0, 2, 1)), (N * (iu + ix), ix))

        u_aug_p = u_diag + eps_u * h
        u_aug_p = np.dstack((np.tile(u_mat, (1, 1, ix)), u_aug_p))
        u_aug_p = np.reshape(np.transpose(u_aug_p, (0, 2, 1)), (N * (iu + ix), iu))

        # numerical difference
        f_change_m = self.forward(x_aug_m, u_aug_m, 0)
        f_change_p = self.forward(x_aug_p, u_aug_p, 0)
        f_change_m = np.reshape(f_change_m, (N, ix + iu, ix))
        f_change_p = np.reshape(f_change_p, (N, ix + iu, ix))
        f_diff = (f_change_p - f_change_m) / (2 * h)
        f_diff = np.transpose(f_diff, [0, 2, 1])
        fx = f_diff[:, :, 0:ix]
        fu = f_diff[:, :, ix:ix + iu]

        # return np.squeeze(fx), np.squeeze(fu)
        return fx, fu

    def diff(self, x, u):
        # state & input size
        ix = self.n_x
        iu = self.n_u

        ndim = np.ndim(x)
        if ndim == 1:  # 1 step state & input
            N = 1
            x = np.expand_dims(x, axis=0)
            u = np.expand_dims(u, axis=0)
        else:
            N = np.size(x, axis=0)

        fx = np.zeros((N, ix, ix))
        fu = np.zeros((N, ix, iu))
        for i in range(N):
            fx[i, :, :] = self.A(x[i, :], u[i, :])
            fu[i, :, :] = self.B(x[i, :], u[i, :])

        # return np.squeeze(fx), np.squeeze(fu)
        return fx, fu

    def diff_discrete_foh_var_vectorized(self, x, u, T):
        # delT = self.delT
        ix = self.n_x
        iu = self.n_u

        ndim = np.ndim(x)
        if ndim == 1:  # 1 step state & input
            N = 1
            x = np.expand_dims(x, axis=0)
            u = np.expand_dims(u, axis=0)
        else:
            N = np.size(x, axis=0)

        def dvdt(t, V, um, up, length):
            assert len(um) == len(up)
            assert len(um) == length
            alpha = 1.0 - t
            beta = t
            u = alpha * um + beta * up
            # IPython.embed()
            V = V.reshape((length, ix + ix * ix + 2 * ix * iu + ix + ix)).transpose()
            x = V[:ix].transpose()
            Phi = V[ix:ix * ix + ix]
            Phi = Phi.transpose().reshape((length, ix, ix))
            Phi_inv = np.linalg.inv(Phi)
            f = self.forward(x, u)
            # if self.type_linearization == "numeric_central":
            #     A, B = self.diff_numeric_central(x, u)
            # elif self.type_linearization == "numeric_forward":
            #     A, B = self.diff_numeric(x, u)
            # elif self.type_linearization == "analytic":
            #     A, B = self.diff(x, u)
            A, B = self.diff_numeric_central(x, u)
            A, B = (A.T * T).T, (B.T * T).T
            dpdt = np.matmul(A, Phi).reshape((length, ix * ix)).transpose()
            dbmdt = np.matmul(Phi_inv, B).reshape((length, ix * iu)).transpose() * alpha
            dbpdt = np.matmul(Phi_inv, B).reshape((length, ix * iu)).transpose() * beta
            dsdt = np.squeeze(np.matmul(Phi_inv, np.expand_dims(f, 2))).transpose()
            dzdt = np.squeeze(np.matmul(Phi_inv, -np.matmul(A, np.expand_dims(x, 2)) - np.matmul(B, np.expand_dims(u,
                                                                                                                   2)))).transpose()
            dv = np.vstack((T * f.transpose(), dpdt, dbmdt, dbpdt, dsdt, dzdt))
            return dv.flatten(order='F')

        A0 = np.eye(ix).flatten()
        Bm0 = np.zeros((ix * iu))
        Bp0 = np.zeros((ix * iu))
        s0 = np.zeros(ix)
        z0 = np.zeros(ix)
        V0 = np.array([np.hstack((x[i], A0, Bm0, Bp0, s0, z0)) for i in range(N)]).transpose()
        V0_repeat = V0.flatten(order='F')

        # sol = solve_ivp(dvdt, (0, 1), V0_repeat, args=(u[0:N], u[1:], N), rtol=1e-6, atol=1e-10)
        sol = solve_ivp(dvdt,(0,self.dt),V0_repeat,args=(u[0:N],u[1:],N))
        # sol = RK4(dvdt, (0, self.dt), V0_repeat, args=(u[0:N], u[1:], N), N_RK=2)
        # IPython.embed()
        idx_state = slice(0, ix)
        idx_A = slice(ix, ix + ix * ix)
        idx_Bm = slice(ix + ix * ix, ix + ix * ix + ix * iu)
        idx_Bp = slice(ix + ix * ix + ix * iu, ix + ix * ix + 2 * ix * iu)
        idx_s = slice(ix + ix * ix + 2 * ix * iu, ix + ix * ix + 2 * ix * iu + ix)
        idx_z = slice(ix + ix * ix + 2 * ix * iu + ix, ix + ix * ix + 2 * ix * iu + ix + ix)
        sol = sol.y[:, -1].reshape((N, -1))
        # xnew = np.zeros((N+1,ix))
        # xnew[0] = x[0]
        # xnew[1:] = sol[:,:ix]
        x_prop = sol[:, idx_state].reshape((-1, ix))
        A = sol[:, idx_A].reshape((-1, ix, ix))
        Bm = np.matmul(A, sol[:, idx_Bm].reshape((-1, ix, iu)))
        Bp = np.matmul(A, sol[:, idx_Bp].reshape((-1, ix, iu)))
        s = np.matmul(A, sol[:, idx_s].reshape((-1, ix, 1))).squeeze()
        z = np.matmul(A, sol[:, idx_z].reshape((-1, ix, 1))).squeeze()

        return A, Bm, Bp, s, z, x_prop
    
    def trapezoidal_rule(self, x, u, sigma):
        ix = self.n_x
        iu = self.n_u
        k_num = x.shape[1]
        A_subs = np.zeros((ix*ix, k_num))
        B_subs = np.zeros((ix*iu, k_num))
        f_subs = np.zeros((ix, k_num))
        c_subs = np.zeros((ix, k_num))
        for i in range(k_num):
            A_subs[:, i] = (sigma * self.A(x[:,i], u[:,i])).reshape((ix*ix, ))
            B_subs[:, i] = (sigma * self.B(x[:,i], u[:,i])).reshape((ix*iu, ))
            f_subs[:, i] = (self.f(x[:,i], u[:,i])).reshape((ix, ))
            c_subs[:, i] = -sigma * np.dot(self.A(x[:,i], u[:,i]),x[:,i])-sigma * np.dot(self.B(x[:,i], u[:,i]),u[:,i])
        return A_subs, B_subs, f_subs, c_subs


