import numpy as np
import os
from scipy.integrate import solve_ivp


def plot_obstacle(ax, R1, H1, p1):

    # Make data
    # self.p1, self.R1 = self.obstacle_rendim(self.p1, self.R1)

    a, b, c = R1/H1[0, 0], R1/H1[1, 1], R1/H1[2, 2]
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = a * np.outer(np.cos(u), np.sin(v)) + p1[0]
    y = b * np.outer(np.sin(u), np.sin(v)) + p1[1]
    z = c * np.outer(np.ones(np.size(u)), np.cos(v)) + p1[2]

    # Plot the surface
    ax.plot_surface(x, y, z, color='tab:blue', alpha=0.5)



def obstacle_constraint_violation(r, H, rj):
    rj = np.expand_dims(rj, axis=-1)
    rj_1 = np.tile(rj, (1, r.shape[1]))
    f = np.linalg.norm(np.dot(H, r-rj_1), axis=0)

    return f


def euler_to_quat(a):
    a = np.deg2rad(a)

    cy = np.cos(a[1] * 0.5)
    sy = np.sin(a[1] * 0.5)
    cr = np.cos(a[0] * 0.5)
    sr = np.sin(a[0] * 0.5)
    cp = np.cos(a[2] * 0.5)
    sp = np.sin(a[2] * 0.5)

    q = np.zeros(4)

    q[0] = cy * cr * cp + sy * sr * sp
    q[1] = cy * sr * cp - sy * cr * sp
    q[3] = cy * cr * sp + sy * sr * cp
    q[2] = sy * cr * cp - cy * sr * sp
    # c1 = np.cos(a[0] * 0.5)
    # s1 = np.sin(a[0] * 0.5)
    # c2 = np.cos(a[1] * 0.5)
    # s2 = np.sin(a[1] * 0.5)
    # c3 = np.cos(a[2] * 0.5)
    # s3 = np.sin(a[2] * 0.5)

    # q = np.zeros(4)

    # q[0] = c1 * c2 * c3 - s1 * s2 * s3
    # q[1] = c1 * s2 * c3 - s1 * c2 * s3
    # q[3] = c1 * c2 * s3 + s1 * s2 * c3
    # q[2] = s1 * c2 * c3 + c1 * s2 * s3

    return q


def format_line(name, value, unit=''):
    """
    Formats a line e.g.
    {Name:}           {value}{unit}
    """
    name += ':'
    if isinstance(value, (float, np.ndarray)):
        value = f'{value:{0}.{4}}'

    return f'{name.ljust(40)}{value}{unit}'


def save_arrays(path, a_dict):
    """
    :param path: Output path
    :param a_dict: A dict containing the name of the array as key.
    """
    path = path.rstrip('/')

    if not os.path.isdir(path):
        os.mkdir(path)

    if len(os.listdir(path)) == 0:
        folder_number = '000'
    else:
        folder_number = str(int(max(os.listdir(path))) + 1).zfill(3)

    os.mkdir(f'{path}/{folder_number}')

    for key in a_dict:
        np.save(f'{path}/{folder_number}/{key}.npy', a_dict[key])


def obstacle_constraint_violation(r, H, rj):
    rj = np.expand_dims(rj, axis=-1)
    rj_1 = np.tile(rj, (1, r.shape[1]))
    f = np.linalg.norm(np.dot(H, r-rj_1), axis=0)

    return f


def hoop_constr(r, r_h, n_h, l_c, p_g, p_c):
    g1 = np.matmul(n_h.T, (r_h-r))-l_c
    g2 = np.matmul(n_h.T, (r-r_h))-l_c
    Nh = np.eye(3)-np.matmul(n_h, n_h.T)
    E = np.matmul(np.matmul(np.matmul((r-r_h).T, Nh.T), Nh), (r-r_h))
    g3 = E-p_g**2
    c1 = E-p_c**2
    sigma1 = -min(0, g1)
    sigma2 = -min(0, g2)
    sigma3 = -min(0, g3)
    f = sigma1*sigma2*sigma3*c1
    # f = sigma1 * sigma2 * c1

    return f, g1, g2, g3


def hoop_constr_diff(r, r_h, n_h, l_c, p_g, p_c):
    g1 = np.matmul(n_h.T, (r_h-r))-l_c
    g2 = np.matmul(n_h.T, (r-r_h))-l_c
    Nh = np.eye(3)-np.matmul(n_h, n_h.T)
    N_bar = np.matmul(Nh.T, Nh)
    same_diff = np.matmul(N_bar+N_bar.T, r-r_h)
    E = np.matmul(np.matmul(np.matmul((r-r_h).T, Nh.T), Nh), (r-r_h))
    g3 = E-p_g**2
    c1 = E-p_c**2
    sigma1 = -min(0, g1)
    sigma2 = -min(0, g2)
    sigma3 = -min(0, g3)
    f = sigma2*sigma3*c1*(-n_h)+sigma1*sigma3*c1*n_h+sigma1*sigma2*c1*same_diff+sigma1*sigma2*sigma3*same_diff
    # f = sigma2*c1*(-n_h)+sigma1*c1*n_h+sigma1*sigma2*c1*same_diff+sigma1*sigma2*same_diff


    return f


# e^At
def expm(a,t):
    a = t*a
    e = np.zeros(a.shape)
    f = np.eye(a.shape[1])
    k = 1
    while np.linalg.norm(e+f-e, 1) > 0:
        e = e+f
        f = np.matmul(a, f)/k
        k += 1
    return e


def lti_discrete(m, delta_t):
    f, A, B = m.get_equations()
    A = A(np.zeros((m.n_x, 1)), np.zeros((m.n_u, 1)))
    A_dim = A.shape[0]
    eAT = expm(A, delta_t)
    # eAT = np.zeros((6, 6))
    V0 = np.zeros((2*A_dim**2,))

    V = solve_ivp(ode_dVdt, [0, delta_t], V0,
                  method='RK45', atol=1e-8).y[:, -1]

    integral1 = V[0:A_dim**2].reshape(A_dim, A_dim)
    integral2 = V[A_dim**2:2*A_dim**2].reshape(A_dim, A_dim)

    return eAT, integral2, integral1-(integral2/delta_t), integral1


# def _ode_dVdt(self, V, t, u_t0, u_t1, sigma):
def ode_dVdt(t, V):
    """
    ODE-function to compute dVdt.

    :param V: Evaluation state V = [x, Phi_A, B_bar, C_bar, S_bar, z_bar]
    :param t: Evaluation time
    :param u_t0: Input at start of interval
    :param u_t1: Input at end of interval
    :param sigma: Total time
    :return: Derivative at current time and state dVdt
    """
    A = np.append(np.zeros((3, 3)), np.eye(3), axis=1)
    A = np.append(A, np.zeros((3, 6)), axis=0)
    A_dim = A.shape[0]
    dVdt = np.zeros_like(V)

    dVdt[0:A_dim**2] = expm(A, t).reshape(-1)

    dVdt[A_dim**2:2*A_dim**2] = (expm(A, t)*t).reshape(-1)

    return dVdt

