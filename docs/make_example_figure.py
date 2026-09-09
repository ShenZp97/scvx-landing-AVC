"""Reproduce docs/example_trajectory.png. Run from the repository root:

    python docs/make_example_figure.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

import scripts.twophase_landing_6dof_server as srv

# Fall back to a solver that ships with CVXPY so the figure can be reproduced
# without a MOSEK license.
srv.solver = cp.MOSEK if cp.MOSEK in cp.installed_solvers() else cp.CLARABEL

goal = np.array([0.0, 0.05, 1.677])
init_p = np.array([-0.5, 0.0, 0.3])

X_traj, time_traj, U_traj = srv.scvx_landing("aims3", goal, 3.0, init_p, plot=False)

guess = np.array(pd.read_csv('scvx_traj1_3dof.csv', index_col=0).values)
X_lc = guess[0:6, :]

K = X_traj.shape[0]
K_phase = int(np.ceil(K / 2))
R1 = 1.0

fig = plt.figure(figsize=(16.5, 5.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0])

# ---------------------------------------------------------------- 3-D view
ax = fig.add_subplot(gs[0, 0], projection='3d')

u = np.linspace(0, 2 * np.pi, 41)
v = np.linspace(0, np.pi, 21)
ax.plot_wireframe(R1 * np.outer(np.cos(u), np.sin(v)) + goal[0],
                  R1 * np.outer(np.sin(u), np.sin(v)) + goal[1],
                  R1 * np.outer(np.ones_like(u), np.cos(v)) + goal[2],
                  rstride=5, cstride=5, color='tab:red', alpha=0.22, linewidth=0.6)

h = np.linspace(0, 1.1, 20)
th = np.linspace(0, 2 * np.pi, 61)
H, TH = np.meshgrid(h, th)
ax.plot_surface(H * np.cos(TH) + goal[0], H * np.sin(TH) + goal[1], H + goal[2],
                color='tab:green', alpha=0.16, linewidth=0)

ax.plot(X_lc[0], X_lc[1], X_lc[2], '--', color='0.4', linewidth=1.5)
ax.plot(X_traj[:, 0], X_traj[:, 1], X_traj[:, 2], '-o', color='tab:blue',
        markersize=3.2, linewidth=1.8)
ax.plot([X_traj[K_phase - 1, 0]], [X_traj[K_phase - 1, 1]], [X_traj[K_phase - 1, 2]],
        's', color='tab:orange', markersize=8)
ax.plot([init_p[0]], [init_p[1]], [init_p[2]], '^', color='k', markersize=8)
ax.plot([goal[0]], [goal[1]], [goal[2]], '*', color='tab:red', markersize=15)

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_zlim(0, 2.8)
ax.set_box_aspect((1, 1, 1.1))
ax.view_init(elev=16, azim=-64)
ax.set_title('Landing trajectory (aims3, static carrier)', fontsize=11)

handles = [
    plt.Line2D([], [], color='0.4', linestyle='--', label='3-DoF guess (LC)'),
    plt.Line2D([], [], color='tab:blue', marker='o', markersize=4, label='6-DoF result (SCP)'),
    plt.Line2D([], [], color='tab:orange', marker='s', linestyle='', label='phase switch'),
    plt.Line2D([], [], color='k', marker='^', linestyle='', label='start'),
    plt.Line2D([], [], color='tab:red', marker='*', markersize=10, linestyle='', label='carrier'),
    plt.Line2D([], [], color='tab:red', alpha=0.5, label='keep-out ball'),
    plt.Line2D([], [], color='tab:green', alpha=0.5, label='landing cone'),
]
ax.legend(handles=handles, loc='upper left', fontsize=7.5, framealpha=0.85,
          borderpad=0.3, labelspacing=0.3)

# -------------------------------------------------- constraints in (d, dz)
ax2 = fig.add_subplot(gs[0, 1])

d = np.linalg.norm(X_traj[:, 0:2] - goal[0:2], axis=1)
dz = X_traj[:, 2] - goal[2]

dz_line = np.linspace(0.0, 2.0, 50)
ax2.plot(dz_line, dz_line, '-', color='tab:green', linewidth=1.6, label='landing cone, 45$^\\circ$')
ax2.fill_between(dz_line, 0.0, dz_line, color='tab:green', alpha=0.10)
ax2.text(1.45, 0.35, 'inside\nthe cone', color='tab:green', fontsize=8.5, ha='center')

phi = np.linspace(-np.pi / 2, np.pi / 2, 100)
ax2.plot(R1 * np.sin(phi), R1 * np.cos(phi), '-', color='tab:red', linewidth=1.6,
         label='keep-out ball, $R_1$ = 1 m')

ax2.plot(dz[:K_phase], d[:K_phase], 'o', color='tab:purple', markersize=5,
         label='phase 1 (approach)')
ax2.plot(dz[K_phase - 1:], d[K_phase - 1:], 'o', color='tab:blue', markersize=5,
         label='phase 2 (landing)')

ax2.set_xlabel('height above carrier $r_z - p_z$ [m]')
ax2.set_ylabel('horizontal distance $\\|r_{xy} - p_{xy}\\|$ [m]')
ax2.set_xlim(-1.6, 1.9)
ax2.set_ylim(0.0, 1.6)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8, loc='upper left')
ax2.set_title('Phase 1 stays outside the ball,\nphase 2 stays inside the cone', fontsize=11)

# ------------------------------------------------------------- rotor thrust
ax3 = fig.add_subplot(gs[0, 2])
for i in range(4):
    ax3.plot(time_traj, U_traj[:, i], '.-', linewidth=1.2, label=f'rotor {i + 1}')
ax3.axhline(25.0 / 4, linestyle='--', color='tab:red', linewidth=1)
ax3.axhline(3.0 / 4, linestyle='--', color='tab:red', linewidth=1)
ax3.text(0.05, 6.32, '$T_{max}/4$', color='tab:red', fontsize=8)
ax3.text(0.05, 0.85, '$T_{min}/4$', color='tab:red', fontsize=8)
ax3.axvline(time_traj[K_phase - 1], linestyle=':', color='tab:orange', linewidth=1.4)
ax3.text(time_traj[K_phase - 1] + 0.04, 5.4, 'phase switch', color='tab:orange', fontsize=8)
ax3.set_xlabel('t [s]')
ax3.set_ylabel('rotor thrust [N]')
ax3.set_ylim(0, 7)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=8, ncol=2, loc='upper right')
ax3.set_title('Individual rotor thrusts', fontsize=11)

fig.tight_layout()
fig.savefig('docs/example_trajectory.png', dpi=150)
print('saved, tf =', time_traj[-1], 'min dist to carrier in phase 1 =',
      np.min(np.linalg.norm(X_traj[:K_phase, 0:3] - goal, axis=1)))
