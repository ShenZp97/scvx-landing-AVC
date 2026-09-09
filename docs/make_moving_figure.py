"""Reproduce docs/example_trajectory_moving.png. Run from the repository root:

    python docs/make_moving_figure.py
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
from scripts.Configuration.parameters_6dof_landing_twophase import carrier_traj_file

# Fall back to a solver that ships with CVXPY so the figure can be reproduced
# without a MOSEK license.
srv.solver = cp.MOSEK if cp.MOSEK in cp.installed_solvers() else cp.CLARABEL

tf_guess = 3.8
goal = np.array([1.706, -0.708 + 0.026, 0.5 + 0.077 + 0.02])
goal_v = np.array([0.7102, 0.7084, 0.0])
init_p = np.array([-1.5, 1.5, 0.5])

X_traj, time_traj, U_traj = srv.scvx_landing_moving_exp(
    "aims3", goal, tf_guess, init_p, plot=False, goal_v=goal_v)

guess = np.array(pd.read_csv('scvx_traj1_3dof.csv', index_col=0).values)
X_lc = guess[0:6, :]

K = X_traj.shape[0]
K_phase = int(np.ceil(K / 2))
R1 = 1.0

# The same apex schedule the subproblem uses: phase-2 node times inside the
# manoeuvre, offset into the recorded log by start_time.
start_time = 3.025
time_points = np.array([2.66, 2.774, 2.888, 3.002, 3.116, 3.23,
                        3.344, 3.458, 3.572, 3.686, 3.8])
log = np.array(pd.read_csv(carrier_traj_file, header=None).values)[1:2623, 1:5]


def carrier_at(t):
    return np.array([np.interp(t, log[:, 0], log[:, i]) for i in (1, 2, 3)])


apex = np.array([carrier_at(t + start_time) for t in time_points])

fig = plt.figure(figsize=(17.5, 5.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.7, 1.0, 1.0])

# ---------------------------------------------------------------- 3-D view
ax = fig.add_subplot(gs[0, 0], projection='3d')

t_window = np.linspace(start_time, start_time + tf_guess, 200)
carrier_path = np.array([carrier_at(t) for t in t_window])
ax.plot(carrier_path[:, 0], carrier_path[:, 1], carrier_path[:, 2], '-',
        color='tab:red', linewidth=2.0, alpha=0.8)

# The phase-1 keep-out ball is not drawn: this approach never gets within R1 of
# the carrier, so the constraint is inactive throughout (see the static figure).

# The cone travels with the carrier: draw it at the start of phase 2 and again
# at touchdown.
h = np.linspace(0, 1.1, 20)
th = np.linspace(0, 2 * np.pi, 61)
H, TH = np.meshgrid(h, th)
ax.plot_wireframe(H * np.cos(TH) + apex[0][0], H * np.sin(TH) + apex[0][1], H + apex[0][2],
                  rstride=10, cstride=6, color='tab:green', alpha=0.35, linewidth=0.6)
ax.plot_surface(H * np.cos(TH) + apex[-1][0], H * np.sin(TH) + apex[-1][1], H + apex[-1][2],
                color='tab:green', alpha=0.16, linewidth=0)

ax.plot(X_lc[0], X_lc[1], X_lc[2], '--', color='0.4', linewidth=1.5)
ax.plot(X_traj[:, 0], X_traj[:, 1], X_traj[:, 2], '-o', color='tab:blue',
        markersize=3.2, linewidth=1.8)
ax.plot(apex[:, 0], apex[:, 1], apex[:, 2], '.', color='tab:red', markersize=7)
ax.plot([X_traj[K_phase - 1, 0]], [X_traj[K_phase - 1, 1]], [X_traj[K_phase - 1, 2]],
        's', color='tab:orange', markersize=8)
ax.plot([init_p[0]], [init_p[1]], [init_p[2]], '^', color='k', markersize=8)
ax.plot([goal[0]], [goal[1]], [goal[2]], '*', color='tab:red', markersize=15)

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_xlim(-2.0, 2.4)
ax.set_ylim(-1.4, 1.8)
ax.set_zlim(0, 2.6)
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_yticks([-1, 0, 1])
ax.set_zticks([0, 1, 2])
ax.set_box_aspect((1.15, 1.0, 0.95))
# Looking along the approach so the climb and the descent do not overlap.
ax.view_init(elev=22, azim=-122)
ax.set_title('Landing trajectory (aims3, moving carrier)', fontsize=11)

handles = [
    plt.Line2D([], [], color='0.4', linestyle='--', label='3-DoF guess (LC)'),
    plt.Line2D([], [], color='tab:blue', marker='o', markersize=4, label='6-DoF result (SCP)'),
    plt.Line2D([], [], color='tab:red', linewidth=2, label='recorded carrier path'),
    plt.Line2D([], [], color='tab:red', marker='.', linestyle='', markersize=9, label='cone apex per node'),
    plt.Line2D([], [], color='tab:orange', marker='s', linestyle='', label='phase switch'),
    plt.Line2D([], [], color='k', marker='^', linestyle='', label='start'),
    plt.Line2D([], [], color='tab:red', marker='*', markersize=10, linestyle='', label='touchdown'),
    plt.Line2D([], [], color='tab:green', alpha=0.5, linestyle='--', label='cone at phase-2 start'),
    plt.Line2D([], [], color='tab:green', alpha=0.5, linewidth=6, label='cone at touchdown'),
]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.04), ncol=3,
          fontsize=7.5, framealpha=0.85, borderpad=0.3, labelspacing=0.3,
          columnspacing=1.2, handletextpad=0.5)

# -------------------------------------------------- constraints in (d, dz)
ax2 = fig.add_subplot(gs[0, 1])

# Every phase-2 node is measured against the apex that belongs to it. The
# keep-out ball is inactive in this scenario (the approach never comes within
# R1 of the carrier), so only phase 2 is shown here.
d2 = np.linalg.norm(X_traj[K_phase - 1:, 0:2] - apex[:, 0:2], axis=1)
dz2 = X_traj[K_phase - 1:, 2] - apex[:, 2]

dz_line = np.linspace(0.0, 1.05 * dz2.max(), 50)
ax2.plot(dz_line, dz_line, '-', color='tab:green', linewidth=1.6, label='landing cone, 45$^\\circ$')
ax2.fill_between(dz_line, 0.0, dz_line, color='tab:green', alpha=0.10)
ax2.text(0.74 * dz2.max(), 0.46 * dz2.max(), 'inside\nthe cone', color='tab:green',
         fontsize=8.5, ha='center')

for k in range(len(d2)):
    ax2.plot([dz2[k], dz2[k]], [d2[k], dz2[k]], '-', color='0.75', linewidth=0.8)
ax2.plot(dz2, d2, 'o-', color='tab:blue', markersize=5, linewidth=1.0,
         label='phase 2 (landing)')
ax2.plot([dz2[-1]], [d2[-1]], '*', color='tab:red', markersize=15, label='touchdown')

ax2.set_xlabel('height above carrier $r_z - p_z(t)$ [m]')
ax2.set_ylabel('horizontal distance $\\|r_{xy} - p_{xy}(t)\\|$ [m]')
ax2.set_xlim(0, 1.05 * dz2.max())
ax2.set_ylim(0, 1.05 * dz2.max())
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8, loc='upper left')
ax2.set_title('Phase 2 stays inside the cone even though\nits apex moves with the carrier', fontsize=11)

# ------------------------------------------------- relative state vs. time
ax3 = fig.add_subplot(gs[0, 2])

carrier_now = np.array([carrier_at(t + start_time) for t in time_traj])
carrier_vel = np.array([(carrier_at(t + start_time + 0.05) - carrier_at(t + start_time - 0.05)) / 0.1
                        for t in time_traj])
# The body frame must settle a fixed offset above the deck rather than on it,
# which is the difference between `goal` and the carrier at touchdown.
offset = goal - carrier_at(start_time + tf_guess)
rel_p = np.linalg.norm(X_traj[:, 0:3] - carrier_now - offset, axis=1)
rel_v = np.linalg.norm(X_traj[:, 3:6] - carrier_vel, axis=1)

ax3.plot(time_traj, rel_p, '.-', color='tab:blue', linewidth=1.4,
         label='$\\|r - p(t) - \\Delta\\|$')
ax3.plot(time_traj, rel_v, '.-', color='tab:green', linewidth=1.4,
         label='$\\|v - \\dot{p}(t)\\|$')
ax3.axvline(time_traj[K_phase - 1], linestyle=':', color='tab:orange', linewidth=1.4)
ax3.text(time_traj[K_phase - 1] - 0.06, 0.5 * max(rel_p.max(), rel_v.max()),
         'phase switch', color='tab:orange', fontsize=8, rotation=90, va='center', ha='right')
ax3.axhline(0.0, color='0.6', linewidth=0.8)
ax3.set_ylim(bottom=-0.08)
ax3.set_xlabel('t [s]')
ax3.set_ylabel('relative position [m] / velocity [m/s]')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9, loc='upper right')
ax3.set_title('Rendezvous: relative position ($\\Delta$ = touchdown offset)\nand velocity both vanish', fontsize=11)

fig.tight_layout()

# 3-D axes reserve a lot of internal padding, so claim some of it back after
# tight_layout has settled the other two panels.
pos = ax.get_position()
ax.set_position([pos.x0 - 0.018, pos.y0 - 0.055, pos.width * 1.10, pos.height * 1.18])

fig.savefig('docs/example_trajectory_moving.png', dpi=150)
print('saved, tf =', time_traj[-1])
print('touchdown  =', np.round(X_traj[-1, 0:3], 3), np.round(X_traj[-1, 3:6], 3))
print('carrier    =', np.round(carrier_at(start_time + tf_guess), 3), 'offset =', np.round(offset, 3))
print('rel_p, rel_v at touchdown =', round(rel_p[-1], 4), round(rel_v[-1], 4))
print('worst cone slack in phase 2 =', round(float(np.min(dz2 - d2)), 4))
print('min dist to carrier in phase 1 =',
      round(float(np.min(np.linalg.norm(X_traj[:K_phase, 0:3] - goal, axis=1))), 4))
