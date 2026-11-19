"""AE 352 quadrotor reference-frame and attitude illustrations.

Produces two vector PDF figures and PNG previews. Requires numpy/matplotlib.
Rotors match the team's report exactly:
1=+b1, 2=+b2, 3=-b1, 4=-b2; body b3 is positive thrust.
The second figure illustrates geometry, NOT numerically simulated motion.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})

INK = '#203145'
SILVER = '#94a3b8'
HUB = '#334155'
ROTOR = '#cbd5e1'
THRUST = '#db8b24'
GRAVITY = '#bc4141'
AXIS_COLORS = ['#2563eb', '#17916a', '#8549ba']
L = 1.0
ROTOR_RADIUS = 0.535   # rotor disk radius / arm length ~ 64.8mm / 120mm


def rotations(phi=0, theta=0, psi=0):
    """Body-to-inertial R = Rz(psi) Ry(theta) Rx(phi). Angles in degrees."""
    a, b, c = np.radians([phi, theta, psi])
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

ROTOR_B = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=float)


def arrow(ax, start, vector, color, width=2.3, arrow_ratio=.17, **kw):
    start = np.asarray(start, dtype=float)
    vector = np.asarray(vector, dtype=float)
    ax.quiver(*start, *vector, color=color, linewidth=width,
              arrow_length_ratio=arrow_ratio, normalize=False, **kw)


def decorate(ax, lim=1.9, zlow=-1.05, zhigh=1.95, elev=26, azim=-55):
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(zlow, zhigh))
    ax.set_box_aspect((2*lim, 2*lim, zhigh-zlow), zoom=1.24)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.dist = 9


def drone(ax, R=np.eye(3), rotor_numbers=True, show_thrust=False, show_gravity=False,
          body_axes=True, inertial_axes=False, label_axes=True, compact=False):
    center = np.array([0., 0., 0.])
    pts = (R @ ROTOR_B.T).T

    # Rotor arms: each arm terminates at the rotor axis, 120 mm from CoM.
    for i, rotor in enumerate(pts):
        xyz = np.vstack((center, rotor))
        ax.plot(*xyz.T, lw=8 if not compact else 5.8, c=SILVER, solid_capstyle='round', zorder=2)
        ax.plot(*xyz.T, lw=3 if not compact else 2.2, c=INK, solid_capstyle='round', zorder=3)

    # 3-bladed rotor outlines, lying in the body b1/b2 plane.
    t = np.linspace(0, 2*np.pi, 90)
    for i, rotor in enumerate(pts):
        offsets = np.column_stack([ROTOR_RADIUS*np.cos(t), ROTOR_RADIUS*np.sin(t), np.zeros_like(t)])
        circle = rotor + offsets @ R.T
        ax.plot(*circle.T, lw=1.25 if not compact else .8, c=ROTOR, alpha=.9)
        for k in range(3):
            ang = k * 2*np.pi/3 + np.pi/9
            blade_outer = rotor + R @ np.array([ROTOR_RADIUS*.78*np.cos(ang), ROTOR_RADIUS*.78*np.sin(ang), 0])
            ax.plot(*np.vstack([rotor, blade_outer]).T,
                    lw=4.4 if not compact else 2.9, c='#75899e', alpha=.82, solid_capstyle='round')
        ax.scatter(*rotor, s=47 if not compact else 24, c=HUB, depthshade=False)

    # Center of mass on the body plane.
    ax.scatter(*center, s=140 if not compact else 80, c=HUB, depthshade=False, zorder=5)
    if rotor_numbers:
        # Move text outward radially so it does not overlap rotors or thrust arrows.
        for i, rotor in enumerate(pts):
            position = rotor + .37 * rotor / np.linalg.norm(rotor) + R @ np.array([0., 0., -.22])
            ax.text(*position, rf'$\mathrm{{Rotor}}\ {i+1}$', color=INK,
                    fontsize=10, ha='center', va='center', weight='medium')

    if body_axes:
        for axis, (color, sym) in enumerate(zip(AXIS_COLORS, [r'$\mathbf{b}_1$', r'$\mathbf{b}_2$', r'$\mathbf{b}_3$'])):
            vector = R @ np.eye(3)[:, axis] * (1.65 if not compact else 1.5)
            arrow(ax, [0, 0, 0], vector, color, width=2.6 if not compact else 1.9)
            if label_axes:
                ax.text(*(vector*1.13), sym, color=color,
                        fontsize=13 if not compact else 10.5, ha='center', va='center')

    if inertial_axes:
        # Faint fixed inertial axes show which orientation is changing.
        for axis, sym in enumerate([r'$\mathbf{a}_1$', r'$\mathbf{a}_2$', r'$\mathbf{a}_3$']):
            v = np.eye(3)[axis]*1.8
            arrow(ax, [0, 0, 0], v, '#c3cbd4', width=1.1)
            if label_axes:
                ax.text(*(v*1.13), sym, color='#8a96a4', fontsize=9)

    if show_thrust:
        for i, rotor in enumerate(pts):
            base = rotor + R @ np.array([0,0,.06])
            v = R @ np.array([0,0,.9])
            arrow(ax, base, v, THRUST, width=2.7, arrow_ratio=.19)
            tip = base + v
            lab_off = R @ np.array([.12 if i % 2 else -.12, -.14, .06])
            ax.text(*(tip + lab_off), rf'$F_{i+1}$', color=THRUST, weight='bold', fontsize=13)

    if show_gravity:
        arrow(ax, [0., 0., 0.], [0.,0.,-.85], GRAVITY, width=3.0)
        ax.text(.18, .13, -.89, r'$mg$', color=GRAVITY, weight='bold', fontsize=13)


def save(fig, stem):
    pdf = OUT / f'{stem}.pdf'
    png = OUT / f'{stem}.png'
    fig.savefig(pdf, bbox_inches='tight', pad_inches=.12)
    fig.savefig(png, dpi=220, bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    print('saved', pdf, png)


def make_forces():
    fig = plt.figure(figsize=(7.4, 6.9), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    # Upper margin for document caption; figure as clean diagram.
    fig.subplots_adjust(left=.01, right=.99, bottom=.08, top=.97)
    drone(ax, show_thrust=True, show_gravity=True)
    decorate(ax, lim=1.72, zlow=-1.10, zhigh=1.55, elev=26, azim=-53)
    fig.text(.5, .035,
        r'$\mathbf{r}_1=+L\mathbf{b}_1\quad \mathbf{r}_2=+L\mathbf{b}_2\quad '
        r'\mathbf{r}_3=-L\mathbf{b}_1\quad \mathbf{r}_4=-L\mathbf{b}_2$',
        ha='center', va='center', fontsize=11, color=INK)
    fig.text(.5, .004,
        r'Orange: rotor thrust $(+\mathbf{b}_3)$   $\bullet$   Red: gravity $( -\mathbf{a}_3)$',
        ha='center', va='center', fontsize=9.5, color=INK)
    save(fig, 'AE352_quadrotor_forces')


def make_attitude():
    fig = plt.figure(figsize=(10.7, 7.9), facecolor='white')
    configurations = [
        ('(a) Level reference', (0,0,0)),
        (r'(b) Positive roll  $\phi=20^\circ$', (20,0,0)),
        (r'(c) Positive pitch  $\theta=20^\circ$', (0,20,0)),
        (r'(d) Positive yaw  $\psi=30^\circ$', (0,0,30)),
    ]
    for i, (heading, (phi, theta, psi)) in enumerate(configurations):
        ax = fig.add_subplot(2,2,i+1, projection='3d')
        drone(ax, R=rotations(phi,theta,psi), rotor_numbers=False,
              body_axes=True, inertial_axes=True, compact=True)
        decorate(ax, lim=1.8, zlow=-1.75, zhigh=1.8, elev=25, azim=-56)
        ax.set_title(heading, fontsize=12, color=INK, pad=8, weight='bold')
    fig.subplots_adjust(left=.01, right=.99, top=.95, bottom=.095, wspace=0, hspace=.16)
    fig.text(.5, .055,
             r'Body axes $\mathbf{b}_i$ (color) rotate with the drone; inertial axes $\mathbf{a}_i$ (gray) remain fixed.',
             ha='center', fontsize=10, color=INK)
    fig.text(.5, .024,
             r'Illustrative orientations only (not simulation results).  $R=R_z(\psi)R_y(\theta)R_x(\phi)$.',
             ha='center', fontsize=10, color=INK)
    save(fig, 'AE352_quadrotor_attitude')

if __name__ == '__main__':
    make_forces()
    make_attitude()
