"""
Geometric representation of the 3D ("cigar") trap and its domain.
Writes figures/geometry_beam.png. No physics is computed
"""
import numpy as np

from common import (w0, depth_kT, L, rayleigh_range, setup_figures, save_fig,
                    C_BEAM, MUTED, SECONDARY, SEQ)

plt = setup_figures()

Lz = 1.5 * L
zR = rayleigh_range()

n_theta = 96
theta = np.linspace(0.0, 2.0 * np.pi, n_theta)


def surface_of_revolution(z, radius):
    """Tensor a profile radius(z) about the z-axis into (X, Y, Z) surface arrays."""
    Z, TH = np.meshgrid(z, theta)
    R, _ = np.meshgrid(radius, theta)
    return R * np.cos(TH), R * np.sin(TH), Z


def trap_equipotential(level):
    """Surface U_beam = -level*kB T: radius rho(z) and its axial extent (see docstring)."""
    z_max = zR * np.sqrt(depth_kT / level - 1.0)
    z = np.linspace(-z_max, z_max, 160)
    s = 1.0 / (1.0 + (z / zR)**2)
    rho2 = (w0**2 / (2.0 * s)) * np.log(depth_kT * s / level)
    return surface_of_revolution(z, np.sqrt(np.clip(rho2, 0.0, None)))


fig = plt.figure(figsize=(6.4, 6.6))
ax = fig.add_subplot(111, projection="3d")
um = 1e6   # metres -> microns for display

# Domain: the cylinder rho < L, |z| < Lz 
zc = np.array([-Lz, Lz])
Xc, Yc, Zc = surface_of_revolution(zc, np.array([L, L]))
ax.plot_surface(Xc * um, Yc * um, Zc * um, color=MUTED, alpha=0.06,
                linewidth=0, shade=False)
for zcap in (-Lz, Lz):                       # rim circles top and bottom
    ax.plot(L * np.cos(theta) * um, L * np.sin(theta) * um,
            zcap * um, color=MUTED, lw=1.0)
# a few verticals to read the cylinder as a solid
for th in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
    ax.plot([L * np.cos(th) * um] * 2, [L * np.sin(th) * um] * 2,
            [-Lz * um, Lz * um], color=MUTED, lw=0.6, alpha=0.5)

# Beam: the 1/e^2 envelope w(z), a faint hyperboloid 
zb = np.linspace(-Lz, Lz, 60)
Xb, Yb, Zb = surface_of_revolution(zb, w0 * np.sqrt(1.0 + (zb / zR)**2))
ax.plot_wireframe(Xb * um, Yb * um, Zb * um, rstride=8, cstride=6,
                  color=C_BEAM, lw=0.6, alpha=0.7)

# Trap region: nested equipotential cigars 
for level, shade, alpha in ((1.0, SEQ[3], 0.32), (4.0, SEQ[6], 0.7)):
    Xt, Yt, Zt = trap_equipotential(level)
    ax.plot_surface(Xt * um, Yt * um, Zt * um, color=shade, alpha=alpha,
                    linewidth=0, antialiased=True, shade=True)

# Beam axis 
ax.plot([0, 0], [0, 0], [-Lz * um, Lz * um], color=SECONDARY, lw=1.0, ls="--")

# Legend proxies (plot_surface/wireframe carry no label)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
handles = [
    Patch(facecolor=MUTED, alpha=0.25, label="domain cylinder"),
    Line2D([0], [0], color=C_BEAM, lw=1.6, label="beam envelope"),
    Patch(facecolor=SEQ[3], alpha=0.65, label="trap region"),
    Patch(facecolor=SEQ[6], alpha=0.85, label="inner shell"),
]
# Plain-word labels (the potential levels are given in the report caption) with a light
# backing, so the legend stays clear of the wireframe behind it.
ax.legend(handles=handles, loc="upper left", fontsize=9, bbox_to_anchor=(0.0, 0.99),
          frameon=True, facecolor="white", framealpha=0.85, edgecolor="none")

ax.set_box_aspect((L, L, Lz))          # true proportions
ax.set_xlim(-L * um, L * um)
ax.set_ylim(-L * um, L * um)
ax.set_zlim(-Lz * um, Lz * um)
ax.set_xlabel(r"$x$ ($\mu$m)", labelpad=2)
ax.set_ylabel(r"$y$ ($\mu$m)", labelpad=2)
ax.set_zlabel(r"$z$ ($\mu$m, beam axis)", labelpad=2)
ax.view_init(elev=16, azim=-58)
ax.grid(False)

save_fig(fig, "geometry_beam.png")
print(f"  cigar (U=-kBT): radius(0) = {w0*np.sqrt(np.log(depth_kT)/2)*1e9:.0f} nm, "
      f"half-length = {zR*np.sqrt(depth_kT-1)*1e9:.0f} nm")
print(f"  domain: L = {L*1e9:.0f} nm, Lz = {Lz*1e9:.0f} nm")
