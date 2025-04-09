import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ----------------------------
# Physical parameters
# ----------------------------
U = 0.13       # Free-stream flow velocity (m/s)
R = 0.025      # Cylinder radius (m)
rho = 1000     # Density (kg/m^3), e.g., water
p_ref = 0      # Reference pressure (p_inf)

# ----------------------------
# Definition of the computational domain
# ----------------------------
x_min, x_max = -0.2, 0.2
y_min, y_max = -0.2, 0.2
nx, ny = 1000, 1000  # High resolution for detailed results
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y)

# ----------------------------
# Conversion to polar coordinates (cylinder center at (0,0))
# ----------------------------
r = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)  # Standard angle measured from the x-axis

# ----------------------------
# Calculation of the polar velocity components for potential flow
# ----------------------------
# Classical formulas for potential flow around a cylinder:
#   v_r = U*cos(theta)*(1 - (R^2)/(r^2))
#   v_theta = -U*sin(theta)*(1 + (R^2)/(r^2))
v_r = U * np.cos(theta) * (1 - (R**2) / (r**2))
v_theta = -U * np.sin(theta) * (1 + (R**2) / (r**2))

# ----------------------------
# Conversion to Cartesian components
# ----------------------------
#   v_x = v_r*cos(theta) - v_theta*sin(theta)
#   v_y = v_r*sin(theta) + v_theta*cos(theta)
v_x = v_r * np.cos(theta) - v_theta * np.sin(theta)
v_y = v_r * np.sin(theta) + v_theta * np.cos(theta)

# ----------------------------
# Masking the interior of the cylinder (r < R)
# ----------------------------
mask = r < R
v_x[mask] = np.nan
v_y[mask] = np.nan

# ----------------------------
# Calculation of scalar fields
# ----------------------------
V = np.sqrt(v_x**2 + v_y**2)         # Magnitude of the velocity field
U_field = np.full_like(X, U)         # Uniform field (constant value U)
Delta = U**2 - V**2                  # Δ = U² - V²

# Pressure field via Bernoulli: p = p_ref + 0.5*rho*(U² - V²)
pressure = p_ref + 0.5 * rho * (U**2 - V**2)

# ----------------------------
# Utility function: add the cylinder to a given axis
# ----------------------------
def add_cylinder(ax):
    cyl = Circle((0, 0), R, edgecolor='k', facecolor='none', lw=2)
    ax.add_patch(cyl)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

# ----------------------------
# Function to plot a scalar field (contourf) with streamlines
# ----------------------------
def plot_field(ax, field, title, cmap, levels=20):
    # Filled contour plot of the scalar field
    cf = ax.contourf(X, Y, field, levels=levels, cmap=cmap)
    # Superimpose streamlines from (v_x, v_y)
    ax.streamplot(x, y, v_x, v_y, density=2, color='k', linewidth=1, arrowsize=1)
    ax.set_title(title)
    add_cylinder(ax)
    fig.colorbar(cf, ax=ax, orientation='vertical')

# ----------------------------
# Create the figure with a 2x3 grid
# ----------------------------
fig, axs = plt.subplots(2, 3, figsize=(14, 18), sharex=True, sharey=True)
plt.tight_layout(pad=3.0)

# Sub-figure 1: Field of v_x
plot_field(axs[0, 0], v_x, r'Field of $v_x$ (m/s)', 'bwr', levels=20)

# Sub-figure 2: Field of v_y
plot_field(axs[0, 1], v_y, r'Field of $v_y$ (m/s)', 'bwr', levels=20)

# Sub-figure 3: Magnitude V
plot_field(axs[0, 2], V, r'Field of $V=\sqrt{v_x^2+v_y^2}$ (m/s)', 'bwr', levels=20)

# Sub-figure 4: Uniform field U
plot_field(axs[1, 0], U_field, r'Uniform field $U$ (m/s)', 'viridis',
           levels=[U-0.001, U+0.001])
axs[1, 0].text(x_min+0.02, y_max-0.05, f"U = {U:.2f} m/s", color='k', fontsize=12)

# Sub-figure 5: Field Δ = U² - V²
plot_field(axs[1, 1], Delta, r'Field of $\Delta=U^2 - V^2$ (m$^2$/s$^2$)', 'bwr',
           levels=20)

# Sub-figure 6: Pressure field
plot_field(axs[1, 2], pressure, r'Pressure field $p$ (Pa)', 'bwr', levels=20)

# ----------------------------
# Function to save the entire figure
# ----------------------------
def save_figure(fig, filename, dpi=300):
    """
    Saves the entire figure (with all subplots) to a file.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    filename : str
        Name of the file to create.
    dpi : int, optional
        Resolution in dots per inch. Default is 300.
    """
    fig.savefig(filename, dpi=dpi)

# Show the figure on screen
plt.show()

# ----------------------------
# Save the figure to a file
# ----------------------------
save_figure(fig, "cylinder_flow.png")
