import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import os

# --- Plotting Style Configuration ---
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
})

# Define color scheme
color_classical = "#4A90E2"  # Blue for Classical Theory
color_phonder = "#75A140"    # Green for ours
color_error = "#D96161"      # Red for error

# --- Load Data ---
current_dir = os.path.dirname(os.path.abspath(__file__))
input_name = 'sweep_reflectance_ideal.csv'
file_path = os.path.join(current_dir, input_name)

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: '{file_path}' not found.")
    print("Please run 'validate_sphere_model.py' first to generate the data.")
    exit()

# --- Create Figure ---
fig = plt.figure(figsize=(11, 10))
fig.set_constrained_layout(True)
gs = fig.add_gridspec(2, 1, height_ratios=[3, 0.6], hspace=0.04)

ax_main = fig.add_subplot(gs[0, 0])
ax_error = fig.add_subplot(gs[1, 0], sharex=ax_main) # Share x-axis with main plot

# --- Main Plot: Classical vs. Ours ---
# Plot Classical Theory results (solid line, filled markers)
ax_main.plot(df['wall_reflectance'], df['theory_flux'], '-', color=color_classical,
             marker='s', markersize=5, label='Classical Theory')

# Plot Ours results (solid line, filled markers, staggered)
# Note: For this simplified plot, distinguishing by marker and stagger for 'ours' is suitable.
ax_main.plot(df['wall_reflectance'], df['simulation_flux_mean'], '-', color=color_phonder,
             marker='o', markersize=5, markevery=2, label='Simulation (Ours)')

# Main plot labels and title
ax_main.set_ylabel('Detector Flux (Normalized)')
ax_main.set_title('Single Sphere Validation: Wall Illumination')
ax_main.grid(True, alpha=0.3)
ax_main.legend()
plt.setp(ax_main.get_xticklabels(), visible=False) # Hide x-tick labels on the main plot

# --- Error Plot ---
# Calculate relative error: (Phonder - Classical) / Classical * 100%
# Handle potential division by zero if classical_flux can be 0.
relative_error = np.divide(
    (df['simulation_flux_mean'] - df['theory_flux']),
    df['theory_flux'],
    out=np.zeros_like(df['theory_flux']),
    where=df['theory_flux'] != 0
) * 100

ax_error.plot(df['wall_reflectance'], relative_error, '-', color=color_error, marker='o', markersize=4)
ax_error.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

# Error plot labels
ax_error.set_xlabel('Wall Reflectance (R_w)')
ax_error.set_ylabel('Relative\nError (%)')

# Set y-axis limits for error plot to focus on small errors
max_err = np.max(np.abs(relative_error))
if not np.isnan(max_err) and max_err > 0:
    ax_error.set_ylim(-max_err * 1.2, max_err * 1.2)
else:
    ax_error.set_ylim(-1, 1) # Default small range if error is tiny or zero
ax_error.set_ylim(-20, 20)

ax_error.grid(True, alpha=0.3)


# --- Final Save ---
output_filename_png = f'{input_name.replace(".csv", "")}.png'
output_path_png = os.path.join(current_dir, output_filename_png)

fig.savefig(output_path_png, dpi=300, bbox_inches='tight')

print(f"Plot saved as '{output_filename_png}'")

# To display the plot, uncomment the following line:
# plt.show()