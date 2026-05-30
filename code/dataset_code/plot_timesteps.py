import numpy as np
import matplotlib.pyplot as plt

# Data from table
# mk01
# steps = np.array([50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
# total_infeas_ops = np.array([33, 15, 14, 13, 19, 17, 24, 25, 23, 31, 37])

# mk10
steps = np.array([50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
total_infeas_ops = np.array([636, 385, 218, 165, 165, 165, 188, 187, 232, 245, 307])


benchmark = "mk10"

plt.figure(figsize=(10, 10))

# Line with markers
plt.plot(
    steps,
    total_infeas_ops,
    marker='o',
    linewidth=2,
    markersize=7,
    label="Total Infeasible Operations"
)

# Annotate points
for x, y in zip(steps, total_infeas_ops):
    plt.annotate(
        str(y),
        (x, y),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )

plt.xlabel("Timesteps")
plt.ylabel("Total Infeasible Operations")
plt.title(f"{benchmark} Total Infeasible Operations over Timesteps")

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save figure
path = "/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/code/dataset_code"
plot_name = f"{benchmark}_infeas_timesteps.png"
plt.savefig(
    f"{path}/{plot_name}",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

