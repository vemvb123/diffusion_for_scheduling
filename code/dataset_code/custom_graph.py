

import numpy as np
import matplotlib.pyplot as plt

# LB rows only
instances = ["mk01", "mk02", "mk03", "mk04", "mk05",
             "mk06", "mk07", "mk10"]

size = np.array([330, 348, 1200, 720, 424, 1500, 500, 3600])
avg_infeas = np.array([0.3, 0.5, 5.0, 3.8, 0.5, 4.4, 1.7, 6.8])




plt.figure(figsize=(10, 6))

# Scatter points
plt.scatter(size, avg_infeas, s=120, alpha=0.8)

# Custom label offsets to avoid overlap
offsets = {
    "mk01": (-25, -15),
    "mk02": (10, 10),
    "mk03": (10, 5),
    "mk04": (-30, 10),
    "mk05": (10, -15),
    "mk06": (10, -5),
    "mk07": (-25, 5),
    "mk10": (10, 5)
}

# Add labels
for x, y, label in zip(size, avg_infeas, instances):
    dx, dy = offsets[label]

    plt.annotate(
        label,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=10
    )

# Connect points in order of increasing size
order = np.argsort(size)
plt.plot(
    size[order],
    avg_infeas[order],
    alpha=0.5,
    linewidth=1.5,
    label="Instance progression"
)

# Linear trend line
z = np.polyfit(size, avg_infeas, 1)
p = np.poly1d(z)

x_line = np.linspace(size.min(), size.max(), 300)

plt.plot(
    x_line,
    p(x_line),
    "--",
    linewidth=2,
    label="Linear trend"
)

plt.xlabel("Size")
plt.ylabel("Average Infeasibility")
plt.title("Brandimarte Instances (LB): Size vs Avg Infeasibility")

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/code/dataset_code'
plot_name = "mk_size_vs_infeas.png"
full_path = f"{path}/{plot_name}"
plt.savefig(full_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved plot")





