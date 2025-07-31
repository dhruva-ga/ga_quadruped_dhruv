import pandas as pd
import matplotlib.pyplot as plt

# 1) Load your combined log (time, <joint>_force, <joint>_cmd, <joint>_act, …)
df = pd.read_csv("logged_data.csv")
time = df["time"].astype(float)

# 2) Define legs and joint categories
legs        = ["rr", "rl", "fr", "fl"]
joint_types = ["hip", "thigh", "calf"]

# 3) For each joint type, create one window with 4 subplots
for jt in joint_types:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    axes = axes.flatten()
    for ax, leg in zip(axes, legs):
        # column names in the combined logger:
        force_col = f"{jt}_{leg}_force"
        cmd_col   = f"{jt}_{leg}_cmd"
        act_col   = f"{jt}_{leg}_act"

        # plot torque
        # ax.plot(time, df[force_col], label="torque")
        # plot commanded vs actual angle
        ax.plot(time, df[cmd_col],   label="cmd angle", linestyle="--")
        ax.plot(time, df[act_col],   label="act angle", linestyle=":")

        ax.set_title(f"{jt.capitalize()} {leg.upper()}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

    fig.suptitle(f"{jt.capitalize()} Joint: Torque & Angles")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 4) Show all three windows
plt.show()
