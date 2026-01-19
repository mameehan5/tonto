import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

logdir = "./logs/"

# Load Monitor logs
# monitor_files = list(logdir.rglob("monitor.csv"))
monitor = pd.read_csv(logdir + 'monitor.csv', skiprows=1, on_bad_lines='skip', engine="python")
# df = pd.concat([pd.read_csv(f, skiprows=1) for f in monitor_files])
# print(monitor.items())

# monitor = pd.read_csv("logs/monitor.csv", skiprows=1)
monitor.columns = ["r","l","t"]
print(monitor["r"])

plt.figure()
plt.plot(monitor["r"])
# plt.plot(monitor["r"].rolling(window=50, min_periods=1).mean())
plt.xlabel("Episodes")
plt.ylabel("Mean Episode Reward")
plt.title("Training Reward")
plt.savefig(logdir + 'mean_episode_reward.png', dpi=300)

plt.figure()
plt.plot(monitor["l"])
plt.xlabel("Episodes")
plt.ylabel("Episode Length")
plt.title("Episode Length")
plt.savefig(logdir + 'mean_episode_length.png', dpi=300)

logdir = Path("logs/progress.csv")  # path to your SB3 CSV logger
output_dir = Path("plots")          # where to save figures
rolling_window = 50                 # smoothing window

os.makedirs(output_dir, exist_ok=True)

# --- LOAD CSV ---
df = pd.read_csv(logdir, engine="python", on_bad_lines='skip')

# --- CLEAN NUMERIC COLUMNS ---
numeric_cols = [
    "train/ep_rew_mean",
    "train/ep_len_mean",
    "train/entropy_loss",
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/explained_variance"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        print(f"Warning: column {col} not found in CSV")

# --- COMPUTE ROLLING MEANS ---
for col in numeric_cols:
    if col in df.columns:
        df[col + "_smooth"] = df[col].rolling(window=rolling_window, min_periods=1).mean()

# --- PLOT FUNCTION ---
def save_plot(y, ylabel, title, filename, color='blue'):
    plt.figure(figsize=(10, 6))
    plt.plot(y, color=color)
    plt.xlabel("Training Updates")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / filename)
    plt.close()

# --- SAVE FIGURES ---
if "train/ep_rew_mean_smooth" in df.columns:
    save_plot(df["train/ep_rew_mean_smooth"], "Reward", "Smoothed Episode Reward", "reward.png")

if "train/ep_len_mean_smooth" in df.columns:
    save_plot(df["train/ep_len_mean_smooth"], "Episode Length", "Episode Length Over Training", "ep_length.png", color='orange')

if "train/entropy_loss_smooth" in df.columns:
    save_plot(-df["train/entropy_loss_smooth"], "Entropy", "Policy Entropy Over Training", "entropy.png", color='green')

if "train/value_loss_smooth" in df.columns:
    save_plot(df["train/value_loss_smooth"], "Value Loss", "Value Loss Over Training", "value_loss.png", color='red')

if "train/explained_variance_smooth" in df.columns:
    save_plot(df["train/explained_variance_smooth"], "Explained Variance", "Value Function Explained Variance", "explained_variance.png", color='purple')

print(f"All plots saved to {output_dir.resolve()}")


