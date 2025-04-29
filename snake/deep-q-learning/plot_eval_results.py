import pandas as pd
import matplotlib.pyplot as plt
import os

csv_path = "eval/eval_summary.csv"

if not os.path.exists(csv_path):
    print("❌ CSV file not found:", csv_path)
    exit()

# Load and process CSV
df = pd.read_csv(csv_path)
df = df[df["model"].str.contains("model")]
df["step"] = df["model"].str.extract(r"model_(\d+)").fillna(0).astype(int)
df.sort_values("step", inplace=True)

# Plot 1: Average Score over Steps
plt.figure()
plt.plot(df["step"], df["avg_score"], marker='o')
plt.xlabel("Training Step")
plt.ylabel("Average Score")
plt.title("Average Evaluation Score Over Training")
plt.grid(True)
plt.tight_layout()
plt.savefig("eval/model_score_over_time.png")
plt.close()

# Plot 2: Stacked Action Distribution
action_df = df[["step", "action_0", "action_1", "action_2"]].set_index("step")
action_df.plot(kind="bar", stacked=True)
plt.title("Action Usage Distribution per Model")
plt.xlabel("Training Step")
plt.ylabel("Action Count")
plt.tight_layout()
plt.grid(True)
plt.savefig("eval/action_distribution_by_model.png")
plt.close()

