import pandas as pd
import matplotlib.pyplot as plt
import os

games = ["snake", "pacman"]

for game in games:
    csv_path = f"eval/{game}_eval_summary.csv"

    if not os.path.exists(csv_path):
        print("❌ CSV file not found:", csv_path)
        exit()

    # Load and process CSV
    df = pd.read_csv(csv_path)

    # Filter rows with "model" in the model column
    df = df[df["model"].astype(str).str.contains("model", na=False)]

    # Extract training step from model name
    df["step"] = df["model"].astype(str).str.extract(r"model_(\d+)")[0].fillna(0).astype(int)
    df.sort_values("step", inplace=True)
    print(df)

    # Plot 1: Average Score over Steps
    if not df.empty and "avg_score" in df.columns:
        plt.figure()
        plt.plot(df["step"], df["avg_score"], marker='o')
        plt.xlabel("Training Step")
        plt.ylabel("Average Score")
        plt.title("Average Evaluation Score Over Training")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"eval/{game}_model_score_over_time.png")
        plt.close()
    else:
        print("⚠️ Skipping score plot: 'avg_score' column missing or no data.")

    # Plot 2: Stacked Action Distribution
    action_cols = [col for col in df.columns if col.startswith("avg_action_")]
    if action_cols:
        action_df = df[["step"] + action_cols].set_index("step")
        if not action_df.empty:
            action_df.plot(kind="bar", stacked=True)
            plt.title("Action Usage Distribution per Model")
            plt.xlabel("Training Step")
            plt.ylabel("Action Count")
            plt.tight_layout()
            plt.grid(True)
            plt.savefig(f"eval/{game}_action_distribution_by_model.png")
            plt.close()
        else:
            print("⚠️ Skipping action plot: No data in action_df.")
    else:
        print("⚠️ Skipping action plot: No avg_action_* columns found.")
