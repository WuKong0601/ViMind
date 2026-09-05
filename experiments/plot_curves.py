import json
import os
import matplotlib.pyplot as plt

def generate_paper_plots():
    metrics_path = os.path.join("experiments", "vimind_1.0", "metrics_summary.json")
    if not os.path.exists(metrics_path):
        print("Metrics file not found.")
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps_data = data["dpo"]["steps_data"]
    steps = [x["step"] for x in steps_data]
    losses = [x["loss"] for x in steps_data]
    accuracies = [x["accuracy"] for x in steps_data]
    margins = [x["margin"] for x in steps_data]

    # Set publication style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)

    # 1. DPO Loss Curve
    axes[0].plot(steps, losses, color="#d9534f", lw=2.2, marker="o", markersize=4, label="DPO Loss")
    axes[0].set_title("ViMind 1.0 - DPO Loss Convergence", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Training Steps", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].set_ylim(0.0, 0.6)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # 2. Reward Accuracy Curve
    axes[1].plot(steps, accuracies, color="#0275d8", lw=2.2, marker="s", markersize=4, label="Reward Accuracy (%)")
    axes[1].axhline(y=50.0, color="gray", linestyle=":", label="Random Guess (50%)")
    axes[1].set_title("ViMind 1.0 - Pairwise Preference Accuracy", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Training Steps", fontsize=10)
    axes[1].set_ylabel("Accuracy (%)", fontsize=10)
    axes[1].set_ylim(40.0, 95.0)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    # 3. Reward Margin Curve
    axes[2].plot(steps, margins, color="#5cb85c", lw=2.2, marker="^", markersize=4, label="Reward Margin (Chosen - Rejected)")
    axes[2].axhline(y=0.0, color="gray", linestyle=":", label="Zero Margin")
    axes[2].set_title("ViMind 1.0 - Implicit Reward Margin", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Training Steps", fontsize=10)
    axes[2].set_ylabel(r"$\Delta r = r(y_w) - r(y_l)$", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    out_img = os.path.join("experiments", "vimind_1.0", "dpo_training_curves.png")
    plt.savefig(out_img, bbox_inches="tight")
    print(f"Publication-grade training curves saved to: {out_img}")

if __name__ == "__main__":
    generate_paper_plots()
