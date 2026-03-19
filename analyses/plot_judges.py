import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


CSV_PATH = "data/llama_vs_mixtral_summary.csv"
OUT_PATH = "plots/llama_vs_mixtral_diff_colored.png"


def main():
    df = pd.read_csv(CSV_PATH)

    margin_col = "mean_margin_llama_minus_mixtral"
    if margin_col not in df.columns:
        raise ValueError(f"Missing required column: {margin_col}")

    metrics = df["metric"]
    diffs = df[margin_col]

    colors = ["#1f77b4" if d > 0 else "#ff7f0e" for d in diffs]

    plt.figure(figsize=(10, 6))
    plt.bar(metrics, diffs, color=colors)
    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Mean difference (LLaMA − Mixtral)")
    plt.title("Which judge scores higher by metric")

    legend_elements = [
        Patch(facecolor="#1f77b4", label="LLaMA scores higher"),
        Patch(facecolor="#ff7f0e", label="Mixtral scores higher"),
    ]
    plt.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plot -> {OUT_PATH}")


if __name__ == "__main__":
    main()