import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# ---- Plot Configuration ----
# Figure sizes for different plot types (width, height in inches)
FIGURE_SIZES = {
    "heatmap": (12, 8),
    "barplot": (12, 6),
    "scatter": (10, 6),
    "lineplot": (12, 6),
}

# Font sizes
FONT_SIZES = {
    "title": 16,
    "axes_labels": 14,
    "tick_labels": 12,
    "legend": 12,
    "annotations": 10,
}

# Apply global matplotlib settings
plt.rcParams.update(
    {
        "font.size": FONT_SIZES["tick_labels"],
        "axes.titlesize": FONT_SIZES["title"],
        "axes.labelsize": FONT_SIZES["axes_labels"],
        "xtick.labelsize": FONT_SIZES["tick_labels"],
        "ytick.labelsize": FONT_SIZES["tick_labels"],
        "legend.fontsize": FONT_SIZES["legend"],
        "figure.titlesize": FONT_SIZES["title"],
    }
)

# Create output directories if they don't exist
output_dir = "analysis_results"
plots_dir = os.path.join(output_dir, "plots")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Load the data
input_file = "diffusion_analysis_results_20250423_191103/diffusion_mask_analysis_detailed_20250423_191103.csv"
df = pd.read_csv(input_file)

# Add computed columns
df["SuccessRate"] = df["CorrectSRVCount"] / df["Samples"]
df["UniquenessRate"] = df["UniqueGenerated"] / df["Samples"]
df["ErrorRate"] = df["ConversionErrors"] / df["Samples"]

# Extract base geometry type
df["GeometryBase"] = df["Geometry"].apply(lambda x: x.split("_")[0])

# Check if it's a baseline unconstrained geometry
df["IsBaseline"] = df["Geometry"] == "Baseline_Unconstrained"

# Create a filtered dataframe without baseline
df_no_baseline = df[~df["IsBaseline"]]

# ---- Tables ----
# Summary by qubit count
summary_by_qubits = (
    df.groupby("Qubits")
    .agg(
        {
            "Samples": "sum",
            "UniqueGenerated": "sum",
            "CorrectSRVCount": "sum",
            "SuccessRate": "mean",
            "UniquenessRate": "mean",
            "MaskEffectiveGates": "mean",
            "ConversionErrors": "sum",
        }
    )
    .reset_index()
)

# Summary by geometry
summary_by_geometry = (
    df.groupby(["GeometryBase", "Qubits"])
    .agg(
        {
            "Samples": "sum",
            "UniqueGenerated": "sum",
            "CorrectSRVCount": "sum",
            "SuccessRate": "mean",
            "UniquenessRate": "mean",
            "MaskEffectiveGates": "mean",
            "ConversionErrors": "sum",
        }
    )
    .reset_index()
)

# Summary by SRV complexity - using filtered data without baseline
summary_by_srv = (
    df_no_baseline.groupby(["Qubits", "SRV_Complexity"])
    .agg(
        {
            "Samples": "sum",
            "UniqueGenerated": "sum",
            "CorrectSRVCount": "sum",
            "SuccessRate": "mean",
            "UniquenessRate": "mean",
            "MaskEffectiveGates": "mean",
            "ConversionErrors": "sum",
        }
    )
    .reset_index()
)

# Performance Analysis by Connectivity Size
df["ConnectivitySize"] = df["Connectivity"].apply(
    lambda x: len(eval(x)) if x != "[]" else 0
)
connectivity_analysis = (
    df.groupby("ConnectivitySize")
    .agg(
        {
            "SuccessRate": ["mean", "std"],
            "UniquenessRate": ["mean", "std"],
            "Samples": "sum",
            "CorrectSRVCount": "sum",
        }
    )
    .reset_index()
)
connectivity_analysis.columns = [
    "_".join(col).strip() for col in connectivity_analysis.columns.values
]

# Save summary tables to CSV
summary_by_qubits.to_csv(os.path.join(output_dir, "summary_by_qubits.csv"), index=False)
summary_by_geometry.to_csv(
    os.path.join(output_dir, "summary_by_geometry.csv"), index=False
)
summary_by_srv.to_csv(os.path.join(output_dir, "summary_by_srv.csv"), index=False)
connectivity_analysis.to_csv(
    os.path.join(output_dir, "connectivity_analysis.csv"), index=False
)

# Create pivot table for heatmap using filtered data
pivot_srv_success = pd.pivot_table(
    summary_by_srv, values="SuccessRate", index="Qubits", columns="SRV_Complexity"
)

# ---- Plots ----
# 1. Success Rate by SRV Complexity Heatmap (excluding baseline)
plt.figure(figsize=FIGURE_SIZES["heatmap"])
heatmap = sns.heatmap(
    pivot_srv_success,
    annot=True,
    cmap="viridis",
    fmt=".3f",
    linewidths=0.5,
    annot_kws={"size": FONT_SIZES["annotations"]},
)
plt.title("Success Rate by SRV Complexity and Qubit Count (Constrained Only)")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "success_rate_by_srv_heatmap.png"), dpi=300)
plt.close()

# 2. Baseline vs Constrained Comparison (keeps both for comparison)
plt.figure(figsize=FIGURE_SIZES["barplot"])
summary_by_baseline = (
    df.groupby(["Qubits", "IsBaseline"]).agg({"SuccessRate": "mean"}).reset_index()
)
# Create a new column with descriptive labels
summary_by_baseline["ArchitectureType"] = summary_by_baseline["IsBaseline"].map(
    {False: "Constrained", True: "Unconstrained"}
)
# Plot using the new column
barplot = sns.barplot(
    data=summary_by_baseline, x="Qubits", y="SuccessRate", hue="ArchitectureType"
)
plt.title("Success Rate: Baseline vs Constrained Architectures")
plt.xlabel("Number of Qubits")
plt.ylabel("Average Success Rate")
plt.legend(title="Architecture Type")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "baseline_vs_constrained.png"), dpi=300)
plt.close()

# 3. MaskEffectiveGates vs Success Rate Scatter (excluding baseline) with discrete colors and log scale
plt.figure(figsize=FIGURE_SIZES["scatter"])

# Define color palette for discrete qubit counts
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
markers = ["o", "s", "^", "D", "P"]  # Different marker shapes for better distinction

# Plot each qubit count separately with a different color
for i, qubit in enumerate(sorted(df_no_baseline["Qubits"].unique())):
    subset = df_no_baseline[df_no_baseline["Qubits"] == qubit]
    plt.scatter(
        subset["MaskEffectiveGates"],
        subset["SuccessRate"],
        alpha=0.7,
        color=colors[i % len(colors)],
        marker=markers[i % len(markers)],
        s=70,  # Point size
        label=f"{qubit} Qubits",
    )

# Set y-axis to logarithmic scale
plt.yscale("log")

plt.xlabel("Mask Gate Sequence Length")
plt.ylabel("Success Rate (log scale)")
plt.title("Success Rate vs Mask Gate Sequence Length (Constrained Only)")
plt.grid(True, alpha=0.3, which="both")  # Add grid lines for both major and minor ticks
plt.legend(title="Qubit Count", loc="best")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "success_rate_vs_mask_gates.png"), dpi=300)
plt.close()

# 4a. SRV Complexity vs Success Rate by Qubit Count - LOG SCALE VERSION
plt.figure(figsize=FIGURE_SIZES["lineplot"])
for qubit in df_no_baseline["Qubits"].unique():
    subset = df_no_baseline[df_no_baseline["Qubits"] == qubit]
    srv_groups = subset.groupby("SRV_Complexity")["SuccessRate"].mean()
    plt.plot(
        srv_groups.index,
        srv_groups.values,
        "o-",
        linewidth=2,
        markersize=8,
        label=f"{qubit} Qubits",
    )

# Set y-axis to logarithmic scale
plt.yscale("log")

plt.xlabel("SRV Complexity")
plt.ylabel("Success Rate (log scale)")
plt.title("Success Rate vs SRV Complexity by Qubit Count (Log Scale, Constrained Only)")
plt.grid(True, alpha=0.3, which="both")  # Add grid lines for both major and minor ticks
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "success_rate_vs_srv_by_qubits_log.png"), dpi=300)
plt.close()

# 4b. SRV Complexity vs Success Rate by Qubit Count - LINEAR SCALE VERSION
plt.figure(figsize=FIGURE_SIZES["lineplot"])
for qubit in df_no_baseline["Qubits"].unique():
    subset = df_no_baseline[df_no_baseline["Qubits"] == qubit]
    srv_groups = subset.groupby("SRV_Complexity")["SuccessRate"].mean()
    plt.plot(
        srv_groups.index,
        srv_groups.values,
        "o-",
        linewidth=2,
        markersize=8,
        label=f"{qubit} Qubits",
    )

# Using default linear scale (no need to set explicitly)
plt.xlabel("SRV Complexity")
plt.ylabel("Success Rate")
plt.title(
    "Success Rate vs SRV Complexity by Qubit Count (Linear Scale, Constrained Only)"
)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(plots_dir, "success_rate_vs_srv_by_qubits_linear.png"), dpi=300
)
plt.close()

# 5a. SRV Complexity vs Error Rate by Qubit Count - LOG SCALE VERSION
plt.figure(figsize=FIGURE_SIZES["lineplot"])
for qubit in df_no_baseline["Qubits"].unique():
    subset = df_no_baseline[df_no_baseline["Qubits"] == qubit]
    srv_groups = subset.groupby("SRV_Complexity")["ErrorRate"].mean()
    plt.plot(
        srv_groups.index,
        srv_groups.values,
        "o-",
        linewidth=2,
        markersize=8,
        label=f"{qubit} Qubits",
    )

# Set y-axis to logarithmic scale
plt.yscale("log")

plt.xlabel("SRV Complexity")
plt.ylabel("Error Rate (log scale)")
plt.title("Error Rate vs SRV Complexity by Qubit Count (Log Scale, Constrained Only)")
plt.grid(True, alpha=0.3, which="both")  # Add grid lines for both major and minor ticks
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "error_rate_vs_srv_by_qubits_log.png"), dpi=300)
plt.close()

# 5b. SRV Complexity vs Error Rate by Qubit Count - LINEAR SCALE VERSION
plt.figure(figsize=FIGURE_SIZES["lineplot"])
for qubit in df_no_baseline["Qubits"].unique():
    subset = df_no_baseline[df_no_baseline["Qubits"] == qubit]
    srv_groups = subset.groupby("SRV_Complexity")["ErrorRate"].mean()
    plt.plot(
        srv_groups.index,
        srv_groups.values,
        "o-",
        linewidth=2,
        markersize=8,
        label=f"{qubit} Qubits",
    )

# Using default linear scale (no need to set explicitly)
plt.xlabel("SRV Complexity")
plt.ylabel("Error Rate")
plt.title(
    "Error Rate vs SRV Complexity by Qubit Count (Linear Scale, Constrained Only)"
)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "error_rate_vs_srv_by_qubits_linear.png"), dpi=300)
plt.close()

print("Analysis completed. Results saved to:", output_dir)
