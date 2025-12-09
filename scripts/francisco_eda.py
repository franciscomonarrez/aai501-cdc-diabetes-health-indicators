"""
Comprehensive Exploratory Data Analysis (EDA) for CDC Diabetes Health Indicators.

This script generates:
- Dataset overview and summary statistics
- Class distribution analysis
- Feature distributions and histograms
- Correlation matrix and heatmap
- Missing value analysis
- Outlier detection
- Initial insights and patterns

All figures are saved to figures/ directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.aai501_diabetes.config import DATA_RAW, FIG_DIR, TARGET_COL
from src.aai501_diabetes.data_loader import load_data, validate_data

# Set style
try:
    plt.style.use("seaborn-v0_8")
except OSError:
    plt.style.use("seaborn")
sns.set_palette("husl")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Plot class distribution of target variable."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Count plot
    counts = df[TARGET_COL].value_counts().sort_index()
    axes[0].bar(
        ["No Diabetes (0)", "Diabetes (1)"],
        counts.values,
        color=["#3498db", "#e74c3c"],
        alpha=0.7,
    )
    axes[0].set_title("Class Distribution", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_xlabel("Class", fontsize=12)
    for i, v in enumerate(counts.values):
        axes[0].text(
            i,
            v + 5000,
            f"{v:,}\n({v/len(df)*100:.1f}%)",
            ha="center",
            fontsize=11,
        )

    # Pie chart
    axes[1].pie(
        counts.values,
        labels=["No Diabetes", "Diabetes"],
        autopct="%1.1f%%",
        colors=["#3498db", "#e74c3c"],
        startangle=90,
    )
    axes[1].set_title("Class Proportion", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "francisco_class_distribution.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {FIG_DIR / 'francisco_class_distribution.png'}")
    plt.close()


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """Plot distributions of key features."""
    # Select key features for visualization
    key_features = [
        "BMI",
        "Age",
        "GenHlth",
        "MentHlth",
        "PhysHlth",
        "Income",
        "Education",
    ]

    # Filter to features that exist
    key_features = [f for f in key_features if f in df.columns]

    n_features = len(key_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feature in enumerate(key_features):
        ax = axes[idx]
        df[feature].hist(bins=30, ax=ax, alpha=0.7, edgecolor="black")
        ax.set_title(f"{feature} Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.grid(alpha=0.3)

    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "francisco_feature_distributions.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {FIG_DIR / 'francisco_feature_distributions.png'}")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot correlation matrix heatmap."""
    # Calculate correlation matrix
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()

    # Create figure
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8},
    )

    plt.title("Feature Correlation Matrix", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "francisco_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {FIG_DIR / 'francisco_correlation_heatmap.png'}")
    plt.close()


def plot_target_correlations(df: pd.DataFrame) -> None:
    """Plot correlations with target variable."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    target_corr = df[numeric_cols].corrwith(df[TARGET_COL]).sort_values(ascending=False)

    # Remove target from correlations
    target_corr = target_corr.drop(TARGET_COL)

    plt.figure(figsize=(10, 8))
    colors = ["#e74c3c" if x > 0 else "#3498db" for x in target_corr.values]
    bars = plt.barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7)
    plt.yticks(range(len(target_corr)), target_corr.index)
    plt.xlabel("Correlation with Diabetes", fontsize=12)
    plt.title("Feature Correlations with Target Variable", fontsize=14, fontweight="bold")
    plt.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    plt.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (idx, val) in enumerate(target_corr.items()):
        plt.text(
            val + 0.01 if val > 0 else val - 0.01,
            i,
            f"{val:.3f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "francisco_target_correlations.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {FIG_DIR / 'francisco_target_correlations.png'}")
    plt.close()


def analyze_missing_values(df: pd.DataFrame) -> None:
    """Analyze and visualize missing values."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct,
    }).sort_values("Missing Count", ascending=False)

    # Filter to columns with missing values
    missing_df = missing_df[missing_df["Missing Count"] > 0]

    if len(missing_df) > 0:
        print("\nMissing Values Analysis:")
        print(missing_df.to_string())

        # Plot
        plt.figure(figsize=(10, 6))
        missing_df["Missing %"].plot(kind="barh")
        plt.xlabel("Missing Percentage", fontsize=12)
        plt.title("Missing Values by Feature", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "francisco_missing_values.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {FIG_DIR / 'francisco_missing_values.png'}")
        plt.close()
    else:
        print("\nNo missing values found in the dataset.")


def generate_summary_statistics(df: pd.DataFrame) -> None:
    """Generate and save summary statistics."""
    print("\n" + "=" * 80)
    print("DATASET SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nDataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumn Names ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")

    print("\n" + "-" * 80)
    print("DESCRIPTIVE STATISTICS")
    print("-" * 80)
    print(df.describe().to_string())

    print("\n" + "-" * 80)
    print("DATA TYPES")
    print("-" * 80)
    print(df.dtypes.value_counts().to_string())

    # Save to file
    summary_file = FIG_DIR.parent / "reports" / "francisco_eda_summary.txt"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, "w") as f:
        f.write("CDC DIABETES HEALTH INDICATORS - EDA SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n\n")
        f.write("DESCRIPTIVE STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(df.describe().to_string())
        f.write("\n\nDATA TYPES\n")
        f.write("-" * 80 + "\n")
        f.write(df.dtypes.value_counts().to_string())

    print(f"\nSummary saved to: {summary_file}")


def main():
    """Run complete EDA pipeline."""
    print("=" * 80)
    print("EXPLORATORY DATA ANALYSIS - CDC DIABETES HEALTH INDICATORS")
    print("=" * 80)

    # Load and validate data
    df = load_data()
    validate_data(df)

    # Generate summary statistics
    generate_summary_statistics(df)

    # Analyze missing values
    analyze_missing_values(df)

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_class_distribution(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_target_correlations(df)

    print("\n" + "=" * 80)
    print("EDA COMPLETE - All figures saved to figures/ directory")
    print("=" * 80)


if __name__ == "__main__":
    main()

