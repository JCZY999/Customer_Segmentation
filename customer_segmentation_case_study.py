"""End-to-end customer segmentation case study.

Run:
    python customer_segmentation_case_study.py
    python customer_segmentation_case_study.py --data Mall_Customers.csv --output-dir outputs

The script validates the Mall Customers dataset, compares K-Means candidates,
fits a reproducible five-segment model, and exports tables and charts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/sharmaroshan/"
    "Clustering-of-Mall-Customers/master/Mall_Customers.csv"
)
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
SEGMENT_ORDER = [
    "VIP Customers", "Affluent but Unengaged", "Core Customers",
    "Promising Spenders", "Budget Conscious",
]
COLORS = {
    "VIP Customers": "#0f766e", "Affluent but Unengaged": "#d97706",
    "Core Customers": "#2563eb", "Promising Spenders": "#7c3aed",
    "Budget Conscious": "#64748b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA_URL,
                        help="CSV path or URL (defaults to the public Mall Customers dataset).")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="Directory for CSV and chart outputs (default: outputs).")
    return parser.parse_args()


def load_and_validate(source: str) -> pd.DataFrame:
    customers = pd.read_csv(source).rename(columns={"Genre": "Gender"})
    required = {"CustomerID", "Gender", "Age", *FEATURES}
    missing = sorted(required.difference(customers.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if customers.empty:
        raise ValueError("The dataset contains no customer records.")
    if not customers["CustomerID"].is_unique:
        raise ValueError("CustomerID must be unique.")
    if customers[list(required)].isna().any().any():
        raise ValueError("Required columns contain missing values.")
    if (customers[FEATURES] < 0).any().any():
        raise ValueError("Income and spending score cannot be negative.")
    return customers


def choose_models(scaled_features, k_min: int = 2,
                  k_max: int = 10) -> tuple[list[float], dict[int, float]]:
    inertias, silhouettes = [], {}
    for k in range(1, k_max + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(scaled_features)
        inertias.append(float(model.inertia_))
        if k >= k_min:
            silhouettes[k] = float(silhouette_score(scaled_features, labels))
    return inertias, silhouettes


def name_segment(income: float, spending_score: float) -> str:
    """Translate a centroid into a stable, business-readable name."""
    if income < 40:
        return "Promising Spenders" if spending_score >= 50 else "Budget Conscious"
    if income > 70:
        return "VIP Customers" if spending_score >= 50 else "Affluent but Unengaged"
    return "Core Customers"


def fit_segments(customers: pd.DataFrame):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(customers[FEATURES])
    model = KMeans(n_clusters=5, random_state=42, n_init=20)
    scored = customers.copy()
    scored["cluster"] = model.fit_predict(scaled)
    centroids = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_), columns=FEATURES
    )
    centroids["cluster"] = centroids.index
    centroids["segment"] = [
        name_segment(row[FEATURES[0]], row[FEATURES[1]])
        for _, row in centroids.iterrows()
    ]
    if centroids["segment"].nunique() != 5:
        raise RuntimeError("Centroid rules did not produce five unique segment names.")
    scored["segment"] = scored["cluster"].map(
        centroids.set_index("cluster")["segment"]
    )
    return scored, centroids


def build_profiles(scored: pd.DataFrame) -> pd.DataFrame:
    profiles = (
        scored.groupby("segment")
        .agg(
            customers=("CustomerID", "count"),
            average_age=("Age", "mean"),
            average_income=(FEATURES[0], "mean"),
            average_spending_score=(FEATURES[1], "mean"),
            female_share=("Gender", lambda values: (values == "Female").mean()),
        )
        .reindex(SEGMENT_ORDER)
    )
    profiles["customer_share"] = profiles["customers"] / len(scored)
    return profiles.reset_index()


def plot_model_selection(inertias, silhouettes, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(range(1, 11), inertias, marker="o", color="#2563eb")
    axes[0].axvline(5, linestyle="--", color="#0f766e", label="Selected k=5")
    axes[0].set(title="Elbow diagnostic", xlabel="Number of clusters", ylabel="Inertia")
    axes[0].legend()
    axes[1].plot(list(silhouettes), list(silhouettes.values()),
                 marker="o", color="#7c3aed")
    axes[1].axvline(5, linestyle="--", color="#0f766e", label="Selected k=5")
    axes[1].set(title="Silhouette diagnostic", xlabel="Number of clusters",
                ylabel="Silhouette score")
    axes[1].legend()
    fig.suptitle("Model selection balances separation and business usefulness")
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_segments(scored: pd.DataFrame, centroids: pd.DataFrame,
                  output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for segment in SEGMENT_ORDER:
        subset = scored[scored["segment"] == segment]
        ax.scatter(subset[FEATURES[0]], subset[FEATURES[1]], s=48, alpha=0.75,
                   color=COLORS[segment],
                   label=f"{segment} (n={len(subset)})")
    ax.scatter(centroids[FEATURES[0]], centroids[FEATURES[1]], marker="X",
               s=190, color="#111827", edgecolor="white", linewidth=1.2,
               label="Centroids")
    ax.set(title="Five actionable customer segments",
           xlabel="Annual income (k$)", ylabel="Spending score (1–100)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_profiles(profiles: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    colors = [COLORS[name] for name in profiles["segment"]]
    axes[0].barh(profiles["segment"], profiles["average_income"], color=colors)
    axes[0].invert_yaxis()
    axes[0].set(title="Average annual income", xlabel="Income (k$)")
    axes[1].barh(profiles["segment"], profiles["average_spending_score"],
                 color=colors)
    axes[1].invert_yaxis()
    axes[1].set(title="Average spending score", xlabel="Score (1–100)")
    fig.suptitle("Segment profiles reveal different commercial opportunities")
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    customers = load_and_validate(args.data)
    scaled = StandardScaler().fit_transform(customers[FEATURES])
    inertias, silhouettes = choose_models(scaled)
    scored, centroids = fit_segments(customers)
    profiles = build_profiles(scored)
    scored.to_csv(args.output_dir / "customer_segment_assignments.csv", index=False)
    profiles.to_csv(args.output_dir / "segment_profiles.csv", index=False)
    centroids.to_csv(args.output_dir / "segment_centroids.csv", index=False)
    plot_model_selection(inertias, silhouettes, args.output_dir / "model_selection.png")
    plot_segments(scored, centroids, args.output_dir / "customer_segments.png")
    plot_profiles(profiles, args.output_dir / "segment_profiles.png")
    print(f"Analyzed {len(scored):,} customers")
    print(f"Five-cluster silhouette score: {silhouettes[5]:.3f}\n")
    print(profiles.to_string(index=False, formatters={
        "average_age": "{:.1f}".format,
        "average_income": "{:.1f}".format,
        "average_spending_score": "{:.1f}".format,
        "female_share": "{:.1%}".format,
        "customer_share": "{:.1%}".format,
    }))
    print(f"\nOutputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
