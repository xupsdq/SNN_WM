from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_ORDER: Tuple[str, ...] = ("static", "dynamic")
MODEL_COLORS: Dict[str, str] = {
    "static": "#8C8C8C",
    "dynamic": "#D55E00",
}
EXPECTED_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "model_type": ("model_type", "stsp_mode"),
    "probe_class": ("probe_class", "probe_label"),
    "sample_class": ("sample_class", "sample_label"),
    "pred_class": ("pred_class", "predicted_label", "prediction_probe", "pred_label"),
}


def canonical_model_type(raw_value: object) -> str:
    text = str(raw_value).strip().lower()
    if text in {"static", "static_frozen", "frozen", "baseline"}:
        return "static"
    if text in {"dynamic", "stsp_on"}:
        return "dynamic"
    return text


def score_candidate(path: Path, columns: Iterable[str]) -> int:
    score = 0
    path_str = str(path).lower()
    if "fixed_probe_varied_sample" in path_str:
        score += 1000
    if "trial_level_predictions" in path.name.lower():
        score += 200
    if "trial_level_results" in path.name.lower():
        score += 120
    if "trial_predictions" in path.name.lower():
        score += 80
    if "metrics" in path_str:
        score += 30
    col_set = {str(col).strip() for col in columns}
    for aliases in EXPECTED_COLUMNS.values():
        if any(alias in col_set for alias in aliases):
            score += 40
        else:
            score -= 200
    return int(score)


def locate_trial_csv(results_root: Path, explicit_path: str | None = None) -> Path:
    if explicit_path is not None and str(explicit_path).strip():
        resolved = Path(explicit_path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Trial CSV not found: {resolved}")
        return resolved

    candidates: List[Tuple[int, float, Path]] = []
    for path in results_root.rglob("*.csv"):
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue
        score = score_candidate(path=path, columns=header.columns.tolist())
        if score <= 0:
            continue
        candidates.append((score, path.stat().st_mtime, path))

    if not candidates:
        raise FileNotFoundError(f"No suitable trial-level CSV was found under {results_root.resolve()}")

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2].resolve()


def adapt_trial_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    mapping: Dict[str, str] = {}
    for canonical_name, aliases in EXPECTED_COLUMNS.items():
        found = None
        for alias in aliases:
            if alias in df_raw.columns:
                found = alias
                break
        if found is None:
            raise ValueError(f"Missing required column for {canonical_name}. Tried aliases: {aliases}")
        mapping[canonical_name] = found

    df = df_raw.rename(columns={raw: canonical for canonical, raw in mapping.items()}).copy()
    df = df[["model_type", "probe_class", "sample_class", "pred_class"]].copy()
    df["model_type"] = df["model_type"].map(canonical_model_type)
    for column in ["probe_class", "sample_class", "pred_class"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["probe_class", "sample_class", "pred_class"]).copy()
    for column in ["probe_class", "sample_class", "pred_class"]:
        df[column] = df[column].astype(np.int64)
    return df


def compute_capture_metrics(df_subset: pd.DataFrame, chance_rate: float) -> Dict[str, float | int]:
    n_trials = int(len(df_subset))
    error_mask = df_subset["pred_class"].to_numpy(dtype=np.int64, copy=False) != df_subset["probe_class"].to_numpy(dtype=np.int64, copy=False)
    n_errors = int(error_mask.sum())
    if n_errors <= 0:
        return {
            "n_trials": n_trials,
            "n_errors": 0,
            "sample_capture": float("nan"),
            "excess_capture": float("nan"),
            "enrichment": float("nan"),
        }

    pred = df_subset["pred_class"].to_numpy(dtype=np.int64, copy=False)
    sample = df_subset["sample_class"].to_numpy(dtype=np.int64, copy=False)
    sample_capture = float(np.mean(pred[error_mask] == sample[error_mask]))
    excess_capture = float(sample_capture - chance_rate)
    enrichment = float(sample_capture / chance_rate) if chance_rate > 0 else float("nan")
    return {
        "n_trials": n_trials,
        "n_errors": n_errors,
        "sample_capture": sample_capture,
        "excess_capture": excess_capture,
        "enrichment": enrichment,
    }


def summarize_overall(df_trials: pd.DataFrame, chance_rate: float) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    for model_type in MODEL_ORDER:
        subset = df_trials[df_trials["model_type"] == model_type].copy()
        stats = compute_capture_metrics(subset, chance_rate=chance_rate)
        rows.append(
            {
                "model_type": str(model_type),
                "total_trials": int(stats["n_trials"]),
                "total_errors": int(stats["n_errors"]),
                "sample_capture": float(stats["sample_capture"]),
                "excess_capture": float(stats["excess_capture"]),
                "enrichment": float(stats["enrichment"]),
            }
        )
    df = pd.DataFrame(rows)
    static_row = df[df["model_type"] == "static"].iloc[0]
    dynamic_row = df[df["model_type"] == "dynamic"].iloc[0]
    delta_sample = float(dynamic_row["sample_capture"] - static_row["sample_capture"])
    delta_excess = float(dynamic_row["excess_capture"] - static_row["excess_capture"])
    delta_enrichment = float(dynamic_row["enrichment"] - static_row["enrichment"])
    df["delta_sample_capture_vs_static"] = np.nan
    df["delta_excess_capture_vs_static"] = np.nan
    df["delta_enrichment_vs_static"] = np.nan
    df.loc[df["model_type"] == "dynamic", "delta_sample_capture_vs_static"] = delta_sample
    df.loc[df["model_type"] == "dynamic", "delta_excess_capture_vs_static"] = delta_excess
    df.loc[df["model_type"] == "dynamic", "delta_enrichment_vs_static"] = delta_enrichment
    return df


def summarize_by_probe(df_trials: pd.DataFrame, chance_rate: float) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    for (model_type, probe_class), subset in df_trials.groupby(["model_type", "probe_class"], sort=True):
        stats = compute_capture_metrics(subset, chance_rate=chance_rate)
        rows.append(
            {
                "model_type": str(model_type),
                "probe_class": int(probe_class),
                "n_trials": int(stats["n_trials"]),
                "n_errors": int(stats["n_errors"]),
                "sample_capture": float(stats["sample_capture"]),
                "excess_capture": float(stats["excess_capture"]),
                "enrichment": float(stats["enrichment"]),
            }
        )
    df = pd.DataFrame(rows).sort_values(["probe_class", "model_type"], kind="stable").reset_index(drop=True)
    static = df[df["model_type"] == "static"][["probe_class", "sample_capture", "excess_capture", "enrichment"]].rename(
        columns={
            "sample_capture": "sample_capture_static",
            "excess_capture": "excess_capture_static",
            "enrichment": "enrichment_static",
        }
    )
    dynamic = df[df["model_type"] == "dynamic"][["probe_class", "sample_capture", "excess_capture", "enrichment"]].rename(
        columns={
            "sample_capture": "sample_capture_dynamic",
            "excess_capture": "excess_capture_dynamic",
            "enrichment": "enrichment_dynamic",
        }
    )
    delta = static.merge(dynamic, on="probe_class", how="inner", validate="one_to_one")
    delta["delta_sample_capture_vs_static"] = delta["sample_capture_dynamic"] - delta["sample_capture_static"]
    delta["delta_excess_capture_vs_static"] = delta["excess_capture_dynamic"] - delta["excess_capture_static"]
    delta["delta_enrichment_vs_static"] = delta["enrichment_dynamic"] - delta["enrichment_static"]
    df = df.merge(
        delta[["probe_class", "delta_sample_capture_vs_static", "delta_excess_capture_vs_static", "delta_enrichment_vs_static"]],
        on="probe_class",
        how="left",
        validate="many_to_one",
    )
    return df


def summarize_by_pair(df_trials: pd.DataFrame, chance_rate: float) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    for (model_type, sample_class, probe_class), subset in df_trials.groupby(["model_type", "sample_class", "probe_class"], sort=True):
        stats = compute_capture_metrics(subset, chance_rate=chance_rate)
        rows.append(
            {
                "model_type": str(model_type),
                "sample_class": int(sample_class),
                "probe_class": int(probe_class),
                "n_trials": int(stats["n_trials"]),
                "n_errors": int(stats["n_errors"]),
                "sample_capture": float(stats["sample_capture"]),
                "excess_capture": float(stats["excess_capture"]),
                "enrichment": float(stats["enrichment"]),
            }
        )
    df = pd.DataFrame(rows).sort_values(["probe_class", "sample_class", "model_type"], kind="stable").reset_index(drop=True)
    static = df[df["model_type"] == "static"][["sample_class", "probe_class", "sample_capture", "excess_capture", "enrichment"]].rename(
        columns={
            "sample_capture": "sample_capture_static",
            "excess_capture": "excess_capture_static",
            "enrichment": "enrichment_static",
        }
    )
    dynamic = df[df["model_type"] == "dynamic"][["sample_class", "probe_class", "sample_capture", "excess_capture", "enrichment"]].rename(
        columns={
            "sample_capture": "sample_capture_dynamic",
            "excess_capture": "excess_capture_dynamic",
            "enrichment": "enrichment_dynamic",
        }
    )
    delta = static.merge(dynamic, on=["sample_class", "probe_class"], how="inner", validate="one_to_one")
    delta["delta_sample_capture_vs_static"] = delta["sample_capture_dynamic"] - delta["sample_capture_static"]
    delta["delta_excess_capture_vs_static"] = delta["excess_capture_dynamic"] - delta["excess_capture_static"]
    delta["delta_enrichment_vs_static"] = delta["enrichment_dynamic"] - delta["enrichment_static"]
    df = df.merge(
        delta[
            [
                "sample_class",
                "probe_class",
                "delta_sample_capture_vs_static",
                "delta_excess_capture_vs_static",
                "delta_enrichment_vs_static",
            ]
        ],
        on=["sample_class", "probe_class"],
        how="left",
        validate="many_to_one",
    )
    return df


def infer_output_dir(trial_csv_path: Path, explicit_output_dir: str | None = None) -> Path:
    if explicit_output_dir is not None and str(explicit_output_dir).strip():
        return Path(explicit_output_dir).resolve()
    parent = trial_csv_path.parent
    if parent.name.lower() == "metrics":
        return parent.parent / "sample_capture"
    return parent / "sample_capture"


def save_csv(df: pd.DataFrame, path: Path, sort_by: List[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_df = df.sort_values(sort_by, kind="stable").reset_index(drop=True)
    out_df.to_csv(path, index=False, encoding="utf-8")
    return path


def make_overall_plot(df_overall: pd.DataFrame, chance_rate: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    plot_df = df_overall.copy().set_index("model_type").reindex(MODEL_ORDER).reset_index()
    x = np.arange(len(plot_df), dtype=np.float64)
    values = plot_df["sample_capture"].to_numpy(dtype=np.float64) * 100.0
    colors = [MODEL_COLORS[str(v)] for v in plot_df["model_type"]]
    ax.bar(x, values, color=colors, width=0.62, edgecolor="black", linewidth=0.7)
    ax.axhline(chance_rate * 100.0, color="#1F77B4", linestyle="--", linewidth=1.3, label="Chance = 11.11%")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v).title() for v in plot_df["model_type"]])
    ax.set_ylabel("SampleCapture (%)")
    ax.set_title("Overall sample capture")
    ax.set_ylim(0.0, max(chance_rate * 100.0 * 1.4, np.nanmax(values) + 5.0))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def make_probe_plot(df_probe: pd.DataFrame, chance_rate: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    probe_classes = sorted(pd.unique(df_probe["probe_class"]).tolist())
    x = np.arange(len(probe_classes), dtype=np.float64)
    width = 0.36
    for idx, model_type in enumerate(MODEL_ORDER):
        subset = (
            df_probe[df_probe["model_type"] == model_type]
            .set_index("probe_class")
            .reindex(probe_classes)
            .reset_index()
        )
        values = subset["sample_capture"].to_numpy(dtype=np.float64) * 100.0
        offset = (-0.5 if idx == 0 else 0.5) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            color=MODEL_COLORS[model_type],
            edgecolor="black",
            linewidth=0.6,
            label=model_type.title(),
        )
    ax.axhline(chance_rate * 100.0, color="#1F77B4", linestyle="--", linewidth=1.2, label="Chance = 11.11%")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in probe_classes])
    ax.set_xlabel("Probe class")
    ax.set_ylabel("SampleCapture (%)")
    ax.set_title("Sample capture by probe")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    return fig


def build_pair_matrix(df_pair: pd.DataFrame, value_column: str, model_type: str, num_classes: int) -> np.ndarray:
    subset = df_pair[df_pair["model_type"] == model_type].copy()
    matrix = np.full((int(num_classes), int(num_classes)), np.nan, dtype=np.float64)
    for row in subset.itertuples(index=False):
        matrix[int(row.sample_class), int(row.probe_class)] = float(getattr(row, value_column))
    for idx in range(int(num_classes)):
        matrix[idx, idx] = np.nan
    return matrix


def make_heatmap(matrix: np.ndarray, title: str, cbar_label: str, cmap_name: str, symmetric: bool) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    display = matrix * 100.0
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#F5F5F5")
    finite = display[np.isfinite(display)]
    if symmetric:
        vmax = float(np.max(np.abs(finite))) if finite.size > 0 else 1.0
        if vmax <= 0.0:
            vmax = 1.0
        vmin = -vmax
    else:
        vmin = 0.0
        vmax = float(np.max(finite)) if finite.size > 0 else 1.0
        if vmax <= 0.0:
            vmax = 1.0
    im = ax.imshow(display, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xticklabels([str(i) for i in range(matrix.shape[1])])
    ax.set_yticklabels([str(i) for i in range(matrix.shape[0])])
    ax.set_xlabel("Probe class")
    ax.set_ylabel("Sample class")
    ax.set_title(title)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(base_path.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.05, dpi=300)


def build_summary_text(
    df_overall: pd.DataFrame,
    df_probe: pd.DataFrame,
    df_pair: pd.DataFrame,
    chance_rate: float,
    input_label: str,
) -> str:
    overall = df_overall.set_index("model_type")
    dynamic = overall.loc["dynamic"]
    static = overall.loc["static"]
    dynamic_probe = (
        df_probe[df_probe["model_type"] == "dynamic"]
        .sort_values(["sample_capture", "n_errors", "probe_class"], ascending=[False, False, True], kind="stable")
        .iloc[0]
    )
    dynamic_pairs = (
        df_pair[df_pair["model_type"] == "dynamic"]
        .sort_values(["sample_capture", "n_errors", "probe_class", "sample_class"], ascending=[False, False, True, True], kind="stable")
        .head(3)
    )
    top_pair_text = "; ".join(
        [
            f"sample {int(row.sample_class)} -> probe {int(row.probe_class)} "
            f"({float(row.sample_capture) * 100.0:.2f}%, n_errors={int(row.n_errors)})"
            for row in dynamic_pairs.itertuples(index=False)
        ]
    )
    lines = [
        "Sample capture summary",
        "",
        f"Input trial CSV: {input_label}",
        f"Chance baseline = {chance_rate * 100.0:.2f}%.",
        f"Dynamic overall SampleCapture: {float(dynamic['sample_capture']) * 100.0:.2f}% ({int(dynamic['total_errors'])} errors out of {int(dynamic['total_trials'])} trials).",
        f"Static overall SampleCapture: {float(static['sample_capture']) * 100.0:.2f}% ({int(static['total_errors'])} errors out of {int(static['total_trials'])} trials).",
        f"Dynamic minus static SampleCapture: {(float(dynamic['sample_capture']) - float(static['sample_capture'])) * 100.0:.2f} pp.",
        f"Dynamic enrichment above chance: {float(dynamic['enrichment']):.2f}x random-error baseline.",
        (
            f"Most sample-captured probe in dynamic: probe {int(dynamic_probe['probe_class'])} "
            f"(SampleCapture = {float(dynamic_probe['sample_capture']) * 100.0:.2f}%, n_errors = {int(dynamic_probe['n_errors'])})."
        ),
        f"Strongest dynamic pairs: {top_pair_text}.",
        "",
        (
            "In dynamic STSP trials, "
            f"{float(dynamic['sample_capture']) * 100.0:.2f}% of all probe errors were redirected specifically to the current sample class, "
            f"compared with {float(static['sample_capture']) * 100.0:.2f}% in the static baseline "
            f"(chance = {chance_rate * 100.0:.2f}%), indicating a {float(dynamic['enrichment']):.2f}-fold enrichment toward the sample label."
        ),
    ]
    return "\n".join(lines) + "\n"


def generate_sample_capture_outputs(
    df_trials: pd.DataFrame,
    output_dir: Path,
    num_classes: int,
    *,
    input_label: str,
) -> Dict[str, object]:
    if int(num_classes) < 2:
        raise ValueError("num_classes must be at least 2.")

    chance_rate = 1.0 / float(int(num_classes) - 1)
    clean_trials = df_trials[["model_type", "probe_class", "sample_class", "pred_class"]].copy()
    clean_trials["model_type"] = clean_trials["model_type"].map(canonical_model_type)
    for column in ["probe_class", "sample_class", "pred_class"]:
        clean_trials[column] = pd.to_numeric(clean_trials[column], errors="coerce")
    clean_trials = clean_trials.dropna(subset=["probe_class", "sample_class", "pred_class"]).copy()
    for column in ["probe_class", "sample_class", "pred_class"]:
        clean_trials[column] = clean_trials[column].astype(np.int64)
    clean_trials = clean_trials[clean_trials["sample_class"] != clean_trials["probe_class"]].copy()
    if clean_trials.empty:
        raise ValueError("No mismatched trials remain after filtering sample_class != probe_class.")

    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df_overall = summarize_overall(df_trials=clean_trials, chance_rate=chance_rate)
    df_probe = summarize_by_probe(df_trials=clean_trials, chance_rate=chance_rate)
    df_pair = summarize_by_pair(df_trials=clean_trials, chance_rate=chance_rate)

    overall_csv = save_csv(df_overall, metrics_dir / "overall_sample_capture.csv", ["model_type"])
    probe_csv = save_csv(df_probe, metrics_dir / "sample_capture_by_probe.csv", ["probe_class", "model_type"])
    pair_csv = save_csv(df_pair, metrics_dir / "sample_capture_by_pair.csv", ["probe_class", "sample_class", "model_type"])

    fig1 = make_overall_plot(df_overall=df_overall, chance_rate=chance_rate)
    save_figure(fig1, figures_dir / "figure1_overall_sample_capture")
    plt.close(fig1)

    fig2 = make_probe_plot(df_probe=df_probe, chance_rate=chance_rate)
    save_figure(fig2, figures_dir / "figure2_sample_capture_by_probe")
    plt.close(fig2)

    dynamic_matrix = build_pair_matrix(df_pair=df_pair, value_column="sample_capture", model_type="dynamic", num_classes=int(num_classes))
    fig3 = make_heatmap(dynamic_matrix, "Dynamic SampleCapture by pair", "SampleCapture (%)", "viridis", symmetric=False)
    save_figure(fig3, figures_dir / "figure3_dynamic_pair_heatmap")
    plt.close(fig3)

    delta_pair = (
        df_pair[df_pair["model_type"] == "dynamic"][
            ["sample_class", "probe_class", "delta_sample_capture_vs_static"]
        ]
        .copy()
        .rename(columns={"delta_sample_capture_vs_static": "delta_sample_capture"})
    )
    delta_matrix = np.full((int(num_classes), int(num_classes)), np.nan, dtype=np.float64)
    for row in delta_pair.itertuples(index=False):
        delta_matrix[int(row.sample_class), int(row.probe_class)] = float(row.delta_sample_capture)
    for idx in range(int(num_classes)):
        delta_matrix[idx, idx] = np.nan
    fig4 = make_heatmap(delta_matrix, "Dynamic - static SampleCapture", "DeltaSampleCapture (pp)", "RdBu_r", symmetric=True)
    save_figure(fig4, figures_dir / "figure4_delta_sample_capture_heatmap")
    plt.close(fig4)

    summary_text = build_summary_text(
        df_overall=df_overall,
        df_probe=df_probe,
        df_pair=df_pair,
        chance_rate=chance_rate,
        input_label=input_label,
    )
    summary_path = output_dir / "sample_capture_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    sentence = summary_text.strip().splitlines()[-1]
    return {
        "chance_rate": chance_rate,
        "overall": df_overall,
        "by_probe": df_probe,
        "by_pair": df_pair,
        "overall_csv": overall_csv,
        "probe_csv": probe_csv,
        "pair_csv": pair_csv,
        "summary_path": summary_path,
        "sentence": sentence,
    }
