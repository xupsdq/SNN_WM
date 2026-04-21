from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    title: str
    legacy_module: str
    output_flag: str
    smoke_args: tuple[str, ...]
    expected_artifacts: tuple[str, ...]
    primary_csv: str | None = None
    csv_x: str | None = None
    csv_y: tuple[str, ...] = ()
    csv_group: str | None = None
    supports_skip_figures: bool = True
    supports_model_path: bool = True
    supports_dataset_root: bool = True
    supports_device: bool = True
    supports_seed: bool = True


EXPERIMENT_SPECS: dict[str, ExperimentSpec] = {
    "engram_decode": ExperimentSpec(
        experiment_id="engram_decode",
        title="Engram Decode",
        legacy_module="src.experiments.engram_decode",
        output_flag="--save-dir",
        smoke_args=("--delay-points-ms", "100", "--train-per-class", "1", "--test-per-class", "1", "--batch-size", "1"),
        expected_artifacts=("run_config.json", "summary.json", "metrics/engram_decode_metrics.csv", "metrics/summary.json"),
        primary_csv="engram_decode_metrics.csv",
        csv_x="delay_ms",
        csv_y=("acc",),
        csv_group="layer",
    ),
    "ux_shuffle_memory_collapse": ExperimentSpec(
        experiment_id="ux_shuffle_memory_collapse",
        title="UX Shuffle Memory Collapse",
        legacy_module="src.experiments.ux_shuffle_memory_collapse",
        output_flag="--save-dir",
        smoke_args=("--trials", "12", "--batch-size", "4", "--num-boot", "64"),
        expected_artifacts=("run_config.json", "summary.json", "metrics/metrics_condition_summary.csv", "metrics/metrics_collapse_summary.csv"),
        primary_csv="metrics_condition_summary.csv",
        csv_x="condition",
        csv_y=("acc_probe", "abs_rate_pred_original_sample"),
        supports_dataset_root=False,
        supports_device=False,
    ),
    "similarity_bias_experiment": ExperimentSpec(
        experiment_id="similarity_bias_experiment",
        title="Similarity Bias Experiment",
        legacy_module="src.experiments.similarity_bias_experiment",
        output_flag="--output-dir",
        smoke_args=("--max-pairs", "20", "--max-samples", "20", "--batch-size", "8", "--repeats", "1"),
        expected_artifacts=("run_config.json", "summary.json", "metrics/bin_accuracy_summary.csv"),
        primary_csv="bin_accuracy_summary.csv",
        csv_x="bin_index",
        csv_y=("acc_dynamic", "acc_static", "acc_drop"),
    ),
    "dms_overlap_ux_support_mechanism_experiment": ExperimentSpec(
        experiment_id="dms_overlap_ux_support_mechanism_experiment",
        title="DMS Overlap UX Support Mechanism",
        legacy_module="src.experiments.dms_overlap_ux_support_mechanism_experiment",
        output_flag="--output-dir",
        smoke_args=("--max-probes", "4", "--max-pairs", "12", "--batch-size", "4", "--save-case-count", "1"),
        expected_artifacts=("run_config.json", "summary.json", "metrics/preprobe_stsp_summary.csv"),
        primary_csv="preprobe_stsp_summary.csv",
        csv_x="model_type",
    ),
    "overlap_causal_input_perturbation_experiment": ExperimentSpec(
        experiment_id="overlap_causal_input_perturbation_experiment",
        title="Overlap Causal Input Perturbation",
        legacy_module="src.experiments.overlap_causal_input_perturbation_experiment",
        output_flag="--output-dir",
        smoke_args=("--max-probes", "2", "--samples-per-probe", "2", "--max-pairs", "8", "--batch-size", "4", "--num-control-candidates", "4", "--save-case-count", "1"),
        expected_artifacts=("run_config.json", "summary.json", "data/pair_condition_pattern_results.csv", "metrics/summary_metrics.json"),
        primary_csv="pair_condition_pattern_results.csv",
    ),
    "l3_accumulator_mechanism_experiment": ExperimentSpec(
        experiment_id="l3_accumulator_mechanism_experiment",
        title="L3 Accumulator Mechanism",
        legacy_module="src.experiments.l3_accumulator_mechanism_experiment",
        output_flag="--output-dir",
        smoke_args=("--max-probes", "2", "--samples-per-probe", "2", "--max-pairs", "8", "--batch-size", "4", "--save-case-count", "1"),
        expected_artifacts=("run_config.json", "summary.json", "data/pair_results.csv", "metrics/summary_metrics.json", "meta/plot_bundle_manifest.json"),
        primary_csv="pair_results.csv",
    ),
    "chunk_step2_fused_state_experiment": ExperimentSpec(
        experiment_id="chunk_step2_fused_state_experiment",
        title="Chunk Step2 Fused State",
        legacy_module="src.experiments.chunk_step2_fused_state_experiment",
        output_flag="--output-dir",
        smoke_args=("--smoke",),
        expected_artifacts=("run_config.json", "summary.json", "metrics/preprobe_fusion_metrics.csv", "meta/plot_bundle_manifest.json"),
        primary_csv="preprobe_fusion_metrics.csv",
    ),
    "chunk_stsp_state_taxonomy": ExperimentSpec(
        experiment_id="chunk_stsp_state_taxonomy",
        title="Chunk STSP State Taxonomy",
        legacy_module="src.experiments.chunk_stsp_state_taxonomy",
        output_flag="--output-dir",
        smoke_args=("--smoke",),
        expected_artifacts=("run_config.json", "summary.json", "metrics/state_similarity_metrics.csv"),
        primary_csv="state_similarity_metrics.csv",
    ),
    "chunk_stsp_multiitem_sequence": ExperimentSpec(
        experiment_id="chunk_stsp_multiitem_sequence",
        title="Chunk STSP Multiitem Sequence",
        legacy_module="src.experiments.chunk_stsp_multiitem_sequence",
        output_flag="--output-dir",
        smoke_args=("--smoke",),
        expected_artifacts=("run_config.json", "summary.json", "metrics/similarity_summary_metrics.csv"),
        primary_csv="similarity_summary_metrics.csv",
    ),
    "chunk_stsp_layer3_anchor_drift_mechanism": ExperimentSpec(
        experiment_id="chunk_stsp_layer3_anchor_drift_mechanism",
        title="Chunk STSP Layer3 Anchor Drift",
        legacy_module="src.experiments.chunk_stsp_layer3_anchor_drift_mechanism",
        output_flag="--output-dir",
        smoke_args=("--smoke",),
        expected_artifacts=("run_config.json", "summary.json", "metrics/layer3_changed_rank_metrics.csv"),
        primary_csv="layer3_changed_rank_metrics.csv",
    ),
}


def get_experiment_spec(experiment_id: str) -> ExperimentSpec:
    try:
        return EXPERIMENT_SPECS[experiment_id]
    except KeyError as exc:
        known = ", ".join(sorted(EXPERIMENT_SPECS))
        raise KeyError(f"Unknown experiment_id={experiment_id!r}. Known: {known}") from exc


__all__ = ["EXPERIMENT_SPECS", "ExperimentSpec", "get_experiment_spec"]
