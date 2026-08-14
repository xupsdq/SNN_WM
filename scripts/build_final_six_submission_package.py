from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


FIGURE_IDS = tuple(f"fig{index}" for index in range(1, 7))
EXPECTED_PANEL_IDS = {
    "fig1": tuple("abcde"),
    "fig2": tuple("abcd"),
    "fig3": tuple("abcdefg"),
    "fig4": tuple("abcd"),
    "fig5": tuple("abcdef"),
    "fig6": tuple("abcdef"),
}
CHAIN_ROLES = {
    "fig1": ("inherit", "继承：活动静默 STSP 状态"),
    "fig2": ("transition", "转移：相同当前输入下的历史条件化更新"),
    "fig3": ("implement", "实现：一次状态转移的局部机制"),
    "fig4": ("recur", "递归：连续输入下的反复改写"),
    "fig5": ("organize", "组织：多成分、序列化且受限的内部结构"),
    "fig6": ("access", "访问：后续输入对结构化状态的条件性利用"),
}
TEXT_SUFFIXES = {".csv", ".json", ".md", ".sha256", ".txt", ".yaml", ".yml"}
MOJIBAKE_MARKERS = ("\ufffd", "锟斤拷", "閿熸枻鎷", "Ã", "Â")


@dataclass(frozen=True)
class CopyRecord:
    source_path: str
    packaged_path: str
    size_bytes: int
    sha256_before: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = content.replace("\r\n", "\n")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(normalized)
    if path.read_text(encoding="utf-8") != normalized:
        raise RuntimeError(f"UTF-8 readback mismatch: {path}")


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    path.read_text(encoding="utf-8")


def run_git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def copy_file(
    source: Path,
    destination: Path,
    repo_root: Path,
    package_root: Path,
    records: list[CopyRecord],
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(destination) != digest:
        raise RuntimeError(f"Copy hash mismatch: {source} -> {destination}")
    records.append(
        CopyRecord(
            source_path=relative_posix(source, repo_root),
            packaged_path=relative_posix(destination, package_root),
            size_bytes=destination.stat().st_size,
            sha256_before=digest,
        )
    )


def deterministic_zip(source_directory: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in sorted(path for path in source_directory.rglob("*") if path.is_file()):
            arcname = (Path(source_directory.name) / source.relative_to(source_directory)).as_posix()
            info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def read_cohort_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            (row["figure_id"], row["panel_id"]): row
            for row in csv.DictReader(handle)
        }


def csv_n_networks(path: Path) -> str:
    if not path.is_file():
        return ""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "n_networks" not in reader.fieldnames:
            return ""
        values = {
            row["n_networks"].strip()
            for row in reader
            if row.get("n_networks", "").strip() not in {"", "NA", "nan"}
        }
    return ";".join(sorted(values, key=lambda value: float(value)))


def panel_type(chart: str) -> str:
    return "schematic" if chart in {"svg_asset", "schematic"} else "quantitative"


def role_for_path(relative_path: str) -> str:
    lower = relative_path.lower()
    if relative_path == "README.md":
        return "package_readme"
    if lower.startswith("01_journal_upload/fig") and lower.endswith(".pdf"):
        return "main_figure_pdf"
    if lower.startswith("01_journal_upload/source_data_") and lower.endswith(".zip"):
        return "main_figure_source_data_archive"
    if lower.startswith("02_data_release/"):
        return "source_data_and_lineage"
    if lower.startswith("03_code_release/"):
        return "code_and_reproducibility"
    if lower.startswith("04_internal_qa/figure_masters/"):
        return "editable_figure_master"
    if lower.startswith("04_internal_qa/manuscript_baseline/"):
        return "pre_final_six_manuscript_baseline"
    if lower.startswith("04_internal_qa/"):
        return "internal_qa_and_story_record"
    return "package_support"


class FinalSixSubmissionPackageBuilder:
    def __init__(self, repo_root: Path, package_id: str, target: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.package_id = package_id
        self.target = target.resolve()
        self.staging = (self.repo_root / ".codex" / "tmp" / f"{package_id}_staging").resolve()
        self.source_bundle = self.repo_root / "results" / "paper_figure_multi_seed" / "final_six_figures_v5_c5_revised_20260804_r2"
        self.formal_root = self.repo_root / "results" / "paper_figures" / "outputs"
        self.promotion_manifest_path = self.formal_root / "main_figures_promotion_manifest.json"
        self.contract_root = self.repo_root / "docs" / "paper" / "results_state_transition_program"
        self.manuscript_baseline = self.repo_root / "docs/archive/paper/v5_direction_reset_20260730/v5_complete_package_20260730/manuscript/v5.docx"
        self.copy_records: list[CopyRecord] = []
        self.panel_rows: list[dict[str, Any]] = []
        self.figure_specs: dict[str, dict[str, Any]] = {}
        self.created_at = utc_now()
        self.git_commit = run_git(self.repo_root, "rev-parse", "HEAD")
        self.git_branch = run_git(self.repo_root, "branch", "--show-current")
        self.git_status = run_git(self.repo_root, "status", "--short").splitlines()
        self.cohort_rows = read_cohort_rows(self.source_bundle / "meta" / "cohort_validation.csv")
        self.promotion_manifest = read_json(self.promotion_manifest_path)

    def require_inputs(self) -> None:
        required = [
            self.source_bundle / "artifact_manifest.json",
            self.source_bundle / "meta" / "validation_report.json",
            self.promotion_manifest_path,
            self.manuscript_baseline,
            self.contract_root / "main_figure_sequence_contract.md",
        ]
        for figure_id in FIGURE_IDS:
            required.extend(
                [
                    self.source_bundle / figure_id / "meta" / "final_plot_spec.json",
                    self.formal_root / figure_id / f"{figure_id}.pdf",
                    self.formal_root / figure_id / f"{figure_id}.png",
                    self.formal_root / figure_id / f"{figure_id}.svg",
                ]
            )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
        if self.target.exists():
            raise FileExistsError(f"Target already exists; refusing to overwrite: {self.target}")
        if self.staging.exists():
            raise FileExistsError(f"Staging path already exists; refusing to overwrite: {self.staging}")
        if self.promotion_manifest.get("user_confirmed") is not True:
            raise RuntimeError("Formal promotion manifest is not user-confirmed")
        if self.promotion_manifest.get("scope") != list(FIGURE_IDS):
            raise RuntimeError("Formal promotion scope is not exactly Fig.1-Fig.6")

    def verify_formal_figure_sources(self) -> None:
        promotion_figures = {row["figure_id"]: row for row in self.promotion_manifest["figures"]}
        for figure_id in FIGURE_IDS:
            record = promotion_figures[figure_id]
            expected = {Path(item["path"]).suffix.lower(): item for item in record["main_files"]}
            for suffix in (".pdf", ".png", ".svg"):
                source = self.formal_root / figure_id / f"{figure_id}{suffix}"
                if sha256_file(source) != expected[suffix]["sha256"]:
                    raise RuntimeError(f"Formal figure no longer matches promotion manifest: {source}")

    def load_current_panel_specs(self) -> None:
        for figure_id in FIGURE_IDS:
            spec = read_json(self.source_bundle / figure_id / "meta" / "final_plot_spec.json")
            panel_ids = tuple(spec["reader_contract"]["task_graph"]["nodes"])
            if panel_ids != EXPECTED_PANEL_IDS[figure_id]:
                raise RuntimeError(
                    f"Unexpected current panel order for {figure_id}: {panel_ids}; "
                    f"expected {EXPECTED_PANEL_IDS[figure_id]}"
                )
            self.figure_specs[figure_id] = spec

    def build_panel_rows(self) -> None:
        for figure_id in FIGURE_IDS:
            spec = self.figure_specs[figure_id]
            source_figure = self.source_bundle / figure_id
            for order, panel_id in enumerate(EXPECTED_PANEL_IDS[figure_id], start=1):
                panel = spec["panels"][panel_id]
                chart = panel["chart"]
                kind = panel_type(chart)
                statistics_path = source_figure / "metrics" / f"panel_{panel_id}_statistics.csv"
                source_manifest_path = source_figure / "meta" / f"panel_{panel_id}_source_manifest.csv"
                if not statistics_path.is_file() or not source_manifest_path.is_file():
                    raise FileNotFoundError(
                        f"Missing statistics/source manifest for current panel {figure_id}{panel_id}"
                    )
                data_files = sorted(
                    path.relative_to(source_figure).as_posix()
                    for path in (source_figure / "data").glob(f"panel_{panel_id}*.csv")
                )
                metric_files = sorted(
                    path.relative_to(source_figure).as_posix()
                    for path in (source_figure / "metrics").glob(f"panel_{panel_id}*.csv")
                )
                cohort = self.cohort_rows.get((figure_id, panel_id), {})
                cohort_status = "not_applicable" if kind == "schematic" else cohort.get("status", "missing")
                self.panel_rows.append(
                    {
                        "figure_id": figure_id,
                        "panel_id": panel_id,
                        "panel_order": order,
                        "chain_role": CHAIN_ROLES[figure_id][0],
                        "panel_type": kind,
                        "chart": chart,
                        "claim": panel["claim"],
                        "role": panel["role"],
                        "primary_source": panel["source"],
                        "statistics_csv": f"metrics/panel_{panel_id}_statistics.csv",
                        "source_manifest_csv": f"meta/panel_{panel_id}_source_manifest.csv",
                        "additional_data_csvs": ";".join(data_files),
                        "additional_statistics_csvs": ";".join(metric_files),
                        "n_networks": "" if kind == "schematic" else csv_n_networks(statistics_path),
                        "cohort_status": cohort_status,
                    }
                )
        if len(self.panel_rows) != 32:
            raise RuntimeError(f"Expected 32 current panels, found {len(self.panel_rows)}")
        quantitative = [row for row in self.panel_rows if row["panel_type"] == "quantitative"]
        if len(quantitative) != 30:
            raise RuntimeError(f"Expected 30 current quantitative panels, found {len(quantitative)}")
        failed = [
            f"{row['figure_id']}{row['panel_id']}"
            for row in quantitative
            if row["cohort_status"] != "pass" or row["n_networks"] != "20"
        ]
        if failed:
            raise RuntimeError("Current quantitative panels failed 20-network validation: " + ", ".join(failed))

    @property
    def panel_fields(self) -> list[str]:
        return [
            "figure_id",
            "panel_id",
            "panel_order",
            "chain_role",
            "panel_type",
            "chart",
            "claim",
            "role",
            "primary_source",
            "statistics_csv",
            "source_manifest_csv",
            "additional_data_csvs",
            "additional_statistics_csvs",
            "n_networks",
            "cohort_status",
        ]

    def copy_figure_assets(self) -> None:
        journal = self.staging / "01_journal_upload"
        masters = self.staging / "04_internal_qa" / "figure_masters"
        for figure_id in FIGURE_IDS:
            figure_number = figure_id.removeprefix("fig")
            formal = self.formal_root / figure_id
            copy_file(
                formal / f"{figure_id}.pdf",
                journal / f"Fig{figure_number}.pdf",
                self.repo_root,
                self.staging,
                self.copy_records,
            )
            for suffix in ("pdf", "png", "svg"):
                copy_file(
                    formal / f"{figure_id}.{suffix}",
                    masters / f"Fig{figure_number}.{suffix}",
                    self.repo_root,
                    self.staging,
                    self.copy_records,
                )
            copy_file(
                formal / "formal_promotion.json",
                masters / "promotion_records" / f"Fig{figure_number}_formal_promotion.json",
                self.repo_root,
                self.staging,
                self.copy_records,
            )

    def copy_current_source_data(self) -> None:
        data_root = self.staging / "02_data_release" / "main_figure_source_data"
        for figure_id in FIGURE_IDS:
            figure_number = figure_id.removeprefix("fig")
            source_figure = self.source_bundle / figure_id
            destination_figure = data_root / f"Fig{figure_number}"
            current_panels = set(EXPECTED_PANEL_IDS[figure_id])

            for subdirectory in ("data", "metrics"):
                for source in sorted((source_figure / subdirectory).glob("*.csv")):
                    parts = source.stem.split("_")
                    if len(parts) < 2 or parts[0] != "panel" or parts[1] not in current_panels:
                        continue
                    copy_file(
                        source,
                        destination_figure / subdirectory / source.name,
                        self.repo_root,
                        self.staging,
                        self.copy_records,
                    )

            general_meta = {
                "final_plot_spec.json",
                "main_figure_panel_index.csv",
                "layout_measurements.csv",
                "visual_qa.json",
                "cohort_validation.csv",
                "schema_validation.csv",
                "plot_source_access.csv",
            }
            for source in sorted((source_figure / "meta").iterdir()):
                if not source.is_file():
                    continue
                include = source.name in general_meta
                if source.name.startswith("panel_"):
                    parts = source.stem.split("_")
                    include = len(parts) >= 2 and parts[1] in current_panels
                if include:
                    copy_file(
                        source,
                        destination_figure / "meta" / source.name,
                        self.repo_root,
                        self.staging,
                        self.copy_records,
                    )

            for filename in ("summary.json", "run_config.json"):
                copy_file(
                    source_figure / filename,
                    destination_figure / "upstream_records" / filename,
                    self.repo_root,
                    self.staging,
                    self.copy_records,
                )

            figure_rows = [row for row in self.panel_rows if row["figure_id"] == figure_id]
            write_csv(destination_figure / "CURRENT_PANEL_INDEX.csv", figure_rows, self.panel_fields)
            spec = self.figure_specs[figure_id]
            write_text(
                destination_figure / "README.md",
                "\n".join(
                    [
                        f"# Fig.{figure_number} current source-data record",
                        "",
                        f"Question: {spec['reader_contract']['figure_question']}",
                        "",
                        f"Terminal inference: {spec['reader_contract']['terminal_inference']}",
                        "",
                        "This directory contains only the panels present in the user-confirmed formal figure. ",
                        "Rows are copied from the validated final-six bundle; no simulation or inference was rerun.",
                        "",
                    ]
                ),
            )
            if figure_id == "fig2":
                forbidden = list(destination_figure.rglob("panel_e*"))
                if forbidden:
                    raise RuntimeError(f"Retired Fig.2e leaked into current source data: {forbidden}")

        write_csv(
            self.staging / "00_manifest" / "current_panel_index.csv",
            self.panel_rows,
            self.panel_fields,
        )

    def build_source_data_archives(self) -> None:
        data_root = self.staging / "02_data_release" / "main_figure_source_data"
        journal = self.staging / "01_journal_upload"
        for figure_number in range(1, 7):
            deterministic_zip(
                data_root / f"Fig{figure_number}",
                journal / f"Source_Data_Fig{figure_number}.zip",
            )

        rows: list[dict[str, Any]] = []
        for path in sorted(path for path in data_root.rglob("*") if path.is_file()):
            relative = relative_posix(path, self.staging)
            source_record = next(
                (record for record in self.copy_records if record.packaged_path == relative),
                None,
            )
            rows.append(
                {
                    "packaged_path": relative,
                    "source_path": source_record.source_path if source_record else "package_generated",
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "lineage_status": "copied_hash_identical" if source_record else "package_generated",
                }
            )
        write_csv(
            self.staging / "02_data_release" / "source_data_manifest.csv",
            rows,
            ["packaged_path", "source_path", "size_bytes", "sha256", "lineage_status"],
        )
        write_text(
            self.staging / "02_data_release" / "README.md",
            "# Main-figure data release\n\n"
            "This domain contains the current Fig.1-Fig.6 plot-ready CSV files, statistical records, "
            "panel source manifests and figure-level QA metadata. The same per-figure directories are "
            "packaged as deterministic ZIP files in `01_journal_upload/`.\n\n"
            "The top-level upstream `panel_index.csv` was not used as current panel authority because it "
            "predates the final Fig.2 and Fig.3 panel-order revisions. `00_manifest/current_panel_index.csv` "
            "is the current 32-panel record.\n",
        )

    def build_code_release(self) -> None:
        code_root = self.staging / "03_code_release"
        code_root.mkdir(parents=True, exist_ok=True)
        archive_path = code_root / f"net_torch_final_six_source_{self.git_commit[:8]}.zip"
        archive_paths = [
            "src/experiments/paper_figures/final_six",
            "src/experiments/paper_figures/fig1/run_task.py",
            "src/experiments/paper_figures/fig2/run_task.py",
            "src/experiments/paper_figures/fig3/run_task.py",
            "src/experiments/paper_figures/fig4/run_task.py",
            "src/experiments/paper_figures/fig5/run_task.py",
            "src/experiments/paper_figures/fig6/run_task.py",
            "src/plotting/paper_fig/final_six",
            "src/plotting/paper_fig/layout_contract.py",
            "src/plotting/paper_fig/LAYOUT_CONTRACT.md",
            "src/plotting/paper_fig/typography.py",
            "src/plotting/common/colors.py",
            "src/plotting/paper_fig/specs/fig1.yaml",
            "src/plotting/paper_fig/specs/fig2.yaml",
            "src/plotting/paper_fig/specs/fig3.yaml",
            "src/plotting/paper_fig/specs/fig4.yaml",
            "src/plotting/paper_fig/specs/fig5.yaml",
            "src/plotting/paper_fig/specs/fig6.yaml",
            "src/plotting/paper_fig/specs/paper_figures.yaml",
            "tests/test_final_six_fig56_revision.py",
            "tests/test_paper_layout_contract.py",
        ]
        subprocess.run(
            [
                "git",
                "archive",
                "--format=zip",
                "--output",
                str(archive_path),
                self.git_commit,
                *archive_paths,
            ],
            cwd=self.repo_root,
            check=True,
        )
        copy_file(
            Path(__file__).resolve(),
            code_root / "build_final_six_submission_package.py",
            self.repo_root,
            self.staging,
            self.copy_records,
        )
        write_json(
            code_root / "SOURCE_IDENTITY.json",
            {
                "schema": "final_six_code_identity_v1",
                "git_commit": self.git_commit,
                "git_branch_at_assembly": self.git_branch,
                "commit_subject": run_git(self.repo_root, "show", "-s", "--format=%s", self.git_commit),
                "commit_time": run_git(self.repo_root, "show", "-s", "--format=%cI", self.git_commit),
                "archive_scope": "committed final-six experiment, plotting, specs, layout and focused tests",
                "uncommitted_paths_not_in_archive": self.git_status,
                "scientific_results_recomputed": False,
            },
        )
        write_text(
            code_root / "README.md",
            "# Code release record\n\n"
            "The ZIP is a Git-committed source snapshot for the final-six statistics/plot-only pipeline. "
            "It intentionally excludes unrelated uncommitted working-tree changes. The package builder is "
            "also copied separately because it is the assembly entrypoint for this working candidate.\n\n"
            "Plot-only replay uses `python -m src.plotting.paper_fig.final_six.figX_plot --input-dir "
            "results/paper_figure_multi_seed/final_six_figures/figX` for X=1,...,6. It must not invoke "
            "training, simulation, rollout or parent regeneration.\n",
        )

    def build_story_and_internal_qa(self) -> None:
        qa_root = self.staging / "04_internal_qa"
        story_root = qa_root / "story_and_contracts"
        validation_root = qa_root / "validation"
        baseline_root = qa_root / "manuscript_baseline"

        contract_names = [
            "main_figure_sequence_contract.md",
            "fig1_panel_contract.md",
            "fig2_panel_contract.md",
            "fig3_panel_contract.md",
            "fig4_panel_contract.md",
            "fig5_panel_contract.md",
            "fig6_panel_contract.md",
            "final_six_figure_statistics_plotting_prompt.md",
        ]
        for name in contract_names:
            copy_file(
                self.contract_root / name,
                story_root / name,
                self.repo_root,
                self.staging,
                self.copy_records,
            )
        copy_file(
            self.repo_root / "src" / "plotting" / "paper_fig" / "LAYOUT_CONTRACT.md",
            story_root / "LAYOUT_CONTRACT.md",
            self.repo_root,
            self.staging,
            self.copy_records,
        )

        copy_file(
            self.manuscript_baseline,
            baseline_root / "v5_pre_final_six.docx",
            self.repo_root,
            self.staging,
            self.copy_records,
        )
        archived_pdf = (
            self.repo_root
            / "docs"
            / "paper"
            / "archive"
            / "v5_direction_reset_20260730"
            / "v5_complete_package_20260730"
            / "manuscript"
            / "v5.pdf"
        )
        if archived_pdf.is_file():
            copy_file(
                archived_pdf,
                baseline_root / "v5_pre_final_six_reading_copy.pdf",
                self.repo_root,
                self.staging,
                self.copy_records,
            )
        write_text(
            baseline_root / "MANUSCRIPT_BASELINE_STATUS.md",
            "# Manuscript baseline status\n\n"
            "`v5_pre_final_six.docx` predates the 2026-08-01 formal promotion of the current six main "
            "figures. It is retained only as the editable writing baseline. Its Results text, figure "
            "citations and legends must be synchronized before any submission candidate can be declared "
            "review-ready. It is deliberately not present in `01_journal_upload/`.\n",
        )

        validation_files = [
            "meta/validation_report.json",
            "meta/plot_replay_validation.json",
            "meta/plot_source_audit.json",
            "meta/require_mode_failure_test.json",
            "meta/export_validation.csv",
            "meta/source_manifest_validation.csv",
            "meta/statistics_consistency.csv",
            "meta/cohort_validation.csv",
            "meta/schema_validation.csv",
            "meta/frozen_protocol_validation.csv",
            "meta/parent_hashes_before.csv",
            "meta/parent_hashes_after.csv",
        ]
        for relative in validation_files:
            source = self.source_bundle / relative
            if source.is_file():
                copy_file(
                    source,
                    validation_root / Path(relative).name,
                    self.repo_root,
                    self.staging,
                    self.copy_records,
                )
        copy_file(
            self.promotion_manifest_path,
            validation_root / "main_figures_promotion_manifest.json",
            self.repo_root,
            self.staging,
            self.copy_records,
        )
        for name in ("summary.json", "run_config.json", "panel_index.csv", "artifact_manifest.json"):
            copy_file(
                self.source_bundle / name,
                validation_root / "upstream_records" / f"upstream_{name}",
                self.repo_root,
                self.staging,
                self.copy_records,
            )

        write_text(story_root / "RESULTS_STORYBOARD.md", self.results_storyboard())
        write_text(story_root / "FIGURE_EVIDENCE_INDEX.md", self.figure_evidence_markdown())
        write_text(story_root / "CURRENT_AUTHORITY_NOTE.md", self.current_authority_note())

    def results_storyboard(self) -> str:
        lines = [
            "# Current six-figure Results storyboard",
            "",
            "The frozen reading chain is `inherit → transition → implement → recur → organize → access`.",
            "It answers: where the silent state comes from, how one later input changes it, how that "
            "transition is implemented, whether it recurs, what organization it creates, and whether "
            "that organization remains usable.",
            "",
            "中文主线：`状态从哪里来 → 一次转移是否存在 → 一次转移如何实现 → 转移能否反复发生 "
            "→ 反复转移形成什么结构 → 该结构是否仍能被使用`。",
            "",
        ]
        for figure_number, figure_id in enumerate(FIGURE_IDS, start=1):
            spec = self.figure_specs[figure_id]
            chain_role, chinese_role = CHAIN_ROLES[figure_id]
            lines.extend(
                [
                    f"## Fig.{figure_number} — {chain_role}",
                    "",
                    f"- 逻辑角色：{chinese_role}",
                    f"- Question: {spec['reader_contract']['figure_question']}",
                    f"- Terminal inference: {spec['reader_contract']['terminal_inference']}",
                    "- Panel evidence:",
                    "",
                ]
            )
            for panel_id in EXPECTED_PANEL_IDS[figure_id]:
                panel = spec["panels"][panel_id]
                lines.append(f"  - {panel_id}: {panel['claim']} — {panel['role']}.")
            lines.append("")
        lines.extend(
            [
                "## Writing boundary",
                "",
                "This file is an evidence storyboard, not final Results prose or a final figure legend. "
                "Exact estimates, uncertainty definitions, tests, multiplicity corrections and exclusions "
                "must be taken from each panel's packaged `metrics/` and `meta/` records.",
                "",
            ]
        )
        return "\n".join(lines)

    def figure_evidence_markdown(self) -> str:
        lines = [
            "# Figure-to-evidence index",
            "",
            "All quantitative panels use seed_1000-seed_1019 as 20 independent network replicates. "
            "Schematic panels have no inferential statistics.",
            "",
        ]
        for figure_number, figure_id in enumerate(FIGURE_IDS, start=1):
            lines.extend([f"## Fig.{figure_number}", ""])
            for row in (item for item in self.panel_rows if item["figure_id"] == figure_id):
                package_prefix = f"02_data_release/main_figure_source_data/Fig{figure_number}"
                lines.extend(
                    [
                        f"### {row['panel_id']}. {row['claim']}",
                        "",
                        f"- Role: {row['role']}",
                        f"- Primary data: `{package_prefix}/{row['primary_source']}`",
                        f"- Statistics: `{package_prefix}/{row['statistics_csv']}`",
                        f"- Source manifest: `{package_prefix}/{row['source_manifest_csv']}`",
                        f"- Cohort: {row['cohort_status']}"
                        + (f"; n_networks={row['n_networks']}" if row["n_networks"] else ""),
                        "",
                    ]
                )
        return "\n".join(lines)

    def current_authority_note(self) -> str:
        return (
            "# Current authority and historical-record note\n\n"
            "Current authority is resolved in this order:\n\n"
            "1. `results/paper_figures/outputs/main_figures_promotion_manifest.json` identifies the user-confirmed "
            "formal Fig.1-Fig.6 artwork and hashes.\n"
        "2. Each figure's `meta/final_plot_spec.json` and, when present, "
        "`meta/main_figure_panel_index.csv` identify the current panel set, order, claim and data source.\n"
        "3. The panel contracts and main-sequence contract provide scientific boundaries and history.\n"
        "4. The top-level upstream `panel_index.csv` and `summary.json` are retained only as upstream build "
        "records because they predate the last plot-only revisions.\n\n"
        "Two concrete overrides are therefore enforced in this package: Fig.2 contains a-d only (the retained "
            "upstream Fig.2e data are not journal-facing Source Data), and the current Fig.3 order is e = L2 "
            "history-dependent write-back, f = STSP causal necessity. Historical submission packages were used "
            "only as directory-structure references; none of their old scientific payload was promoted here.\n"
        )

    def write_package_readmes_and_controls(self) -> None:
        write_text(
            self.staging / "README.md",
            f"# Final-six Results submission working package\n\n"
            f"Package ID: `{self.package_id}`  \n"
            f"Status: `results_consolidated_working_candidate`  \n"
            f"Submission ready: `false`\n\n"
            "This package consolidates the user-confirmed Fig.1-Fig.6 artwork, current-panel Source Data, "
            "statistics, evidence lineage, code snapshot, Results storyboard and internal QA records. No "
            "training, simulation, rollout, inference or figure redesign was performed during assembly.\n\n"
            "## Start here\n\n"
            "- Formal figure PDFs: `01_journal_upload/Fig1.pdf`-`Fig6.pdf`\n"
            "- Per-figure Source Data archives: `01_journal_upload/Source_Data_Fig1.zip`-`Source_Data_Fig6.zip`\n"
            "- Current 32-panel index: `00_manifest/current_panel_index.csv`\n"
            "- Six-figure Results storyboard: `04_internal_qa/story_and_contracts/RESULTS_STORYBOARD.md`\n"
            "- Figure-to-evidence map: `04_internal_qa/story_and_contracts/FIGURE_EVIDENCE_INDEX.md`\n"
            "- Package status and open work: `00_manifest/PACKAGE_STATUS.json` and `00_manifest/OPEN_ITEMS.md`\n\n"
            "The retained v5 manuscript predates the final-six promotion and is stored only under "
            "`04_internal_qa/manuscript_baseline/`. It is not an upload artifact.\n",
        )
        write_text(
            self.staging / "00_manifest" / "README.md",
            "# Package controls\n\n"
            "This is a non-promotable working Results package. `package_manifest.csv`, "
            "`package_manifest.json` and `checksums.sha256` cover every file outside `00_manifest/`. "
            "The current scientific payload is identified by the formal promotion manifest and the "
            "package-generated current-panel index.\n",
        )
        write_text(
            self.staging / "01_journal_upload" / "README_NOT_READY_FOR_UPLOAD.md",
            "# Not ready for upload\n\n"
            "This folder currently contains only the six confirmed main-figure PDFs and six deterministic "
            "per-figure CSV Source Data archives. Article text, final legends, Supplementary Information, "
            "journal-specific Source Data formatting and policy forms remain open.\n",
        )
        write_text(
            self.staging / "00_manifest" / "OPEN_ITEMS.md",
            "# Open items before a real submission candidate\n\n"
            "1. Rewrite and synchronize the Results text to the current six-figure chain.\n"
            "2. Write final figure legends with exact n, uncertainty definitions, tests, corrections and exclusions.\n"
            "3. Update all in-text figure citations and reconcile Abstract, Introduction and Discussion claims.\n"
            "4. Audit Methods and Statistics against the final current-panel evidence routes.\n"
            "5. Decide the journal-facing Source Data workbook/ZIP format and validate it against current guidance.\n"
            "6. Reconcile Supplementary Information and supplementary figures with the new main-figure allocation.\n"
            "7. Perform full-page Article/SI rendering, clean-copy and citation-closure QA after manuscript editing.\n"
            "8. Complete author-, policy-, disclosure-, reporting-summary- and submission-system items.\n",
        )
        write_json(
            self.staging / "00_manifest" / "excluded_evidence_register.json",
            {
                "schema": "final_six_excluded_evidence_v1",
                "entries": [
                    {
                        "item": "fig2e",
                        "status": "retained_upstream_not_in_current_main_figure",
                        "reason": "User-confirmed Fig.2 formal artwork contains panels a-d only.",
                        "upstream_bundle": "results/paper_figure_multi_seed/final_six_figures/fig2",
                    },
                    {
                        "item": "history_rewrite_bridge",
                        "status": "internal_provenance_only",
                        "reason": "The main-figure sequence contract excludes it from manuscript-facing figures.",
                    },
                ],
            },
        )

    def write_source_receipt(self) -> None:
        unique_sources: dict[str, CopyRecord] = {}
        for record in self.copy_records:
            unique_sources.setdefault(record.source_path, record)
        rows: list[dict[str, Any]] = []
        for source_path, record in sorted(unique_sources.items()):
            source = self.repo_root / Path(source_path)
            after = sha256_file(source)
            rows.append(
                {
                    "source_path": source_path,
                    "sha256_before": record.sha256_before,
                    "sha256_after": after,
                    "unchanged": str(after == record.sha256_before).lower(),
                }
            )
        if any(row["unchanged"] != "true" for row in rows):
            raise RuntimeError("At least one copied source changed during package assembly")
        write_csv(
            self.staging / "04_internal_qa" / "validation" / "source_parent_receipt.csv",
            rows,
            ["source_path", "sha256_before", "sha256_after", "unchanged"],
        )

    def write_source_identity(self) -> None:
        validation_report = self.source_bundle / "meta" / "validation_report.json"
        write_json(
            self.staging / "00_manifest" / "source_identity.json",
            {
                "schema": "final_six_results_package_source_identity_v1",
                "package_id": self.package_id,
                "created_at": self.created_at,
                "formal_promotion_id": self.promotion_manifest["promotion_id"],
                "formal_promotion_manifest": relative_posix(self.promotion_manifest_path, self.repo_root),
                "formal_promotion_manifest_sha256": sha256_file(self.promotion_manifest_path),
                "source_bundle": relative_posix(self.source_bundle, self.repo_root),
                "source_bundle_artifact_manifest_sha256": sha256_file(
                    self.source_bundle / "artifact_manifest.json"
                ),
                "source_bundle_validation_report_sha256": sha256_file(validation_report),
                "git_commit": self.git_commit,
                "git_branch_at_assembly": self.git_branch,
                "manuscript_baseline": relative_posix(self.manuscript_baseline, self.repo_root),
                "manuscript_baseline_sha256": sha256_file(self.manuscript_baseline),
                "experiments_rerun": False,
                "statistics_recomputed": False,
                "figures_redrawn": False,
                "canonical_sources_modified": False,
            },
        )

    def write_submission_inventory(self) -> None:
        rows: list[dict[str, Any]] = []
        for figure_number in range(1, 7):
            for filename, role, status, lineage in (
                (
                    f"Fig{figure_number}.pdf",
                    "Main Figure",
                    "confirmed_formal_asset",
                    "formal_promotion_hash_matched",
                ),
                (
                    f"Source_Data_Fig{figure_number}.zip",
                    "Source Data",
                    "working_csv_package_pending_journal_format_review",
                    "current_panel_csv_consolidation",
                ),
            ):
                path = self.staging / "01_journal_upload" / filename
                rows.append(
                    {
                        "relative_path": f"01_journal_upload/{filename}",
                        "submission_role": role,
                        "present": "true",
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "lineage_class": lineage,
                        "status": status,
                    }
                )
        for filename, role, status in (
            ("Article.docx", "Article", "missing_pending_results_and_legend_sync"),
            ("Figure_Legends.docx", "Figure Legends", "missing_pending_evidence_bounded_drafting"),
            ("Supplementary_Information.pdf", "Supplementary Information", "missing_pending_reconciliation"),
            ("Supplementary_Source_Data.zip", "Supplementary Source Data", "missing_pending_reconciliation"),
            ("Reporting_Summary.pdf", "Policy Form", "missing_external_author_item"),
        ):
            rows.append(
                {
                    "relative_path": f"01_journal_upload/{filename}",
                    "submission_role": role,
                    "present": "false",
                    "size_bytes": "",
                    "sha256": "",
                    "lineage_class": "not_yet_materialized",
                    "status": status,
                }
            )
        write_csv(
            self.staging / "00_manifest" / "submission_inventory.csv",
            rows,
            [
                "relative_path",
                "submission_role",
                "present",
                "size_bytes",
                "sha256",
                "lineage_class",
                "status",
            ],
        )

    def write_package_manifest(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(path for path in self.staging.rglob("*") if path.is_file()):
            relative = relative_posix(path, self.staging)
            if relative.startswith("00_manifest/"):
                continue
            rows.append(
                {
                    "relative_path": relative,
                    "domain": relative.split("/", 1)[0] if "/" in relative else "root",
                    "role": role_for_path(relative),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        fields = ["relative_path", "domain", "role", "size_bytes", "sha256"]
        write_csv(self.staging / "00_manifest" / "package_manifest.csv", rows, fields)
        write_json(
            self.staging / "00_manifest" / "package_manifest.json",
            {
                "schema": "final_six_results_package_manifest_v1",
                "package_id": self.package_id,
                "coverage": "all files outside 00_manifest",
                "file_count": len(rows),
                "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
                "files": rows,
            },
        )
        checksum_lines = [f"{row['sha256']}  {row['relative_path']}" for row in rows]
        write_text(
            self.staging / "00_manifest" / "checksums.sha256",
            "\n".join(checksum_lines) + "\n",
        )
        return rows

    def validate_package(self, manifest_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        manifest_failures: list[str] = []
        for row in manifest_rows:
            path = self.staging / Path(row["relative_path"])
            if not path.is_file():
                manifest_failures.append(f"missing:{row['relative_path']}")
                continue
            if path.stat().st_size != int(row["size_bytes"]):
                manifest_failures.append(f"size:{row['relative_path']}")
            if sha256_file(path) != row["sha256"]:
                manifest_failures.append(f"sha256:{row['relative_path']}")

        zip_failures: list[str] = []
        for archive_path in sorted((self.staging / "01_journal_upload").glob("Source_Data_Fig*.zip")):
            with zipfile.ZipFile(archive_path, "r") as archive:
                bad = archive.testzip()
                if bad:
                    zip_failures.append(f"{archive_path.name}:{bad}")

        utf8_failures: list[str] = []
        mojibake_failures: list[str] = []
        for path in sorted(path for path in self.staging.rglob("*") if path.is_file()):
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                utf8_failures.append(f"{relative_posix(path, self.staging)}:{exc}")
                continue
            markers = [marker for marker in MOJIBAKE_MARKERS if marker in text]
            if markers:
                mojibake_failures.append(
                    f"{relative_posix(path, self.staging)}:{','.join(markers)}"
                )

        current_quantitative = [
            row for row in self.panel_rows if row["panel_type"] == "quantitative"
        ]
        cohort_pass = all(
            row["cohort_status"] == "pass" and row["n_networks"] == "20"
            for row in current_quantitative
        )
        formal_assets_pass = all(
            sha256_file(self.formal_root / figure_id / f"{figure_id}.{suffix}")
            == sha256_file(
                self.staging
                / "04_internal_qa"
                / "figure_masters"
                / f"Fig{figure_id.removeprefix('fig')}.{suffix}"
            )
            for figure_id in FIGURE_IDS
            for suffix in ("pdf", "png", "svg")
        )
        fig2e_absent = not any(
            (self.staging / "02_data_release" / "main_figure_source_data" / "Fig2").rglob(
                "panel_e*"
            )
        )
        status = "pass" if all(
            [
                not manifest_failures,
                not zip_failures,
                not utf8_failures,
                not mojibake_failures,
                cohort_pass,
                formal_assets_pass,
                fig2e_absent,
            ]
        ) else "fail"
        report = {
            "schema": "final_six_results_package_validation_v1",
            "package_id": self.package_id,
            "validated_at": utc_now(),
            "status": status,
            "manifest_files_checked": len(manifest_rows),
            "manifest_failures": manifest_failures,
            "source_data_archives_checked": 6,
            "zip_failures": zip_failures,
            "utf8_failures": utf8_failures,
            "mojibake_failures": mojibake_failures,
            "current_panel_count": len(self.panel_rows),
            "current_quantitative_panel_count": len(current_quantitative),
            "current_schematic_panel_count": len(self.panel_rows) - len(current_quantitative),
            "all_current_quantitative_panels_have_20_networks": cohort_pass,
            "formal_figure_assets_match_sources": formal_assets_pass,
            "retired_fig2e_absent_from_journal_facing_source_data": fig2e_absent,
            "source_results_recomputed": False,
            "source_figures_modified": False,
        }
        write_json(self.staging / "00_manifest" / "package_validation.json", report)
        if status != "pass":
            raise RuntimeError("Package validation failed: " + json.dumps(report, ensure_ascii=False))
        return report

    def write_final_status(self, validation: dict[str, Any], manifest_rows: Sequence[dict[str, Any]]) -> None:
        write_json(
            self.staging / "00_manifest" / "PACKAGE_STATUS.json",
            {
                "schema": "final_six_results_working_package_status_v1",
                "package_id": self.package_id,
                "created_at": self.created_at,
                "status": "results_consolidated_working_candidate",
                "review_ready": False,
                "submission_ready": False,
                "promotion_allowed": False,
                "main_figures": "user_confirmed_formal",
                "main_figure_count": 6,
                "current_panel_count": 32,
                "current_quantitative_panel_count": 30,
                "source_data": "current_panels_consolidated",
                "article": "pending_final_six_results_and_legend_sync",
                "supplementary_information": "pending_reconciliation",
                "journal_policy_materials": "pending",
                "package_validation": validation["status"],
                "manifest_file_count": len(manifest_rows),
                "experiments_rerun": False,
                "statistics_recomputed": False,
                "figures_redrawn": False,
            },
        )
        write_json(
            self.staging / "00_manifest" / "temporary_artifact_disposition.json",
            {
                "schema": "temporary_artifact_disposition_v1",
                "staging_path": relative_posix(self.staging, self.repo_root),
                "disposition": "promoted_atomically_into_candidate_root",
                "temporary_artifacts_remaining": [],
                "deletion_count": 0,
            },
        )

    def assemble(self) -> dict[str, Any]:
        self.require_inputs()
        self.verify_formal_figure_sources()
        self.load_current_panel_specs()
        self.build_panel_rows()
        self.staging.mkdir(parents=True)
        for domain in (
            "00_manifest",
            "01_journal_upload",
            "02_data_release",
            "03_code_release",
            "04_internal_qa",
        ):
            (self.staging / domain).mkdir(parents=True, exist_ok=True)

        self.copy_figure_assets()
        self.copy_current_source_data()
        self.build_source_data_archives()
        self.build_code_release()
        self.build_story_and_internal_qa()
        self.write_package_readmes_and_controls()
        self.write_source_receipt()
        self.write_source_identity()
        self.write_submission_inventory()

        source_validation = read_json(self.source_bundle / "meta" / "validation_report.json")
        write_json(
            self.staging / "04_internal_qa" / "validation" / "assembly_scope_validation.json",
            {
                "schema": "final_six_results_assembly_scope_validation_v1",
                "status": "pass",
                "source_bundle_validation_status": source_validation["status"],
                "source_parent_hashes_unchanged": source_validation["parent_hashes_unchanged"],
                "formal_promotion_user_confirmed": self.promotion_manifest["user_confirmed"],
                "current_panel_authority": "per-figure final_plot_spec.json",
                "historical_submission_payload_reused": False,
                "experiments_rerun": False,
                "statistics_recomputed": False,
                "figures_redrawn": False,
            },
        )

        manifest_rows = self.write_package_manifest()
        validation = self.validate_package(manifest_rows)
        self.write_final_status(validation, manifest_rows)

        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.staging.replace(self.target)
        return {
            "status": "complete",
            "package_id": self.package_id,
            "package_root": str(self.target),
            "main_figures": 6,
            "current_panels": 32,
            "quantitative_panels": 30,
            "manifest_files": len(manifest_rows),
            "validation": validation["status"],
            "submission_ready": False,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble the user-confirmed final-six figures and evidence into a submission-like working package."
    )
    parser.add_argument(
        "--package-id",
        default="communications_biology_20260801_final_six_results_candidate",
    )
    parser.add_argument(
        "--target",
        type=Path,
        help="Target package root. Defaults to docs/paper/submission_packages/<package-id>.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    target = args.target or (
        repo_root / "docs" / "paper" / "submission_packages" / args.package_id
    )
    builder = FinalSixSubmissionPackageBuilder(repo_root, args.package_id, target)
    result = builder.assemble()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
