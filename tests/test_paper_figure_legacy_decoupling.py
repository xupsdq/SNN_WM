from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap

import pytest


FIGURE_LEGACY_MODULES = (
    (2, "src.experiments.paper_figures.fig2_pair_fused_stsp_state_experiment"),
    (3, "src.experiments.paper_figures.fig3_multiitem_peak_landscape_experiment"),
    (4, "src.experiments.paper_figures.fig4_overlap_reentry_experiment"),
    (5, "src.experiments.paper_figures.fig5_local_support_competition_experiment"),
    (6, "src.experiments.paper_figures.fig6_peak_amplified_reentry_experiment"),
)


@pytest.mark.parametrize(("figure_number", "legacy_module"), FIGURE_LEGACY_MODULES)
def test_paper_figure_task_modules_import_without_legacy_monolith(
    figure_number: int,
    legacy_module: str,
) -> None:
    script = textwrap.dedent(
        f"""
        import importlib
        import importlib.abc
        import pkgutil
        import sys

        legacy_module = {legacy_module!r}
        package_name = "src.experiments.paper_figures.fig{figure_number}"

        class RejectLegacyImport(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == legacy_module:
                    raise ImportError(f"legacy monolith import rejected: {{fullname}}")
                return None

        sys.meta_path.insert(0, RejectLegacyImport())
        package = importlib.import_module(package_name)
        module_names = [f"{{package_name}}.run_task"]
        module_names.extend(
            module_info.name
            for module_info in pkgutil.walk_packages(package.__path__, prefix=f"{{package_name}}.")
            if ".subexperiments." in module_info.name
        )
        for module_name in sorted(set(module_names)):
            importlib.import_module(module_name)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(("figure_number", "legacy_module"), FIGURE_LEGACY_MODULES)
def test_extracted_figure_constants_match_legacy_values(
    figure_number: int,
    legacy_module: str,
) -> None:
    constants = importlib.import_module(f"src.experiments.paper_figures.fig{figure_number}.constants")
    legacy = importlib.import_module(legacy_module)

    for name in constants.__all__:
        assert getattr(constants, name) == getattr(legacy, name), name
