"""CI installs only requirements-ci.txt (no torch, mlflow or open_clip).

Every module CI touches must import without them; heavy imports belong inside
the functions that need them. This test fails fast if one creeps to the top level.
"""

import importlib

import pytest

CI_MODULES = ["hfa.accession", "hfa.clean", "hfa.dates", "hfa.manifest", "hfa.splits", "hfa.rubric",
              "hfa.label_sample", "hfa.label_server", "hfa.label_audit", "hfa.attr_keyword", "hfa.attr_eval",
              "hfa.recsys", "hfa.rec_eval", "hfa.search", "hfa.search_eval", "hfa.rag", "hfa.ci_check", "hfa.refresh"]


@pytest.mark.parametrize("name", CI_MODULES)
def test_module_imports_without_heavy_dependencies(name):
    importlib.import_module(name)
